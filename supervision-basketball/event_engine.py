"""
EventEngine — detecta eventos de juego a partir del movimiento de la pelota.

Eventos detectados:
  POSSESSION  — un jugador tiene la pelota (la pica o la agarra)
  PASS        — la pelota viaja rápido entre dos jugadores
  SHOT        — la pelota va en dirección al aro
  BASKET      — la pelota pasó por el aro (posible canasta)
  LOOSE_BALL  — la pelota está suelta (rebote, robo, etc.)
  DEAD_BALL   — la pelota está quieta (falta, fuera, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from ball_tracker import BallState, RIM_LEFT, RIM_RIGHT


# ---------------------------------------------------------------------------
# Umbrales de detección — ajustar según calidad del video y cámara
# ---------------------------------------------------------------------------
POSSESSION_MAX_DIST_M    = 1.8    # jugador a menos de 1.8m de la pelota = posesión
POSSESSION_MIN_FRAMES    = 8      # mínimo de frames seguidos para confirmar posesión
DRIBBLE_MAX_SPEED_MS     = 3.5    # picando: la pelota no viaja lejos (< 3.5 m/s)
PASS_MIN_SPEED_MS        = 6.0    # pase: la pelota va a > 6 m/s
PASS_MAX_DIST_PLAYER_M   = 2.5    # al recibir, el jugador está a < 2.5m
SHOT_RIM_APPROACH_M      = 4.0    # umbral: la pelota está a menos de 4m del aro
SHOT_MIN_SPEED_MS        = 4.0    # el tiro va a > 4 m/s
BASKET_RIM_MAX_M         = 1.5    # canasta: pelota pasó a menos de 150cm del aro (margen por inaccuracidad de perspectiva)
DEAD_BALL_MAX_SPEED_MS   = 0.5    # pelota casi quieta
DEAD_BALL_MIN_FRAMES     = 15     # frames quieta para declarar pelota muerta


# ---------------------------------------------------------------------------
# Definición de eventos
# ---------------------------------------------------------------------------
class EventType(Enum):
    POSSESSION  = auto()
    PASS        = auto()
    SHOT        = auto()
    BASKET      = auto()
    LOOSE_BALL  = auto()
    DEAD_BALL   = auto()


@dataclass
class GameEvent:
    type:        EventType
    frame_idx:   int
    player_id:   int | None = None        # jugador protagonista
    receiver_id: int | None = None        # receptor (en pase)
    team:        int | None = None        # 0=A, 1=B
    ball_pos:    np.ndarray | None = None # posición de la pelota en cancha (cm)
    speed_ms:    float = 0.0
    metadata:    dict = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"[frame {self.frame_idx:04d}] {self.type.name}"]
        if self.player_id is not None:
            team_s = ("Eq.A" if self.team == 0 else "Eq.B") if self.team in (0,1) else ""
            parts.append(f"jugador #{self.player_id} {team_s}")
        if self.receiver_id is not None:
            parts.append(f"→ #{self.receiver_id}")
        if self.speed_ms > 0:
            parts.append(f"({self.speed_ms:.1f} m/s)")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Máquina de estados
# ---------------------------------------------------------------------------
class BallPhase(Enum):
    UNKNOWN    = auto()
    POSSESSED  = auto()   # jugador tiene la pelota
    IN_FLIGHT  = auto()   # pelota en el aire (pase o tiro)
    LOOSE      = auto()   # pelota suelta
    DEAD       = auto()   # pelota quieta / fuera de juego


class EventEngine:
    """
    Lee el BallState frame a frame y emite GameEvents.

    Uso:
        engine = EventEngine(fps=30)
        for frame in video:
            state = ball_tracker.get_state(...)
            events = engine.update(frame_idx, state, team_cache)
            for ev in events:
                stats.record(ev)
    """

    def __init__(
        self,
        fps: float,
        on_event: Callable[[GameEvent], None] | None = None,
    ) -> None:
        self.fps = fps
        self.on_event = on_event  # callback opcional para cada evento

        # Estado actual
        self.phase = BallPhase.UNKNOWN
        self.phase_frames = 0                # frames en la fase actual

        # Quién tenía la pelota antes
        self.possessor_id:   int | None = None
        self.possessor_team: int | None = None
        self.possession_frame: int = 0

        # Para detectar si el tiro fue canasta
        self._shot_active  = False
        self._shot_rim:    np.ndarray | None = None  # aro al que apunta
        self._shot_player: int | None = None
        self._shot_team:   int | None = None
        self._shot_frame:  int = 0
        self._min_rim_dist = float("inf")    # mínima distancia al aro durante el tiro

        # Historial de eventos emitidos
        self.events: list[GameEvent] = []

    # ------------------------------------------------------------------
    def update(
        self,
        frame_idx: int,
        state: BallState,
        team_cache: dict[int, int],
    ) -> list[GameEvent]:
        """
        Procesar el estado de la pelota en este frame.
        Devuelve lista de eventos ocurridos (puede ser vacía).
        """
        if not state.detected:
            self.phase_frames += 1
            return []

        emitted: list[GameEvent] = []

        # Distancia al aro más cercano
        near_rim, rim_dist = self._nearest_rim(state)

        # ---- Actualizar seguimiento de tiro en vuelo ----
        if self._shot_active:
            self._min_rim_dist = min(self._min_rim_dist, rim_dist)
            # Si la pelota llegó muy cerca del aro → posible canasta
            if rim_dist < BASKET_RIM_MAX_M:
                ev = self._emit(GameEvent(
                    type=EventType.BASKET,
                    frame_idx=frame_idx,
                    player_id=self._shot_player,
                    team=self._shot_team,
                    ball_pos=state.court_xy.copy(),
                    speed_ms=state.speed_ms,
                ))
                emitted.append(ev)
                self._shot_active = False
            # Si la pelota frena o se acerca a un jugador → miss / rebote
            elif state.speed_ms < 1.0 or state.nearest_dist_m < POSSESSION_MAX_DIST_M:
                self._shot_active = False

        # ---- Transiciones de fase ----
        new_phase = self._classify_phase(state, rim_dist)

        if new_phase != self.phase:
            events = self._handle_transition(frame_idx, state, team_cache, new_phase, near_rim)
            emitted.extend(events)
            self.phase = new_phase
            self.phase_frames = 0
        else:
            self.phase_frames += 1

        return emitted

    # ------------------------------------------------------------------
    def _classify_phase(self, state: BallState, rim_dist: float) -> BallPhase:
        """Determinar en qué fase está la pelota ahora."""
        # Pelota muerta (casi quieta)
        if state.speed_ms < DEAD_BALL_MAX_SPEED_MS:
            return BallPhase.DEAD

        # Cerca de un jugador y lenta → posesión
        if (
            state.nearest_dist_m < POSSESSION_MAX_DIST_M
            and state.speed_ms < PASS_MIN_SPEED_MS
        ):
            return BallPhase.POSSESSED

        # Rápida y alejada de jugadores → en vuelo (pase o tiro)
        if state.speed_ms >= PASS_MIN_SPEED_MS:
            return BallPhase.IN_FLIGHT

        # Default: suelta
        return BallPhase.LOOSE

    def _handle_transition(
        self,
        frame_idx: int,
        state: BallState,
        team_cache: dict[int, int],
        new_phase: BallPhase,
        near_rim: np.ndarray,
    ) -> list[GameEvent]:
        """Emitir eventos según la transición de fase."""
        emitted = []
        pid  = state.nearest_player_id
        team = team_cache.get(pid, None) if pid is not None else None

        # POSSESSED: alguien agarró la pelota
        if new_phase == BallPhase.POSSESSED:
            # Cambio de poseedor = pase recibido
            if (
                self.phase == BallPhase.IN_FLIGHT
                and self.possessor_id is not None
                and pid != self.possessor_id
            ):
                ev = self._emit(GameEvent(
                    type=EventType.PASS,
                    frame_idx=frame_idx,
                    player_id=self.possessor_id,
                    receiver_id=pid,
                    team=self.possessor_team,
                    ball_pos=state.court_xy.copy(),
                    speed_ms=state.speed_ms,
                ))
                emitted.append(ev)

            # Registrar nueva posesión
            if pid != self.possessor_id:
                ev = self._emit(GameEvent(
                    type=EventType.POSSESSION,
                    frame_idx=frame_idx,
                    player_id=pid,
                    team=team,
                    ball_pos=state.court_xy.copy(),
                    speed_ms=state.speed_ms,
                    metadata={"bouncing": state.is_bouncing},
                ))
                emitted.append(ev)
                self.possessor_id   = pid
                self.possessor_team = team
                self.possession_frame = frame_idx

        # IN_FLIGHT: la pelota salió volando
        elif new_phase == BallPhase.IN_FLIGHT:
            is_toward_rim = (
                state.dist_to_rim_left_m  < SHOT_RIM_APPROACH_M
                or state.dist_to_rim_right_m < SHOT_RIM_APPROACH_M
            ) and state.speed_ms >= SHOT_MIN_SPEED_MS

            if is_toward_rim:
                ev = self._emit(GameEvent(
                    type=EventType.SHOT,
                    frame_idx=frame_idx,
                    player_id=self.possessor_id,
                    team=self.possessor_team,
                    ball_pos=state.court_xy.copy(),
                    speed_ms=state.speed_ms,
                    metadata={"rim": near_rim.tolist()},
                ))
                emitted.append(ev)
                self._shot_active   = True
                self._shot_rim      = near_rim
                self._shot_player   = self.possessor_id
                self._shot_team     = self.possessor_team
                self._shot_frame    = frame_idx
                self._min_rim_dist  = float("inf")
            else:
                # Pase iniciado (se confirmará cuando alguien lo reciba)
                pass

        # LOOSE: pelota suelta
        elif new_phase == BallPhase.LOOSE:
            if self.phase in (BallPhase.POSSESSED, BallPhase.IN_FLIGHT):
                ev = self._emit(GameEvent(
                    type=EventType.LOOSE_BALL,
                    frame_idx=frame_idx,
                    ball_pos=state.court_xy.copy(),
                ))
                emitted.append(ev)

        # DEAD: pelota quieta
        elif new_phase == BallPhase.DEAD:
            if self.phase != BallPhase.DEAD:
                ev = self._emit(GameEvent(
                    type=EventType.DEAD_BALL,
                    frame_idx=frame_idx,
                    ball_pos=state.court_xy.copy(),
                ))
                emitted.append(ev)
                self.possessor_id = None

        return emitted

    # ------------------------------------------------------------------
    def _nearest_rim(self, state: BallState) -> tuple[np.ndarray, float]:
        if state.dist_to_rim_left_m <= state.dist_to_rim_right_m:
            return RIM_LEFT, state.dist_to_rim_left_m
        return RIM_RIGHT, state.dist_to_rim_right_m

    def _emit(self, event: GameEvent) -> GameEvent:
        self.events.append(event)
        if self.on_event:
            self.on_event(event)
        return event
