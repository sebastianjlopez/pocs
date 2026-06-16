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
SHOT_MIN_SPEED_MS        = 4.0    # el tiro va a > 4 m/s
SHOT_MIN_COS_ANGLE       = 0.25   # cos del ángulo máx. hacia el aro (~75°)
PASS_LOOKBACK_FRAMES_S   = 2.0    # segundos hacia atrás para detectar pase demorado
BASKET_RIM_MAX_M         = 1.5    # canasta: pelota pasó a menos de 150cm del aro (margen por inaccuracidad de perspectiva)
DEAD_BALL_MAX_SPEED_MS   = 0.5    # pelota casi quieta
DEAD_BALL_MIN_FRAMES     = 15     # frames CONSECUTIVOS quieta para declarar pelota muerta


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

        # Último poseedor conocido (persiste a través de DEAD_BALL para atribuir tiros/pases)
        self._last_possessor_id:    int | None = None
        self._last_possessor_team:  int | None = None
        self._last_possession_frame: int = 0

        # Frames consecutivos con pelota lenta (para hysteresis de DEAD_BALL)
        self._slow_frames: int = 0

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
        # Pelota muerta (casi quieta) — requiere N frames CONSECUTIVOS para evitar
        # falsos positivos por frames donde YOLO pierde momentáneamente la pelota.
        if state.speed_ms < DEAD_BALL_MAX_SPEED_MS:
            self._slow_frames += 1
            if self.phase == BallPhase.DEAD or self._slow_frames >= DEAD_BALL_MIN_FRAMES:
                return BallPhase.DEAD
            # Pendiente de confirmación: mantener fase actual hasta acumular frames
            return self.phase
        else:
            self._slow_frames = 0

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

        _lookback = int(self.fps * PASS_LOOKBACK_FRAMES_S)

        # Para PASES: ventana corta (PASS_LOOKBACK_FRAMES_S) — el receptor debe aparecer pronto.
        # Para TIROS: usamos el último poseedor absoluto, sin límite de tiempo.
        # En básquet, si la pelota sale volando hacia el aro, siempre fue alguien quien la tiró.
        effective_possessor = self.possessor_id
        effective_team      = self.possessor_team
        if effective_possessor is None and self._last_possessor_id is not None:
            effective_possessor = self._last_possessor_id   # sin límite de tiempo para tiros
            effective_team      = self._last_possessor_team

        # Para pases: restringir a ventana corta
        pass_possessor = self.possessor_id
        pass_team      = self.possessor_team
        if pass_possessor is None and self._last_possessor_id is not None:
            if frame_idx - self._last_possession_frame <= _lookback:
                pass_possessor = self._last_possessor_id
                pass_team      = self._last_possessor_team

        # POSSESSED: alguien agarró la pelota
        if new_phase == BallPhase.POSSESSED:
            # Pase directo: IN_FLIGHT → POSSESSED con diferente jugador
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

            # Pase demorado: pasó por DEAD/LOOSE pero hubo un poseedor reciente
            # del mismo equipo → contar como pase en vez de nueva posesión aislada.
            elif (
                self.phase in (BallPhase.LOOSE, BallPhase.DEAD)
                and pass_possessor is not None
                and pid != pass_possessor
                and team_cache.get(pid) == pass_team
            ):
                ev = self._emit(GameEvent(
                    type=EventType.PASS,
                    frame_idx=frame_idx,
                    player_id=pass_possessor,
                    receiver_id=pid,
                    team=pass_team,
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
                # Actualizar último poseedor ANTES de pisarlo
                self._last_possessor_id     = pid
                self._last_possessor_team   = team
                self._last_possession_frame = frame_idx
                self.possessor_id    = pid
                self.possessor_team  = team
                self.possession_frame = frame_idx

        # IN_FLIGHT: la pelota salió volando
        elif new_phase == BallPhase.IN_FLIGHT:
            # Tiro: velocidad suficiente + dirección apunta al aro
            is_shot = (
                state.speed_ms >= SHOT_MIN_SPEED_MS
                and self._is_moving_toward_rim(state, near_rim)
                and effective_possessor is not None   # sabemos quién tiró
            )

            if is_shot:
                ev = self._emit(GameEvent(
                    type=EventType.SHOT,
                    frame_idx=frame_idx,
                    player_id=effective_possessor,
                    team=effective_team,
                    ball_pos=state.court_xy.copy(),
                    speed_ms=state.speed_ms,
                    metadata={"rim": near_rim.tolist()},
                ))
                emitted.append(ev)
                self._shot_active   = True
                self._shot_rim      = near_rim
                self._shot_player   = effective_possessor
                self._shot_team     = effective_team
                self._shot_frame    = frame_idx
                self._min_rim_dist  = float("inf")

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
                # _last_possessor_id se mantiene para atribuir el próximo tiro/pase
                self.possessor_id = None

        return emitted

    # ------------------------------------------------------------------
    def _is_moving_toward_rim(self, state: BallState, near_rim: np.ndarray) -> bool:
        """
        Devuelve True si el vector de velocidad de la pelota apunta
        aproximadamente hacia el aro (cos del ángulo > SHOT_MIN_COS_ANGLE).
        """
        vel = state.velocity_court          # cm/frame
        vel_norm = float(np.linalg.norm(vel))
        if vel_norm < 1.0:
            return False
        to_rim = near_rim - state.court_xy
        rim_norm = float(np.linalg.norm(to_rim))
        if rim_norm < 1.0:
            return True   # ya está en el aro
        cos_angle = float(np.dot(vel / vel_norm, to_rim / rim_norm))
        return cos_angle > SHOT_MIN_COS_ANGLE

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
