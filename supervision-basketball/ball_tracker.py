"""
BallTracker — historial de posición de la pelota y métricas básicas.

Guarda los últimos N frames de posición (píxeles + cancha real),
calcula velocidad, detecta rebotes y encuentra el jugador más cercano.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# Posiciones del aro en centímetros (cancha NBA completa 2865×1524 cm)
RIM_LEFT  = np.array([152.0,  762.0])   # aro izquierdo
RIM_RIGHT = np.array([2713.0, 762.0])   # aro derecho
RIM_RADIUS_CM = 45.7 / 2               # radio del aro


@dataclass
class BallSample:
    """Una observación de la pelota en un frame."""
    frame_idx:   int
    pixel_xy:    np.ndarray   # coordenadas en píxeles
    court_xy:    np.ndarray   # coordenadas en cm sobre la cancha
    confidence:  float = 1.0
    detected:    bool  = True


@dataclass
class BallState:
    """Estado calculado de la pelota en el frame actual."""
    court_xy:         np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed_ms:         float = 0.0       # velocidad en m/s
    velocity_court:   np.ndarray = field(default_factory=lambda: np.zeros(2))  # cm/frame
    is_bouncing:      bool = False      # pica en el piso
    nearest_player_id: int | None = None
    nearest_dist_m:   float = float("inf")
    dist_to_rim_left_m:  float = float("inf")
    dist_to_rim_right_m: float = float("inf")
    detected:         bool = False


class BallTracker:
    """
    Mantiene el historial de posición de la pelota y calcula:
    - Velocidad en m/s
    - Si está rebotando (pica)
    - Jugador más cercano
    - Distancia a cada aro
    """

    def __init__(self, fps: float, history_seconds: float = 1.5) -> None:
        self.fps = fps
        self.history: deque[BallSample] = deque(maxlen=int(fps * history_seconds))
        self.pixel_history: deque[np.ndarray] = deque(maxlen=int(fps * 0.5))  # para detectar rebotes
        self._missing_frames = 0
        self.MAX_MISSING = int(fps * 0.5)   # si no aparece 0.5s → se pierde

    # ------------------------------------------------------------------
    def update(
        self,
        frame_idx: int,
        pixel_xy: np.ndarray | None,
        court_xy: np.ndarray | None,
        confidence: float = 1.0,
    ) -> None:
        """Registrar la posición de la pelota en este frame."""
        if pixel_xy is None or court_xy is None:
            self._missing_frames += 1
            if self._missing_frames <= self.MAX_MISSING and len(self.history) > 0:
                # Interpolar: mantener última posición conocida con menor confianza
                last = self.history[-1]
                self.history.append(BallSample(
                    frame_idx=frame_idx,
                    pixel_xy=last.pixel_xy.copy(),
                    court_xy=last.court_xy.copy(),
                    confidence=0.1,
                    detected=False,
                ))
            return

        self._missing_frames = 0
        self.history.append(BallSample(
            frame_idx=frame_idx,
            pixel_xy=pixel_xy.copy(),
            court_xy=court_xy.copy(),
            confidence=confidence,
            detected=True,
        ))
        self.pixel_history.append(pixel_xy.copy())

    # ------------------------------------------------------------------
    def get_state(
        self,
        player_ids: np.ndarray | None,
        player_court_coords: np.ndarray | None,
    ) -> BallState:
        """Calcular el estado actual de la pelota."""
        state = BallState()

        if len(self.history) == 0:
            return state

        latest = self.history[-1]
        state.court_xy = latest.court_xy
        state.detected = latest.detected

        # Velocidad (promedio de las últimas N muestras)
        state.speed_ms, state.velocity_court = self._calc_speed()

        # Rebote (oscillación vertical en píxeles)
        state.is_bouncing = self._detect_bounce()

        # Jugador más cercano
        if player_ids is not None and player_court_coords is not None and len(player_ids) > 0:
            dists = np.linalg.norm(player_court_coords - latest.court_xy, axis=1)
            idx = int(np.argmin(dists))
            state.nearest_player_id = int(player_ids[idx])
            state.nearest_dist_m = float(dists[idx]) / 100.0

        # Distancia a los aros
        state.dist_to_rim_left_m  = float(np.linalg.norm(latest.court_xy - RIM_LEFT))  / 100.0
        state.dist_to_rim_right_m = float(np.linalg.norm(latest.court_xy - RIM_RIGHT)) / 100.0

        return state

    # ------------------------------------------------------------------
    def court_positions(self) -> np.ndarray:
        """Devuelve el historial de posiciones en cancha (N×2)."""
        return np.array([s.court_xy for s in self.history])

    def pixel_positions(self) -> np.ndarray:
        """Devuelve el historial de posiciones en píxeles (N×2)."""
        return np.array([s.pixel_xy for s in self.history])

    # Velocidad máxima física de una pelota de básquet (~35 m/s = tiro de toda cancha)
    MAX_SPEED_MS = 35.0

    # ------------------------------------------------------------------
    def _calc_speed(self) -> tuple[float, np.ndarray]:
        """
        Velocidad promedio sobre los últimos ~0.3s.
        Solo usa muestras realmente detectadas (excluye interpolaciones).
        Si el gap entre las dos últimas detecciones reales es mayor a 1s,
        devuelve 0 para evitar velocidades ficticias por saltos de detección.
        """
        window = min(len(self.history), max(2, int(self.fps * 0.3)))
        if window < 2:
            return 0.0, np.zeros(2)

        samples = list(self.history)[-window:]

        # Filtrar solo muestras con detección real
        real = [s for s in samples if s.detected]
        if len(real) < 2:
            return 0.0, np.zeros(2)

        # Si el gap entre las dos detecciones reales más cercanas supera 1s → resetear
        gap_frames = real[-1].frame_idx - real[-2].frame_idx
        if gap_frames > self.fps:          # > 1 segundo de gap
            return 0.0, np.zeros(2)

        deltas = []
        for i in range(1, len(real)):
            dt = max(1, real[i].frame_idx - real[i-1].frame_idx)
            delta_cm = (real[i].court_xy - real[i-1].court_xy) / dt
            deltas.append(delta_cm)

        avg_delta = np.mean(deltas, axis=0)  # cm/frame
        speed_cm_per_s = float(np.linalg.norm(avg_delta)) * self.fps
        speed_ms = min(speed_cm_per_s / 100.0, self.MAX_SPEED_MS)
        return speed_ms, avg_delta  # m/s, cm/frame

    def _detect_bounce(self) -> bool:
        """
        Detecta si la pelota está picando mirando si el Y en píxeles
        alterna entre subir y bajar al menos 2 veces.
        """
        if len(self.pixel_history) < 6:
            return False

        ys = [p[1] for p in self.pixel_history]
        # contar cambios de dirección vertical
        direction_changes = 0
        prev_dir = 0
        for i in range(1, len(ys)):
            diff = ys[i] - ys[i-1]
            if abs(diff) < 2:
                continue
            cur_dir = 1 if diff > 0 else -1
            if prev_dir != 0 and cur_dir != prev_dir:
                direction_changes += 1
            prev_dir = cur_dir

        return direction_changes >= 2
