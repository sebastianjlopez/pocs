"""
CourtDetector — detecta automáticamente las esquinas de la cancha.

Estrategia:
  1. Segmenta el piso de la cancha por color (madera = tono naranja-marrón)
  2. Encuentra el contorno más grande (la cancha)
  3. Aproxima un rectángulo con las 4 esquinas
  4. Si falla, intenta con detección de líneas (Hough)
  5. Fallback: usa los bordes del frame con margen

Uso:
    detector = CourtDetector()
    corners = detector.detect_from_video("video.mp4", preview=True)
    # corners → array (4, 2) con las esquinas en píxeles
"""

from __future__ import annotations

import cv2
import numpy as np


# Rangos HSV de respaldo (se intentan si el adaptativo falla)
HSV_RANGES_FALLBACK = [
    # (h_min, h_max, s_min, s_max, v_min, v_max)
    (10, 30,  30,  160, 100, 230),   # madera NBA clásica (naranja/tan)
    ( 5, 40,  20,  180,  80, 240),   # rango ampliado
    ( 0, 50,  10,  200,  60, 255),   # muy permisivo
]


def _adaptive_hsv_range(frame: np.ndarray) -> list[tuple]:
    """
    Muestrea el color del piso en la zona central-inferior del frame
    (donde suele haber cancha y pocos jugadores) y construye un rango HSV
    adaptativo alrededor de ese color.
    """
    h, w = frame.shape[:2]
    # Zona central-inferior: evita el fondo/tribuna arriba y los extremos
    roi = frame[int(h * 0.55) : int(h * 0.80), int(w * 0.25) : int(w * 0.75)]
    if roi.size == 0:
        return HSV_RANGES_FALLBACK

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)

    # Usar mediana (más robusta a outliers como jugadores o líneas)
    hm = float(np.median(hsv_roi[:, 0]))
    sm = float(np.median(hsv_roi[:, 1]))
    vm = float(np.median(hsv_roi[:, 2]))

    # Rango ±20 en H (circular), ±40 en S y V — amplio para capturar variaciones
    h_slack, s_slack, v_slack = 20, 40, 50
    h1 = max(0,   int(hm - h_slack))
    h2 = min(179, int(hm + h_slack))
    s1 = max(0,   int(sm - s_slack))
    s2 = min(255, int(sm + s_slack))
    v1 = max(40,  int(vm - v_slack))
    v2 = min(255, int(vm + v_slack))

    adaptive = (h1, h2, s1, s2, v1, v2)
    # Devolver el rango adaptativo primero, luego los fallbacks
    return [adaptive] + HSV_RANGES_FALLBACK


class CourtDetector:
    """Detecta automáticamente las 4 esquinas de la cancha en un frame."""

    def __init__(self, min_area_ratio: float = 0.15) -> None:
        """
        min_area_ratio: la cancha debe ocupar al menos este % del frame.
        Reducir si la cámara está muy lejos.
        """
        self.min_area_ratio = min_area_ratio
        self._last_corners: np.ndarray | None = None
        self._debug_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Detecta las esquinas de la cancha en un frame.
        Devuelve array (4, 2) o None si no encontró nada confiable.
        """
        h, w = frame.shape[:2]
        min_area = h * w * self.min_area_ratio

        # Intento 1: segmentación por color de piso
        corners = self._detect_by_color(frame, min_area)
        if corners is not None:
            self._last_corners = corners
            return corners

        # Intento 2: detección de líneas (Hough)
        corners = self._detect_by_lines(frame)
        if corners is not None:
            self._last_corners = corners
            return corners

        # Fallback: rectángulo con margen del frame
        corners = self._fallback_corners(w, h)
        self._last_corners = corners
        return corners

    # ------------------------------------------------------------------
    def detect_from_video(
        self,
        video_path: str,
        sample_frames: int = 8,
        preview: bool = True,
        preview_seconds: float = 3.0,
    ) -> np.ndarray:
        """
        Analiza varios frames del video y elige la mejor detección.
        Si preview=True, muestra la detección antes de continuar.
        Usa PyAV para decodificar (soporta AV1, H.264, H.265, VP9, etc.)
        """
        from video_reader import get_video_info, sample_frames as sample_video_frames

        info  = get_video_info(video_path)
        fps   = info.fps
        w     = info.width
        h     = info.height

        sampled = sample_video_frames(video_path, n=sample_frames, window=0.3)

        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []  # (score, corners, frame)

        for _frame_idx, frame in sampled:
            corners = self.detect(frame)
            if corners is not None:
                score = self._score_corners(corners, w, h)
                candidates.append((score, corners.copy(), frame.copy()))

        if not candidates:
            print("No se detectó la cancha. Usando bordes del frame como fallback.")
            return self._fallback_corners(w, h)

        # Elegir el candidato con mejor score
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_corners, best_frame = candidates[0]
        print(f"Cancha detectada (score: {best_score:.2f})")

        if preview:
            self._show_preview(best_frame, best_corners, fps, preview_seconds)

        return best_corners

    # ------------------------------------------------------------------
    def _detect_by_color(
        self, frame: np.ndarray, min_area: float
    ) -> np.ndarray | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Usar rango adaptativo basado en el color real del piso
        ranges = _adaptive_hsv_range(frame)

        for (h1, h2, s1, s2, v1, v2) in ranges:
            mask = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))

            # Limpiar ruido
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < min_area:
                continue

            self._debug_mask = mask.copy()

            # Envolvente convexa → aproximar a 4 esquinas
            hull = cv2.convexHull(largest)
            corners = self._approx_quad(hull, frame.shape[:2])
            if corners is not None:
                return corners

        return None

    def _detect_by_lines(self, frame: np.ndarray) -> np.ndarray | None:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100, minLineLength=frame.shape[1] // 5, maxLineGap=30
        )
        if lines is None or len(lines) < 4:
            return None

        # Filtrar líneas casi horizontales y casi verticales
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 20 or angle > 160:
                h_lines.append(line[0])
            elif 70 < angle < 110:
                v_lines.append(line[0])

        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        # Tomar las líneas más extremas
        top_y    = min(min(l[1], l[3]) for l in h_lines)
        bottom_y = max(max(l[1], l[3]) for l in h_lines)
        left_x   = min(min(l[0], l[2]) for l in v_lines)
        right_x  = max(max(l[0], l[2]) for l in v_lines)

        h, w = frame.shape[:2]
        # Verificar que el rectángulo sea razonable
        if (right_x - left_x) < w * 0.3 or (bottom_y - top_y) < h * 0.2:
            return None

        return np.array([
            [left_x,  bottom_y],
            [right_x, bottom_y],
            [right_x, top_y],
            [left_x,  top_y],
        ], dtype=np.float32)

    def _approx_quad(
        self, contour: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray | None:
        """Aproxima un contorno a un cuadrilátero de 4 puntos."""
        peri    = cv2.arcLength(contour, True)
        epsilon = 0.02 * peri

        for factor in [0.02, 0.04, 0.06, 0.08, 0.10]:
            approx = cv2.approxPolyDP(contour, factor * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                return self._sort_corners(pts)

        # Si no se pudo aproximar a 4 puntos, usar el bounding rect
        rect = cv2.minAreaRect(contour)
        box  = cv2.boxPoints(rect).astype(np.float32)
        return self._sort_corners(box)

    def _sort_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        Ordena las esquinas: inferior-izq, inferior-der, superior-der, superior-izq.
        (el orden que espera ViewTransformer)
        """
        # Ordenar por Y (inferior = Y mayor en imagen)
        pts = pts[np.argsort(pts[:, 1])]
        top    = pts[:2][np.argsort(pts[:2, 0])]    # izq, der (arriba)
        bottom = pts[2:][np.argsort(pts[2:, 0])]    # izq, der (abajo)
        return np.array([
            bottom[0],  # inferior-izquierda
            bottom[1],  # inferior-derecha
            top[1],     # superior-derecha
            top[0],     # superior-izquierda
        ], dtype=np.float32)

    def _score_corners(self, corners: np.ndarray, w: int, h: int) -> float:
        """
        Puntúa una detección. Mayor puntaje = mejor.
        Premia rectángulos grandes, centrados y con relación de aspecto razonable.
        """
        area = cv2.contourArea(corners)
        frame_area = w * h

        # Qué porcentaje del frame ocupa
        area_ratio = area / frame_area

        # Centrado
        cx = np.mean(corners[:, 0])
        cy = np.mean(corners[:, 1])
        center_score = 1.0 - (abs(cx - w/2) / (w/2) * 0.5 + abs(cy - h/2) / (h/2) * 0.5)

        # Relación de aspecto (cancha NBA: ~1.88:1)
        width  = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])
        if height == 0:
            return 0.0
        aspect = width / height
        ideal_aspect = 1.88
        aspect_score = 1.0 - min(abs(aspect - ideal_aspect) / ideal_aspect, 1.0)

        return area_ratio * 0.5 + center_score * 0.3 + aspect_score * 0.2

    def _fallback_corners(self, w: int, h: int, margin: float = 0.05) -> np.ndarray:
        """Usa el frame completo con un pequeño margen como fallback."""
        mx, my = int(w * margin), int(h * margin)
        return np.array([
            [mx,     h - my],
            [w - mx, h - my],
            [w - mx, my],
            [mx,     my],
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def _show_preview(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        fps: float,
        seconds: float,
    ) -> None:
        preview = frame.copy()
        pts = corners.astype(np.int32)

        # Dibujar el polígono detectado
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 100), thickness=3)

        labels = ["INF-IZQ", "INF-DER", "SUP-DER", "SUP-IZQ"]
        colors = [(0,255,255), (0,165,255), (0,255,0), (255,0,255)]
        for i, (pt, label, color) in enumerate(zip(pts, labels, colors)):
            cv2.circle(preview, tuple(pt), 10, color, -1)
            cv2.putText(preview, label, (pt[0]+12, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Overlay de instrucciones
        overlay = preview.copy()
        cv2.rectangle(overlay, (0, preview.shape[0]-60),
                      (preview.shape[1], preview.shape[0]), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, preview, 0.4, 0, preview)
        cv2.putText(preview, "Cancha detectada automaticamente  |  ENTER=continuar  R=reintentar  Q=salir",
                    (12, preview.shape[0]-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 100), 2)

        win = "Detección de cancha — verificar y confirmar"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1280, preview.shape[1]), min(720, preview.shape[0]))

        wait_ms = int(1000 / max(fps, 1))
        total_ms = int(seconds * 1000)
        elapsed  = 0

        while elapsed < total_ms:
            cv2.imshow(win, preview)
            key = cv2.waitKey(wait_ms) & 0xFF
            elapsed += wait_ms
            if key == 13:    # ENTER → continuar
                break
            elif key == ord("r"):   # reintentar
                cv2.destroyWindow(win)
                return
            elif key == ord("q"):
                cv2.destroyWindow(win)
                raise SystemExit("Cancelado por el usuario.")

        cv2.destroyWindow(win)
