"""
Basketball Analytics POC usando roboflow/supervision.

Pipeline:
  1. Detectar jugadores + pelota con YOLO
  2. Trackear con ByteTrack
  3. Proyectar posiciones a coordenadas de cancha real (ViewTransformer)
  4. Analizar zonas tácticas (PolygonZone)
  5. Calcular velocidad, heatmap y tiempo en zona
  6. Exportar datos + video anotado
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

import supervision as sv

# ---------------------------------------------------------------------------
# Coordenadas de la cancha (SOURCE = esquinas en píxeles del frame de TV)
# Ajustar estos puntos para cada cámara específica
# ---------------------------------------------------------------------------
COURT_SOURCE = np.array([
    [320, 680],   # esquina inferior izquierda
    [1600, 680],  # esquina inferior derecha
    [1480, 320],  # esquina superior derecha
    [440, 320],   # esquina superior izquierda
], dtype=np.float32)

# Cancha NBA: 28.65m x 15.24m — usamos cm para evitar decimales
COURT_WIDTH_CM = 2865
COURT_HEIGHT_CM = 1524

COURT_TARGET = np.array([
    [0, COURT_HEIGHT_CM],
    [COURT_WIDTH_CM, COURT_HEIGHT_CM],
    [COURT_WIDTH_CM, 0],
    [0, 0],
], dtype=np.float32)

# ---------------------------------------------------------------------------
# Zonas tácticas en coordenadas reales de la cancha (cm)
# ---------------------------------------------------------------------------
PAINT_LEFT = np.array([
    [0, 427], [488, 427], [488, 1097], [0, 1097]
])

PAINT_RIGHT = np.array([
    [2377, 427], [2865, 427], [2865, 1097], [2377, 1097]
])

THREE_POINT_LEFT = np.array([
    [0, 0], [670, 0], [670, 1524], [0, 1524]
])

THREE_POINT_RIGHT = np.array([
    [2195, 0], [2865, 0], [2865, 1524], [2195, 1524]
])

MIDCOURT_LEFT = np.array([
    [0, 0], [1432, 0], [1432, 1524], [0, 1524]
])


class ViewTransformer:
    """Proyecta puntos de imagen a coordenadas de cancha real via perspectiva."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32),
            target.astype(np.float32),
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)


class CourtMap:
    """Dibuja un mini-mapa 2D de la cancha con posiciones de jugadores."""

    MAP_W = 400
    MAP_H = 213  # proporcional a 28.65 x 15.24

    TEAM_COLORS = {
        0: (255, 80, 80),   # equipo A — azul
        1: (80, 255, 80),   # equipo B — verde
        -1: (255, 255, 0),  # sin equipo / árbitro
    }

    def __init__(self) -> None:
        self.scale_x = self.MAP_W / COURT_WIDTH_CM
        self.scale_y = self.MAP_H / COURT_HEIGHT_CM

    def render(
        self,
        tracker_ids: np.ndarray,
        court_coords: np.ndarray,
        team_ids: dict[int, int],
    ) -> np.ndarray:
        canvas = np.zeros((self.MAP_H, self.MAP_W, 3), dtype=np.uint8)
        # fondo de cancha
        cv2.rectangle(canvas, (0, 0), (self.MAP_W - 1, self.MAP_H - 1), (40, 40, 40), -1)
        cv2.rectangle(canvas, (0, 0), (self.MAP_W - 1, self.MAP_H - 1), (100, 100, 100), 1)
        # línea de medio campo
        cv2.line(canvas, (self.MAP_W // 2, 0), (self.MAP_W // 2, self.MAP_H), (80, 80, 80), 1)

        for tid, (cx, cy) in zip(tracker_ids, court_coords):
            px = int(cx * self.scale_x)
            py = int(cy * self.scale_y)
            px = max(4, min(self.MAP_W - 4, px))
            py = max(4, min(self.MAP_H - 4, py))
            team = team_ids.get(int(tid), -1)
            color = self.TEAM_COLORS.get(team, (200, 200, 200))
            cv2.circle(canvas, (px, py), 5, color, -1)
            cv2.putText(canvas, str(tid), (px + 6, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        return canvas


def assign_teams_by_color(
    frame: np.ndarray,
    detections: sv.Detections,
    tracker_team: dict[int, int],
) -> dict[int, int]:
    """
    Asigna equipo a cada tracker_id usando clustering K-means sobre el color
    promedio del torso del bbox. Simple heurística: rojo vs azul dominante.
    """
    if detections.tracker_id is None:
        return tracker_team

    for tid, box in zip(detections.tracker_id, detections.xyxy):
        if tid in tracker_team:
            continue
        x1, y1, x2, y2 = map(int, box)
        # recorta la mitad superior del bbox (torso)
        mid_y = (y1 + y2) // 2
        crop = frame[y1:mid_y, x1:x2]
        if crop.size == 0:
            continue
        # promedio BGR
        mean_b, mean_g, mean_r = cv2.mean(crop)[:3]
        # heurística básica: R > B → equipo 0 (rojo), B > R → equipo 1 (azul)
        tracker_team[tid] = 0 if mean_r > mean_b else 1

    return tracker_team


def main(
    source_video_path: str,
    target_video_path: str | None = None,
    model_weights: str = "yolo11x.pt",
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.5,
) -> None:
    """
    Corre el pipeline de Basketball Analytics sobre un video.

    Args:
        source_video_path: Ruta al video de entrada
        target_video_path: Ruta de salida (None = solo display)
        model_weights: Pesos del modelo YOLO (idealmente fine-tuned en basquet)
        confidence_threshold: Umbral de confianza del detector
        iou_threshold: IOU threshold para NMS
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Instalar ultralytics: pip install ultralytics")

    model = YOLO(model_weights)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps

    # --- Tracker ---
    byte_track = sv.ByteTrack(
        frame_rate=fps,
        track_activation_threshold=confidence_threshold,
    )

    # --- Transformers ---
    view_transformer = ViewTransformer(COURT_SOURCE, COURT_TARGET)

    # --- Zonas tácticas (en coordenadas reales) ---
    # Para filtrar por zona, trabajamos sobre court_coords en vez de pixel_coords
    # PolygonZone sobre imagen requeriría invertir la perspectiva; aquí lo hacemos
    # directamente en espacio de cancha via contains_point

    # --- Anotadores ---
    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.TOP_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(fps * 3),
        position=sv.Position.BOTTOM_CENTER,
    )
    heat_map_annotator = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        opacity=0.4,
        radius=40,
        kernel_size=25,
    )

    # --- Estado acumulado ---
    speed_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=int(fps)))
    zone_time: dict[int, dict[str, int]] = defaultdict(lambda: {"paint": 0, "perimeter": 0, "midcourt": 0})
    tracker_team: dict[int, int] = {}
    court_map = CourtMap()
    smoother = sv.DetectionsSmoother()

    # --- Zonas en espacio cancha (polígonos) ---
    def in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(
            polygon.astype(np.float32), (float(point[0]), float(point[1])), False
        ) >= 0

    # --- Exportación CSV ---
    csv_sink = sv.CSVSink(f"{Path(source_video_path).stem}_tracking.csv")

    frame_generator = sv.get_video_frames_generator(source_video_path)
    frame_idx = 0

    sink_ctx = sv.VideoSink(target_video_path, video_info) if target_video_path else None

    with csv_sink:
        for frame in frame_generator:
            # --- Detección ---
            results = model(frame, conf=confidence_threshold, iou=iou_threshold)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Filtrar solo personas (class_id=0 en COCO) y pelota (32)
            mask = np.isin(detections.class_id, [0, 32])
            detections = detections[mask]

            # --- Smoothing + Tracking ---
            detections = smoother.update_with_detections(detections)
            detections = byte_track.update_with_detections(detections)

            # --- Asignación de equipos por color ---
            tracker_team = assign_teams_by_color(frame, detections, tracker_team)

            # --- Proyección a coordenadas de cancha ---
            pixel_anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            court_coords = view_transformer.transform_points(pixel_anchors)

            # --- Velocidad + Zonas ---
            labels = []
            if detections.tracker_id is not None:
                for tid, court_pt in zip(detections.tracker_id, court_coords):
                    speed_history[tid].append(court_pt)

                    # Velocidad en m/s
                    speed_str = ""
                    if len(speed_history[tid]) >= 2:
                        p1 = speed_history[tid][0]
                        p2 = speed_history[tid][-1]
                        dist_cm = np.linalg.norm(p2 - p1)
                        dist_m = dist_cm / 100.0
                        time_s = len(speed_history[tid]) / fps
                        speed_ms = dist_m / time_s if time_s > 0 else 0
                        speed_str = f" {speed_ms:.1f}m/s"

                    # Zona táctica
                    zone = "mid"
                    if in_polygon(court_pt, PAINT_LEFT) or in_polygon(court_pt, PAINT_RIGHT):
                        zone = "paint"
                        zone_time[tid]["paint"] += 1
                    elif in_polygon(court_pt, THREE_POINT_LEFT) or in_polygon(court_pt, THREE_POINT_RIGHT):
                        zone = "3pt"
                        zone_time[tid]["perimeter"] += 1
                    else:
                        zone_time[tid]["midcourt"] += 1

                    team = tracker_team.get(int(tid), -1)
                    team_str = ["A", "B"].get(team, "?") if isinstance(team, int) and team in [0, 1] else "?"
                    labels.append(f"#{tid}[{team_str}] {zone}{speed_str}")
            else:
                labels = [f"cls={c}" for c in (detections.class_id or [])]

            # --- Exportar a CSV ---
            csv_sink.append(detections, custom_data={"frame": frame_idx})

            # --- Anotaciones visuales ---
            annotated = frame.copy()
            annotated = heat_map_annotator.annotate(annotated, detections)
            annotated = trace_annotator.annotate(annotated, detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels)

            # --- Mini-mapa ---
            if detections.tracker_id is not None and len(court_coords) > 0:
                mini_map = court_map.render(detections.tracker_id, court_coords, tracker_team)
                map_h, map_w = mini_map.shape[:2]
                annotated[10:10+map_h, 10:10+map_w] = mini_map

            # --- Frame info ---
            cv2.putText(
                annotated,
                f"Frame {frame_idx} | Jugadores: {len(detections)}",
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

            if sink_ctx:
                sink_ctx.write_frame(annotated)

            cv2.imshow("Basketball Analytics", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    if sink_ctx:
        sink_ctx.__exit__(None, None, None)

    cv2.destroyAllWindows()

    # --- Resumen de tiempo en zona ---
    print("\n=== Tiempo en zona por jugador (frames) ===")
    print(f"{'Jugador':>8} | {'Equipo':>6} | {'Paint':>8} | {'Perimeter':>10} | {'Midcourt':>9}")
    print("-" * 55)
    for tid, zones in sorted(zone_time.items()):
        team = tracker_team.get(tid, -1)
        team_str = ["A", "B"][team] if team in [0, 1] else "?"
        total = sum(zones.values()) or 1
        print(
            f"{tid:>8} | {team_str:>6} | "
            f"{zones['paint']:>8} ({zones['paint']/total*100:.0f}%) | "
            f"{zones['perimeter']:>8} ({zones['perimeter']/total*100:.0f}%) | "
            f"{zones['midcourt']:>8} ({zones['midcourt']/total*100:.0f}%)"
        )

    print(f"\nDatos de tracking exportados a: {Path(source_video_path).stem}_tracking.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basketball Analytics con supervision")
    parser.add_argument("source_video", help="Video de entrada")
    parser.add_argument("--output", "-o", default=None, help="Video de salida (opcional)")
    parser.add_argument("--weights", default="yolo11x.pt", help="Pesos del modelo YOLO")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold")
    args = parser.parse_args()

    main(
        source_video_path=args.source_video,
        target_video_path=args.output,
        model_weights=args.weights,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
    )
