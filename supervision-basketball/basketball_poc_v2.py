"""
Basketball Analytics POC v2
----------------------------
Pipeline:
  1. Detección con YOLOv11 (jugadores + pelota)
  2. Tracking con ByteTrack (supervision)
  3. Asignación de equipos con CLIP (zero-shot, sin etiquetado)
  4. ViewTransformer → coordenadas de cancha real
  5. Velocidad, tiempo en zona, heatmap, mini-mapa
  6. Exportación CSV + JSON

Uso rápido:
    python basketball_poc_v2.py --source test_basketball.mp4

Con calibración:
    python basketball_poc_v2.py --source video.mp4 --court-config court_config.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

import supervision as sv

# ---------------------------------------------------------------------------
# Configuración por defecto de la cancha (si no hay court_config.json)
# Ajustar SOURCE para cada cámara.
# Orden: inferior-izq, inferior-der, superior-der, superior-izq
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_POINTS = np.array([
    [100,  620],
    [1180, 620],
    [1080, 100],
    [200,  100],
], dtype=np.float32)

COURT_WIDTH_CM  = 2865   # NBA: 28.65m
COURT_HEIGHT_CM = 1524   # NBA: 15.24m

# Zonas tácticas (cm) — cancha completa
# Pintura izquierda: 0–488cm de ancho, 427–1097cm de alto
ZONES = {
    "paint_left":  np.array([[0, 427], [488, 427], [488, 1097], [0, 1097]]),
    "paint_right": np.array([[2377, 427], [2865, 427], [2865, 1097], [2377, 1097]]),
    "three_left":  np.array([[0, 0], [670, 0], [670, 1524], [0, 1524]]),
    "three_right": np.array([[2195, 0], [2865, 0], [2865, 1524], [2195, 1524]]),
    "midcourt":    np.array([[1282, 0], [1583, 0], [1583, 1524], [1282, 1524]]),
}

TEAM_A_COLOR = (230, 230, 230)   # BGR blanco para anotaciones
TEAM_B_COLOR = (80,  80,  220)   # BGR azul  para anotaciones
BALL_COLOR   = (0,  140,  255)   # BGR naranja

# ---------------------------------------------------------------------------
# ViewTransformer
# ---------------------------------------------------------------------------
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.M = cv2.getPerspectiveTransform(
            source.astype(np.float32),
            target.astype(np.float32),
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Team Classifier con CLIP
# ---------------------------------------------------------------------------
class CLIPTeamClassifier:
    """
    Clasifica jugadores en equipos usando CLIP de forma zero-shot.
    Requiere: pip install open-clip-torch
    """

    def __init__(
        self,
        team_a_prompt: str = "basketball player wearing white jersey uniform",
        team_b_prompt: str = "basketball player wearing dark colored jersey uniform",
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        self.team_a_prompt = team_a_prompt
        self.team_b_prompt = team_b_prompt
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        try:
            import open_clip
            import torch
            self._torch = torch
            print("Cargando CLIP (ViT-B/32)...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.model.eval()
            print("CLIP cargado.")
        except ImportError:
            print("open-clip-torch no instalado. Usando clasificador por color como fallback.")
            print("Instalar con: pip install open-clip-torch")
            self.model = None

    def classify_crops(
        self,
        crops: list[np.ndarray],
        tracker_ids: list[int],
        cache: dict[int, int],
    ) -> dict[int, int]:
        """
        Clasifica una lista de crops BGR → {tracker_id: team (0 o 1)}.
        Usa cache para no re-clasificar IDs ya vistos.
        """
        to_classify = [(i, tid) for i, tid in enumerate(tracker_ids) if tid not in cache]
        if not to_classify:
            return cache

        if self.model is not None:
            return self._classify_clip(crops, tracker_ids, to_classify, cache)
        else:
            return self._classify_color(crops, tracker_ids, to_classify, cache)

    def _classify_clip(self, crops, tracker_ids, to_classify, cache) -> dict[int, int]:
        from PIL import Image
        import torch

        texts = self.tokenizer([self.team_a_prompt, self.team_b_prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Procesar en batches
        for batch_start in range(0, len(to_classify), self.batch_size):
            batch = to_classify[batch_start : batch_start + self.batch_size]
            images = []
            valid = []
            for crop_idx, tid in batch:
                crop = crops[crop_idx]
                if crop is None or crop.size == 0:
                    continue
                # Recortar el tercio superior (torso/jersey)
                h = crop.shape[0]
                torso = crop[h // 4 : h * 3 // 4, :]
                if torso.size == 0:
                    torso = crop
                pil_img = Image.fromarray(cv2.cvtColor(torso, cv2.COLOR_BGR2RGB))
                images.append(self.preprocess(pil_img))
                valid.append(tid)

            if not images:
                continue

            img_tensor = torch.stack(images).to(self.device)
            with torch.no_grad():
                img_features = self.model.encode_image(img_tensor)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                logits = (img_features @ text_features.T) * 100
                probs = logits.softmax(dim=-1).cpu().numpy()

            for tid, prob in zip(valid, probs):
                cache[tid] = int(np.argmax(prob))

        return cache

    def _classify_color(self, crops, tracker_ids, to_classify, cache) -> dict[int, int]:
        """Fallback: asigna equipo por brillo promedio del torso (blanco vs oscuro)."""
        for crop_idx, tid in to_classify:
            crop = crops[crop_idx]
            if crop is None or crop.size == 0:
                cache[tid] = 0
                continue
            h = crop.shape[0]
            torso = crop[h // 4 : h * 3 // 4, :]
            if torso.size == 0:
                torso = crop
            gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            # Equipo 0 = camisetas claras (brightness > 128)
            cache[tid] = 0 if brightness > 110 else 1
        return cache


# ---------------------------------------------------------------------------
# Mini-mapa de cancha
# ---------------------------------------------------------------------------
class CourtMiniMap:
    MAP_W = 420
    MAP_H = 224  # proporcional a 28.65 × 15.24

    COURT_COLOR    = (50, 40, 30)
    LINE_COLOR     = (200, 200, 200)
    PAINT_COLOR    = (40, 40, 60)

    def __init__(self) -> None:
        self.sx = self.MAP_W / COURT_WIDTH_CM
        self.sy = self.MAP_H / COURT_HEIGHT_CM
        self._base = self._draw_base()

    def _px(self, x_cm: float, y_cm: float) -> tuple[int, int]:
        return int(x_cm * self.sx), int(y_cm * self.sy)

    def _draw_base(self) -> np.ndarray:
        img = np.full((self.MAP_H, self.MAP_W, 3), self.COURT_COLOR, dtype=np.uint8)
        lc = self.LINE_COLOR
        # borde
        cv2.rectangle(img, (0, 0), (self.MAP_W - 1, self.MAP_H - 1), lc, 1)
        # línea central
        cv2.line(img, self._px(COURT_WIDTH_CM / 2, 0),
                 self._px(COURT_WIDTH_CM / 2, COURT_HEIGHT_CM), lc, 1)
        # pinturas (relleno)
        cv2.rectangle(img, self._px(0, 427), self._px(488, 1097), self.PAINT_COLOR, -1)
        cv2.rectangle(img, self._px(2377, 427), self._px(2865, 1097), self.PAINT_COLOR, -1)
        # pinturas (borde)
        cv2.rectangle(img, self._px(0, 427), self._px(488, 1097), lc, 1)
        cv2.rectangle(img, self._px(2377, 427), self._px(2865, 1097), lc, 1)
        # círculo central
        cx, cy = self._px(COURT_WIDTH_CM / 2, COURT_HEIGHT_CM / 2)
        cv2.circle(img, (cx, cy), int(183 * self.sx), lc, 1)
        return img

    def render(
        self,
        tracker_ids: np.ndarray,
        court_coords: np.ndarray,
        team_cache: dict[int, int],
        ball_court: np.ndarray | None = None,
    ) -> np.ndarray:
        canvas = self._base.copy()

        # Pelota
        if ball_court is not None and len(ball_court) > 0:
            bx, by = self._px(float(ball_court[0]), float(ball_court[1]))
            bx = max(4, min(self.MAP_W - 4, bx))
            by = max(4, min(self.MAP_H - 4, by))
            cv2.circle(canvas, (bx, by), 6, (0, 140, 255), -1)

        # Jugadores
        for tid, (cx, cy) in zip(tracker_ids, court_coords):
            px = int(np.clip(cx * self.sx, 4, self.MAP_W - 4))
            py = int(np.clip(cy * self.sy, 4, self.MAP_H - 4))
            team = team_cache.get(int(tid), -1)
            color = TEAM_A_COLOR if team == 0 else (TEAM_B_COLOR if team == 1 else (128, 128, 128))
            cv2.circle(canvas, (px, py), 6, color, -1)
            cv2.circle(canvas, (px, py), 6, (0, 0, 0), 1)
            cv2.putText(canvas, str(tid), (px + 7, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
        return canvas


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(
        polygon.astype(np.float32),
        (float(point[0]), float(point[1])),
        False,
    ) >= 0


def get_zone(pt: np.ndarray) -> str:
    for name, poly in ZONES.items():
        if in_polygon(pt, poly):
            return name
    return "open_court"


def team_color(team: int) -> tuple[int, int, int]:
    return TEAM_A_COLOR if team == 0 else (TEAM_B_COLOR if team == 1 else (128, 128, 128))


def extract_crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main(
    source_video_path: str,
    target_video_path: str | None = None,
    court_config_path: str | None = None,
    model_weights: str = "yolo11n.pt",
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.5,
    team_a_prompt: str = "basketball player wearing white jersey",
    team_b_prompt: str = "basketball player wearing dark jersey",
    clip_device: str = "cpu",
    show_heatmap: bool = True,
    show_minimap: bool = True,
) -> None:
    print(f"\n{'='*60}")
    print("  Basketball Analytics POC v2")
    print(f"{'='*60}")
    print(f"  Video:  {source_video_path}")
    print(f"  Modelo: {model_weights}")
    print(f"  Team A: {team_a_prompt}")
    print(f"  Team B: {team_b_prompt}")
    print(f"{'='*60}\n")

    # --- Cargar configuración de cancha ---
    source_points = DEFAULT_SOURCE_POINTS
    if court_config_path and Path(court_config_path).exists():
        with open(court_config_path) as f:
            cfg = json.load(f)
        source_points = np.array(cfg["source_points"], dtype=np.float32)
        print(f"Configuración de cancha cargada: {court_config_path}")
    else:
        print("Usando coordenadas de cancha por defecto (ejecutá calibrate_court.py para mejorar precisión)")

    target_points = np.array([
        [0,               COURT_HEIGHT_CM],
        [COURT_WIDTH_CM,  COURT_HEIGHT_CM],
        [COURT_WIDTH_CM,  0],
        [0,               0],
    ], dtype=np.float32)

    # --- Modelo y video ---
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Instalar ultralytics: pip install ultralytics")

    print(f"Cargando modelo {model_weights}...")
    model = YOLO(model_weights)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps
    print(f"Video: {video_info.width}×{video_info.height} @ {fps:.1f}fps")

    # --- Inicializar componentes ---
    view_transformer = ViewTransformer(source_points, target_points)
    team_classifier  = CLIPTeamClassifier(team_a_prompt, team_b_prompt, device=clip_device)
    mini_map         = CourtMiniMap()

    byte_track = sv.ByteTrack(
        frame_rate=fps,
        track_activation_threshold=confidence_threshold,
    )
    smoother = sv.DetectionsSmoother()

    # --- Anotadores ---
    thickness  = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)

    trace_annotator    = sv.TraceAnnotator(
        thickness=max(1, thickness - 1),
        trace_length=int(fps * 4),
        position=sv.Position.BOTTOM_CENTER,
    )
    box_annotator      = sv.BoxAnnotator(thickness=thickness)
    label_annotator    = sv.LabelAnnotator(
        text_scale=text_scale * 0.85,
        text_thickness=max(1, thickness - 1),
        text_position=sv.Position.TOP_CENTER,
        text_padding=4,
    )
    heatmap_annotator  = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        opacity=0.35,
        radius=50,
        kernel_size=31,
    )
    ellipse_annotator  = sv.EllipseAnnotator(thickness=thickness)

    # --- Exportación ---
    stem = Path(source_video_path).stem
    csv_sink  = sv.CSVSink(f"{stem}_tracking.csv")
    zone_log: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    speed_history: dict[int, deque]     = defaultdict(lambda: deque(maxlen=int(fps * 2)))
    team_cache: dict[int, int]          = {}

    # --- Controles de teclado ---
    show_hm  = show_heatmap
    show_mm  = show_minimap
    paused   = False

    frame_generator = sv.get_video_frames_generator(source_video_path)

    sink_ctx = sv.VideoSink(target_video_path, video_info) if target_video_path else None

    print("\nIniciando procesamiento...")
    print("Controles: Q=salir | H=heatmap | M=mapa | S=screenshot | P=pausa\n")

    t_start  = time.time()
    frame_idx = 0

    with csv_sink:
        for frame in frame_generator:
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord("p"): paused = False
                elif key == ord("q"): break
                continue

            # --- Detección ---
            results = model(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(results)

            # Separar pelota (class 32 en COCO) de jugadores (class 0)
            ball_mask    = detections.class_id == 32
            player_mask  = detections.class_id == 0

            ball_dets    = detections[ball_mask]
            player_dets  = detections[player_mask]

            # --- Tracking solo en jugadores ---
            player_dets = smoother.update_with_detections(player_dets)
            player_dets = byte_track.update_with_detections(player_dets)

            # --- Asignación de equipos con CLIP ---
            if player_dets.tracker_id is not None and len(player_dets) > 0:
                crops = [extract_crop(frame, box) for box in player_dets.xyxy]
                team_cache = team_classifier.classify_crops(
                    crops, player_dets.tracker_id.tolist(), team_cache
                )

            # --- Proyección a cancha ---
            player_anchors = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            court_coords   = view_transformer.transform_points(player_anchors)

            ball_court = None
            if len(ball_dets) > 0:
                ball_anchor = ball_dets.get_anchors_coordinates(sv.Position.CENTER)
                ball_court  = view_transformer.transform_points(ball_anchor)
                if len(ball_court) > 0:
                    ball_court = ball_court[0]

            # --- Velocidad + Zonas ---
            labels: list[str] = []
            if player_dets.tracker_id is not None:
                for tid, court_pt in zip(player_dets.tracker_id, court_coords):
                    speed_history[tid].append(court_pt.copy())
                    zone = get_zone(court_pt)
                    zone_log[tid][zone] += 1

                    speed_str = ""
                    hist = speed_history[tid]
                    if len(hist) >= max(2, int(fps * 0.5)):
                        d_cm = float(np.linalg.norm(hist[-1] - hist[0]))
                        t_s  = len(hist) / fps
                        speed_ms = (d_cm / 100.0) / t_s
                        speed_str = f" {speed_ms:.1f}m/s"

                    team = team_cache.get(int(tid), -1)
                    team_str = ("A" if team == 0 else "B") if team in (0, 1) else "?"
                    short_zone = zone.replace("_left", "L").replace("_right", "R").replace("_", "")
                    labels.append(f"#{tid}[{team_str}]{speed_str}\n{short_zone}")
            else:
                labels = ["" for _ in player_dets]

            # --- Exportar CSV ---
            csv_sink.append(player_dets, custom_data={"frame": frame_idx})

            # --- Render ---
            annotated = frame.copy()

            if show_hm:
                annotated = heatmap_annotator.annotate(annotated, player_dets)

            annotated = trace_annotator.annotate(annotated, player_dets)

            # Colorear boxes por equipo
            if player_dets.tracker_id is not None:
                for i, (box, tid) in enumerate(zip(player_dets.xyxy, player_dets.tracker_id)):
                    team  = team_cache.get(int(tid), -1)
                    color = team_color(team)
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            annotated = label_annotator.annotate(annotated, player_dets, labels)

            # Pelota
            if len(ball_dets) > 0:
                for box in ball_dets.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    r = max(8, (x2 - x1) // 2)
                    cv2.circle(annotated, (cx, cy), r, BALL_COLOR, -1)
                    cv2.circle(annotated, (cx, cy), r, (255, 255, 255), 1)

            # Mini-mapa
            if show_mm and player_dets.tracker_id is not None and len(court_coords) > 0:
                mm = mini_map.render(
                    player_dets.tracker_id, court_coords, team_cache, ball_court
                )
                mm_h, mm_w = mm.shape[:2]
                margin = 10
                oy, ox = annotated.shape[0] - mm_h - margin, annotated.shape[1] - mm_w - margin
                # borde
                cv2.rectangle(annotated, (ox - 2, oy - 2),
                              (ox + mm_w + 2, oy + mm_h + 2), (200, 200, 200), 1)
                annotated[oy:oy + mm_h, ox:ox + mm_w] = mm

            # HUD
            elapsed = time.time() - t_start
            real_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            players_a = sum(1 for v in team_cache.values() if v == 0)
            players_b = sum(1 for v in team_cache.values() if v == 1)

            hud_lines = [
                f"Frame {frame_idx:04d}  |  {real_fps:.1f} fps",
                f"Detectados: {len(player_dets)} jugadores  |  Equipo A: {players_a}  Equipo B: {players_b}",
                f"H=heatmap({'ON' if show_hm else 'OFF'})  M=mapa({'ON' if show_mm else 'OFF'})  P=pausa  S=screenshot  Q=salir",
            ]
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 20 + 22 * len(hud_lines)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
            for i, line in enumerate(hud_lines):
                cv2.putText(annotated, line, (10, 20 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

            if sink_ctx:
                sink_ctx.write_frame(annotated)

            cv2.imshow("Basketball Analytics v2", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                show_hm = not show_hm
            elif key == ord("m"):
                show_mm = not show_mm
            elif key == ord("p"):
                paused = True
                print("Pausado. Presioná P para continuar.")
            elif key == ord("s"):
                screenshot_path = f"screenshot_frame_{frame_idx:04d}.jpg"
                cv2.imwrite(screenshot_path, annotated)
                print(f"Screenshot guardado: {screenshot_path}")

            frame_idx += 1

    if sink_ctx:
        sink_ctx.__exit__(None, None, None)

    cv2.destroyAllWindows()

    # --- Resumen final ---
    _print_summary(zone_log, team_cache, fps, stem)


def _print_summary(
    zone_log: dict,
    team_cache: dict,
    fps: float,
    stem: str,
) -> None:
    print(f"\n{'='*70}")
    print("  RESUMEN — Tiempo en zona por jugador")
    print(f"{'='*70}")
    header = f"{'ID':>4} | {'Equipo':>6} | {'Pintura':>10} | {'3pt zone':>10} | {'Medioc.':>9} | {'Open':>8}"
    print(header)
    print("-" * 70)

    for tid in sorted(zone_log.keys()):
        zones   = zone_log[tid]
        team    = team_cache.get(tid, -1)
        team_s  = ("A" if team == 0 else "B") if team in (0, 1) else "?"
        total   = max(1, sum(zones.values()))
        paint   = zones.get("paint_left", 0)  + zones.get("paint_right", 0)
        three   = zones.get("three_left", 0)  + zones.get("three_right", 0)
        mid     = zones.get("midcourt", 0)
        open_c  = zones.get("open_court", 0)

        def fmt(n):
            return f"{n/fps:4.1f}s ({n/total*100:3.0f}%)"

        print(f"{tid:>4} | {team_s:>6} | {fmt(paint):>10} | {fmt(three):>10} | {fmt(mid):>9} | {fmt(open_c):>8}")

    # Guardar JSON
    summary = {
        str(tid): {
            "team": ("A" if team_cache.get(tid) == 0 else "B") if team_cache.get(tid) in (0, 1) else "?",
            "zones_frames": dict(zones),
            "zones_seconds": {k: round(v / fps, 2) for k, v in zones.items()},
        }
        for tid, zones in zone_log.items()
    }
    json_path = f"{stem}_zone_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResumen guardado en: {json_path}")
    print(f"Tracking CSV en:     {stem}_tracking.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Basketball Analytics POC v2 — YOLO + ByteTrack + CLIP"
    )
    parser.add_argument("--source", required=True, help="Video de entrada")
    parser.add_argument("--output", "-o", default=None, help="Video de salida (opcional)")
    parser.add_argument("--court-config", default=None,
                        help="JSON de calibración de cancha (court_config.json)")
    parser.add_argument("--weights", default="yolo11n.pt",
                        help="Pesos del modelo YOLO (default: yolo11n.pt — se descarga automático)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou",  type=float, default=0.50, help="IOU threshold")
    parser.add_argument("--team-a", default="basketball player wearing white jersey",
                        help="Prompt CLIP para equipo A")
    parser.add_argument("--team-b", default="basketball player wearing dark jersey",
                        help="Prompt CLIP para equipo B")
    parser.add_argument("--clip-device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device para CLIP (default: cpu)")
    parser.add_argument("--no-heatmap", action="store_true", help="Deshabilitar heatmap inicial")
    parser.add_argument("--no-minimap", action="store_true", help="Deshabilitar mini-mapa inicial")
    args = parser.parse_args()

    main(
        source_video_path=args.source,
        target_video_path=args.output,
        court_config_path=args.court_config,
        model_weights=args.weights,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        team_a_prompt=args.team_a,
        team_b_prompt=args.team_b,
        clip_device=args.clip_device,
        show_heatmap=not args.no_heatmap,
        show_minimap=not args.no_minimap,
    )
