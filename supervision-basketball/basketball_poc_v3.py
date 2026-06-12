"""
Basketball Analytics POC v3
----------------------------
Integra detección, tracking, y el motor de eventos de pelota.

Eventos detectados:
  POSSESSION  — jugador con la pelota
  PASS        — pase entre jugadores
  SHOT        — tiro al aro
  BASKET      — canasta
  LOOSE_BALL  — pelota suelta
  DEAD_BALL   — pelota quieta

Uso:
    # Video sintético (sin modelo, ideal para testear la lógica)
    python basketball_poc_v3.py --source test_basketball.mp4 --synthetic-mode

    # Con YOLO real
    python basketball_poc_v3.py --source video.mp4

    # Con calibración de cancha
    python basketball_poc_v3.py --source video.mp4 --court-config court_config.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

import supervision as sv

from ball_tracker import BallTracker, RIM_LEFT, RIM_RIGHT
from event_engine import EventEngine, EventType, GameEvent
from stats_collector import StatsCollector

# ---------------------------------------------------------------------------
# Configuración de cancha por defecto
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_POINTS = np.array([
    [100,  620],
    [1180, 620],
    [1080, 100],
    [200,  100],
], dtype=np.float32)

COURT_WIDTH_CM  = 2865
COURT_HEIGHT_CM = 1524

TEAM_A_COLOR = (230, 230, 230)
TEAM_B_COLOR = (80,   80, 220)
BALL_COLOR   = (0,   160, 255)
EVENT_COLORS = {
    EventType.POSSESSION: (0, 255, 200),
    EventType.PASS:       (0, 200, 255),
    EventType.SHOT:       (0, 100, 255),
    EventType.BASKET:     (0, 255,   0),
    EventType.LOOSE_BALL: (0, 180, 255),
    EventType.DEAD_BALL:  (120, 120, 120),
}

# ---------------------------------------------------------------------------
# ViewTransformer
# ---------------------------------------------------------------------------
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.M = cv2.getPerspectiveTransform(
            source.astype(np.float32),
            target.astype(np.float32),
        )
        self.M_inv = cv2.getPerspectiveTransform(
            target.astype(np.float32),
            source.astype(np.float32),
        )

    def to_court(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)

    def to_pixel(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.M_inv).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Asignación de equipos por color (sin CLIP para simplificar)
# ---------------------------------------------------------------------------
def assign_team_by_brightness(frame: np.ndarray, box: np.ndarray) -> int:
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    # Recortar el tercio del torso
    ty1, ty2 = y1 + h // 4, y1 + h * 3 // 4
    crop = frame[max(0,ty1):ty2, max(0,x1):x2]
    if crop.size == 0:
        return 0
    brightness = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
    return 0 if brightness > 110 else 1


# ---------------------------------------------------------------------------
# Mini-mapa con eventos
# ---------------------------------------------------------------------------
class CourtMiniMap:
    W, H = 420, 224

    def __init__(self, transformer: ViewTransformer) -> None:
        self.sx = self.W / COURT_WIDTH_CM
        self.sy = self.H / COURT_HEIGHT_CM
        self.transformer = transformer
        self._base = self._draw_base()

    def _px(self, x: float, y: float) -> tuple[int, int]:
        return int(np.clip(x * self.sx, 0, self.W - 1)), int(np.clip(y * self.sy, 0, self.H - 1))

    def _draw_base(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), (45, 35, 25), dtype=np.uint8)
        lc = (180, 180, 180)
        cv2.rectangle(img, (0, 0), (self.W-1, self.H-1), lc, 1)
        # pinturas
        cv2.rectangle(img, self._px(0, 427), self._px(488, 1097), (40, 40, 70), -1)
        cv2.rectangle(img, self._px(2377, 427), self._px(2865, 1097), (40, 40, 70), -1)
        cv2.rectangle(img, self._px(0, 427), self._px(488, 1097), lc, 1)
        cv2.rectangle(img, self._px(2377, 427), self._px(2865, 1097), lc, 1)
        # línea central
        cv2.line(img, self._px(COURT_WIDTH_CM/2, 0), self._px(COURT_WIDTH_CM/2, COURT_HEIGHT_CM), lc, 1)
        # círculo central
        cx, cy = self._px(COURT_WIDTH_CM/2, COURT_HEIGHT_CM/2)
        cv2.circle(img, (cx, cy), int(183*self.sx), lc, 1)
        # aros
        for rim in (RIM_LEFT, RIM_RIGHT):
            cv2.circle(img, self._px(*rim), max(2, int(23*self.sx)), (0, 120, 255), 2)
        return img

    def render(
        self,
        tracker_ids:  np.ndarray,
        court_coords: np.ndarray,
        team_cache:   dict[int, int],
        ball_court:   np.ndarray | None,
        recent_events: list[GameEvent],
    ) -> np.ndarray:
        canvas = self._base.copy()

        # Eventos recientes sobre el mapa
        for ev in recent_events[-3:]:
            if ev.ball_pos is not None:
                epx, epy = self._px(*ev.ball_pos)
                color = EVENT_COLORS.get(ev.type, (255, 255, 255))
                cv2.circle(canvas, (epx, epy), 10, color, 1)
                cv2.putText(canvas, ev.type.name[:4], (epx+3, epy-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

        # Jugadores
        for tid, (cx, cy) in zip(tracker_ids, court_coords):
            team = team_cache.get(int(tid), -1)
            color = TEAM_A_COLOR if team == 0 else TEAM_B_COLOR
            px, py = self._px(cx, cy)
            cv2.circle(canvas, (px, py), 6, color, -1)
            cv2.circle(canvas, (px, py), 6, (0, 0, 0), 1)
            cv2.putText(canvas, str(tid), (px+5, py+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

        # Pelota
        if ball_court is not None:
            bx, by = self._px(*ball_court)
            cv2.circle(canvas, (bx, by), 7, BALL_COLOR, -1)
            cv2.circle(canvas, (bx, by), 7, (255, 255, 255), 1)

        return canvas


# ---------------------------------------------------------------------------
# Overlay de eventos recientes en pantalla
# ---------------------------------------------------------------------------
class EventFeed:
    """Muestra los últimos N eventos en pantalla como un feed."""

    def __init__(self, max_events: int = 6, ttl_seconds: float = 4.0, fps: float = 30) -> None:
        self.max = max_events
        self.ttl_frames = int(ttl_seconds * fps)
        self.items: deque[tuple[GameEvent, int]] = deque(maxlen=max_events)  # (event, frame_added)

    def add(self, event: GameEvent, current_frame: int) -> None:
        self.items.append((event, current_frame))

    def render(self, frame: np.ndarray, current_frame: int) -> np.ndarray:
        # Filtrar eventos expirados
        active = [(ev, f) for ev, f in self.items if current_frame - f < self.ttl_frames]
        if not active:
            return frame

        x_start = frame.shape[1] - 340
        y_start = 80
        line_h  = 26

        # Fondo
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x_start - 8, y_start - 20),
                      (frame.shape[1] - 8, y_start + line_h * len(active) + 4),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "EVENTOS", (x_start, y_start - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        for i, (ev, f) in enumerate(reversed(active)):
            age_ratio = 1.0 - (current_frame - f) / self.ttl_frames
            color = EVENT_COLORS.get(ev.type, (200, 200, 200))
            alpha_color = tuple(int(c * age_ratio) for c in color)

            team_s = ("A" if ev.team == 0 else "B") if ev.team in (0, 1) else ""
            player_s = f"#{ev.player_id}" if ev.player_id is not None else ""

            if ev.type == EventType.PASS and ev.receiver_id:
                text = f"{ev.type.name}  {player_s} → #{ev.receiver_id} [{team_s}]"
            elif ev.type == EventType.BASKET:
                text = f"CANASTA!  {player_s} [{team_s}]  🏀"
            elif ev.type == EventType.SHOT:
                text = f"TIRO  {player_s} [{team_s}]  {ev.speed_ms:.1f}m/s"
            else:
                text = f"{ev.type.name}  {player_s} [{team_s}]"

            cv2.putText(frame, text, (x_start, y_start + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, alpha_color, 1)

        return frame


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main(
    source_video_path:  str,
    target_video_path:  str | None = None,
    court_config_path:  str | None = None,
    model_weights:      str = "yolo11n.pt",
    confidence:         float = 0.35,
    iou:                float = 0.50,
    synthetic_mode:     bool = False,
) -> None:
    print(f"\n{'='*60}")
    print("  Basketball Analytics POC v3 — Motor de Eventos")
    print(f"{'='*60}\n")

    # --- Cancha ---
    source_pts = DEFAULT_SOURCE_POINTS
    if court_config_path and Path(court_config_path).exists():
        cfg = json.load(open(court_config_path))
        source_pts = np.array(cfg["source_points"], dtype=np.float32)

    target_pts = np.array([
        [0,              COURT_HEIGHT_CM],
        [COURT_WIDTH_CM, COURT_HEIGHT_CM],
        [COURT_WIDTH_CM, 0],
        [0,              0],
    ], dtype=np.float32)

    transformer = ViewTransformer(source_pts, target_pts)

    # --- Video ---
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps
    print(f"Video: {video_info.width}×{video_info.height} @ {fps:.1f}fps")

    # --- Motor de eventos y estadísticas ---
    stats   = StatsCollector(fps=fps)
    engine  = EventEngine(fps=fps, on_event=stats.record)
    ball_tr = BallTracker(fps=fps, history_seconds=2.0)
    mini_map= CourtMiniMap(transformer)
    feed    = EventFeed(max_events=6, ttl_seconds=4.0, fps=fps)
    recent_events: list[GameEvent] = []

    def on_event(ev: GameEvent) -> None:
        feed.add(ev, ev.frame_idx)
        recent_events.append(ev)
        print(str(ev))

    engine.on_event = on_event

    # --- Modelo (opcional en modo sintético) ---
    if not synthetic_mode:
        try:
            from ultralytics import YOLO
            model = YOLO(model_weights)
            print(f"Modelo cargado: {model_weights}")
        except ImportError:
            print("ultralytics no encontrado — activando modo sintético")
            synthetic_mode = True

    # --- Tracker y anotadores ---
    byte_track = sv.ByteTrack(frame_rate=fps, track_activation_threshold=confidence)
    smoother   = sv.DetectionsSmoother()

    thickness  = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)

    trace_ann = sv.TraceAnnotator(thickness=max(1, thickness-1),
                                  trace_length=int(fps*3),
                                  position=sv.Position.BOTTOM_CENTER)
    label_ann = sv.LabelAnnotator(text_scale=text_scale*0.8,
                                  text_thickness=max(1, thickness-1),
                                  text_position=sv.Position.TOP_CENTER,
                                  text_padding=3)

    # --- Sink de salida ---
    stem     = Path(source_video_path).stem
    sink_ctx = sv.VideoSink(target_video_path, video_info) if target_video_path else None

    team_cache: dict[int, int] = {}
    frame_generator = sv.get_video_frames_generator(source_video_path)

    print("\nProcesando... (Q=salir, P=pausa, S=screenshot)\n")
    paused = False

    for frame_idx, frame in enumerate(frame_generator):
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key == ord("p"): paused = False
            elif key == ord("q"): break
            continue

        # ---- Detección ----
        if synthetic_mode:
            player_dets, ball_pixel, ball_conf = _synthetic_detections(frame_idx, fps)
        else:
            results     = model(frame, conf=confidence, iou=iou, verbose=False)[0]
            all_dets    = sv.Detections.from_ultralytics(results)
            player_dets = all_dets[all_dets.class_id == 0]
            ball_dets   = all_dets[all_dets.class_id == 32]
            ball_pixel  = None
            ball_conf   = 0.0
            if len(ball_dets) > 0:
                anchors   = ball_dets.get_anchors_coordinates(sv.Position.CENTER)
                best      = int(np.argmax(ball_dets.confidence))
                ball_pixel = anchors[best]
                ball_conf  = float(ball_dets.confidence[best])

        # ---- Tracking de jugadores ----
        player_dets = smoother.update_with_detections(player_dets)
        player_dets = byte_track.update_with_detections(player_dets)

        # Asignación de equipos por brillo
        if player_dets.tracker_id is not None:
            for tid, box in zip(player_dets.tracker_id, player_dets.xyxy):
                if tid not in team_cache:
                    team_cache[tid] = assign_team_by_brightness(frame, box)

        # ---- Coordenadas de cancha ----
        player_anchors = player_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        court_coords   = transformer.to_court(player_anchors) if len(player_anchors) > 0 else np.zeros((0, 2))

        # Pelota en coordenadas de cancha
        ball_court = None
        if ball_pixel is not None:
            ball_court = transformer.to_court(ball_pixel.reshape(1, 2))[0]

        # ---- BallTracker + EventEngine ----
        ball_tr.update(frame_idx, ball_pixel, ball_court, ball_conf)
        ball_state = ball_tr.get_state(
            player_ids    = player_dets.tracker_id if player_dets.tracker_id is not None else np.array([]),
            player_court_coords = court_coords,
        )
        events = engine.update(frame_idx, ball_state, team_cache)
        stats.tick(frame_idx)

        # ---- Render ----
        annotated = frame.copy()

        # Traces de jugadores
        annotated = trace_ann.annotate(annotated, player_dets)

        # Boxes de jugadores con color por equipo
        if player_dets.tracker_id is not None:
            for box, tid in zip(player_dets.xyxy, player_dets.tracker_id):
                color = TEAM_A_COLOR if team_cache.get(int(tid)) == 0 else TEAM_B_COLOR
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Labels de jugadores
        if player_dets.tracker_id is not None:
            labels = []
            for tid in player_dets.tracker_id:
                t = team_cache.get(int(tid), -1)
                ts = "A" if t == 0 else ("B" if t == 1 else "?")
                is_possessor = engine.possessor_id == int(tid)
                marker = " ●" if is_possessor else ""
                labels.append(f"#{tid}[{ts}]{marker}")
            annotated = label_ann.annotate(annotated, player_dets, labels)

        # Pelota
        if ball_pixel is not None:
            bx, by = int(ball_pixel[0]), int(ball_pixel[1])
            cv2.circle(annotated, (bx, by), 14, BALL_COLOR, -1)
            cv2.circle(annotated, (bx, by), 14, (255,255,255), 2)
            # Mostrar velocidad sobre la pelota
            if ball_state.speed_ms > 0.5:
                cv2.putText(annotated, f"{ball_state.speed_ms:.1f}m/s",
                            (bx + 16, by - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, BALL_COLOR, 2)

        # Traza de trayectoria de la pelota
        ball_pix_hist = ball_tr.pixel_positions()
        if len(ball_pix_hist) >= 2:
            for i in range(1, len(ball_pix_hist)):
                alpha = i / len(ball_pix_hist)
                color = tuple(int(c * alpha) for c in BALL_COLOR)
                pt1 = tuple(ball_pix_hist[i-1].astype(int))
                pt2 = tuple(ball_pix_hist[i].astype(int))
                cv2.line(annotated, pt1, pt2, color, 2)

        # Resaltar poseedor
        if engine.possessor_id is not None and player_dets.tracker_id is not None:
            mask = player_dets.tracker_id == engine.possessor_id
            if mask.any():
                box = player_dets.xyxy[mask][0]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 200), 2)

        # Feed de eventos
        annotated = feed.render(annotated, frame_idx)

        # Mini-mapa
        if len(court_coords) > 0 and player_dets.tracker_id is not None:
            mm = mini_map.render(
                player_dets.tracker_id, court_coords, team_cache,
                ball_court, recent_events[-5:]
            )
            mh, mw = mm.shape[:2]
            ox, oy = 10, annotated.shape[0] - mh - 10
            cv2.rectangle(annotated, (ox-2, oy-2), (ox+mw+2, oy+mh+2), (180,180,180), 1)
            annotated[oy:oy+mh, ox:ox+mw] = mm

        # HUD superior
        _draw_hud(annotated, frame_idx, fps, engine, stats, ball_state)

        if sink_ctx:
            sink_ctx.write_frame(annotated)

        cv2.imshow("Basketball Analytics v3", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = True
        elif key == ord("s"):
            path = f"screenshot_{frame_idx:04d}.jpg"
            cv2.imwrite(path, annotated)
            print(f"Screenshot: {path}")

    if sink_ctx:
        sink_ctx.__exit__(None, None, None)
    cv2.destroyAllWindows()

    # --- Resumen final ---
    stats.print_summary()
    stats.export_json(f"{stem}_stats.json")


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def _draw_hud(frame, frame_idx, fps, engine, stats, ball_state):
    possession_s = ""
    if engine.possessor_id is not None:
        pid  = engine.possessor_id
        team = ("Eq.A" if engine.possessor_team == 0 else "Eq.B") if engine.possessor_team in (0,1) else ""
        possession_s = f"POSESION: #{pid} {team}"

    phase_s = engine.phase.name
    speed_s = f"Pelota: {ball_state.speed_ms:.1f}m/s" if ball_state.detected else "Pelota: no detectada"

    hud = [
        f"Frame {frame_idx:04d}  |  {phase_s}  |  {speed_s}",
        possession_s,
    ]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 18 + 22 * len(hud)), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, line in enumerate(hud):
        cv2.putText(frame, line, (10, 16 + 22*i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220,220,220), 1)


# ---------------------------------------------------------------------------
# Detecciones sintéticas para testear sin modelo
# ---------------------------------------------------------------------------
def _synthetic_detections(frame_idx: int, fps: float):
    """
    Simula 10 jugadores y una pelota con movimiento básico.
    Devuelve sv.Detections con xyxy, class_id, confidence.
    """
    import math

    t = frame_idx / fps

    # 10 jugadores estáticos con pequeño movimiento
    players_pos = []
    for i in range(10):
        angle = (i / 10) * 2 * math.pi + t * 0.3
        rx = 400 + i * 50
        ry = 300 + int(math.sin(angle) * 80)
        # ancho/alto del bounding box
        w, h = 40, 80
        players_pos.append([rx - w//2, ry - h//2, rx + w//2, ry + h//2])

    # Pelota: trayectoria simple de izquierda a derecha y rebote
    bx = 200 + int((frame_idx % 300) * 3.0)
    by = 360 + int(abs(math.sin(frame_idx * 0.25)) * -60)
    ball_pixel = np.array([float(bx), float(by)])

    xyxy = np.array(players_pos, dtype=np.float32)
    class_ids = np.zeros(len(players_pos), dtype=int)
    confs = np.ones(len(players_pos), dtype=float) * 0.9

    dets = sv.Detections(xyxy=xyxy, class_id=class_ids, confidence=confs)
    return dets, ball_pixel, 0.9


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Basketball Analytics v3 — Motor de eventos de pelota"
    )
    parser.add_argument("--source",        required=True,       help="Video de entrada")
    parser.add_argument("--output",  "-o", default=None,        help="Video de salida")
    parser.add_argument("--court-config",  default=None,        help="court_config.json")
    parser.add_argument("--weights",       default="yolo11n.pt",help="Pesos YOLO")
    parser.add_argument("--conf",  type=float, default=0.35)
    parser.add_argument("--iou",   type=float, default=0.50)
    parser.add_argument("--synthetic-mode", action="store_true",
                        help="Usar detecciones sintéticas (no requiere YOLO)")
    args = parser.parse_args()

    main(
        source_video_path = args.source,
        target_video_path = args.output,
        court_config_path = args.court_config,
        model_weights     = args.weights,
        confidence        = args.conf,
        iou               = args.iou,
        synthetic_mode    = args.synthetic_mode,
    )
