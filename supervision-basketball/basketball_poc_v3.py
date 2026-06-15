"""
Basketball Analytics POC v3
----------------------------
Uso:
    python basketball_poc_v3.py --source video.mp4
    python basketball_poc_v3.py --source video.mp4 --output resultado.mp4
    python basketball_poc_v3.py --source video.mp4 --weights yolo11x.pt

La cancha se detecta automáticamente en los primeros frames.
Se muestra una preview 3 segundos — presioná ENTER para continuar.

Controles durante el video:
  Q  — salir
  P  — pausar / continuar
  S  — guardar screenshot
  H  — toggle heatmap
  M  — toggle mini-mapa
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

import supervision as sv

from ball_tracker import BallTracker, RIM_LEFT, RIM_RIGHT
from video_reader import get_video_info, frames_generator as pyav_frames
from court_detector import CourtDetector
from event_engine import EventEngine, EventType, GameEvent
from stats_collector import StatsCollector

# ---------------------------------------------------------------------------
# Dimensiones reales de cancha NBA (cm)
# ---------------------------------------------------------------------------
COURT_W = 2865
COURT_H = 1524

# Colores
TEAM_A  = (230, 230, 230)
TEAM_B  = (80,   80, 220)
BALL_C  = (0,   160, 255)
EVENT_COLORS = {
    EventType.POSSESSION: (0, 255, 200),
    EventType.PASS:       (255, 200,  0),
    EventType.SHOT:       (0,  100, 255),
    EventType.BASKET:     (0,  255,   0),
    EventType.LOOSE_BALL: (0,  200, 255),
    EventType.DEAD_BALL:  (120,120, 120),
}


# ---------------------------------------------------------------------------
# ViewTransformer
# ---------------------------------------------------------------------------
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.M     = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))
        self.M_inv = cv2.getPerspectiveTransform(target.astype(np.float32), source.astype(np.float32))

    def to_court(self, pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return pts
        return cv2.perspectiveTransform(pts.reshape(-1,1,2).astype(np.float32), self.M).reshape(-1,2)

    def to_pixel(self, pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return pts
        return cv2.perspectiveTransform(pts.reshape(-1,1,2).astype(np.float32), self.M_inv).reshape(-1,2)


# ---------------------------------------------------------------------------
# Asignación de equipos por brillo del jersey — k-means adaptativo
# ---------------------------------------------------------------------------
def _jersey_brightness(frame: np.ndarray, box: np.ndarray) -> float:
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    crop = frame[max(0, y1 + h//4) : max(0, y1 + h*3//4), max(0, x1):x2]
    if crop.size == 0:
        return 128.0
    return float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))


class TeamAssigner:
    """
    Asigna equipos usando k-means adaptativo sobre el brillo del jersey.
    Usa threshold fijo hasta tener suficientes jugadores; luego re-ajusta
    los centros con k-means cada vez que aparece un jugador nuevo.
    """

    MIN_PLAYERS = 6        # mínimo de jugadores para activar k-means
    REFIT_EVERY = 60       # re-ajustar centros cada N jugadores nuevos

    def __init__(self) -> None:
        self._brightness: dict[int, float] = {}   # tracker_id → brillo
        self._teams:      dict[int, int]   = {}   # tracker_id → equipo (0 o 1)
        self._centers:    tuple[float, float] | None = None   # (dark, bright)
        self._new_since_refit = 0

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, tracker_ids: np.ndarray,
               boxes: np.ndarray) -> None:
        """Registrar jugadores nuevos y re-ajustar k-means si corresponde."""
        new_player = False
        for tid, box in zip(tracker_ids, boxes):
            if int(tid) not in self._brightness:
                self._brightness[int(tid)] = _jersey_brightness(frame, box)
                new_player = True
                self._new_since_refit += 1

        if new_player and len(self._brightness) >= self.MIN_PLAYERS:
            if self._new_since_refit >= 1:
                self._refit()
                self._new_since_refit = 0

    def get(self, tracker_id: int) -> int:
        """Devuelve el equipo asignado (0 o 1). Si aún no clasificado, usa fallback."""
        if tracker_id in self._teams:
            return self._teams[tracker_id]
        # Fallback hasta tener k-means: brillo > 110 → equipo 0
        b = self._brightness.get(tracker_id, 128.0)
        return 0 if b > 110 else 1

    # ------------------------------------------------------------------
    def _refit(self) -> None:
        """K-means manual con k=2 sobre todos los brillos conocidos."""
        values = np.array(list(self._brightness.values()), dtype=float)
        # Inicializar centros con percentiles 25 y 75 para robustez
        c0 = float(np.percentile(values, 25))
        c1 = float(np.percentile(values, 75))
        if abs(c1 - c0) < 5:          # jerseys indistinguibles → no re-ajustar
            return
        for _ in range(20):
            labels = (np.abs(values - c1) < np.abs(values - c0)).astype(int)
            new_c0 = float(values[labels == 0].mean()) if (labels == 0).any() else c0
            new_c1 = float(values[labels == 1].mean()) if (labels == 1).any() else c1
            if abs(new_c0 - c0) < 0.1 and abs(new_c1 - c1) < 0.1:
                break
            c0, c1 = new_c0, new_c1
        self._centers = (c0, c1)   # (dark_center, bright_center)
        # Re-asignar todos los jugadores conocidos
        for tid, b in self._brightness.items():
            self._teams[tid] = 0 if abs(b - c1) < abs(b - c0) else 1


# ---------------------------------------------------------------------------
# Mini-mapa
# ---------------------------------------------------------------------------
class MiniMap:
    W, H = 420, 224

    def __init__(self) -> None:
        self.sx = self.W / COURT_W
        self.sy = self.H / COURT_H
        self._base = self._make_base()

    def _p(self, x: float, y: float) -> tuple[int,int]:
        return int(np.clip(x*self.sx, 0, self.W-1)), int(np.clip(y*self.sy, 0, self.H-1))

    def _make_base(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), (45,35,25), np.uint8)
        lc = (180,180,180)
        cv2.rectangle(img, (0,0), (self.W-1,self.H-1), lc, 1)
        cv2.rectangle(img, self._p(0,427),  self._p(488,1097),  (40,40,70), -1)
        cv2.rectangle(img, self._p(2377,427), self._p(2865,1097), (40,40,70), -1)
        cv2.rectangle(img, self._p(0,427),  self._p(488,1097),  lc, 1)
        cv2.rectangle(img, self._p(2377,427), self._p(2865,1097), lc, 1)
        cv2.line(img, self._p(COURT_W/2,0), self._p(COURT_W/2,COURT_H), lc, 1)
        cx, cy = self._p(COURT_W/2, COURT_H/2)
        cv2.circle(img, (cx,cy), int(183*self.sx), lc, 1)
        for rim in (RIM_LEFT, RIM_RIGHT):
            cv2.circle(img, self._p(*rim), max(2,int(23*self.sx)), (0,120,255), 2)
        return img

    def render(
        self,
        tracker_ids:  np.ndarray,
        court_coords: np.ndarray,
        teams:        dict[int,int],
        ball:         np.ndarray | None,
        events:       list[GameEvent],
    ) -> np.ndarray:
        canvas = self._base.copy()
        for ev in events[-3:]:
            if ev.ball_pos is not None:
                px, py = self._p(*ev.ball_pos)
                c = EVENT_COLORS.get(ev.type, (255,255,255))
                cv2.circle(canvas, (px,py), 10, c, 1)
                cv2.putText(canvas, ev.type.name[:4], (px+3,py-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, c, 1)
        for tid, (cx,cy) in zip(tracker_ids, court_coords):
            c = TEAM_A if teams.get(int(tid))==0 else TEAM_B
            px, py = self._p(cx, cy)
            cv2.circle(canvas, (px,py), 6, c, -1)
            cv2.circle(canvas, (px,py), 6, (0,0,0), 1)
            cv2.putText(canvas, str(tid), (px+5,py+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, c, 1)
        if ball is not None:
            bx, by = self._p(*ball)
            cv2.circle(canvas, (bx,by), 7, BALL_C, -1)
            cv2.circle(canvas, (bx,by), 7, (255,255,255), 1)
        return canvas


# ---------------------------------------------------------------------------
# Feed de eventos en pantalla
# ---------------------------------------------------------------------------
class EventFeed:
    def __init__(self, fps: float, max_n: int = 6, ttl_s: float = 4.0) -> None:
        self.ttl = int(ttl_s * fps)
        self.items: deque[tuple[GameEvent,int]] = deque(maxlen=max_n)

    def add(self, ev: GameEvent, frame: int) -> None:
        self.items.append((ev, frame))

    def render(self, img: np.ndarray, frame: int) -> np.ndarray:
        active = [(ev,f) for ev,f in self.items if frame - f < self.ttl]
        if not active:
            return img

        x0 = img.shape[1] - 345
        y0 = 80
        lh = 26

        ov = img.copy()
        cv2.rectangle(ov, (x0-8, y0-22), (img.shape[1]-8, y0+lh*len(active)+4), (0,0,0), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        cv2.putText(img, "EVENTOS", (x0, y0-7), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180,180,180), 1)

        for i, (ev, f) in enumerate(reversed(active)):
            ratio = 1.0 - (frame - f) / self.ttl
            c = tuple(int(ch*ratio) for ch in EVENT_COLORS.get(ev.type,(200,200,200)))
            ts = ("A" if ev.team==0 else "B") if ev.team in (0,1) else ""
            ps = f"#{ev.player_id}" if ev.player_id is not None else ""
            if ev.type == EventType.BASKET:
                txt = f"CANASTA  {ps} [{ts}]"
            elif ev.type == EventType.PASS and ev.receiver_id:
                txt = f"PASE  {ps} → #{ev.receiver_id}"
            elif ev.type == EventType.SHOT:
                txt = f"TIRO  {ps} [{ts}]  {ev.speed_ms:.1f}m/s"
            else:
                txt = f"{ev.type.name}  {ps} [{ts}]"
            cv2.putText(img, txt, (x0, y0+i*lh), cv2.FONT_HERSHEY_SIMPLEX, 0.47, c, 1)

        return img


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def _parse_corners(s: str) -> np.ndarray:
    """Parsea '0,684;1280,684;1280,36;0,36' → array (4,2)."""
    pts = []
    for part in s.split(";"):
        x, y = part.strip().split(",")
        pts.append([float(x), float(y)])
    if len(pts) != 4:
        raise ValueError(f"Se esperan 4 esquinas separadas por ';', se recibieron {len(pts)}")
    return np.array(pts, dtype=np.float32)


def main(
    source: str,
    output: str | None = None,
    weights: str = "yolo11n.pt",
    conf: float = 0.35,
    ball_conf: float = 0.15,
    iou:  float = 0.50,
    show_heatmap: bool = True,
    show_minimap: bool = True,
    headless: bool = False,
    max_frames: int | None = None,
    court_corners: str | None = None,
) -> None:
    print(f"\n{'='*60}")
    print("  Basketball Analytics POC v3")
    print(f"{'='*60}")
    print(f"  Video  : {source}")
    print(f"  Modelo : {weights}")
    print(f"{'='*60}\n")

    # ── 1. Detectar cancha ───────────────────────────────────────────
    if court_corners:
        corners = _parse_corners(court_corners)
        print(f"Cancha: esquinas forzadas por CLI\n  {corners.astype(int).tolist()}\n")
    else:
        print("Detectando cancha...")
        detector = CourtDetector()
        corners  = detector.detect_from_video(source, preview=not headless)
        print(f"  Esquinas: {corners.astype(int).tolist()}\n")

    target = np.array([
        [0,       COURT_H],
        [COURT_W, COURT_H],
        [COURT_W, 0],
        [0,       0],
    ], dtype=np.float32)
    transformer = ViewTransformer(corners, target)

    # ── 2. Cargar modelo ─────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        print(f"Cargando {weights} ...")
        model = YOLO(weights)
    except ImportError:
        raise SystemExit("Instalá ultralytics:  pip install ultralytics")

    # ── 3. Supervision / tracking ────────────────────────────────────
    video_info = get_video_info(source)   # PyAV — soporta AV1, H.264, H.265, VP9…
    fps = video_info.fps
    print(f"Video: {video_info.width}×{video_info.height} @ {fps:.1f} fps\n")

    # Fix 5: track_buffer más largo + mínimo 3 frames consecutivos para confirmar track
    byte_track = sv.ByteTrack(
        frame_rate=fps,
        track_activation_threshold=conf,
        lost_track_buffer=int(fps * 3),        # 3s antes de dar track por perdido
        minimum_consecutive_frames=3,           # evitar tracks efímeros
    )
    smoother   = sv.DetectionsSmoother()

    thick = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    tscl  = sv.calculate_optimal_text_scale(video_info.resolution_wh)

    trace_ann = sv.TraceAnnotator(thickness=max(1,thick-1),
                                  trace_length=int(fps*3),
                                  position=sv.Position.BOTTOM_CENTER)
    label_ann = sv.LabelAnnotator(text_scale=tscl*0.8,
                                  text_thickness=max(1,thick-1),
                                  text_position=sv.Position.TOP_CENTER,
                                  text_padding=3)
    heat_ann  = sv.HeatMapAnnotator(position=sv.Position.BOTTOM_CENTER,
                                    opacity=0.35, radius=50, kernel_size=31)

    # ── 4. Módulos de análisis ───────────────────────────────────────
    ball_tr = BallTracker(fps=fps)
    stats   = StatsCollector(fps=fps)
    engine  = EventEngine(fps=fps)
    feed    = EventFeed(fps=fps)
    minimap = MiniMap()
    recent_events: list[GameEvent] = []

    def on_event(ev: GameEvent) -> None:
        stats.record(ev)
        feed.add(ev, ev.frame_idx)
        recent_events.append(ev)
        print(str(ev))

    engine.on_event = on_event

    # ── 5. Exportación ───────────────────────────────────────────────
    stem     = Path(source).stem
    csv_sink = sv.CSVSink(f"{stem}_tracking.csv")
    sink_ctx = sv.VideoSink(output, video_info) if output else None

    team_assigner = TeamAssigner()
    team_cache: dict[int,int] = {}
    show_hm = show_heatmap
    show_mm = show_minimap
    paused  = False
    frames  = pyav_frames(source, end=max_frames)   # PyAV — soporta AV1
    if max_frames:
        print(f"  Límite: {max_frames} frames ({max_frames/fps:.0f}s de video)\n")

    print("Procesando... (Q=salir  P=pausa  S=screenshot  H=heatmap  M=mapa)\n")

    with csv_sink:
        for frame_idx, frame in enumerate(frames):

            # ── Pausa ──
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord("p"): paused = False
                elif key == ord("q"): break
                continue

            # ── Detección YOLO ──
            # Fix 1: correr con ball_conf (más bajo) para mejorar recall de pelota
            results  = model(frame, conf=min(conf, ball_conf), iou=iou, verbose=False)[0]
            all_dets = sv.Detections.from_ultralytics(results)
            p_dets   = all_dets[(all_dets.class_id == 0) &
                                 (all_dets.confidence >= conf)]    # personas: conf normal
            b_dets   = all_dets[(all_dets.class_id == 32) &
                                 (all_dets.confidence >= ball_conf)]  # pelota: conf baja

            # Filtro de tamaño/forma: pelota de básquet debe ser ~cuadrada y 6–80px de alto
            if len(b_dets) > 0:
                xyxy    = b_dets.xyxy
                heights = xyxy[:, 3] - xyxy[:, 1]
                widths  = xyxy[:, 2] - xyxy[:, 0]
                aspect  = np.where(heights > 0, widths / heights, 0.0)
                valid   = (heights >= 6) & (heights <= 80) & (aspect >= 0.5) & (aspect <= 2.0)
                b_dets  = b_dets[valid]

            # ── Tracking jugadores ──
            # Fix 4: ByteTrack PRIMERO (asigna tracker_id), smoother DESPUÉS
            p_dets = byte_track.update_with_detections(p_dets)
            p_dets = smoother.update_with_detections(p_dets)

            # Fix 3: k-means adaptativo para asignación de equipos
            if p_dets.tracker_id is not None:
                team_assigner.update(frame, p_dets.tracker_id, p_dets.xyxy)
                for tid in p_dets.tracker_id:
                    team_cache[int(tid)] = team_assigner.get(int(tid))

            # ── Coordenadas de cancha ──
            anchors      = p_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            court_coords = transformer.to_court(anchors) if len(anchors) > 0 else np.zeros((0,2))

            ball_pixel = ball_court = None
            if len(b_dets) > 0:
                best       = int(np.argmax(b_dets.confidence))
                ball_pixel = b_dets.get_anchors_coordinates(sv.Position.CENTER)[best]
                ball_court = transformer.to_court(ball_pixel.reshape(1,2))[0]

            # ── Motor de eventos ──
            ball_tr.update(frame_idx, ball_pixel, ball_court)
            ball_state = ball_tr.get_state(
                player_ids=p_dets.tracker_id if p_dets.tracker_id is not None else np.array([]),
                player_court_coords=court_coords,
            )
            engine.update(frame_idx, ball_state, team_cache)
            stats.tick(frame_idx)

            # ── Exportar CSV ──
            csv_sink.append(p_dets, custom_data={"frame": frame_idx})

            # ── Render ──
            vis = frame.copy()

            if show_hm:
                vis = heat_ann.annotate(vis, p_dets)

            vis = trace_ann.annotate(vis, p_dets)

            # Boxes con color de equipo
            if p_dets.tracker_id is not None:
                for box, tid in zip(p_dets.xyxy, p_dets.tracker_id):
                    c = TEAM_A if team_cache.get(int(tid))==0 else TEAM_B
                    x1,y1,x2,y2 = map(int,box)
                    cv2.rectangle(vis,(x1,y1),(x2,y2),c,thick)
                    # Resaltar poseedor
                    if engine.possessor_id == int(tid):
                        cv2.rectangle(vis,(x1-3,y1-3),(x2+3,y2+3),(0,255,180),2)

                labels = []
                for tid in p_dets.tracker_id:
                    t  = team_cache.get(int(tid),-1)
                    ts = "A" if t==0 else ("B" if t==1 else "?")
                    mk = " ●" if engine.possessor_id==int(tid) else ""
                    labels.append(f"#{tid}[{ts}]{mk}")
                vis = label_ann.annotate(vis, p_dets, labels)

            # Pelota + trayectoria
            hist = ball_tr.pixel_positions()
            if len(hist) >= 2:
                for i in range(1, len(hist)):
                    a = i / len(hist)
                    c = tuple(int(ch*a) for ch in BALL_C)
                    cv2.line(vis, tuple(hist[i-1].astype(int)), tuple(hist[i].astype(int)), c, 2)

            if ball_pixel is not None:
                bx, by = int(ball_pixel[0]), int(ball_pixel[1])
                cv2.circle(vis,(bx,by),14,BALL_C,-1)
                cv2.circle(vis,(bx,by),14,(255,255,255),2)
                if ball_state.speed_ms > 0.5:
                    cv2.putText(vis, f"{ball_state.speed_ms:.1f}m/s",
                                (bx+16,by-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BALL_C, 2)

            # Mini-mapa
            if show_mm and p_dets.tracker_id is not None and len(court_coords)>0:
                mm = minimap.render(
                    p_dets.tracker_id, court_coords, team_cache,
                    ball_court, recent_events[-5:]
                )
                mh, mw = mm.shape[:2]
                oy = vis.shape[0]-mh-10
                cv2.rectangle(vis,(8,oy-2),(10+mw+2,oy+mh+2),(180,180,180),1)
                vis[oy:oy+mh, 10:10+mw] = mm

            # Feed de eventos
            vis = feed.render(vis, frame_idx)

            # HUD
            phase_s = engine.phase.name
            speed_s = f"Pelota: {ball_state.speed_ms:.1f}m/s" if ball_state.detected else "Pelota: no detectada"
            poss_s  = ""
            if engine.possessor_id is not None:
                t = ("Eq.A" if engine.possessor_team==0 else "Eq.B") if engine.possessor_team in (0,1) else ""
                poss_s = f"POSESION: #{engine.possessor_id} {t}"

            ov = vis.copy()
            cv2.rectangle(ov,(0,0),(vis.shape[1],64),(0,0,0),-1)
            cv2.addWeighted(ov, 0.5, vis, 0.5, 0, vis)
            cv2.putText(vis, f"Frame {frame_idx:04d}  |  {phase_s}  |  {speed_s}",
                        (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,(220,220,220),1)
            if poss_s:
                cv2.putText(vis, poss_s, (10,42), cv2.FONT_HERSHEY_SIMPLEX, 0.52,(0,255,180),1)

            if sink_ctx:
                sink_ctx.write_frame(vis)

            if not headless:
                cv2.imshow("Basketball Analytics v3", vis)
                key = cv2.waitKey(1) & 0xFF
                if   key == ord("q"): break
                elif key == ord("p"): paused = True
                elif key == ord("h"): show_hm = not show_hm
                elif key == ord("m"): show_mm = not show_mm
                elif key == ord("s"):
                    p = f"screenshot_{frame_idx:04d}.jpg"
                    cv2.imwrite(p, vis)
                    print(f"Screenshot: {p}")
            elif frame_idx % 100 == 0:
                print(f"  frame {frame_idx:04d}...")

    if sink_ctx:
        sink_ctx.__exit__(None, None, None)
    if not headless:
        cv2.destroyAllWindows()

    stats.print_summary()
    stats.export_json(f"{stem}_stats.json")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Basketball Analytics v3 — detección automática de cancha + motor de eventos"
    )
    ap.add_argument("--source",   required=True,        help="Video de entrada (.mp4)")
    ap.add_argument("--output",   default=None,         help="Video de salida anotado (opcional)")
    ap.add_argument("--weights",  default="yolo11n.pt", help="Pesos YOLO (default: yolo11n.pt)")
    ap.add_argument("--conf",     type=float, default=0.35)
    ap.add_argument("--iou",      type=float, default=0.50)
    ap.add_argument("--no-heatmap",  action="store_true")
    ap.add_argument("--no-minimap",  action="store_true")
    ap.add_argument("--headless",    action="store_true",
                    help="Batch mode: skip all cv2 windows, print progress every 100 frames")
    ap.add_argument("--max-frames",  type=int, default=None,
                    help="Procesar sólo los primeros N frames (útil para videos largos)")
    ap.add_argument("--ball-conf",   type=float, default=0.15,
                    help="Confianza mínima para detectar la pelota (default: 0.15, más bajo = más recall)")
    ap.add_argument("--court-corners", default=None,
                    help="Esquinas de la cancha (override manual): 'x1,y1;x2,y2;x3,y3;x4,y4' "
                         "en orden INF-IZQ;INF-DER;SUP-DER;SUP-IZQ")
    args = ap.parse_args()

    main(
        source         = args.source,
        output         = args.output,
        weights        = args.weights,
        conf           = args.conf,
        ball_conf      = args.ball_conf,
        iou            = args.iou,
        show_heatmap   = not args.no_heatmap,
        show_minimap   = not args.no_minimap,
        headless       = args.headless,
        max_frames     = args.max_frames,
        court_corners  = args.court_corners,
    )
