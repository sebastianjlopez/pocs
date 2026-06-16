"""
Herramienta interactiva para calibrar las esquinas de la cancha.

Uso:
    python calibrate_court.py --source video.mp4
    python calibrate_court.py --source video.mp4 --frame 60  # usar frame específico

Hacé click en las 4 esquinas de la cancha en este orden:
  1. Esquina inferior-izquierda
  2. Esquina inferior-derecha
  3. Esquina superior-derecha
  4. Esquina superior-izquierda

Guardado automáticamente en court_config.json.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CORNER_LABELS = [
    "1. Inferior-Izquierda",
    "2. Inferior-Derecha",
    "3. Superior-Derecha",
    "4. Superior-Izquierda",
]
CORNER_COLORS = [
    (0, 255, 255),    # amarillo
    (0, 165, 255),    # naranja
    (0, 255, 0),      # verde
    (255, 0, 255),    # magenta
]

INSTRUCTIONS = [
    "Hacé click en: INFERIOR-IZQUIERDA de la cancha",
    "Hacé click en: INFERIOR-DERECHA de la cancha",
    "Hacé click en: SUPERIOR-DERECHA de la cancha",
    "Hacé click en: SUPERIOR-IZQUIERDA de la cancha",
]


class CourtCalibrator:
    def __init__(self, frame: np.ndarray, output_path: str = "court_config.json"):
        self.original = frame.copy()
        self.frame = frame.copy()
        self.output_path = output_path
        self.points: list[tuple[int, int]] = []
        self.done = False

        # Dimensiones reales cancha NBA (cm)
        self.court_width_cm = 2865
        self.court_height_cm = 1524

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"  Punto {len(self.points)}: ({x}, {y}) — {CORNER_LABELS[len(self.points)-1]}")
            self._redraw()

        elif event == cv2.EVENT_MOUSEMOVE:
            self._redraw(cursor=(x, y))

    def _redraw(self, cursor: tuple[int, int] | None = None) -> None:
        self.frame = self.original.copy()

        # Dibujar puntos ya capturados
        for i, (px, py) in enumerate(self.points):
            cv2.circle(self.frame, (px, py), 8, CORNER_COLORS[i], -1)
            cv2.circle(self.frame, (px, py), 10, (255, 255, 255), 2)
            cv2.putText(self.frame, CORNER_LABELS[i], (px + 12, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CORNER_COLORS[i], 2)

        # Líneas entre puntos capturados
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(self.frame, self.points[i], self.points[i + 1],
                         (200, 200, 200), 1, cv2.LINE_AA)
        if len(self.points) == 4:
            cv2.line(self.frame, self.points[3], self.points[0],
                     (200, 200, 200), 1, cv2.LINE_AA)

        # Crosshair del cursor
        if cursor and len(self.points) < 4:
            cv2.line(self.frame, (cursor[0] - 15, cursor[1]), (cursor[0] + 15, cursor[1]),
                     (255, 255, 255), 1)
            cv2.line(self.frame, (cursor[0], cursor[1] - 15), (cursor[0], cursor[1] + 15),
                     (255, 255, 255), 1)

        # Instrucción actual
        if len(self.points) < 4:
            instruction = INSTRUCTIONS[len(self.points)]
            color = CORNER_COLORS[len(self.points)]
        else:
            instruction = "Listo! Presiona ENTER para guardar o R para reiniciar"
            color = (0, 255, 0)

        # Fondo semitransparente para el texto
        overlay = self.frame.copy()
        cv2.rectangle(overlay, (0, self.frame.shape[0] - 55),
                      (self.frame.shape[1], self.frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, self.frame, 0.4, 0, self.frame)

        cv2.putText(self.frame, instruction,
                    (15, self.frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Contador de puntos
        cv2.putText(self.frame, f"Puntos: {len(self.points)}/4",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(self.frame, "R = reiniciar  |  Q = cancelar",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    def run(self) -> dict | None:
        win = "Calibración de cancha — click en 4 esquinas"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1280, self.original.shape[1]),
                         min(720, self.original.shape[0]))
        cv2.setMouseCallback(win, self.mouse_callback)

        print("\n=== CALIBRACIÓN DE CANCHA ===")
        print("Hacé click en las 4 esquinas de la cancha en el orden indicado.")
        print("R = reiniciar | ENTER = guardar | Q = cancelar\n")

        self._redraw()

        while True:
            cv2.imshow(win, self.frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("r") or key == ord("R"):
                print("Reiniciando puntos...")
                self.points = []
                self._redraw()

            elif key == 13 and len(self.points) == 4:  # ENTER
                cv2.destroyAllWindows()
                return self._save()

            elif key == ord("q") or key == ord("Q"):
                print("Calibración cancelada.")
                cv2.destroyAllWindows()
                return None

        return None

    def _save(self) -> dict:
        config = {
            "source_points": self.points,
            "court_width_cm": self.court_width_cm,
            "court_height_cm": self.court_height_cm,
            "corner_order": [
                "inferior_izquierda",
                "inferior_derecha",
                "superior_derecha",
                "superior_izquierda",
            ],
            "notes": (
                "source_points son coordenadas en píxeles del frame. "
                "El ViewTransformer mapea estos 4 puntos al rectángulo "
                "0,0 -> court_width_cm,court_height_cm."
            ),
        }
        with open(self.output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nConfiguración guardada en: {self.output_path}")
        print(f"Puntos capturados:")
        for label, pt in zip(CORNER_LABELS, self.points):
            print(f"  {label}: {pt}")

        # Mostrar preview de la proyección
        self._show_preview()
        return config

    def _show_preview(self) -> None:
        target = np.array([
            [0, self.court_height_cm],
            [self.court_width_cm, self.court_height_cm],
            [self.court_width_cm, 0],
            [0, 0],
        ], dtype=np.float32)

        src = np.array(self.points, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, target)

        # Normalizar para display
        scale = 400 / self.court_width_cm
        preview_w = int(self.court_width_cm * scale)
        preview_h = int(self.court_height_cm * scale)
        target_display = target * scale

        M_display = cv2.getPerspectiveTransform(src, target_display)
        warped = cv2.warpPerspective(self.original, M_display, (preview_w, preview_h))

        # Superponer grilla de cancha
        cv2.rectangle(warped, (0, 0), (preview_w - 1, preview_h - 1), (0, 255, 0), 2)
        cv2.line(warped, (preview_w // 2, 0), (preview_w // 2, preview_h), (0, 200, 0), 1)

        cv2.imshow("Preview — Vista de pájaro (presiona cualquier tecla)", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_frame(video_path: str, frame_number: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = min(frame_number, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"No se pudo leer el frame {frame_number} de {video_path}")
    return frame


def main():
    parser = argparse.ArgumentParser(description="Calibración interactiva de cancha de básquet")
    parser.add_argument("--source", required=True, help="Video de entrada")
    parser.add_argument("--frame", type=int, default=30,
                        help="Número de frame a usar para calibrar (default: 30)")
    parser.add_argument("--output", default="court_config.json",
                        help="Archivo de salida para la configuración")
    parser.add_argument("--court-width", type=int, default=2865,
                        help="Ancho real de la cancha en cm (NBA: 2865)")
    parser.add_argument("--court-height", type=int, default=1524,
                        help="Alto real de la cancha en cm (NBA: 1524)")
    args = parser.parse_args()

    if not Path(args.source).exists():
        print(f"Error: No se encontró el video '{args.source}'")
        print("Primero descargá el video con: python download_video.py")
        return

    print(f"Cargando frame {args.frame} de {args.source}...")
    frame = get_frame(args.source, args.frame)

    calibrator = CourtCalibrator(frame, args.output)
    calibrator.court_width_cm = args.court_width
    calibrator.court_height_cm = args.court_height
    calibrator.run()


if __name__ == "__main__":
    main()
