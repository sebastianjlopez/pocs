"""
Descarga un video de prueba de básquet NBA de YouTube.
Requiere: pip install yt-dlp

Uso:
    python download_video.py
    python download_video.py --url "https://www.youtube.com/watch?v=..."
    python download_video.py --output mi_video.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Videos públicos de NBA highlights (cámara fija, buena visión de cancha)
DEFAULT_URLS = [
    # NBA highlights con ángulo de cámara estático — ideales para tracking
    "https://www.youtube.com/watch?v=sM4K0MzI5iI",   # NBA tracking demo
    "https://www.youtube.com/watch?v=wqZKMBvNWJA",   # NBA game footage
]

def download(url: str, output: str = "test_basketball.mp4", max_height: int = 720) -> bool:
    """Descarga el video en la resolución indicada."""
    print(f"Descargando: {url}")
    print(f"Destino:     {output}")

    cmd = [
        "yt-dlp",
        "--format", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best",
        "--output", output,
        "--no-playlist",
        "--merge-output-format", "mp4",
        url,
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if Path(output).exists():
            size_mb = Path(output).stat().st_size / 1_048_576
            print(f"\nDescargado: {output} ({size_mb:.1f} MB)")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error al descargar: {e}")
    except FileNotFoundError:
        print("yt-dlp no encontrado. Instalá con: pip install yt-dlp")
    return False


def create_synthetic_video(output: str = "test_basketball.mp4") -> None:
    """
    Genera un video sintético de básquet con jugadores simulados
    para testear el pipeline sin necesidad de descargar nada.
    """
    import cv2
    import numpy as np
    import random
    import math

    print("Generando video sintético de básquet para testing...")

    W, H, FPS, DURATION = 1280, 720, 30, 20
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, FPS, (W, H))

    # Cancha de básquet (fondo)
    def draw_court(frame: np.ndarray) -> np.ndarray:
        frame[:] = (60, 120, 60)  # verde madera
        # piso
        cv2.rectangle(frame, (100, 100), (W - 100, H - 100), (180, 130, 80), -1)
        # líneas
        court_color = (255, 255, 255)
        cv2.rectangle(frame, (100, 100), (W - 100, H - 100), court_color, 2)
        # línea central
        cv2.line(frame, (W // 2, 100), (W // 2, H - 100), court_color, 2)
        # círculo central
        cv2.circle(frame, (W // 2, H // 2), 80, court_color, 2)
        # pinturas
        cv2.rectangle(frame, (100, 260), (280, 460), court_color, 2)
        cv2.rectangle(frame, (W - 280, 260), (W - 100, 460), court_color, 2)
        # arcos 3pt (simplificado como semicírculos)
        cv2.ellipse(frame, (190, 360), (200, 200), 0, -90, 90, court_color, 2)
        cv2.ellipse(frame, (W - 190, 360), (200, 200), 0, 90, 270, court_color, 2)
        return frame

    # Estado de cada jugador
    class Player:
        def __init__(self, pid: int, team: int, x: float, y: float):
            self.pid = pid
            self.team = team  # 0=blanco, 1=azul
            self.x = x
            self.y = y
            self.vx = random.uniform(-3, 3)
            self.vy = random.uniform(-2, 2)
            self.color = (230, 230, 230) if team == 0 else (50, 80, 200)

        def step(self, frame_idx: int):
            # movimiento con algo de oscilación
            t = frame_idx * 0.05
            self.x += self.vx + math.sin(t + self.pid) * 1.5
            self.y += self.vy + math.cos(t * 0.7 + self.pid) * 1.0
            # rebotar en los límites de la cancha
            if self.x < 150 or self.x > W - 150:
                self.vx *= -1
            if self.y < 150 or self.y > H - 150:
                self.vy *= -1
            self.x = max(150, min(W - 150, self.x))
            self.y = max(150, min(H - 150, self.y))

        def draw(self, frame: np.ndarray):
            cx, cy = int(self.x), int(self.y)
            # cuerpo
            cv2.ellipse(frame, (cx, cy + 25), (18, 28), 0, 0, 360, self.color, -1)
            # cabeza
            cv2.circle(frame, (cx, cy - 5), 14, (210, 180, 140), -1)
            # número
            cv2.putText(frame, str(self.pid), (cx - 7, cy + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    class Ball:
        def __init__(self):
            self.x = W / 2
            self.y = H / 2
            self.vx = random.uniform(-5, 5)
            self.vy = random.uniform(-4, 4)

        def step(self):
            self.x += self.vx
            self.y += self.vy
            if self.x < 120 or self.x > W - 120:
                self.vx *= -0.9
            if self.y < 120 or self.y > H - 120:
                self.vy *= -0.9
            self.x = max(120, min(W - 120, self.x))
            self.y = max(120, min(H - 120, self.y))

        def draw(self, frame: np.ndarray):
            cv2.circle(frame, (int(self.x), int(self.y)), 12, (0, 140, 255), -1)
            cv2.circle(frame, (int(self.x), int(self.y)), 12, (0, 100, 200), 2)

    # Crear jugadores: 5 por equipo + pelota
    players = []
    for i in range(5):
        players.append(Player(i + 1, 0, random.uniform(200, 600), random.uniform(200, 500)))
    for i in range(5):
        players.append(Player(i + 6, 1, random.uniform(700, 1100), random.uniform(200, 500)))
    ball = Ball()

    total_frames = FPS * DURATION
    for frame_idx in range(total_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame = draw_court(frame)
        for p in players:
            p.step(frame_idx)
        ball.step()
        for p in players:
            p.draw(frame)
        ball.draw(frame)

        # Info overlay
        cv2.putText(frame, f"SYNTHETIC BASKETBALL — Frame {frame_idx}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Team A: white  |  Team B: blue",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(frame)

    out.release()
    size_mb = Path(output).stat().st_size / 1_048_576
    print(f"Video sintético generado: {output} ({size_mb:.1f} MB, {total_frames} frames a {FPS}fps)")


def main():
    parser = argparse.ArgumentParser(description="Descarga video de prueba de básquet")
    parser.add_argument("--url", default=None, help="URL de YouTube (opcional)")
    parser.add_argument("--output", default="test_basketball.mp4", help="Archivo de salida")
    parser.add_argument("--resolution", type=int, default=720, help="Resolución máxima (altura px)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generar video sintético sin descargar nada")
    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_video(args.output)
        return

    # Intentar descargar
    url = args.url or DEFAULT_URLS[0]
    success = download(url, args.output, args.resolution)

    if not success:
        print("\nNo se pudo descargar. Generando video sintético como fallback...")
        create_synthetic_video(args.output)


if __name__ == "__main__":
    main()
