"""
video_reader.py — Lector de video compatible con AV1 usando PyAV.

OpenCV (opencv-python de pip) no incluye soporte para el codec AV1.
Este módulo usa PyAV (que incluye libdav1d) para decodificar cualquier
formato que ffmpeg soporte, incluyendo AV1.

Uso:
    from video_reader import get_video_info, frames_generator

    info   = get_video_info("video.mp4")   # sv.VideoInfo compatible
    for frame in frames_generator("video.mp4"):
        # frame es un numpy array BGR (H, W, 3) uint8
        process(frame)
"""

from __future__ import annotations

from typing import Generator

import av
import numpy as np
import supervision as sv


# ──────────────────────────────────────────────────────────────────────────────

def get_video_info(path: str) -> sv.VideoInfo:
    """
    Devuelve un sv.VideoInfo leyendo el archivo con PyAV.
    Funciona con H.264, H.265, AV1, VP9, etc.
    """
    container = av.open(path)
    stream = container.streams.video[0]
    fps    = float(stream.average_rate)
    width  = stream.width
    height = stream.height
    total  = stream.frames if stream.frames else None
    container.close()
    return sv.VideoInfo(width=width, height=height, fps=fps, total_frames=total)


def frames_generator(
    path: str,
    stride: int = 1,
    start: int = 0,
    end: int | None = None,
    thread_type: str = "AUTO",
) -> Generator[np.ndarray, None, None]:
    """
    Generador de frames BGR (numpy uint8) usando PyAV.

    Args:
        path:        Ruta al video.
        stride:      Saltar frames (1 = todos, 2 = uno de cada dos, etc.)
        start:       Primer frame a procesar (0-indexed).
        end:         Último frame a procesar (exclusivo). None = hasta el final.
        thread_type: Tipo de threading de decodificación ('AUTO', 'FRAME', 'SLICE').
    """
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = thread_type

    frame_idx = 0
    for av_frame in container.decode(stream):
        if end is not None and frame_idx >= end:
            break
        if frame_idx >= start and (frame_idx - start) % stride == 0:
            yield av_frame.to_ndarray(format="bgr24")
        frame_idx += 1

    container.close()


def sample_frames(
    path: str,
    n: int = 8,
    window: float = 0.3,
) -> list[tuple[int, np.ndarray]]:
    """
    Muestrea n frames distribuidos en los primeros `window` (fracción) del video.
    Devuelve lista de (frame_idx, frame_bgr).
    """
    info  = get_video_info(path)
    total = info.total_frames or 1000
    limit = int(total * window)

    indices = [int(limit * (i + 1) / (n + 1)) for i in range(n)]
    index_set = set(indices)

    result: list[tuple[int, np.ndarray]] = []
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    frame_idx = 0
    for av_frame in container.decode(stream):
        if frame_idx in index_set:
            result.append((frame_idx, av_frame.to_ndarray(format="bgr24")))
        if frame_idx > max(index_set):
            break
        frame_idx += 1

    container.close()
    return result
