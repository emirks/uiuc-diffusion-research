"""Minimal MP4 read/write for the procedural engine (PyAV)."""

from __future__ import annotations

import pathlib

import av
import numpy as np


def read_clip(path: str | pathlib.Path) -> np.ndarray:
    """Decode an MP4 to an (N, H, W, 3) uint8 array."""
    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    return np.stack(frames)


def write_clip(path: str | pathlib.Path, frames: np.ndarray, fps: int = 24,
               crf: int = 18, threads: int = 4) -> None:
    """Encode to H.264. `threads` is capped on purpose: the software GL renderer
    already saturates the box, and letting swscale/x264 spawn per-core thread
    pools on top of it trips the per-user thread limit (EAGAIN on a login node).
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.thread_count = threads
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for arr in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
        container.mux(stream.encode())


def filmstrip(frames: np.ndarray, indices, pad: int = 2) -> np.ndarray:
    """Horizontal contact sheet of the given frame indices."""
    sel = [frames[i] for i in indices]
    h, w = sel[0].shape[:2]
    strip = np.full((h, len(sel) * w + (len(sel) - 1) * pad, 3), 255, np.uint8)
    for k, img in enumerate(sel):
        x = k * (w + pad)
        strip[:, x:x + w] = img
    return strip
