"""Thread-capped MP4 read/write.

`exp_075/engine/videoio.py` caps threads on the *encoder* only; PyAV's decoder
still spawns a per-core frame-thread pool, which trips the per-user thread limit
on a login node (`av.error.BlockingIOError: [Errno 11]`). Everything here pins
thread_count so the same code runs on a login node and on a compute node.
"""

from __future__ import annotations

import pathlib

import av
import numpy as np


def read_clip(path: str | pathlib.Path, threads: int = 1) -> np.ndarray:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_count = threads
        s.thread_type = "NONE" if threads == 1 else "AUTO"
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    return np.stack(frames)


def write_clip(path: str | pathlib.Path, frames: np.ndarray, fps: int = 24,
               crf: int = 20, threads: int = 2) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    with av.open(str(path), mode="w") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height = w, h
        st.pix_fmt = "yuv420p"
        st.thread_count = threads
        st.options = {"crf": str(crf), "preset": "veryfast"}
        for arr in frames:
            c.mux(st.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
        c.mux(st.encode())


def filmstrip(frames: np.ndarray, indices, pad: int = 2) -> np.ndarray:
    sel = [frames[i] for i in indices]
    h, w = sel[0].shape[:2]
    strip = np.full((h, len(sel) * w + (len(sel) - 1) * pad, 3), 255, np.uint8)
    for k, img in enumerate(sel):
        strip[:, k * (w + pad):k * (w + pad) + w] = img
    return strip
