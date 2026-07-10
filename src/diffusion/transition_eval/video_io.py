"""Frame I/O for the eval harness (PyAV). All frames are uint8 [T, H, W, 3]."""

from __future__ import annotations

import fractions
import pathlib

import av
import numpy as np
from PIL import Image


def load_frames(path: pathlib.Path, short_side: int | None = 256) -> tuple[np.ndarray, float]:
    """Decode a whole video. Frames are resized during decode (BILINEAR) so the
    shortest side equals `short_side` (never upscaled); returns (frames, fps)."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        for frame in container.decode(stream):
            img = frame.to_image()  # materialize during decode (PyAV buffer reuse)
            if short_side is not None:
                w, h = img.size
                s = short_side / min(w, h)
                if s < 1.0:
                    img = img.resize((max(2, round(w * s)), max(2, round(h * s))), Image.BILINEAR)
            frames.append(np.asarray(img, dtype=np.uint8))
    if not frames:
        raise ValueError(f"no video frames decoded from {path}")
    return np.stack(frames), fps


def resize_cover_crop(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize-to-cover + center-crop (mirrors the LTX ValidationRunner's
    conditioning-media preprocessing) so condition clips align with generated
    frames of a different resolution/aspect."""
    out = np.empty((len(frames), height, width, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        s = max(height / h, width / w)
        rw, rh = max(width, round(w * s)), max(height, round(h * s))
        img = Image.fromarray(f).resize((rw, rh), Image.BILINEAR)
        left, top = (rw - width) // 2, (rh - height) // 2
        out[i] = np.asarray(img.crop((left, top, left + width, top + height)), dtype=np.uint8)
    return out


def write_video(frames: np.ndarray, path: pathlib.Path, fps: float = 24.0) -> None:
    """Write uint8 frames as H.264 mp4 (yuv420p needs even dims — crops by 1px if odd)."""
    h, w = frames.shape[1] // 2 * 2, frames.shape[2] // 2 * 2
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fractions.Fraction(fps).limit_denominator(1000))
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for f in frames:
            av_frame = av.VideoFrame.from_ndarray(f[:h, :w], format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
