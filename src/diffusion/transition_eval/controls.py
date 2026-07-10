"""Floor controls. The lerp control is the canonical degenerate transition:
hold endpoint A, linear crossfade to endpoint B, hold B. Run through the
identical metric pipeline it anchors the floor of every score — transformation
depth ~0, degenerate morph profile, no effect medium, no motion."""

from __future__ import annotations

import numpy as np

from .video_io import resize_cover_crop


def make_lerp(prefix_frames: np.ndarray, suffix_frames: np.ndarray, total_frames: int) -> np.ndarray:
    """Copy the prefix frames, crossfade prefix[-1] -> suffix[0] over the
    middle, copy the suffix frames. Suffix is cover-cropped to prefix geometry
    when the two clips differ."""
    h, w = prefix_frames.shape[1:3]
    if suffix_frames.shape[1:3] != (h, w):
        suffix_frames = resize_cover_crop(suffix_frames, h, w)
    n_mid = total_frames - len(prefix_frames) - len(suffix_frames)
    if n_mid < 1:
        raise ValueError(f"total_frames={total_frames} too short for endpoints")
    src = prefix_frames[-1].astype(np.float32)
    dst = suffix_frames[0].astype(np.float32)
    alphas = np.linspace(0.0, 1.0, n_mid + 2)[1:-1]
    mid = np.stack([(1 - a) * src + a * dst for a in alphas]).round().astype(np.uint8)
    return np.concatenate([prefix_frames, mid, suffix_frames])
