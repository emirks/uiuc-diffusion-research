"""Floor controls. The lerp control is the canonical degenerate transition:
hold endpoint A, linear crossfade to endpoint B, hold B. Run through the
identical metric pipeline it anchors the floor of every score — transformation
depth ~0, degenerate morph profile, no effect medium, no motion."""

from __future__ import annotations

import numpy as np

from .video_io import resize_cover_crop


def make_static_hold(prefix_frames: np.ndarray, suffix_frames: np.ndarray | None,
                     total_frames: int) -> np.ndarray:
    """v3 one-sided / prefix-only degenerate control (SPEC §4): copy the prefix,
    HOLD its last frame through the middle, then the suffix as given (if any).
    This is the canonical lazy solution for one-sided items — freeze, then snap
    — and unlike lerp it (a) injects zero end-state information into the floor
    and (b) exists for prefix-only items, which have nothing to lerp to."""
    n_suf = len(suffix_frames) if suffix_frames is not None else 0
    n_mid = total_frames - len(prefix_frames) - n_suf
    if n_mid < 1:
        raise ValueError(f"total_frames={total_frames} too short for endpoints")
    mid = np.repeat(prefix_frames[-1:], n_mid, axis=0)
    parts = [prefix_frames, mid]
    if suffix_frames is not None:
        h, w = prefix_frames.shape[1:3]
        if suffix_frames.shape[1:3] != (h, w):
            suffix_frames = resize_cover_crop(suffix_frames, h, w)
        parts.append(suffix_frames)
    return np.concatenate(parts)


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
