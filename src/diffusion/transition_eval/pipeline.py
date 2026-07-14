"""One cached processing path for any video entering the harness — real
reference, generated output, or synthetic control — so every number in a
report comes from identical machinery."""

from __future__ import annotations

import pathlib

import numpy as np

from .features import DinoExtractor, array_features, feature_cache_path, file_key
from .morph import core_mask, derived_scalars, morph_profile
from .motion import Tracker, track_cache_path
from .video_io import load_frames, probe_fps


class VideoBundle(dict):
    """dict with attribute access: feats, profile, scalars, core, tracks, vis."""

    __getattr__ = dict.__getitem__


def process_video(frames: np.ndarray | None, key: str, cache_dir: pathlib.Path,
                  extractor: DinoExtractor, tracker: Tracker | None = None,
                  n_prefix: int = 9, n_suffix: int = 8, n_endpoints: int = 2) -> VideoBundle:
    """uint8 frames -> features (cached), morph profile + scalars, core mask,
    and (optionally) tracklets (cached)."""
    feats = array_features(frames, key, cache_dir, extractor)
    profile = morph_profile(feats, n_prefix=n_prefix, n_suffix=n_suffix, n_endpoints=n_endpoints)
    bundle = VideoBundle(
        key=key, feats=feats, profile=profile,
        scalars=derived_scalars(profile), core=core_mask(profile),
        tracks=None, vis=None,
    )
    if tracker is not None:
        bundle["tracks"], bundle["vis"] = tracker.cached_track(frames, key + ":tracks", cache_dir)
    return bundle


def process_video_file(path: pathlib.Path, cache_dir: pathlib.Path,
                       extractor: DinoExtractor, tracker: Tracker | None = None,
                       short_side: int = 256, need_frames: bool = True,
                       **morph_kw) -> tuple[VideoBundle, np.ndarray | None]:
    """Decode once, process, and return (bundle, frames) — callers that also
    need pixels (controls, LPIPS, judge) reuse the decoded frames.

    need_frames=False skips the decode entirely when every requested cache is
    warm (frames returns None; fps read from the container header). A cache
    miss still decodes, so the flag never changes any number — only whether
    pixels are materialized for the caller."""
    key = file_key(path, extractor.model_name, str(short_side))
    warm = (feature_cache_path(key, cache_dir).exists()
            and (tracker is None
                 or track_cache_path(f"{key}:tracks:{tracker.CACHE_TAG}",
                                     cache_dir).exists()))
    if not need_frames and warm:
        frames, fps = None, probe_fps(path)
    else:
        frames, fps = load_frames(path, short_side=short_side)
    bundle = process_video(frames, key, cache_dir, extractor, tracker, **morph_kw)
    bundle["fps"] = fps
    bundle["path"] = str(path)
    return bundle, frames
