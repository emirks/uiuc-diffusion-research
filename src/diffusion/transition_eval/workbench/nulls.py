"""Rendered-lerp nulls (RUNBOOK §4.0) — the per-pair calibration object.

    "For every corpus clip, synthesize the degenerate video for its own endpoint
     pair (alpha-blend at matched progress; reuse core_degenerate machinery),
     embed and cache its whitened curve. This is the per-pair calibration object
     — the geometric chord is kept only as the coordinate frame, never as the
     null."

WHY A RENDERED NULL AND NOT THE GEOMETRIC CHORD. The straight line between e_A
and e_B in embedding space is not where a real crossfade goes: alpha-blending two
images produces a ghosted superposition whose DINO embedding leaves the chord
(the encoder is not linear in pixel space). So "distance from nothing" has to be
measured against the null that was actually RENDERED for this clip's own
endpoints — otherwise every clip's excursion is measured from a point no
degenerate video occupies, and the quotient re-imports the very content it claims
to remove.

CONSTRUCTION (pinned, deployed code reused verbatim):
  - the clip's own conditioned windows: prefix = frames[:n_prefix] (9),
    suffix = frames[-n_suffix:] (8), from the SAME short_side=256 decode the
    certified features come from, so the null's endpoint pixels are the same
    pixels the incumbent saw;
  - controls.make_lerp(prefix, suffix, total_frames=121) — the deployed
    alpha-blend, not a reimplementation. "Alpha-blend at matched progress" means
    a matched frame budget and matched conditioned windows (121 total, 9/8
    endpoints), so sigma aligns between clip and null; it is NOT a per-frame alpha
    schedule tracking the clip's own a_hat.
  - EVERY clip gets a lerp null, one-sided classes included: every corpus clip
    carries both conditioned windows (the std contract is 9 prefix + 8 suffix
    frames), so a lerp is always constructible. make_static_hold is the certified
    ONE-SIDED control arm, but §4.0 registers the lerp as the null, and swapping
    it for one-sided classes would be an unregistered change of the calibration
    object.

The null's features are cached in $WB_CACHE under this module's own tag. They are
NEVER written into the certified shared cache.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

from ..controls import make_lerp
from ..features import file_key
from ..video_io import load_frames
from . import paths

CACHE_TAG = "lerpnull-v1"
TOTAL_FRAMES = 121          # the corpus std contract
N_PREFIX, N_SUFFIX = 9, 8   # the deployed morph defaults the certification used

PINS = {
    "kind": "rendered_lerp",
    "builder": "transition_eval.controls.make_lerp (deployed, verbatim)",
    "total_frames": TOTAL_FRAMES,
    "n_prefix": N_PREFIX,
    "n_suffix": N_SUFFIX,
    "decode_short_side": paths.FEATURE_SHORT_SIDE,
    "embedder": paths.DINO_MODEL,
    "applies_to": "all 223 clips (one-sided included)",
    "cache_tag": CACHE_TAG,
}


def null_cache_path(key: str, cache_dir: pathlib.Path) -> pathlib.Path:
    h = hashlib.sha1(f"{key}:{CACHE_TAG}".encode()).hexdigest()[:16]
    return pathlib.Path(cache_dir) / f"null_{h}.npz"


def clip_null_key(path: pathlib.Path) -> str:
    return file_key(path, "lerpnull", paths.DINO_MODEL,
                    str(paths.FEATURE_SHORT_SIDE), str(TOTAL_FRAMES))


def render_null(frames: np.ndarray) -> np.ndarray:
    """The clip's own endpoints -> its degenerate twin, via deployed make_lerp."""
    prefix = frames[:N_PREFIX]
    suffix = frames[-N_SUFFIX:]
    return make_lerp(prefix, suffix, TOTAL_FRAMES)


def build_clip_null(path: pathlib.Path, extractor, cache_dir: pathlib.Path) -> pathlib.Path:
    """Render + embed one clip's lerp null. Idempotent; writes only to cache_dir
    (which is $WB_CACHE — never the certified shared cache)."""
    key = clip_null_key(path)
    cache = null_cache_path(key, cache_dir)
    if cache.exists():
        return cache
    frames, _ = load_frames(path, short_side=paths.FEATURE_SHORT_SIDE)
    null_frames = render_null(frames)
    feats = extractor.extract(null_frames)              # [121, 768] L2-normed CLS
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, feats=feats, src=str(path), n_frames=len(null_frames))
    tmp.replace(cache)
    return cache


def load_null_feats(path: pathlib.Path, cache_dir: pathlib.Path) -> np.ndarray:
    """[121, 768] null features from cache. Raises on a miss — nulls are built
    once, in the cache-build job."""
    cache = null_cache_path(clip_null_key(path), cache_dir)
    if not cache.exists():
        raise RuntimeError(f"lerp-null cache miss for {path} ({cache.name}) — run "
                           f"the cache-build job (OPERATIONS §6 step 3)")
    return np.load(cache)["feats"]
