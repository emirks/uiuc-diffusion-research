"""One cached processing path for any video entering the harness — real
reference, generated output, or synthetic control — so every number in a
report comes from identical machinery.

I/O has two backends (SPEC §9): a :class:`~diffusion.feature_store.FeatureStore`
(the deployed path, keyed by video path) or a legacy ``cache_dir`` (hashed
filenames — retained for ``certify/`` and ``workbench/`` and their tests). The
metric machinery is identical either way; only where the arrays are read /
written differs. ``process_video_file`` dispatches on its backend argument."""

from __future__ import annotations

import pathlib

import numpy as np

from . import store_io
from ..feature_store import FeatureStore
from .features import DinoExtractor, array_features, feature_cache_path, file_key
from .morph import core_mask, derived_scalars, morph_profile
from .motion import Tracker, track_cache_path
from .video_io import load_frames, probe_fps


class VideoBundle(dict):
    """dict with attribute access: feats, profile, scalars, core, tracks, vis."""

    __getattr__ = dict.__getitem__


def _is_store(backend) -> bool:
    return isinstance(backend, (FeatureStore, store_io.HarnessStore))


def process_video(frames: np.ndarray | None, key: str, cache_dir: pathlib.Path,
                  extractor: DinoExtractor, tracker: Tracker | None = None,
                  n_prefix: int = 9, n_suffix: int = 8, n_endpoints: int = 2) -> VideoBundle:
    """uint8 frames -> features (cached), morph profile + scalars, core mask,
    and (optionally) tracklets (cached). Legacy ``cache_dir`` backend."""
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


def process_video_store(frames: np.ndarray | None, identity, hstore: "store_io.HarnessStore",
                        extractor: DinoExtractor, tracker: Tracker | None = None,
                        n_prefix: int = 9, n_suffix: int = 8, n_endpoints: int = 2) -> VideoBundle:
    """Store backend of :func:`process_video`: DINO features and tracklets are
    read from / written to the feature store by ``identity`` (a real video path
    or a control next to its gen). Numerically identical to the legacy path — a
    store hit returns the same arrays the hashed cache held; a miss extracts and
    persists exactly what the legacy path wrote."""
    got = hstore.get(identity, store_io.DINO_NS)
    if got is not None:
        feats = got["feats"]
    else:
        feats = extractor.extract(frames)
        hstore.put(identity, store_io.DINO_NS, {"feats": feats})
    profile = morph_profile(feats, n_prefix=n_prefix, n_suffix=n_suffix, n_endpoints=n_endpoints)
    bundle = VideoBundle(
        key=hstore.key_str(identity), identity=identity, feats=feats, profile=profile,
        scalars=derived_scalars(profile), core=core_mask(profile),
        tracks=None, vis=None,
    )
    if tracker is not None:
        got_t = hstore.get(identity, store_io.TRACK_NS)
        if got_t is not None:
            tracks, vis = got_t["tracks"], got_t["vis"]
        else:
            tracks, vis = tracker.track(frames)
            hstore.put(identity, store_io.TRACK_NS, {"tracks": tracks, "vis": vis})
        bundle["tracks"], bundle["vis"] = tracks, vis
    return bundle


def process_video_file(path: pathlib.Path, backend, extractor: DinoExtractor,
                       tracker: Tracker | None = None, short_side: int = 256,
                       need_frames: bool = True,
                       **morph_kw) -> tuple[VideoBundle, np.ndarray | None]:
    """Decode once, process, and return (bundle, frames) — callers that also
    need pixels (controls, LPIPS, judge) reuse the decoded frames.

    ``backend`` is a ``FeatureStore`` / ``HarnessStore`` (deployed store path) or
    a legacy ``cache_dir`` path. ``need_frames=False`` skips the decode entirely
    when every requested cache is warm (frames returns None; fps from the
    container header). A cache miss still decodes, so the flag never changes any
    number — only whether pixels are materialized for the caller."""
    if _is_store(backend):
        hstore = backend if isinstance(backend, store_io.HarnessStore) else store_io.HarnessStore(backend)
        identity = store_io.RealVideo(path)
        warm = (hstore.has(identity, store_io.DINO_NS)
                and (tracker is None or hstore.has(identity, store_io.TRACK_NS)))
        if not need_frames and warm:
            frames, fps = None, probe_fps(path)
        else:
            frames, fps = load_frames(path, short_side=short_side)
        bundle = process_video_store(frames, identity, hstore, extractor, tracker, **morph_kw)
        bundle["fps"] = fps
        bundle["path"] = str(path)
        return bundle, frames

    cache_dir = backend
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
