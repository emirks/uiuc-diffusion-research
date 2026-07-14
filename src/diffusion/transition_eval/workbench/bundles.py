"""Warm corpus bundles — the one way the workbench reads the certified cache.

The shared cache (952 MB, 1933 entries) is READ-ONLY (OPERATIONS §1.3): a
polluted entry can fail the certified warm-determinism bar (1e-6) in a later
certification, and the directory is group-writable, so "read-only" has to be
enforced, not assumed. The deployed features.array_features WRITES to cache_dir
on a miss — so this module hands the deployed pipeline a ReadOnlyExtractor whose
extract() raises. A cache HIT never calls it; a cache MISS raises instead of
computing a feature and writing it. Polluting the shared cache is therefore
impossible by construction rather than by discipline, and no GPU (and no torch
CUDA context) is needed to read 223 warm bundles.

tracker=None throughout: no workbench metric consumes cotracker tracks. The
incumbent motion baselines come from the frozen npz, and the new motion metrics
are built on optical flow (RUNBOOK §3.1). This also keeps the track cache out of
the write path entirely.

CLI = OPERATIONS §4 items 3 and 4 (warm-cache probe + bitwise round-trip):

    PYTHONPATH=$WB/src python -m diffusion.transition_eval.workbench.bundles
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from ..certify import exam
from ..pipeline import process_video_file
from . import paths


class ReadOnlyExtractor:
    """Duck-types features.DinoExtractor for the cached path only.

    Carries the model name (it is part of the cache key) and refuses to compute.
    Any call to extract() means a cache miss, which means the corpus path or the
    pins are wrong — a loud stop, never a silent GPU re-extraction into a
    read-only cache."""

    def __init__(self, model_name: str = paths.DINO_MODEL):
        self.model_name = model_name
        self.calls = 0

    def extract(self, frames, batch_size: int = 64):
        self.calls += 1
        raise RuntimeError(
            "COLD CACHE: the deployed pipeline asked ReadOnlyExtractor to compute "
            "features. That means a cache miss against the certified shared cache "
            "— refusing (a write here can fail a future certification's "
            "warm-determinism bar). Check CORPUS_ROOT, DINO_MODEL and "
            "FEATURE_SHORT_SIDE against versioning.PINS.")

    def free(self) -> None:
        pass


def load_corpus_bundles(keys: list[str] | None = None, verbose: bool = False) -> list[dict]:
    """The 223 corpus bundles (feats [T,768] L2-normed CLS, morph profile), in
    sorted-key order — the row order every distance matrix shares. Zero decodes,
    zero GPU, zero writes.

    Bundle construction is byte-for-byte what run_certification does: same
    cache, same model, same short_side, same default morph kwargs (n_prefix=9,
    n_suffix=8, n_endpoints=2) — the bitwise round-trip below is what proves it.
    """
    corpus = paths.load_corpus()
    keys = keys or paths.corpus_keys(corpus)
    extractor = ReadOnlyExtractor()
    bundles = []
    for i, key in enumerate(keys):
        b, frames = process_video_file(
            paths.clip_path(key), paths.SHARED_CACHE, extractor, tracker=None,
            short_side=paths.FEATURE_SHORT_SIDE, need_frames=False)
        assert frames is None, f"{key}: decoded frames on a warm path"
        bundles.append(b)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [bundles] {i + 1}/{len(keys)}")
    assert extractor.calls == 0, "ReadOnlyExtractor was called — cache was cold"
    return bundles


def main() -> int:
    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    labels = paths.labels_of(corpus, keys)
    sidedness = paths.sidedness_of(corpus, keys)

    # --- OPERATIONS §4.3 — warm-cache probe -----------------------------------
    t0 = time.time()
    try:
        bundles = load_corpus_bundles(keys, verbose=True)
    except RuntimeError as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 1
    dt = time.time() - t0
    shapes = {b["feats"].shape[0] for b in bundles}
    print(f"[step0] warm-cache probe: {len(bundles)} bundles in {dt:.1f}s, zero decodes, "
          f"zero GPU; frame counts {sorted(shapes)}")

    # --- OPERATIONS §4.4 — bitwise round-trip ---------------------------------
    # Rebuild the incumbent appearance matrix with DEPLOYED code from the warm
    # bundles and compare to the frozen npz. Anything but exact 0.0 means the
    # bundles this workbench feeds its candidates are not the bundles that
    # produced the pinned baselines, and no head-to-head is trustworthy.
    t0 = time.time()
    D = exam.appearance_distance_matrix(bundles, "v3_sided", sidedness, n_jobs=4)
    dt = time.time() - t0
    frozen = np.load(paths.NPZ)["m1a__v3_sided"]
    delta = float(np.abs(D - frozen).max())
    print(f"[step0] m1a__v3_sided rebuilt in {dt:.1f}s; max|Δ| vs frozen npz = {delta!r}")
    if delta != 0.0:
        print(f"STOP: round-trip is not bitwise (max|Δ| = {delta!r}) — owner review "
              f"(OPERATIONS §4.4)", file=sys.stderr)
        return 1
    print("[step0] bitwise round-trip EXACT — warm bundles reproduce the pinned incumbent")

    out = paths.WB_OUT / "step0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "freeze_check.json").write_text(json.dumps({
        "warm_cache_probe": {
            "n_bundles": len(bundles), "decodes": 0, "gpu": False,
            "extractor_calls": 0, "seconds": round(dt, 2),
            "cache_dir": str(paths.SHARED_CACHE),
            "corpus_root": str(paths.CORPUS_ROOT),
            "pins": {"dino_model": paths.DINO_MODEL,
                     "feature_short_side": paths.FEATURE_SHORT_SIDE},
        },
        "bitwise_roundtrip": {
            "metric": "m1a__v3_sided",
            "built_by": "certify.exam.appearance_distance_matrix (deployed)",
            "max_abs_delta_vs_frozen_npz": delta,
            "exact": delta == 0.0,
        },
    }, indent=1))
    print(f"[step0] wrote {out / 'freeze_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
