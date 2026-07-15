"""GPU pass — the IV2 probe cache (E1PRIME_DIRECTIVE §2.3, PREREG §P6).

IV2 asks whether the signature can tell a SNAP from NOTHING: a hard cut at the
conditioning handoff, against the crossfade that would have been there instead.

  cuts  — certify.probes.build_hard_cut, the DEPLOYED Bar-6 construction, IMPORTED and
          not reimplemented, on each n>=2 class's bar_pair (also deployed:
          certify.probes.sibling_pairs, deterministic and corpus-only). Prefix of A
          (9 frames) + cover-cropped body of B.
  lerps — nulls.render_null(cut_frames): the crossfade built from THE CUT'S OWN
          endpoints. Same §4.0 per-pair null object, same deployed builder. This is what
          makes IV2 "snap vs nothing" and not "snap vs some unrelated video".

WHY THIS IS A GPU JOB AND NOT A CACHE READ. The certification embedded its own hard-cut
probes, and those features exist. They are DELIBERATELY NOT REUSED: cuts embedded by a
prior GPU run, paired against lerps embedded now, would let a DEVICE-DRIFT SIGNATURE
separate IV2's two classes — and IV2 could then pass for a reason that has nothing to do
with the signature. Both classes are extracted TOGETHER, ON ONE DEVICE, in this job.

Nothing is written to the certified shared cache. The corpus is still read through
bundles.ReadOnlyExtractor; only $WB_CACHE is written.
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..certify.probes import build_hard_cut, sibling_pairs
from ..pipeline import process_video_file
from ..video_io import load_frames
from . import bundles, nulls, paths

IV2_DIR = paths.WB_CACHE / "iv2"
PROBE_DIR = IV2_DIR / "probes"


def log(m: str) -> None:
    print(f"[iv2 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("REFUSING: GPU work on CPU. Submit through Slurm.")
        return 1

    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    bs = bundles.load_corpus_bundles(keys)          # warm, read-only, zero writes
    bmap = {k: b for k, b in zip(keys, bs)}

    pairs = sibling_pairs(bmap, corpus)             # deployed, deterministic
    log(f"bar pairs: {len(pairs)} classes with n>=2")

    from ..features import DinoExtractor
    dino = DinoExtractor(paths.DINO_MODEL, device=device)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    t0 = time.time()
    for i, (cls, info) in enumerate(sorted(pairs.items())):
        a, b = info["bar_pair"]
        cut_path = PROBE_DIR / f"hardcut__{cls}.mp4"
        if not cut_path.exists():
            build_hard_cut(paths.clip_path(a), paths.clip_path(b), cut_path)

        # the cut's own bundle: features + S-profile, written to $WB_CACHE ONLY
        cut_bundle, cut_frames = process_video_file(
            cut_path, paths.WB_CACHE, dino, tracker=None,
            short_side=paths.FEATURE_SHORT_SIDE, need_frames=True)

        # the matched "nothing": the crossfade with THE CUT'S OWN endpoints
        lerp_cache = IV2_DIR / f"lerp__{cls}.npz"
        if not lerp_cache.exists():
            lerp_frames = nulls.render_null(cut_frames)
            feats = dino.extract(lerp_frames)                  # SAME device, SAME job
            tmp = lerp_cache.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, feats=feats, n_frames=len(lerp_frames))
            tmp.replace(lerp_cache)

        manifest[cls] = {
            "bar_pair": [a, b],
            "cut_video": str(cut_path),
            "lerp_feats": str(lerp_cache),
            "n_frames_cut": int(len(cut_bundle["feats"])),
            "sidedness": corpus["classes"][cls]["sidedness"],
        }
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(pairs)} ({time.time() - t0:.0f}s)")

    dino.free() if hasattr(dino, "free") else None
    IV2_DIR.mkdir(parents=True, exist_ok=True)
    (IV2_DIR / "manifest.json").write_text(json.dumps({
        "purpose": "IV2 (snap vs nothing) — E1PRIME_DIRECTIVE §2.3",
        "cut_construction": "certify.probes.build_hard_cut (deployed Bar-6, imported)",
        "pair_selection": "certify.probes.sibling_pairs -> bar_pair (deployed)",
        "null_construction": "nulls.render_null == controls.make_lerp on the CUT's own "
                             "endpoints (deployed)",
        "device_consistency": "cuts and lerps embedded TOGETHER on ONE device in this "
                              "job; the certification's own probe features are "
                              "deliberately NOT reused (a device-drift signature could "
                              "otherwise separate IV2's classes for free)",
        "embedder": paths.DINO_MODEL,
        "short_side": paths.FEATURE_SHORT_SIDE,
        "n_classes": len(manifest),
        "classes": manifest,
    }, indent=1))
    log(f"IV2 CACHE COMPLETE: {len(manifest)} cuts + {len(manifest)} lerps "
        f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
