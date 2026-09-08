#!/usr/bin/env python
"""EffectData gapper screen — warm the scorer cache for every screen clip and compute kernel ceilings for the screen classes.

Same kernel as scripts/grid_v3/ceilings_kernel.py (deployed pool-score kernel appearance_s3 / m1a_pair against the FROZEN
reference_v4 populations, core_mask_v3 by sidedness 'onesided', mean app_ref over ordered same-class pairs). Runs on a GPU
node (refvfx env, cert worktree on PYTHONPATH). Parts -> misc/2026-09-08_ed_gapper_screen/ceilings/part_i.json;
--aggregate -> ceilings_screen.json (schema of eval_ladder/ceilings_v3.json; used through LADDER_CEILINGS_EXTRA).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "misc/2026-09-08_ed_gapper_screen"
STD = REPO / "data/processed/transitions_std121"
CACHE = REPO / "misc/refvfx_baseline/probe/cache"
PARTS = CAMP / "ceilings"


def classes():
    return sorted({d["cls"] for d in json.loads((CAMP / "selection.json").read_text())["effects"]})


def compute(shard, n):
    import numpy as np, torch
    from diffusion.transition_eval import versioning
    from diffusion.transition_eval.features import DinoExtractor
    from diffusion.transition_eval.motion import Tracker
    from diffusion.transition_eval.pipeline import process_video_file
    from diffusion.transition_eval.s_structure import core_mask_v3
    from diffusion.transition_eval.m1_transfer import appearance_s3
    from diffusion.transition_eval import reference_stats as RS
    ref = RS.load_reference()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DinoExtractor(versioning.PINS["dino_model"], device=dev); tracker = Tracker(device=dev); short = versioning.PINS["feature_short_side"]
    out = {}
    for cls in classes()[shard::n]:
        bundles = []
        for p in sorted((STD / cls).glob("*.mp4")):
            b, _ = process_video_file(p, CACHE, extractor, tracker, short_side=short, need_frames=False)
            core, _m = core_mask_v3(b.profile, "onesided"); bundles.append((p.stem, b, core))
        vals = []
        for gi, (gs, gb, gc) in enumerate(bundles):
            for ri, (rs, rb, rc) in enumerate(bundles):
                if gi != ri:
                    vals.append(float(appearance_s3(gb.feats, gc, gb.profile["n_prefix"], gb.profile["n_suffix"], rb.feats, rc, rb.profile["n_prefix"], rb.profile["n_suffix"], ref)["app_ref"]))
        out[cls] = {"ceiling": float(np.mean(vals)) if vals else None, "n_clips": len(bundles), "n_pairs": len(vals), "sd_pairs": float(np.std(vals)) if vals else None, "sidedness": "onesided"}
        print(f"  {cls:50s} n={len(bundles):2d} ceiling={out[cls]['ceiling']}", flush=True)
    PARTS.mkdir(parents=True, exist_ok=True); (PARTS / f"part_{shard}.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"[done] shard {shard}/{n}: {len(out)} classes")


def aggregate():
    merged = {}
    for f in sorted(PARTS.glob("part_*.json")):
        merged.update(json.loads(f.read_text()))
    out = {"method": "deployed pool-score kernel (appearance_s3 / m1a_pair) vs the FROZEN reference_v4 populations, core_mask_v3 onesided, mean app_ref over ordered same-class pairs (scripts/ed_screen/warm_ceilings.py)",
           "ceilings": {c: v for c, v in sorted(merged.items()) if v["ceiling"] is not None}, "n_classes": len(merged)}
    (CAMP / "ceilings_screen.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{len(out['ceilings'])} classes -> ceilings_screen.json; missing: {sorted(set(classes()) - set(merged))[:5]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args(); aggregate() if a.aggregate else compute(a.shard, a.num_shards)
