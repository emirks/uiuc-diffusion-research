#!/usr/bin/env python
"""grid v3 — class ceilings for the new corpus classes, by the DEPLOYED pool-score kernel against the frozen reference.

Why not `score.py`: the v4 instrument pins its reference populations (reference_v4.npz) to the 222-clip corpus
manifest and refuses any other manifest (SPEC §4/§7) — correct for scoring generations, and exactly the guarantee
the overlay keeps: the populations are never rebuilt. This script uses the SAME deployed kernel a generation's pool
score uses (m1_transfer.appearance_s3 -> reference_stats.m1a_pair against the frozen populations, core mask
core_mask_v3 by class sidedness, bundle profiles from the shared feature cache) and applies it to every ordered
same-class pair of corpus clips. The per-class mean over pairs is the ceiling "a perfect generation scores in this
exact setting" (POOL_YARDSTICK.md) for classes the certified matrix does not cover. Five original classes are scored
the same way as a CALIBRATION set against their certified ceilings; the aggregate reports the ratio.

Bundles come from the warm cache (shard 0 of job 3108966 extracted every new clip); a cold clip is extracted on the
fly by the same pipeline. Runs on a GPU node (refvfx env, eval-v4-cert worktree on PYTHONPATH).

Run (per shard):  python scripts/grid_v3/ceilings_kernel.py --shard i --num-shards n
Aggregate:        python scripts/grid_v3/ceilings_kernel.py --aggregate   -> eval_ladder/ceilings_v3.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WT = REPO / ".claude/worktrees/eval-v4-cert"
sys.path.insert(0, str(REPO / "scripts/grid_v3"))
STD = REPO / "data/processed/transitions_std121"
CACHE = REPO / "misc/refvfx_baseline/probe/cache"
PARTS = REPO / "misc/2026-09-07_eval_grid_v2/ceilings/kernel"
CEIL_OUT = REPO / "eval_ladder/ceilings_v3.json"
NPZ_V4 = WT / "outputs/eval/certification/4.0.0-draft.1/analysis/distance_matrices.npz"
CALIBRATION = ["shadow", "portal", "cotton_cloud", "hero_flight", "shadow_smoke"]


def compute(shard: int, num_shards: int) -> None:
    import numpy as np
    import torch
    from diffusion.transition_eval import versioning
    from diffusion.transition_eval.features import DinoExtractor
    from diffusion.transition_eval.motion import Tracker
    from diffusion.transition_eval.pipeline import process_video_file
    from diffusion.transition_eval.s_structure import core_mask_v3
    from diffusion.transition_eval.m1_transfer import appearance_s3
    from diffusion.transition_eval import reference_stats as RS
    from ceilings_manifest import classes_to_score

    ref = RS.load_reference()          # frozen populations; the corpus pin is deliberately NOT asserted (overlay semantics)
    print(f"[ref] corpus_sha of the frozen artifact: {str(ref['corpus_sha'])[:12]} (222-clip corpus; populations reused unchanged)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DinoExtractor(versioning.PINS["dino_model"], device=device)
    tracker = Tracker(device=device)
    short = versioning.PINS["feature_short_side"]

    classes = classes_to_score()
    todo = sorted(classes)[shard::num_shards]
    out = {}
    for cls in todo:
        sided = classes[cls]
        clips = sorted(p for p in (STD / cls).glob("*.mp4"))
        bundles = []
        for p in clips:
            b, _ = process_video_file(p, CACHE, extractor, tracker, short_side=short, need_frames=False)
            core, _meta = core_mask_v3(b.profile, sided)
            bundles.append((p.stem, b, core))
        vals, rows = [], []
        for gi, (gs, gb, gc) in enumerate(bundles):
            for ri, (rs, rb, rc) in enumerate(bundles):
                if gi == ri:
                    continue
                r = appearance_s3(gb.feats, gc, gb.profile["n_prefix"], gb.profile["n_suffix"],
                                  rb.feats, rc, rb.profile["n_prefix"], rb.profile["n_suffix"], ref)
                vals.append(float(r["app_ref"]))
                rows.append({"gen": gs, "ref": rs, "app_ref": float(r["app_ref"]), "saturated": bool(r["app_saturated"])})
        out[cls] = {"ceiling": float(np.mean(vals)), "n_clips": len(bundles), "n_pairs": len(vals),
                    "sd_pairs": float(np.std(vals)), "sidedness": sided, "pairs": rows}
        print(f"  {cls:44s} n={len(bundles):2d} pairs={len(vals):4d} ceiling={out[cls]['ceiling']:.4f} sd={out[cls]['sd_pairs']:.4f}", flush=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"part_{shard}.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"[done] shard {shard}/{num_shards}: {len(out)} classes -> {PARTS / f'part_{shard}.json'}")


def aggregate() -> None:
    import numpy as np
    merged = {}
    for f in sorted(PARTS.glob("part_*.json")):
        merged.update(json.loads(f.read_text()))
    z = np.load(NPZ_V4, allow_pickle=True)
    S = 1.0 - z["m1a_S3"]
    names = [str(x) for x in z["keys"]]
    idx = collections.defaultdict(list)
    for i, n in enumerate(names):
        idx[n.split("/")[0]].append(i)
    certified = {c: float(S[np.ix_(ii, ii)][~np.eye(len(ii), dtype=bool)].mean()) for c, ii in idx.items() if len(ii) >= 2}
    calib = {}
    for c in CALIBRATION:
        if c in merged and c in certified:
            # the calibration must compare the SAME clip set: restrict the kernel pairs to the 222-corpus clips
            orig = {names[i].split("/")[1][:-4] for i in idx[c]}
            vals = [p["app_ref"] for p in merged[c]["pairs"] if p["gen"] in orig and p["ref"] in orig]
            calib[c] = {"kernel_222clips": round(float(np.mean(vals)), 4), "kernel_all_clips": round(merged[c]["ceiling"], 4),
                        "certified": round(certified[c], 4), "n_pairs_222": len(vals), "ratio": round(float(np.mean(vals)) / certified[c], 4)}
    ratios = [v["ratio"] for v in calib.values()]
    ceilings = {c: {k: v for k, v in d.items() if k != "pairs"} for c, d in sorted(merged.items())}
    out = {"method": "deployed pool-score kernel (m1_transfer.appearance_s3 / reference_stats.m1a_pair) against the FROZEN reference_v4 "
                     "populations, core_mask_v3 by class sidedness, mean app_ref over ordered same-class pairs of corpus clips. "
                     "Classes absent from the certified matrix take this value (run_eval.ceilings() overlay); certified classes keep theirs.",
           "calibration": calib, "calibration_ratio_mean": round(float(np.mean(ratios)), 4) if ratios else None,
           "ceilings": ceilings, "n_classes": len(ceilings)}
    CEIL_OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    (PARTS / "pairs_all.json").write_text(json.dumps({c: d["pairs"] for c, d in merged.items()}, sort_keys=True))
    print(f"{len(ceilings)} classes -> {CEIL_OUT.relative_to(REPO)}")
    for c, v in calib.items():
        print(f"  calibration {c:14s} kernel(222 clips) {v['kernel_222clips']:.4f} vs certified {v['certified']:.4f}  ratio {v['ratio']:.3f}  (n {v['n_pairs_222']}; all clips {v['kernel_all_clips']:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        compute(a.shard, a.num_shards)


if __name__ == "__main__":
    main()
