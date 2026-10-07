#!/usr/bin/env python
"""grid v3 — ceilings for the new corpus classes: score the corpus AGAINST ITSELF with the production scorer.

The pool yardstick (POOL_YARDSTICK.md) divides a generation's mean pool app_ref by the class CEILING = the
same-class off-diagonal mean of the certified v4 matrix (run_eval.ceilings(), keyed by the 39 original classes).
New classes are absent from that matrix. This builds a scorer manifest in which every clip of a class plays
the "generation" against every OTHER clip of the class as the "reference" (ordered pairs), so the per-class mean
app_ref over pairs IS the ceiling, computed by the same instrument path (reference_v4.npz kernel, one machine)
that scores real generations. No conditioning windows are attached (the certification matrix was built from
whole clips); the scorer then assumes its default 9-frame prefix mask — the same for every pair.

Classes scored: the 7 new Higgsfield zero-shot classes, the 34 EffectData donor classes, the 17 topped-up
classes (recomputed with their enlarged pools), and 5 ORIGINAL classes as a CALIBRATION set whose certified
ceilings are known — the aggregate compares scorer-path vs certified values before any new ceiling is used.

Run:  python scripts/grid_v3/ceilings_manifest.py [--chunks 8]     -> misc/2026-09-07_eval_grid_v2/ceilings/manifests/eval_c{N}.json
      python scripts/grid_v3/ceilings_manifest.py --aggregate      -> eval_ladder/ceilings_v3.json (+ calibration report)
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
STD = REPO / "data/processed/transitions_std121"
GRID = yaml.safe_load((REPO / "eval_ladder/grid_v3.yaml").read_text())
OUT_DIR = REPO / "misc/2026-09-07_eval_grid_v2/ceilings"
SCORES = REPO / "outputs/eval/grid_v3_ceilings"
CEIL_OUT = REPO / "eval_ladder/ceilings_v3.json"
CALIBRATION = ["shadow", "portal", "cotton_cloud", "hero_flight", "shadow_smoke"]
NPZ_V4 = (REPO / ".claude/worktrees/eval-v4-cert/outputs/eval/certification/4.0.0-draft.1/analysis/distance_matrices.npz")


def classes_to_score() -> dict[str, str]:
    man = json.loads((STD / "corpus_manifest.json").read_text())
    donors = set(GRID["new_zero_shot_classes"])
    for key in ("tier1_topups", "tier1_reference_only", "flame_additions", "zs_pool_topups"):
        donors |= set(GRID[key])
    rows = [json.loads(l) for l in (REPO / "eval_ladder/registry_v3.jsonl").read_text().splitlines()]
    donors |= {r["donor_class"] for r in rows if r["arm"] == "ic_gen" and r["donor_class"].startswith(GRID["effectdata"]["clip_prefix"] + ".")}
    donors |= set(CALIBRATION)
    out = {}
    for c in sorted(donors):
        n = len(list((STD / c).glob("*.mp4")))
        if n >= 2:
            out[c] = man["classes"][c]["sidedness"]
    return out


def build(chunks: int) -> None:
    cls = classes_to_score()
    items = []
    for c, sided in cls.items():
        clips = sorted(p.stem for p in (STD / c).glob("*.mp4"))
        for g in clips:
            for r in clips:
                if r == g:
                    continue
                items.append({
                    "item_id": f"ceil__{c}__{g}__ref_{r}",
                    "generated_video": str((STD / c / f"{g}.mp4").relative_to(REPO)),
                    "reference_video": str((STD / c / f"{r}.mp4").relative_to(REPO)),
                    "style": c,
                    "n_endpoints": 2 if sided == "twosided" else 1,
                    "arm": "corpus_ceiling",
                    "twin_of": None,
                    "notes": f"grid v3 ceiling pair; class {c}; no conditioning (whole-clip, certification convention)",
                })
    mdir = OUT_DIR / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    for old in mdir.glob("eval_c*.json"):
        old.unlink()
    for i in range(chunks):
        part = items[i::chunks]
        (mdir / f"eval_c{i}.json").write_text(json.dumps(part, indent=1))
    per = collections.Counter(it["style"] for it in items)
    (OUT_DIR / "classes.json").write_text(json.dumps({"classes": cls, "pairs_per_class": per, "calibration": CALIBRATION}, indent=1, sort_keys=True))
    print(f"{len(cls)} classes, {len(items)} ordered pairs -> {chunks} chunks in {mdir.relative_to(REPO)}; "
          f"pairs/class min {min(per.values())} median {st.median(per.values())} max {max(per.values())}")


def aggregate() -> None:
    import numpy as np
    rows = []
    for f in sorted(SCORES.glob("*/items.jsonl")):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("app_ref") is not None:
                    rows.append(r)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["style"]].append(float(r["app_ref"]))
    z = np.load(NPZ_V4, allow_pickle=True)
    S = 1.0 - z["m1a_S3"]
    names = [str(x) for x in z["keys"]]
    idx = collections.defaultdict(list)
    for i, n in enumerate(names):
        idx[n.split("/")[0]].append(i)
    certified = {c: float(S[np.ix_(ii, ii)][~np.eye(len(ii), dtype=bool)].mean()) for c, ii in idx.items() if len(ii) >= 2}
    calib = {c: {"scorer_path": round(st.mean(by[c]), 4), "certified": round(certified[c], 4), "n_pairs": len(by[c])}
             for c in CALIBRATION if c in by and c in certified}
    ratio = [v["scorer_path"] / v["certified"] for v in calib.values()]
    out = {"method": "scorer-path ceiling: mean app_ref over ordered same-class pairs of corpus clips, v4 kernel (reference_v4.npz), "
                     "no conditioning; classes absent from the certified matrix get THIS value; original classes keep the certified one",
           "calibration": calib, "calibration_ratio_mean": round(st.mean(ratio), 4) if ratio else None,
           "ceilings": {c: {"ceiling": round(st.mean(v), 4), "n_pairs": len(v), "sd_pairs": round(st.pstdev(v), 4)} for c, v in sorted(by.items())},
           "overlay_rule": "run_eval.ceilings(): npz value if the class is in the certified matrix, else ceilings_v3.json"}
    CEIL_OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{len(by)} classes aggregated -> {CEIL_OUT.relative_to(REPO)}")
    for c, v in calib.items():
        print(f"  calibration {c:14s} scorer-path {v['scorer_path']:.4f} vs certified {v['certified']:.4f}  ratio {v['scorer_path']/v['certified']:.3f}  (n {v['n_pairs']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        build(a.chunks)


if __name__ == "__main__":
    main()
