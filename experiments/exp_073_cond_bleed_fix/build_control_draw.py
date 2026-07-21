#!/usr/bin/env python
"""Pre-registered stratified draw of the n=24 one-sided ic3 negative-control rows (CONSULT 2).

12 from manifest_ic3 one-sided (prefix_only), proportional across tiers A/B/C.
12 from manifest_ic3_x (all one-sided), proportional across recipient classes.
Within each stratum: sort IDs lexicographically, draw with fixed RNG seed 20260721
(largest-remainder proportional allocation; deterministic). Also reports whether any
drawn one-sided row shares a clip with the 38 CORRECTED two-sided target clips (advisor's
force-include check). Writes control_draw.json. DETERMINISTIC — reproducible.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20260721
N_EACH = 12
REPO = Path(__file__).resolve().parents[2]
G = REPO / "experiments/exp_065_ladder_v3_grid/dataset"
PAIRS = REPO / "experiments/exp_064_ic3_aligned_retrain/dataset/pairs.json"
OUT = Path(__file__).resolve().parent / "control_draw.json"


def largest_remainder(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Allocate `total` across strata proportional to sizes (largest-remainder, deterministic)."""
    s = sum(sizes.values())
    raw = {k: total * v / s for k, v in sizes.items()}
    base = {k: int(v) for k, v in raw.items()}
    rem = total - sum(base.values())
    # distribute remaining by largest fractional part, tie-break lexicographic
    order = sorted(sizes, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:rem]:
        base[k] += 1
    return base


def main() -> None:
    corrected = {(p["class"], p["target"]) for p in json.loads(PAIRS.read_text())
                 if p["sidedness"] == "twosided"}  # 38 corrected two-sided target clips
    rng = random.Random(SEED)

    # ---- manifest_ic3 one-sided, stratified by tier ----
    ic3 = json.loads((G / "manifest_ic3.json").read_text())["rows"]
    ic3_os = [r for r in ic3 if r.get("prefix_only") and not r.get("deferred")]
    by_tier = defaultdict(list)
    for r in ic3_os:
        by_tier[r["tier"]].append(r)
    alloc_t = largest_remainder({t: len(v) for t, v in by_tier.items()}, N_EACH)
    pick_ic3 = []
    for t in sorted(by_tier):
        ids = sorted(r["id"] for r in by_tier[t])
        pick_ic3 += rng.sample(ids, alloc_t[t])

    # ---- manifest_ic3_x (all one-sided), stratified by recipient class ----
    icx = json.loads((G / "manifest_ic3_x.json").read_text())["rows"]
    icx_os = [r for r in icx if r.get("prefix_only") and not r.get("deferred")]
    by_cls = defaultdict(list)
    for r in icx_os:
        by_cls[r["class"]].append(r)
    alloc_c = largest_remainder({c: len(v) for c, v in by_cls.items()}, N_EACH)
    pick_icx = []
    for c in sorted(by_cls):
        ids = sorted(r["id"] for r in by_cls[c])
        pick_icx += rng.sample(ids, alloc_c[c])

    picked_ids = set(pick_ic3) | set(pick_icx)

    # ---- subclass-sharing check: do drawn rows' clips overlap the 38 corrected clips? ----
    rowmap = {r["id"]: r for r in ic3_os + icx_os}
    overlap = []
    for iid in sorted(picked_ids):
        r = rowmap[iid]
        clips = {(r["class"], r.get("clip")), (r["class"], r.get("endpoints"))}
        # ic3_x endpoint belongs to donor_class
        if r.get("donor_class"):
            clips.add((r["donor_class"], r.get("clip")))
            clips.add((r["donor_class"], r.get("endpoints")))
        if clips & corrected:
            overlap.append({"id": iid, "shared": sorted(str(x) for x in (clips & corrected))})

    result = {
        "seed": SEED, "n": len(picked_ids),
        "alloc_ic3_by_tier": alloc_t, "alloc_ic3x_by_class": alloc_c,
        "ic3_onesided_ids": sorted(pick_ic3), "ic3x_ids": sorted(pick_icx),
        "all24": sorted(picked_ids),
        "n_corrected_clips": len(corrected),
        "subclass_overlap": overlap,
        "note": "one-sided classes use their own clips; ic3_x endpoints come from donor classes. "
                "Training-data overlap with the 38 corrected two-sided TARGET clips is checked above.",
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"[draw] n={len(picked_ids)} (ic3 {len(pick_ic3)} by tier {alloc_t}; ic3_x {len(pick_icx)} by class {alloc_c})")
    print(f"[draw] subclass_overlap with 38 corrected clips: {len(overlap)} rows")
    for o in overlap:
        print("   ", o)
    print(f"[draw] wrote {OUT}")


if __name__ == "__main__":
    main()
