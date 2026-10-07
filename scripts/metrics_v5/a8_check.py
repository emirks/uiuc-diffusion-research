#!/usr/bin/env python
"""metrics v5 A8 — the roster-driven `pairs` path reproduces the imported 047 pairs.

Pushes ``refvfx_effect_v3`` (roster id ``refvfx_teg``, 276 pairs) through v5.py's pairs path and
compares Mu / D / pxd / fid (and the 6 pillars + per-gen bl_333) against the imported 047 arm, for
BOTH gen-DINO sources:
  store   = dino_cls@dinov2b-r256 (the brief default; what every Round-2 arm uses)
  harness = score_v3_mf's own feats_or_extract loader (the source the 047 import was built from)

Run:  PYTHONPATH=<repo>/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PILLAR_JOBS=1 \
      $LAB/envs-aarch64/ltx2/bin/python scripts/metrics_v5/a8_check.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
import v5  # noqa: E402
import blend  # noqa: E402

ARM = "refvfx_teg"
IMPORTED = REPO / "store/evals/047_transport_v5_gridv3__dai__2026-09-23/refvfx_effect_v3"


def _load_pairs(d):
    return {(r["item_id"], r["seed"], r["ref"]): r
            for r in (json.loads(l) for l in (d / "pairs.jsonl").read_text().splitlines())}


def _load_pg(d):
    return {(r["item_id"], r["seed"]): r
            for r in (json.loads(l) for l in (d / "per_gen.jsonl").read_text().splitlines())}


def main() -> int:
    ctx = v5._pairs_context()
    fs = v5.FeatureStore(REPO)
    pop, fpop, ceil = blend.build_from_reference(blend.load_reference(v5.REFERENCE))
    arm = ctx["by_id"][ARM]
    rows, subs, label, mode = v5._pairs_rows(arm, ctx)
    print(f"[A8] arm {ARM} ({label}, {mode}): {len(rows)} rows, "
          f"{len({r['_gen'] for r in rows})} gens, {len({r['_refkey'] for r in rows})} refs")

    imp_pairs = _load_pairs(IMPORTED)
    imp_pg = _load_pg(IMPORTED)
    chans = ["Mu", "D", "pxd", "fid"]
    pillars = ["T", "Tshape", "E", "Mw"]
    ok = False
    for src in ("store", "harness"):
        cache = v5._build_pair_cache(rows, subs, ctx, fs, gen_dino=src)
        pr, pg = blend.score_cache(cache, v5._grid_map(
            [f"store/gens/{s}" for s, _ in v5._parts_of(arm["gens"])]), pop, fpop, ceil, label, mode)
        got = {(r["item_id"], r["seed"], r["ref"]): r for r in pr}
        got_pg = {(r["item_id"], r["seed"]): r for r in pg}
        shared = set(got) & set(imp_pairs)
        miss = (set(got) ^ set(imp_pairs))
        maxd = {c: max(abs(got[k][c] - imp_pairs[k][c]) for k in shared) for c in chans + pillars}
        shared_g = set(got_pg) & set(imp_pg)
        bl = max(abs(got_pg[k]["bl_333"] - imp_pg[k]["bl_333"]) for k in shared_g)
        cols = list(blend.WEIGHTS)
        allw = max(abs(got_pg[k][c] - imp_pg[k][c]) for k in shared_g for c in cols)
        passed = all(maxd[c] <= 1e-9 for c in chans) and bl <= 1e-9
        print(f"\n[A8 gen-dino={src}] pairs {len(got)} vs imported {len(imp_pairs)}; "
              f"shared {len(shared)}, key mismatches {len(miss)}")
        print("   PAIR max|diff|:  " + "  ".join(f"{c}={maxd[c]:.3e}" for c in chans))
        print("   pillars max|diff|: " + "  ".join(f"{c}={maxd[c]:.3e}" for c in pillars))
        print(f"   PER-GEN bl_333 max|diff| ({len(shared_g)} gens) = {bl:.3e}; all 10 weight cols = {allw:.3e}")
        print(f"   -> abs<=1e-9 on Mu/D/pxd/fid + bl_333: {'PASS' if passed else 'no'}")
        if src == "harness":
            ok = passed
    print(f"\n[A8] VERDICT (gen-dino=harness, the import's own DINO loader): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
