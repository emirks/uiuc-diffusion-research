#!/bin/bash
# ctt_v2 — window-close procedure: rebuild the content pool with the expansion, re-freeze both
# grids against it, verify every invariant, and launch the S2 mass render.
#
# The advisor's Q3 ruling makes this unconditional: "S2 launches at window close, unconditionally,
# on the largest gate-passing pool." build_content_pool.py enforces Gates A/B and, if either
# fails, greedily trims the most-redundant NEW clips (the v1 clips are untouchable) until they
# pass, logging every removal. The v1 floor passes by construction, so this always terminates in
# a launchable pool. Nothing here is allowed to hold S2 past this point.
#
#   bash experiments/exp_081_s2_stratum/refreeze_and_launch.sh [--dry-run]

set -eo pipefail
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
REPO=$LAB/diffusion-research
PY=$LAB/envs/diffusion/bin/python
DRY=${1:-}
cd "$REPO"

echo "=== 1/5  content pool (v1 clean + expansion), Gates A/B, deterministic trim ==="
$PY experiments/exp_081_s2_stratum/build_content_pool.py --with-v2

echo
echo "=== 2/5  re-freeze the S2 grid against the expanded pool ==="
$PY experiments/exp_081_s2_stratum/plan_s2.py

echo
echo "=== 3/5  re-freeze the S3 grid (same pool; keeps the cross-modality pair overlap) ==="
$PY experiments/exp_082_s3_stratum/plan_s3.py

echo
echo "=== 4/5  verify every invariant before a single clip is rendered ==="
$PY - <<'PYEOF'
import json, collections, sys
import numpy as np
R = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/"
pool = json.load(open(R + "data/processed/ctt_v2_strata/CONTENT_POOL.json"))
lb = set(json.load(open(R + "data/processed/ctt_v2_strata/LETTERBOX_AUDIT.json"))["excluded"])
res = {e["clip_id"] for e in pool["reserved"]}
fail = []
for name, plan_path, holdout_key in (
        ("S2", "experiments/exp_081_s2_stratum/PLAN_S2.json", "shaders"),
        ("S3", "experiments/exp_082_s3_stratum/PLAN_S3.json", None)):
    p = json.load(open(R + plan_path))
    ops, pairs, m = p["ops"], p["pairs"], p["design"]["contents_per_op"]
    used = {c for x in pairs for c in (x["A"], x["B"])}
    deg = collections.Counter()
    nd = 0
    for o in ops:
        prim = o["candidates"][:m]
        cl = [c for pi in prim for c in (pairs[pi]["A"], pairs[pi]["B"])]
        if len(set(cl)) != 2 * m:
            nd += 1
        for pi in prim:
            deg[pi] += 1
    d = list(deg.values())
    print(f"  {name}: {len(ops)} ops x {m} = {len(ops)*m} clips | {len(pairs)} pairs | "
          f"{len(used)} contents | ops/pair {min(d)}/{int(np.median(d))}/{max(d)}")
    for label, bad in (("letterboxed content", used & lb),
                       ("reserved content", used & res),
                       ("non-disjoint ops", nd),
                       ("duplicate op_id", len(ops) - len({o['op_id'] for o in ops}))):
        if bad:
            fail.append(f"{name}: {label} -> {bad}")
    if holdout_key:
        leak = {o["shader"] for o in ops} & set(p["holdout"][holdout_key])
        if leak:
            fail.append(f"{name}: held-out shader in roster -> {leak}")
    else:
        if any(o["family"] == p["holdout"]["family"] for o in ops):
            fail.append(f"{name}: held-out camera family in roster")
g = pool["gates"]["after_trim"]
print(f"  pool: {pool['n_training']} training + {pool['n_reserved']} reserved | "
      f"Gate A {g['gate_a_mean_cos']} (<= {g['gate_a_bar']}) | "
      f"Gate B {g['gate_b_matched_pr']} (>= {g['gate_b_bar']}) | trimmed {len(pool['trim_log'])}")
if not g["pass"]:
    fail.append("pool gates FAIL after trim")
if fail:
    print("\n  INVARIANT FAILURES:")
    for f in fail:
        print("   -", f)
    sys.exit(1)
print("  all invariants OK")
PYEOF

echo
if [ "$DRY" = "--dry-run" ]; then
  echo "=== 5/5  --dry-run: not launching ==="
  exit 0
fi
echo "=== 5/5  launch the S2 mass render (CPU array, 20 shards) ==="
JID=$(sbatch --parsable experiments/exp_081_s2_stratum/job_render_array.sbatch)
echo "s2render array -> $JID"
sed -i '/^s2_launched=/d' "$LAB/misc/ctt_v2_s2s3/watchdog.state"
echo "s2_launched=1" >> "$LAB/misc/ctt_v2_s2s3/watchdog.state"
echo "watchdog armed to track and resubmit the array"
