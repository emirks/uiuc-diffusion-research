"""exp_060 — σ_seed compute (SPEC §6.4). Join the certified items.jsonl back
to the probe-group map (score.py does not propagate the manifest's
`probe_group`; the instrument is never modified — SPEC §7), inject
probe_group, then call the certified certify.seeds.sigma_seed to get the
per-metric pooled between-seed std + MDE@n table.

Runs numpy-only, login-node safe (SPEC §8 `--from-items` re-reporting). MUST
import seeds from the CERTIFIED worktree src (PYTHONPATH=$WT/src).

    PYTHONPATH=$WT/src python compute_sigma.py \
        --items <out>/adapter/items.jsonl --groups dataset/probe_groups.json \
        --out <out>/sigma_seed.json
"""

import argparse
import json
import pathlib

import numpy as np

from diffusion.transition_eval.certify import seeds as seeds_mod

EXP = pathlib.Path(__file__).resolve().parent
EXPECTED_SEEDS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="scored items.jsonl (certified)")
    ap.add_argument("--groups", default=str(EXP / "dataset/probe_groups.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    groups = json.loads(pathlib.Path(args.groups).read_text())
    rows = [json.loads(l) for l in pathlib.Path(args.items).read_text().splitlines() if l.strip()]

    # ---- certification / stamp sanity ------------------------------------
    certified = all(r.get("provenance", {}).get("certified") for r in rows if "provenance" in r)
    error_rows = [r["item_id"] for r in rows if "error" in r]

    # ---- inject probe_group (join on item_id) ----------------------------
    injected, unmatched = [], []
    for r in rows:
        g = groups.get(r["item_id"])
        if g is None:
            unmatched.append(r["item_id"])
            continue
        injected.append({**r, "probe_group": g})

    aug_path = pathlib.Path(args.out).with_name("items_with_probe_group.jsonl")
    aug_path.write_text("\n".join(json.dumps(r) for r in injected) + "\n")

    # ---- per-group finiteness audit (report loudly) ----------------------
    by_group: dict[str, list] = {}
    for r in injected:
        by_group.setdefault(r["probe_group"], []).append(r)
    audit = {}
    for g, rs in sorted(by_group.items()):
        finite = {}
        for m in seeds_mod.METRICS:
            vals = [r.get(m) for r in rs]
            finite[m] = sum(1 for x in vals if x is not None and np.isfinite(x))
        audit[g] = {"n_rows": len(rs), "finite": finite}

    bad_app_ref = {g: a["finite"]["app_ref"] for g, a in audit.items()
                   if a["finite"]["app_ref"] != EXPECTED_SEEDS}

    # ---- compute sigma_seed via the CERTIFIED protocol -------------------
    result = seeds_mod.sigma_seed(aug_path, args.out)
    result["_audit"] = {
        "certified_all_rows": bool(certified),
        "n_rows": len(rows),
        "n_injected": len(injected),
        "error_rows": error_rows,
        "unmatched_item_ids": unmatched,
        "expected_seeds_per_group": EXPECTED_SEEDS,
        "groups_without_5_finite_app_ref": bad_app_ref,
        "per_group_finite": audit,
    }
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))

    print(f"[sigma_seed] {result['n_groups']} probe_groups; certified={certified}")
    if error_rows:
        print(f"[WARN] {len(error_rows)} ERROR ROWS: {error_rows}")
    if unmatched:
        print(f"[WARN] {len(unmatched)} unmatched item_ids: {unmatched}")
    if bad_app_ref:
        print(f"[WARN] groups without {EXPECTED_SEEDS} finite app_ref: {bad_app_ref}")
    print("\n=== sigma (pooled between-seed std) ===")
    for m, s in result["sigma"].items():
        mde = result["mde"][m]
        print(f"  {m:12s} sigma={s:.4f}  MDE n5={mde['5']} n10={mde['10']} "
              f"n20={mde['20']} n40={mde['40']}")
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
