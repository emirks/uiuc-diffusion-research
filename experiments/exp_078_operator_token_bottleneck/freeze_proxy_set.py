"""Freeze the round-1 proxy evaluation set (advisor round-1 verdict, item D).

The proxy set is the cheap loop's readout. It is FROZEN before first use and never changed
after the one permitted round-1 refreeze, so every variant is judged on identical items.

v2 (the ONE permitted round-1 refreeze, spent before anything was trained): the owner moved the
headline from Transfer Index to **pool-% of the donor-class GT ceiling**, with +6pp targets on FOUR
focused cells. Two of them — G-unseen-same and G-zs-same — were previously a guard and absent
respectively, so the proxy could not read the headline. Both now use their ENTIRE full-instrument
grids at 2 seeds, making proxy same-cell reads instrument-grade modulo seed count.

Composition (advisor-specified, v2):
    unseen-cross          25 items x seeds 42,43   -> gates the branch in phase 1
    zs-cross              20 items x seeds 42,43   -> carries the goal; gates only in phase 2
    unseen-same           13 items x seeds 42,43   -> full grid; goal cell + shelter guard
    zs-same                8 items x seeds 42,43   -> full grid; goal cell (n=8, seed-sensitive)
    mismatched-demo probe  8 items x seed 42       -> channel-liveness detector

Derived from `eval_ladder/registry.jsonl` (the ladder's single source of truth) — never authored.
Subsets are the deterministic first-N by `item_id`, matching the pool-yardstick convention.

    python experiments/exp_078_operator_token_bottleneck/freeze_proxy_set.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "eval_ladder" / "registry.jsonl"
PASSB = REPO_ROOT / "eval_ladder" / "dominance_passB.json"
OUT = Path(__file__).resolve().parent / "proxy_set_v2.json"

# Cells the proxy draws from, and how many items x how many seeds. n=None means the whole grid.
SPEC = {
    "unseen_cross": {"cell": "G-unseen-cross", "n": None, "seeds": [42, 43]},
    "zs_cross": {"cell": "G-zs-cross", "n": None, "seeds": [42, 43]},
    "unseen_same": {"cell": "G-unseen-same", "n": None, "seeds": [42, 43]},
    "zs_same": {"cell": "G-zs-same", "n": None, "seeds": [42, 43]},
    "mismatch_probe": {"cell": "G-ref-control", "n": 8, "seeds": [42]},
}

# Headline baselines the +6pp targets are measured against (v2.0.0-R1 §7.2, ic_gen, deduped).
# raw app_ref / GT ceiling = pool-%. Recorded here so the target is auditable from the frozen file.
BASELINES = {
    "G-unseen-cross": {"raw": 0.6302, "ceiling": 0.8735, "pool_pct": 72.9, "target_pool_pct": 78.9},
    "G-zs-cross": {"raw": 0.6086, "ceiling": 0.8722, "pool_pct": 72.8, "target_pool_pct": 78.8},
    "G-unseen-same": {"raw": 0.7711, "ceiling": 0.8735, "pool_pct": 88.7, "target_pool_pct": 94.7},
    "G-zs-same": {"raw": 0.7523, "ceiling": 0.8635, "pool_pct": 90.8, "target_pool_pct": 96.8},
}


def main() -> None:
    registry = [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]
    # The bars the advisor set (unseen-cross meanTI 0.224, n=25) were computed on the rows that
    # survived amendment-1's Pass B. Freeze on exactly those so the comparison is like-for-like.
    passb_ids = {row["item_id"] for row in json.loads(PASSB.read_text())}

    frozen: dict[str, dict] = {}
    excluded: list[dict] = []

    for name, spec in SPEC.items():
        rows = sorted(
            (r for r in registry if r["cell"] == spec["cell"] and r["arm"] == "ic_gen"),
            key=lambda r: r["item_id"],
        )
        # Cross cells carry the Transfer Index, so they must match Pass B's surviving rows.
        if spec["cell"].endswith("cross"):
            kept, dropped = [], []
            for r in rows:
                (kept if r["item_id"] in passb_ids else dropped).append(r)
            for r in dropped:
                excluded.append(
                    {
                        "item_id": r["item_id"],
                        "cell": r["cell"],
                        "reason": "absent from dominance_passB.json in v2.0.0-R1 "
                        "(degeneracy guard / missing recipient-pool score); excluded so the "
                        "proxy is like-for-like with the pre-registered meanTI baselines",
                    }
                )
            rows = kept
        if spec["n"] is not None:
            rows = rows[: spec["n"]]

        frozen[name] = {
            "cell": spec["cell"],
            "seeds": spec["seeds"],
            "n_items": len(rows),
            "n_generations": len(rows) * len(spec["seeds"]),
            "items": [
                {
                    "item_id": r["item_id"],
                    "donor_class": r["donor_class"],
                    "endpoint": r["endpoint"],
                    "endpoint_class": r["endpoint_class"],
                    "reference": r["reference"],
                    "sided": r["sided"],
                    "pct_type": r["pct_type"],
                    "input_key": r["input_key"],
                }
                for r in rows
            ],
        }

    total = sum(b["n_generations"] for b in frozen.values())
    payload = {
        "version": "proxy_set_v2",
        "frozen_by": "exp_078 operator-token bottleneck campaign, advisor round-1 verdict item D, "
        "refrozen under the pool-% headline ruling (B'/C')",
        "source": "eval_ladder/registry.jsonl + dominance_passB.json (derived, never authored)",
        "rule": "FROZEN. This consumed the ONE permitted round-1 refreeze — spent before any "
        "training and before any promotion decision. No further changes.",
        "headline_metric": "pool-% of donor-class GT ceiling; margin vs ic_gen (identical content cap)",
        "baselines": BASELINES,
        "total_generations": total,
        "blocks": frozen,
        "excluded": excluded,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"[freeze] wrote {OUT.relative_to(REPO_ROOT)}")
    for name, block in frozen.items():
        donors = sorted({i["donor_class"] for i in block["items"]})
        print(
            f"  {name:20s} cell={block['cell']:16s} items={block['n_items']:3d} "
            f"seeds={block['seeds']} gens={block['n_generations']:3d} donors={len(donors)}"
        )
    print(f"  {'TOTAL':20s} {'':22s} {'':10s} {'':11s} gens={total}")
    for e in excluded:
        print(f"  [excluded] {e['item_id']}")


if __name__ == "__main__":
    main()
