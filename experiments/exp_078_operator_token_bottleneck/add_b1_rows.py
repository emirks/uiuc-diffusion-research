"""Add `b1` (operator-token bottleneck) rows to the WORKTREE registry — derived, never authored.

run_gen and run_eval are both registry-driven: they act on the rows whose `arm` matches. So the
bottleneck arm needs its own rows, or `run_gen --arm b1` asserts "no registry rows for arm=b1" and
the chained proxy-generation fails.

A b1 row is exactly its ic_gen twin with two fields renamed — `arm` and the `arm` token inside
`item_id` — and everything else preserved, crucially `input_key` (so the keyed base-twin join is
unchanged) and `gt_pool_class` / `reference` / `endpoint` (so the copy-guarded pool is identical).
The ONLY thing that differs between a b1 generation and its ic_gen twin is the adapter: same task,
same pool, same yardstick.

Scope: the five proxy_set_v2 cells. Idempotent — rewrites the b1 block each run.

    python experiments/exp_078_operator_token_bottleneck/add_b1_rows.py [--check]

This mutates ONLY the worktree's own tracked registry.jsonl; the parallel campaign reads the main
checkout, which is untouched.
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "eval_ladder" / "registry.jsonl"

PROXY_CELLS = {"G-unseen-cross", "G-zs-cross", "G-unseen-same", "G-zs-same", "G-ref-control"}
SRC_ARM = "ic_gen"
DST_ARM = "b1"


def derive_b1(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r["arm"] != SRC_ARM or r["cell"] not in PROXY_CELLS:
            continue
        b = dict(r)
        b["arm"] = DST_ARM
        b["item_id"] = r["item_id"].replace(f"__{SRC_ARM}__", f"__{DST_ARM}__", 1)
        out.append(b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    rows = [json.loads(x) for x in REGISTRY.read_text().splitlines() if x.strip()]
    kept = [r for r in rows if r["arm"] != DST_ARM]  # drop any prior b1 block (idempotent)
    b1 = derive_b1(kept)

    # Invariants that would otherwise fail silently downstream.
    ids = [r["item_id"] for r in kept] + [r["item_id"] for r in b1]
    assert len(ids) == len(set(ids)), "item_id collision after adding b1 rows"
    keys = {r["input_key"] for r in kept if r["arm"] == "base"}
    orphan = [r["item_id"] for r in b1 if r["input_key"] not in keys]
    assert not orphan, f"{len(orphan)} b1 rows have no base twin, e.g. {orphan[:3]}"

    by_cell = {}
    for r in b1:
        by_cell.setdefault(r["cell"], 0)
        by_cell[r["cell"]] += 1
    print(f"[b1] {len(b1)} rows across {len(by_cell)} cells: " +
          ", ".join(f"{c}={n}" for c, n in sorted(by_cell.items())))
    print(f"[b1] every b1 row has a base twin ({len(keys)} base input_keys); no item_id collisions")

    if args.check:
        print("[b1] --check: nothing written")
        return

    with REGISTRY.open("w") as f:
        for r in kept + b1:
            f.write(json.dumps(r) + "\n")
    print(f"[b1] registry rewritten: {len(kept)} original + {len(b1)} b1 = {len(kept) + len(b1)} rows")


if __name__ == "__main__":
    main()
