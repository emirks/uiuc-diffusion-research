"""eval_ladder v2.1.0 — append the two CLEAN baselines to the registry.

Owner call 2026-07-23: the base+reference arm is a COPIER, not a baseline. The honest baselines
for every specialist/generalist task are (1) prompt only and (2) prompt + endpoint conditioning
(sidedness-aware), NEITHER carrying a reference. This script derives one of each for every
distinct treatment task — (donor_class, endpoint, sidedness) — and appends them to the frozen
registry WITHOUT touching the original 444 rows (purely additive; that is the v2.1.0 bump).

    python eval_ladder/add_baselines.py [--write]

Video is per-task (simple: one item_id → one video → one score). A prompt-only clip for an
endpoint is content-identical across donors, so this regenerates it per task — cheap (base
weights, no adapter) and it keeps the gen/eval pipeline byte-for-byte the standard one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompts  # noqa: E402

REGISTRY = HERE / "registry.jsonl"
BASELINE_ARMS = ("base_prompt", "base_cond")


def load() -> list[dict]:
    return [json.loads(x) for x in REGISTRY.read_text().splitlines() if x.strip()]


def baseline_rows(rows: list[dict]) -> list[dict]:
    # one representative treatment row per (donor, endpoint, sided): it carries the rendered
    # prompt, the %-type and the GT pool we must score the baseline against.
    tasks: dict[tuple, dict] = {}
    for r in rows:
        if r["arm"].startswith("spec_") or r["arm"] == "ic_gen":
            tasks.setdefault((r["donor_class"], r["endpoint"], r["sided"]), r)

    out = []
    for (donor, endpoint, sided), r in sorted(tasks.items()):
        for arm in BASELINE_ARMS:
            cond = "none" if arm == "base_prompt" else ("two" if sided == "two" else "prefix")
            item_id = f"{arm}__{donor}__{endpoint}"
            out.append({
                "item_id": item_id,
                "mismatched_reference": False,
                "cell": "BL-prompt" if arm == "base_prompt" else "BL-cond",
                "priority": "P0",
                "arm": arm,
                "ref_novelty": "none",
                "content": r["content"],
                "donor_class": donor,
                "endpoint": endpoint,
                "endpoint_class": r["endpoint_class"],
                "endpoint_split": r["endpoint_split"],
                "sided": sided,
                "reference": None,
                "reference_split": None,
                # base_prompt: prompt only (the model is NOT given the endpoint). base_cond: the
                # same rendered prompt + endpoint conditioning. Same renderer as everything else.
                "prompt": r["prompt"],
                "pct_type": r["pct_type"],
                "gt_pool_class": r["gt_pool_class"],
                # conditioning drives run_gen / run_eval: 'none' => no prefix; else prefix(+suffix)
                "conditioning": cond,
                "input_key": item_id,          # baselines are never twinned; unique key
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = load()
    existing = {r["item_id"] for r in rows}
    new = [r for r in baseline_rows(rows) if r["item_id"] not in existing]

    # invariants
    ids = [r["item_id"] for r in new]
    assert len(ids) == len(set(ids)), "duplicate baseline item_id"
    for r in new:
        assert prompts.MARKER not in r["prompt"], f"{r['item_id']}: outcome leaked"
        assert (r["conditioning"] == "none") == (r["arm"] == "base_prompt")

    n_ep = len({(r["endpoint"], r["sided"]) for r in new})
    print(f"[baselines] {len(new)} new rows over {n_ep} distinct endpoints "
          f"({len(new)//2} tasks × 2 baselines); {len(rows)} existing rows untouched")
    for arm in BASELINE_ARMS:
        print(f"   {arm}: {sum(1 for r in new if r['arm']==arm)}")
    if args.write:
        with REGISTRY.open("a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
        print(f"[baselines] appended -> {REGISTRY.name} (now {len(rows)+len(new)} rows)")
    else:
        print("[baselines] dry run — pass --write to append")


if __name__ == "__main__":
    main()
