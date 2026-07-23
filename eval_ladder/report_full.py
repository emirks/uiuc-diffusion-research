"""ladder2 — the COMPLETE metric readout, emitted as markdown for REPORT.md.

`run_eval.py --mode report` prints the headline yardstick only (pool-% + margin). This prints
EVERY v4 field the instrument produces, per cell x arm, so the report is auditable without
re-running anything.

Aggregation order matters and is fixed here:

    (generation, seed) x pool-reference   -- scored rows
      -> mean over pool references        -- pair metrics (app_ref, cam_zpr, obj_csls, copy_max...)
                                             average; per-generation metrics (scalar_*, prefix_*,
                                             seam, margin) are CONSTANT across refs, so the mean
                                             is the value itself and no special-casing is needed
      -> mean over seeds                  -- one number per registry item
      -> mean over items                  -- the cell x arm cell

Flags are reported as RATES over (generation, seed), never averaged as floats.

Metric directions come from the v4 SPEC (M1a..M3b); they are printed in the header so a reader
never has to guess whether lower is better.

    python eval_ladder/report_full.py            # markdown to stdout
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

import run_eval  # noqa: E402  (registry loader + ceilings + the NPZ pin)

SCORES = REPO_ROOT / "outputs/eval/ladder2"

#: (field, label, direction, decimals). Direction is from the SPEC, printed in the legend.
PAIR_METRICS = [
    ("app_ref", "M1a app_ref", "up", 3),
    ("cam_zpr", "M1b cam_zpr", "down", 3),
    ("obj_csls", "M1c obj_csls", "down", 3),
    ("copy_max", "M2a copy_max", "down", 3),
    ("cam_dtw", "cam_dtw", "up", 3),
    ("cam_corr", "cam_corr", "up", 3),
    ("obj_match", "obj_match", "up", 3),
    ("app_ref_v3", "app_ref_v3", "up", 3),
    ("cross", "cross", "info", 3),
]
GEN_METRICS = [
    ("margin", "M2b margin", "up", 3),
    ("app_target", "app_target", "up", 3),
    ("prefix_dino", "M3a pre_dino", "up", 3),
    ("prefix_lpips", "M3a pre_lpips", "down", 4),
    ("max_seam_z", "M3b seam_z", "down", 2),
    ("scalar_depth", "depth", "info", 3),
    ("scalar_depart", "depart", "info", 3),
    ("scalar_arrive", "arrive", "info", 3),
    ("scalar_core_frac", "core_frac", "info", 3),
]
FLAGS = [("near_copy", "near_copy"), ("cross_high", "cross_high"),
         ("app_saturated", "app_sat"), ("core_degenerate", "core_degen"),
         ("intruder", "intruder")]

ALL = PAIR_METRICS + GEN_METRICS


def load_scored() -> dict[tuple[str, int], list[dict]]:
    """(registry item_id, seed) -> its scored pool rows. Deduped on the full eval id: scoring
    ran in several incremental passes and a row could in principle appear twice."""
    seen: set[str] = set()
    out: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for f in sorted(SCORES.glob("*/items.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["item_id"] in seen:
                continue
            seen.add(r["item_id"])
            head, _, _ref = r["item_id"].rpartition("__ref_")
            item, _, seed = head.rpartition("__s")
            out[(item, int(seed))].append(r)
    return out


def collapse(rows: list[dict]) -> dict:
    """One scored generation: mean over its pool references, flags as rates."""
    out: dict[str, float] = {}
    for field, _label, _dir, _dp in ALL:
        vals = [r[field] for r in rows
                if r.get(field) is not None and isinstance(r.get(field), (int, float))]
        if vals:
            out[field] = st.mean(vals)
    for field, _label in FLAGS:
        vals = [r.get(field) for r in rows if field in r]
        if field == "intruder":  # a class name or null, not a bool
            out[field] = st.mean([1.0 if v else 0.0 for v in vals]) if vals else float("nan")
        else:
            vals = [v for v in vals if isinstance(v, bool)]
            out[field] = st.mean([1.0 if v else 0.0 for v in vals]) if vals else float("nan")
    return out


def mean_or_nan(vals: list[float]) -> float:
    vals = [v for v in vals if v == v]
    return st.mean(vals) if vals else float("nan")


def fmt(v: float, dp: int) -> str:
    return "—" if v != v else f"{v:.{dp}f}"


def main() -> None:
    registry = {r["item_id"]: r for r in run_eval.load_registry()}
    ceil = run_eval.ceilings()
    scored = load_scored()

    # (item_id) -> metric -> mean over seeds; plus the pool-% yardstick
    per_item: dict[str, dict[str, float]] = {}
    for (item, _seed), rows in scored.items():
        if item not in registry:
            continue
        per_item.setdefault(item, collections.defaultdict(list))
        c = collapse(rows)
        for k, v in c.items():
            per_item[item][k].append(v)
        cls = registry[item]["gt_pool_class"]
        if cls in ceil and "app_ref" in c:
            per_item[item]["pct"].append(c["app_ref"] / ceil[cls])
    per_item = {k: {m: mean_or_nan(v) for m, v in d.items()} for k, d in per_item.items()}

    # ------------------------------------------------------------------ cell x arm aggregation
    groups: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for item in per_item:
        row = registry[item]
        arm = row["arm"] if row["arm"] in ("base", "text_floor", "ic_gen") else "specialist"
        groups[(row["cell"], arm)].append(item)

    order = ["specialist", "ic_gen", "base", "text_floor"]

    def emit(title: str, fields: list[tuple], note: str) -> None:
        print(f"\n#### {title}\n")
        print(note + "\n")
        head = "| cell | arm | n | " + " | ".join(l for _f, l, _d, _dp in fields) + " |"
        print(head)
        print("|" + "---|" * (3 + len(fields)))
        for (cell, arm) in sorted(groups, key=lambda k: (k[0], order.index(k[1]))):
            items = groups[(cell, arm)]
            cells = [fmt(mean_or_nan([per_item[i].get(f, float("nan")) for i in items]), dp)
                     for f, _l, _d, dp in fields]
            print(f"| {cell} | {arm} | {len(items)} | " + " | ".join(cells) + " |")

    print("<!-- generated by eval_ladder/report_full.py — do not hand-edit -->")

    # ---------------------------------------------------------------- headline: raw/ceiling/pct
    print("\n#### Headline yardstick — raw · ceiling · %\n")
    print("| cell | arm | n | raw app_ref | GT ceiling | pool-% | %type |")
    print("|---|---|---|---|---|---|---|")
    for (cell, arm) in sorted(groups, key=lambda k: (k[0], order.index(k[1]))):
        items = groups[(cell, arm)]
        raw = mean_or_nan([per_item[i].get("app_ref", float("nan")) for i in items])
        cl = mean_or_nan([ceil.get(registry[i]["gt_pool_class"], float("nan")) for i in items])
        pc = mean_or_nan([per_item[i].get("pct", float("nan")) for i in items])
        pt = {registry[i]["pct_type"] for i in items}
        print(f"| {cell} | {arm} | {len(items)} | {fmt(raw, 4)} | {fmt(cl, 4)} | "
              f"{'—' if pc != pc else f'{pc:.1%}'} | {'/'.join(sorted(pt))} |")

    emit("Reference-relative metrics (mean over the same-class GT pool)", PAIR_METRICS,
         "Every field is scored generation-vs-pool-clip and averaged over the pool, then over "
         "seeds. `app_ref` ↑ · `cam_zpr` ↓ · `obj_csls` ↓ · `copy_max` ↓.")
    emit("Per-generation metrics (constant across pool references)", GEN_METRICS,
         "`margin` ↑ (M2b intrusion: positive = the intended class wins) · `prefix_dino` ↑ / "
         "`prefix_lpips` ↓ (M3a endpoint fidelity) · `max_seam_z` ↓ (M3b flag, z≳3 = snap).")

    print("\n#### Flag rates (fraction of scored generations)\n")
    print("| cell | arm | n | " + " | ".join(l for _f, l in FLAGS) + " |")
    print("|" + "---|" * (3 + len(FLAGS)))
    for (cell, arm) in sorted(groups, key=lambda k: (k[0], order.index(k[1]))):
        items = groups[(cell, arm)]
        vals = [fmt(mean_or_nan([per_item[i].get(f, float("nan")) for i in items]), 3)
                for f, _l in FLAGS]
        print(f"| {cell} | {arm} | {len(items)} | " + " | ".join(vals) + " |")

    # ------------------------------------------------------------------------- per-arm rollup
    by_arm: dict[str, list[str]] = collections.defaultdict(list)
    for item in per_item:
        by_arm[registry[item]["arm"]].append(item)
    print("\n#### Per-arm rollup (all cells that arm appears in)\n")
    print("| arm | items | raw app_ref | M2a copy_max | near_copy | M3a pre_dino | M3b seam_z |")
    print("|---|---|---|---|---|---|---|")
    for arm in sorted(by_arm):
        items = by_arm[arm]
        g = lambda f: mean_or_nan([per_item[i].get(f, float("nan")) for i in items])  # noqa: E731
        print(f"| {arm} | {len(items)} | {fmt(g('app_ref'), 4)} | {fmt(g('copy_max'), 3)} | "
              f"{fmt(g('near_copy'), 3)} | {fmt(g('prefix_dino'), 3)} | {fmt(g('max_seam_z'), 2)} |")

    tot = sum(len(v) for v in scored.values())
    print(f"\n*{tot} scored (generation × pool-reference) rows over "
          f"{len(scored)} (generation × seed) pairs, {len(per_item)} registry items.*")


if __name__ == "__main__":
    main()
