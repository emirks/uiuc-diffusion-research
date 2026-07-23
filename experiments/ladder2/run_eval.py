"""ladder2 — evaluation. Orchestration only; the metric math lives in the v4 instrument.

    --mode plan     registry rows + generated videos -> v4 eval manifests (pool references,
                    copy-guarded, mask derived from the row's conditioning)
    --mode report   scored rows -> the campaign table, with the %-typing firewall enforced

The yardstick (owner's preferred readout) is the POOL PERCENTAGE: score each generation
against every same-class corpus clip of the GT pool (the DONOR class — the class the arm was
supposed to produce), average, and divide by that class's GT ceiling (the same-class
off-diagonal mean of the v4 distance matrix). The ceiling carries the same class-spread
penalty as the score, so it cancels — which is what makes % comparable ACROSS classes.

The one rule that keeps % honest:

    %_same   endpoint class == donor class -> fair, cross-class comparable, headline
    %_proxy  cross / foreign               -> content-capped: the generation can never look
                                              fully like the donor class because its CONTENT
                                              comes from elsewhere. Absolute level is
                                              ranking-only; the CLAIM is the margin
                                              Delta-pp = %(treat) - %(base) on the same input,
                                              where the identical cap cancels.

`report` therefore prints % everywhere but refuses to aggregate across % types.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import encode_conditioning as ec  # noqa: E402
import prompts  # noqa: E402

STD = REPO_ROOT / "data/processed/transitions_std121"
REGISTRY = HERE / "registry.jsonl"
GENS = REPO_ROOT / "outputs/videos/ladder2"
EVAL_DIR = HERE / "eval"
SCORES = REPO_ROOT / "outputs/eval/ladder2"
NPZ = (REPO_ROOT / ".claude/worktrees/eval-v4-cert/outputs/eval/certification"
       / "4.0.0-draft.1/analysis/distance_matrices.npz")
MATRIX = "m1a_S3"          # the v4 appearance kernel (owner directive 2026-07-20: v4 is the lane)
MAX_POOL_REFS = 8          # deterministic first-N by clip name


def load_registry() -> list[dict]:
    return [json.loads(x) for x in REGISTRY.read_text().splitlines() if x.strip()]


def gen_path(row: dict, seed: int) -> Path:
    return GENS / row["arm"] / f"{row['item_id']}__s{seed}.mp4"


# ------------------------------------------------------------------------------- plan
def pool_refs(row: dict) -> list[str]:
    """Same-class GT pool for this row, copy-guarded (never the reference, never the endpoint)."""
    cls = row["gt_pool_class"]
    clips = sorted(p.stem for p in (STD / cls).glob("*.mp4"))
    banned = {row.get("reference"), row.get("endpoint")}
    return [c for c in clips if c not in banned][:MAX_POOL_REFS]


def plan(seeds: list[int], chunks: int) -> None:
    rows = load_registry()
    by_key = {r["input_key"]: r for r in rows if r["arm"] == "base"}

    items, missing_gen, missing_twin = [], [], []
    for row in rows:
        for seed in seeds:
            path = gen_path(row, seed)
            if not path.exists():
                missing_gen.append(f"{row['item_id']}__s{seed}")
                continue
            # seatbelt 4: the keyed join is EXACT. A treatment row with no base twin is an
            # error, never a silently-dropped or folded row. (This is the defect that flipped
            # two verdicts in the previous ladder.)
            twin = None
            if row["arm"] not in ("base", "text_floor"):
                base = by_key.get(row["input_key"])
                if base is None:
                    missing_twin.append(row["item_id"])
                    continue
                twin = f"{base['item_id']}__s{seed}"

            cond = {}
            if row.get("conditioning") != "none" and row["endpoint"]:
                # seatbelt 5: the mask is a pure function of the row's CONDITIONING,
                # never of the class label.
                paths = ec.cond_paths(row["endpoint"], row["sided"])
                cond["condition_prefix"] = {"video": str(paths["prefix"].relative_to(REPO_ROOT)),
                                            "num_frames": ec.PX_PREFIX}
                if row["sided"] == "two":
                    cond["condition_suffix"] = {"video": str(paths["suffix"].relative_to(REPO_ROOT)),
                                                "num_frames": ec.SUFFIX_GEN_FRAMES}
            for ref_clip in pool_refs(row):
                items.append({
                    "item_id": f"{row['item_id']}__s{seed}__ref_{ref_clip}",
                    "generated_video": str(path),
                    "reference_video": f"data/processed/transitions_std121/"
                                       f"{prompts.clip_class(ref_clip)}/{ref_clip}.mp4",
                    "style": row["gt_pool_class"],
                    "n_endpoints": 2 if row["sided"] == "two" else 1,
                    "arm": row["arm"],
                    "twin_of": twin,
                    "notes": f"ladder2 {row['cell']} pct={row['pct_type']} seed={seed}",
                    **cond,
                })

    if missing_twin:
        sys.exit(f"[plan] KEYED-JOIN FAILURE: {len(missing_twin)} rows without a base twin: "
                 f"{missing_twin[:5]}")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(chunks):
        part = items[i::chunks]
        (EVAL_DIR / f"eval_c{i}.json").write_text(json.dumps(part, indent=1))
    print(f"[plan] {len(items)} (generation x pool-reference) rows -> {chunks} chunks "
          f"in {EVAL_DIR.relative_to(REPO_ROOT)}")
    if missing_gen:
        print(f"[plan] {len(missing_gen)} generations not yet rendered (skipped): "
              f"{missing_gen[:4]}{' ...' if len(missing_gen) > 4 else ''}")


# ----------------------------------------------------------------------------- report
def ceilings() -> dict[str, float]:
    import numpy as np

    z = np.load(NPZ, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    sim = 1.0 - z[MATRIX]
    by_cls: dict[str, list[int]] = collections.defaultdict(list)
    for i, k in enumerate(keys):
        by_cls[k.split("/")[0]].append(i)
    out = {}
    for cls, idx in by_cls.items():
        if len(idx) < 2:
            continue
        block = sim[np.ix_(idx, idx)]
        out[cls] = float(block[~np.eye(len(idx), dtype=bool)].mean())
    return out


def report() -> None:
    rows = {r["item_id"]: r for r in load_registry()}
    ceil = ceilings()

    # (item_id, seed) -> pool mean over references
    pool: dict[tuple[str, int], list[float]] = collections.defaultdict(list)
    for f in sorted(SCORES.glob("*/items.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("app_ref") is None:
                continue
            head, _, _ref = r["item_id"].rpartition("__ref_")
            base_id, _, seed = head.rpartition("__s")
            pool[(base_id, int(seed))].append(r["app_ref"])
    if not pool:
        sys.exit(f"[report] no scored rows under {SCORES} — run the scorer first")

    # item -> pct of its GT ceiling (mean over seeds of the per-seed pool mean)
    pct: dict[str, float] = {}
    for (item_id, _seed), vals in pool.items():
        row = rows.get(item_id)
        if row is None or row["gt_pool_class"] not in ceil:
            continue
        pct.setdefault(item_id, [])
        pct[item_id].append(st.mean(vals) / ceil[row["gt_pool_class"]])
    pct = {k: st.mean(v) for k, v in pct.items()}

    base_by_key = {r["input_key"]: r["item_id"] for r in rows.values() if r["arm"] == "base"}

    cells: dict[str, list[tuple[str, float, float | None]]] = collections.defaultdict(list)
    unpaired: dict[str, int] = collections.defaultdict(int)
    for item_id, value in pct.items():
        row = rows[item_id]
        if row["arm"] == "base":
            continue
        twin = base_by_key.get(row["input_key"])
        if twin is None:
            sys.exit(f"[report] KEYED-JOIN FAILURE: {item_id} has no base twin in the registry")
        base_value = pct.get(twin)
        if row["arm"] != "text_floor" and base_value is None:
            # the twin exists in the registry but has not been scored yet — say so out loud
            # rather than letting the row quietly vanish from the margin statistics
            unpaired[row["cell"]] += 1
        cells[row["cell"]].append((row["donor_class"], value,
                                   (value - base_value) if base_value is not None else None))

    print(f"\n{'cell':16s} {'n':>3s} {'%type':>6s} {'level %':>9s} {'Δpp vs base':>12s} "
          f"{'donors +':>9s}  verdict")
    print("-" * 78)
    for cell in sorted(cells):
        vals = cells[cell]
        ptype = {rows[i]["pct_type"] for i in pct if rows[i]["cell"] == cell}
        ptype = ptype.pop() if len(ptype) == 1 else "MIXED"
        assert ptype != "MIXED", f"{cell}: mixed % types in one cell — firewall violation"
        level = st.mean(v for _c, v, _d in vals)
        deltas = [d for _c, _v, d in vals if d is not None]
        # unit of analysis is the CLASS: per-donor mean of the margin, then a sign test
        per_donor: dict[str, list[float]] = collections.defaultdict(list)
        for cls, _v, d in vals:
            if d is not None:
                per_donor[cls].append(d)
        pos = sum(1 for v in per_donor.values() if st.mean(v) > 0)
        n_d = len(per_donor)
        dpp = st.mean(deltas) * 100 if deltas else float("nan")
        headline = f"{level:8.1%}" if ptype == "same" else f"({level:7.1%})"
        verdict = "—" if not n_d else ("positive" if pos / n_d >= 0.8 else "weak")
        flag = f"  [{unpaired[cell]} unpaired]" if unpaired.get(cell) else ""
        print(f"{cell:16s} {len(vals):3d} {ptype:>6s} {headline:>9s} {dpp:>+11.1f}  "
              f"{pos:>4d}/{n_d:<4d}  {verdict}{flag}")
    print("\n%_same = fair, cross-class comparable, headline-eligible.")
    print("(%_proxy) = content-capped level, ranking only — the CLAIM for those cells is Δpp.")
    if unpaired:
        print(f"[report] {sum(unpaired.values())} rows had a registered base twin that is not "
              f"scored yet — they contribute a level but no Δpp. Not a fold, but not complete.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plan", "report"], required=True)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--chunks", type=int, default=8)
    args = ap.parse_args()
    if args.mode == "plan":
        plan([int(s) for s in args.seeds.split(",")], args.chunks)
    else:
        report()


if __name__ == "__main__":
    main()
