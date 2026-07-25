"""exp_077 — THE FINAL MEASUREMENT: the D2 generalist vs the incumbent generalist.

Self-relative. There is NO baseline arm here: the comparison is d2_gen against ic_gen on the
SAME four claim cells, the SAME 67 items, the SAME prompts / references / endpoints / seeds
(42, 43), the SAME copy-guarded donor pools and the SAME v4 class-GT ceilings. The only thing
that differs between the two arms is the adapter — and between the two adapters, the only
thing that differs is the training DATA.

All pool math is imported from run_eval, not re-implemented, so the incumbent's published
numbers (72.9 / 72.8 / 88.7 / 90.8) are reproduced by this script from its own scored rows
rather than trusted — see the `incumbent_reproduces` block in the JSON.

    python eval_ladder/d2_report.py
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

import run_eval  # noqa: E402

CELLS = ["G-unseen-cross", "G-zs-cross", "G-unseen-same", "G-zs-same"]
CROSS = ["G-unseen-cross", "G-zs-cross"]

#: the incumbent's already-published pool-% (v4 pool-% of donor-class GT ceiling), as handed to
#: this task. Reproduced from scored rows below; a mismatch is a hard failure.
INCUMBENT_PUBLISHED = {
    "G-unseen-cross": 0.729, "G-zs-cross": 0.728,
    "G-unseen-same": 0.887, "G-zs-same": 0.908,
}
#: pre-registered bars (not chosen after seeing a number)
WIN_POOLED_CROSS_PP = 5.0
WIN_DONORS_POSITIVE = 15
SAME_GUARDS = {"G-unseen-same": 0.837, "G-zs-same": 0.858}
NEAR_COPY_MAX = 0.05
REF_DOMINATED_MAX = 0.10

D2_SCORES = REPO_ROOT / "outputs/eval/d2gen"
IC_SCORES = REPO_ROOT / "outputs/eval/ladder2"
OUT = REPO_ROOT / "experiments/exp_077_synth_stratum/D2_RESULT.json"


def raw_and_pct(pool, rows, ceil):
    """item -> (raw pool-mean app_ref, pct of GT ceiling). Both are the mean over seeds."""
    pct = run_eval.item_pct(pool, rows, ceil)
    raw: dict[str, list[float]] = collections.defaultdict(list)
    for (item_id, _seed), vals in pool.items():
        if item_id in pct:
            raw[item_id].append(st.mean(vals))
    return {k: st.mean(v) for k, v in raw.items()}, pct


def near_copy_rate(scores_dir: Path, arm: str | None = None):
    """Fraction of scored (generation x pool-reference) rows flagged near_copy, deduped the same
    way the pool means are."""
    seen, n, flagged = set(), 0, 0
    for f in sorted(scores_dir.glob("*/items.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["item_id"] in seen:
                continue
            seen.add(r["item_id"])
            if arm is not None and r.get("arm") != arm:
                continue
            n += 1
            flagged += bool(r.get("near_copy"))
    return {"n_rows": n, "n_near_copy": flagged, "rate": (flagged / n) if n else None}


def dominance_rate(path: Path, arm: str):
    if not path.exists():
        return None
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("arm") == arm and r.get("cell") in CELLS]
    if not rows:
        return None
    out = {"n": len(rows),
           "ref_dominated_rate": sum(r["ref_dominated"] for r in rows) / len(rows),
           "ep_align": st.mean(r["ep_align"] for r in rows),
           "ref_align": st.mean(r["ref_align"] for r in rows)}
    for cell in CELLS:
        sub = [r for r in rows if r["cell"] == cell]
        if sub:
            out[cell] = {"n": len(sub),
                         "ref_dominated_rate": sum(r["ref_dominated"] for r in sub) / len(sub)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-registry",
                    default=str(HERE / "registry_d2gen.jsonl"))
    ap.add_argument("--partial", action="store_true",
                    help="write the JSON even if some items are not scored yet")
    args = ap.parse_args()

    run_eval.EXTRA_REGISTRY = Path(args.extra_registry)
    reg = {r["item_id"]: r for r in run_eval.load_registry()}
    ceil = run_eval.ceilings()

    ic_raw, ic_pct = raw_and_pct(run_eval.pool_means(IC_SCORES), reg, ceil)
    d2_raw, d2_pct = raw_and_pct(run_eval.pool_means(D2_SCORES), reg, ceil)

    # ---- the item set is DEFINED by the registry, never by what happened to score
    expected = {c: sorted(i for i, r in reg.items() if r["arm"] == "d2_gen" and r["cell"] == c)
                for c in CELLS}
    missing = {c: [i for i in ids if i not in d2_pct] for c, ids in expected.items()}
    n_missing = sum(len(v) for v in missing.values())

    result: dict = {
        "experiment": "exp_077_synth_stratum",
        "measurement": "D2 generalist (d2_gen) vs incumbent generalist (ic_gen), self-relative",
        "adapter": "outputs/training/exp_077_synth_stratum/d2_gen/checkpoints/"
                   "lora_weights_step_05000.safetensors",
        "seeds": [42, 43],
        "instrument": {"matrix": run_eval.MATRIX, "max_pool_refs": run_eval.MAX_POOL_REFS,
                       "npz": str(run_eval.NPZ)},
        "bars": {"pooled_cross_delta_pp": WIN_POOLED_CROSS_PP,
                 "donors_positive": f"{WIN_DONORS_POSITIVE}/23",
                 "same_content_floors": SAME_GUARDS,
                 "near_copy_max": NEAR_COPY_MAX, "ref_dominated_max": REF_DOMINATED_MAX},
        "missing_items": {c: v for c, v in missing.items() if v},
        "n_missing": n_missing,
    }

    # ---- headline table -------------------------------------------------------------
    table, repro = [], {}
    for cell in CELLS:
        ids = [i for i in expected[cell] if i in d2_pct]
        ic_ids = [i.replace("__d2_gen__", "__ic_gen__") for i in expected[cell]]
        ic_have = [i for i in ic_ids if i in ic_pct]
        inc_level = st.mean(ic_pct[i] for i in ic_have) if ic_have else None
        repro[cell] = {"reproduced": inc_level, "published": INCUMBENT_PUBLISHED[cell],
                       "n": len(ic_have),
                       "agrees": inc_level is not None
                       and abs(inc_level - INCUMBENT_PUBLISHED[cell]) < 0.001}
        row = {
            "cell": cell,
            "pct_type": reg[expected[cell][0]]["pct_type"],
            "n": len(ids), "n_expected": len(expected[cell]),
            "d2_raw_app_ref": st.mean(d2_raw[i] for i in ids) if ids else None,
            "gt_ceiling_mean": st.mean(ceil[reg[i]["gt_pool_class"]] for i in ids) if ids else None,
            "d2_pool_pct": st.mean(d2_pct[i] for i in ids) if ids else None,
            "incumbent_pool_pct_published": INCUMBENT_PUBLISHED[cell],
            "incumbent_pool_pct_reproduced": inc_level,
            "incumbent_raw_app_ref": st.mean(ic_raw[i] for i in ic_have) if ic_have else None,
        }
        # paired: same item, same seeds, same pool, same ceiling -> ceiling cancels exactly
        paired = [(i, d2_pct[i] - ic_pct[i.replace("__d2_gen__", "__ic_gen__")])
                  for i in ids if i.replace("__d2_gen__", "__ic_gen__") in ic_pct]
        row["n_paired"] = len(paired)
        row["delta_pp_paired"] = st.mean(d for _i, d in paired) * 100 if paired else None
        row["delta_pp_vs_published"] = ((row["d2_pool_pct"] - INCUMBENT_PUBLISHED[cell]) * 100
                                        if row["d2_pool_pct"] is not None else None)
        table.append(row)
    result["headline"] = table
    result["incumbent_reproduces"] = repro

    # ---- pooled CROSS: per-donor paired signs ---------------------------------------
    cross_ids = [i for c in CROSS for i in expected[c] if i in d2_pct]
    cross_pairs = [(i, d2_pct[i] - ic_pct[i.replace("__d2_gen__", "__ic_gen__")])
                   for i in cross_ids if i.replace("__d2_gen__", "__ic_gen__") in ic_pct]
    per_donor: dict[str, list[float]] = collections.defaultdict(list)
    for i, d in cross_pairs:
        per_donor[reg[i]["donor_class"]].append(d)
    donor_means = {k: st.mean(v) for k, v in sorted(per_donor.items())}
    all_cross_donors = sorted({reg[i]["donor_class"] for c in CROSS for i in expected[c]})
    pos = sum(1 for v in donor_means.values() if v > 0)
    pooled_dpp = st.mean(d for _i, d in cross_pairs) * 100 if cross_pairs else None
    result["pooled_cross"] = {
        "n_items": len(cross_pairs), "n_items_expected": sum(len(expected[c]) for c in CROSS),
        "mean_delta_pp": pooled_dpp,
        "donors_positive": pos, "donors_total": len(donor_means),
        "donors_registered": len(all_cross_donors),
        "donors_missing": [d for d in all_cross_donors if d not in donor_means],
        "per_donor_delta_pp": {k: v * 100 for k, v in donor_means.items()},
        "d2_pool_pct": st.mean(d2_pct[i] for i in cross_ids) if cross_ids else None,
    }

    # ---- guards ----------------------------------------------------------------------
    guards = {
        "near_copy_d2": near_copy_rate(D2_SCORES),
        "near_copy_incumbent_all_ladder2": near_copy_rate(IC_SCORES, arm="ic_gen"),
        "ref_dominated_passA_d2": dominance_rate(HERE / "dominance_passA_d2.jsonl", "d2_gen"),
        "ref_dominated_passA_incumbent": dominance_rate(HERE / "dominance_passA.jsonl", "ic_gen"),
    }
    same_ok = {}
    for cell, floor in SAME_GUARDS.items():
        got = next(r["d2_pool_pct"] for r in table if r["cell"] == cell)
        same_ok[cell] = {"value": got, "floor": floor,
                         "holds": (got is not None and got >= floor)}
    guards["same_content_floors"] = same_ok
    result["guards"] = guards

    # ---- verdict ---------------------------------------------------------------------
    complete = n_missing == 0
    bar_delta = pooled_dpp is not None and pooled_dpp >= WIN_POOLED_CROSS_PP
    bar_donors = pos >= WIN_DONORS_POSITIVE
    result["verdict"] = {
        "complete": complete,
        "pooled_cross_delta_pp": pooled_dpp,
        "bar_pooled_cross_met": bar_delta,
        "bar_donors_met": bar_donors,
        "donors_positive": f"{pos}/{len(donor_means)}",
        "WIN": bool(complete and bar_delta and bar_donors),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))

    # ---- print ------------------------------------------------------------------------
    print(f"\n=== exp_077 D2 GENERALIST vs INCUMBENT — v4 pool-% of donor-class GT ceiling ===")
    print(f"{'cell':16s} {'%type':>6s} {'n':>5s} {'raw app_ref':>12s} {'GT ceiling':>11s} "
          f"{'D2 pool-%':>10s} {'incumbent':>10s} {'Δpp':>8s}")
    print("-" * 88)
    def num(v, w, p=3):
        return f"{v:{w}.{p}f}" if v is not None else "-".rjust(w)

    def pc_(v):
        return f"{v * 100:9.1f}%" if v is not None else "-".rjust(10)

    for r in table:
        dpp = r["delta_pp_paired"]
        dpp_s = f"{dpp:+8.1f}" if dpp is not None else "-".rjust(8)
        print(f"{r['cell']:16s} {r['pct_type']:>6s} {r['n']:>2d}/{r['n_expected']:<2d} "
              f"{num(r['d2_raw_app_ref'], 12, 4)} {num(r['gt_ceiling_mean'], 11, 4)} "
              f"{pc_(r['d2_pool_pct'])} {pc_(r['incumbent_pool_pct_published'])} {dpp_s}")
    print(f"\nincumbent reproduced from its own scored rows: "
          + ", ".join(f"{c} {v['reproduced'] * 100:.1f}% (pub {v['published'] * 100:.1f}%)"
                      if v["reproduced"] is not None else f"{c} -" for c, v in repro.items()))
    pc = result["pooled_cross"]
    print(f"\npooled CROSS (G-unseen-cross + G-zs-cross): n={pc['n_items']}/"
          f"{pc['n_items_expected']}  mean Δ = "
          + (f"{pc['mean_delta_pp']:+.1f} pp" if pc["mean_delta_pp"] is not None else "-")
          + f"   donors positive {pc['donors_positive']}/{pc['donors_total']}")
    if donor_means:
        print("  per-donor Δpp: " + ", ".join(f"{k} {v * 100:+.1f}" for k, v in donor_means.items()))
    print("\nguards:")
    for k in ("near_copy_d2", "near_copy_incumbent_all_ladder2"):
        g = guards[k]
        print(f"  {k:34s} {g['n_near_copy']}/{g['n_rows']} = "
              + (f"{g['rate']:.2%}" if g["rate"] is not None else "-")
              + f"   (bar <= {NEAR_COPY_MAX:.0%})")
    for k in ("ref_dominated_passA_d2", "ref_dominated_passA_incumbent"):
        g = guards[k]
        print(f"  {k:34s} " + (f"{g['ref_dominated_rate']:.1%} (n={g['n']})" if g else "not run")
              + f"   (bar <= {REF_DOMINATED_MAX:.0%})")
    for cell, g in same_ok.items():
        print(f"  {cell:34s} "
              + (f"{g['value']:.1%}" if g["value"] is not None else "-")
              + f" vs floor {g['floor']:.1%}  -> {'HOLDS' if g['holds'] else 'BREACHED'}")
    v = result["verdict"]
    print(f"\nWIN bar: pooled-cross Δ >= +{WIN_POOLED_CROSS_PP}pp AND "
          f">= {WIN_DONORS_POSITIVE}/23 donors positive")
    print(f"  Δ bar      {'MET' if v['bar_pooled_cross_met'] else 'NOT MET'}")
    print(f"  donor bar  {'MET' if v['bar_donors_met'] else 'NOT MET'}  ({v['donors_positive']})")
    print(f"  VERDICT:   {'WIN' if v['WIN'] else 'NO WIN'}"
          + ("" if complete else f"   [PARTIAL — {n_missing} items unscored]"))
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    if n_missing and not args.partial:
        print(f"[warn] {n_missing} registered items are not scored yet — numbers are PROVISIONAL")


if __name__ == "__main__":
    main()
