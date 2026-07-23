"""ladder2 — ic_gen convergence diagnostic (advisor amendment, 2026-07-23).

Replaces the original inline "is the OOD sample still improving between step 4500 and 5000?"
eyeball check, which was dropped because its only advantage was earliness — and earliness is
worth nothing here: every checkpoint persists, and a follow-up run would launch after step 5000
either way. This version answers the same question quantitatively, on claim-bearing eval items,
with the v4 instrument.

THE QUESTION: 5000 steps was validated only under the OLD LEAKY prompts. ladder2's leak-free
regime is plausibly harder. If ic_gen scores below the specialists, that is only attributable to
"generalist vs specialist" if the generalist actually CONVERGED — otherwise it is confounded with
under-training. This buys that attribution for one extra generation pass.

STATUS: DIAGNOSTIC, never a claim. Its output never enters the report's claim tables. Its only
power is to authorise a separate, clearly-labelled non-pre-registered follow-up run.

PRE-DECLARED BAR (fixed before any ic_gen score was seen):
  UNDERSHOT  Δ >= +2.0 pp on either cell AND >= 2/3 of that cell's items improve
             -> follow-up run authorised; flag ic_gen-vs-specialist as budget-confounded
  CONVERGED  |Δ| < 2.0 pp on both cells, or improvement is non-systematic  -> 5000 stands
  AMBIGUOUS  Δ in [1.0, 2.0) pp with a systematic sign -> also generate from ckpt-4000 and read
             the 4000->4500->5000 slope; monotone rise with total gain >= 2 pp = UNDERSHOT

    # 1. generate the two cells from ckpt-4500 (writes to arm dir ic_gen__ck4500)
    python experiments/ladder2/convergence_diagnostic.py --mode gen-cmd
    # 2. plan + score those generations
    python experiments/ladder2/convergence_diagnostic.py --mode plan
    # 3. verdict
    python experiments/ladder2/convergence_diagnostic.py --mode report
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

import run_eval  # noqa: E402  (reuse pool_refs, ceilings, the same v4 conventions)

CELLS = ("G-unseen-same", "G-zs-cross")
BASE_STEP, PROBE_STEP = 5000, 4500
PROBE_ARM = f"ic_gen__ck{PROBE_STEP}"
EVAL_DIR = HERE / "eval_diag"
SCORES = REPO_ROOT / "outputs/eval/ladder2_diag"
GENS = REPO_ROOT / "outputs/videos/ladder2"
#: pre-declared bar — do not edit after the first score is read
DELTA_PP = 2.0
FRACTION = 2 / 3


def rows_of_interest() -> list[dict]:
    return [r for r in run_eval.load_registry() if r["cell"] in CELLS and r["arm"] == "ic_gen"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen-cmd", "plan", "report"], required=True)
    ap.add_argument("--seeds", default="42,43")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = rows_of_interest()

    if args.mode == "gen-cmd":
        n = len(rows) * len(seeds)
        print(f"# {len(rows)} items x {len(seeds)} seeds = {n} generations "
              f"({'both seeds' if n <= 120 else 'SUBSAMPLE to one seed'})")
        for seed in seeds:
            print(f"sbatch --job-name=ladder2_diag_s{seed} --array=0-1 --partition=secondary "
                  f"--account=campusclusterusers --gres=gpu:H100:1 --time=02:00:00 --nice=300 "
                  f"--export=ALL,ARM=ic_gen,SEED={seed},NCHUNKS=2,"
                  f"CELLS={'|'.join(CELLS)},STEP={PROBE_STEP} "
                  f"experiments/ladder2/job_gen.sbatch")
        return

    if args.mode == "plan":
        items, missing = [], 0
        for row in rows:
            for seed in seeds:
                p = GENS / PROBE_ARM / f"{row['item_id']}__s{seed}.mp4"
                if not p.exists():
                    missing += 1
                    continue
                for ref_clip in run_eval.pool_refs(row):
                    items.append({
                        "item_id": f"DIAG__{row['item_id']}__s{seed}__ref_{ref_clip}",
                        "generated_video": str(p),
                        "reference_video": f"data/processed/transitions_std121/"
                                           f"{run_eval.prompts.clip_class(ref_clip)}/{ref_clip}.mp4",
                        "style": row["gt_pool_class"],
                        "n_endpoints": 2 if row["sided"] == "two" else 1,
                        "arm": PROBE_ARM,
                        "notes": f"convergence diagnostic ckpt{PROBE_STEP} {row['cell']}",
                    })
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        for old in EVAL_DIR.glob("eval_c*.json"):
            old.unlink()
        for i in range(2):
            (EVAL_DIR / f"eval_c{i}.json").write_text(json.dumps(items[i::2], indent=1))
        print(f"[diag] {len(items)} rows -> 2 chunks in {EVAL_DIR.relative_to(REPO_ROOT)}"
              f"  ({missing} ckpt-{PROBE_STEP} generations still missing)")
        return

    # ---- report: compare ckpt-4500 vs the pinned ckpt-5000 scores, item by item
    ceil = run_eval.ceilings()
    registry = {r["item_id"]: r for r in run_eval.load_registry()}

    def pct_by_item(score_dirs, strip_diag: bool) -> dict[str, float]:
        pool: dict[str, list[float]] = collections.defaultdict(list)
        for d in score_dirs:
            for f in d.glob("*/items.jsonl"):
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("app_ref") is None:
                        continue
                    ident = r["item_id"]
                    if strip_diag != ident.startswith("DIAG__"):
                        continue
                    ident = ident.removeprefix("DIAG__")
                    head, _, _ref = ident.rpartition("__ref_")
                    item, _, _seed = head.rpartition("__s")
                    if item in registry and registry[item]["cell"] in CELLS:
                        pool[item].append(r["app_ref"])
        return {k: st.mean(v) / ceil[registry[k]["gt_pool_class"]]
                for k, v in pool.items() if registry[k]["gt_pool_class"] in ceil}

    at_5000 = pct_by_item([run_eval.SCORES], strip_diag=False)
    at_4500 = pct_by_item([SCORES], strip_diag=True)
    common = sorted(set(at_5000) & set(at_4500))
    if not common:
        sys.exit("[diag] no items scored at BOTH checkpoints yet")

    print(f"\n{'cell':16s} {'n':>3s} {'ckpt4500':>9s} {'ckpt5000':>9s} {'Δpp':>7s} {'improved':>10s}")
    print("-" * 62)
    verdicts = []
    for cell in CELLS:
        ids = [i for i in common if registry[i]["cell"] == cell]
        if not ids:
            continue
        a = st.mean(at_4500[i] for i in ids) * 100
        b = st.mean(at_5000[i] for i in ids) * 100
        improved = sum(1 for i in ids if at_5000[i] > at_4500[i])
        frac = improved / len(ids)
        print(f"{cell:16s} {len(ids):3d} {a:8.1f}% {b:8.1f}% {b - a:+7.1f} {improved:5d}/{len(ids):<4d}")
        verdicts.append((cell, b - a, frac))

    undershot = [c for c, d, fr in verdicts if d >= DELTA_PP and fr >= FRACTION]
    ambiguous = [c for c, d, fr in verdicts if 1.0 <= d < DELTA_PP and fr >= FRACTION]
    print()
    if undershot:
        print(f"VERDICT: **UNDERSHOT** on {undershot} — Δ>={DELTA_PP}pp with >={FRACTION:.0%} of items "
              f"improving.\n  => a separate, clearly-labelled NON-pre-registered follow-up run is "
              f"authorised;\n  => ic_gen-vs-specialist comparisons must be flagged budget-confounded.")
    elif ambiguous:
        print(f"VERDICT: **AMBIGUOUS** on {ambiguous} — generate the same items from ckpt-4000 and "
              f"read the\n  4000->4500->5000 slope; monotone rise with total gain >=2pp = UNDERSHOT.")
    else:
        print("VERDICT: **CONVERGED** — 5000 steps stands; no follow-up run. ic_gen-vs-specialist "
              "comparisons\n  are NOT budget-confounded.")
    print("\n(diagnostic only — never enters the report's claim tables)")


if __name__ == "__main__":
    main()
