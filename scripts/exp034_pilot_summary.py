"""Aggregate exp_034 pilot results (recipe A and B) vs exp_033 baselines for
the same clips (ss0, ss5). Reads run.log + summary.yaml from the latest run
of each pilot config, prints a comparison table, and writes a CSV.

Pre-registered decision rule (Ledger It-4):
  CONFIRMED:   median PSNR across {ss0, ss5} >= exp_033 baseline median + 3 dB
               AND no clip below 18 PSNR. Promote to full batch.
  REJECTED:    both variants regress > 2 dB on either pilot clip. No full batch.
  INCONCLUSIVE: mixed results.
"""

import argparse
import csv
import pathlib
from typing import Dict, List

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP034_DIR = REPO_ROOT / "outputs/videos/exp_034_ltx2_rf_inv_anchor_quality"
EXP033_DIR = REPO_ROOT / "outputs/videos/exp_033_ltx2_rf_inv_drop1/run_0001"

PILOT_CLIPS = ["shadow_smoke_0", "shadow_smoke_5"]


def load_psnr(summary_path: pathlib.Path) -> Dict[str, float]:
    """Return {clip: recon_psnr_mean} from a summary.yaml."""
    if not summary_path.exists():
        return {}
    summary = yaml.safe_load(summary_path.read_text())
    out: Dict[str, float] = {}
    for s in summary.get("samples", []):
        out[s["sample_id"]] = float(s["recon_psnr_mean"])
    return out


def find_run_with_recipe(recipe: str) -> pathlib.Path | None:
    """Find the most recent run_NNNN whose config_snapshot.yaml has recipe == X
    and only contains the pilot samples (ss0, ss5)."""
    if not EXP034_DIR.exists():
        return None
    runs = sorted(EXP034_DIR.glob("run_*/"), reverse=True)
    for run in runs:
        snap = run / "config_snapshot.yaml"
        if not snap.exists():
            continue
        try:
            cfg = yaml.safe_load(snap.read_text())
        except Exception:
            continue
        r = str(cfg.get("inversion", {}).get("recipe", "")).upper()
        sample_ids = {s["sample_id"] for s in cfg.get("samples", [])}
        if r == recipe.upper() and sample_ids == set(PILOT_CLIPS):
            return run
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=pathlib.Path,
                        default=REPO_ROOT / "scripts/exp034_pilot_summary.csv")
    args = parser.parse_args()

    baseline = load_psnr(EXP033_DIR / "summary.yaml")
    print("=" * 72)
    print("exp_033 baseline (reference) — PSNR on pilot clips:")
    for clip in PILOT_CLIPS:
        print(f"  {clip}: {baseline.get(clip, float('nan')):.2f} dB")
    base_med = sorted([baseline[c] for c in PILOT_CLIPS if c in baseline])
    base_median = base_med[len(base_med) // 2] if base_med else float("nan")
    print(f"  median across {PILOT_CLIPS}: {base_median:.2f} dB")

    rows: List[Dict] = []
    for recipe in ("A", "B"):
        run_dir = find_run_with_recipe(recipe)
        print()
        print("=" * 72)
        if run_dir is None:
            print(f"recipe={recipe}: no matching run found")
            continue
        print(f"recipe={recipe} → {run_dir.name}")
        psnr_by_clip = load_psnr(run_dir / "summary.yaml")
        for clip in PILOT_CLIPS:
            p = psnr_by_clip.get(clip, float("nan"))
            b = baseline.get(clip, float("nan"))
            delta = p - b if (b == b and p == p) else float("nan")
            print(f"  {clip}: {p:.2f} dB  (exp_033 baseline {b:.2f}, Δ {delta:+.2f})")
            rows.append({
                "recipe": recipe, "clip": clip, "psnr": p,
                "exp033_psnr": b, "delta_dB": delta,
            })
        vals = [psnr_by_clip[c] for c in PILOT_CLIPS if c in psnr_by_clip]
        if vals:
            med = sorted(vals)[len(vals) // 2]
            med_delta = med - base_median if base_median == base_median else float("nan")
            print(f"  median: {med:.2f} dB  (Δ vs exp_033 {med_delta:+.2f})")

    print()
    print("=" * 72)
    print("DECISION-RULE CHECK")
    print("=" * 72)
    print(f"  exp_033 baseline median across pilot clips: {base_median:.2f} dB")
    print("  CONFIRMED if: variant median >= baseline + 3 dB AND no clip < 18 dB")
    print("  REJECTED if : variant regresses > 2 dB on either pilot clip")
    print("  INCONCLUSIVE: mixed (use 3-clip mini-batch on less-regressed variant)")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["recipe", "clip", "psnr", "exp033_psnr", "delta_dB"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[done] wrote {args.csv}")


if __name__ == "__main__":
    main()
