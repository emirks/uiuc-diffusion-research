"""Trainer-equivalence verdict, against the advisor's pre-registered rule.

The bottleneck campaign does its surgery in a private trainer worktree, so B1-vs-ic_gen would
cross trainer versions. This scores whether that crossing is safe.

RULE (pre-registered by the advisor BEFORE the control ran — do not edit to fit the numbers):

    R_step = mean|diff|(mine, lineage)      / mean|diff|(same-code floor pair)
    R_mean = |delta run-mean|(mine, lineage)/ |delta run-mean|(same-code floor pair)
    PASS  <=>  BOTH <= 3

    - The sign split is DESCRIPTIVE ONLY, never gating: once two trajectories separate, the
      per-step diffs are a random walk on diverged weights, and long one-sided excursions are the
      arcsine law rather than evidence of bias. A paired sign test here would be invalid.
    - max|diff| is descriptive only (dominated by single-step tail events).
    - Both statistics are required: per-step drift passing while run-mean fails is the signature of
      a one-sided systematic component, and must trigger the same-binary fallback.
    - CONTROL VALIDITY GATE: the floor run's step-0 loss must be bitwise identical to its partner's.
      If it is not, the run landed in a different kernel environment and the floor is invalid.
    - Degenerate floors (identical runs, or mean|diff| < 5e-4) fall out as FAIL automatically: a
      deterministic trainer means the divergence is attributable to the diff.
    - With two floors available, the denominator is the MAX, guarding against one unluckily-quiet
      floor sample.

    python experiments/exp_078_operator_token_bottleneck/equivalence_report.py
"""

import glob
import re
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS = REPO_ROOT / "outputs" / "logs" / "slurm"

PASS_RATIO = 3.0
DEGENERATE_FLOOR = 5e-4


def loss_sequence(job_glob: str) -> list[float]:
    """Per-step losses from a trainer log, collapsing the progress bar's duplicate re-renders."""
    files = sorted(LOGS.glob(job_glob))
    if not files:
        return []
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", files[-1].read_text(errors="ignore"))
    out: list[float] = []
    for v in (float(m) for m in re.findall(r"Loss: ([0-9.]+)", text)):
        if not out or v != out[-1]:
            out.append(v)
    return out


def compare(a: list[float], b: list[float]) -> dict:
    n = min(len(a), len(b))
    d = [a[i] - b[i] for i in range(n)]
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    onset = next((i for i, x in enumerate(d) if x != 0), None)
    return {
        "n": n,
        "step0_equal": bool(n and a[0] == b[0]),
        "mean_abs": st.mean(map(abs, d)) if n else float("nan"),
        "max_abs": max(map(abs, d)) if n else float("nan"),
        "delta_run_mean": abs(st.mean(a[:n]) - st.mean(b[:n])) if n else float("nan"),
        "sign": f"{pos}/{neg}/{n - pos - neg}",
        "onset": onset,
    }


def main() -> None:
    runs = {
        "mine": loss_sequence("bneck_equiv_mine-*.out"),
        "lineage": loss_sequence("bneck_equiv_lineage-*.out"),
        "lineage2": loss_sequence("bneck_equiv_lineage2-*.out"),
        "mine2": loss_sequence("bneck_equiv_mine2-*.out"),
    }
    print("runs found: " + ", ".join(f"{k}={len(v)}" for k, v in runs.items()))
    if not (runs["mine"] and runs["lineage"]):
        raise SystemExit("[fatal] the cross-trainer pair is missing")

    cross = compare(runs["mine"], runs["lineage"])
    floors = {}
    if runs["lineage2"]:
        floors["lineage-vs-lineage2"] = compare(runs["lineage"], runs["lineage2"])
    if runs["mine2"]:
        floors["mine-vs-mine2"] = compare(runs["mine"], runs["mine2"])

    hdr = f"{'pair':24s} {'n':>4s} {'step0=':>7s} {'mean|d|':>9s} {'max|d|':>9s} {'Drunmean':>9s} {'sign +/-/=':>12s} {'onset':>6s}"
    print("\n" + hdr)
    print("-" * len(hdr))

    def row(name: str, c: dict) -> None:
        print(f"{name:24s} {c['n']:4d} {str(c['step0_equal']):>7s} {c['mean_abs']:9.6f} "
              f"{c['max_abs']:9.6f} {c['delta_run_mean']:9.6f} {c['sign']:>12s} {str(c['onset']):>6s}")

    row("CROSS mine-vs-lineage", cross)
    for name, c in floors.items():
        row(f"FLOOR {name}", c)

    if not floors:
        raise SystemExit("\n[pending] no same-code floor has completed yet — no verdict possible")

    # Validity gate, then the max-of-floors denominator.
    valid = {k: c for k, c in floors.items() if c["step0_equal"]}
    for k, c in floors.items():
        if not c["step0_equal"]:
            print(f"\n[warn] floor '{k}' FAILS the validity gate (step-0 losses differ) — excluded; "
                  "it landed in a different kernel environment")
    if not valid:
        raise SystemExit("[fatal] no valid floor — rerun a floor pinned to the same GPU model")

    den_step = max(c["mean_abs"] for c in valid.values())
    den_mean = max(c["delta_run_mean"] for c in valid.values())

    print(f"\ndenominator = max over {len(valid)} valid floor(s): "
          f"mean|d|={den_step:.6f}  Drunmean={den_mean:.6f}")

    if den_step < DEGENERATE_FLOOR:
        print(f"\nVERDICT: FAIL — floor is degenerate (mean|d| {den_step:.2e} < {DEGENERATE_FLOOR:.0e}); "
              "the trainer is effectively deterministic, so the divergence is attributable to the diff.")
        print("  -> pre-registered remedy: retrain the ic_gen comparator on the private trainer so "
              "every claim-bearing comparison shares one binary. B1 is NOT blocked.")
        return

    r_step = cross["mean_abs"] / den_step
    r_mean = cross["delta_run_mean"] / den_mean if den_mean > 0 else float("inf")
    ok_step, ok_mean = r_step <= PASS_RATIO, r_mean <= PASS_RATIO

    print(f"\n  R_step = {cross['mean_abs']:.6f} / {den_step:.6f} = {r_step:.2f}   "
          f"{'PASS' if ok_step else 'FAIL'} (bar <= {PASS_RATIO})")
    print(f"  R_mean = {cross['delta_run_mean']:.6f} / {den_mean:.6f} = {r_mean:.2f}   "
          f"{'PASS' if ok_mean else 'FAIL'} (bar <= {PASS_RATIO})")
    print(f"\nVERDICT: {'PASS — the diff is a no-op for the full-ref path' if ok_step and ok_mean else 'FAIL — same-binary comparator required'}")
    if not (ok_step and ok_mean):
        print("  -> pre-registered remedy: retrain the ic_gen comparator on the private trainer "
              "(~3.9 h). B1 is NOT blocked either way.")


if __name__ == "__main__":
    main()
