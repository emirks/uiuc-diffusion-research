"""Floor/ceiling normalization and the harness's own exam (retrieval eval).

Raw metric values are uninterpretable ("is 0.71 good?"); every reported score
is anchored between a floor control (lerp crossfade / base model) and a
ceiling (real same-style clips, leave-one-out) run through the identical
pipeline. No single composite number is produced — collapsing axes hides
exactly the trade-offs the work is about.
"""

from __future__ import annotations

import numpy as np


def normalize_score(raw: float, floor: float, ceiling: float, higher_better: bool = True) -> float:
    """(raw - floor) / (ceiling - floor), clipped to [0, 1]; orientation-aware."""
    if not higher_better:
        raw, floor, ceiling = -raw, -floor, -ceiling
    denom = ceiling - floor
    if abs(denom) < 1e-9:
        return float("nan")
    return float(np.clip((raw - floor) / denom, 0.0, 1.0))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion — the honest error
    bar for exam accuracies at n=41 and judge pass rates at n=6."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else float(max(0.0, center - half))
    hi = 1.0 if k == n else float(min(1.0, center + half))  # fp-exact at bounds
    return (lo, hi)


def retrieval_eval(dist: np.ndarray, labels: list[str]) -> dict:
    """Leave-one-out 1-NN style retrieval on a symmetric distance matrix —
    the style-discrimination exam. Also reports within- vs cross-style
    separation (Cohen's d) since accuracy alone saturates."""
    n = len(labels)
    D = dist.copy().astype(float)
    np.fill_diagonal(D, np.inf)
    D[np.isnan(D)] = np.inf
    row_valid = np.isfinite(D).any(axis=1)  # untrackable items can't be retrieved
    pred = [labels[int(np.argmin(D[i]))] if row_valid[i] else None for i in range(n)]
    correct = np.array([p == l for p, l, v in zip(pred, labels, row_valid) if v])
    wilson = wilson_interval(int(correct.sum()), len(correct))

    classes = sorted(set(labels))
    lab = np.array(labels)
    per_class = {c: float(np.mean([p == l for p, l, v in zip(pred, labels, row_valid)
                                   if v and l == c] or [np.nan])) for c in classes}
    confusion = {c: {c2: int(sum(1 for p, l in zip(pred, labels) if l == c and p == c2))
                     for c2 in classes} for c in classes}

    off = ~np.eye(n, dtype=bool) & np.isfinite(dist) & ~np.isnan(dist)
    same = (lab[:, None] == lab[None, :]) & off
    within, cross = dist[same], dist[off & ~same]
    pooled = np.sqrt(0.5 * (within.std() ** 2 + cross.std() ** 2)) + 1e-12
    return {
        "accuracy_1nn": float(correct.mean()),
        "accuracy_wilson95": wilson,
        "coverage": float(row_valid.mean()),
        "chance": float(max(np.bincount([classes.index(l) for l in labels])) / n),
        "per_class_recall": per_class,
        "confusion": confusion,
        "within_mean": float(within.mean()), "cross_mean": float(cross.mean()),
        "separation_cohens_d": float((cross.mean() - within.mean()) / pooled),
    }


def md_table(headers: list[str], rows: list[list]) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


# --- trust flags + the standard score-report shape (exp_053) -----------------
#
# The exam (run_validation) certifies each metric PER STYLE; scores on styles
# where a metric failed its exam, or whose LOO ceiling rests on too few clips,
# are printed but flagged — a number the exam couldn't certify must not look
# like one it did.

def trust_flags(validation_results: dict, ref_counts: dict[str, int],
                motion_recall_min: float = 0.5, min_ceiling_clips: int = 4) -> dict:
    """Per-style reliability from the validation run's results.json.
    motion_trusted: the exam retrieved this style via motion fidelity.
    ceiling_trusted: the LOO ceiling has >= min_ceiling_clips-1 neighbors."""
    recall = validation_results["retrieval"]["motion_fidelity"]["per_class_recall"]
    flags = {}
    for style, n in ref_counts.items():
        r = recall.get(style)
        flags[style] = {
            "motion_trusted": bool(r is not None and np.isfinite(r) and r >= motion_recall_min),
            "ceiling_trusted": n >= min_ceiling_clips,
            "n_ref_clips": n,
            "motion_recall": None if r is None else float(r),
        }
    return flags


def _mean_std(vals: list) -> tuple[float, float, int]:
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def _cell(vals: list, flagged: bool = False) -> str:
    m, s, n = _mean_std(vals)
    if n == 0:
        return "—"
    txt = f"{m:.2f}±{s:.2f}"
    return f"({txt})†" if flagged else txt


def score_tables(rows: list[dict], trust: dict | None = None,
                 judge_by_arm: dict | None = None) -> str:
    """The standard two-table report: HEADLINE (the axes that discriminate
    under endpoint conditioning — appearance / motion / judge / endpoints+seam
    / leakage) and ANALYSIS (M1 profile/timing scalars — necessary-not-
    sufficient there; they earn headline status only in unconditioned or
    timing-focused comparisons). All cells mean±std over items; † = the exam
    could not certify this metric for this item's style; ‡ = ceiling rests on
    <4 reference clips."""
    trust = trust or {}
    arms = sorted({r["arm"] for r in rows})

    def vals(sub, col):
        return [r.get(col) for r in sub]

    def arm_flag(sub, key):
        st = {r["style"] for r in sub}
        return any(not trust.get(s, {}).get(key, True) for s in st)

    head_rows, ana_rows = [], []
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        mflag = arm_flag(sub, "motion_trusted")
        cflag = arm_flag(sub, "ceiling_trusted")
        ep = vals(sub, "prefix_dino") + vals(sub, "suffix_dino")
        judge = (judge_by_arm or {}).get(arm, {})
        jp = judge.get("all_pass")
        head_rows.append([
            arm, len(sub),
            _cell(vals(sub, "norm_appearance_best"), flagged=cflag),
            _cell(vals(sub, "norm_motion_fidelity_mean"), flagged=mflag or cflag),
            f"{jp:.2f}" if jp is not None and np.isfinite(jp) else "—",
            _cell(ep),
            _cell(vals(sub, "max_seam_z")),
            _cell(vals(sub, "leak_max_sim_target")),
        ])
        ana_rows.append([
            arm,
            _cell(vals(sub, "norm_profile_dtw_best"), flagged=cflag),
            _cell(vals(sub, "scalar_depth")),
            _cell(vals(sub, "scalar_depart")),
            _cell(vals(sub, "scalar_arrive")),
            _cell(vals(sub, "scalar_core_frac")),
            _cell(vals(sub, "leak_excess")),
            str(sum(1 for r in sub if r.get("scalar_cross_high"))),
        ])

    out = ["## Headline — discriminative axes (mean±std per arm)\n",
           md_table(["arm", "n", "appearance", "motion", "judge pass",
                     "endpoint DINO", "max seam z", "leak max sim"], head_rows),
           "\n† metric not exam-certified for this style · ‡/() see trust flags · "
           "seam z < 0 = no seam · leak max sim ≥ ~0.9 = near-copy regime\n",
           "\n## Analysis — profile/timing scalars (saturate under endpoint conditioning)\n",
           md_table(["arm", "profile DTW (norm)", "depth", "depart", "arrive",
                     "core frac", "leak excess", "cross>0.85 items"], ana_rows)]
    return "\n".join(out)
