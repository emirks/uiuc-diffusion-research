"""Exam machinery + trust flags (v3 surface).

Kept: retrieval_eval (the LOO 1-NN style-discrimination exam — certify/exam.py
runs it per metric per variant), wilson_interval (honest error bars for exam
accuracies), trust_flags (per-style reliability from the certification exam),
md_table (report rendering).

Retired to git history (SPEC v3 §3 "Deleted from v2"): normalize_score and
score_tables — floor/ceiling normalization no longer exists; v3 reports raw
scores with control arms and the paired twin table (score.py).
"""

from __future__ import annotations

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion — the honest error
    bar for exam accuracies."""
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


# --- trust flags ---------------------------------------------------------------
#
# The certification exam certifies each metric PER STYLE; scores on styles
# where a metric failed its exam are printed but flagged — a number the exam
# couldn't certify must not look like one it did.

def trust_flags(validation_results: dict, ref_counts: dict[str, int],
                motion_recall_min: float = 0.5, min_ceiling_clips: int = 4) -> dict:
    """Per-style reliability from the exam's results.json.
    motion_trusted: the exam retrieved this style via motion.
    ceiling_trusted: enough real clips for real-sibling context (SPEC §4)."""
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
