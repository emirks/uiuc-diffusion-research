"""Curve conventions (RUNBOOK §1.3), shared by both phases.

    "Parameterize by normalized arc length sigma in [0,1] within the S-mask (not
     raw time, not progress — progress can be non-monotone). Resample all
     signature channels to 64 points (motion descriptors: 32). Z-score per
     channel over the corpus. Distance: L2 default; banded DTW with <=10% band as
     fallback alignment. Never widen the band — timing is M1d's property. No
     differential invariants (curvature/torsion): noise amplifiers at 20-100
     jittery frames. Integral quantities only."

Arc length, not time: a clip that dwells then lurches and a clip that moves
steadily trace the same PATH; parameterizing by path length compares shapes
rather than schedules. Progress (a_hat/b_hat) is explicitly rejected as the
parameter because it can be non-monotone, which would fold the curve back on
itself.

Z-scoring is per channel and fitted ON THE CORPUS (a frozen scale, applied
identically to every clip and every candidate) — never per clip, which would
destroy exactly the amplitude information the descriptors carry.
"""

from __future__ import annotations

import numpy as np

N_SIGNATURE = 64        # appearance signature channels (§1.3)
N_MOTION = 32           # motion descriptors (§1.3)
DTW_BAND_FRAC = 0.10    # <=10%, frozen — never widen (§1.3)


def arc_length_sigma(X: np.ndarray) -> np.ndarray:
    """Normalized cumulative chordal arc length sigma in [0,1] for a curve [T, D].

    Degenerate curves (a clip that never moves — total path length ~0) have no
    meaningful arc-length parameter; they fall back to uniform spacing, which
    keeps the resample defined and lets the caller's definedness gate, not a
    divide-by-zero, decide what to do with them."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) == 1:
        return np.zeros(1)
    steps = np.linalg.norm(np.diff(X, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    total = s[-1]
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(X))
    return s / total


def resample(X: np.ndarray, n: int) -> np.ndarray:
    """Resample a curve [T, D] to n points at uniform arc length -> [n, D]."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if len(X) == 0:
        return np.full((n, X.shape[1]), np.nan)
    if len(X) == 1:
        return np.repeat(X, n, axis=0)
    sigma = arc_length_sigma(X)
    grid = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(grid, sigma, X[:, d]) for d in range(X.shape[1])], axis=1)


def fit_channel_scaler(curves: list[np.ndarray]) -> dict:
    """Per-channel mean/std over the CORPUS (frozen, applied to every clip).

    Undefined clips (all-NaN curves) are excluded from the fit and never
    contribute to the scale."""
    stack = np.stack([c for c in curves if c is not None and np.isfinite(c).all()])
    return {"mean": stack.mean(axis=(0, 1)), "std": stack.std(axis=(0, 1)) + 1e-12,
            "n_curves": len(stack)}


def zscore(curve: np.ndarray, scaler: dict) -> np.ndarray:
    return (np.asarray(curve, dtype=np.float64) - scaler["mean"]) / scaler["std"]


def l2(a: np.ndarray, b: np.ndarray) -> float:
    """Default distance: L2 between two resampled, z-scored curves."""
    if a is None or b is None or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def banded_dtw(a: np.ndarray, b: np.ndarray, band_frac: float = DTW_BAND_FRAC) -> float:
    """Sakoe-Chiba banded DTW — the fallback alignment (§1.3).

    The band is a hard <=10% of the curve length: timing is M1d's property, and a
    wide band would let a metric claiming to measure SHAPE quietly absorb
    arbitrary time warps."""
    if a is None or b is None or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    n, m = len(a), len(b)
    band = max(1, int(round(band_frac * max(n, m))))
    INF = np.inf
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF)
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
        prev[0] = INF          # only the origin may have zero cost
    d = prev[m]
    return float(d) if np.isfinite(d) else float("nan")


def distance_matrix(curves: list[np.ndarray], metric: str = "l2") -> np.ndarray:
    """Symmetric [n, n] distance matrix, NaN where either curve is undefined.

    NaN is the definedness channel the frozen exam kernel already understands
    (retrieval_eval treats NaN as +inf for retrieval and drops it from coverage),
    so §1.5's discipline falls out of the kernel rather than being bolted on."""
    fn = l2 if metric == "l2" else banded_dtw
    n = len(curves)
    D = np.full((n, n), np.nan)
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = fn(curves[i], curves[j])
    return D
