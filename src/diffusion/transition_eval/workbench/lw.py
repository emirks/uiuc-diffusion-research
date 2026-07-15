"""Ledoit-Wolf shrinkage whitener — E1' arm C ONLY (diagnostic, NON-GATING).

WHY THIS ARM EXISTS. Escalation (a) left one question open: the E1 kill was recorded
against a whitener whose regularization (`eig_floor_ratio = 1e-6`) was EXECUTOR-CHOSEN,
and the whitened no-subtraction control scored at chance while the raw one scored 0.605.
Was the whitening the instrument failure, or was that particular floor?

Ledoit-Wolf answers it without introducing a second executor choice, because it HAS NO
FREE PARAMETER: the shrinkage intensity is a closed-form function of the data. If a
principled, parameter-free whitener also flattens the signal, the failure belongs to
whitening-on-this-manifold, not to one arbitrary floor. If it does not, the floor is
implicated. Either way the executor chose nothing.

`sklearn` is not installed in this env, so the estimator is implemented from the
published formula (Ledoit & Wolf 2004, "A well-conditioned estimator for
large-dimensional covariance matrices") and unit-tested against its own algebraic
identities — trace preservation is exact, so the test is a real check, not a tautology.

    S    = (1/n) sum_k x_k x_k^T                (x_k centered)
    m    = tr(S)/p
    d^2  = ||S - m I||_F^2 / p
    b~^2 = (1/(n^2 p)) * [ sum_k ||x_k||^4 - n ||S||_F^2 ]      (exact, see below)
    b^2  = min(b~^2, d^2)
    Sigma* = (b^2/d^2) m I + (1 - b^2/d^2) S

The b~^2 identity: sum_k ||x_k x_k^T - S||_F^2 = sum_k ||x_k||^4 - 2 sum_k x_k^T S x_k
+ n||S||_F^2, and sum_k x_k^T S x_k = tr(S sum_k x_k x_k^T) = n tr(S^2) = n||S||_F^2 for
symmetric S. So the middle term collapses and the whole quantity is computable from
sum_k ||x_k||^4 and ||S||_F^2 alone — no p x p per-sample outer products.

Sigma* has lambda_min >= (b^2/d^2) * m > 0 by construction, so the whitener needs NO
eigenvalue floor. That is the entire point.
"""

from __future__ import annotations

import numpy as np


def shrinkage(X: np.ndarray) -> dict:
    """Ledoit-Wolf shrinkage intensity and the shrunk covariance, from [N, p] data."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu

    S = (Xc.T @ Xc) / n                       # LW's convention: 1/n, not 1/(n-1)
    S = 0.5 * (S + S.T)
    m = float(np.trace(S)) / p
    S_fro2 = float(np.sum(S * S))             # ||S||_F^2

    d2 = float(np.sum((S - m * np.eye(p)) ** 2)) / p
    sq_norms = np.einsum("ij,ij->i", Xc, Xc)  # ||x_k||^2
    b_bar2 = float(np.sum(sq_norms ** 2) - n * S_fro2) / (n * n * p)
    b_bar2 = max(b_bar2, 0.0)                 # numerical guard; the quantity is >= 0
    b2 = min(b_bar2, d2)
    delta = 0.0 if d2 <= 0 else b2 / d2

    Sigma = delta * m * np.eye(p) + (1.0 - delta) * S
    return {"Sigma": Sigma, "shrinkage": float(delta), "mu": mu, "m": m,
            "d2": d2, "b_bar2": b_bar2, "n": int(n), "p": int(p),
            "emp_cov": S}


def fit(X: np.ndarray) -> dict:
    """The shrinkage-whitening artifact: W = Sigma*^(-1/2), same affine convention as
    whitening.whiten (W @ (x - mu)) so the two arms differ ONLY in the covariance
    estimator — never in how the map is applied."""
    lw = shrinkage(X)
    lam, V = np.linalg.eigh(lw["Sigma"])
    lam = np.clip(lam, 0.0, None)
    if lam.min() <= 0:                        # cannot happen for delta > 0; assert it
        raise RuntimeError("Ledoit-Wolf Sigma* is singular — the shrinkage failed")
    W = (V * (1.0 / np.sqrt(lam))) @ V.T
    lam_raw = np.linalg.eigvalsh(lw["emp_cov"])
    lam_raw = np.clip(lam_raw, 0.0, None)
    return {
        "mean": lw["mu"], "W": W,
        "shrinkage": lw["shrinkage"],
        "eigvals": lam,
        "n_frames": int(lw["n"]), "dim": int(lw["p"]),
        "eig_floor_ratio": None,              # THERE IS NONE. That is the point.
        "condition_number_shrunk": float(lam.max() / lam.min()),
        "condition_number_raw": float(lam_raw.max() / max(lam_raw.min(), 1e-300)),
        "n_floored": 0,                       # nothing is floored; Sigma* is PD already
        "tag": "ledoit-wolf-v1",
    }


def whiten(art: dict, X: np.ndarray) -> np.ndarray:
    """Affine, identical in form to whitening.whiten."""
    return np.asarray(X, dtype=np.float64) @ art["W"].T - (art["W"] @ art["mean"])
