"""Corpus ZCA (RUNBOOK §1.1).

    "Fit corpus-level ZCA on per-frame masked DINOv2 embeddings (per A3: CLS
     embeddings, S-mask frame selection). Fit once, freeze, persist the matrix.
     All inner-product geometry downstream (anchors, projections, residuals,
     distances) operates in whitened space. Raw DINO is anisotropic; unwhitened
     chords and angles are measured with a bent ruler."

FIT POPULATION (pinned, not open): the S-mask core frames of all 223 corpus
clips, pooled — §1.1's own parenthetical says "S-mask frame selection", and A3
adds only that these are CLS embeddings (the cache holds no patch tokens) pooled
across the corpus. Fitting on all frames instead would be quiet drift on a frozen
scientific choice. The consequence — endpoint vectors and rendered-null curves
are whitened by a map fitted off their own manifold — is a property the freeze
accepted; it is parked in IDEAS_NEXT_CYCLE.md, not litigated here.

ZCA (not PCA): W = V diag(1/sqrt(lambda + eps)) V^T is the symmetric whitener,
so whitened axes stay aligned with the original feature axes and the map is a
pure "unbend the ruler" operation. eps is a deterministic eigenvalue floor
(EIG_FLOOR_RATIO * lambda_max), recorded in the artifact and frozen before any
candidate runs — DINO's covariance has a long tail of near-zero directions and
dividing by their square roots would amplify pure noise.

The fit is deterministic: same bundles -> same matrix, bitwise.
"""

from __future__ import annotations

import pathlib

import numpy as np

from ..s_structure import core_mask_v3

EIG_FLOOR_RATIO = 1e-6      # frozen: lambda_i <- max(lambda_i, ratio * lambda_max)
ZCA_TAG = "zca-core-v1"     # cache/artifact tag


def core_frames(bundles: list[dict], sidedness: list[str]) -> np.ndarray:
    """The fit population: every clip's S-mask core frames, pooled [N, 768].

    Uses the deployed core_mask_v3 with the manifest's sidedness — the same mask
    the certified M1a operates on, which is what makes the whitened space the
    same space the incumbent is judged in."""
    out = []
    for b, s in zip(bundles, sidedness):
        mask, _ = core_mask_v3(b["profile"], s)
        idx = np.flatnonzero(mask)
        if idx.size:
            out.append(b["feats"][idx])
    return np.concatenate(out).astype(np.float64)


def fit_zca(X: np.ndarray, eig_floor_ratio: float = EIG_FLOOR_RATIO) -> dict:
    """ZCA from a [N, D] frame matrix. Returns the frozen whitening artifact."""
    mu = X.mean(axis=0)
    Xc = X - mu
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    C = 0.5 * (C + C.T)                       # exact symmetry -> eigh is stable
    lam, V = np.linalg.eigh(C)                # ascending, real
    lam = np.clip(lam, 0.0, None)
    floor = eig_floor_ratio * float(lam.max())
    lam_f = np.maximum(lam, floor)
    W = (V * (1.0 / np.sqrt(lam_f))) @ V.T    # V diag(lam^-1/2) V^T
    W_inv = (V * np.sqrt(lam_f)) @ V.T
    return {
        "mean": mu, "W": W, "W_inv": W_inv, "eigvals": lam,
        "eig_floor": np.float64(floor),
        "eig_floor_ratio": np.float64(eig_floor_ratio),
        "n_frames": np.int64(len(X)), "dim": np.int64(X.shape[1]),
        "n_floored": np.int64(int((lam < floor).sum())),
        "condition_number_raw": np.float64(float(lam.max() / max(lam.min(), 1e-300))),
        "tag": ZCA_TAG,
    }


def whiten(zca: dict, X: np.ndarray) -> np.ndarray:
    """Apply the frozen map to [.., D] embeddings. Affine: W @ (x - mu).

    Note this is affine, not linear: mu cancels in any DIFFERENCE of whitened
    vectors (the E1 delta, the endpoint chord) but not in a whitened vector
    itself."""
    return np.asarray(X, dtype=np.float64) @ zca["W"].T - (zca["W"] @ zca["mean"])


def save(zca: dict, path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **zca)
    return path


def load(path: pathlib.Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def sanity(zca: dict, X: np.ndarray) -> dict:
    """Whitened covariance should be ~I on the fit population — the check that
    the ruler is actually straight now."""
    Z = whiten(zca, X)
    C = np.cov(Z, rowvar=False)
    d = C.shape[0]
    off = C - np.diag(np.diag(C))
    return {
        "mean_abs_whitened_mean": float(np.abs(Z.mean(axis=0)).mean()),
        "mean_diag": float(np.diag(C).mean()),
        "max_abs_offdiag": float(np.abs(off).max()),
        "frobenius_dev_from_I": float(np.linalg.norm(C - np.eye(d)) / np.sqrt(d)),
    }
