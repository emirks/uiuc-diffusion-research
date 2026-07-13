"""M1 — Morph Profile: the content-invariant transition signature.

Per-frame cosine similarity to the video's OWN endpoints gives curves a(t)
(similarity to endpoint A = mean feature of the first `n_prefix` frames) and
b(t) (endpoint B = mean of the last `n_suffix` frames). Curves are
floor-normalized by cross = cos(eA, eB) — the "unrelated content" baseline for
that specific pair — so a_hat ≈ 1 means "is endpoint A" and a_hat ≈ 0 means
"neither endpoint", with thresholds that transfer across content.

Pure numpy: unit-testable on the login node without torch/GPU.
"""

from __future__ import annotations

import numpy as np

# Default endpoint windows match the C2V conditioning contract:
# first 2 latent frames = 9 px frames, last 1 latent frame = 8 px frames.
N_PREFIX = 9
N_SUFFIX = 8

# Above this endpoint cross-similarity the (1 - cross) normalization divides
# by a small number — profile scalars for such items are flagged, not trusted.
CROSS_HIGH_THRESH = 0.85


def morph_profile(feats: np.ndarray, n_prefix: int = N_PREFIX, n_suffix: int = N_SUFFIX,
                  n_endpoints: int = 2) -> dict:
    """feats: L2-normalized [T, D]. Returns raw + floor-normalized curves."""
    if len(feats) < n_prefix + n_suffix + 4:
        raise ValueError(f"video too short for morph profile: T={len(feats)}")
    eA = feats[:n_prefix].mean(axis=0)
    eA /= np.linalg.norm(eA) + 1e-12
    a = feats @ eA
    if n_endpoints == 2:
        eB = feats[-n_suffix:].mean(axis=0)
        eB /= np.linalg.norm(eB) + 1e-12
        b = feats @ eB
        cross = float(eA @ eB)
    else:
        # 1-endpoint (portal-style): no B available; the floor is the video's
        # own most-dissimilar frame (weaker contract, stated in the paper).
        b = None
        cross = float(np.percentile(a, 5))
    denom = max(1.0 - cross, 1e-6)
    a_hat = np.clip((a - cross) / denom, -0.25, 1.25)
    b_hat = np.clip((b - cross) / denom, -0.25, 1.25) if b is not None else None
    # Edge guard: semantically-close endpoints (same subject before/after the
    # effect — common in portal-style clips) shrink the denominator and make
    # â/b̂ unstable. Downstream reports flag rather than trust these items.
    return {"a": a, "b": b, "a_hat": a_hat, "b_hat": b_hat, "cross": cross,
            "cross_high": bool(cross > CROSS_HIGH_THRESH),
            "n_prefix": n_prefix, "n_suffix": n_suffix, "n_endpoints": n_endpoints}


def _endpoint_envelope(profile: dict) -> np.ndarray:
    """max(a_hat, b_hat) per frame — 'how much is this frame either endpoint'."""
    if profile["b_hat"] is None:
        return profile["a_hat"]
    return np.maximum(profile["a_hat"], profile["b_hat"])


def derived_scalars(profile: dict) -> dict:
    """Content-invariant scalars read off the normalized curves."""
    a_hat, b_hat = profile["a_hat"], profile["b_hat"]
    T = len(a_hat)
    n_pre, n_suf = profile["n_prefix"], profile["n_suffix"]
    mid = slice(n_pre, T - n_suf)
    env = _endpoint_envelope(profile)
    t = np.arange(T) / max(T - 1, 1)

    depth = float(np.clip(1.0 - env[mid].min(), 0.0, 1.0))
    below_a = np.flatnonzero(a_hat < 0.5)
    depart = float(t[below_a[0]]) if len(below_a) else 1.0
    hold_idx = np.flatnonzero(a_hat < 0.9)
    hold = float(t[hold_idx[0]]) if len(hold_idx) else 1.0
    out = {"depth": depth, "depart": depart, "hold": hold,
           "core_frac": float((env[mid] < 0.5).mean())}
    if b_hat is not None:
        below_b = np.flatnonzero(b_hat < 0.5)
        out["arrive"] = float(t[below_b[-1]]) if len(below_b) else 0.0
    return out


def core_mask(profile: dict, thresh: float = 0.5) -> np.ndarray:
    """Frames that are 'neither endpoint' — the effect medium lives here.
    Fallback for shallow transitions (crossfades): the single deepest frame."""
    env = _endpoint_envelope(profile)
    T = len(env)
    mask = env < thresh
    mask[:profile["n_prefix"]] = False
    mask[T - profile["n_suffix"]:] = False
    if not mask.any():
        mid_env = env.copy()
        mid_env[:profile["n_prefix"]] = np.inf
        mid_env[T - profile["n_suffix"]:] = np.inf
        mask[int(np.argmin(mid_env))] = True
    return mask


# --- profile comparison ------------------------------------------------------

def resample_curve(x: np.ndarray, n: int = 96) -> np.ndarray:
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(x)), x)


def znorm(x: np.ndarray) -> np.ndarray:
    s = x.std()
    if s < 1e-6:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def dtw_distance(X: np.ndarray, Y: np.ndarray, band_frac: float = 0.15) -> float:
    """Sakoe-Chiba-banded DTW on multi-channel curves X, Y [n, c] (same n),
    Euclidean local cost, normalized by path length."""
    n = len(X)
    band = max(1, int(band_frac * n))
    inf = np.inf
    D = np.full((n + 1, n + 1), inf)
    D[0, 0] = 0.0
    cost = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        for j in range(lo, hi + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return float(D[n, n] / (2 * n))


def profile_distance(p: dict, q: dict, n: int = 96) -> dict:
    """Compare two morph profiles on their shared channels (a always; b when
    both have 2 endpoints). Returns DTW distance (lower = more similar) and
    mean Pearson r on the linearly resampled curves (no warping)."""
    channels = ["a_hat"] + (["b_hat"] if (p["b_hat"] is not None and q["b_hat"] is not None) else [])
    P = np.stack([znorm(resample_curve(p[c], n)) for c in channels], axis=1)
    Q = np.stack([znorm(resample_curve(q[c], n)) for c in channels], axis=1)
    pearsons = []
    for k in range(P.shape[1]):
        if P[:, k].std() < 1e-6 or Q[:, k].std() < 1e-6:
            pearsons.append(0.0)
        else:
            pearsons.append(float(np.corrcoef(P[:, k], Q[:, k])[0, 1]))
    return {"dtw": dtw_distance(P, Q), "pearson": float(np.mean(pearsons))}
