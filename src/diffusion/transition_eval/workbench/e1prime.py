"""E1' — the gamma-scalar signature. THE KILL TEST (E1PRIME_DIRECTIVE.md §2).

    "Per clip, three scalar channels over arc-length sigma in [0,1] within the S-mask:
     a_hat(sigma), b_hat(sigma) — the endpoint-progress coordinates S already computes;
     m~(sigma) = m(sigma) - m_lerp(sigma) — sided residual magnitude, null-calibrated.
     Geometry: RAW embedding space. No ZCA anywhere in the gating arm."

WHY THIS IS NOT A RE-RUN OF E1. The owner's own spec-error item (b): the derivation
"curves are integrals of what delta summarizes, so if delta fails the curves must" is
FALSE. E1's delta is NET DISPLACEMENT, ||sum_t rho(t)||; this signature's m~ channel is
sum_t ||rho(t)|| — a path length. An excursion that leaves the chord and returns
annihilates in the first and survives in full in the second. ||sum rho|| does not bound
sum ||rho||. E1's kill therefore does not transfer, and that is the entire licence for
E1'.

EVERY READING BELOW WAS PRE-DECLARED IN E1PRIME_PREREG.md AND FROZEN AT aace78d,
BEFORE ANY CANDIDATE NUMBER EXISTED. The three that mattered:

  sigma  (§P1) — arc length of the 3-CHANNEL SIGNATURE CURVE in R^3, native units,
         shared across the channels. NOT per-channel arc length: that maps any
         monotone channel to a straight line for every clip (test), which would
         annihilate a_hat corpus-wide.
  rho    (§P2) — projection onto the LINEAR span{e_A, e_B} (two-sided) or span{e_A}
         (one-sided), NOT the affine chord line. The one-sided clause decides it:
         "projection onto e_A alone" has no affine reading.
  m~ = 0 (§P5) — make_lerp is idempotent on its own endpoints, so a rendered null's own
         m~ is EXACTLY zero. Disclosed before the IV ran; it is why the IV certifies
         less than its name suggests.
"""

from __future__ import annotations

import numpy as np

from ..s_structure import core_mask_v3
from . import anchors, curves, paths

N_SIGMA = 64                 # §1.3 (signature channels)
RANK_EPS = 1e-8              # degenerate-basis guard (§P2)

CHANNELS = {                 # the closed arm list (§2.2) — nothing else is computed
    "A_gating": ("a_hat", "b_hat", "m_tilde"),
    "B_no_null_sub": ("a_hat", "b_hat", "m"),
    "C_ledoit_wolf": ("a_hat", "b_hat", "m_tilde"),
    "D_m_tilde_alone": ("m_tilde",),
}


# --- geometry -----------------------------------------------------------------

def sided_basis(eA: np.ndarray, eB: np.ndarray, sidedness: str) -> np.ndarray | None:
    """Orthonormal basis of the LINEAR subspace to project OUT (§P2).

    two-sided -> span{e_A, e_B} (2-D);  one-sided -> span{e_A} (1-D).

    Returns None when the two-sided basis is degenerate (e_A, e_B collinear) — the
    clip is then UNDEFINED with a reason and COUNTED, never silently demoted to
    rank-1 (§1.5: undefined is not zero, and it is not a quiet fallback either)."""
    nA = float(np.linalg.norm(eA))
    if nA < RANK_EPS:
        return None
    q1 = eA / nA
    if sidedness == "onesided":
        return q1[:, None]
    r = eB - (eB @ q1) * q1
    nr = float(np.linalg.norm(r))
    if nr < RANK_EPS * max(float(np.linalg.norm(eB)), 1.0):
        return None
    return np.stack([q1, r / nr], axis=1)


def sided_m(feats: np.ndarray, Q: np.ndarray, D: float) -> np.ndarray:
    """m(sigma) = ||rho(sigma)|| / D, rho = f - Q Q^T f (§2.1)."""
    rho = feats - (feats @ Q) @ Q.T
    return np.linalg.norm(rho, axis=1) / D


# --- the signature ------------------------------------------------------------

def clip_signature(bundle: dict, sidedness: str, null_feats: np.ndarray,
                   eA: np.ndarray, eB: np.ndarray, low_D: bool,
                   whiten_fn=None, sigma_source: str = "signature",
                   feats: np.ndarray | None = None) -> dict:
    """One clip's gamma-signature: [64, k] per arm, plus its definedness reason.

    `whiten_fn` is None for the raw arms (the gating arm is RAW — no ZCA anywhere,
    per the directive) and the Ledoit-Wolf map for arm C.

    `sigma_source` selects the PRE-DECLARED sigma:
      "signature" — GATING: arc length of the 3-channel curve in R^3 (§P1).
      "embedding" — the NON-GATING sensitivity column: arc length of the raw 768-d
                    embedding path. It cannot change any verdict; it exists so that a
                    sigma-sensitive verdict is adjudicable without a re-run.

    `feats` overrides the frames the signature is computed FROM, while the core mask,
    the anchors and `null_feats` stay the clip's. This is the seam the IV needs: a
    rendered null's signature is "computed identically" (directive 2.1) — the clip's
    core indices, the clip's anchors (the null shares the pair's endpoints) — with the
    null's own frames in the numerator AND in the null slot, because make_lerp is
    idempotent on its own endpoints, so a null's own rendered null IS that null. That
    is exactly why m~ comes out identically 0 for every "nothing" object (PREREG §P5a),
    and it is a property of the registered construction, not a shortcut taken here."""
    if low_D:
        return {"defined": False, "reason": "low_D (below the frozen 5th-pct raw chord "
                                            "floor; every channel divides by D)"}
    mask, meta = core_mask_v3(bundle["profile"], sidedness)
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return {"defined": False, "reason": "core mask < 2 frames"}
    src = bundle["feats"] if feats is None else feats
    if idx.max() >= len(null_feats) or idx.max() >= len(src):
        return {"defined": False, "reason": "null shorter than the clip's core index"}

    f = np.asarray(src, dtype=np.float64)[idx]
    g = np.asarray(null_feats, dtype=np.float64)[idx]   # SAME indices (frozen pooling)
    a, b = np.asarray(eA, dtype=np.float64), np.asarray(eB, dtype=np.float64)
    if whiten_fn is not None:
        f, g = whiten_fn(f), whiten_fn(g)
        a, b = whiten_fn(a[None])[0], whiten_fn(b[None])[0]

    D = float(np.linalg.norm(b - a))
    if D < 1e-12:
        return {"defined": False, "reason": "degenerate chord (D ~ 0)"}
    Q = sided_basis(a, b, sidedness)
    if Q is None:
        return {"defined": False, "reason": "degenerate anchor basis (e_A, e_B collinear)"}

    p = anchors.endpoint_progress(f, a, b)          # a_hat, b_hat — as S computes them
    m = sided_m(f, Q, D)                            # sided residual magnitude
    m_lerp = sided_m(g, Q, D)                       # ... on the clip's own rendered null
    m_tilde = m - m_lerp

    ch = {"a_hat": p["a_hat"], "b_hat": p["b_hat"], "m": m, "m_tilde": m_tilde}
    if not all(np.isfinite(v).all() for v in ch.values()):
        return {"defined": False, "reason": "non-finite channel"}

    sigma = None
    if sigma_source == "embedding":
        sigma = curves.arc_length_sigma(f)          # raw embedding path (NON-GATING)

    out = {"defined": True, "reason": None, "n_core": int(idx.size),
           "D": D, "core_degenerate": bool(meta.get("core_degenerate", False)),
           "m_mean": float(m.mean()), "m_lerp_mean": float(m_lerp.mean()),
           "m_tilde_mean": float(m_tilde.mean())}

    for arm, names in CHANNELS.items():
        if arm == "C_ledoit_wolf" and whiten_fn is None:
            continue
        if arm != "C_ledoit_wolf" and whiten_fn is not None:
            continue
        X = np.stack([ch[n] for n in names], axis=1)          # [T, k], native units
        if arm == "D_m_tilde_alone":
            # "where the signal lives": arm D is arm A's OWN m~ channel in isolation,
            # so it rides arm A's sigma rather than re-deriving one from a 1-D curve.
            XA = np.stack([ch[n] for n in CHANNELS["A_gating"]], axis=1)
            s = sigma if sigma is not None else curves.arc_length_sigma(XA)
            out[arm] = _resample_on(X, s, N_SIGMA)
        else:
            s = sigma if sigma is not None else curves.arc_length_sigma(X)
            out[arm] = _resample_on(X, s, N_SIGMA)
    return out


def _resample_on(X: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
    """Resample channels [T, k] onto a uniform grid of the GIVEN sigma -> [n, k].

    Equivalent to curves.resample(X, n) when sigma is X's own arc length (asserted by
    test); the explicit form is needed because the non-gating sensitivity column
    parameterizes the SAME channels by a DIFFERENT sigma."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[0] == 1:
        return np.repeat(X, n, axis=0)
    grid = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(grid, sigma, X[:, d]) for d in range(X.shape[1])], axis=1)


# --- distances ----------------------------------------------------------------

def banded_dtw_batch(A: np.ndarray, B: np.ndarray, band: int) -> np.ndarray:
    """Vectorized-over-pairs Sakoe-Chiba banded DTW for SINGLE-channel curves.

    A, B: [P, n]. Returns [P]. Bit-identical to curves.banded_dtw pair-by-pair
    (asserted by test) — this is an optimization, never a change of semantics: the
    scalar form would need ~62M Python-level cell updates for one arm's 223x223 matrix.
    The DP is sequential in both i and j, so the vectorization is over PAIRS, not over
    the recurrence."""
    P, n = A.shape
    m = B.shape[1]
    prev = np.full((P, m + 1), np.inf)
    prev[:, 0] = 0.0
    for i in range(1, n + 1):
        cur = np.full((P, m + 1), np.inf)
        lo, hi = max(1, i - band), min(m, i + band)
        ai = A[:, i - 1]
        for j in range(lo, hi + 1):
            cost = np.abs(ai - B[:, j - 1])
            cur[:, j] = cost + np.minimum(np.minimum(prev[:, j], cur[:, j - 1]),
                                          prev[:, j - 1])
        prev = cur
        prev[:, 0] = np.inf                 # only the origin may have zero cost
    d = prev[:, m]
    return np.where(np.isfinite(d), d, np.nan)


def distance_matrix(sigs: list[np.ndarray | None], band_frac: float = curves.DTW_BAND_FRAC
                    ) -> np.ndarray:
    """Banded DTW per channel, then the EQUAL-WEIGHT MEAN of the channel distances
    (the directive's frozen combination rule). NaN where either signature is
    undefined — the frozen kernel reads NaN as 'cannot retrieve' and drops it from
    coverage (§1.5)."""
    n = len(sigs)
    D = np.full((n, n), np.nan)
    ok = [i for i, s in enumerate(sigs) if s is not None]
    if len(ok) < 2:
        return D
    S = np.stack([sigs[i] for i in ok])                       # [N, 64, k]
    N, L, K = S.shape
    band = max(1, int(round(band_frac * L)))
    iu, ju = np.triu_indices(N, k=1)
    acc = np.zeros(len(iu))
    for c in range(K):
        acc += banded_dtw_batch(S[iu, :, c], S[ju, :, c], band)
    acc /= K
    for i in ok:
        D[i, i] = 0.0
    idx = np.array(ok)
    D[idx[iu], idx[ju]] = acc
    D[idx[ju], idx[iu]] = acc
    return D


def zscore_signatures(sigs: list[np.ndarray | None], scaler: dict
                      ) -> list[np.ndarray | None]:
    return [None if s is None else (s - scaler["mean"]) / scaler["std"] for s in sigs]


def fit_scaler(sigs: list[np.ndarray | None]) -> dict:
    """Per-channel corpus z-norm, fitted ONCE on the 223 REAL clips (§P7) and applied
    unchanged to nulls, cuts and IV2 lerps — a calibration object must not be allowed
    to move the scale it is measured against."""
    stack = np.stack([s for s in sigs if s is not None])       # [N, 64, k]
    return {"mean": stack.mean(axis=(0, 1)), "std": stack.std(axis=(0, 1)) + 1e-12,
            "n_curves": int(len(stack))}
