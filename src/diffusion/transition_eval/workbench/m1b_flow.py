"""M1b_flow — camera metric on optical flow (RUNBOOK §3.2).

    "Model: 4-parameter similarity (tx, ty, log-scale, rotation) fit to the dense
     flow field per frame pair. Fit: Huber IRLS, delta ~ 1.5 px. Where S provides a
     spatial effect mask, fit on its complement; else rely on Huber inlier
     weighting. Definedness: inlier fraction < 40% or low-texture frame -> frame
     undefined. > 30% of core frames undefined -> clip undefined. Descriptor: the
     4 parameter trajectories over core frames -> resample 32 -> z-score per dim
     -> L2 (banded DTW fallback)."

S PROVIDES NO SPATIAL MASK. The certified S-block (s_structure.core_mask_v3) is a
TEMPORAL mask — it selects frames, not pixels; the feature cache holds CLS tokens
only, and no patch tokens or spatial effect masks exist anywhere in the certified
tree. So the second branch of §3.2's fit rule is the operative one: "else rely on
Huber inlier weighting". The effect region is not masked out geometrically; it is
DOWN-WEIGHTED as outlier motion, which is exactly what Huber IRLS is for — the
camera is the dominant, globally-consistent motion, and the effect is the
minority that disagrees with it.

WHY THE MODEL IS LINEAR. A similarity warp sends p -> s*R(theta)*p + t. Writing
a = s*cos(theta), b = s*sin(theta), the two flow equations become

    u + x = a*x - b*y + tx
    v + y = b*x + a*y + ty

which are LINEAR in (a, b, tx, ty). So the design matrix depends only on the pixel
grid — identical for every frame of every clip — and is built exactly once; each
frame is then a weighted 4x4 normal-equation solve. s and theta are recovered as
s = hypot(a, b), theta = atan2(b, a), and the descriptor uses log(s) so that
zoom-in and zoom-out are symmetric about 0.

Coordinates are centered on the image, so tx/ty mean "translation at the image
center" rather than translation of a corner, and rotation/scale do not leak into
them.
"""

from __future__ import annotations

import numpy as np

from . import curves

HUBER_DELTA_PX = 1.5        # §3.2
MIN_INLIER_FRAC = 0.40      # §3.2 — below this the FRAME is undefined
MAX_UNDEFINED_CORE = 0.30   # §3.2 — above this the CLIP is undefined
N_RESAMPLE = 32             # §1.3 motion descriptors
IRLS_ITERS = 20            # converged: 8 iters still drifts (measured), 20 is a fixed point
PIXEL_STRIDE = 3            # subsample the flow grid for the fit (~15k points)

PARAM_NAMES = ("tx", "ty", "log_scale", "rotation")


def design_matrix(h: int, w: int, stride: int = PIXEL_STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """The fixed [2N, 4] design matrix and the [N, 2] sampled pixel grid.

    Built once for the whole corpus — every frame pair of every clip shares it."""
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    x = xs.ravel().astype(np.float64) - (w - 1) / 2.0     # centered
    y = ys.ravel().astype(np.float64) - (h - 1) / 2.0
    n = len(x)
    A = np.zeros((2 * n, 4))
    A[0::2, 0] = x;  A[0::2, 1] = -y;  A[0::2, 2] = 1.0   # u-equation
    A[1::2, 0] = y;  A[1::2, 1] = x;   A[1::2, 3] = 1.0   # v-equation
    return A, np.stack([x, y], axis=1)


def fit_similarity(flow: np.ndarray, A: np.ndarray, grid: np.ndarray,
                   stride: int = PIXEL_STRIDE, delta: float = HUBER_DELTA_PX,
                   iters: int = IRLS_ITERS, valid: np.ndarray | None = None) -> dict:
    """Huber-IRLS similarity fit to ONE dense flow field [H, W, 2].

    Returns the 4 parameters, the inlier fraction, and the robust residual. The
    effect region is not excluded — it is down-weighted as outlier motion.

    MEASURED PROPERTY OF THE PRE-REGISTERED ESTIMATOR (not a bug, and not fixable
    without changing the estimator §3.2 pins). Huber is bounded-influence but
    LOW-BREAKDOWN: a large effect region that moves coherently biases the fit
    toward itself. Measured on constructed truth (a rigid object translating
    ~11 px, camera = 2.0 px pan), max parameter error vs contaminated area:

        5% -> 0.05 px | 15% -> 0.24 | 25% -> 0.52 | 33% -> 0.75
        40% -> 1.68   | 45% -> 2.92 | 55% -> 5.52

    Initializing IRLS from a high-breakdown median-flow estimate changes NOTHING
    (verified: identical to 4 decimals) — the bias is the estimator's fixed point,
    not a bad starting basin. What protects the metric is the DEFINEDNESS gate,
    not the estimator: past ~40% contamination the inlier fraction collapses
    (0.146 at 40%, 0.031 at 45%) and the frame exits as UNDEFINED rather than
    confidently wrong. This is reported, not corrected."""
    f = flow[::stride, ::stride].reshape(-1, 2).astype(np.float64)
    x, y = grid[:, 0], grid[:, 1]
    # targets: (u + x, v + y), interleaved to match A's row order
    t = np.empty(2 * len(f))
    t[0::2] = f[:, 0] + x
    t[1::2] = f[:, 1] + y

    # §3.2's first branch: "where S provides a spatial effect mask, fit on its
    # COMPLEMENT". On the injected-trajectory probes the invalid region is known
    # exactly — the pixels a synthetic warp fills from BORDER_REFLECT are mirror
    # content moving the WRONG WAY, and they are not part of the constructed truth.
    # Excluding them is what keeps "the ground truth is exact" true. Everywhere else
    # valid is None and the fit is unchanged.
    m_valid = None
    if valid is not None:
        m_valid = valid[::stride, ::stride].ravel().astype(bool)
        if m_valid.sum() < 16:
            return {"params": np.full(4, np.nan), "inlier_frac": 0.0,
                    "residual_px": float("nan"), "defined": False,
                    "valid_frac": float(m_valid.mean())}

    w = np.ones(len(f))                      # per-POINT weights (residual is a 2-vector)
    if m_valid is not None:
        w = w * m_valid                      # invalid pixels never vote
    p = np.zeros(4)
    for _ in range(iters):
        W = np.repeat(w, 2)
        AtW = A.T * W
        try:
            p = np.linalg.solve(AtW @ A, AtW @ t)
        except np.linalg.LinAlgError:
            return {"params": np.full(4, np.nan), "inlier_frac": 0.0,
                    "residual_px": float("nan"), "defined": False}
        r = (A @ p - t).reshape(-1, 2)
        rn = np.linalg.norm(r, axis=1)
        w = np.where(rn <= delta, 1.0, delta / np.maximum(rn, 1e-9))   # Huber
        if m_valid is not None:
            w = w * m_valid          # re-apply EVERY iteration: the Huber update
                                     # above would otherwise hand the invalid
                                     # border pixels their vote back

    a, b, tx, ty = p
    s = float(np.hypot(a, b))
    # the inlier fraction is over the VALID support — an invalid pixel is not an
    # outlier, it is not a pixel at all
    sel = m_valid if m_valid is not None else np.ones(len(rn), bool)
    inlier = float(np.mean(rn[sel] <= delta)) if sel.any() else 0.0
    return {
        "params": np.array([tx, ty,
                            np.log(s) if s > 1e-9 else np.nan,
                            float(np.arctan2(b, a))]),
        "inlier_frac": inlier,
        "residual_px": float(np.median(rn[sel])) if sel.any() else float("nan"),
        "defined": bool(inlier >= MIN_INLIER_FRAC and s > 1e-9),
        "valid_frac": float(sel.mean()),
    }


def clip_camera_trajectory(flow: np.ndarray, texture_ok: np.ndarray | None = None,
                           stride: int = PIXEL_STRIDE,
                           valid: np.ndarray | None = None) -> dict:
    """Fit every frame pair of one clip -> [T-1, 4] parameter trajectory + flags.

    texture_ok: per-PAIR boolean (a low-texture frame makes its flow unreliable,
    so §3.2 declares the frame undefined regardless of what the fit reports —
    a confident fit to noise is still noise)."""
    T = len(flow)
    A, grid = design_matrix(flow.shape[1], flow.shape[2], stride)
    params = np.full((T, 4), np.nan)
    inlier = np.zeros(T)
    defined = np.zeros(T, dtype=bool)
    for i in range(T):
        v = valid[i] if valid is not None else None
        r = fit_similarity(flow[i].astype(np.float32), A, grid, stride, valid=v)
        params[i] = r["params"]
        inlier[i] = r["inlier_frac"]
        defined[i] = r["defined"]
    if texture_ok is not None:
        defined &= texture_ok[:T]
        params[~defined] = np.nan
    return {"params": params, "inlier_frac": inlier, "defined": defined}


def clip_descriptor(traj: dict, core_pairs: np.ndarray) -> dict:
    """§3.2 descriptor: the 4 parameter trajectories over CORE frame pairs,
    resampled to 32 by arc length. Returns NaN (undefined, never zero) when too
    much of the core is undefined — an undefined clip must exit the exam as an
    undefined ROW, which the frozen kernel already understands."""
    idx = np.flatnonzero(core_pairs)
    if idx.size == 0:
        return {"curve": None, "undefined_frac": 1.0, "defined": False,
                "reason": "no core frame pairs"}
    ok = traj["defined"][idx]
    undef = float(1.0 - ok.mean())
    if undef > MAX_UNDEFINED_CORE:
        return {"curve": None, "undefined_frac": undef, "defined": False,
                "reason": f"{undef:.0%} of core pairs undefined > {MAX_UNDEFINED_CORE:.0%}"}
    P = traj["params"][idx][ok]                    # defined core pairs only
    if len(P) < 2:
        return {"curve": None, "undefined_frac": undef, "defined": False,
                "reason": "fewer than 2 defined core pairs"}
    return {"curve": curves.resample(P, N_RESAMPLE), "undefined_frac": undef,
            "defined": True, "reason": None}


def corpus_scaler(per_clip: list[dict]) -> dict | None:
    """The per-channel corpus z-scale. Exposed because ANY curve compared against a
    corpus descriptor must go through this same scaler — a raw curve and a z-scored
    curve are not commensurable, and an L2 between them is a meaningless number that
    happens to be finite."""
    defined = [d["curve"] for d in per_clip if d["defined"]]
    return curves.fit_channel_scaler(defined) if defined else None


def corpus_descriptors(per_clip: list[dict]) -> list[np.ndarray | None]:
    """Z-score each of the 4 channels over the CORPUS (a frozen scale applied
    identically to every clip — never per clip, which would destroy the amplitude
    information that distinguishes a slow pan from a fast one)."""
    scaler = corpus_scaler(per_clip)
    if scaler is None:
        return [None] * len(per_clip)
    return [curves.zscore(d["curve"], scaler) if d["defined"] else None
            for d in per_clip]
