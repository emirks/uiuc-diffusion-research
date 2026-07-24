"""exp_077 D2 quality metrics — Assert1 (pure-phase identity), Assert2 (seam), M1 (mush), M2 (near-cut).

All four are computed on the RAW uint8 render (before the lossy mp4 encode) against the REAL
source streams, so Assert1 is a genuine identity check and not a codec-noise measurement.

Assert1  pure-phase identity — MAE of every pure-A frame vs A_src[t] and every pure-B frame vs
         B_src[t]. Progress is pinned 0/1 there, so a gate-passing shader must reproduce the
         source frames exactly (expected ~0).

Assert2  seam — exp_076 `ops3d.seam_error` semantics: the frame step at each handoff divided by
         that bucket's OWN mean frame delta. Raw MAE is not comparable across clips (a near-static
         bucket runs ~1.7, a fast one ~25), so we report the ratio; ~1 means the join is as smooth
         as the content's natural motion, >>1 is a visible cut. Threshold <= 2.0 at BOTH handoffs.

M1       MUSH — s(t) = max(zNCC(F[t], A_src[t]), zNCC(F[t], B_src[t])) on 96x72 grayscale,
         compared against the source frames AT THE SAME t (the streams move, so the endpoint
         blocks are the wrong reference). Clip score = p10 of s(t) over the ramp. A melted /
         warp-destroyed / graphic-matte frame resembles NEITHER stream => low s(t).

M2       NEAR-CUT — q(t) = zNCC(F[t], B_src[t]) - zNCC(F[t], A_src[t]), min-max rescaled per clip.
         Fail if any single-frame |dq| > 0.5, i.e. more than half the A->B swing happens in one
         frame (effectively a cut).
"""

from __future__ import annotations

import numpy as np

M1_GRAY_HW = (96, 72)        # (H, W) — exactly preserves the 640x480 portrait aspect (4:3)
SEAM_MAX = 2.0
M2_MAX_DQ = 0.5


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def to_small_gray(frames: np.ndarray, hw: tuple[int, int] = M1_GRAY_HW) -> np.ndarray:
    """(N,H,W,3) uint8 -> (N, h, w) float32 grayscale, area-averaged down to `hw`."""
    import cv2
    h, w = hw
    out = np.empty((len(frames), h, w), np.float32)
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        out[i] = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    return out


def zncc(x: np.ndarray, y: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation (Pearson r) of two equal-shaped arrays."""
    a = x.ravel().astype(np.float64)
    b = y.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mean_frame_delta(block: np.ndarray) -> float:
    if len(block) < 2:
        return 0.0
    return float(np.abs(np.diff(block.astype(np.float32), axis=0)).mean())


# --------------------------------------------------------------------------
# Assert 1 — pure-phase identity
# --------------------------------------------------------------------------
def assert1_pure_phase(clip: np.ndarray, a_src: np.ndarray, b_src: np.ndarray,
                       i0: int, j0: int) -> dict:
    """MAE over pure-A frames [0..i0] vs A_src and pure-B frames [j0..] vs B_src."""
    da = np.abs(clip[: i0 + 1].astype(np.float32) - a_src[: i0 + 1].astype(np.float32))
    db = np.abs(clip[j0:].astype(np.float32) - b_src[j0:].astype(np.float32))
    # the pinned 9-frame anchor blocks specifically (the conditioning frames)
    d_anchor_a = np.abs(clip[:9].astype(np.float32) - a_src[:9].astype(np.float32)).mean()
    d_anchor_b = np.abs(clip[-9:].astype(np.float32) - b_src[-9:].astype(np.float32)).mean()
    return {"mae_pure_a": float(da.mean()), "mae_pure_b": float(db.mean()),
            "max_pure": float(max(da.max(), db.max())),
            "mae_anchor_a9": float(d_anchor_a), "mae_anchor_b9": float(d_anchor_b),
            "n_pure_a": i0 + 1, "n_pure_b": int(len(clip) - j0)}


# --------------------------------------------------------------------------
# Assert 2 — seam continuity at the two handoffs (exp_076 ops3d.seam_error semantics)
# --------------------------------------------------------------------------
def assert2_seam(clip: np.ndarray, i0: int, j0: int) -> dict:
    """Step at each handoff / that bucket's own mean frame delta."""
    d0 = float(np.abs(clip[i0].astype(np.float32) - clip[i0 + 1].astype(np.float32)).mean())
    d1 = float(np.abs(clip[j0].astype(np.float32) - clip[j0 - 1].astype(np.float32)).mean())
    ref0 = max(_mean_frame_delta(clip[: i0 + 1]), 1e-3)
    ref1 = max(_mean_frame_delta(clip[j0:]), 1e-3)
    return {"seam_mae": [d0, d1], "seam_ratio": [d0 / ref0, d1 / ref1],
            "bucket_delta": [ref0, ref1], "seam_max_ratio": max(d0 / ref0, d1 / ref1)}


# --------------------------------------------------------------------------
# M1 / M2 — resemblance to the two REAL streams at the SAME t
# --------------------------------------------------------------------------
def m1_m2(clip: np.ndarray, a_src: np.ndarray, b_src: np.ndarray, i0: int, j0: int,
          *, gray: tuple | None = None) -> dict:
    gf, ga, gb = gray if gray is not None else (
        to_small_gray(clip), to_small_gray(a_src), to_small_gray(b_src))
    T = len(clip)
    ncc_a = np.array([zncc(gf[t], ga[t]) for t in range(T)])
    ncc_b = np.array([zncc(gf[t], gb[t]) for t in range(T)])
    s = np.maximum(ncc_a, ncc_b)
    ramp = np.arange(i0 + 1, j0)
    s_ramp = s[ramp] if len(ramp) else s
    q = ncc_b - ncc_a
    lo, hi = float(q.min()), float(q.max())
    qn = (q - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(q)
    dq = np.abs(np.diff(qn))
    return {
        "m1_p10": float(np.percentile(s_ramp, 10)),
        "m1_min": float(s_ramp.min()), "m1_mean": float(s_ramp.mean()),
        "m1_p10_frame": int(ramp[int(np.argsort(s_ramp)[max(0, int(0.10 * len(s_ramp)) - 1)])])
        if len(ramp) else -1,
        "m2_max_dq": float(dq.max()) if len(dq) else 0.0,
        "m2_max_dq_frame": int(np.argmax(dq)) if len(dq) else -1,
        "n_ramp": int(len(ramp)),
        "ncc_a": [round(float(v), 4) for v in ncc_a],
        "ncc_b": [round(float(v), 4) for v in ncc_b],
    }


def score_clip(clip: np.ndarray, a_src: np.ndarray, b_src: np.ndarray,
               i0: int, j0: int) -> dict:
    """All four metrics for one rendered clip against its two REAL source streams."""
    out = {"assert1": assert1_pure_phase(clip, a_src, b_src, i0, j0),
           "assert2": assert2_seam(clip, i0, j0)}
    out.update(m1_m2(clip, a_src, b_src, i0, j0))
    return out


def verdict(row: dict, tau: float, *, assert1_tol: float = 0.5,
            seam_max: float = SEAM_MAX, m2_max: float = M2_MAX_DQ) -> dict:
    """Apply the frozen D2 gate to a metric row."""
    # BUGFIX 2026-07-24: this used the MEAN over the pure phase (mae_pure_a/b), which hides a
    # catastrophic localised violation — a shader that breaks p=0/p=1 identity on a few frames or
    # a small region has a near-zero mean but max_pure up to 240 (i.e. the "real frames" region is
    # visibly NOT the source). 20/448 audit tuples passed this leg with max_pure 29-240, all on the
    # known identity-breaking shaders (Radial, BlockDissolve, undulatingBurnOut, BowTieVertical).
    # The spec is "pure-phase frames must be byte-close to the source" = a MAX condition.
    a1 = row["assert1"]["max_pure"] <= assert1_tol
    a2 = row["assert2"]["seam_max_ratio"] <= seam_max
    m1 = row["m1_p10"] >= tau
    m2 = row["m2_max_dq"] <= m2_max
    return {"assert1": bool(a1), "assert2": bool(a2), "m1": bool(m1), "m2": bool(m2),
            "pass": bool(a1 and a2 and m1 and m2)}
