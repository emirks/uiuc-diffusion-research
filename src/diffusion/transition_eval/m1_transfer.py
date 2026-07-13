"""M1 — Transfer fidelity (SPEC §3): does the generation match the reference demo?

M1a appearance   — set-similarity between gen core and REFERENCE core (reference-
                   centric: works for singleton/unseen references, no corpus needed).
M1b camera match — per-step robust similarity-transform fit on tracklets
                   (dx, dy, log-scale, rotation) -> 4-channel trajectory compared
                   by banded DTW + correlation. The SfM-free projection of
                   RotErr/TransErr, chosen because transition scenes are
                   non-rigid/dissolving (SPEC §3 M1b).
M1c object match — Yatim-style MFS velocity-direction correlation on RESIDUAL
                   velocities after removing the M1b global fit per step.

Robust weighting for the M1b fit: 2-round median trim (primary) or Huber
IRLS (O7 secondary). The draft.7 exam triggered the O7 conditional
(camera-stratum mean recall 0.346 < 0.5), so draft.8 examines Huber under
the SAME pre-registered adoption rule; the exam's winner is what score.py
deploys (--camera-fit). No other weighting schemes enter.

Pure numpy; GPU only upstream (tracking/features). Unit-tested synthetically.
"""

from __future__ import annotations

import numpy as np

from .appearance import set_similarity
from .morph import dtw_distance, resample_curve, znorm

MIN_FIT_POINTS = 6      # below this the per-step camera fit is invalid
TRIM_FACTOR = 3.0       # residual > TRIM_FACTOR * median -> dropped, one refit
N_STEPS = 64            # shared resampling grid (matches v2 motion protocol)


# --- M1a ----------------------------------------------------------------------

def appearance_ref(gen_feats: np.ndarray, gen_core: np.ndarray,
                   ref_feats: np.ndarray, ref_core: np.ndarray) -> float:
    """Symmetric mean-of-max cosine between gen core and reference core frames."""
    return set_similarity(gen_feats[gen_core], ref_feats[ref_core])


# --- M1b ----------------------------------------------------------------------

def _fit_similarity(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares similarity transform Q ≈ M P + t with M = [[a,-b],[b,a]].
    Closed form on centered coordinates. P, Q: [N,2]."""
    Pm, Qm = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - Pm, Q - Qm
    denom = (Pc ** 2).sum() + 1e-12
    a = (Pc * Qc).sum() / denom
    b = (Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0]).sum() / denom
    M = np.array([[a, -b], [b, a]])
    t = Qm - M @ Pm
    return M, t


def _fit_similarity_weighted(P: np.ndarray, Q: np.ndarray,
                             w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted closed form of _fit_similarity (Huber IRLS inner step)."""
    ws = w.sum() + 1e-12
    Pm, Qm = (w[:, None] * P).sum(axis=0) / ws, (w[:, None] * Q).sum(axis=0) / ws
    Pc, Qc = P - Pm, Q - Qm
    denom = (w[:, None] * Pc ** 2).sum() + 1e-12
    a = (w[:, None] * Pc * Qc).sum() / denom
    b = (w * (Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])).sum() / denom
    M = np.array([[a, -b], [b, a]])
    t = Qm - M @ Pm
    return M, t


def _fit_similarity_huber(P: np.ndarray, Q: np.ndarray, iters: int = 5,
                          k: float = 1.345) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Huber IRLS similarity fit — the O7 pre-registered SECONDARY scheme,
    examined under the same adoption rule only because the draft.7 exam
    landed camera-stratum mean recall 0.346 < 0.5. MAD-scaled residuals,
    standard k=1.345. Returns (M, t, final_weights)."""
    M, t = _fit_similarity(P, Q)
    w = np.ones(len(P))
    for _ in range(iters):
        res = np.linalg.norm(Q - (P @ M.T + t), axis=1)
        s = 1.4826 * np.median(res) + 1e-12
        u = res / (k * s)
        w = np.where(u <= 1.0, 1.0, 1.0 / u)
        M, t = _fit_similarity_weighted(P, Q, w)
    return M, t, w


def _smooth_tracks(tracks: np.ndarray) -> np.ndarray:
    """Box-smooth along time (same protocol as v2 motion) to kill sub-pixel jitter."""
    T = len(tracks)
    win = max(3, T // 32)
    kern = np.ones(win, dtype=np.float32) / win
    pad = np.pad(tracks, ((win // 2, win - 1 - win // 2), (0, 0), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda x: np.convolve(x, kern, mode="valid"), 0, pad)


def camera_trajectory(tracks: np.ndarray, vis: np.ndarray, fit: str = "median") -> dict:
    """Per-step global (camera) motion from tracklets.

    tracks [T,N,2] in normalized coords, vis [T,N]. fit: 'median' (2-round
    median trim, primary) or 'huber' (IRLS, O7 secondary — deployed only if
    the exam's pre-registered adoption rule selects it). Returns
      params  [T-1, 4]  (dx, dy, dlog_scale, dtheta) per step — NaN where invalid
      Ms, ts            the fitted transforms (for residualization)
      n_points [T-1]    points used per step
      valid    [T-1]    fit had >= MIN_FIT_POINTS after trimming
    """
    if fit not in ("median", "huber"):
        raise ValueError(f"unknown fit scheme {fit!r}")
    if tracks.shape[1] == 0:
        T = len(tracks)
        return {"params": np.full((T - 1, 4), np.nan, dtype=np.float32),
                "Ms": np.tile(np.eye(2, dtype=np.float32), (T - 1, 1, 1)),
                "ts": np.zeros((T - 1, 2), dtype=np.float32),
                "n_points": np.zeros(T - 1, dtype=int),
                "valid": np.zeros(T - 1, dtype=bool)}
    tr = _smooth_tracks(tracks)
    T = len(tr)
    params = np.full((T - 1, 4), np.nan, dtype=np.float32)
    Ms = np.tile(np.eye(2, dtype=np.float32), (T - 1, 1, 1))
    ts = np.zeros((T - 1, 2), dtype=np.float32)
    n_pts = np.zeros(T - 1, dtype=int)
    for s in range(T - 1):
        ok = (vis[s] >= 0.5) & (vis[s + 1] >= 0.5)
        P, Q = tr[s][ok], tr[s + 1][ok]
        if len(P) < MIN_FIT_POINTS:
            continue
        if fit == "huber":
            M, t, w = _fit_similarity_huber(P, Q)
            n_pts[s] = int((w >= 0.5).sum())
        else:
            M, t = _fit_similarity(P, Q)
            # median trim (primary): one trimmed refit on residual outliers
            res = np.linalg.norm(Q - (P @ M.T + t), axis=1)
            keep = res <= TRIM_FACTOR * (np.median(res) + 1e-9)
            if keep.sum() >= MIN_FIT_POINTS and keep.sum() < len(P):
                M, t = _fit_similarity(P[keep], Q[keep])
            n_pts[s] = int(keep.sum()) if keep.sum() >= MIN_FIT_POINTS else int(len(P))
        Ms[s], ts[s] = M, t
        a, b = M[0, 0], M[1, 0]
        center_disp = Q.mean(axis=0) - P.mean(axis=0)   # interpretable translation
        params[s] = [center_disp[0], center_disp[1],
                     0.5 * np.log(a * a + b * b + 1e-12), np.arctan2(b, a)]
    valid = n_pts >= MIN_FIT_POINTS
    return {"params": params, "Ms": Ms, "ts": ts, "n_points": n_pts, "valid": valid}


def camera_match(camA: dict, camB: dict, n: int = N_STEPS) -> dict:
    """Compare two camera trajectories: per-channel duration-normalized resample,
    z-norm, 4-channel banded DTW (lower better) + mean Pearson (higher better).
    Steps are scaled to per-normalized-duration units so 121f and 242f compare."""
    out = {"cam_dtw": float("nan"), "cam_corr": float("nan"),
           "cam_valid": bool(camA["valid"].mean() > 0.5 and camB["valid"].mean() > 0.5)}
    if not out["cam_valid"]:
        return out
    chans_a, chans_b, corrs = [], [], []
    for c in range(4):
        xa, xb = camA["params"][:, c].copy(), camB["params"][:, c].copy()
        xa[~camA["valid"]] = 0.0
        xb[~camB["valid"]] = 0.0
        xa, xb = xa * len(xa), xb * len(xb)          # per-duration units
        ra, rb = resample_curve(xa, n), resample_curve(xb, n)
        za, zb = znorm(ra), znorm(rb)
        chans_a.append(za)
        chans_b.append(zb)
        if ra.std() > 1e-8 and rb.std() > 1e-8:
            corrs.append(float(np.corrcoef(ra, rb)[0, 1]))
    A = np.stack(chans_a, axis=1)
    B = np.stack(chans_b, axis=1)
    out["cam_dtw"] = dtw_distance(A, B)
    out["cam_corr"] = float(np.mean(corrs)) if corrs else float("nan")
    return out


# --- M1c ----------------------------------------------------------------------

def _residual_directions(tracks: np.ndarray, vis: np.ndarray, cam: dict,
                         n_steps: int = N_STEPS, min_vis: float = 0.2,
                         speed_floor: float = 0.1, min_moving_frac: float = 0.05,
                         ) -> np.ndarray:
    """Unit velocity directions AFTER subtracting the fitted global motion per
    step — same gating protocol as v2 motion (visibility, per-duration speed
    floor, moving fraction), so M1c differs from v2 M2 ONLY by residualization."""
    keep = vis.mean(axis=0) >= min_vis
    if keep.sum() == 0:   # low-texture video (splice loops, near-static gens):
        return np.zeros((0, n_steps, 2), dtype=np.float32)  # -> object_match NaN
    tr, vs = _smooth_tracks(tracks[:, keep, :]), vis[:, keep]
    T = len(tr)
    v = np.diff(tr, axis=0)                                     # [T-1, N, 2]
    pred = np.einsum("sij,snj->sni", cam["Ms"], tr[:-1]) + cam["ts"][:, None, :] - tr[:-1]
    v_res = (v - pred) * (T - 1)                                # per-duration units
    vv = 0.5 * (vs[:-1] + vs[1:])
    src = np.linspace(0.0, 1.0, v_res.shape[0])
    dst = np.linspace(0.0, 1.0, n_steps)
    vr = np.stack([np.stack([np.interp(dst, src, v_res[:, i, d]) for d in (0, 1)], axis=-1)
                   for i in range(v_res.shape[1])])
    vis_r = np.stack([np.interp(dst, src, vv[:, i]) for i in range(vv.shape[1])])
    speed = np.linalg.norm(vr, axis=-1)
    active = (speed > speed_floor) & (vis_r > 0.5)
    keep2 = active.mean(axis=1) >= min_moving_frac
    vr, speed, active = vr[keep2], speed[keep2], active[keep2]
    return np.where(active[..., None], vr / (speed[..., None] + 1e-9), 0.0).astype(np.float32)


def object_match(tracksA: np.ndarray, visA: np.ndarray,
                 tracksB: np.ndarray, visB: np.ndarray,
                 camA: dict | None = None, camB: dict | None = None,
                 n_steps: int = N_STEPS) -> float:
    """Bidirectional mean-of-max residual-direction correlation. NaN when either
    side has no residually-moving tracklets (a pure camera move — reported, not
    imputed; the camera channel carries that item)."""
    camA = camA or camera_trajectory(tracksA, visA)
    camB = camB or camera_trajectory(tracksB, visB)
    D1 = _residual_directions(tracksA, visA, camA, n_steps)
    D2 = _residual_directions(tracksB, visB, camB, n_steps)
    if len(D1) == 0 or len(D2) == 0:
        return float("nan")
    C = np.einsum("itc,jtc->ij", D1, D2) / n_steps
    return float(0.5 * (C.max(axis=1).mean() + C.max(axis=0).mean()))
