"""Null-family (endpoint-line) instrument, revised 2026-09-08.

Measures how close a generated middle is to the interpolation family of the paper's Eq. (lerp):
    v_t = (1 - alpha_t) a + alpha_t b,   a = last conditioned start frame, b = first conditioned end frame
(or the final generated frame when there is no end anchor). Pixel space, 128x128, Gaussian blur sigma=1,
same geometry as misc/2026-08-24_lerp_collapse so numbers are comparable.

Per clip we return CONTINUOUS descriptors, not a class:
  confinement : resid profile r_t = || (v_t - a) - proj_line(v_t - a) || / ||b - a||   (interior frames)
                DR_med = median r_t, DR_mean = mean r_t, online_frac = frac(r_t <= theta)
  schedule    : tau_t = <(v_t - a), u_hat> / ||b - a||;  M = frac(tau in [.25,.75]),
                R = p95 - p5 of tau, S = Spearman(tau, frame index)
  scale       : gap = ||b - a||, gap_rel = gap / mean(||a||, ||b||), PR (participation ratio) of interior
  nuisance    : motion_prefix / motion_suffix = mean adjacent-frame L2 inside the conditioning windows / gap
The class label (STATIC / DISSOLVE / CUT / FREEZE / REAL) is derived afterwards and is descriptive only.
Change vs the Aug-24 classifier: DISSOLVE no longer requires Spearman == 1.0 (unreachable for noisy
outputs); it is M >= M_thr on an on-line clip. S is reported as a continuous descriptor instead.
"""
import hashlib
import numpy as np
import cv2

cv2.setNumThreads(1)

SIZE, BLUR = 128, 1.0
THETA = 0.12          # on-line cut point, kept from the Aug-24 campaign for comparability (sensitivity reported)
M_THR = 0.25          # mid-band coverage: synthetic dissolve = 0.50, synthetic cut/freeze = 0.00
R_THR = 0.50


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_matrix(path, size=SIZE, blur=BLUR):
    """Decode mp4 -> (T, d) float32 in [0,1] at size x size, blurred. Returns None if unreadable."""
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        if blur and blur > 0:
            f = cv2.GaussianBlur(f, (0, 0), blur)
        fr.append(f.astype(np.float32) / 255.0)
    cap.release()
    if not fr:
        return None
    a = np.stack(fr)
    return a.reshape(a.shape[0], -1)


def window_indices(T, prefix, suffix, two_sided):
    a_idx = prefix - 1
    b_idx = (T - suffix) if (two_sided and suffix > 0) else (T - 1)
    return a_idx, b_idx, slice(a_idx + 1, b_idx)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan")
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def participation_ratio(M):
    Mc = M - M.mean(axis=0, keepdims=True)
    G = Mc @ Mc.T
    lam = np.clip(np.linalg.eigvalsh(G), 0, None)
    s = lam.sum()
    return float(s * s / (lam * lam).sum()) if s > 0 else 1.0


def motion_energy(frames, gap):
    """Mean adjacent-frame L2 distance normalised by the endpoint gap. NaN if < 2 frames."""
    if frames.shape[0] < 2 or not np.isfinite(gap) or gap <= 0:
        return float("nan")
    d = np.linalg.norm(np.diff(frames, axis=0), axis=1)
    return float(d.mean() / gap)


def measure(M, prefix, suffix, two_sided, theta=THETA, eps=1e-8):
    """Full descriptor set for one decoded clip matrix M (T, d)."""
    T = M.shape[0]
    a_idx, b_idx, interior = window_indices(T, prefix, suffix, two_sided)
    out = dict(T=T, a_idx=a_idx, b_idx=b_idx)
    if b_idx - a_idx < 3:
        out.update(n_interior=0)
        return out, None, None
    a, b = M[a_idx], M[b_idx]
    u = b - a
    gap = float(np.linalg.norm(u))
    out["gap"] = gap
    out["gap_rel"] = float(gap / (0.5 * (np.linalg.norm(a) + np.linalg.norm(b)) + eps))
    X = M[interior]
    n = X.shape[0]
    out["n_interior"] = int(n)
    if gap < eps:
        out.update(DR_med=np.nan, DR_mean=np.nan, online_frac=np.nan, M=np.nan, R=np.nan, S=np.nan,
                   PR=participation_ratio(X), motion_prefix=np.nan, motion_suffix=np.nan)
        return out, None, None
    uhat = u / gap
    V = X - a
    t = V @ uhat
    resid = np.linalg.norm(V - np.outer(t, uhat), axis=1) / gap
    tau = t / gap
    out["DR_med"] = float(np.median(resid))
    out["DR_mean"] = float(resid.mean())
    out["online_frac"] = float((resid <= theta).mean())
    out["M"] = float(((tau >= 0.25) & (tau <= 0.75)).mean())
    out["R"] = float(np.percentile(tau, 95) - np.percentile(tau, 5))
    out["S"] = spearman(tau, np.arange(n))
    out["tau_med"] = float(np.median(tau))
    out["PR"] = participation_ratio(X)
    # nuisance: motion inside the conditioning windows (held shots with camera motion read as off-line)
    out["motion_prefix"] = motion_energy(M[: a_idx + 1], gap)
    out["motion_suffix"] = motion_energy(M[b_idx:], gap) if (two_sided and suffix > 1) else float("nan")
    return out, resid.astype(np.float32), tau.astype(np.float32)


def classify(DR_med, M, R, static, theta=THETA, m_thr=M_THR, r_thr=R_THR):
    """Descriptive label. STATIC = endpoints too close for the gap-normalised residual to mean anything."""
    if static or not np.isfinite(DR_med):
        return "STATIC"
    if DR_med <= theta:
        if np.isfinite(M) and M >= m_thr:
            return "DISSOLVE"
        if np.isfinite(R) and R >= r_thr:
            return "CUT"
        return "FREEZE"
    return "REAL"
