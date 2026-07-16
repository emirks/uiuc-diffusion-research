"""m1b (camera) candidate distances — pure-motion, from cached camera trajectories.

Inherits the incumbent camera_trajectory + camera_match preprocessing bit-exact
(zero-invalid -> x*len -> resample-64), changing ONLY the channel commensuration
(z-norm vs physical-unit) and/or the distance (banded DTW vs no-warp Euclidean).
The cam_valid = >50%-valid-both NaN-gating is preserved so coverage stays
apples-to-apples with the frozen m1b matrix.

Physical-unit (P) commensuration (fable D2 fix): no z-norm; keep (dx,dy) native and
rescale (dlog_scale, dtheta) by the grid RMS radius r so a scale/rotation change is
expressed as the coordinate DISPLACEMENT it induces at a typical grid point. This
preserves amplitude (which z-norm destroys) and self-suppresses the near-noise
scale/rotation channels — parameter-free (r is a geometric constant of the substrate).
"""

from __future__ import annotations

import numpy as np

from ...morph import dtw_distance, resample_curve, znorm

N_STEPS = 64
# RMS radius of the 20x20 CoTracker query grid on [0,1]^2 (geometric constant).
_g = np.stack(np.meshgrid((np.arange(20) + 0.5) / 20, (np.arange(20) + 0.5) / 20), -1).reshape(-1, 2)
RGRID = float(np.sqrt(((_g - _g.mean(0)) ** 2).sum(1).mean()))     # 0.40773766...


def traj_features(params: np.ndarray, valid: np.ndarray, scheme: str,
                  channels=(0, 1, 2, 3)) -> np.ndarray:
    """[N_STEPS, k] resampled trajectory for one clip under a commensuration scheme."""
    cols = []
    for c in channels:
        x = params[:, c].copy()
        x[~valid] = 0.0
        x = x * len(x)                                # per-duration units (incumbent)
        cols.append(resample_curve(x, N_STEPS))
    A = np.stack(cols, axis=1)                         # [64, k]
    if scheme == "Z":
        A = np.stack([znorm(A[:, i]) for i in range(A.shape[1])], axis=1)
    elif scheme == "P":
        A = A.copy()
        for i, c in enumerate(channels):
            if c in (2, 3):                            # dlog_scale, dtheta -> displacement
                A[:, i] = A[:, i] * RGRID
    return A


def camera_matrix(params_all: np.ndarray, valid_all: np.ndarray, scheme: str,
                  channels=(0, 1, 2, 3), dist: str = "dtw") -> np.ndarray:
    """Serial (DTW ~0.75 ms/pair -> ~19s full; forking the torch-laden parent cost
    more than it saved). NaN where either clip is not cam_valid."""
    n = len(params_all)
    T = [traj_features(params_all[i], valid_all[i], scheme, channels) for i in range(n)]
    cv = [bool(valid_all[i].mean() > 0.5) for i in range(n)]
    D = np.full((n, n), np.nan)
    for i in range(n):
        if not cv[i]:
            continue
        D[i, i] = 0.0
        Ti = T[i]
        for j in range(i + 1, n):
            if not cv[j]:
                continue
            if dist == "dtw":
                v = dtw_distance(Ti, T[j])
            else:
                v = float(np.sqrt(((Ti - T[j]) ** 2).sum(axis=1)).mean())
            D[i, j] = D[j, i] = v
    return D
