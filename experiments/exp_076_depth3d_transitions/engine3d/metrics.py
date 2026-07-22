"""Parallax Index — a cheap certificate that a clip really is a 3D camera move.

The defining signature of a translating camera is that **near pixels move more
than far pixels, by a ratio the geometry predicts**. Every 2D effect — every wipe,
dissolve and shader warp in the exp_075 bank — has parallax index ≈ 1 by
construction, because it has no notion of depth at all.

So this is not just a quality check, it is the measurement that separates the two
operator banks and validates the label we would attach to each clip.

Cost: one dense optical-flow call per frame pair (OpenCV DIS, CPU) plus the depth
map we already cached.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import spearmanr


# OpenCV's flow solver opens a per-core worker pool; stacked on top of the
# software GL rasteriser that trips the per-user thread limit (EAGAIN).
cv2.setNumThreads(4)


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)


def parallax_index(frames: np.ndarray, view_z: np.ndarray, *,
                   pairs: int = 3) -> dict:
    """Measure parallax over the first `pairs` frame steps of `frames`.

    Returns:
        pi        median flow magnitude in the nearest depth decile
                  / the same in the farthest decile. 1.0 = flat (2D).
        pi_pred   what the depth distribution predicts for a pure translation
                  (z_far / z_near), i.e. the target.
        ratio     pi / pi_pred. ~1 means the rendered parallax matches the
                  geometry the operator commanded.
        rho       Spearman correlation between 1/z and flow magnitude. The single
                  most interpretable "is it 3D" number: >0.6 for a translating
                  camera, ~0 for any 2D effect.
    """
    z = view_z.astype(np.float32)
    lo, hi = np.percentile(z, 10), np.percentile(z, 90)
    near, far = z <= lo, z >= hi

    mags, rhos = [], []
    for i in range(min(pairs, len(frames) - 1)):
        f = _flow(frames[i], frames[i + 1])
        m = np.linalg.norm(f, axis=-1)
        mags.append(m)
        sub = slice(None, None, 7)                       # subsample for speed
        r = spearmanr((1.0 / np.maximum(z, 1e-3)).ravel()[sub], m.ravel()[sub])
        rhos.append(float(r.statistic))
    m = np.mean(mags, axis=0)

    mn = float(np.median(m[near])) if near.any() else 0.0
    mf = float(np.median(m[far])) if far.any() else 0.0
    pi = mn / max(mf, 1e-4)
    pi_pred = float(np.median(z[far]) / max(np.median(z[near]), 1e-4))
    return {
        "pi": round(pi, 3),
        "pi_pred": round(pi_pred, 3),
        "ratio": round(pi / max(pi_pred, 1e-4), 3),
        "rho": round(float(np.mean(rhos)), 3),
        "flow_px": round(float(np.median(m)), 3),
    }
