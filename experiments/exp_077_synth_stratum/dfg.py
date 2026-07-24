"""exp_077 D2-FULL — DEGENERATE-FRAME GATE (DFG).

WHY (2026-07-24 ruling, after the parameter-clamp was ABANDONED PERMANENTLY)
---------------------------------------------------------------------------
Every observed D2 failure mode — near-black frames, white blowout, flat mattes (brown /
lavender / orange / olive), saturated washes, geometry blank-outs — is a DEGENERATE FRAME:
extreme mean luma, or near-zero spatial variance. So detect it in PIXEL space, not in
parameter space (the clamp tried parameter space and collapsed two kept shaders onto the
destructive constant; `param_clamp.py` is kept on disk as a record and NEVER runs).

This also dodges the earlier per-frame-floor failure. A per-frame zNCC floor could not tell
decorrelation-by-NOISE (legitimate StaticFade grain) from decorrelation-by-FLATNESS (a junk
matte): both decorrelate from the sources. RAW LUMA STATISTICS separate them trivially —
grain has HIGH pixel variance, a matte has NONE.

THE GATE
--------
Additive accept-criterion evaluated ONLY on clips that already passed the frozen gate, i.e.
AND-composed DOWNSTREAM of it. The accepted set is therefore a SUBSET of the frozen gate's,
so tau = 0.2543 and the frozen gate are UNTOUCHED and no recalibration is permitted.

Per frame t in the TRANSITION WINDOW ONLY (ramp = range(i0+1, j0), identical to M1's ramp —
the pure phases are byte-identical REAL frames and must never be flagged):

  near-black  L(t) < theta_black,  GUARDED by min(La(t), Lb(t)) > 0.15
              (a genuinely dark real scene can never false-positive)
  near-white  L(t) > theta_white,  GUARDED by max(La(t), Lb(t)) < 0.85
  flat        S(t) < theta_flat,   NO guard (real frames are never near-zero variance).
              This single test covers every solid fill regardless of hue.
  sat_wash    sat(t) > theta_sat AND S(t) < sat_flat_mult * theta_flat   [OPTIONAL]
              (a saturated flat colour whose luma is perfectly legal — e.g. pure green
              [0.05, 0.92, 0.13] has luma 0.67)

L(t) = mean luma, S(t) = luma std, both on the 96x72 grayscale M1 already computes, in [0, 1].
sat(t) = mean per-pixel (max - min) over a 96x72 area-downsampled RGB, in [0, 1].
La/Lb = mean luma of the two REAL SOURCE streams at the same t (both are in hand at render
time), also on the 96x72 grayscale M1 already computes for the sources.

A clip is REJECTED if at least K frames of the window are flagged — a 1-2 frame stylistic
flash (FilmBurn, a light leak) is tolerated; sustained degeneracy is junk. There are NO shader
exceptions: a fade THROUGH a held solid counts as BAD.
"""

from __future__ import annotations

import numpy as np

DFG_GRAY_HW = (96, 72)          # == d2_metrics.M1_GRAY_HW

SRC_DARK_GUARD = 0.15           # black test only fires if BOTH sources are brighter than this
SRC_BRIGHT_GUARD = 0.85         # white test only fires if BOTH sources are darker than this

# starting points; the calibration decides the shipped values (DFG_CALIB.json)
DEFAULT_CONFIG: dict = {
    "theta_black": 0.05,
    "theta_white": 0.95,
    "theta_flat": 0.03,
    "K": 3,
    "theta_sat": None,          # None => the saturated-wash test is DISABLED
    "sat_flat_mult": 2.5,
}

TESTS = ("black", "white", "flat", "sat")


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def to_small_rgb(frames: np.ndarray, hw: tuple[int, int] = DFG_GRAY_HW) -> np.ndarray:
    """(N,H,W,3) uint8 -> (N,h,w,3) float32, area-averaged down to `hw` (same as to_small_gray)."""
    import cv2
    h, w = hw
    out = np.empty((len(frames), h, w, 3), np.float32)
    for i, f in enumerate(frames):
        out[i] = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    return out


def ramp_indices(n_frames: int, i0: int, j0: int) -> np.ndarray:
    """The transition window: exactly M1's ramp. Empty -> the whole clip (never happens in D2)."""
    r = np.arange(i0 + 1, j0)
    return r if len(r) else np.arange(n_frames)


def features(clip: np.ndarray, i0: int, j0: int, *, gray: np.ndarray, gray_a: np.ndarray,
             gray_b: np.ndarray, rgb: np.ndarray | None = None, need_sat: bool = True) -> dict:
    """Per-frame DFG features over the transition window.

    `gray`, `gray_a`, `gray_b` are 96x72 float32 grayscales in 0..255 (d2_metrics.to_small_gray)
    of the rendered clip and of the two REAL source streams. `rgb` (96x72x3, 0..255) is only
    needed for the saturated-wash feature and is computed here if absent.
    """
    r = ramp_indices(len(clip), i0, j0)
    g = gray[r] / 255.0
    L = g.reshape(len(r), -1).mean(axis=1)
    S = g.reshape(len(r), -1).std(axis=1)
    La = (gray_a[r] / 255.0).reshape(len(r), -1).mean(axis=1)
    Lb = (gray_b[r] / 255.0).reshape(len(r), -1).mean(axis=1)
    if need_sat:
        c = (to_small_rgb(clip[r]) if rgb is None else rgb[r]) / 255.0
        sat = (c.max(axis=-1) - c.min(axis=-1)).reshape(len(r), -1).mean(axis=1)
    else:
        sat = np.zeros(len(r), np.float32)
    return {
        "i0": int(i0), "j0": int(j0), "n_window": int(len(r)),
        "t": [int(x) for x in r],
        "L": [round(float(x), 5) for x in L],
        "S": [round(float(x), 5) for x in S],
        "sat": [round(float(x), 5) for x in sat],
        "La": [round(float(x), 5) for x in La],
        "Lb": [round(float(x), 5) for x in Lb],
    }


# --------------------------------------------------------------------------
# decision
# --------------------------------------------------------------------------
def evaluate(feat: dict, cfg: dict | None = None) -> dict:
    """Apply a DFG config to a feature record. Returns the per-test flags and the verdict."""
    c = dict(DEFAULT_CONFIG)
    c.update(cfg or {})
    L = np.asarray(feat["L"], np.float64)
    S = np.asarray(feat["S"], np.float64)
    sat = np.asarray(feat["sat"], np.float64)
    La = np.asarray(feat["La"], np.float64)
    Lb = np.asarray(feat["Lb"], np.float64)

    f_black = (L < c["theta_black"]) & (np.minimum(La, Lb) > SRC_DARK_GUARD)
    f_white = (L > c["theta_white"]) & (np.maximum(La, Lb) < SRC_BRIGHT_GUARD)
    f_flat = S < c["theta_flat"]
    if c.get("theta_sat") is None:
        f_sat = np.zeros(len(L), bool)
    else:
        f_sat = (sat > c["theta_sat"]) & (S < c["sat_flat_mult"] * c["theta_flat"])

    flagged = f_black | f_white | f_flat | f_sat
    t = np.asarray(feat["t"])
    return {
        "reject": bool(int(flagged.sum()) >= c["K"]),
        "n_flag": int(flagged.sum()),
        "n_window": int(len(L)),
        "by_test": {"black": int(f_black.sum()), "white": int(f_white.sum()),
                    "flat": int(f_flat.sum()), "sat": int(f_sat.sum())},
        "flag_frames": [int(x) for x in t[flagged]],
        "worst": {"L_min": round(float(L.min()), 5), "L_max": round(float(L.max()), 5),
                  "S_min": round(float(S.min()), 5), "sat_max": round(float(sat.max()), 5)},
        "config": {k: c[k] for k in ("theta_black", "theta_white", "theta_flat", "K",
                                     "theta_sat", "sat_flat_mult")},
    }


def compact(feat: dict, res: dict) -> dict:
    """Rejection-log payload: the verdict plus ONLY the flagged frames' features."""
    idx = {t: i for i, t in enumerate(feat["t"])}
    return {
        "n_flag": res["n_flag"], "n_window": res["n_window"], "by_test": res["by_test"],
        "worst": res["worst"], "window": [feat["i0"], feat["j0"]],
        "frames": [{"t": t, "L": feat["L"][idx[t]], "S": feat["S"][idx[t]],
                    "sat": feat["sat"][idx[t]], "La": feat["La"][idx[t]],
                    "Lb": feat["Lb"][idx[t]]} for t in res["flag_frames"]],
    }
