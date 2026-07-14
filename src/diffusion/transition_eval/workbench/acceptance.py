"""Phase 1 acceptance tests — constructed truth (RUNBOOK §3.4).

    "1. Reversal probe (inherited from old Bar 5): time-reversed reference must be
        distinguished from the original — camera parameter trajectories negate;
        descriptor distance to reversed self must exceed the median within-class
        distance. (Known blind spot to design against, recorded in
        certify/probes.py: per-channel z-norm makes time-antisymmetric
        trajectories self-identical — the probe must check parameter negation,
        not only descriptor distance.)
     2. Injected-trajectory recovery: apply known synthetic pans/zooms/rotations
        to static clips; recovered parameter trajectories must match ground truth
        (per-parameter correlation >= 0.9 and relative amplitude error <= 10%).

     Failure of either = fix or stop; the exam is not run on a metric that fails
     constructed truth."

These are the tests that decide whether M1b_flow measures CAMERA MOTION at all,
as opposed to producing self-consistent numbers. A metric can retrieve well and
still be measuring nothing: the incumbent M1c retrieved polygon 58% of the time.
Constructed truth is the only place where we know the answer independently of the
metric, which is why §3.4 puts it BEFORE the exam and makes failure terminal.

THE BLIND SPOT, CONCRETELY. Under time reversal a similarity trajectory negates:
if the forward step is p -> s*R(theta)*p + t, the backward step is its inverse,
with log-scale and rotation negated. But the descriptor z-scores each channel over
the corpus, and an L2 distance between a trajectory and its own negation is NOT
zero — yet a purely antisymmetric trajectory resampled by ARC LENGTH can traverse
the same path in reverse and land arbitrarily close to itself. So a descriptor-only
reversal check can pass while the metric is direction-blind. This module therefore
grades reversal on BOTH: the parameter-negation correlation (does the fit actually
see the sign flip?) AND the descriptor distance.
"""

from __future__ import annotations

import numpy as np

from . import curves, m1b_flow

CORR_MIN = 0.90        # §3.4 injected-trajectory: per-parameter correlation
AMP_ERR_MAX = 0.10     # §3.4 injected-trajectory: relative amplitude error
NEG_CORR_MIN = 0.90    # reversal: correlation with the NEGATED trajectory


# --- similarity algebra (centered pixel coordinates) --------------------------

def sim_matrix(tx: float, ty: float, log_s: float, theta: float) -> np.ndarray:
    """3x3 similarity in CENTERED coordinates: p -> s*R(theta)*p + t."""
    s = np.exp(log_s)
    c, sn = np.cos(theta), np.sin(theta)
    return np.array([[s * c, -s * sn, tx],
                     [s * sn, s * c, ty],
                     [0.0, 0.0, 1.0]])


def params_of(M: np.ndarray) -> np.ndarray:
    """Inverse of sim_matrix -> (tx, ty, log_s, theta), the m1b parameterization."""
    a, b = M[0, 0], M[1, 0]
    s = float(np.hypot(a, b))
    return np.array([float(M[0, 2]), float(M[1, 2]),
                     float(np.log(s)) if s > 1e-12 else np.nan,
                     float(np.arctan2(b, a))])


def to_cv2_affine(M: np.ndarray, h: int, w: int) -> np.ndarray:
    """Centered-coordinate similarity -> cv2's 2x3 pixel-coordinate affine."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    T = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    Tinv = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    return (Tinv @ M @ T)[:2]


# --- injected trajectories ----------------------------------------------------

def trajectory(kind: str, T: int, amp: float) -> np.ndarray:
    """Ground-truth CUMULATIVE camera pose per frame [T, 4] = (tx, ty, log_s, theta).

    EASE-IN-OUT, not constant velocity. This is not decoration: a constant-velocity
    pan has a CONSTANT per-pair parameter (every step is the same step), and a
    correlation against a constant is undefined — the grader would silently decline
    to grade the very channel under test. An eased profile makes the per-pair
    parameters vary over time, which is both what a real camera move does and what
    makes "the recovered trajectory tracks the true one" a statement with content.
    """
    u = np.linspace(0.0, 1.0, T)
    u = 0.5 * (1.0 - np.cos(np.pi * u))       # ease-in-out: velocity varies
    z = np.zeros(T)
    if kind == "pan_x":
        return np.stack([amp * u, z, z, z], axis=1)
    if kind == "pan_y":
        return np.stack([z, amp * u, z, z], axis=1)
    if kind == "zoom":
        return np.stack([z, z, amp * u, z], axis=1)
    if kind == "rotate":
        return np.stack([z, z, z, amp * u], axis=1)
    if kind == "pan_zoom":                       # a compound move, still known exactly
        return np.stack([amp * u, 0.5 * amp * u, 0.3 * amp * u, z], axis=1)
    raise ValueError(f"unknown trajectory {kind}")


def relative_params(cum: np.ndarray) -> np.ndarray:
    """Cumulative poses [T,4] -> the per-PAIR relative similarity [T-1,4].

    A point at q in warped frame t sits at M_{t+1} M_t^{-1} q in frame t+1, so the
    relative transform — not the cumulative pose — is what the flow between the two
    frames encodes, and therefore what m1b_flow must recover."""
    out = []
    for t in range(len(cum) - 1):
        M0 = sim_matrix(*cum[t])
        M1 = sim_matrix(*cum[t + 1])
        out.append(params_of(M1 @ np.linalg.inv(M0)))
    return np.stack(out)


def warp_frames(frames: np.ndarray, cum: np.ndarray) -> np.ndarray:
    """Apply the cumulative trajectory to a clip -> a synthetic camera move.

    Applied to a STATIC clip, the only motion in the result is the injected one,
    so the ground truth is exact rather than approximate."""
    import cv2

    h, w = frames.shape[1:3]
    out = []
    for t in range(len(cum)):
        A = to_cv2_affine(sim_matrix(*cum[t]), h, w)
        out.append(cv2.warpAffine(frames[min(t, len(frames) - 1)], A, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT))
    return np.stack(out)


def grade_injection(recovered: np.ndarray, truth: np.ndarray,
                    defined: np.ndarray) -> dict:
    """§3.4 test 2, graded per parameter against the frozen thresholds.

    Correlation says the SHAPE of the recovered trajectory tracks the truth;
    relative amplitude error says the SCALE does too. Both are required — a metric
    that recovers a perfectly-correlated trajectory at half the true amplitude has
    not recovered the camera."""
    ok = defined[:len(truth)]
    rows = {}
    for i, name in enumerate(m1b_flow.PARAM_NAMES):
        r, t = recovered[:len(truth), i][ok], truth[ok, i]
        if len(t) < 3 or np.std(t) < 1e-12:
            # a channel the trajectory never moves: nothing to correlate. It is
            # not graded, and it is not silently passed either.
            rows[name] = {"corr": None, "amp_err": None, "graded": False,
                          "moved": False}
            continue
        corr = float(np.corrcoef(r, t)[0, 1]) if np.std(r) > 1e-12 else 0.0
        amp_r = float(np.abs(r).max())
        amp_t = float(np.abs(t).max())
        amp_err = float(abs(amp_r - amp_t) / max(amp_t, 1e-12))
        rows[name] = {"corr": corr, "amp_err": amp_err, "graded": True, "moved": True,
                      "corr_ok": bool(corr >= CORR_MIN),
                      "amp_ok": bool(amp_err <= AMP_ERR_MAX),
                      "pass": bool(corr >= CORR_MIN and amp_err <= AMP_ERR_MAX)}
    graded = [v for v in rows.values() if v["graded"]]
    return {"params": rows,
            "n_graded": len(graded),
            "pass": bool(graded and all(v["pass"] for v in graded))}


# --- reversal -----------------------------------------------------------------

def expected_reversed_params(params: np.ndarray) -> np.ndarray:
    """What the per-pair parameters of a TIME-REVERSED clip must be.

    Reversing the clip reverses the order of the steps AND inverts each one: the
    step that took frame t to t+1 now takes t+1 to t. The inverse of
    p -> s*R(theta)*p + t is p -> (1/s)*R(-theta)*(p - t), so log-scale and rotation
    NEGATE and the translation maps to -(1/s)R(-theta)t. This is the ground truth
    the fitted reversed trajectory is checked against — the check the RUNBOOK's
    blind-spot note demands."""
    out = []
    for p in params[::-1]:                       # reversed order
        M = sim_matrix(*p)
        out.append(params_of(np.linalg.inv(M)))  # each step inverted
    return np.stack(out)


def grade_reversal(fwd: dict, rev: dict, descriptor_fwd: np.ndarray | None,
                   descriptor_rev: np.ndarray | None,
                   median_within_class: float) -> dict:
    """§3.4 test 1 — graded on BOTH legs.

    Leg A (the one the blind-spot note insists on): do the fitted parameters of the
    reversed clip actually match the NEGATED forward trajectory? If they do not,
    the fit is not seeing direction at all.
    Leg B (the inherited Bar-5 form): is the descriptor distance between a clip and
    its own reversal larger than the median within-class distance? If not, the
    metric cannot tell a clip from its own reverse, and any retrieval it does is
    direction-blind."""
    # Index alignment matters and is easy to get backwards: expect[j] and got[j] are
    # the REVERSED clip's pair j, which is the FORWARD clip's pair (n-1-j) inverted.
    # So the definedness mask must be the forward flags REVERSED, ANDed with the
    # reversed clip's own flags — not the other way round.
    both = fwd["defined"][::-1] & rev["defined"]      # pairs defined in both directions
    expect = expected_reversed_params(fwd["params"])  # what an honest fit must return
    blind = fwd["params"][::-1]                       # what a DIRECTION-BLIND fit returns:
                                                      # same steps, re-ordered, NOT inverted
    got = rev["params"]

    n = min(len(expect), len(got))
    mask = both[:n]
    if mask.sum() < 3:
        return {"parameter_negation": {"pass": False, "reason": "fewer than 3 pairs "
                                       "defined in both directions", "corr": {}},
                "descriptor_distance": {"pass": False},
                "pass": False}

    # LEG A1 — threshold-free and immune to constant channels: is the fitted
    # reversed trajectory closer to the NEGATED prediction than to the
    # direction-blind one? A metric that cannot see direction fails this outright,
    # even when every channel is constant and no correlation is computable.
    d_neg = float(np.linalg.norm(got[:n][mask] - expect[:n][mask]))
    d_blind = float(np.linalg.norm(got[:n][mask] - blind[:n][mask]))
    closer_ok = bool(d_neg < d_blind)

    # LEG A2 — for channels that actually vary, the correlation floor (§3.4's own
    # 0.9; a constant channel has no correlation to speak of and is not graded).
    corrs = {}
    for i, name in enumerate(m1b_flow.PARAM_NAMES):
        e, g = expect[:n, i][mask], got[:n, i][mask]
        if len(e) < 3 or np.std(e) < 1e-12 or np.std(g) < 1e-12:
            corrs[name] = None
            continue
        corrs[name] = float(np.corrcoef(e, g)[0, 1])
    graded = {k: v for k, v in corrs.items() if v is not None}
    corr_ok = all(v >= NEG_CORR_MIN for v in graded.values())   # vacuously true if none vary

    neg_ok = bool(closer_ok and corr_ok)

    d = curves.l2(descriptor_fwd, descriptor_rev)
    dist_ok = bool(np.isfinite(d) and np.isfinite(median_within_class)
                   and d > median_within_class)
    return {
        "parameter_negation": {
            "corr": corrs,
            "n_pairs": int(mask.sum()),
            "min_corr": min(graded.values()) if graded else None,
            "n_channels_graded": len(graded),
            "dist_to_negated": d_neg,
            "dist_to_direction_blind": d_blind,
            "closer_to_negated": closer_ok,
            "threshold": NEG_CORR_MIN,
            "pass": neg_ok,
        },
        "descriptor_distance": {"d_self_vs_reversed": d,
                                "median_within_class": median_within_class,
                                "pass": dist_ok},
        "pass": bool(neg_ok and dist_ok),
    }


def reversal_sensitive(params: np.ndarray, defined: np.ndarray) -> bool:
    """A palindromic move (a pan out and back) is genuinely identical to its own
    reverse — grading it would be grading noise. Sensitivity is enumerated from the
    trajectory itself, before any reversed flow is computed (corpus-only)."""
    if defined.sum() < 8:
        return False
    P = params[defined]
    E = expected_reversed_params(params)[defined[::-1]]
    if len(P) < 3 or len(E) < 3:
        return False
    n = min(len(P), len(E))
    # a clip is reversal-SENSITIVE if its trajectory is not already its own reverse
    return bool(np.linalg.norm(P[:n] - E[:n]) > 1e-6
                and np.abs(P[:n, 2:]).max() > 1e-4)   # some rotation or zoom exists
