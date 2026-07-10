"""S — structure substrate v3 (SPEC §3 S).

Sidedness-aware core mask with a flagged, distribution-aware fallback,
replacing v2's one-size-fits-all envelope mask and silent single-frame
fallback. Builds on morph.py's profile (curves unchanged — exam-validated
on two-sided data).

Why sidedness-aware: for one-sided transitions the effect's terminal state
IS endpoint B (the transformed subject / emptied scene), so the v2 envelope
max(â,b̂) excludes the effect's own appearance from the core by construction
— measured collapse to 0–7% core_frac on shadow/wireframe/super_fast_run
(2026-07-10 items.jsonl audit). Departure-from-A alone is the correct
"transition is happening" notion there.

Pure numpy; unit-tested in tests/test_transition_eval_v3.py.
"""

from __future__ import annotations

import numpy as np

CORE_THRESH = 0.5          # semantic: closer to 'unrelated' than to an endpoint
FALLBACK_MIN_FRAMES = 8    # DRAFT (SPEC O3) — below this the fallback engages
FALLBACK_DELTA = 0.05      # DRAFT (SPEC O3) — valley expansion width


def envelope(profile: dict, sidedness: str) -> np.ndarray:
    """Per-frame 'how much is this frame an endpoint' under the item's mode.
    two-sided -> max(â, b̂); one-sided -> â alone (b̂ intentionally ignored
    even when a suffix window exists: conditioning fact ≠ class fact)."""
    if sidedness == "onesided" or profile["b_hat"] is None:
        return profile["a_hat"]
    return np.maximum(profile["a_hat"], profile["b_hat"])


def core_mask_v3(profile: dict, sidedness: str, thresh: float = CORE_THRESH,
                 min_frames: int = FALLBACK_MIN_FRAMES,
                 delta: float = FALLBACK_DELTA) -> tuple[np.ndarray, dict]:
    """(mask, meta). Primary threshold stays ABSOLUTE (a video with no core is
    a finding — crossfade/hold — not a failure). If the strict mask is smaller
    than min_frames, expand to the envelope valley (env <= env_min + delta)
    and flag core_degenerate — an honest, sized mask instead of v2's silent
    argmin frame. Conditioned windows are always excluded."""
    env = envelope(profile, sidedness)
    T = len(env)
    n_pre, n_suf = profile["n_prefix"], profile["n_suffix"]
    window = np.zeros(T, dtype=bool)
    window[n_pre:T - n_suf] = True

    strict = (env < thresh) & window
    meta = {
        "mode": "a_only" if (sidedness == "onesided" or profile["b_hat"] is None) else "envelope",
        "core_frames_strict": int(strict.sum()),
        "core_frac_strict": float(strict.sum() / max(window.sum(), 1)),
        "core_degenerate": False,
    }
    if strict.sum() >= min_frames:
        meta["core_frames"] = int(strict.sum())
        return strict, meta

    mid_env = np.where(window, env, np.inf)
    valley = mid_env <= (mid_env.min() + delta)
    mask = valley & window
    meta["core_degenerate"] = True
    meta["core_frames"] = int(mask.sum())
    return mask, meta


def structure_flags(profile: dict, core_meta: dict) -> dict:
    """The S-block fields every item row carries (trust-flag inputs)."""
    return {
        "cross": float(profile["cross"]),
        "cross_high": bool(profile["cross_high"]),
        "core_mode": core_meta["mode"],
        "core_frames": core_meta["core_frames"],
        "core_frac_strict": core_meta["core_frac_strict"],
        "core_degenerate": core_meta["core_degenerate"],
    }
