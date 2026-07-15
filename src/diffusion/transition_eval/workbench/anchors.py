"""Endpoint anchors, chord length, min-D guard (RUNBOOK §1.2).

    "e_A, e_B = mean whitened embedding of the flanking stable frames outside the
     S-mask (never single frames). Chord length D = ||e_B - e_A||. Persist the
     corpus D distribution. Min-D guard: floor at the 5th percentile of the
     corpus D distribution. Below floor -> clip flagged low_D, excluded from
     normalized scores, never zeroed."

ANCHOR CONVENTION (pinned; advisor consult C1). The anchors are the DEPLOYED
certify.probes.endpoint_vecs, whitened afterwards — OPERATIONS §5 pins that
function by name. endpoint_vecs means the conditioned windows (the first
n_prefix and last n_suffix frames), which ARE the flanking stable frames outside
the S-mask: core_mask_v3 excludes the conditioned windows by construction. It
L2-renormalizes the raw mean before returning, so "endpoint_vecs then whiten" is
not identical to "mean of whitened frames" — the renorm is nonlinear. Both
readings are defensible; this one is chosen because endpoint_vecs is the
endpoint definition the certified instrument uses EVERYWHERE it reasons about
endpoints (endpoint_distance, bar-pair selection, the content-invariance audit),
and a workbench anchor defined differently would silently decouple the two.

The renorm cancels in the E1 delta (no anchors in that formula) and is
self-consistent inside the min-D guard (the 5th percentile is taken over the
same D distribution it gates). It reaches only E2's projection coordinates. The
alternative convention is computed and persisted as a DIAGNOSTIC column
(d_chord_norenorm) so the guard's insensitivity is auditable — it is never used
to score anything.

LOW-D SCOPE (pinned; advisor consult C1): low_D excludes a clip from
D-NORMALIZED scores — E2's a_hat/b_hat/m~ channels and E3's Gram, where D
actually appears in a denominator. E1's delta contains no D (no division, no
projection), so E1 scores every clip. The flag is computed here, at cache time,
from corpus-only facts, and recorded; it never silently drops a clip from a
score that does not divide by D.
"""

from __future__ import annotations

import numpy as np

from ..certify.probes import endpoint_vecs
from . import whitening

MIN_D_PERCENTILE = 5.0      # frozen (§1.2)


def raw_anchors(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Deployed endpoint definition, unwhitened."""
    return endpoint_vecs(bundle)


def _mean_frames_norenorm(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic alternative: plain means of the conditioned windows, NO L2
    renorm (the other reading of §1.2). Never used for scoring."""
    f = bundle["feats"]
    n_pre, n_suf = bundle["profile"]["n_prefix"], bundle["profile"]["n_suffix"]
    eA = f[:n_pre].mean(axis=0)
    eB = f[-n_suf:].mean(axis=0) if n_suf else eA
    return eA, eB


def clip_anchors(bundle: dict, zca: dict) -> dict:
    """Whitened e_A / e_B and the chord for one clip."""
    eA_raw, eB_raw = raw_anchors(bundle)
    eA, eB = whitening.whiten(zca, np.stack([eA_raw, eB_raw]))
    chord = eB - eA
    D = float(np.linalg.norm(chord))

    aA, aB = _mean_frames_norenorm(bundle)
    wA, wB = whitening.whiten(zca, np.stack([aA, aB]))
    D_alt = float(np.linalg.norm(wB - wA))

    return {"e_A": eA, "e_B": eB, "chord": chord, "D": D,
            "D_norenorm_diagnostic": D_alt}


def corpus_anchors(bundles: list[dict], zca: dict,
                   min_d_percentile: float = MIN_D_PERCENTILE) -> dict:
    """All 223 anchors + the corpus D distribution + the frozen min-D floor.

    The floor is a corpus-only calibration (no candidate has run) and is frozen
    into the anchors artifact; low_D clips are FLAGGED, never zeroed and never
    silently dropped (§1.2, §1.5)."""
    per_clip = [clip_anchors(b, zca) for b in bundles]
    Ds = np.array([a["D"] for a in per_clip])
    floor = float(np.percentile(Ds, min_d_percentile))
    low_d = Ds < floor
    return {
        "e_A": np.stack([a["e_A"] for a in per_clip]),
        "e_B": np.stack([a["e_B"] for a in per_clip]),
        "D": Ds,
        "D_norenorm_diagnostic": np.array([a["D_norenorm_diagnostic"] for a in per_clip]),
        "min_d_floor": np.float64(floor),
        "min_d_percentile": np.float64(min_d_percentile),
        "low_D": low_d,
        "n_low_D": np.int64(int(low_d.sum())),
    }


def endpoint_progress(feats_w: np.ndarray, eA: np.ndarray, eB: np.ndarray) -> dict:
    """E0/E2 coordinates (§4.2/§4.3): where each whitened frame sits relative to
    the clip's own endpoint chord.

      a_hat(sigma) = <f - e_A, u> / D        (progress along the chord, u = chord/D)
      b_hat(sigma) = ||f - e_B|| / D         (distance from the far endpoint, in chord units)
      rho(sigma)   = (f - e_A) - a_hat * D * u   (the excursion OFF the chord)
      m(sigma)     = ||rho|| / D             (residual magnitude, in chord units)

    Everything is expressed in units of that clip's own chord, which is what
    makes the coordinates comparable across clips with different content — the
    whole point of endpoint normalization. D enters every denominator here, which
    is why low_D clips are excluded from these channels."""
    chord = eB - eA
    D = float(np.linalg.norm(chord))
    if D < 1e-12:
        n = len(feats_w)
        nan = np.full(n, np.nan)
        return {"a_hat": nan, "b_hat": nan, "m": nan,
                "rho": np.full_like(feats_w, np.nan), "D": D}
    u = chord / D
    rel = feats_w - eA
    a_hat = rel @ u / D
    rho = rel - np.outer(a_hat * D, u)
    m = np.linalg.norm(rho, axis=1) / D
    b_hat = np.linalg.norm(feats_w - eB, axis=1) / D
    return {"a_hat": a_hat, "b_hat": b_hat, "m": m, "rho": rho, "D": D}
