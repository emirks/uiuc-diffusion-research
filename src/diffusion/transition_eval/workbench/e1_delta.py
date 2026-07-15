"""E1 — the effect-delta vector. THE KILL TEST (RUNBOOK §4.1).

    "Definition: v_clip = mean whitened masked embedding over core frames;
     v_null = same pooling over the clip's rendered lerp; delta = v_clip - v_null.
     One vector per clip. Distance: L2.
     Exam: LOO + d, head-to-head vs m1a__v3_sided on the frozen corpus and splits.
     KILL RULE (verbatim, pre-registered): if delta fails to beat raw M1a on BOTH
     Cohen's d and misretrieved count (pinned: 73/223), the endpoint-normalization
     program is dead at the appearance level. One appendix paragraph, full stop.
     E2/E3 do not run."

THE IDEA. M1a compares masked core frames ABSOLUTELY, so two clips of the same
effect over different content look different, and two clips of different effects
over similar content look alike — the 0.82 content-invariance correlation is that
bill arriving. The delta asks a different question: not "what does this clip look
like" but "what did the EFFECT do to this content, relative to what a degenerate
crossfade of the same endpoints would have done". Subtracting the clip's own
rendered null is meant to cancel the content and leave the effect.

TWO CHOICES PINNED BEFORE THE RUN (advisor consult C1; they cannot be revisited
after seeing a number):

1. v_null is pooled over THE CLIP'S OWN core frame indices, not over the null's
   own S-mask. A lerp's endpoint envelope stays above the core threshold
   throughout its crossfade, so its strict core is empty-to-degenerate by
   construction and core_mask_v3's fallback would pick an arbitrary sliver near
   the envelope minimum. The null is a calibration object FOR THIS CLIP: it
   inherits the clip's structure, and the difference of two means only cancels
   content if both means are taken over the same sigma-support.

2. low_D clips are NOT excluded. §1.2 excludes them from D-NORMALIZED scores, and
   the delta contains no D — no division, no projection. E1 therefore scores all
   223 clips, and coverage stays 1.0 (the incumbent's), so the §7 definedness
   condition is not quietly bought with a shrunken support.

Whitening is AFFINE (W @ (x - mu)), so the ZCA mean cancels exactly in the
difference: delta = W @ (mean_core_raw - mean_null_raw). The whitener still
matters — it is what makes the L2 between two deltas a fair distance rather than
one measured with a bent ruler.
"""

from __future__ import annotations

import numpy as np

from ..s_structure import core_mask_v3
from . import nulls, paths, whitening


def clip_delta(bundle: dict, sidedness: str, null_feats: np.ndarray,
               zca: dict) -> dict:
    """delta = mean(whitened clip core frames) - mean(whitened null, SAME indices)."""
    mask, meta = core_mask_v3(bundle["profile"], sidedness)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {"delta": None, "defined": False, "reason": "empty core mask",
                "n_core": 0, "core_degenerate": meta.get("core_degenerate", False)}
    if idx.max() >= len(null_feats):
        return {"delta": None, "defined": False,
                "reason": f"null has {len(null_feats)} frames, core index "
                          f"{int(idx.max())} out of range",
                "n_core": int(idx.size), "core_degenerate": meta.get("core_degenerate", False)}

    v_clip = whitening.whiten(zca, bundle["feats"][idx]).mean(axis=0)
    v_null = whitening.whiten(zca, null_feats[idx]).mean(axis=0)   # SAME indices
    return {
        "delta": v_clip - v_null,
        "defined": True,
        "reason": None,
        "n_core": int(idx.size),
        "core_degenerate": bool(meta.get("core_degenerate", False)),
        "delta_norm": float(np.linalg.norm(v_clip - v_null)),
        "clip_norm": float(np.linalg.norm(v_clip)),
        "null_norm": float(np.linalg.norm(v_null)),
    }


def corpus_deltas(bundles: list[dict], sidedness: list[str], keys: list[str],
                  zca: dict, cache_dir=None) -> list[dict]:
    cache_dir = cache_dir or paths.WB_CACHE
    out = []
    for b, s, k in zip(bundles, sidedness, keys):
        nf = nulls.load_null_feats(paths.clip_path(k), cache_dir)
        out.append(clip_delta(b, s, nf, zca))
    return out


def distance_matrix(deltas: list[dict]) -> np.ndarray:
    """L2 between delta vectors; NaN where a clip is undefined (§1.5 — the frozen
    kernel reads NaN as 'cannot retrieve' and drops it from coverage)."""
    n = len(deltas)
    D = np.full((n, n), np.nan)
    V = [d["delta"] if d["defined"] else None for d in deltas]
    for i in range(n):
        if V[i] is None:
            continue
        D[i, i] = 0.0
        for j in range(i + 1, n):
            if V[j] is None:
                continue
            D[i, j] = D[j, i] = float(np.linalg.norm(V[i] - V[j]))
    return D


def kill_rule(cand: dict, gates: dict) -> dict:
    """§4.1, verbatim and mechanical.

    The candidate SURVIVES only by beating raw M1a on BOTH Cohen's d and the
    misretrieved count. Failing either is death for the endpoint-normalization
    program at the appearance level; E2 and E3 do not run. Terminal — no rescue
    variant, no threshold adjustment, no second attempt (§9)."""
    must = gates["phase2"]["e1"]["must_beat"]
    d_ok = bool(cand["separation_cohens_d"] > must["cohens_d"])
    m_ok = bool(cand["misretrieved"] < must["misretrieved"])
    return {
        "rule": ("RUNBOOK §4.1: if delta fails to beat raw M1a on BOTH Cohen's d "
                 "and misretrieved count, the endpoint-normalization program is "
                 "dead at the appearance level. One appendix paragraph, full stop. "
                 "E2/E3 do not run."),
        "beats_cohens_d": {"candidate": cand["separation_cohens_d"],
                           "incumbent": must["cohens_d"], "pass": d_ok},
        "beats_misretrieved": {"candidate": cand["misretrieved"],
                               "incumbent": must["misretrieved"], "pass": m_ok},
        "survives": bool(d_ok and m_ok),
        "verdict": "PASS — E2 may run" if (d_ok and m_ok) else
                   "KILL — endpoint-normalization is dead at the appearance level; "
                   "E2/E3 do not run",
    }
