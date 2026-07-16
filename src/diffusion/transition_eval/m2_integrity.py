"""M2 — Integrity (SPEC §3): did the generation cheat?

Three separate diseases, three separate instruments (never one number):

M2a copy         — reference content replayed. Max sim of GENERATED MID frames
                   (everything outside conditioned windows — near-endpoint bleed
                   is in scope, the exp_056 briefcase lesson) against the
                   REFERENCE's NON-CORE frames (its scenes A/B — content that
                   must never appear). Corpus-free: works for singleton and
                   unseen references.
M2b intrusion    — wrong style imported (smoke into gas). Appearance profile
                   against EVERY corpus class; the margin names the intruder.
                   Replaces v2's `excess`, whose sign moved the WRONG way under
                   intrusion (the intruding class raised mean_max_other).
M2c memorization — training clips regurgitated. Retrieval against the training
                   pool with clip attribution; interpreted ONLY via tier B<->C
                   contrasts (same-class similarity is legitimate per-item).

Every retrieval-type number carries argmax provenance (frame + source) so any
flag is auditable in the viewer. Pure numpy.
"""

from __future__ import annotations

import numpy as np

from .appearance import set_similarity

TAU_COPY = 0.858  # owner-adopted 2026-07-14 (amendment-1 midpoint rule); v4
                  # certification re-derives under the same frozen rule


def mid_mask(T: int, n_prefix: int, n_suffix: int) -> np.ndarray:
    """All frames outside the conditioned windows (n_suffix=0 for prefix-only)."""
    m = np.zeros(T, dtype=bool)
    m[n_prefix:T - n_suffix if n_suffix else T] = True
    return m


def copy_score(gen_feats: np.ndarray, gen_mid: np.ndarray,
               ref_feats: np.ndarray, ref_core: np.ndarray,
               tau: float = TAU_COPY) -> dict:
    """M2a. ref non-core = the demo's own scenes; matching them is unambiguous
    copying. Matching ref CORE frames is the ambiguous gray zone (style vs
    layout-tracking) — that similarity is M1a's business and the base twin
    arbitrates it, so it is deliberately NOT in this score."""
    gen_idx = np.flatnonzero(gen_mid)
    ref_idx = np.flatnonzero(~ref_core)
    if len(gen_idx) == 0 or len(ref_idx) == 0:
        return {"copy_max": float("nan"), "near_copy": None,
                "copy_gen_frame": None, "copy_ref_frame": None}
    S = gen_feats[gen_idx] @ ref_feats[ref_idx].T
    gi, ri = np.unravel_index(int(S.argmax()), S.shape)
    mx = float(S.max())
    return {"copy_max": mx, "near_copy": bool(mx >= tau),
            "copy_gen_frame": int(gen_idx[gi]), "copy_ref_frame": int(ref_idx[ri])}


def intrusion_margin(gen_feats: np.ndarray, gen_core: np.ndarray,
                     class_core_pools: dict[str, np.ndarray],
                     target: str) -> dict:
    """M2b. class_core_pools: class -> concatenated CORE features of its real
    clips. margin = app(target) − best other; negative margin names the
    intruder. Advisory for camera classes (appearance ill-defined there —
    the caller flags by taxonomy, not here)."""
    core = gen_feats[gen_core]
    apps = {c: set_similarity(core, pool)
            for c, pool in class_core_pools.items() if len(pool)}
    if target not in apps:
        return {"app_target": float("nan"), "margin": float("nan"),
                "intruder": None, "apps_top3": []}
    others = {c: v for c, v in apps.items() if c != target}
    best_other = max(others, key=others.get) if others else None
    top3 = sorted(apps.items(), key=lambda kv: -kv[1])[:3]
    return {
        "app_target": float(apps[target]),
        "margin": float(apps[target] - others[best_other]) if best_other else float("nan"),
        "intruder": best_other if best_other and others[best_other] > apps[target] else None,
        "apps_top3": [(c, round(float(v), 4)) for c, v in top3],
    }


def memorization_score(gen_feats: np.ndarray, gen_mid: np.ndarray,
                       train_pools: dict[str, np.ndarray]) -> dict:
    """M2c. train_pools: training clip key -> its frame features. Off-headline
    audit — meaningless without tier metadata joined by the caller."""
    gen_idx = np.flatnonzero(gen_mid)
    if len(gen_idx) == 0 or not train_pools:
        return {"mem_max": float("nan"), "mem_clip": None, "mem_top3": []}
    G = gen_feats[gen_idx]
    per_clip = {}
    for key, F in train_pools.items():
        if len(F):
            per_clip[key] = float((G @ F.T).max())
    if not per_clip:
        return {"mem_max": float("nan"), "mem_clip": None, "mem_top3": []}
    top = sorted(per_clip.items(), key=lambda kv: -kv[1])
    return {"mem_max": top[0][1], "mem_clip": top[0][0],
            "mem_top3": [(k, round(v, 4)) for k, v in top[:3]]}
