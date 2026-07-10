"""M3 — Effect Appearance and M6 — Leakage. Both operate on core frames
(the 'neither endpoint' frames from the morph profile), which isolates the
effect medium (smoke / ravens / water) from endpoint content by construction.

Appearance wants distribution-level similarity to the reference core frames
(symmetric mean-of-max). Leakage wants near-duplicate retrieval (a single max)
against ALL reference frames, contrasted with the same retrieval against
unrelated styles so style-driven similarity doesn't read as leakage.
"""

from __future__ import annotations

import numpy as np


def set_similarity(F1: np.ndarray, F2: np.ndarray) -> float:
    """Symmetric mean-of-max cosine between two L2-normalized feature sets."""
    if len(F1) == 0 or len(F2) == 0:
        return float("nan")
    S = F1 @ F2.T
    return float(0.5 * (S.max(axis=1).mean() + S.max(axis=0).mean()))


def effect_similarity(gen_feats: np.ndarray, gen_core: np.ndarray,
                      ref_feats: np.ndarray, ref_core: np.ndarray) -> float:
    return set_similarity(gen_feats[gen_core], ref_feats[ref_core])


def leakage(gen_core_feats: np.ndarray, target_frames: np.ndarray,
            other_styles_frames: dict[str, np.ndarray]) -> dict:
    """gen core frames retrieved against every frame of the target style's
    reference set vs against each unrelated style's frames. `excess` > 0 means
    the generation is closer to specific target-reference content than generic
    cross-style similarity explains; near-1.0 max_sim_target = near-copy."""
    S = gen_core_feats @ target_frames.T
    gi, ri = np.unravel_index(int(S.argmax()), S.shape)
    others = {name: float((gen_core_feats @ F.T).max())
              for name, F in other_styles_frames.items() if len(F)}
    mean_other = float(np.mean(list(others.values()))) if others else float("nan")
    return {"max_sim_target": float(S.max()), "mean_max_other": mean_other,
            "excess": float(S.max()) - mean_other,
            "argmax_gen_core_frame": int(gi), "argmax_ref_frame": int(ri)}
