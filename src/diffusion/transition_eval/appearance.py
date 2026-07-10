"""Set-similarity substrate — feeds M1a (appearance-to-reference) and M2b
(per-class attribution pools). The v2 `leakage`/`effect_similarity` functions
were retired by m2_integrity.py (SPEC v3: copy/intrusion/memorization split);
git history holds them."""

from __future__ import annotations

import numpy as np


def set_similarity(F1: np.ndarray, F2: np.ndarray) -> float:
    """Symmetric mean-of-max cosine between two L2-normalized feature sets."""
    if len(F1) == 0 or len(F2) == 0:
        return float("nan")
    S = F1 @ F2.T
    return float(0.5 * (S.max(axis=1).mean() + S.max(axis=0).mean()))
