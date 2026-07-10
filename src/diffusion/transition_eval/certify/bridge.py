"""Bridge check (SPEC §6 check 4): rescore the archived exp_056–058 items
under v3 and grade two PRE-REGISTERED predictions against the stored v2 rows.

B1 continuity  — on TWO-SIDED items the v3 core mask is the same envelope rule
    v2 used, so v3 app_ref must rank-correlate strongly with v2
    appearance_best (Spearman >= bars.bridge.twosided_spearman_min),
    computed over items non-degenerate under both versions.
B2 degeneracy collapse — the claim that motivated v3: on ONE-SIDED items the
    v2 envelope core collapsed (data-confirmed, core_frac ~ 1 frame); the v3
    sided mask must bring the core_degenerate rate down to
    <= bars.bridge.onesided_degenerate_max. The v2 rate is recomputed here
    from the archived rows (core_frac proxy) and recorded as context.

This doubles as v3 score.py's first-real-data shakedown: the v3 rescore must
be COMPLETE (every archived item scored, no crashes, flags computable) before
either prediction is graded.

v2 rows come from the archived items.jsonl (exp_056/057/058); v3 rows from
score.py rerun on the same manifests. Field mapping is explicit below — the
two schemas share item_id.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

# An archived v2 row whose core_frac is at or below ~1 frame of the ~104
# scoreable frames had the collapsed single-frame fallback core.
V2_DEGENERATE_CORE_FRAC = 0.015


def _rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation, numpy-only (average ranks for ties)."""
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(1, len(v) + 1)
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def compare_v2_v3(v2_items: list[pathlib.Path], v3_items: list[pathlib.Path],
                  spearman_min: float, degenerate_max: float) -> dict:
    """Grade B1/B2. v2/v3 rows are matched by item_id across all given files."""
    v2 = {r["item_id"]: r for p in v2_items for r in _rows(p)}
    v3 = {r["item_id"]: r for p in v3_items for r in _rows(p)}
    shared = sorted(set(v2) & set(v3))

    two = [i for i in shared if v3[i].get("sidedness") == "twosided"]
    one = [i for i in shared if v3[i].get("sidedness") == "onesided"]

    # B1: continuity on two-sided, non-degenerate under both versions
    pairs = [(v2[i]["appearance_best"], v3[i]["app_ref"]) for i in two
             if np.isfinite(v2[i].get("appearance_best", np.nan))
             and np.isfinite(v3[i].get("app_ref", np.nan))
             and v2[i].get("scalar_core_frac", 1.0) > V2_DEGENERATE_CORE_FRAC
             and not v3[i].get("core_degenerate", False)]
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 5 else float("nan")

    # B2: one-sided degeneracy rate, v2 proxy vs v3 flag
    v2_degen = [v2[i].get("scalar_core_frac", 1.0) <= V2_DEGENERATE_CORE_FRAC for i in one]
    v3_degen = [bool(v3[i].get("core_degenerate", False)) for i in one]
    v2_rate = float(np.mean(v2_degen)) if v2_degen else float("nan")
    v3_rate = float(np.mean(v3_degen)) if v3_degen else float("nan")

    return {
        "n_v2": len(v2), "n_v3": len(v3), "n_shared": len(shared),
        "missing_in_v3": sorted(set(v2) - set(v3)),
        "b1_continuity": {"n_pairs": len(pairs), "spearman": rho,
                          "bar": spearman_min,
                          "pass": bool(np.isfinite(rho) and rho >= spearman_min)},
        "b2_degeneracy": {"n_onesided": len(one),
                          "v2_rate_proxy": v2_rate, "v3_rate": v3_rate,
                          "bar": degenerate_max,
                          "pass": bool(np.isfinite(v3_rate) and v3_rate <= degenerate_max)},
    }
