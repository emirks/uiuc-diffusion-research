"""Stability checks (SPEC §6.4): a measuring device that doesn't repeat itself
measures nothing.

warm_rerun   — same items scored twice with a hot cache must be bit-identical
               (v2 precedent: twin Slurm jobs agreed to the 7th decimal).
cold_rerun   — cache cleared: per-headline-metric abs delta <= bars tolerance
               (GPU nondeterminism bound, measured not assumed).
anchors      — designated anchor items must reproduce raw metrics within the
               bars anchor_reproduction bound across corpus/version changes
               (exp_057 §4c precedent: ±0.04).

sigma_seed lives in seeds.py; judged jointly at certification.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HEADLINE = ("app_ref", "margin", "copy_max", "cam_dtw", "obj_match",
            "prefix_dino", "suffix_dino", "max_seam_z")


def _rows(path: pathlib.Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in
            (json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip())}


def compare_runs(items_a: pathlib.Path, items_b: pathlib.Path,
                 tolerance: float = 0.0) -> dict:
    """Per-metric max abs delta between two scoring runs of the same manifest.
    tolerance=0.0 -> warm-rerun bit-identity check."""
    A, B = _rows(items_a), _rows(items_b)
    shared = sorted(set(A) & set(B))
    deltas = {}
    for m in HEADLINE:
        ds = [abs(A[i].get(m, np.nan) - B[i].get(m, np.nan))
              for i in shared
              if np.isfinite(A[i].get(m, np.nan)) and np.isfinite(B[i].get(m, np.nan))]
        deltas[m] = float(max(ds)) if ds else None
    worst = max((d for d in deltas.values() if d is not None), default=None)
    return {"n_shared": len(shared), "missing_either": sorted(set(A) ^ set(B)),
            "per_metric_max_abs_delta": deltas, "worst": worst,
            "pass": bool(worst is not None and worst <= tolerance)}
