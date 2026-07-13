"""Constructed-truth probes (SPEC §6.2, Block B) — items whose correct score is
KNOWN by construction, so the metrics are graded against truth, not vibes.

Roster (bars.yaml `probes`):
  siblings      — all within-class pairs scored (descriptive + content-invariance
                  audit); hard bars only on the max-endpoint-distance pair per
                  class (bar 2: M1a > class control AND M2a silent).
  controls      — lerp (two-sided) / static-hold (one-sided) arms: M1a floor +
                  core_degenerate/timing flags (bar 3).
  copy splices  — reference NON-CORE frames into gen-mid (core-frame splices sit
                  outside M2a's comparison pool and would fail spuriously);
                  verbatim + ONE pinned perturbation level (re-rendered-copy
                  proxy); honest set = sibling M2a distribution; tau_copy :=
                  gap midpoint, set here, tested in Block C (bar 4).
  reversal      — reversed-reference M1b drop, on pre-enumerated reversal-
                  sensitive camera pairs only (bar 5).
  m3 panel      — endpoint-swap (true prefix beats a wrong-CLASS prefix) +
                  hard-cut (constructed cut must fire max_seam_z) (bar 6).

Killed by the 2026-07-13 design review (SPEC changelog draft.6): cross-label
probe (the exam's pool-level readout is the same estimator on unimpeachable
truth; on generations it is circular), self-memorization probe (a unit test),
enforcement probes (pytest territory).

All builders emit standard v3 eval manifests + probe videos — probes are scored
by the SAME score.py as real items (a probe path through special code would
certify the special code, not the instrument).

STATUS: splice builder implemented (perturbation level pending); sibling /
reversal / endpoint-swap / hard-cut builders pending — implemented after bars
freeze, before the certification run.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from ..video_io import load_frames, resize_cover_crop, write_video


def build_copy_splice(gen_path: pathlib.Path, ref_path: pathlib.Path,
                      out_path: pathlib.Path, n_prefix: int = 9, n_suffix: int = 8,
                      short_side: int = 256) -> pathlib.Path:
    """Honest generation with its mid-segment replaced by the reference's —
    a ground-truth copy that keeps the original conditioned windows intact."""
    gen, fps = load_frames(gen_path, short_side=short_side)
    ref, _ = load_frames(ref_path, short_side=short_side)
    T = len(gen)
    mid_lo, mid_hi = n_prefix, T - n_suffix
    n_mid = mid_hi - mid_lo
    r_lo = max(0, (len(ref) - n_mid) // 2)
    seg = ref[r_lo:r_lo + n_mid]
    if len(seg) < n_mid:  # short reference: loop-pad, still literal ref content
        seg = np.concatenate([seg, seg[: n_mid - len(seg)]])
    seg = resize_cover_crop(seg, gen.shape[1], gen.shape[2])
    spliced = np.concatenate([gen[:mid_lo], seg, gen[mid_hi:]])
    write_video(spliced, out_path, fps=fps)
    return out_path


def grade_copy_probes(rows: list[dict], honest_rows: list[dict], tau: float) -> dict:
    splice_scores = [r["copy_max"] for r in rows if np.isfinite(r.get("copy_max", np.nan))]
    honest_scores = [r["copy_max"] for r in honest_rows if np.isfinite(r.get("copy_max", np.nan))]
    return {
        "splice_min": min(splice_scores) if splice_scores else None,
        "honest_max": max(honest_scores) if honest_scores else None,
        "all_splices_flagged": bool(splice_scores and min(splice_scores) >= tau),
        "gap": (min(splice_scores) - max(honest_scores))
               if splice_scores and honest_scores else None,
        "tau_recalibrated": (0.5 * (min(splice_scores) + max(honest_scores)))
                            if splice_scores and honest_scores else None,
    }
