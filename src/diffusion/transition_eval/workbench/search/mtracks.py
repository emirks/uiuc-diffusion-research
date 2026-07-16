"""Motion-track substrate + m1b/m1c apples-to-apples exam (CoTracker tracklets).

The appearance search (harness.py) never loads tracks. m1b (camera) and m1c (object)
are built entirely from CoTracker3 tracklets, so this module adds the track substrate
(read warm & read-only via the reproduced cache key <file_key>:tracks:v2, no torch, no
GPU) and rebuilds the two incumbents bit-exact from the deployed construction — the
same base-touch discipline that anchored the m1a search.

Exam: a candidate camera/object distance matrix is scored by the SAME frozen kernel,
and judged on its OWN stratum (macro per-class recall over stratum ∩ n>=4-eligible)
plus full-matrix Cohen's d + hubness — RUNBOOK §3.6 / verdict_vs_incumbent.

  m1b_camera : beat camera-stratum recall 0.346230 AND d 0.519962, pass hubness
  m1c_object : beat object-stratum recall 0.034259 AND d 0.247601, pass hubness
               (the incumbent m1c FAILS hubness — a polygon sink — so clearing the
                gate is itself part of the win)
"""

from __future__ import annotations

import time

import numpy as np

from .. import exam as wb_exam
from .. import paths
from . import harness as H

TRACKS = paths.WB_CACHE / "search" / "tracks_substrate.npz"


def _track_key(clip_key: str) -> str:
    from ...features import file_key
    from ...motion import Tracker
    fk = file_key(paths.clip_path(clip_key), paths.DINO_MODEL, str(paths.FEATURE_SHORT_SIDE))
    return f"{fk}:tracks:{Tracker.CACHE_TAG}"


def load_track_substrate(ctx: dict, rebuild: bool = False) -> dict:
    if TRACKS.exists() and not rebuild:
        z = np.load(TRACKS)
        return {"tracks": z["tracks"], "vis": z["vis"]}
    from ...motion import track_cache_path
    keys = ctx["keys"]
    tr, vs = [], []
    t0 = time.time()
    for k in keys:
        p = track_cache_path(_track_key(k), paths.SHARED_CACHE)
        z = np.load(p)
        tr.append(np.asarray(z["tracks"], dtype=np.float32))
        vs.append(np.asarray(z["vis"], dtype=np.float32))
    tracks = np.stack(tr)                                 # [223,121,800,2]
    vis = np.stack(vs)                                    # [223,121,800]
    TRACKS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TRACKS, tracks=tracks, vis=vis)
    print(f"[tracks] {len(keys)} warm track bundles in {time.time()-t0:.1f}s -> {tracks.shape}")
    return {"tracks": tracks, "vis": vis}


def deployed_motion_matrices(sub: dict, n_jobs: int = 8) -> dict:
    from ...certify.exam import motion_distance_matrices
    bundles = [{"tracks": t, "vis": v} for t, v in zip(sub["tracks"], sub["vis"])]
    return motion_distance_matrices(bundles, n_jobs=n_jobs)


# --- exam (stratum-judged) ----------------------------------------------------

def motion_report(ctx: dict, name: str, D: np.ndarray, stratum: str,
                  reasons=None, quiet: bool = False) -> dict:
    """Score a camera/object candidate; report the JUDGED stratum recall + d +
    coverage + hubness + verdict vs the pinned incumbent target."""
    r = wb_exam.evaluate(name, np.asarray(D, float), list(ctx["keys"]), ctx["labels"],
                         ctx["gates"], ctx["corpus_facts"], reasons=reasons, stratum=stratum)
    tgt = ctx["gates"]["stratum_targets"][stratum]
    d_inc = ctx["gates"]["baselines"][tgt["incumbent"]]["cohens_d"]
    sr = r["stratum_recalls"][stratum]["value"]
    beats_recall = np.isfinite(sr) and sr > tgt["value"]
    beats_d = r["separation_cohens_d"] > d_inc
    hub = r["hubness"]["pass"]
    win = bool(beats_recall and beats_d and hub)
    if not quiet:
        print(f"{name:30s} {stratum}-recall {sr:.4f} (>{tgt['value']:.4f}? {beats_recall}) "
              f" d {r['separation_cohens_d']:.4f} (>{d_inc:.4f}? {beats_d})  "
              f"cov {r['coverage']:.3f}  hub {'PASS' if hub else 'FAIL'}  "
              f"=> {'WIN' if win else 'no'}")
    return {"result": r, "stratum_recall": float(sr) if np.isfinite(sr) else float("nan"),
            "cohens_d": r["separation_cohens_d"], "coverage": r["coverage"],
            "hubness_pass": hub, "beats_recall": bool(beats_recall),
            "beats_d": bool(beats_d), "win": win,
            "target_recall": tgt["value"], "target_d": d_inc}


def verify(ctx: dict, sub: dict) -> dict:
    """Base touch: rebuild m1b/m1c from tracks with deployed code, prove bit-exact."""
    mats = deployed_motion_matrices(sub)
    frozen = np.load(ctx["npz_frozen"])
    out = {}
    for key in ("m1b_camera", "m1c_object", "m_incumbent"):
        D = mats[key]
        fz = frozen[key]
        both = np.isfinite(D) & np.isfinite(fz)
        delta = float(np.abs(D[both] - fz[both]).max()) if both.any() else float("nan")
        nan_match = bool((np.isnan(D) == np.isnan(fz)).all())
        out[key] = {"D": D, "max_abs_delta": delta, "nan_pattern_match": nan_match}
        print(f"[verify] {key}: max|Δ| {delta:.2e}  nan-pattern-match {nan_match}")
    return out


if __name__ == "__main__":
    ctx = H.load_context()
    sub = load_track_substrate(ctx)
    v = verify(ctx, sub)
    print("\n--- incumbents through the stratum exam ---")
    motion_report(ctx, "m1b_camera (rebuilt)", v["m1b_camera"]["D"], "camera")
    motion_report(ctx, "m1c_object (rebuilt)", v["m1c_object"]["D"], "object")
