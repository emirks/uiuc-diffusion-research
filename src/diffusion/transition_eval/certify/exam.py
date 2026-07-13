"""Certification exam (SPEC §6.1, Block A) — the harness's own test, zero
generated videos needed.

Two readouts over the corpus manifest, both computed with IMPORTED deployed
metric code — never reimplemented; pytest guards that the exam's statistic
equals score.py's on the same item:

  R1 clip-level — LOO 1-NN retrieval + Cohen's d for M1a/M1b/M1c (deployed as
      clip-to-clip comparisons), under every contested variant (core-mask
      v2_envelope / v3_sided / all_frames; motion incumbent / decomposed).
  R2 pool-level — LOO class-pool margin classification for M2b (deployed as
      pool comparisons; clip excluded from its own pool). R1 trust does NOT
      transfer to M2b — different estimator; R2 exists because of that.

Grades variants against bars.yaml's pre-registered adoption rule (per-class
sign test on n>=4 classes + regression guard + mask non-degeneracy criterion;
O7 Huber conditional) and refreshes the per-class trust map (recall; M1c also
definedness) the scorer consumes.

Needs GPU only for uncached features/tracks; the standing corpus is fully
cached from prior runs (outputs/eval/cache), so a re-exam is login-node numpy.

Run contract (first execution = part of the v3 certification run):
    python -m diffusion.transition_eval.certify.exam \
        --corpus data/processed/transitions_std121/corpus_manifest.json \
        --cache outputs/eval/cache --out outputs/eval/certification/<version>/exam/

STATUS: R1 machinery implemented; R2 readout + sign-test grading pending —
implemented after bars freeze, before the certification run. UNEXECUTED until
bars.yaml is frozen (O3) — running it earlier would de-register the bars.
"""

from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np

from ..m1_transfer import camera_match, camera_trajectory, object_match
from ..morph import profile_distance
from ..appearance import set_similarity
from ..report import retrieval_eval
from ..s_structure import core_mask_v3


def appearance_distance_matrix(bundles: list[dict], variant: str,
                               sidedness: list[str]) -> np.ndarray:
    """1 - set_similarity on core frames under the chosen core-mask variant."""
    cores = []
    for b, side in zip(bundles, sidedness):
        if variant == "all_frames":
            T = len(b["feats"])
            m = np.ones(T, dtype=bool)
            m[:b["profile"]["n_prefix"]] = False
            m[T - b["profile"]["n_suffix"]:] = False
        elif variant == "v3_sided":
            m, _ = core_mask_v3(b["profile"], side)
        else:  # v2_envelope
            m, _ = core_mask_v3(b["profile"], "twosided")
        cores.append(b["feats"][m])
    n = len(bundles)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        D[i, j] = D[j, i] = 1.0 - set_similarity(cores[i], cores[j])
    return D


def motion_distance_matrices(bundles: list[dict]) -> dict[str, np.ndarray]:
    """Incumbent MFS distance vs decomposed camera/object distances."""
    from ..motion import motion_fidelity  # incumbent (v2)

    n = len(bundles)
    cams = [camera_trajectory(b["tracks"], b["vis"]) for b in bundles]
    D_inc = np.full((n, n), np.nan)
    D_cam = np.full((n, n), np.nan)
    D_obj = np.full((n, n), np.nan)
    for i, j in itertools.combinations(range(n), 2):
        mf = motion_fidelity(bundles[i]["tracks"], bundles[i]["vis"],
                             bundles[j]["tracks"], bundles[j]["vis"])
        D_inc[i, j] = D_inc[j, i] = 1.0 - mf if np.isfinite(mf) else np.nan
        cm = camera_match(cams[i], cams[j])
        D_cam[i, j] = D_cam[j, i] = cm["cam_dtw"] if cm["cam_valid"] else np.nan
        om = object_match(bundles[i]["tracks"], bundles[i]["vis"],
                          bundles[j]["tracks"], bundles[j]["vis"], cams[i], cams[j])
        D_obj[i, j] = D_obj[j, i] = 1.0 - om if np.isfinite(om) else np.nan
    return {"m_incumbent": D_inc, "m1b_camera": D_cam, "m1c_object": D_obj}


def grade_variants(results: dict, bars: dict, strata: dict[str, list[str]]) -> dict:
    """Apply bars.exam.adoption_rule: overall >= incumbent, target stratum
    improvement, no trusted-class regression. Returns per-variant verdicts —
    the piece of certification that decides which variant SHIPS."""
    verdicts = {}
    for name, res in results.items():
        if name.endswith("incumbent") or name.endswith("v2_envelope"):
            continue
        base_key = [k for k in results if k.endswith("incumbent") or k.endswith("v2_envelope")]
        base = results[base_key[0]] if base_key else None
        verdicts[name] = {
            "overall_ok": bool(base is None or res["accuracy_1nn"] >= base["accuracy_1nn"]),
            "stratum": {s: None for s in strata},   # filled from per_class_recall
            "verdict": "PENDING_BARS_FREEZE",
        }
    return verdicts


def run_exam(corpus_manifest: pathlib.Path, cache_dir: pathlib.Path,
             out_dir: pathlib.Path, bars: dict) -> dict:
    """Full exam orchestration. Deliberately unexecuted until bars are frozen —
    the caller (certification driver) asserts bars['frozen'] is True."""
    if not bars.get("frozen"):
        raise RuntimeError(
            "bars.yaml is DRAFT (frozen: false) — freeze the bars at the "
            "health-design session before executing the exam (SPEC §6).")
    raise NotImplementedError(
        "wire: load corpus manifest -> process_video_file per clip (cached) -> "
        "appearance_distance_matrix per variant + motion_distance_matrices -> "
        "retrieval_eval per matrix -> grade_variants -> trust flags + report")
