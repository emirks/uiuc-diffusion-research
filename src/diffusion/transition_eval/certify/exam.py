"""Certification exam — the harness's own test, zero generated videos needed.

Ports exp_052's run_validation.py to v3: for every real clip in the corpus
manifest, build S/M1 features under EVERY contested variant (core-mask
v2_envelope / v3_sided / all_frames; motion incumbent / decomposed), compute
pairwise distance matrices, run LOO 1-NN retrieval + Cohen's d per metric per
variant, and grade variants against bars.yaml's adoption rule. Refreshes the
per-class trust flags the scorer consumes.

Needs GPU only for uncached features/tracks; the standing corpus is fully
cached from prior runs (outputs/eval/cache), so a re-exam is login-node numpy.

Run contract (first execution = part of the v3 certification run):
    python -m diffusion.transition_eval.certify.exam \
        --corpus data/processed/transitions_std121/corpus_manifest.json \
        --cache outputs/eval/cache --out outputs/eval/certification/<version>/exam/

STATUS: orchestration complete; UNEXECUTED until bars.yaml is frozen (O3) —
running it earlier would de-register the bars.
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


def motion_distance_matrices(bundles: list[dict], weighting: str = "trim2",
                             ) -> dict[str, np.ndarray]:
    """Incumbent MFS distance vs decomposed camera/object distances.
    `weighting` selects the O7 M1b fit variant (trim2 incumbent / huber)."""
    from ..motion import motion_fidelity  # incumbent (v2)

    n = len(bundles)
    cams = [camera_trajectory(b["tracks"], b["vis"], weighting=weighting)
            for b in bundles]
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


def timing_distance_matrices(bundles: list[dict]) -> dict[str, np.ndarray]:
    """M1d/O8 roster: profile DTW without vs with the Δ-novelty channel."""
    n = len(bundles)
    D_base = np.zeros((n, n))
    D_nov = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        pi, pj = bundles[i]["profile"], bundles[j]["profile"]
        D_base[i, j] = D_base[j, i] = profile_distance(pi, pj)["dtw"]
        D_nov[i, j] = D_nov[j, i] = profile_distance(pi, pj, use_novelty=True)["dtw"]
    return {"timing_incumbent": D_base, "timing_novelty": D_nov}


def _stratum_recall(res: dict, classes: list[str]) -> float:
    """Mean per-class recall over a stratum (NaN if none present)."""
    rec = res["per_class_recall"]
    vals = [rec[c] for c in classes if c in rec]
    return float(np.mean(vals)) if vals else float("nan")


def grade_variant(challenger: dict, incumbent: dict, stratum_classes: list[str],
                  bars_exam: dict) -> dict:
    """Apply bars.exam.adoption_rule to one challenger/incumbent pair:
    overall >= incumbent, target-stratum recall better by the frozen margin,
    no incumbent-trusted class dropping below trust_min. This is the piece of
    certification that decides which variant SHIPS."""
    margin = float(bars_exam.get("target_stratum_margin", 0.05))
    trust_min = float(bars_exam.get("trust_min_recall", 0.5))
    inc_rec, ch_rec = incumbent["per_class_recall"], challenger["per_class_recall"]
    overall_ok = bool(challenger["accuracy_1nn"] >= incumbent["accuracy_1nn"])
    s_inc = _stratum_recall(incumbent, stratum_classes)
    s_ch = _stratum_recall(challenger, stratum_classes)
    stratum_ok = bool(np.isfinite(s_ch) and np.isfinite(s_inc)
                      and s_ch >= s_inc + margin)
    regressed = sorted(c for c, r in inc_rec.items()
                       if r >= trust_min and ch_rec.get(c, 0.0) < trust_min)
    verdict = "ADOPT" if (overall_ok and stratum_ok and not regressed) else "KEEP_INCUMBENT"
    return {"overall_ok": overall_ok,
            "incumbent_accuracy": incumbent["accuracy_1nn"],
            "challenger_accuracy": challenger["accuracy_1nn"],
            "stratum_recall_incumbent": s_inc, "stratum_recall_challenger": s_ch,
            "stratum_ok": stratum_ok, "regressed_classes": regressed,
            "verdict": verdict}


def _testable_accuracy(res: dict, counts: dict[str, int]) -> float:
    """Query accuracy excluding singleton classes (untestable by construction:
    a lone clip has no same-class neighbor). Singletons stay in the matrix as
    distractors; they just don't count against the instrument."""
    num = sum(res["per_class_recall"].get(c, 0.0) * n
              for c, n in counts.items() if n >= 2)
    den = sum(n for n in counts.values() if n >= 2)
    return float(num / den) if den else float("nan")


def _retrieval(D: np.ndarray, labels: list[str], counts: dict[str, int]) -> dict:
    """retrieval_eval on a matrix that may hold NaNs (untrackable pairs):
    NaN -> +inf distance (never retrieved), coverage recorded."""
    Dc = D.copy()
    nan_frac = float(np.isnan(Dc[np.triu_indices(len(Dc), 1)]).mean())
    Dc[np.isnan(Dc)] = np.inf
    res = retrieval_eval(Dc, labels)
    res["pair_coverage"] = 1.0 - nan_frac
    res["accuracy_testable"] = _testable_accuracy(res, counts)
    return res


def run_exam(corpus_manifest: pathlib.Path, cache_dir: pathlib.Path,
             out_dir: pathlib.Path, bars: dict, corpus_root: pathlib.Path | None = None,
             device: str = "cpu") -> dict:
    """Full exam orchestration (SPEC §6 check 1). Refuses unfrozen bars —
    running against draft bars would de-register them. With a warm cache this
    is login-node numpy; models load only on cache misses."""
    if not bars.get("frozen"):
        raise RuntimeError(
            "bars.yaml is DRAFT (frozen: false) — freeze the bars at the "
            "health-design session before executing the exam (SPEC §6).")

    from ..features import DinoExtractor
    from ..motion import Tracker
    from ..pipeline import process_video_file

    corpus = json.loads(pathlib.Path(corpus_manifest).read_text())
    root = pathlib.Path(corpus_root) if corpus_root else pathlib.Path(corpus_manifest).parent
    clip_keys = sorted(corpus["clips"])
    labels = [k.split("/")[0] for k in clip_keys]
    sidedness = [corpus["classes"][c]["sidedness"] for c in labels]
    counts = {c: labels.count(c) for c in set(labels)}

    extractor = DinoExtractor(device=device)
    tracker = Tracker(device=device)
    bundles = []
    for k in clip_keys:
        b, _frames = process_video_file(root / k, cache_dir, extractor, tracker)
        bundles.append(b)

    # --- distance matrices, every contested variant, one feature pass -------
    matrices: dict[str, np.ndarray] = {}
    for variant in bars["exam"]["variants"]["core_mask"]:
        matrices[f"m1a__{variant}"] = appearance_distance_matrix(bundles, variant, sidedness)
    for weighting in bars["exam"]["variants"].get("m1b_weighting", ["trim2"]):
        for name, D in motion_distance_matrices(bundles, weighting=weighting).items():
            matrices[f"{name}__{weighting}"] = D
    matrices.update(timing_distance_matrices(bundles))

    results = {name: _retrieval(D, labels, counts) for name, D in matrices.items()}

    # --- adoption verdicts (challenger vs incumbent, frozen margins) --------
    onesided = sorted(c for c, m in corpus["classes"].items() if m["sidedness"] == "onesided")
    camera = sorted(c for c, m in corpus["classes"].items() if "camera" in m.get("tags", []))
    objectc = sorted(c for c, m in corpus["classes"].items() if "object" in m.get("tags", []))
    verdicts = {
        "core_mask_v3_sided": grade_variant(
            results["m1a__v3_sided"], results["m1a__v2_envelope"], onesided, bars["exam"]),
        "motion_decomposed_m1b": grade_variant(
            results["m1b_camera__trim2"], results["m_incumbent__trim2"], camera, bars["exam"]),
        "motion_decomposed_m1c": grade_variant(
            results["m1c_object__trim2"], results["m_incumbent__trim2"], objectc, bars["exam"]),
        "timing_novelty": grade_variant(
            results["timing_novelty"], results["timing_incumbent"], onesided, bars["exam"]),
    }
    if "huber" in bars["exam"]["variants"].get("m1b_weighting", []):
        verdicts["m1b_weighting_huber"] = grade_variant(
            results["m1b_camera__huber"], results["m1b_camera__trim2"], camera, bars["exam"])

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "distance_matrices.npz",
                        labels=np.array(labels), **matrices)
    report = {"n_clips": len(clip_keys), "n_classes": len(counts),
              "chance": max(counts.values()) / len(clip_keys),
              "singletons": sorted(c for c, n in counts.items() if n == 1),
              "results": results, "verdicts": verdicts}
    (out_dir / "exam_results.json").write_text(json.dumps(report, indent=2, default=float))
    return report
