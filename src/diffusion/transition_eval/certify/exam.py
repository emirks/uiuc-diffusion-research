"""Certification exam (SPEC §6.1, Block A) — the harness's own test, zero
generated videos needed.

Two readouts over the corpus manifest, both computed with IMPORTED deployed
metric code — never reimplemented; pytest guards that the exam's statistic
equals score.py's on the same item:

  R1 clip-level — LOO 1-NN retrieval + Cohen's d for M1a/M1b/M1c (deployed as
      clip-to-clip comparisons), under every contested variant (core-mask
      v2_envelope / v3_sided / all_frames; motion incumbent / decomposed).
  R2 pool-level — LOO class-pool margin classification for M2b (deployed as
      pool comparisons via m2_integrity.intrusion_margin; the clip is excluded
      from its own class pool). R1 trust does NOT transfer to M2b — different
      estimator; R2 exists because of that.

Adoption (bars.yaml `exam.adoption`, pre-registered, applied mechanically):
one-sided exact binomial sign test on per-class recall over n>=4 classes of
the target stratum, plus a no-trusted-class-regression guard, plus (core mask
only) a real-clip non-degeneracy criterion. O7 conditional: Huber examined
only if camera-stratum recall < trust floor. Motion contingency: if the
decomposition is not adopted, the incumbent MFS ships as the single headline
motion metric (pre-registered fallback).

Needs GPU only for uncached features/tracks; with a warm cache the exam is
CPU numpy (the 223^2/2 motion pair grid takes ~10-20 min single-core).

STATUS: refuses to run until bars.yaml is frozen (O3) — running it earlier
would de-register the bars.
"""

from __future__ import annotations

import itertools
import json
import math
import pathlib

import numpy as np

from ..appearance import set_similarity
from ..m1_transfer import camera_match, camera_trajectory, object_match
from ..m2_integrity import intrusion_margin
from ..report import retrieval_eval
from ..s_structure import core_mask_v3
from . import diagnostics

CORE_VARIANTS = ("v2_envelope", "v3_sided", "all_frames")


# --- variant core masks -----------------------------------------------------------

def variant_core(bundle: dict, sidedness: str, variant: str) -> tuple[np.ndarray, dict]:
    """The three contested core-mask rules, all through deployed core_mask_v3
    (all_frames = the trivial mask: every non-conditioned frame)."""
    if variant == "all_frames":
        T = len(bundle["feats"])
        m = np.zeros(T, dtype=bool)
        m[bundle["profile"]["n_prefix"]:T - bundle["profile"]["n_suffix"]] = True
        return m, {"core_degenerate": False, "mode": "all_frames"}
    if variant == "v3_sided":
        return core_mask_v3(bundle["profile"], sidedness)
    return core_mask_v3(bundle["profile"], "twosided")   # v2_envelope


# --- R1: clip-level distance matrices ----------------------------------------------

def _map_pairs(fn, pairs: list, n_jobs: int | None,
               min_pairs_for_pool: int) -> list:
    """Map fn over independent (i, j) pairs, on a fork pool when the pair count
    justifies it. fn must be a module-level numpy-only function reading
    fork-inherited state; results are identical to the serial path."""
    import concurrent.futures
    import multiprocessing
    import os

    if n_jobs is None:
        n_jobs = min(16, os.cpu_count() or 1)
    if n_jobs > 1 and len(pairs) >= min_pairs_for_pool and hasattr(os, "fork"):
        ctx = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_jobs, mp_context=ctx) as ex:
            return list(ex.map(fn, pairs,
                               chunksize=max(1, len(pairs) // (n_jobs * 8))))
    return [fn(p) for p in pairs]


_APPEARANCE_STATE: dict = {}   # fork-inherited worker state for appearance_distance_matrix


def _appearance_pair(pair: tuple[int, int]) -> float:
    i, j = pair
    cores = _APPEARANCE_STATE["cores"]
    return 1.0 - set_similarity(cores[i], cores[j])


def appearance_distance_matrix(bundles: list[dict], variant: str,
                               sidedness: list[str], n_jobs: int | None = None,
                               min_pairs_for_pool: int = 256) -> np.ndarray:
    """1 - set_similarity on core frames under the chosen core-mask variant.
    Pairwise cells are independent pure numpy — fork-parallel, bit-identical."""
    cores = [b["feats"][variant_core(b, s, variant)[0]]
             for b, s in zip(bundles, sidedness)]
    n = len(bundles)
    pairs = list(itertools.combinations(range(n), 2))
    _APPEARANCE_STATE["cores"] = cores
    try:
        vals = _map_pairs(_appearance_pair, pairs, n_jobs, min_pairs_for_pool)
    finally:
        _APPEARANCE_STATE.clear()
    D = np.zeros((n, n))
    for (i, j), v in zip(pairs, vals):
        D[i, j] = D[j, i] = v
    return D


_MOTION_STATE: dict = {}   # fork-inherited worker state for motion_distance_matrices


def _motion_pair(pair: tuple[int, int]) -> tuple[float, float, float]:
    """One (i, j) cell of the three motion matrices — numpy only, identical
    whether run inline or in a forked worker."""
    from ..motion import motion_fidelity  # incumbent (v2)

    i, j = pair
    tracks, vis, cams = (_MOTION_STATE["tracks"], _MOTION_STATE["vis"],
                         _MOTION_STATE["cams"])
    mf = motion_fidelity(tracks[i], vis[i], tracks[j], vis[j])
    cm = camera_match(cams[i], cams[j])
    om = object_match(tracks[i], vis[i], tracks[j], vis[j], cams[i], cams[j])
    return (1.0 - mf if np.isfinite(mf) else np.nan,
            cm["cam_dtw"] if cm["cam_valid"] else np.nan,
            1.0 - om if np.isfinite(om) else np.nan)


def motion_distance_matrices(bundles: list[dict], n_jobs: int | None = None,
                             min_pairs_for_pool: int = 256) -> dict[str, np.ndarray]:
    """Incumbent MFS distance vs decomposed camera/object distances.

    The pairwise loop is embarrassingly parallel (each cell independent, pure
    numpy) and runs on a fork pool when the pair count justifies it; workers
    never touch torch/CUDA. Results are identical to the serial path."""
    n = len(bundles)
    cams = [camera_trajectory(b["tracks"], b["vis"]) for b in bundles]
    _MOTION_STATE.update(tracks=[b["tracks"] for b in bundles],
                         vis=[b["vis"] for b in bundles], cams=cams)
    pairs = list(itertools.combinations(range(n), 2))
    try:
        results = _map_pairs(_motion_pair, pairs, n_jobs, min_pairs_for_pool)
    finally:
        _MOTION_STATE.clear()
    D_inc = np.full((n, n), np.nan)
    D_cam = np.full((n, n), np.nan)
    D_obj = np.full((n, n), np.nan)
    for (i, j), (d_inc, d_cam, d_obj) in zip(pairs, results):
        D_inc[i, j] = D_inc[j, i] = d_inc
        D_cam[i, j] = D_cam[j, i] = d_cam
        D_obj[i, j] = D_obj[j, i] = d_obj
    return {"m_incumbent": D_inc, "m1b_camera": D_cam, "m1c_object": D_obj}


# --- R2: pool-level margin classification ------------------------------------------

def pool_margin_exam(bundles: list[dict], labels: list[str],
                     sidedness: list[str], variant: str) -> dict:
    """LOO class-pool margin classification with the DEPLOYED M2b statistic.

    For each clip: pools = per-class concatenated core features with THIS
    clip's frames removed from its own class pool; correct <=> margin > 0.
    Uses m2_integrity.intrusion_margin verbatim — the whole point of R2."""
    cores = [b["feats"][variant_core(b, s, variant)[0]]
             for b, s in zip(bundles, sidedness)]
    by_class: dict[str, list[int]] = {}
    for idx, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(idx)

    full_pools = {c: np.concatenate([cores[i] for i in idxs])
                  for c, idxs in by_class.items()}
    rows = []
    for i, (lab, core) in enumerate(zip(labels, cores)):
        others = [cores[j] for j in by_class[lab] if j != i]
        if not others:            # singleton class: LOO pool undefined
            rows.append({"label": lab, "margin": None, "correct": None})
            continue
        pools = dict(full_pools)
        pools[lab] = np.concatenate(others)
        r = intrusion_margin(bundles[i]["feats"],
                             variant_core(bundles[i], sidedness[i], variant)[0],
                             pools, target=lab)
        rows.append({"label": lab, "margin": r["margin"],
                     "intruder": r["intruder"],
                     "correct": bool(np.isfinite(r["margin"]) and r["margin"] > 0)})
    graded = [r for r in rows if r["correct"] is not None]
    per_class = {c: float(np.mean([r["correct"] for r in graded if r["label"] == c]))
                 for c in sorted(set(r["label"] for r in graded))}
    return {"accuracy": float(np.mean([r["correct"] for r in graded])) if graded else None,
            "n_graded": len(graded),
            "n_singleton_excluded": len(rows) - len(graded),
            "per_class_recall": per_class,
            "margins_mean": float(np.nanmean([r["margin"] for r in graded
                                              if r["margin"] is not None])),
            "rows": rows}


# --- adoption machinery -------------------------------------------------------------

def sign_test_p(wins: int, losses: int) -> float:
    """One-sided exact binomial P(X >= wins | n=wins+losses, p=0.5)."""
    m = wins + losses
    if m == 0:
        return 1.0
    return float(sum(math.comb(m, k) for k in range(wins, m + 1)) / 2 ** m)


def class_sign_test(recall_new: dict, recall_old: dict,
                    eligible: set[str]) -> dict:
    """Per-class recall win/loss sign test on the eligible (n>=4) classes."""
    wins = losses = ties = 0
    for c in sorted(eligible):
        a, b = recall_new.get(c), recall_old.get(c)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a > b:
            wins += 1
        elif a < b:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties,
            "p_one_sided": sign_test_p(wins, losses)}


def nondegenerate_rate(bundles: list[dict], labels: list[str],
                       sidedness: list[str], variant: str,
                       stratum: set[str]) -> float:
    """Fraction of real clips of the stratum whose STRICT core (no fallback)
    is non-degenerate under the variant — the mask-fitness criterion."""
    flags = [not variant_core(b, s, variant)[1].get("core_degenerate", False)
             for b, s, lab in zip(bundles, sidedness, labels) if lab in stratum]
    return float(np.mean(flags)) if flags else float("nan")


def adopt_core_mask(r1: dict, bundles, labels, sidedness, bars: dict,
                    onesided_classes: set[str], eligible: set[str]) -> dict:
    """Pre-registered rule (bars.exam.adoption): challenger v3_sided vs
    incumbent v2_envelope. all_frames is the no-mask baseline — reported, and
    adopted only if BOTH masks fail their criteria and it wins the same test."""
    alpha = bars["exam"]["adoption"]["alpha"]
    tol = bars["exam"]["adoption"]["overall_tolerance"]
    ndg_min = bars["exam"]["adoption"]["mask_nondegenerate_min"]
    trust_min = bars["exam"]["trust"]["trust_min_recall"]

    inc, cha = r1["m1a__v2_envelope"], r1["m1a__v3_sided"]
    stratum_eligible = onesided_classes & eligible
    st = class_sign_test(cha["per_class_recall"], inc["per_class_recall"], stratum_eligible)
    regressions = [c for c in eligible
                   if inc["per_class_recall"].get(c, 0) >= trust_min
                   and cha["per_class_recall"].get(c, 1) < trust_min]
    ndg = {v: nondegenerate_rate(bundles, labels, sidedness, v, onesided_classes)
           for v in CORE_VARIANTS}
    checks = {
        "overall_ok": bool(cha["accuracy_1nn"] >= inc["accuracy_1nn"] - tol),
        "stratum_sign_test": st,
        "stratum_ok": bool(st["p_one_sided"] < alpha),
        "regression_guard_ok": not regressions,
        "regressed_classes": regressions,
        "nondegenerate_rates": ndg,
        "nondegenerate_ok": bool(ndg["v3_sided"] >= ndg_min),
    }
    adopted = (checks["overall_ok"] and checks["stratum_ok"]
               and checks["regression_guard_ok"] and checks["nondegenerate_ok"])
    return {"winner": "v3_sided" if adopted else "v2_envelope",
            "adopted_challenger": bool(adopted), "checks": checks}


def adopt_motion(r1: dict, bars: dict, camera_classes: set[str],
                 eligible: set[str], m1c_definedness: dict) -> dict:
    """Pre-registered rule: decomposition (M1b+M1c) vs incumbent MFS on the
    camera stratum; M1c guard on defined classes. Contingency: not adopted ->
    incumbent MFS ships as the single headline motion metric."""
    alpha = bars["exam"]["adoption"]["alpha"]
    guard_drop = bars["exam"]["adoption"]["m1c_overall_guard_drop"]
    inc, cam, obj = r1["m_incumbent"], r1["m1b_camera"], r1["m1c_object"]
    stratum_eligible = camera_classes & eligible
    st = class_sign_test(cam["per_class_recall"], inc["per_class_recall"], stratum_eligible)
    defined = {c for c, f in m1c_definedness.items()
               if f >= bars["exam"]["trust"]["m1c_min_definedness"]} & eligible
    inc_med = float(np.nanmedian([inc["per_class_recall"].get(c, np.nan) for c in defined])) if defined else float("nan")
    obj_med = float(np.nanmedian([obj["per_class_recall"].get(c, np.nan) for c in defined])) if defined else float("nan")
    checks = {
        "stratum_sign_test": st,
        "stratum_ok": bool(st["p_one_sided"] < alpha),
        "m1c_median_recall_defined": obj_med,
        "incumbent_median_recall_defined": inc_med,
        "m1c_guard_ok": bool(np.isnan(inc_med) or np.isnan(obj_med)
                             or obj_med >= inc_med - guard_drop),
    }
    adopted = checks["stratum_ok"] and checks["m1c_guard_ok"]
    return {"winner": "v3_decomposed" if adopted else "v2_mfs_incumbent",
            "adopted_challenger": bool(adopted), "checks": checks}


# --- trust map ---------------------------------------------------------------------

def m1c_definedness_per_class(D_obj: np.ndarray, labels: list[str]) -> dict:
    """Fraction of a class's same-class pairs where M1c is defined (non-NaN).
    NaN prevalence is a trust fact — recall over 2 defined pairs is not trust."""
    lab = np.array(labels)
    out = {}
    for c in sorted(set(labels)):
        idx = np.flatnonzero(lab == c)
        if len(idx) < 2:
            out[c] = 0.0
            continue
        vals = [D_obj[i, j] for i, j in itertools.combinations(idx, 2)]
        out[c] = float(np.mean(np.isfinite(vals)))
    return out


def trust_map(r1_winner: dict, r2_winner: dict, class_n: dict,
              m1c_definedness: dict, bars: dict) -> dict:
    """Per-class, per-metric trust consumed by every model report (SPEC §6.1)."""
    tmin = bars["exam"]["trust"]["trust_min_recall"]
    nmin = bars["exam"]["trust"]["min_class_n"]
    dmin = bars["exam"]["trust"]["m1c_min_definedness"]
    out = {}
    for c, n in sorted(class_n.items()):
        if n < nmin:
            out[c] = {"n_clips": n, "eligible": False,
                      "m1a": False, "m1b": False, "m1c": False, "m2b": False,
                      "reason": f"n={n} < {nmin} (singletons permanently untrusted)"}
            continue
        rec = {m: r1_winner[m]["per_class_recall"].get(c) for m in ("m1a", "m1b", "m1c")}
        out[c] = {
            "n_clips": n, "eligible": True,
            "m1a": bool(rec["m1a"] is not None and rec["m1a"] >= tmin),
            "m1b": bool(rec["m1b"] is not None and rec["m1b"] >= tmin),
            "m1c": bool(rec["m1c"] is not None and rec["m1c"] >= tmin
                        and m1c_definedness.get(c, 0) >= dmin),
            "m1c_definedness": m1c_definedness.get(c),
            "m2b": bool(r2_winner["per_class_recall"].get(c, 0) >= tmin),
            "recall": rec,
        }
    return out


# --- orchestration -----------------------------------------------------------------

def run_exam(bundles: list[dict], labels: list[str], sidedness: list[str],
             corpus: dict, bars: dict, out_dir: pathlib.Path,
             analysis_dir: pathlib.Path | None = None) -> dict:
    """Full Block A. Caller supplies processed bundles (cached pipeline).
    Refuses unfrozen bars — running earlier would de-register them.

    When analysis_dir is given, the full diagnostic state (matrices, confusion,
    per-clip rows, per-tag accuracy — see certify.diagnostics) is persisted
    there; a diagnostics failure never gates the exam."""
    if not bars.get("frozen"):
        raise RuntimeError(
            "bars.yaml is DRAFT (frozen: false) — freeze the bars before "
            "executing the exam (SPEC §6).")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_n = {c: corpus["classes"][c]["n_clips"] for c in corpus["classes"]}
    eligible = {c for c, n in class_n.items()
                if n >= bars["exam"]["trust"]["min_class_n"]}
    onesided = {c for c, v in corpus["classes"].items() if v["sidedness"] == "onesided"}
    camera = {c for c, v in corpus["classes"].items() if "camera" in v.get("tags", [])}

    # R1 appearance under every mask variant
    r1, mats = {}, {}
    for v in CORE_VARIANTS:
        D = appearance_distance_matrix(bundles, v, sidedness)
        mats[f"m1a__{v}"] = D
        r1[f"m1a__{v}"] = retrieval_eval(D, labels)
    # R1 motion
    Dm = motion_distance_matrices(bundles)
    for k, D in Dm.items():
        mats[k] = D
        r1[k] = retrieval_eval(D, labels)
    m1c_def = m1c_definedness_per_class(Dm["m1c_object"], labels)

    # adoption (pre-registered, mechanical)
    mask_verdict = adopt_core_mask(r1, bundles, labels, sidedness, bars, onesided, eligible)
    motion_verdict = adopt_motion(r1, bars, camera, eligible, m1c_def)
    mask_w = mask_verdict["winner"]

    # R2 pool-level under the winning mask
    r2 = pool_margin_exam(bundles, labels, sidedness, mask_w)

    # O7 conditional (pre-registered): only fires on camera-stratum trust failure
    cam_recall = [r1["m1b_camera"]["per_class_recall"].get(c) for c in camera & eligible]
    cam_mean = float(np.nanmean([r for r in cam_recall if r is not None])) if cam_recall else float("nan")
    o7 = {"camera_stratum_mean_recall": cam_mean,
          "huber_triggered": bool(np.isfinite(cam_mean)
                                  and cam_mean < bars["exam"]["trust"]["trust_min_recall"]),
          "note": "if triggered: implement Huber IRLS variant, re-run this exam "
                  "under the same adoption rule (new draft version)"}

    # bar 1: M1a separation floor on the winning variant — d only. The accuracy
    # conjunct was deleted by owner decision at the draft.8 inspection with the
    # outcome known (0.673 vs a 0.80 floor calibrated on the 11-style v2
    # corpus); accuracy stays reported here as a descriptive statistic.
    win = r1[f"m1a__{mask_w}"]
    bar1 = {"acc": win["accuracy_1nn"], "d": win["separation_cohens_d"],
            "d_min": bars["exam"]["bar1_m1a_floor"]["d_min"]}
    bar1["pass"] = bool(bar1["d"] >= bar1["d_min"])

    winner_r1 = {"m1a": r1[f"m1a__{mask_w}"],
                 "m1b": r1["m1b_camera"], "m1c": r1["m1c_object"]}
    tmap = trust_map(winner_r1, r2, class_n, m1c_def, bars)

    # diagnostic state (representation only — never feeds a verdict)
    by_tag = None
    try:
        keys = sorted(corpus["clips"])
        assert len(keys) == len(bundles), "bundle order must be sorted corpus keys"
        ana = diagnostics.build_analysis(corpus, keys, labels, sidedness,
                                         r1, mats, r2, mask_w)
        by_tag = ana["by_tag"]
        if analysis_dir is not None:
            diagnostics.write_analysis(analysis_dir, ana, mats, keys)
    except Exception as e:  # noqa: BLE001 — diagnostics must never gate the exam
        print(f"[exam] diagnostics failed (non-gating): {type(e).__name__}: {e}",
              flush=True)

    result = {"r1": {k: {kk: vv for kk, vv in v.items() if kk != "confusion"}
                     for k, v in r1.items()},
              "r2": {k: v for k, v in r2.items() if k != "rows"},
              "mask_adoption": mask_verdict, "motion_adoption": motion_verdict,
              "o7_conditional": o7, "bar1": bar1, "by_tag": by_tag,
              "m1c_definedness": m1c_def, "trust_map": tmap}
    (out_dir / "exam.json").write_text(json.dumps(result, indent=1, default=str))
    (out_dir / "trust_map.json").write_text(json.dumps(tmap, indent=1))
    return result
