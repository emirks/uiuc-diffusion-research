"""Exam-design primitives (design-agnostic building blocks).

The exam-design phase optimizes the EXAM's explanatory/diagnostic/measurement power,
not a score. These are the neutral tools any design will compose: ranking-based
retrieval quality (mAP@R — sees the full ranking, top-weighted, unlike 1-NN), a
label-permutation null (empirical chance robust to class imbalance + small-n), a
clip-level bootstrap (uncertainty / callable deltas), and a content-similarity matrix
(the confound the exam must causally control for — endpoint content, matching the
certified content_invariance_audit's 0.82 alarm).

All operate on a 223x223 distance matrix + the single per-clip style labels.
"""

from __future__ import annotations

import numpy as np


# --- content-similarity (the confound to control) -----------------------------

def content_sim_endpoint(feats: np.ndarray, n_prefix: np.ndarray, n_suffix: np.ndarray
                         ) -> np.ndarray:
    """Endpoint content similarity S_c[i,j] = 0.5(eA_i·eA_j + eB_i·eB_j), the same
    endpoint/content notion the certified content_invariance_audit correlates M1a
    against (the 0.82 confound). Unit anchors -> cosine. [223,223] in [-1,1]."""
    from . import kernels as K
    n = len(feats)
    eA = np.zeros((n, feats.shape[2])); eB = np.zeros((n, feats.shape[2]))
    for i in range(n):
        a, b = K.endpoint_anchors(feats[i], int(n_prefix[i]), int(n_suffix[i]))
        eA[i], eB[i] = a, b
    return 0.5 * (eA @ eA.T + eB @ eB.T)


# --- ranking-based retrieval quality ------------------------------------------

def _rank_matrix(D: np.ndarray) -> np.ndarray:
    """Per row, argsort of the other clips by ascending distance (NaN -> last)."""
    M = D.copy().astype(float)
    np.fill_diagonal(M, np.inf)
    M[np.isnan(M)] = np.inf
    return M


def ap_at_r_per_anchor(D: np.ndarray, labels: list[str],
                       gallery_mask: np.ndarray | None = None) -> np.ndarray:
    """AP@R for every anchor. R = # same-class positives available in the (masked)
    gallery. gallery_mask[i] = boolean [n] of which clips are in anchor i's gallery
    (self always excluded). Returns AP per anchor (NaN where R=0 or anchor undefined)."""
    M = _rank_matrix(D)
    lab = np.asarray(labels)
    n = len(lab)
    ap = np.full(n, np.nan)
    for i in range(n):
        allowed = np.ones(n, dtype=bool) if gallery_mask is None else gallery_mask[i].copy()
        allowed[i] = False
        allowed &= np.isfinite(M[i])
        if allowed.sum() == 0:
            continue
        pos = allowed & (lab == lab[i])
        R = int(pos.sum())
        if R == 0:
            continue
        order = np.argsort(M[i][allowed], kind="mergesort")
        idx = np.flatnonzero(allowed)[order]           # gallery clips, nearest first
        rel = (lab[idx] == lab[i]).astype(float)
        topR = rel[:R]
        prec = np.cumsum(topR) / (np.arange(R) + 1.0)
        ap[i] = float((prec * topR).sum() / R)
    return ap


def macro_map_at_r(D: np.ndarray, labels: list[str], eligible: set[str],
                   gallery_mask: np.ndarray | None = None,
                   min_positives: int = 1) -> dict:
    """Macro mAP@R over eligible classes (n>=4 tag-eligible). Reports per-class mAP,
    the macro mean, and anchor coverage (fraction of eligible anchors with R>=min)."""
    ap = ap_at_r_per_anchor(D, labels, gallery_mask)
    lab = np.asarray(labels)
    per_class, covered, total = {}, 0, 0
    for c in sorted(eligible):
        idx = np.flatnonzero(lab == c)
        total += len(idx)
        vals = ap[idx][np.isfinite(ap[idx])]
        covered += len(vals)
        per_class[c] = float(vals.mean()) if len(vals) else float("nan")
    defined = [v for v in per_class.values() if np.isfinite(v)]
    return {
        "macro_map": float(np.mean(defined)) if defined else float("nan"),
        "per_class": per_class,
        "n_classes_defined": len(defined),
        "n_classes_eligible": len(eligible),
        "anchor_coverage": covered / total if total else 0.0,
        "ap_per_anchor": ap,
    }


# --- permutation null + bootstrap ---------------------------------------------

def permutation_null(D: np.ndarray, labels: list[str], eligible: set[str],
                     gallery_mask: np.ndarray | None = None, n_perm: int = 1000,
                     rng_seed: int = 0) -> dict:
    """Empirical chance distribution of macro mAP@R under label permutation (the
    geometry/gallery held fixed, only labels shuffled). Robust to imbalance + small-n
    chance inflation. Returns null mean/std/quantiles + the observed excess."""
    rng = np.random.default_rng(rng_seed)
    obs = macro_map_at_r(D, labels, eligible, gallery_mask)["macro_map"]
    lab = np.asarray(labels)
    null = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(len(lab))
        null[p] = macro_map_at_r(D, list(lab[perm]), eligible, gallery_mask)["macro_map"]
    null = null[np.isfinite(null)]
    return {
        "observed": obs,
        "null_mean": float(null.mean()), "null_std": float(null.std()),
        "null_q95": float(np.quantile(null, 0.95)),
        "excess": float(obs - null.mean()),
        "z": float((obs - null.mean()) / (null.std() + 1e-12)),
        "p_value": float((1 + (null >= obs).sum()) / (1 + len(null))),
        "n_perm": len(null),
    }


# --- causal axis: content-matched hard negatives (fable's key revision) --------

def controlled_gallery_mask(labels: list[str], content_sim: np.ndarray) -> np.ndarray:
    """Per anchor i, gallery = all same-class positives + the R_i hardest
    content-confounded negatives (most content-similar different-class clips), where
    R_i = #positives. Asks the causal question: does D rank a same-style,
    different-content clip above a same-content, different-style clip? Positives are
    NEVER deleted (no coverage collapse). content_sim is LABEL-FREE, so this is clean
    under label permutation."""
    lab = np.asarray(labels)
    n = len(lab)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        same = (lab == lab[i]); same[i] = False
        R = int(same.sum())
        mask[i] |= same
        if R == 0:
            continue
        diff = np.flatnonzero(lab != lab[i])
        if len(diff) == 0:
            continue
        hardest = diff[np.argsort(-content_sim[i, diff])[:R]]   # top-R content-similar negatives
        mask[i, hardest] = True
    return mask


def random_gallery_mask(labels: list[str], rng: np.random.Generator) -> np.ndarray:
    """Same shape as controlled_gallery_mask but the R_i negatives are RANDOM
    different-class clips instead of the content-hardest. The control for the
    confound-susceptibility field: comparing controlled skill (hard negatives) to
    random-negative skill isolates how much a metric is attracted to content-matched
    negatives, independent of the gallery-shrinkage that both share."""
    lab = np.asarray(labels)
    n = len(lab)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        same = (lab == lab[i]); same[i] = False
        R = int(same.sum())
        mask[i] |= same
        if R == 0:
            continue
        diff = np.flatnonzero(lab != lab[i])
        if len(diff) == 0:
            continue
        pick = rng.choice(diff, size=min(R, len(diff)), replace=False)
        mask[i, pick] = True
    return mask


def controlled_excess(D: np.ndarray, labels: list[str], content_sim: np.ndarray,
                      eligible: set[str], n_perm: int = 1000, n_boot: int = 400,
                      rng_seed: int = 0) -> dict:
    """The headline barred quantity: standardized content-controlled discrimination
    excess. Observed controlled-mAP@R minus a size-preserving permutation null (gallery
    rebuilt under each permuted labeling), standardized by null std (z), with a
    clip-bootstrap CI. Also returns the UNCONTROLLED excess (for retention ratio)."""
    lab = np.asarray(labels)
    gm = controlled_gallery_mask(labels, content_sim)
    obs = macro_map_at_r(D, labels, eligible, gm)["macro_map"]
    rng = np.random.default_rng(rng_seed)
    null = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(len(lab))
        pl = list(lab[perm])
        gmp = controlled_gallery_mask(pl, content_sim)
        null[p] = macro_map_at_r(D, pl, eligible, gmp)["macro_map"]
    null = null[np.isfinite(null)]
    # Delete-d SUBSAMPLING bootstrap without replacement + Politis-Romano root-n
    # rescaling (a with-replacement bootstrap injects distance-0 twins that
    # self-retrieve and inflate the CI). d_b = sqrt(m/n)*(theta_b - theta_hat);
    # CI = theta_hat + quantiles(d_b). The sqrt(m/n) corrects the subsample's inflated
    # variance back to full-sample scale.
    n = len(lab)
    m = int(0.8 * n)
    dev = []
    for _ in range(n_boot):
        samp = rng.choice(n, size=m, replace=False)
        Db = D[np.ix_(samp, samp)]
        cs = content_sim[np.ix_(samp, samp)]
        gmb = controlled_gallery_mask(list(lab[samp]), cs)
        r = macro_map_at_r(Db, list(lab[samp]), eligible, gmb)["macro_map"]
        if np.isfinite(r):
            dev.append(np.sqrt(m / n) * (r - obs))
    dev = np.array(dev)
    boot_lo = float(obs + np.quantile(dev, 0.025))
    boot_hi = float(obs + np.quantile(dev, 0.975))
    floor = 2.0 * float(null.std())                    # resolution-derived margin
    excess = float(obs - null.mean())
    p_value = float((1 + (null >= obs).sum()) / (1 + len(null)))
    nm = float(null.mean())
    return {
        "controlled_map": float(obs),
        "null_mean": nm, "null_std": float(null.std()),
        "skill": float((obs - nm) / (1.0 - nm)) if nm < 1 else float("nan"),
        "excess": excess, "z": float(excess / (null.std() + 1e-12)),
        "p_value": p_value,
        "boot_lo95": boot_lo, "boot_hi95": boot_hi,
        "resolution_floor": floor,
        # conjunctive gate: lower-CI excess clears the floor AND permutation-significant
        "PASS": bool((boot_lo - nm > floor) and (p_value < 0.01)),
    }


def skill_score(mAP: float, null_mean: float) -> float:
    """Chance- and ceiling-corrected skill: (mAP - null)/(1 - null) in [0,1]."""
    return float((mAP - null_mean) / (1.0 - null_mean)) if null_mean < 1 else float("nan")


def full_datasheet(D: np.ndarray, labels: list[str], content_sim: np.ndarray,
                   eligible: set[str], n_perm: int = 300, n_boot: int = 300,
                   rng_seed: int = 0, hub_k: int = 10,
                   content_sim_alt: np.ndarray | None = None) -> dict:
    """The instrument datasheet for one metric. One permutation loop yields both the
    controlled (content-matched) and uncontrolled statistics, macro AND per-class, so
    every field is chance-corrected on its own null.

    CAUSAL GATE (health-validated 2026-07-16, replaces the old above-chance gate).
    The prior binary PASS ((boot_lo-null)>floor AND p<0.01) is a PROVEN NO-OP for any
    content-monotone metric: for D_cont=1-content_sim the controlled top-R equals the
    uncontrolled top-R (the negatives ARE selected by content_sim), so controlled skill
    ≡ uncontrolled skill for observed AND every permuted labeling — the gate can never
    fail a pure-content metric on a corpus where content carries class signal. It is
    demoted here to a DESCRIPTIVE `above_chance` field. The real gate is CAUSAL EXCESS
    over an explicit content baseline B=D_cont: causal_PASS iff
    causal_excess = Cn - Cn_B  >=  max(0.10, 2*floor)  AND  paired lo95(oc - oc_B) > 0.
    Both arms are individually insufficient (random D passes the paired-CI arm; a
    sub-floor metric passes the excess arm) — the conjunction is load-bearing, do NOT
    simplify to one arm. `causal_excess_maxproxy` (excess over max(DINO, alt) baseline,
    when content_sim_alt is supplied) ships DESCRIPTIVE: the DINO baseline is proxy-
    specific and understates the content ceiling on some strata. `confound_susceptibility`
    = Cn - C_rand (skill vs random negatives) is descriptive (negative => content-matched
    negatives attract the metric; tolerated when causal_excess is large).
    Fields: Reliability (hubness+hub), Validity (uncontrolled skill U), Causal (Cn,
    content_baseline_Cn, causal_excess, causal_PASS + descriptive above_chance/maxproxy/
    susceptibility), Shortcut (Δ=Cn-U per class), Trust map, Power (CI width + null scale)."""
    from .. import hubness as HB
    lab = np.asarray(labels)
    classes = sorted(eligible)
    gm = controlled_gallery_mask(labels, content_sim)

    oc = macro_map_at_r(D, labels, eligible, gm)          # controlled observed
    ou = macro_map_at_r(D, labels, eligible)             # uncontrolled observed
    rng = np.random.default_rng(rng_seed)
    nc_macro, nu_macro = np.empty(n_perm), np.empty(n_perm)
    nc_cls = {c: np.empty(n_perm) for c in classes}
    for p in range(n_perm):
        perm = rng.permutation(len(lab)); pl = list(lab[perm])
        gmp = controlled_gallery_mask(pl, content_sim)
        rc = macro_map_at_r(D, pl, eligible, gmp)
        ru = macro_map_at_r(D, pl, eligible)
        nc_macro[p], nu_macro[p] = rc["macro_map"], ru["macro_map"]
        for c in classes:
            nc_cls[c][p] = rc["per_class"][c]

    def sk(x, nm): return float((x - nm) / (1 - nm)) if nm < 1 and np.isfinite(x) else float("nan")
    ncm, num = float(np.nanmean(nc_macro)), float(np.nanmean(nu_macro))
    Cn, U = sk(oc["macro_map"], ncm), sk(ou["macro_map"], num)
    z_c = float((oc["macro_map"] - ncm) / (np.nanstd(nc_macro) + 1e-12))
    p_c = float((1 + (nc_macro >= oc["macro_map"]).sum()) / (1 + n_perm))

    # Politis-Romano subsample CI on controlled macro
    n = len(lab); mm = int(0.8 * n); dev = []
    for _ in range(n_boot):
        s = rng.choice(n, size=mm, replace=False)
        r = macro_map_at_r(D[np.ix_(s, s)], list(lab[s]), eligible,
                           controlled_gallery_mask(list(lab[s]), content_sim[np.ix_(s, s)]))["macro_map"]
        if np.isfinite(r): dev.append(np.sqrt(mm / n) * (r - oc["macro_map"]))
    dev = np.array(dev)
    boot_lo = float(oc["macro_map"] + np.quantile(dev, 0.025))
    floor = 2.0 * float(np.nanstd(nc_macro))
    above_chance = bool((boot_lo - ncm > floor) and (p_c < 0.01))   # OLD gate (no-op) → descriptive

    # --- CAUSAL-EXCESS GATE over an explicit content baseline B = D_cont ---
    def _controlled_skill(Db):
        """Controlled skill of matrix Db on the SAME eligible set + content galleries."""
        obs = macro_map_at_r(Db, labels, eligible, gm)["macro_map"]
        rb = np.random.default_rng(rng_seed + 4242)
        nb = np.empty(n_perm)
        for p in range(n_perm):
            perm = rb.permutation(len(lab)); pl = list(lab[perm])
            nb[p] = macro_map_at_r(Db, pl, eligible, controlled_gallery_mask(pl, content_sim))["macro_map"]
        nmb = float(np.nanmean(nb))
        return obs, sk(obs, nmb)
    D_cont = 1.0 - np.asarray(content_sim, float); np.fill_diagonal(D_cont, 0.0)
    ocB, Cn_B = _controlled_skill(D_cont)
    causal_excess = float(Cn - Cn_B)
    # paired Politis-Romano subsample lo95 of the raw controlled-mAP difference (oc - ocB)
    paired_diff = float(oc["macro_map"] - ocB)
    rngp = np.random.default_rng(rng_seed + 777); dev2 = []
    for _ in range(n_boot):
        s = rngp.choice(n, size=mm, replace=False)
        cs = content_sim[np.ix_(s, s)]; gms = controlled_gallery_mask(list(lab[s]), cs)
        dd = (macro_map_at_r(D[np.ix_(s, s)], list(lab[s]), eligible, gms)["macro_map"]
              - macro_map_at_r(D_cont[np.ix_(s, s)], list(lab[s]), eligible, gms)["macro_map"])
        if np.isfinite(dd): dev2.append(np.sqrt(mm / n) * (dd - paired_diff))
    paired_lo95 = float(paired_diff + np.quantile(np.array(dev2), 0.025)) if dev2 else float("nan")
    causal_gate_thr = float(max(0.10, floor))          # floor is already 2*null_std
    causal_PASS = bool(causal_excess >= causal_gate_thr and paired_lo95 > 0)
    # descriptive: excess over the STRONGER of DINO and an alternative content proxy
    causal_excess_maxproxy = None
    if content_sim_alt is not None:
        D_alt = 1.0 - np.asarray(content_sim_alt, float); np.fill_diagonal(D_alt, 0.0)
        _, Cn_alt = _controlled_skill(D_alt)
        causal_excess_maxproxy = float(Cn - max(Cn_B, Cn_alt))
    # descriptive: confound susceptibility = Cn - skill(random negatives)
    rr = np.random.default_rng(rng_seed + 909)
    obs_rand = np.nanmean([macro_map_at_r(D, labels, eligible, random_gallery_mask(labels, rr))["macro_map"]
                           for _ in range(10)])
    nrand = np.empty(max(1, n_perm // 3))
    for p in range(len(nrand)):
        perm = rr.permutation(len(lab)); pl = list(lab[perm])
        nrand[p] = macro_map_at_r(D, pl, eligible, random_gallery_mask(np.asarray(pl), rr))["macro_map"]
    C_rand = sk(float(obs_rand), float(np.nanmean(nrand)))
    confound_susceptibility = float(Cn - C_rand)

    # per-class trust map
    trust = {}
    for c in classes:
        nmc = float(np.nanmean(nc_cls[c]))
        obsc = oc["per_class"][c]
        npos = int((lab == c).sum())
        if not np.isfinite(obsc) or npos < 4:
            stamp = "UNRATABLE"
        else:
            pc = float((1 + (nc_cls[c] >= obsc).sum()) / (1 + n_perm))
            skc = sk(obsc, nmc)
            stamp = "TRUSTED" if pc < 0.01 and skc > 0.15 else "WEAK"
        trust[c] = {"controlled_map": obsc, "skill": sk(obsc, nmc) if np.isfinite(obsc) else float("nan"),
                    "delta": (sk(obsc, nmc) - sk(ou["per_class"][c], num)) if np.isfinite(obsc) else float("nan"),
                    "n": npos, "stamp": stamp}

    from . import kernels as _K
    hub = HB.gate(HB.hubness_stats(D, labels, k=hub_k),
                  {"hubness": {"gating_k": hub_k, "max_hubness_skew": 3.0,
                               "min_pred_entropy_norm": 0.70, "max_pred_class_share": 0.25}})
    sink, share = _K.dominant_sink(D, labels)
    return {
        "reliability": {"hubness_pass": hub["pass"], "skew": hub["stats"]["hubness_skew"],
                        "entropy": hub["stats"]["pred_entropy_norm"],
                        "maxpred": hub["stats"]["max_pred_class_share"],
                        "hub_class": sink, "hub_share": share},
        "validity": {"uncontrolled_map": ou["macro_map"], "U_skill": U, "null": num},
        "causal": {"controlled_map": oc["macro_map"], "Cn_skill": Cn, "null": ncm,
                   "z": z_c, "p": p_c, "boot_lo": boot_lo, "floor": floor,
                   # NEW causal-excess gate (the real gate) — see docstring
                   "content_baseline_Cn": Cn_B, "causal_excess": causal_excess,
                   "paired_diff": paired_diff, "paired_lo95": paired_lo95,
                   "causal_gate_thr": causal_gate_thr, "causal_PASS": causal_PASS,
                   # descriptive-only fields
                   "causal_excess_maxproxy": causal_excess_maxproxy,
                   "confound_susceptibility": confound_susceptibility,
                   "above_chance": above_chance,   # OLD binary gate, demoted (proven no-op)
                   "PASS": causal_PASS},           # back-compat alias → now the FIXED gate
        "shortcut": {"delta_macro": (Cn - U), "note": "Δ=Cn-U; ≥0 content-robust, ≪0 shortcut"},
        "trust_map": trust,
        "coverage": {"anchor_cov_controlled": oc["anchor_coverage"],
                     "classes_defined": oc["n_classes_defined"], "classes_eligible": len(classes)},
    }


def ensemble_ceiling(mats: dict, labels: list[str], content_sim: np.ndarray,
                     eligible: set[str], n_perm: int = 200) -> dict:
    """Per class, the best controlled skill over ALL metrics. A class no metric beats
    chance on is CORPUS-LIMITED (untestable on these features), not metric-limited."""
    lab = np.asarray(labels)
    classes = sorted(eligible)
    per_metric = {}
    for nm, D in mats.items():
        gm = controlled_gallery_mask(labels, content_sim)
        obs = macro_map_at_r(D, labels, eligible, gm)["per_class"]
        rng = np.random.default_rng(0)
        nullc = {c: [] for c in classes}
        for _ in range(n_perm):
            perm = rng.permutation(len(lab)); pl = list(lab[perm])
            r = macro_map_at_r(D, pl, eligible, controlled_gallery_mask(pl, content_sim))["per_class"]
            for c in classes: nullc[c].append(r[c])
        per_metric[nm] = {c: (float((obs[c] - np.nanmean(nullc[c])) / (1 - np.nanmean(nullc[c])))
                              if np.isfinite(obs[c]) else float("nan")) for c in classes}
    ceiling = {c: max((per_metric[m][c] for m in mats if np.isfinite(per_metric[m][c])), default=float("nan"))
               for c in classes}
    corpus_limited = [c for c in classes if not (np.isfinite(ceiling[c]) and ceiling[c] > 0.15)]
    return {"per_metric_skill": per_metric, "ceiling": ceiling,
            "corpus_limited_classes": corpus_limited}


def bootstrap_excess_ci(D: np.ndarray, labels: list[str], eligible: set[str],
                        gallery_mask: np.ndarray | None = None, n_boot: int = 500,
                        rng_seed: int = 0) -> dict:
    """Clip-level bootstrap CI of the macro mAP@R (uncertainty on the headline).
    Resamples clips with replacement, recomputing over the induced sub-matrix."""
    rng = np.random.default_rng(rng_seed)
    lab = np.asarray(labels)
    n = len(lab)
    vals = []
    for _ in range(n_boot):
        samp = rng.integers(0, n, n)
        Db = D[np.ix_(samp, samp)]
        gm = None if gallery_mask is None else gallery_mask[np.ix_(samp, samp)]
        r = macro_map_at_r(Db, list(lab[samp]), eligible, gm)["macro_map"]
        if np.isfinite(r):
            vals.append(r)
    vals = np.array(vals)
    return {"boot_mean": float(vals.mean()), "boot_lo95": float(np.quantile(vals, 0.025)),
            "boot_hi95": float(np.quantile(vals, 0.975)), "n_boot": len(vals)}
