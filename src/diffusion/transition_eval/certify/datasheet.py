"""Instrument datasheet + the causal-excess gate (Block A / bar 9, v4).

Ports the health-validated exam-design machinery (workbench/search/exam_design.py,
REPORT_exam.md + REPORT_health_validation.md) into the certified instrument.

THE GATE'S HISTORY, DISCLOSED: the first datasheet design gated on "controlled
skill above permutation chance" — proven a mathematical NO-OP for any
content-monotone metric (for D_cont = 1-content_sim the controlled top-R equals
the uncontrolled top-R under the observed AND every permuted labeling; measured
U = Cn = 0.2068 exactly, and pure DINO content PASSED). That gate survives only
as the descriptive `above_chance` field. The real gate is CAUSAL EXCESS over an
explicit content baseline B = D_cont:

    causal_PASS  iff  causal_excess = Cn - Cn_B >= max(0.10, 2*null_std)
                  AND paired-subsample lo95(controlled-mAP difference) > 0

Both arms are individually insufficient (a random matrix passes the paired-CI
arm; a sub-floor metric passes the excess arm): the conjunction is load-bearing
— do NOT simplify to one arm. Bar 9 additionally requires the three negative
controls (D_cont, the independent pixel proxy, random-D) to FAIL the same gate
on the same galleries — a self-verifying bar: if a falsifier passes, the gate
is broken and the bar fails.

Baseline choice, disclosed (advisor decision under the owner's 2026-07-16
delegation, outcome-aware): the DINO endpoint baseline GATES; the max-over-
proxies excess (`causal_excess_maxproxy`) is a mandatory NON-GATING record
field. A gate needs a pinned, closed-form baseline; "strongest known proxy" is
an open roster that moves with search effort. On the object stratum the choice
is consequential: M1c's excess is 0.156 over DINO but 0.0991 over the color
proxy (the 0.10 bar) — both numbers print in the record.
"""

from __future__ import annotations

import numpy as np

from ..reference_stats import endpoint_anchors

# Frozen hubness thresholds (calibrated on the incumbents alone, pre-freeze,
# corpus-only — the metric-workbench gates.yaml numbers, carried verbatim).
HUB_GATING_K = 10
HUB_MAX_SKEW = 3.0
HUB_MIN_PRED_ENTROPY = 0.70
HUB_MAX_PRED_SHARE = 0.25

RANDOM_D_SEED = 999          # pinned seed of the random-D negative control


# ---------------------------------------------------------------------------
# Content proxies (the confounds the gate controls for)
# ---------------------------------------------------------------------------

def content_sim_endpoint(feats_list, n_prefix, n_suffix) -> np.ndarray:
    """Endpoint content similarity S_c[i,j] = 0.5(eA_i.eA_j + eB_i.eB_j) — the
    same endpoint/content notion the content-invariance audit correlates M1a
    against. Unit anchors -> cosine."""
    n = len(feats_list)
    d = np.asarray(feats_list[0]).shape[1]
    eA = np.zeros((n, d))
    eB = np.zeros((n, d))
    for i in range(n):
        a, b = endpoint_anchors(feats_list[i], int(n_prefix[i]), int(n_suffix[i]))
        eA[i], eB[i] = a, b
    return 0.5 * (eA @ eA.T + eB @ eB.T)


def content_sim_pixel(clip_paths: list, n_prefix, n_suffix, bins: int = 8,
                      short_side: int = 256) -> np.ndarray:
    """B': independent PIXEL content proxy (no DINO anywhere). Endpoint
    descriptor = L2-normalized sqrt joint-RGB histogram (bins^3) over the
    conditioned endpoint frames; sim = 0.5*(cos(hA_i,hA_j) + cos(hB_i,hB_j)).
    Measured Spearman vs the DINO endpoint proxy: 0.09 — a genuinely
    independent falsifier."""
    from ..video_io import load_frames

    def hist(frames):
        q = frames.astype(np.int64) * bins // 256
        idx = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]
        bc = np.bincount(idx.ravel(), minlength=bins ** 3).astype(np.float64)
        h = np.sqrt(bc / max(bc.sum(), 1))
        return h / (np.linalg.norm(h) + 1e-12)

    n = len(clip_paths)
    HA = np.zeros((n, bins ** 3))
    HB = np.zeros((n, bins ** 3))
    for i, p in enumerate(clip_paths):
        fr, _ = load_frames(p, short_side=short_side)
        HA[i] = hist(fr[:int(n_prefix[i])])
        HB[i] = hist(fr[-int(n_suffix[i]):] if int(n_suffix[i]) else fr[:int(n_prefix[i])])
    return 0.5 * (HA @ HA.T + HB @ HB.T)


def random_distance_matrix(n: int, seed: int = RANDOM_D_SEED) -> np.ndarray:
    """The random-D negative control: symmetric uniform noise, pinned seed."""
    rng = np.random.default_rng(seed)
    R = rng.random((n, n))
    R = (R + R.T) / 2
    np.fill_diagonal(R, 0.0)
    return R


# ---------------------------------------------------------------------------
# Ranking machinery (mAP@R over masked galleries)
# ---------------------------------------------------------------------------

def _rank_matrix(D: np.ndarray) -> np.ndarray:
    M = D.copy().astype(float)
    np.fill_diagonal(M, np.inf)
    M[np.isnan(M)] = np.inf
    return M


def ap_at_r_per_anchor(D: np.ndarray, labels, gallery_mask=None) -> np.ndarray:
    """AP@R per anchor; R = #same-class positives in the (masked) gallery."""
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
        idx = np.flatnonzero(allowed)[order]
        rel = (lab[idx] == lab[i]).astype(float)
        topR = rel[:R]
        prec = np.cumsum(topR) / (np.arange(R) + 1.0)
        ap[i] = float((prec * topR).sum() / R)
    return ap


def macro_map_at_r(D: np.ndarray, labels, eligible: set, gallery_mask=None) -> dict:
    """Macro mAP@R over eligible (n>=4) classes + anchor coverage."""
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
        "anchor_coverage": covered / total if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Galleries
# ---------------------------------------------------------------------------

def controlled_gallery_mask(labels, content_sim: np.ndarray) -> np.ndarray:
    """Per anchor: all same-class positives + the R hardest content-confounded
    negatives (most content-similar different-class clips). Positives are never
    deleted; content_sim is label-free, so the construction is clean under
    label permutation."""
    lab = np.asarray(labels)
    n = len(lab)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        same = (lab == lab[i])
        same[i] = False
        R = int(same.sum())
        mask[i] |= same
        if R == 0:
            continue
        diff = np.flatnonzero(lab != lab[i])
        if len(diff) == 0:
            continue
        hardest = diff[np.argsort(-content_sim[i, diff])[:R]]
        mask[i, hardest] = True
    return mask


def random_gallery_mask(labels, rng: np.random.Generator) -> np.ndarray:
    """Controlled-gallery shape with RANDOM different-class negatives — isolates
    confound susceptibility from gallery shrinkage."""
    lab = np.asarray(labels)
    n = len(lab)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        same = (lab == lab[i])
        same[i] = False
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


# ---------------------------------------------------------------------------
# Hubness (reliability field; frozen thresholds above)
# ---------------------------------------------------------------------------

def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m, s = x.mean(), x.std()
    return 0.0 if s < 1e-12 else float(((x - m) ** 3).mean() / s ** 3)


def _k_occurrence(D: np.ndarray, k: int) -> np.ndarray:
    M = _rank_matrix(D)
    n = len(M)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        finite = np.flatnonzero(np.isfinite(M[i]))
        if finite.size == 0:
            continue
        nn = finite[np.argsort(M[i, finite], kind="stable")[:min(k, finite.size)]]
        counts[nn] += 1
    return counts


def hubness_field(D: np.ndarray, labels, k: int = HUB_GATING_K) -> dict:
    """k-occurrence skew + 1-NN prediction-column collapse, gated on the frozen
    thresholds. A sink column fails regardless of accuracy."""
    M = _rank_matrix(D)
    lab = np.asarray(labels)
    classes = sorted(set(labels))
    valid = np.isfinite(M).any(axis=1)
    nk = _k_occurrence(D, k)
    preds = [labels[int(np.argmin(M[i]))] for i in range(len(M)) if valid[i]]
    counts = np.array([preds.count(c) for c in classes], dtype=float)
    p = counts / counts.sum() if counts.sum() else counts
    nz = p[p > 0]
    H_norm = float(-(nz * np.log(nz)).sum() / np.log(len(classes))) if len(classes) > 1 else 0.0
    skew = _skew(nk)
    max_share = float(counts.max() / counts.sum()) if counts.sum() else 0.0
    sink = classes[int(np.argmax(counts))] if counts.sum() else None
    hub_pass = bool(skew <= HUB_MAX_SKEW and H_norm >= HUB_MIN_PRED_ENTROPY
                    and max_share <= HUB_MAX_PRED_SHARE)
    return {"hubness_pass": hub_pass, "skew": skew, "entropy": H_norm,
            "maxpred": max_share, "hub_class": sink,
            "hub_share": float(nk.max() / max(nk.sum(), 1)),
            "coverage": float(valid.mean())}


# ---------------------------------------------------------------------------
# The datasheet
# ---------------------------------------------------------------------------

def full_datasheet(D: np.ndarray, labels, content_sim: np.ndarray,
                   eligible: set, n_perm: int = 1000, n_boot: int = 400,
                   rng_seed: int = 0, hub_k: int = HUB_GATING_K,
                   content_sim_alt: np.ndarray | None = None) -> dict:
    """The instrument datasheet for one metric matrix (see module docstring for
    the gate's definition and history). Ported verbatim from the health-validated
    implementation; verified to reproduce all six 2026-07-16 panel verdicts at
    n_perm=1000 before the v4 freeze."""
    lab = np.asarray(labels)
    classes = sorted(eligible)
    gm = controlled_gallery_mask(labels, content_sim)

    oc = macro_map_at_r(D, labels, eligible, gm)
    ou = macro_map_at_r(D, labels, eligible)
    rng = np.random.default_rng(rng_seed)
    nc_macro, nu_macro = np.empty(n_perm), np.empty(n_perm)
    nc_cls = {c: np.empty(n_perm) for c in classes}
    for p in range(n_perm):
        perm = rng.permutation(len(lab))
        pl = list(lab[perm])
        gmp = controlled_gallery_mask(pl, content_sim)
        rc = macro_map_at_r(D, pl, eligible, gmp)
        ru = macro_map_at_r(D, pl, eligible)
        nc_macro[p], nu_macro[p] = rc["macro_map"], ru["macro_map"]
        for c in classes:
            nc_cls[c][p] = rc["per_class"][c]

    def sk(x, nm):
        return float((x - nm) / (1 - nm)) if nm < 1 and np.isfinite(x) else float("nan")

    ncm, num = float(np.nanmean(nc_macro)), float(np.nanmean(nu_macro))
    Cn, U = sk(oc["macro_map"], ncm), sk(ou["macro_map"], num)
    z_c = float((oc["macro_map"] - ncm) / (np.nanstd(nc_macro) + 1e-12))
    p_c = float((1 + (nc_macro >= oc["macro_map"]).sum()) / (1 + n_perm))

    n = len(lab)
    mm = int(0.8 * n)
    dev = []
    for _ in range(n_boot):
        s = rng.choice(n, size=mm, replace=False)
        r = macro_map_at_r(D[np.ix_(s, s)], list(lab[s]), eligible,
                           controlled_gallery_mask(list(lab[s]), content_sim[np.ix_(s, s)]))["macro_map"]
        if np.isfinite(r):
            dev.append(np.sqrt(mm / n) * (r - oc["macro_map"]))
    dev = np.array(dev)
    boot_lo = float(oc["macro_map"] + np.quantile(dev, 0.025))
    floor = 2.0 * float(np.nanstd(nc_macro))
    above_chance = bool((boot_lo - ncm > floor) and (p_c < 0.01))   # OLD gate (no-op) -> descriptive

    # --- the causal-excess gate over B = D_cont ---
    def _controlled_skill(Db):
        obs = macro_map_at_r(Db, labels, eligible, gm)["macro_map"]
        rb = np.random.default_rng(rng_seed + 4242)
        nb = np.empty(n_perm)
        for p in range(n_perm):
            perm = rb.permutation(len(lab))
            pl = list(lab[perm])
            nb[p] = macro_map_at_r(Db, pl, eligible, controlled_gallery_mask(pl, content_sim))["macro_map"]
        nmb = float(np.nanmean(nb))
        return obs, sk(obs, nmb)

    D_cont = 1.0 - np.asarray(content_sim, float)
    np.fill_diagonal(D_cont, 0.0)
    ocB, Cn_B = _controlled_skill(D_cont)
    causal_excess = float(Cn - Cn_B)
    paired_diff = float(oc["macro_map"] - ocB)
    rngp = np.random.default_rng(rng_seed + 777)
    dev2 = []
    for _ in range(n_boot):
        s = rngp.choice(n, size=mm, replace=False)
        cs = content_sim[np.ix_(s, s)]
        gms = controlled_gallery_mask(list(lab[s]), cs)
        dd = (macro_map_at_r(D[np.ix_(s, s)], list(lab[s]), eligible, gms)["macro_map"]
              - macro_map_at_r(D_cont[np.ix_(s, s)], list(lab[s]), eligible, gms)["macro_map"])
        if np.isfinite(dd):
            dev2.append(np.sqrt(mm / n) * (dd - paired_diff))
    paired_lo95 = float(paired_diff + np.quantile(np.array(dev2), 0.025)) if dev2 else float("nan")
    causal_gate_thr = float(max(0.10, floor))
    causal_PASS = bool(causal_excess >= causal_gate_thr and paired_lo95 > 0)

    causal_excess_maxproxy = None
    if content_sim_alt is not None:
        D_alt = 1.0 - np.asarray(content_sim_alt, float)
        np.fill_diagonal(D_alt, 0.0)
        _, Cn_alt = _controlled_skill(D_alt)
        causal_excess_maxproxy = float(Cn - max(Cn_B, Cn_alt))

    rr = np.random.default_rng(rng_seed + 909)
    obs_rand = np.nanmean([macro_map_at_r(D, labels, eligible, random_gallery_mask(labels, rr))["macro_map"]
                           for _ in range(10)])
    nrand = np.empty(max(1, n_perm // 3))
    for p in range(len(nrand)):
        perm = rr.permutation(len(lab))
        pl = list(lab[perm])
        nrand[p] = macro_map_at_r(D, pl, eligible, random_gallery_mask(np.asarray(pl), rr))["macro_map"]
    C_rand = sk(float(obs_rand), float(np.nanmean(nrand)))
    confound_susceptibility = float(Cn - C_rand)

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
        trust[c] = {"controlled_map": obsc,
                    "skill": sk(obsc, nmc) if np.isfinite(obsc) else float("nan"),
                    "delta": (sk(obsc, nmc) - sk(ou["per_class"][c], num)) if np.isfinite(obsc) else float("nan"),
                    "n": npos, "stamp": stamp}

    return {
        "reliability": hubness_field(D, labels, k=hub_k),
        "validity": {"uncontrolled_map": ou["macro_map"], "U_skill": U, "null": num},
        "causal": {"controlled_map": oc["macro_map"], "Cn_skill": Cn, "null": ncm,
                   "z": z_c, "p": p_c, "boot_lo": boot_lo, "floor": floor,
                   "content_baseline_Cn": Cn_B, "causal_excess": causal_excess,
                   "paired_diff": paired_diff, "paired_lo95": paired_lo95,
                   "causal_gate_thr": causal_gate_thr, "causal_PASS": causal_PASS,
                   "causal_excess_maxproxy": causal_excess_maxproxy,
                   "confound_susceptibility": confound_susceptibility,
                   "above_chance": above_chance},
        "shortcut": {"delta_macro": (Cn - U),
                     "note": "delta=Cn-U; >=0 content-robust, <<0 shortcut"},
        "trust_map": trust,
        "coverage": {"anchor_cov_controlled": oc["anchor_coverage"],
                     "classes_defined": oc["n_classes_defined"],
                     "classes_eligible": len(classes)},
    }


# ---------------------------------------------------------------------------
# Bar 9 grader (self-verifying: headliners pass AND falsifiers fail)
# ---------------------------------------------------------------------------

def grade_bar9(headline_sheets: dict, control_sheets: dict) -> dict:
    """Bar 9: every headline metric's causal_PASS is True AND every negative
    control's causal_PASS is False. A passing falsifier means the gate itself is
    broken — the bar fails on it exactly as on a failing headline metric."""
    metrics = {k: bool(v["causal"]["causal_PASS"]) for k, v in headline_sheets.items()}
    controls = {k: bool(v["causal"]["causal_PASS"]) for k, v in control_sheets.items()}
    ok = all(metrics.values()) and not any(controls.values())
    return {"pass": bool(ok), "metrics_causal_PASS": metrics,
            "controls_causal_PASS": controls,
            "form": "all headline causal_PASS AND all negative controls fail"}
