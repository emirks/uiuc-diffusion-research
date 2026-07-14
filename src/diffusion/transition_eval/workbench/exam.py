"""The workbench's own exam driver.

Candidates are NEVER run through certify/run_certification.py or certify/exam.py
(OPERATIONS §1 prohibition 2) — those are the certified instrument's, their bars
are frozen, and appending a candidate id to them would de-register them. But the
STATISTIC is imported, not reimplemented: report.retrieval_eval is the frozen exam
kernel (LOO 1-NN + Cohen's d, NaN = undefined, coverage accounting), and
certify.diagnostics supplies the per-clip rows. A candidate is therefore judged by
exactly the function that judged the incumbent — which is the only reason a
head-to-head against RUNBOOK §B means anything.

Every candidate gets, in one place and always together (OPERATIONS §7):
  accuracy + Cohen's d + COVERAGE (an accuracy win on a shrunken support is not a
  win), misretrieved under the pinned convention, the §1.4 hubness verdict, the
  §3.6 stratum recalls, a definedness report, and the per-clip margins.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from ..certify import diagnostics
from ..report import retrieval_eval
from ..s_structure import core_mask_v3
from . import hubness, paths

SEAM_PAIRS = (8, 112)     # structural: n_prefix-1 and T-n_suffix-1 (endpoints.seam_scores)


def core_pair_mask(bundle: dict, sidedness: str) -> np.ndarray:
    """Which of a clip's 120 adjacent frame pairs lie inside the S-mask core.

    A PAIR is core when BOTH its frames are core — a step with one foot in the
    conditioned window is not a step through the effect. §3.1 also excludes the
    M3b seam frames from all fits; those sit at the conditioning handoffs
    (pair 8 = frames 8->9, pair 112 = frames 112->113), so requiring both frames to
    be core already removes them. Asserted rather than assumed."""
    mask, _ = core_mask_v3(bundle["profile"], sidedness)
    pairs = mask[:-1] & mask[1:]
    for s in SEAM_PAIRS:
        if s < len(pairs):
            assert not pairs[s], f"seam pair {s} survived the core mask"
    return pairs


def stratum_recall(per_class_recall: dict, stratum: set[str], eligible: set[str]) -> dict:
    """The frozen §3.6 statistic (gates.yaml conventions.stratum_recall): macro mean
    of per-class recall over (stratum INTERSECT n>=4-eligible), a NaN class counted
    as 0.0 rather than dropped — dropping would let a candidate raise its score by
    NaN-ing its hardest classes."""
    cells = sorted(stratum & eligible)
    vals, nan_classes = [], []
    for c in cells:
        r = per_class_recall.get(c)
        if r is None or not np.isfinite(r):
            vals.append(0.0)
            nan_classes.append(c)
        else:
            vals.append(float(r))
    return {
        "value": float(np.mean(vals)) if vals else float("nan"),
        "n_classes": len(cells),
        "n_classes_nan": len(nan_classes),
        "nan_classes": nan_classes,
        "class_coverage": float(1 - len(nan_classes) / len(cells)) if cells else 0.0,
        "per_class": {c: v for c, v in zip(cells, vals)},
    }


def definedness_report(D: np.ndarray, labels: list[str], reasons: list[str | None]) -> dict:
    """Why each undefined row is undefined — counted, never quietly dropped (§1.5)."""
    M = D.copy().astype(float)
    np.fill_diagonal(M, np.inf)
    M[np.isnan(M)] = np.inf
    undefined = ~np.isfinite(M).any(axis=1)
    by_reason: dict[str, int] = {}
    for i in np.flatnonzero(undefined):
        r = reasons[i] or "unspecified"
        by_reason[r] = by_reason.get(r, 0) + 1
    return {
        "n_clips": len(labels),
        "n_undefined_rows": int(undefined.sum()),
        "coverage": float(1 - undefined.mean()),
        "undefined_by_reason": by_reason,
        "nan_cell_fraction": float(np.isnan(D).mean()),
    }


def evaluate(name: str, D: np.ndarray, keys: list[str], labels: list[str],
             gates: dict, corpus_facts: dict, reasons: list[str | None] | None = None,
             stratum: str | None = None) -> dict:
    """One candidate, judged by the frozen kernel + every mandatory companion."""
    r = retrieval_eval(D, labels)
    rows = diagnostics.per_clip_rows(D, keys, labels)
    mis = sum(1 for x in rows if x["pred"] != x["label"])     # pinned convention

    hub_stats = hubness.hubness_stats(D, labels, k=gates["hubness"]["gating_k"])
    hub = hubness.gate(hub_stats, gates)

    eligible = set(corpus_facts["eligible_n_ge_4"])
    strata = {"camera": set(corpus_facts["camera_classes"]),
              "object": set(corpus_facts["object_classes"])}
    recalls = {s: stratum_recall(r["per_class_recall"], cls, eligible)
               for s, cls in strata.items()}

    return {
        "metric": name,
        "accuracy_1nn": r["accuracy_1nn"],
        "separation_cohens_d": r["separation_cohens_d"],
        "coverage": r["coverage"],
        "misretrieved": mis,
        "n_clips": len(labels),
        "chance": r["chance"],
        "accuracy_wilson95": list(r["accuracy_wilson95"]),
        "per_class_recall": r["per_class_recall"],
        "stratum_recalls": recalls,
        "primary_stratum": stratum,
        "hubness": hub,
        "definedness": definedness_report(D, labels,
                                          reasons or [None] * len(labels)),
        "rows": rows,
    }


def verdict_vs_incumbent(cand: dict, incumbent: dict, gates: dict,
                         stratum: str) -> dict:
    """§3.6, mechanically: beats the incumbent on Cohen's d AND within-stratum
    recall, and passes the §1.4 hubness gate. Each condition is a computed FACT;
    the adoption call itself is owner-side (OPERATIONS §8)."""
    tgt = gates["stratum_targets"][stratum]
    d_inc = incumbent["cohens_d"]
    r_inc = tgt["value"] if "value" in tgt else tgt["value"]
    cand_recall = cand["stratum_recalls"][stratum]["value"]
    conds = {
        "beats_cohens_d": {
            "candidate": cand["separation_cohens_d"], "incumbent": d_inc,
            "pass": bool(cand["separation_cohens_d"] > d_inc)},
        "beats_stratum_recall": {
            "candidate": cand_recall, "incumbent": r_inc, "stratum": stratum,
            "pass": bool(np.isfinite(cand_recall) and cand_recall > r_inc)},
        "hubness_gate": {"pass": bool(cand["hubness"]["pass"]),
                         "stats": cand["hubness"]["checks"]},
    }
    return {
        "conditions": conds,
        "all_pass": bool(all(c["pass"] for c in conds.values())),
        "rule": "RUNBOOK §3.6 — beats incumbent M1b/M1c on Cohen's d AND "
                "within-stratum recall -> tier upgrade proposed (owner's call, "
                "v3.1 re-cert). Misses -> stays analysis-tier, NO second attempt "
                "this cycle. Hub after the energy gate -> the descriptor is dead.",
    }


def save(out_dir: pathlib.Path, name: str, D: np.ndarray, result: dict,
         extra: dict | None = None) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}_distance_matrix.npz", D=D)
    payload = {k: v for k, v in result.items() if k != "rows"}
    payload["per_clip_rows"] = result["rows"]
    if extra:
        payload.update(extra)
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=1, default=str))
    return out_dir / f"{name}.json"


def summary_line(r: dict) -> str:
    h = r["hubness"]
    return (f"{r['metric']:14s} acc {r['accuracy_1nn']:.4f}  d {r['separation_cohens_d']:.4f}  "
            f"cov {r['coverage']:.4f}  mis {r['misretrieved']}/{r['n_clips']}  "
            f"hub {'PASS' if h['pass'] else 'FAIL'} (skew {h['stats']['hubness_skew']:.2f}, "
            f"H {h['stats']['pred_entropy_norm']:.3f}, "
            f"maxpred {h['stats']['max_pred_class_share']:.3f})")
