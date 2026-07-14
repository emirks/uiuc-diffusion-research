"""Hubness gate (RUNBOOK §1.4) — mandatory per candidate space/descriptor.

    "Compute prediction-column entropy and k-occurrence skew on the exam's
     distance matrix. A candidate with a sink column (M1c/polygon pattern)
     fails regardless of accuracy."

This module does not exist anywhere in the certified tree — it is new code, and
the RUNBOOK states the gate without stating its numbers. The thresholds are
therefore CALIBRATED ON THE INCUMBENTS ALONE, before any candidate exists:
m1c_object is the known-dead positive control (its polygon sink is the pattern
§1.4 names), m1a__v3_sided is the known-healthy negative control. That is
corpus-only, outcome-independent calibration — permitted pre-freeze under the
two-kind calibration rule — and the resulting numbers are frozen in gates.yaml
before any candidate descriptor is computed.

The statistics, on a symmetric distance matrix with NaN = undefined:

  k-occurrence N_k(i): how many of the other clips have i in their k nearest
    neighbours. In a healthy space N_k concentrates around k; a hub/sink absorbs
    neighbourhoods and the distribution grows a long right tail. Reported as
    Hubness skew S_k = skew(N_k) (Radovanovic's statistic) and as the share of
    all k-NN slots taken by the single greediest clip.

  Prediction-column entropy: the 1-NN predicted-CLASS distribution (the column
  of predictions the exam's LOO 1-NN emits), normalised by log(n_classes). A
  sink column drives most clips to predict one class, collapsing this toward 0.
  Reported with the largest single-class prediction share.

Undefined rows (all-NaN) never vote and are excluded from the denominators; they
are counted and reported (RUNBOOK §1.5 — an undefined clip is not a clip that
agrees with you).
"""

from __future__ import annotations

import numpy as np


def _masked(D: np.ndarray) -> np.ndarray:
    """The exam kernel's masking, verbatim: self-distance and NaN are +inf."""
    M = D.copy().astype(float)
    np.fill_diagonal(M, np.inf)
    M[np.isnan(M)] = np.inf
    return M


def skew(x: np.ndarray) -> float:
    """Fisher-Pearson moment coefficient of skewness (population form) — the
    hubness statistic. Zero variance (every clip equally popular) is skew 0."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(((x - m) ** 3).mean() / s ** 3)


def k_occurrence(D: np.ndarray, k: int = 10) -> np.ndarray:
    """N_k(i) — the number of clips whose k nearest neighbours include i.

    Rows with fewer than k finite distances contribute only the neighbours they
    have; fully undefined rows contribute nothing (they retrieve no one)."""
    M = _masked(D)
    n = len(M)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        finite = np.flatnonzero(np.isfinite(M[i]))
        if finite.size == 0:
            continue                                  # undefined row: casts no votes
        kk = min(k, finite.size)
        nn = finite[np.argsort(M[i, finite], kind="stable")[:kk]]
        counts[nn] += 1
    return counts


def prediction_column(D: np.ndarray, labels: list[str]) -> tuple[list, np.ndarray]:
    """The exam's 1-NN prediction column (None where the row is undefined)."""
    M = _masked(D)
    valid = np.isfinite(M).any(axis=1)
    pred = [labels[int(np.argmin(M[i]))] if valid[i] else None for i in range(len(M))]
    return pred, valid


GATING_K = 10               # frozen: the k whose skew gates (§1.4)
DIAGNOSTIC_KS = (1, 5, 10)  # persisted for k-sensitivity; only GATING_K gates


def hubness_stats(D: np.ndarray, labels: list[str], k: int = GATING_K) -> dict:
    """Everything §1.4 asks for, on one candidate distance matrix.

    Computed on the SAME matrix the exam grades, over the defined subpool only,
    with denominators explicit and NO coverage correction — a renormalization
    would be a new bar form. Definedness gates (§3.2/§3.3) are SUPPOSED to remove
    near-static clips, and a shrunken pool mechanically concentrates hub
    statistics; that is a property of the candidate, not an artifact to correct
    away. Coverage therefore sits beside every stat, always."""
    n = len(labels)
    classes = sorted(set(labels))

    nk = k_occurrence(D, k=k)
    total_slots = int(nk.sum())
    pred, valid = prediction_column(D, labels)
    graded = [p for p in pred if p is not None]

    # predicted-class distribution -> normalised entropy
    counts = np.array([graded.count(c) for c in classes], dtype=float)
    p = counts / counts.sum() if counts.sum() else counts
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    H_norm = float(H / np.log(len(classes))) if len(classes) > 1 else 0.0

    # who absorbs the neighbourhoods
    top_clip = int(np.argmax(nk))
    top_class = classes[int(np.argmax(counts))] if counts.sum() else None
    return {
        "k": k,
        "n_clips": n,
        "n_undefined_rows": int((~valid).sum()),
        "coverage": float(valid.mean()) if n else 0.0,
        "hubness_skew": skew(nk),
        # non-gating k-sensitivity columns (§1.4 mandates persisting stats).
        # ONLY hubness_skew at GATING_K enters a verdict; these never do.
        "skew_by_k_diagnostic": {str(kk): skew(k_occurrence(D, k=kk))
                                 for kk in DIAGNOSTIC_KS},
        "k_occurrence_max": int(nk.max()) if n else 0,
        "k_occurrence_mean": float(nk.mean()) if n else 0.0,
        "max_clip_knn_share": float(nk.max() / total_slots) if total_slots else 0.0,
        "hub_clip_index": top_clip,
        "pred_entropy_norm": H_norm,
        "max_pred_class_share": float(counts.max() / counts.sum()) if counts.sum() else 0.0,
        "sink_class": top_class,
        "n_graded": len(graded),
    }


def gate(stats: dict, gates: dict) -> dict:
    """The §1.4 verdict — mechanical, against frozen numbers in gates.yaml.

    A candidate FAILS if the k-occurrence distribution is skewed past the frozen
    bound (a clip absorbing neighbourhoods) OR the prediction column has
    collapsed past its bounds (a class absorbing predictions — the M1c/polygon
    sink). OR-structured on purpose: the two instruments see different failures.
    The prediction column is a k=1 phenomenon, so rank-2..10 absorption casts no
    1-NN votes and is invisible to it — which is exactly how the incumbent
    m1b_camera decouples (skew 2.52, entropy 0.917). §3.6's kill rule ("if
    M1c_flow STILL EXHIBITS A HUB after the energy gate → the descriptor is
    dead") is k-occurrence language and has no operational referent without the
    skew conjunct.

    A failed hubness gate is TERMINAL regardless of accuracy (§1.4), and it
    applies to every candidate space/descriptor — motion and appearance alike
    (§7 condition 3).

    Gap-band disclosure: the thresholds were placed at the midpoint of an empty
    band between the pass and dead incumbent populations. A candidate landing
    INSIDE that band gets the frozen verdict — terminal, no re-adjudication — but
    it is flagged so the report can state that the verdict is sensitive to
    threshold placement. Judging whether the placement was right is owner-side
    review, never the executor's."""
    g = gates["hubness"]
    checks = {
        "skew_ok": bool(stats["hubness_skew"] <= g["max_hubness_skew"]),
        "entropy_ok": bool(stats["pred_entropy_norm"] >= g["min_pred_entropy_norm"]),
        "pred_share_ok": bool(stats["max_pred_class_share"] <= g["max_pred_class_share"]),
    }
    band = g.get("calibration_band", {})
    in_band = {}
    for stat, key in (("hubness_skew", "skew"),
                      ("pred_entropy_norm", "pred_entropy_norm"),
                      ("max_pred_class_share", "max_pred_class_share")):
        b = band.get(key)
        if b:
            in_band[stat] = bool(b[0] <= stats[stat] <= b[1])
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {k: v for k, v in g.items() if k != "calibration_band"},
        "landed_in_calibration_band": bool(any(in_band.values())),
        "band_membership": in_band,
        "calibration_band": band,
        "stats": stats,
    }


def main() -> int:
    """Run the FROZEN gate against the six incumbent matrices and persist the
    calibration artifact. The gate must reproduce RUNBOOK §0's diagnosis exactly:
    m1c_object and m_incumbent FAIL (hub-collapsed), the m1a family and
    m1b_camera PASS. If it does not, the gate is miscalibrated and no candidate
    may be judged by it."""
    import json

    import numpy as _np

    from . import paths

    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    labels = paths.labels_of(corpus, keys)
    gates = paths.load_gates()
    z = _np.load(paths.NPZ)

    expected = {"m1a__v3_sided": True, "m1a__v2_envelope": True, "m1a__all_frames": True,
                "m1b_camera": True, "m_incumbent": False, "m1c_object": False}
    out, errors = {}, []
    print(f"{'metric':18s} {'skew':>7s} {'H_norm':>7s} {'max_pred':>9s} {'cov':>6s} "
          f"{'verdict':>8s} {'expected':>9s}  sink")
    for name, should_pass in expected.items():
        s = hubness_stats(z[name], labels, k=gates["hubness"]["gating_k"])
        v = gate(s, gates)
        out[name] = v
        ok = "OK" if v["pass"] == should_pass else "MISMATCH"
        print(f"{name:18s} {s['hubness_skew']:7.3f} {s['pred_entropy_norm']:7.4f} "
              f"{s['max_pred_class_share']:9.4f} {s['coverage']:6.3f} "
              f"{'PASS' if v['pass'] else 'FAIL':>8s} {'PASS' if should_pass else 'FAIL':>9s}  "
              f"{s['sink_class'] if not v['pass'] else ''} [{ok}]")
        if v["pass"] != should_pass:
            errors.append(f"{name}: gate says pass={v['pass']}, RUNBOOK §0 says {should_pass}")
    if errors:
        print("STOP: the frozen hubness gate does not reproduce RUNBOOK §0's diagnosis:")
        for e in errors:
            print("  " + e)
        return 1
    print("[step0] frozen hubness gate reproduces RUNBOOK §0 on all six incumbents")

    d = paths.WB_OUT / "step0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "hubness_incumbents.json").write_text(json.dumps({
        "gating_k": gates["hubness"]["gating_k"],
        "thresholds": {k: v for k, v in gates["hubness"].items()
                       if k in ("max_hubness_skew", "min_pred_entropy_norm",
                                "max_pred_class_share")},
        "derivation": "midpoint of the empty gap between the pass and dead "
                      "incumbent populations (deployed convention: "
                      "certify.probes.grade_splices tau_copy)",
        "authority": "RUNBOOK §0 — M1c is 'hub-collapsed (polygon column = sink "
                     "artifact)'; M1b is merely weak ('discrimination only inside "
                     "camera-tagged strata')",
        "incumbents": out,
    }, indent=1, default=str))
    print(f"[step0] wrote {d / 'hubness_incumbents.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
