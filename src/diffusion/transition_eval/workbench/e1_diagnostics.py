"""E1 instrument diagnostics — LABELLED DIAGNOSTICS, NOT CANDIDATE SCORES.

    PURPOSE. The §4.1 kill rule fired on E1 and the verdict is RECORDED, TERMINAL,
    and NOT REVISED by anything in this file. This module exists only so that
    owner-side review never needs a re-run to adjudicate the escalated
    owner-reserved matter (OPERATIONS §8: reports must be complete enough that the
    reviewer never needs a re-run to make the call).

    NOT a candidate. NOT a rescue variant. NOT a second attempt. NO row here is
    scored against §4.1, §1.4 or §7, and no row carries a verdict. Nothing here
    feeds E2/E3 or any gate — E2/E3 do not run, per the recorded KILL.

    The swept grid was PRE-DECLARED in CONSULTATIONS.md (C3) BEFORE being computed;
    that declaration is what makes this a diagnostic rather than a search:

        eig_floor_ratio in {1e-6 (REGISTERED), 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}

    Only that scalar is swept — the single whitening parameter RUNBOOK §1.1 leaves
    free (it mandates whitening; it does not pin the regularization). The ZCA fit
    population is NOT swept: §1.1's parenthetical pins it to the S-mask core
    frames, and sweeping it would relitigate a frozen scientific choice. No new
    whitener families. No floor is recommended or selected — selecting one is
    owner-reserved.

    The 1e-6 row is the REGISTERED configuration and is the only row that is the
    §4.1 candidate.
"""

from __future__ import annotations

import json

import numpy as np

from ..certify import diagnostics
from ..report import retrieval_eval
from ..s_structure import core_mask_v3
from . import bundles, hubness, nulls, paths, whitening

GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]     # pre-declared (C3)
REGISTERED = 1e-6


def _stats(V: list, labels: list[str], keys: list[str]) -> dict:
    n = len(V)
    D = np.full((n, n), np.nan)
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.linalg.norm(V[i] - V[j]))
    r = retrieval_eval(D, labels)
    h = hubness.hubness_stats(D, labels)
    rows = diagnostics.per_clip_rows(D, keys, labels)
    return {
        "accuracy_1nn": r["accuracy_1nn"],
        "cohens_d": r["separation_cohens_d"],
        "coverage": r["coverage"],
        "misretrieved": sum(1 for x in rows if x["pred"] != x["label"]),
        "hubness_skew": h["hubness_skew"],
        "pred_entropy_norm": h["pred_entropy_norm"],
        "max_pred_class_share": h["max_pred_class_share"],
        "sink_class": h["sink_class"],
    }, D, rows, h


def main() -> int:
    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    labels = paths.labels_of(corpus, keys)
    side = paths.sidedness_of(corpus, keys)
    bs = bundles.load_corpus_bundles(keys)

    # raw (floor-independent) means, computed once
    Craw, Nraw, cores = [], [], []
    for b, s, k in zip(bs, side, keys):
        nf = nulls.load_null_feats(paths.clip_path(k), paths.WB_CACHE)
        m, _ = core_mask_v3(b["profile"], s)
        idx = np.flatnonzero(m)
        cores.append((b["feats"][idx], nf[idx]))
        Craw.append(b["feats"][idx].mean(0))
        Nraw.append(nf[idx].mean(0))
    Craw, Nraw = np.stack(Craw), np.stack(Nraw)

    X = whitening.core_frames(bs, side)          # the §1.1 fit population (NOT swept)
    mu = X.mean(axis=0)
    Xc = X - mu
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    C = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(C)
    lam = np.clip(lam, 0.0, None)

    out = {
        "purpose": ("DIAGNOSTIC ONLY. The §4.1 verdict is recorded, terminal, and "
                    "NOT revised by anything here. Not a candidate, not a rescue "
                    "variant, not a second attempt. No row is scored against §4.1, "
                    "§1.4 or §7; no row carries a verdict. Nothing here feeds E2/E3 "
                    "or any gate. The grid was PRE-DECLARED in CONSULTATIONS.md (C3) "
                    "before being computed. No floor is recommended or selected — "
                    "that is owner-reserved."),
        "provenance": {
            "verdict_commit": "72a5bd4",
            "gates_freeze_commit": "694afc7",
            "registered_eig_floor_ratio": REGISTERED,
            "note": ("RUNBOOK §1.1 mandates whitening but does NOT pin its "
                     "regularization. eig_floor_ratio was an executor-chosen free "
                     "parameter, frozen in good faith before any candidate ran."),
        },
        "zca_spectrum": {
            "dim": int(len(lam)),
            "eig_max": float(lam.max()),
            "eig_min": float(lam.min()),
            "condition_number": float(lam.max() / max(lam.min(), 1e-300)),
            "quantiles": {f"q{q}": float(np.quantile(lam, q / 100))
                          for q in (1, 5, 10, 25, 50, 75, 90, 99)},
            "cumulative_variance_dims": {
                f"dims_for_{p}pct": int(np.searchsorted(
                    np.cumsum(lam[::-1]) / lam.sum(), p / 100) + 1)
                for p in (50, 80, 90, 95, 99)},
        },
        "raw_reference_rows_floor_independent": {},
        "sweep": [],
    }

    # raw arms — floor-independent reference rows
    s_ctl, _, _, _ = _stats(list(Craw), labels, keys)
    s_dlt, _, _, _ = _stats(list(Craw - Nraw), labels, keys)
    out["raw_reference_rows_floor_independent"] = {
        "raw_no_subtraction_control": s_ctl,
        "raw_delta": s_dlt,
    }
    print("RAW reference rows (floor-independent):")
    print(f"  no-subtraction control  acc {s_ctl['accuracy_1nn']:.4f} d {s_ctl['cohens_d']:.4f} "
          f"mis {s_ctl['misretrieved']}/223 H {s_ctl['pred_entropy_norm']:.3f}")
    print(f"  delta                   acc {s_dlt['accuracy_1nn']:.4f} d {s_dlt['cohens_d']:.4f} "
          f"mis {s_dlt['misretrieved']}/223 H {s_dlt['pred_entropy_norm']:.3f}")

    print("\nFLOOR SWEEP (pre-declared grid; NO verdicts, NO pass/fail):")
    hdr = (f"{'floor':>8s} {'flrd':>5s} | {'CONTROL (no subtraction)':^42s} | "
           f"{'DELTA (E1 form)':^42s} | {'norm ratio':>10s}")
    print(hdr)
    print(f"{'':>8s} {'':>5s} | {'acc':>6s} {'d':>6s} {'mis':>7s} {'H':>6s} {'skew':>6s} "
          f"{'':>6s} | {'acc':>6s} {'d':>6s} {'mis':>7s} {'H':>6s} {'skew':>6s} {'':>6s} | "
          f"{'null/clip':>10s}")
    for ratio in GRID:
        floor = ratio * float(lam.max())
        lam_f = np.maximum(lam, floor)
        W = (V * (1.0 / np.sqrt(lam_f))) @ V.T
        z = {"W": W, "mean": mu}

        Cw = np.stack([whitening.whiten(z, c).mean(0) for c, _ in cores])
        Nw = np.stack([whitening.whiten(z, n).mean(0) for _, n in cores])
        ctl, _, _, _ = _stats(list(Cw), labels, keys)
        dlt, _, _, _ = _stats(list(Cw - Nw), labels, keys)
        nr = float(np.linalg.norm(Nw, axis=1).mean() / np.linalg.norm(Cw, axis=1).mean())
        n_flr = int((lam < floor).sum())

        tag = "  <-- REGISTERED (the §4.1 candidate)" if ratio == REGISTERED else ""
        out["sweep"].append({
            "eig_floor_ratio": ratio, "eig_floor": floor, "n_floored": n_flr,
            "effective_dims": int(len(lam) - n_flr),
            "whitened_no_subtraction_control": ctl,
            "whitened_delta": dlt,
            "whitened_null_to_clip_norm_ratio": nr,
            "registered": bool(ratio == REGISTERED),
        })
        print(f"{ratio:8.0e} {n_flr:5d} | {ctl['accuracy_1nn']:6.4f} {ctl['cohens_d']:6.3f} "
              f"{ctl['misretrieved']:4d}/223 {ctl['pred_entropy_norm']:6.3f} "
              f"{ctl['hubness_skew']:6.2f} {'':>6s} | "
              f"{dlt['accuracy_1nn']:6.4f} {dlt['cohens_d']:6.3f} {dlt['misretrieved']:4d}/223 "
              f"{dlt['pred_entropy_norm']:6.3f} {dlt['hubness_skew']:6.2f} {'':>6s} | "
              f"{nr:10.3f}{tag}")

    # per-clip distributions the reviewer may want
    cos_raw = np.array([float(c @ n / (np.linalg.norm(c) * np.linalg.norm(n)))
                        for c, n in zip(Craw, Nraw)])
    out["per_clip_distributions"] = {
        "cos_clip_null_raw": {"mean": float(cos_raw.mean()), "min": float(cos_raw.min()),
                              "max": float(cos_raw.max()),
                              "quantiles": {f"q{q}": float(np.quantile(cos_raw, q / 100))
                                            for q in (5, 25, 50, 75, 95)}},
    }

    d = paths.WB_OUT / "e1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "e1_floor_sensitivity.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {d / 'e1_floor_sensitivity.json'}")
    print("NO floor is recommended or selected — that is owner-reserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
