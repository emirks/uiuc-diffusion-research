"""E1' driver — the final workbench cycle (E1PRIME_DIRECTIVE.md).

Order is not negotiable and is enforced here:

  calibrate  the raw-space min-D floor. CORPUS-ONLY (it reads the frozen corpus and
             nothing else; no candidate signature or distance exists yet), frozen in its
             own commit before any candidate distance — the same discipline gates.yaml
             used for texture_percentile and the energy gate.
  run        IV1/IV2 FIRST (§2.3), then the single gating arm and the three diagnostics
             (§2.2's closed list — nothing else), then the kill rule, the §7 conditions,
             the hubness gate and P1/P2/P4.

Every verdict here is mechanical. The executor computes facts; it does not make the
adoption call, does not adjust a threshold, and does not choose among outcomes (§2.6:
all four close the workbench).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..certify.probes import endpoint_vecs
from . import bundles, curves, e1prime, exam, hubness, iv, lw, nulls, paths, whitening
from ..s_structure import core_mask_v3

OUT = paths.WB_OUT / "e1prime"
FLOOR_JSON = OUT / "mind_raw_frozen.json"


def _gates_e1p() -> dict:
    import yaml
    g = yaml.safe_load((paths.GATES.parent / "gates_e1prime.yaml").read_text())
    if not g.get("frozen"):
        raise RuntimeError("gates_e1prime.yaml is not frozen — refusing to run.")
    return g


def _raw_anchors(bs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RAW-space anchors (no ZCA — the directive's geometry) and the chord."""
    eA = np.stack([endpoint_vecs(b)[0] for b in bs])
    eB = np.stack([endpoint_vecs(b)[1] for b in bs])
    D = np.linalg.norm(eB - eA, axis=1)
    return eA, eB, D


# --- step 1: corpus-only calibration (before any candidate distance) -----------

def calibrate() -> int:
    g = _gates_e1p()
    pct = float(g["min_d"]["percentile"])
    keys = paths.corpus_keys(paths.load_corpus())
    bs = bundles.load_corpus_bundles(keys)
    _, _, D = _raw_anchors(bs)
    floor = float(np.percentile(D, pct))
    low = D < floor

    OUT.mkdir(parents=True, exist_ok=True)
    FLOOR_JSON.write_text(json.dumps({
        "what": "the §1.2 min-D floor, recomputed in RAW embedding space",
        "why": "the persisted floor was fitted in WHITENED space and is meaningless "
               "under the directive's raw geometry. The PERCENTILE is frozen "
               "(gates.yaml min_d.percentile = 5.0) and was NOT re-chosen.",
        "class": "CORPUS-ONLY CALIBRATION — reads the frozen corpus and nothing else. "
                 "No candidate signature, distance, IV number or exam exists at this "
                 "point. Frozen in its own commit before any candidate distance.",
        "percentile": pct,
        "floor_value": floor,
        "n_clips": len(keys),
        "n_low_D": int(low.sum()),
        "low_D_clips": [k for k, f in zip(keys, low) if f],
        "D_distribution": {"min": float(D.min()), "p5": floor, "p50": float(np.median(D)),
                           "p95": float(np.percentile(D, 95)), "max": float(D.max()),
                           "mean": float(D.mean())},
        "scope": "low_D clips are FLAGGED and EXCLUDED from the gamma-signature (every "
                 "channel divides by D) and NEVER zeroed. Coverage reported beside "
                 "accuracy.",
    }, indent=1))
    print(f"[calibrate] raw chord D: min {D.min():.4f}  p5 {floor:.4f}  "
          f"p50 {np.median(D):.4f}  max {D.max():.4f}")
    print(f"[calibrate] min-D floor (5th pct, raw) = {floor:.6f} -> "
          f"{int(low.sum())}/{len(keys)} clips flagged low_D")
    print(f"[calibrate] wrote {FLOOR_JSON}")
    return 0


# --- step 2: the run ----------------------------------------------------------

def _signatures(bs, side, keys, eA, eB, low_D, whiten_fn=None, sigma_source="signature"):
    out = []
    for i, (b, s, k) in enumerate(zip(bs, side, keys)):
        nf = nulls.load_null_feats(paths.clip_path(k), paths.WB_CACHE)
        out.append(e1prime.clip_signature(b, s, nf, eA[i], eB[i], bool(low_D[i]),
                                          whiten_fn=whiten_fn, sigma_source=sigma_source))
    return out


def _arm_matrix(sigs: list[dict], arm: str) -> tuple[np.ndarray, list, list]:
    """z-score on the corpus scaler, then the frozen distance. Returns (D, reasons, z)."""
    raw = [s[arm] if s["defined"] and arm in s else None for s in sigs]
    scaler = e1prime.fit_scaler(raw)
    z = e1prime.zscore_signatures(raw, scaler)
    D = e1prime.distance_matrix(z)
    reasons = [None if s["defined"] else s["reason"] for s in sigs]
    return D, reasons, z, scaler


def run() -> int:
    g_run = paths.load_gates()                 # RUNBOOK gates (hubness, strata) — unchanged
    g = _gates_e1p()                           # E1' gates
    if not FLOOR_JSON.exists():
        raise RuntimeError("run `calibrate` first — the raw min-D floor must be frozen "
                           "in its own commit BEFORE any candidate distance.")
    floor = json.loads(FLOOR_JSON.read_text())["floor_value"]

    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    labels = paths.labels_of(corpus, keys)
    side = paths.sidedness_of(corpus, keys)
    facts = json.loads((paths.WB_OUT / "step0/baselines.json").read_text())["corpus_facts"]
    bs = bundles.load_corpus_bundles(keys)
    eA, eB, Dchord = _raw_anchors(bs)
    low_D = Dchord < floor
    print(f"[E1'] {len(keys)} clips; low_D {int(low_D.sum())} (raw floor {floor:.6f})")

    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    # ---- the four arms (§2.2's CLOSED list) ---------------------------------
    sig_raw = _signatures(bs, side, keys, eA, eB, low_D)                 # arms A, B, D
    lw_art = lw.fit(whitening.core_frames(bs, side))                     # arm C's whitener
    print(f"[E1'] Ledoit-Wolf shrinkage delta = {lw_art['shrinkage']:.6f}  "
          f"(cond raw {lw_art['condition_number_raw']:.3e} -> "
          f"shrunk {lw_art['condition_number_shrunk']:.3e}; NO eigenvalue floor)")
    sig_lw = _signatures(bs, side, keys, eA, eB, low_D,
                         whiten_fn=lambda X: lw.whiten(lw_art, X))       # arm C

    ARMS = [("A_gating", sig_raw, True), ("B_no_null_sub", sig_raw, False),
            ("C_ledoit_wolf", sig_lw, False), ("D_m_tilde_alone", sig_raw, False)]
    mats, scalers = {}, {}
    for arm, sigs, _gating in ARMS:
        D, reasons, z, sc = _arm_matrix(sigs, arm)
        mats[arm], scalers[arm] = (D, reasons, z), sc
        r = exam.evaluate(arm, D, keys, labels, g_run, facts, reasons=reasons)
        results[arm] = r
        print("  " + exam.summary_line(r))

    cand = results["A_gating"]

    # ---- IV1 / IV2 (§2.3) — these decide whether the kill BINDS --------------
    zA = mats["A_gating"][2]
    null_sigs = []
    for i, (b, s, k) in enumerate(zip(bs, side, keys)):
        nf = nulls.load_null_feats(paths.clip_path(k), paths.WB_CACHE)
        ns = e1prime.clip_signature(b, s, nf, eA[i], eB[i], bool(low_D[i]), feats=nf)
        null_sigs.append(ns["A_gating"] if ns["defined"] else None)
    null_z = e1prime.zscore_signatures(null_sigs, scalers["A_gating"])
    iv1 = iv.check("IV1_effect_vs_nothing", list(zA) + list(null_z),
                   ["real"] * len(keys) + ["null"] * len(keys),
                   g["instrument_validity"]["iv1_effect_vs_nothing"]["min_accuracy"])
    print(f"\n[IV1] effect vs nothing: acc {iv1['accuracy']:.4f} "
          f"(min {iv1['min_accuracy']}) -> {'PASS' if iv1['pass'] else 'FAIL'}   "
          f"[a_hat,b_hat only: {iv1['disclosure_a_hat_b_hat_only']['accuracy']:.4f}]")

    iv2 = _iv2(corpus, scalers["A_gating"], floor,
               g["instrument_validity"]["iv2_snap_vs_nothing"]["min_accuracy"])
    print(f"[IV2] snap vs nothing:   acc {iv2['accuracy']:.4f} "
          f"(min {iv2['min_accuracy']}) -> {'PASS' if iv2['pass'] else 'FAIL'}   "
          f"[a_hat,b_hat only: {iv2['disclosure_a_hat_b_hat_only']['accuracy']:.4f}]")
    iv_gate = iv.gate(iv1, iv2)
    print(f"[IV] {iv_gate['verdict']}\n")

    # ---- kill rule (§2.4) ---------------------------------------------------
    mb = g["kill_rule"]["must_beat"]
    d_ok = bool(cand["separation_cohens_d"] > mb["cohens_d"])
    m_ok = bool(cand["misretrieved"] < mb["misretrieved"])
    kill = {
        "rule": g["kill_rule"]["form"],
        "beats_cohens_d": {"candidate": cand["separation_cohens_d"],
                           "incumbent": mb["cohens_d"], "pass": d_ok},
        "beats_misretrieved": {"candidate": cand["misretrieved"],
                               "incumbent": mb["misretrieved"], "pass": m_ok},
        "survives": bool(d_ok and m_ok),
        "binding_on_the_hypothesis": bool(iv_gate["instrument_valid"]),
        "verdict": ("SURVIVES the kill rule" if (d_ok and m_ok) else
                    "KILL — the candidate fails to beat m1a__v3_sided on both"),
    }
    if not iv_gate["instrument_valid"]:
        kill["verdict"] = ("INSTRUMENT-INVALID — " + kill["verdict"] +
                           ", but the IV preconditions failed, so this outcome does NOT "
                           "bind the hypothesis. The program closes UNADJUDICATED "
                           "(§2.6 case 2). No repair attempts.")
    print(f"[KILL RULE] d {cand['separation_cohens_d']:.6f} vs {mb['cohens_d']} "
          f"({'beats' if d_ok else 'FAILS'}); misretrieved {cand['misretrieved']} vs "
          f"{mb['misretrieved']} ({'beats' if m_ok else 'FAILS'})")
    print(f"[KILL RULE] {kill['verdict']}\n")

    # ---- §7 adoption conditions, as computed FACTS --------------------------
    a = g["adoption_m1a"]
    sec7 = {
        "cohens_d_ge_1.772006": {"value": cand["separation_cohens_d"],
                                 "threshold": a["cohens_d_min"],
                                 "pass": bool(cand["separation_cohens_d"] >= a["cohens_d_min"])},
        "misretrieved_lt_73": {"value": cand["misretrieved"],
                               "threshold": a["misretrieved_must_drop_below"],
                               "pass": bool(cand["misretrieved"] < a["misretrieved_must_drop_below"])},
        "hubness_gate": {"value": cand["hubness"]["stats"], "pass": bool(cand["hubness"]["pass"])},
        "coverage_not_materially_narrower": {
            "value": cand["coverage"], "incumbent": 1.0,
            "shortfall_pp": round(100 * (1.0 - cand["coverage"]), 2),
            "pass": None,
            "OWNER_RESERVED": "§7 cond. 4 says coverage must be 'not MATERIALLY narrower' "
                              "than the incumbent's 1.0000. 'Materially' is not a "
                              "threshold, and the executor does not invent one "
                              "(§2.7: ambiguity -> escalate, never choose). The FACT is "
                              "reported: 0.9417 vs 1.0000, i.e. 13/223 clips undefined "
                              "(12 low_D + 1 empty core). The shortfall is entirely a "
                              "consequence of the FROZEN §1.2 min-D guard, which the "
                              "gamma-signature triggers and E1's delta did not (the "
                              "delta contains no D; every gamma channel divides by it). "
                              "MOOT for the §7 call: conditions 1 and 2 fail "
                              "independently and terminally.",
        },
        "probe_battery": {"value": "IV1/IV2 are the only probe-battery elements this "
                                   "cycle registers; the full §7 battery (twins through "
                                   "M2a, sibling-vs-control per class, lerp at the floor) "
                                   "is NOT run — E1' is a kill test, not an adoption run.",
                          "pass": None},
    }
    sec7["all_pass"] = bool(all(v["pass"] for v in sec7.values()
                                if isinstance(v, dict) and v.get("pass") is not None))
    sec7["all_pass_note"] = ("False because conditions 1 (Cohen's d) and 2 (misretrieved) "
                             "FAIL. Two conditions are not computed as pass/fail: the "
                             "coverage condition is owner-reserved (see above) and the "
                             "full probe battery is not part of a kill test. Neither can "
                             "change the §7 outcome.")

    # ---- predictions (descriptive, NEVER gating) ----------------------------
    preds = _predictions(corpus, keys, labels, side, sig_raw, mats["A_gating"][0],
                         zA, null_z, facts)

    # ---- the pre-declared NON-GATING sigma sensitivity column ---------------
    sig_emb = _signatures(bs, side, keys, eA, eB, low_D, sigma_source="embedding")
    D_emb, reasons_emb, _z, _sc = _arm_matrix(sig_emb, "A_gating")
    r_emb = exam.evaluate("A_gating__sigma_emb", D_emb, keys, labels, g_run, facts,
                          reasons=reasons_emb)
    r_emb["gating"] = False
    r_emb["why"] = ("PRE-DECLARED NON-GATING sensitivity column (PREREG §P1). The gating "
                    "sigma is the arc length of the 3-channel signature curve; this "
                    "column re-parameterizes the SAME channels by the arc length of the "
                    "raw 768-d embedding path — the one reading the frozen text does not "
                    "exclude. IT CANNOT CHANGE THE VERDICT. It exists so that a "
                    "sigma-sensitive verdict is adjudicable in owner review WITHOUT a "
                    "re-run.")
    results["A_gating__sigma_emb_NONGATING"] = r_emb
    print("  " + exam.summary_line(r_emb) + "   [NON-GATING sigma column]")

    payload = {
        "authority": "E1PRIME_DIRECTIVE.md §2, frozen at aace78d",
        "instrument_validity": {"iv1": iv1, "iv2": iv2, "gate": iv_gate},
        "kill_rule": kill,
        "section7_conditions": sec7,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in results.items()},
        "predictions": preds,
        "min_d_floor_raw": floor,
        "n_low_D": int(low_D.sum()),
        "ledoit_wolf": {"shrinkage": lw_art["shrinkage"],
                        "condition_number_raw": lw_art["condition_number_raw"],
                        "condition_number_shrunk": lw_art["condition_number_shrunk"],
                        "eig_floor_ratio": None,
                        "note": "no free parameter; no eigenvalue floor (that is the "
                                "point of the arm)"},
    }
    (OUT / "e1prime.json").write_text(json.dumps(payload, indent=1, default=str))
    for arm, (D, _r, _z) in mats.items():
        np.savez_compressed(OUT / f"{arm}_distance_matrix.npz", D=D)
    np.savez_compressed(OUT / "A_gating__sigma_emb_distance_matrix.npz", D=D_emb)
    print(f"\n[E1'] wrote {OUT}/e1prime.json")
    return 0


def _iv2(corpus: dict, scaler: dict, floor: float, min_acc: float) -> dict:
    """IV2 — snap vs nothing, on the deployed Bar-6 hard cuts."""
    from ..pipeline import process_video_file
    man = json.loads((paths.WB_CACHE / "iv2/manifest.json").read_text())
    ro = bundles.ReadOnlyExtractor()
    cut_sigs, lerp_sigs, classes = [], [], []
    for cls, info in sorted(man["classes"].items()):
        b, _ = process_video_file(paths.WB_CACHE / f"iv2/probes/hardcut__{cls}.mp4",
                                  paths.WB_CACHE, ro, tracker=None,
                                  short_side=paths.FEATURE_SHORT_SIDE, need_frames=False)
        lf = np.load(paths.WB_CACHE / f"iv2/lerp__{cls}.npz")["feats"]
        a, bb = endpoint_vecs(b)
        low = bool(float(np.linalg.norm(bb - a)) < floor)
        s = info["sidedness"]
        cs = e1prime.clip_signature(b, s, lf, a, bb, low)
        ls = e1prime.clip_signature(b, s, lf, a, bb, low, feats=lf)
        cut_sigs.append(cs["A_gating"] if cs["defined"] else None)
        lerp_sigs.append(ls["A_gating"] if ls["defined"] else None)
        classes.append(cls)
    z = e1prime.zscore_signatures(cut_sigs + lerp_sigs, scaler)   # the CORPUS scaler
    r = iv.check("IV2_snap_vs_nothing", z,
                 ["cut"] * len(classes) + ["lerp"] * len(classes), min_acc)
    r["n_pairs"] = len(classes)
    r["pair_coverage"] = float(sum(c is not None and l is not None
                                   for c, l in zip(cut_sigs, lerp_sigs)) / len(classes))
    r["classes"] = classes
    return r


def _predictions(corpus, keys, labels, side, sigs, DA, zA, null_z, facts) -> dict:
    """§5, in registered form. DESCRIPTIVE — nothing here gates anything."""
    eligible = set(facts["eligible_n_ge_4"])
    by_class: dict[str, list[int]] = {}
    for i, c in enumerate(labels):
        by_class.setdefault(c, []).append(i)

    # P1: sibling gamma-distance < clip-to-own-null gamma-distance, per eligible class
    pooled = e1prime.distance_matrix(list(zA) + list(null_z))
    n = len(keys)
    p1 = {}
    for c, idx in sorted(by_class.items()):
        if c not in eligible:
            continue
        wins = tot = 0
        for i in idx:
            own_null = pooled[i, n + i]
            sibs = [DA[i, j] for j in idx if j != i and np.isfinite(DA[i, j])]
            if not sibs or not np.isfinite(own_null):
                continue
            tot += 1
            wins += int(min(sibs) < own_null)
        if tot:
            p1[c] = {"clips": tot, "sibling_closer_than_own_null": wins}
    p1_all = sum(1 for v in p1.values() if v["sibling_closer_than_own_null"] == v["clips"])

    # P2: sidedness AUC from s-asymmetry (e0's statistic, on E1's shared-sigma curves)
    def asym(s):
        a, b = s["A_gating"][:, 0], s["A_gating"][:, 1]
        return float(np.mean(a) - np.mean(1.0 - b))
    one = [asym(s) for s, sd in zip(sigs, side) if s["defined"] and sd == "onesided"]
    two = [asym(s) for s, sd in zip(sigs, side) if s["defined"] and sd == "twosided"]
    auc = _auc(one, two)

    # P4: one-sided classes concentrate excursion mass in early sigma
    def centroid(v):
        g = np.linspace(0, 1, len(v))
        return float(np.sum(g * np.abs(v)) / max(float(np.sum(np.abs(v))), 1e-12))
    c_one_mt = [centroid(s["A_gating"][:, 2]) for s, sd in zip(sigs, side)
                if s["defined"] and sd == "onesided"]
    c_two_mt = [centroid(s["A_gating"][:, 2]) for s, sd in zip(sigs, side)
                if s["defined"] and sd == "twosided"]
    c_one_m = [centroid(s["B_no_null_sub"][:, 2]) for s, sd in zip(sigs, side)
               if s["defined"] and sd == "onesided"]
    c_two_m = [centroid(s["B_no_null_sub"][:, 2]) for s, sd in zip(sigs, side)
               if s["defined"] and sd == "twosided"]

    return {
        "status": "DESCRIPTIVE, NON-GATING (§2.5). No verdict depends on any number here.",
        "P1_sibling_closer_than_own_null": {
            "form": "sibling gamma-distance < clip-to-own-rendered-null gamma-distance, "
                    "per n>=4-eligible class — now DIRECTLY checkable (the gamma "
                    "signature exists; e0 could only compute an m-mean analogue)",
            "per_class": p1,
            "classes_where_every_clip_closer_to_sibling": p1_all,
            "n_eligible_classes": len(p1),
        },
        "P2_sidedness_from_s_asymmetry": {
            "statistic": "mean(a_hat) - mean(1 - b_hat) over sigma (e0's, verbatim)",
            "rank_auc_onesided_vs_twosided": auc,
            "onesided": {"n": len(one), "mean": float(np.mean(one)) if one else None},
            "twosided": {"n": len(two), "mean": float(np.mean(two)) if two else None},
            "e0_figure_for_comparison": 0.523,
            "note": "e0's 0.523 was computed on PER-CHANNEL arc-length curves, which "
                    "linearize any monotone channel (E1PRIME_AMENDMENTS.md §A). This "
                    "figure is on E1''s shared-sigma curves. Both are descriptive.",
        },
        "P4_onesided_excursion_mass_early_sigma": {
            "m_tilde_centroid": {"onesided": float(np.mean(c_one_mt)),
                                 "twosided": float(np.mean(c_two_mt)),
                                 "onesided_earlier": bool(np.mean(c_one_mt) < np.mean(c_two_mt))},
            "m_centroid_for_e0_comparability": {
                "onesided": float(np.mean(c_one_m)), "twosided": float(np.mean(c_two_m)),
                "onesided_earlier": bool(np.mean(c_one_m) < np.mean(c_two_m))},
            "note": "centroid < 0.5 = mass early in sigma",
        },
    }


def _auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    v = np.array(pos + neg)
    lab = np.array([1] * len(pos) + [0] * len(neg))
    order = np.argsort(v)
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    n1, n0 = lab.sum(), (1 - lab).sum()
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "run"])
    a = ap.parse_args()
    return calibrate() if a.cmd == "calibrate" else run()


if __name__ == "__main__":
    raise SystemExit(main())
