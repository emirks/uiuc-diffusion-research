"""E0 — anatomy of the endpoint-normalized coordinates (RUNBOOK §4.2).

    "E0 — anatomy plots (rides the E1 run, no gating numbers). gamma-curves per
     clip: a_hat(sigma), b_hat(sigma) endpoint-progress coordinates (from S),
     m(sigma) = ||rho(sigma)||/D residual magnitude. Eyeball checks: flame-family
     detaches from the chord, bloom-family hugs it; the 26/13 sidedness split is
     recoverable from s-asymmetry."

E0 RIDES THE E1 RUN AND IS NOT CONDITIONED ON E1 PASSING — it carries no gating
numbers, so the §4.1 kill does not remove it. It is produced here as registered.

Everything below is DESCRIPTIVE. No number here gates anything, and none is
scored against §4.1, §1.4 or §7.

The coordinates are computed in the REGISTERED whitened space (§1.1 mandates
whitening for all inner-product geometry). The instrument state that E1's
diagnostics document therefore also conditions these curves; that is a fact about
the registered instrument, recorded, not corrected.

Also checks the §5 pre-registered predictions that remain computable after the
§4.1 kill (E2/E3 did not run, so the two predictions stated over the gamma-signature
are marked NOT CHECKABLE rather than quietly skipped).
"""

from __future__ import annotations

import json

import numpy as np

from ..s_structure import core_mask_v3
from . import anchors, bundles, curves, nulls, paths, whitening

N_SIGMA = 64


def clip_curves(bundle: dict, sidedness: str, null_feats: np.ndarray,
                zca: dict, eA: np.ndarray, eB: np.ndarray) -> dict | None:
    """a_hat(sigma), b_hat(sigma), m(sigma) for one clip, plus the same for its
    own rendered null (the null is what 'hugging the chord' is measured against)."""
    mask, _ = core_mask_v3(bundle["profile"], sidedness)
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return None
    fw = whitening.whiten(zca, bundle["feats"][idx])
    nw = whitening.whiten(zca, null_feats[idx])
    p = anchors.endpoint_progress(fw, eA, eB)
    pn = anchors.endpoint_progress(nw, eA, eB)
    if not np.isfinite(p["m"]).all():
        return None

    def rs(v):
        return curves.resample(np.asarray(v)[:, None], N_SIGMA)[:, 0]

    return {
        "a_hat": rs(p["a_hat"]), "b_hat": rs(p["b_hat"]), "m": rs(p["m"]),
        "m_null": rs(pn["m"]),
        "m_mean": float(np.mean(p["m"])),
        "m_null_mean": float(np.mean(pn["m"])),
        # excursion mass centroid in sigma: where along the path the effect lives
        "sigma_centroid": float(
            np.sum(np.linspace(0, 1, len(p["m"])) * np.abs(p["m"])) /
            max(float(np.sum(np.abs(p["m"]))), 1e-12)),
        "n_core": int(idx.size),
    }


def main() -> int:
    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    labels = paths.labels_of(corpus, keys)
    side = paths.sidedness_of(corpus, keys)
    zca = whitening.load(paths.WB_CACHE / "zca.npz")
    anc = dict(np.load(paths.WB_CACHE / "anchors.npz"))
    bs = bundles.load_corpus_bundles(keys)

    per_clip, ok_keys = {}, []
    for i, (b, s, k) in enumerate(zip(bs, side, keys)):
        nf = nulls.load_null_feats(paths.clip_path(k), paths.WB_CACHE)
        c = clip_curves(b, s, nf, zca, anc["e_A"][i], anc["e_B"][i])
        if c is not None:
            per_clip[k] = c
            ok_keys.append(k)
    print(f"[E0] curves for {len(ok_keys)}/{len(keys)} clips")

    by_class: dict[str, list[str]] = {}
    for k in ok_keys:
        by_class.setdefault(corpus["clips"][k]["class"], []).append(k)

    # --- chord detachment per class (the flame-vs-bloom eyeball check) ---------
    det = {c: {"m_mean": float(np.mean([per_clip[k]["m_mean"] for k in ks])),
               "m_null_mean": float(np.mean([per_clip[k]["m_null_mean"] for k in ks])),
               "detachment_ratio": float(
                   np.mean([per_clip[k]["m_mean"] for k in ks]) /
                   max(np.mean([per_clip[k]["m_null_mean"] for k in ks]), 1e-12)),
               "n": len(ks),
               "sidedness": corpus["classes"][c]["sidedness"]}
           for c, ks in by_class.items()}
    ranked = sorted(det.items(), key=lambda kv: kv[1]["detachment_ratio"], reverse=True)
    print("\n[E0] chord detachment (clip residual magnitude / its own rendered null's), "
          "descriptive:")
    print("  most detached:")
    for c, v in ranked[:5]:
        print(f"    {c:22s} ratio {v['detachment_ratio']:.3f}  (m {v['m_mean']:.3f} vs "
              f"null {v['m_null_mean']:.3f})  n={v['n']}")
    print("  most chord-hugging:")
    for c, v in ranked[-5:]:
        print(f"    {c:22s} ratio {v['detachment_ratio']:.3f}  (m {v['m_mean']:.3f} vs "
              f"null {v['m_null_mean']:.3f})  n={v['n']}")

    # --- §5 predictions that survive the E1 kill -------------------------------
    preds = {}

    # P1 (adapted): the gamma-signature form is NOT computable (E2 did not run).
    # The computable analogue in the registered E1 space: is a clip closer to a
    # same-class sibling than to its OWN rendered null?
    sib_vs_null = {}
    for c, ks in by_class.items():
        if corpus["classes"][c]["n_clips"] < 4:
            continue
        wins = tot = 0
        for k in ks:
            a = per_clip[k]
            d_null = abs(a["m_mean"] - a["m_null_mean"])
            others = [per_clip[o] for o in ks if o != k]
            if not others:
                continue
            d_sib = min(abs(a["m_mean"] - o["m_mean"]) for o in others)
            tot += 1
            wins += int(d_sib < d_null)
        if tot:
            sib_vs_null[c] = {"clips": tot, "closer_to_sibling": wins}
    preds["P1_sibling_closer_than_own_null"] = {
        "status": "PARTIAL — the gamma-signature form is NOT CHECKABLE (E2 did not "
                  "run, per the §4.1 kill). Computed here on the m(sigma) mean in the "
                  "registered whitened space as the nearest available analogue.",
        "per_class": sib_vs_null,
        "classes_where_all_clips_closer_to_sibling":
            sum(1 for v in sib_vs_null.values() if v["closer_to_sibling"] == v["clips"]),
        "n_eligible_classes": len(sib_vs_null),
    }

    # P2: is the 26/13 sidedness split recoverable from the asymmetry of the
    # endpoint-progress coordinates?
    def asym(k):
        a, b = per_clip[k]["a_hat"], per_clip[k]["b_hat"]
        return float(np.mean(a) - np.mean(1.0 - b))
    one = [asym(k) for k in ok_keys if corpus["classes"][labels[keys.index(k)]]["sidedness"] == "onesided"]
    two = [asym(k) for k in ok_keys if corpus["classes"][labels[keys.index(k)]]["sidedness"] == "twosided"]
    # separability without fitting anything: rank-based AUC
    allv = np.array(one + two)
    lab = np.array([1] * len(one) + [0] * len(two))
    order = np.argsort(allv)
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    n1, n0 = lab.sum(), (1 - lab).sum()
    auc = float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    preds["P2_sidedness_recoverable_from_asymmetry"] = {
        "statistic": "mean(a_hat) - mean(1 - b_hat) over sigma",
        "onesided": {"n": len(one), "mean": float(np.mean(one)), "std": float(np.std(one))},
        "twosided": {"n": len(two), "mean": float(np.mean(two)), "std": float(np.std(two))},
        "rank_auc_onesided_vs_twosided": auc,
        "note": "descriptive, non-gating; AUC 0.5 = no separation, 1.0 = perfect",
    }

    # P3: nature_bloom stays lerp-adjacent (blind-spot theorem, not a failure)
    if "nature_bloom" in det:
        rr = [v["detachment_ratio"] for v in det.values()]
        preds["P3_nature_bloom_lerp_adjacent"] = {
            "nature_bloom_detachment_ratio": det["nature_bloom"]["detachment_ratio"],
            "corpus_median_detachment_ratio": float(np.median(rr)),
            "rank_of_nature_bloom_low_to_high":
                int(np.searchsorted(np.sort(rr), det["nature_bloom"]["detachment_ratio"]) + 1),
            "n_classes": len(rr),
            "note": "descriptive; n=2 class, never gating (RUNBOOK §6)",
        }

    # P4: one-sided classes concentrate excursion mass in early sigma
    cen_one = [per_clip[k]["sigma_centroid"] for k in ok_keys
               if corpus["classes"][labels[keys.index(k)]]["sidedness"] == "onesided"]
    cen_two = [per_clip[k]["sigma_centroid"] for k in ok_keys
               if corpus["classes"][labels[keys.index(k)]]["sidedness"] == "twosided"]
    preds["P4_onesided_excursion_mass_early_sigma"] = {
        "onesided_sigma_centroid_mean": float(np.mean(cen_one)),
        "twosided_sigma_centroid_mean": float(np.mean(cen_two)),
        "onesided_earlier": bool(np.mean(cen_one) < np.mean(cen_two)),
        "note": "descriptive, non-gating; centroid < 0.5 means mass early in sigma",
    }

    print("\n[E0] §5 pre-registered prediction checks (descriptive, non-gating):")
    p2 = preds["P2_sidedness_recoverable_from_asymmetry"]
    print(f"  P2 sidedness from asymmetry: AUC {p2['rank_auc_onesided_vs_twosided']:.3f} "
          f"(onesided mean {p2['onesided']['mean']:+.3f}, twosided {p2['twosided']['mean']:+.3f})")
    if "P3_nature_bloom_lerp_adjacent" in preds:
        p3 = preds["P3_nature_bloom_lerp_adjacent"]
        print(f"  P3 nature_bloom detachment {p3['nature_bloom_detachment_ratio']:.3f} "
              f"(corpus median {p3['corpus_median_detachment_ratio']:.3f}; rank "
              f"{p3['rank_of_nature_bloom_low_to_high']}/{p3['n_classes']} low->high)")
    p4 = preds["P4_onesided_excursion_mass_early_sigma"]
    print(f"  P4 excursion centroid: onesided {p4['onesided_sigma_centroid_mean']:.3f} vs "
          f"twosided {p4['twosided_sigma_centroid_mean']:.3f} "
          f"(onesided earlier: {p4['onesided_earlier']})")
    p1 = preds["P1_sibling_closer_than_own_null"]
    print(f"  P1 (analogue): {p1['classes_where_all_clips_closer_to_sibling']}/"
          f"{p1['n_eligible_classes']} eligible classes have EVERY clip closer to a "
          f"sibling than to its own null")

    out = paths.WB_OUT / "e0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e0_anatomy.json").write_text(json.dumps({
        "status": "E0 rides the E1 run and carries NO gating numbers (§4.2). It is "
                  "produced as registered despite the §4.1 kill, which does not "
                  "remove it. Everything here is DESCRIPTIVE.",
        "space": "registered whitened space (§1.1); the instrument state documented "
                 "in e1_floor_sensitivity.json also conditions these curves",
        "n_clips_with_curves": len(ok_keys),
        "per_class_chord_detachment": det,
        "prediction_checks": preds,
    }, indent=1, default=str))

    np.savez_compressed(out / "e0_curves.npz",
                        keys=np.array(ok_keys),
                        a_hat=np.stack([per_clip[k]["a_hat"] for k in ok_keys]),
                        b_hat=np.stack([per_clip[k]["b_hat"] for k in ok_keys]),
                        m=np.stack([per_clip[k]["m"] for k in ok_keys]),
                        m_null=np.stack([per_clip[k]["m_null"] for k in ok_keys]))
    print(f"\n[E0] wrote {out}/e0_anatomy.json + e0_curves.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
