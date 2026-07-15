"""Pre-exam definedness report (RUNBOOK §1.5; OPERATIONS §7/§8).

    "Undefined != zero, everywhere. Frames or clips failing gates are excluded from
     fits and averages and COUNTED in a definedness report per metric. A candidate's
     definedness coverage is reported next to its accuracy (an accuracy win on a
     shrunken support is not a win)."

Written BEFORE the exam, from the frozen gates alone. Nothing here is a retrieval
score; nothing here gates. Its job is to make the §3.6 numbers' floor a
pre-registered corpus fact rather than a post-hoc discovery: under the frozen
convention, an eligible class with zero covered clips contributes 0.0 to the macro
stratum recall, so enumerating those classes NOW — before any distance is computed
— tells the reviewer exactly what the candidate is starting from.

Also records the mechanism as a CORPUS fact rather than an inference from
constructed truth: the effect-area fraction on defined vs undefined core frames.
"""

from __future__ import annotations

import json

import numpy as np

from . import m1b_flow, m1c_flow, paths

REASONS = ("no_core_pairs", "inlier_collapse", "texture_gated", "energy_gated", "defined")


def main() -> int:
    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    labels = paths.labels_of(corpus, keys)
    gates = paths.load_gates()
    tex_min = gates["phase1"]["m1b_flow"]["min_pair_texture"]
    eps = gates["phase1"]["m1c_flow"]["energy_gate_epsilon"]
    cap = m1b_flow.MAX_UNDEFINED_CORE

    f = np.load(paths.WB_OUT / "phase1/camera_fits.npz")
    tex = np.load(paths.WB_CACHE / "texture.npz")["pair_texture"]
    Dfn, E, C, I = f["defined"], f["energy"], f["core_pairs"], f["inlier_frac"]

    rows = []
    for i, k in enumerate(keys):
        core = C[i]
        n_core = int(core.sum())
        if n_core < 2:
            rows.append({"key": k, "class": labels[i], "n_core_pairs": n_core,
                         "m1b_defined": False, "m1c_defined": False,
                         "reason": "no_core_pairs"})
            continue
        cam_ok = Dfn[i] & core
        tex_ok = cam_ok & (tex[i] >= tex_min)
        eng_ok = tex_ok & (np.where(Dfn[i], E[i], np.nan) >= eps)

        u_cam = 1 - cam_ok.sum() / n_core
        u_tex = 1 - tex_ok.sum() / n_core
        u_eng = 1 - eng_ok.sum() / n_core
        m1b_def = bool(u_tex <= cap)
        m1c_def = bool(u_eng <= cap)
        reason = "defined"
        if not m1b_def:
            reason = "inlier_collapse" if u_cam > cap else "texture_gated"
        elif not m1c_def:
            reason = "energy_gated"
        rows.append({
            "key": k, "class": labels[i], "n_core_pairs": n_core,
            "undefined_frac_after_camera_fit": float(u_cam),
            "undefined_frac_after_texture": float(u_tex),
            "undefined_frac_after_energy": float(u_eng),
            "median_inlier_frac": float(np.median(I[i][core])),
            "m1b_defined": m1b_def, "m1c_defined": m1c_def, "reason": reason,
        })

    by_reason: dict[str, int] = {}
    for r in rows:
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
    print("[definedness] clips by reason:", by_reason)

    zero_cov = [r for r in rows if not r["m1b_defined"]]
    print(f"[definedness] m1b defined {sum(r['m1b_defined'] for r in rows)}/223; "
          f"m1c defined {sum(r['m1c_defined'] for r in rows)}/223")

    # --- the §3.6 floor, as a pre-registered corpus fact ----------------------
    facts = json.loads((paths.WB_OUT / "step0/baselines.json").read_text())["corpus_facts"]
    eligible = set(facts["eligible_n_ge_4"])
    strata = {"camera": set(facts["camera_classes"]), "object": set(facts["object_classes"])}
    floor = {}
    for sname, scls in strata.items():
        cells = sorted(scls & eligible)
        key_metric = "m1b_defined" if sname == "camera" else "m1c_defined"
        zero = []
        for c in cells:
            covered = [r for r in rows if r["class"] == c and r[key_metric]]
            if not covered:
                zero.append(c)
        floor[sname] = {
            "n_eligible_classes": len(cells),
            "classes_with_zero_covered_clips": zero,
            "n_zero": len(zero),
            "mechanical_recall_ceiling": float(1 - len(zero) / len(cells)) if cells else 0.0,
            "note": ("Under the frozen convention a zero-covered eligible class "
                     "contributes 0.0 to the macro stratum recall, so the candidate's "
                     "§3.6 recall cannot exceed this ceiling however well it retrieves "
                     "the classes it does cover."),
        }
        print(f"[definedness] {sname} stratum: {len(zero)}/{len(cells)} eligible classes "
              f"have ZERO covered clips -> mechanical recall ceiling "
              f"{floor[sname]['mechanical_recall_ceiling']:.4f} "
              f"(incumbent {gates['stratum_targets'][sname]['value']:.4f})")
        if zero:
            print(f"    zero-covered: {zero}")

    # --- mechanism as a CORPUS fact (not an inference from constructed truth) --
    from . import flowcache
    areas_def, areas_undef = [], []
    for i, k in enumerate(keys[:60]):                 # a 60-clip sample is plenty
        core = np.flatnonzero(C[i])
        if core.size == 0:
            continue
        flow = flowcache.load_clip_flow(paths.clip_path(k), paths.WB_CACHE).astype(np.float32)
        for j in core[::7]:
            if not np.isfinite(f["params"][i][j]).all():
                continue
            res = m1c_flow.residual_field(flow[j], f["params"][i][j])
            a = m1c_flow.effect_area_fraction(res)
            (areas_def if Dfn[i][j] else areas_undef).append(a)
    mech = {
        "effect_area_fraction_defined_frames": {
            "n": len(areas_def), "mean": float(np.mean(areas_def)) if areas_def else None,
            "median": float(np.median(areas_def)) if areas_def else None},
        "effect_area_fraction_undefined_frames": {
            "n": len(areas_undef), "mean": float(np.mean(areas_undef)) if areas_undef else None,
            "median": float(np.median(areas_undef)) if areas_undef else None},
        "note": ("Effect area = fraction of pixels the camera fit calls an outlier. "
                 "If undefined frames carry a much larger effect area, the coverage "
                 "loss is the large-effect-region mechanism, measured on the corpus "
                 "rather than inferred from constructed truth."),
    }
    print(f"[definedness] effect area: defined frames median "
          f"{mech['effect_area_fraction_defined_frames']['median']}, "
          f"undefined frames median {mech['effect_area_fraction_undefined_frames']['median']}")

    # --- Huber breakdown, as a committed artifact -----------------------------
    H, W = 64, 48
    A, grid = m1b_flow.design_matrix(H, W, stride=1)
    truth = np.array([2.0, -1.0, 0.01, 0.02])
    hub = []
    for frac in (0.05, 0.15, 0.25, 0.33, 0.40, 0.45, 0.55):
        fl = m1c_flow.camera_field(truth, H, W)
        fl[:, : int(W * frac)] += np.array([9.0, -7.0])
        r = m1b_flow.fit_similarity(fl, A, grid, stride=1)
        hub.append({"contaminated_area": frac,
                    "max_param_error": float(np.abs(r["params"] - truth).max()),
                    "inlier_frac": r["inlier_frac"], "defined": r["defined"]})

    out = paths.WB_OUT / "phase1"
    (out / "definedness_report.json").write_text(json.dumps({
        "purpose": "Pre-exam definedness (§1.5). No retrieval score here; nothing gates.",
        "frozen_gates": {"min_pair_texture": tex_min, "energy_gate_epsilon": eps,
                         "min_inlier_fraction": m1b_flow.MIN_INLIER_FRAC,
                         "max_undefined_core_frac": cap},
        "clips_by_reason": by_reason,
        "m1b_defined": sum(r["m1b_defined"] for r in rows),
        "m1c_defined": sum(r["m1c_defined"] for r in rows),
        "n_clips": len(rows),
        "stratum_recall_floor_pre_exam": floor,
        "mechanism_corpus": mech,
        "huber_breakdown_constructed_truth": hub,
        "per_clip": rows,
    }, indent=1, default=str))
    print(f"[definedness] wrote {out / 'definedness_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
