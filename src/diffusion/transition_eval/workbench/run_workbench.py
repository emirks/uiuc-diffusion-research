"""Workbench driver (OPERATIONS §5/§6).

Enforces the freeze: paths.load_gates() REFUSES to return unfrozen gates, so no
candidate can be scored against numbers that could still move. E2/E3 additionally
refuse to run unless E1's recorded verdict is a pass — the §4.1 kill rule is
terminal, and the driver, not the operator's memory, is what makes it terminal.

Subcommands:
  e1          §4.1 kill test — the effect-delta vector, head-to-head vs the pinned
              incumbent. Runs first; gates all of E2/E3.
  fit-motion  Fit camera trajectories for all 223 clips from the cached flow, and
              emit the corpus residual-energy and texture distributions that
              §3.3's epsilon and §3.2's low-texture threshold are calibrated from.
              Produces NO candidate score — calibration inputs only.
  phase1      §3.2/§3.3 descriptors -> exam -> §1.4 hubness -> §3.6 verdict.

Usage:
    PYTHONPATH=$WB/src python -m diffusion.transition_eval.workbench.run_workbench <cmd>
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import (bundles, e1_delta, exam, m1b_flow, m1c_flow, paths, whitening)


def log(msg: str) -> None:
    print(f"[wb {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _context():
    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    labels = paths.labels_of(corpus, keys)
    sidedness = paths.sidedness_of(corpus, keys)
    gates = paths.load_gates()                      # refuses if not frozen
    facts = json.loads((paths.WB_OUT / "step0/baselines.json").read_text())["corpus_facts"]
    return corpus, keys, labels, sidedness, gates, facts


# --- E1: the kill test --------------------------------------------------------

def cmd_e1(args) -> int:
    corpus, keys, labels, sidedness, gates, facts = _context()
    zca = whitening.load(paths.WB_CACHE / "zca.npz")
    log(f"ZCA warm: {int(zca['n_frames'])} core frames, {int(zca['dim'])} dims")

    bs = bundles.load_corpus_bundles(keys)
    log(f"{len(bs)} warm bundles (certified cache untouched)")

    deltas = e1_delta.corpus_deltas(bs, sidedness, keys, zca)
    n_def = sum(d["defined"] for d in deltas)
    log(f"E1 deltas: {n_def}/{len(deltas)} defined "
        f"({sum(d.get('core_degenerate', False) for d in deltas)} clips have a "
        f"degenerate core)")

    D = e1_delta.distance_matrix(deltas)
    reasons = [d["reason"] for d in deltas]
    r = exam.evaluate("e1_delta", D, keys, labels, gates, facts, reasons)
    log(exam.summary_line(r))

    kill = e1_delta.kill_rule(r, gates)
    inc = gates["baselines"]["m1a__v3_sided"]
    log(f"incumbent m1a__v3_sided: d {inc['cohens_d']:.6f}, mis {inc['misretrieved']}/223, "
        f"cov {inc['coverage']}")
    log(f"§4.1 beats Cohen's d?     {kill['beats_cohens_d']['pass']} "
        f"({r['separation_cohens_d']:.6f} vs {inc['cohens_d']})")
    log(f"§4.1 beats misretrieved?  {kill['beats_misretrieved']['pass']} "
        f"({r['misretrieved']} vs {inc['misretrieved']})")
    log(f"§4.1 VERDICT: {kill['verdict']}")

    extra = {
        "kill_rule": kill,
        "incumbent": inc,
        "pinned_choices": {
            "v_null_pooling": gates["phase2"]["e1"]["v_null_pooling"],
            "low_D_excluded": False,
            "note": "both pinned in gates.yaml before any E1 number existed",
        },
        "delta_norms": {
            "mean": float(np.mean([d["delta_norm"] for d in deltas if d["defined"]])),
            "min": float(np.min([d["delta_norm"] for d in deltas if d["defined"]])),
            "max": float(np.max([d["delta_norm"] for d in deltas if d["defined"]])),
        },
    }
    out = exam.save(paths.WB_OUT / "e1", "e1_delta", D, r, extra)
    log(f"wrote {out}")
    return 0


# --- Phase 1 calibration inputs (no candidate score) --------------------------

def cmd_fit_motion(args) -> int:
    from . import flowcache

    corpus, keys, labels, sidedness, gates, facts = _context()
    bs = bundles.load_corpus_bundles(keys)

    tex_path = paths.WB_CACHE / "texture.npz"
    tex = np.load(tex_path)["pair_texture"] if tex_path.exists() else None
    if tex is None:
        log("WARNING: texture.npz missing — the §3.2 low-texture gate cannot be "
            "applied. Run the texture job first.")

    out_dir = paths.WB_OUT / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    params, defined, inliers, energies, core_masks = [], [], [], [], []

    t0 = time.time()
    for i, k in enumerate(keys):
        flow = flowcache.load_clip_flow(paths.clip_path(k), paths.WB_CACHE)
        traj = m1b_flow.clip_camera_trajectory(flow.astype(np.float32))
        st = m1c_flow.clip_residual_stats(flow.astype(np.float32), traj)
        params.append(traj["params"])
        defined.append(traj["defined"])
        inliers.append(traj["inlier_frac"])
        energies.append(st["energy"])
        core_masks.append(exam.core_pair_mask(bs[i], sidedness[i]))
        if (i + 1) % 25 == 0:
            log(f"  fit {i + 1}/{len(keys)} ({time.time() - t0:.0f}s)")

    P = np.stack(params); Dfn = np.stack(defined); I = np.stack(inliers)
    E = np.stack(energies); C = np.stack(core_masks)
    np.savez_compressed(out_dir / "camera_fits.npz", keys=np.array(keys), params=P,
                        defined=Dfn, inlier_frac=I, energy=E, core_pairs=C)

    # --- calibration inputs (corpus-only; no candidate has been scored) --------
    core_E = E[C]
    core_E = core_E[np.isfinite(core_E)]
    e_pcts = {f"p{p}": float(np.percentile(core_E, p))
              for p in (1, 2, 5, 10, 20, 25, 50, 75, 95)}
    inl = I[C]
    report = {
        "n_clips": len(keys),
        "camera_fit": {
            "defined_pair_frac_overall": float(Dfn[C].mean()),
            "inlier_frac_percentiles": {f"p{p}": float(np.percentile(inl, p))
                                        for p in (5, 25, 50, 75, 95)},
            "clips_with_no_defined_core_pair": int((~(Dfn & C)).all(axis=1).sum()),
        },
        "residual_energy_core_frames": {
            "n": int(core_E.size), "percentiles": e_pcts,
            "note": "§3.3's epsilon is chosen from THIS distribution (corpus-only, "
                    "outcome-independent) and frozen in gates.yaml BEFORE the "
                    "Phase 1 exam runs.",
        },
    }
    if tex is not None:
        tx = tex[C]
        report["pair_texture_core_frames"] = {
            "percentiles": {f"p{p}": float(np.percentile(tx, p))
                            for p in (1, 2, 5, 10, 25, 50)},
            "note": "§3.2's low-texture threshold is chosen from THIS distribution.",
        }
    (out_dir / "calibration_inputs.json").write_text(json.dumps(report, indent=1))

    log(f"camera fits: {Dfn[C].mean():.4f} of core pairs defined; "
        f"{report['camera_fit']['clips_with_no_defined_core_pair']} clips have no "
        f"defined core pair")
    log("residual energy (core frames) percentiles: " +
        "  ".join(f"{k2}={v:.6f}" for k2, v in e_pcts.items()))
    log(f"wrote {out_dir / 'calibration_inputs.json'}")
    return 0


# --- Phase 1: acceptance (§3.4) then the exam (§3.5/§3.6) ---------------------

def _traj_of(i, gates, tex, fits):
    """One clip's camera trajectory with the FROZEN §3.2 gates applied."""
    tex_min = gates["phase1"]["m1b_flow"]["min_pair_texture"]
    traj = {"params": fits["params"][i], "defined": fits["defined"][i].copy(),
            "inlier_frac": fits["inlier_frac"][i]}
    if tex is not None:
        traj["defined"] = traj["defined"] & (tex[i] >= tex_min)   # §3.2 low-texture gate
    return traj


def _m1b_descriptors(keys, gates, tex, fits):
    """m1b needs only the cached camera fits — no flow, so no 15 GB re-read."""
    return [m1b_flow.clip_descriptor(_traj_of(i, gates, tex, fits), fits["core_pairs"][i])
            for i in range(len(keys))]


def _m1c_descriptors(keys, gates, tex, fits):
    """m1c needs the dense residual, so it does read the flow cache."""
    from . import flowcache

    eps = gates["phase1"]["m1c_flow"]["energy_gate_epsilon"]
    out = []
    for i, k in enumerate(keys):
        flow = flowcache.load_clip_flow(paths.clip_path(k), paths.WB_CACHE).astype(np.float32)
        out.append(m1c_flow.clip_curve(flow, _traj_of(i, gates, tex, fits),
                                       fits["core_pairs"][i], eps))
    return out


def cmd_phase1(args) -> int:
    from . import acceptance, curves, flowcache

    corpus, keys, labels, sidedness, gates, facts = _context()
    if gates["phase1"]["m1c_flow"]["energy_gate_epsilon"] is None:
        log("REFUSING: gates.yaml m1c_flow.energy_gate_epsilon is null. The §3.3 "
            "calibration must be chosen from the corpus residual distribution and "
            "FROZEN in its own commit before the exam runs.")
        return 1
    bs = bundles.load_corpus_bundles(keys)
    fits = np.load(paths.WB_OUT / "phase1/camera_fits.npz")
    tex_p = paths.WB_CACHE / "texture.npz"
    tex = np.load(tex_p)["pair_texture"] if tex_p.exists() else None

    log("building descriptors under the frozen gates")
    m1b_raw = _m1b_descriptors(keys, gates, tex, fits)
    m1c_raw = _m1c_descriptors(keys, gates, tex, fits)
    m1b_z = m1b_flow.corpus_descriptors(m1b_raw)
    m1c_z = m1c_flow.corpus_descriptors(m1c_raw)
    log(f"m1b_flow defined {sum(d['defined'] for d in m1b_raw)}/{len(keys)}; "
        f"m1c_flow defined {sum(d['defined'] for d in m1c_raw)}/{len(keys)} "
        f"({sum(d['n_gated'] for d in m1c_raw)} frames energy-gated)")

    D_b = curves.distance_matrix(m1b_z)
    D_c = curves.distance_matrix(m1c_z)

    # ---- §3.4 ACCEPTANCE: both must pass BEFORE the exam runs ----------------
    acc_path = paths.WB_OUT / "phase1/acceptance.json"
    if not acc_path.exists():
        log("REFUSING: §3.4 acceptance has not been graded. The exam is not run on "
            "a metric that has not faced constructed truth. Run `acceptance` first.")
        return 1
    acc = json.loads(acc_path.read_text())
    if not acc["pass"]:
        log("§3.4 ACCEPTANCE FAILED — the exam is NOT run on a metric that fails "
            "constructed truth (RUNBOOK §3.4). Recording and stopping.")
        (paths.WB_OUT / "phase1" / "VERDICT.json").write_text(json.dumps({
            "phase": 1, "status": "STOPPED at §3.4 acceptance",
            "exam_run": False, "acceptance": acc}, indent=1))
        return 0
    log("§3.4 acceptance PASSED — the exam may run")

    # ---- exam (frozen kernel) ------------------------------------------------
    out = paths.WB_OUT / "phase1"
    results = {}
    for name, D, raws, stratum in (("m1b_flow", D_b, m1b_raw, "camera"),
                                   ("m1c_flow", D_c, m1c_raw, "object")):
        reasons = [d["reason"] for d in raws]
        r = exam.evaluate(name, D, keys, labels, gates, facts, reasons, stratum)
        inc = {"cohens_d": gates["phase1"]["adoption"][name]["beat_cohens_d"]}
        v = exam.verdict_vs_incumbent(r, inc, gates, stratum)
        r["verdict"] = v
        results[name] = r
        log(exam.summary_line(r))
        sr = r["stratum_recalls"][stratum]
        log(f"    {stratum}-stratum recall {sr['value']:.5f} vs incumbent "
            f"{gates['stratum_targets'][stratum]['value']:.5f} "
            f"({sr['n_classes']} eligible classes, {sr['n_classes_nan']} NaN)")
        log(f"    §3.6 all conditions pass: {v['all_pass']}")
        exam.save(out, name, D, r, {"verdict": v, "acceptance": acc})

    (out / "VERDICT.json").write_text(json.dumps({
        "phase": 1, "status": "exam run", "exam_run": True,
        "acceptance": acc,
        "verdicts": {n: r["verdict"] for n, r in results.items()},
        "note": "Each §3.6 condition is a computed FACT. The adoption call is "
                "owner-side (OPERATIONS §8).",
    }, indent=1, default=str))
    log(f"wrote {out / 'VERDICT.json'}")
    return 0


def cmd_acceptance(args) -> int:
    """§3.4 — reversal + injected-trajectory, on constructed truth."""
    from . import acceptance, build_probes, curves, flowcache

    corpus, keys, labels, sidedness, gates, facts = _context()
    bs = bundles.load_corpus_bundles(keys)
    fits = np.load(paths.WB_OUT / "phase1/camera_fits.npz")
    fkeys = [str(k) for k in fits["keys"]]
    man = json.loads((paths.WB_CACHE / "probes/manifest.json").read_text())
    tex_p = paths.WB_CACHE / "texture.npz"
    tex = np.load(tex_p)["pair_texture"] if tex_p.exists() else None

    # ---- test 2: injected-trajectory recovery -------------------------------
    # The probe base clips must clear the SAME frozen gates the exam uses. A base
    # clip the metric itself declares undefined (here: textureless) is not a valid
    # substrate for constructed truth — grading on it would fail the acceptance test
    # for a reason that has nothing to do with the metric's ability to recover a
    # camera. Excluded EXPLICITLY and recorded, never silently dropped (§1.5).
    tex_min = gates["phase1"]["m1b_flow"]["min_pair_texture"]
    excluded = []
    base_clips = []
    for k in man["static_clips"]:
        i = fkeys.index(k)
        core = fits["core_pairs"][i]
        below = float((tex[i][core] < tex_min).mean()) if tex is not None else 0.0
        if below > m1b_flow.MAX_UNDEFINED_CORE:
            excluded.append({"clip": k, "frac_core_pairs_below_texture_gate": below,
                             "reason": "base clip is texture-gated by the frozen §3.2 "
                                       "rule; not a valid constructed-truth substrate"})
            log(f"  EXCLUDED base clip {k}: {below:.0%} of core pairs below the frozen "
                f"texture gate")
            continue
        base_clips.append(k)

    inj_rows = []
    for k in base_clips:
        for kind in man["inject_kinds"]:
            p = build_probes.probe_path("inj", k, kind)
            z = np.load(p)
            flow = z["flow"].astype(np.float32)
            truth = z["truth"]
            traj = m1b_flow.clip_camera_trajectory(flow)
            g = acceptance.grade_injection(traj["params"], truth, traj["defined"])
            inj_rows.append({"clip": k, "kind": kind, "pass": g["pass"],
                             "n_graded": g["n_graded"], "params": g["params"]})
            log(f"  inject {kind:9s} {k[:28]:28s} pass={g['pass']} "
                + "  ".join(f"{n}: r={v['corr']:.3f} amp={v['amp_err']:.3f}"
                            for n, v in g["params"].items() if v["graded"]))
    inj_pass = all(r["pass"] for r in inj_rows)

    # ---- test 1: reversal ----------------------------------------------------
    m1b_raw = _m1b_descriptors(keys, gates, tex, fits)   # no flow needed for m1b
    m1b_z = m1b_flow.corpus_descriptors(m1b_raw)
    D_b = curves.distance_matrix(m1b_z)
    lab = np.array(labels)

    rev_rows = []
    for k in man["reversed_clips"]:
        i = fkeys.index(k)
        fwd = {"params": fits["params"][i], "defined": fits["defined"][i]}
        if not acceptance.reversal_sensitive(fwd["params"], fwd["defined"]):
            rev_rows.append({"clip": k, "graded": False,
                             "reason": "palindromic / not reversal-sensitive"})
            continue
        z = np.load(build_probes.probe_path("rev", k))
        rtraj = m1b_flow.clip_camera_trajectory(z["flow"].astype(np.float32))
        core = fits["core_pairs"][i]
        rdesc = m1b_flow.clip_descriptor(rtraj, core[::-1])
        same = np.flatnonzero((lab == labels[i]))
        same = same[same != i]
        within = [D_b[i, j] for j in same if np.isfinite(D_b[i, j])]
        med = float(np.median(within)) if within else float("nan")
        g = acceptance.grade_reversal(fwd, rtraj, m1b_z[i],
                                      rdesc["curve"] if rdesc["defined"] else None, med)
        rev_rows.append({"clip": k, "graded": True, "pass": g["pass"],
                         "parameter_negation": g["parameter_negation"],
                         "descriptor_distance": g["descriptor_distance"]})
    graded_rev = [r for r in rev_rows if r.get("graded")]
    rev_pass = bool(graded_rev and all(r["pass"] for r in graded_rev))
    log(f"  reversal: {sum(r['pass'] for r in graded_rev)}/{len(graded_rev)} graded "
        f"clips pass (of {len(rev_rows)} camera-tagged; "
        f"{len(rev_rows) - len(graded_rev)} not reversal-sensitive)")

    out = {
        "rule": "RUNBOOK §3.4 — both tests must pass before the exam runs. Failure "
                "of either = fix or stop; the exam is not run on a metric that fails "
                "constructed truth.",
        "injected_trajectory": {"pass": inj_pass, "n": len(inj_rows), "rows": inj_rows,
                                "base_clips_used": base_clips,
                                "base_clips_excluded": excluded,
                                "thresholds": {"corr_min": acceptance.CORR_MIN,
                                               "amp_err_max": acceptance.AMP_ERR_MAX}},
        "reversal": {"pass": rev_pass, "n_graded": len(graded_rev),
                     "n_total": len(rev_rows), "rows": rev_rows},
        "pass": bool(inj_pass and rev_pass),
    }
    p = paths.WB_OUT / "phase1/acceptance.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str))
    log(f"§3.4 ACCEPTANCE: injected={inj_pass} reversal={rev_pass} -> "
        f"{'PASS — exam may run' if out['pass'] else 'FAIL — exam is NOT run'}")
    log(f"wrote {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("e1")
    sub.add_parser("fit-motion")
    sub.add_parser("acceptance")
    sub.add_parser("phase1")
    args = ap.parse_args()
    return {"e1": cmd_e1, "fit-motion": cmd_fit_motion,
            "acceptance": cmd_acceptance, "phase1": cmd_phase1}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
