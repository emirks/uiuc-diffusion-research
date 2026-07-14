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


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("e1")
    sub.add_parser("fit-motion")
    args = ap.parse_args()
    return {"e1": cmd_e1, "fit-motion": cmd_fit_motion}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
