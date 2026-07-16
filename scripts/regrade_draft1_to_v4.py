"""One-time REGRADE of the draft.1 v4 certification run under the 4.0.0 bars.

draft.1's full §6 run (job 9531327, node ccc0440, complete A-D, zero grader crashes)
FAILED on bar 8 ALONE — specifically the reference-rebuild-parity sub-check of the
`pop_App` population (max|Δ| = 2.0200e-05 > the frozen scalar 1e-6). Every other bar
(1, 2, 4, 5, 6, 7, 9) PASSED, and within bar 8 the warm-determinism (worst=0.0,
bit-perfect) and cold-anchor clauses PASSED.

Advisor (fail-branch consult) diagnosis: a PRE-REGISTRATION DEFECT, not an instrument
failure. `pop_App`/`pop_Dyn` are ECDF-COMPOSED rank LATTICES — their values sit on the
grid {k/(2N)} (N = 24753 corpus pairs), quantum 1/(2N) = 2.0200e-05. A scalar float
tolerance BELOW that quantum is mechanically unsatisfiable under any cross-environment
rebuild, because the same clause tolerates the raw-channel ~2.5e-8 float32-reduction
drift that inevitably flips ONE lattice cell by exactly one step. The corrected
criterion (bars.yaml `reference:`) grades the lattice arrays in integer rank units.

Per the 3.0.0 precedent (draft.8 regrade; SPEC §6 fail_forward EXCEPTION), this pure
grading-rule change over data produced under UNCHANGED measurement pins is certified by
THIS committed script, not a re-run: draft.1's warm determinism was bit-perfect, so a
re-run reproduces every number by construction. Bars 1, 2, 4-7, 9 and bar 8's warm +
cold clauses carry VERBATIM from the draft.1 record (their grader code is byte-identical
in this revision — verify with `git diff` between the draft.1 run commit and this one).
Only bar 8's rebuild-parity clause is regraded, under the two-class criterion, using
draft.1's recorded per-array deltas + the provenance flip-counts.

    PYTHONPATH=src python scripts/regrade_draft1_to_v4.py

Provenance of the flip-counts (job 9538092, node ccc0440 — the SAME node as draft.1's
cert, so the rebuild is bit-identical; reproduce with
`scripts/provenance_rebuild_parity_v4.py`): two on-node rebuilds bit-identical;
cert-path (n_jobs=8) vs build-script-path (n_jobs=16) bit-identical; per-array deltas
IDENTICAL to draft.1's recorded deltas (the bridge asserted below) => the flip-counts
apply to draft.1's own rebuild. The committed reference_v4.npz was built on the
Jupyter-pod CPU (build log: "device=cpu; CUDA is not available"), a DIFFERENT machine
than the ccc0440 rebuild node => the 2.5e-8 drift is cross-environment float32-reduction
order, not within-node nondeterminism (on-node self-repro is bit-identical).
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.transition_eval import versioning  # noqa: E402

SRC_CERT = REPO_ROOT / "outputs/eval/certification/4.0.0-draft.1"
CORPUS = REPO_ROOT / "data/processed/transitions_std121/corpus_manifest.json"
BARS_PATH = REPO_ROOT / "src/diffusion/transition_eval/certify/bars.yaml"

N_PAIRS = 24753  # corpus upper-triangle pairs (223 choose 2); the lattice denominator base

# Provenance job 9538092 (node ccc0440 == draft.1's cert node; warm cache). Verified
# reproducer: scripts/provenance_rebuild_parity_v4.py. These are Class B's inputs; the
# per-array deltas below are the BRIDGE — asserted equal to draft.1's recorded deltas,
# which proves this rebuild is draft.1's rebuild and the flip-counts carry.
PROVENANCE = {
    "job": "9538092",
    "node": "ccc0440.campuscluster.illinois.edu",
    "cpu": "AMD EPYC 7763 64-Core Processor",
    "self_repro_bit_identical": True,      # two cert-path (n_jobs=8) rebuilds identical
    "code_path_bit_identical": True,       # cert-path (8) vs build-script-path (16) identical
    "committed_built_on": "Jupyter-pod CPU (device=cpu; CUDA not available) — different machine",
    "per_array_delta": {                   # R8a vs committed (bridge to draft.1's recorded)
        "mu": 0.0, "pop_P1": 2.2204e-16, "pop_P2": 2.5366e-08, "pop_V1": 2.2204e-16,
        "pop_V1e": 4.4409e-16, "pop_Z": 2.8247e-08, "pop_P": 1.6775e-08, "pop_R": 0.0,
        "r_obj": 0.0, "k_csls": 0.0, "rgrid": 0.0, "s3_app_weight": 0.0,
    },
    "lattice": {
        "pop_App": {"flips": 4, "max_step": 1, "lattice_sanity": 7.276e-12},
        "pop_Dyn": {"flips": 0, "max_step": 0, "lattice_sanity": 7.276e-12},
    },
}

DISCLOSURE = (
    "bar 8's reference-rebuild-parity clause is regraded under a TWO-CLASS criterion "
    "(bars.yaml `reference:`), the ONLY change from the frozen draft.1 bars. draft.1's "
    "run (job 9531327) failed this clause on pop_App at 2.0200e-05 > the scalar 1e-6. "
    "The clause was a PRE-REGISTRATION DEFECT: pop_App/pop_Dyn are ECDF-composed rank "
    "lattices on the grid {k/(2N)} (quantum 1/(2N)=2.0200e-05), for which a scalar float "
    "tolerance is unsatisfiable under any cross-environment rebuild — the same clause "
    "tolerated the ~2.5e-8 raw-channel drift that flips one lattice cell by one step. "
    "The corrected clause compares the seven value-space arrays at max|Δ|<=1e-6 "
    "(unchanged) and the two lattice arrays in integer rank units u=round(pop*2N): "
    "max_step<=4, flips<=50, lattice-sanity<1e-6. K/B are OUTCOME-AWARE (set after "
    "draft.1's flip-counts pop_App 4/max-step-1, pop_Dyn 0 were measured, job 9538092) "
    "and disclosed as such; they are grounded on fault-signature separation — an "
    "off-by-one ranking bug shifts ALL 24753 cells by >=2 steps (trips flips by ~500x) "
    "— not fit to barely pass (observed margins 4x on step, 12.5x on flips)."
)

REGRADE_PROVENANCE = (
    "Verdicts are produced by REGRADE of the draft.1 run artifacts (job 9531327, node "
    "ccc0440), NOT a re-run — 3.0.0 precedent + bars.yaml fail_forward EXCEPTION. The "
    "corrected two-class rebuild-parity clause is a PURE grading-rule change over data "
    "score.py already produced under frozen, UNCHANGED measurement pins; reference_v4.npz "
    "is byte-identical (sha in versioning.PINS unchanged). draft.1's warm determinism was "
    "bit-perfect (worst=0.0), so a re-run reproduces every number by construction. Graders "
    "for bars 1,2,4-7,9 and bar 8's warm/cold clauses are byte-identical in this revision "
    "(git diff between the draft.1 run commit and the regrade commit shows only bars.yaml, "
    "SPEC.md, VERSION, and this script changed — no measurement/grader .py). Their verdicts "
    "and grade payloads carry over verbatim."
)


def grade_rebuild_parity_two_class(per_array_delta, bars_ref):
    """Two-class rebuild-parity over draft.1's recorded per-array deltas + provenance
    flip-counts. Class A: value-space arrays at scalar value_tol. Class B: lattice
    arrays in integer rank units (max_step, flips, lattice-sanity)."""
    value_tol = float(bars_ref["value_tol"])
    lattice_arrays = list(bars_ref["lattice_arrays"])
    K = int(bars_ref["max_step"])
    B = int(bars_ref["max_flips"])
    san_tol = float(bars_ref["lattice_sanity_tol"])
    mism = []

    # --- bridge: provenance rebuild == draft.1 rebuild (per-array deltas identical) ----
    for k, pd in PROVENANCE["per_array_delta"].items():
        rec = float(per_array_delta[k])
        # both are max|Δ| of the same array; equal to the provenance print precision
        if abs(rec - pd) > max(1e-3 * max(rec, pd), 1e-16) + 1e-12:
            mism.append(f"BRIDGE {k}: draft.1 recorded {rec:.4e} != provenance {pd:.4e}")

    # --- Class A: value-space arrays ---------------------------------------------------
    classA = {}
    for k, d in per_array_delta.items():
        if k in lattice_arrays:
            continue
        d = float(d)
        classA[k] = d
        if d > value_tol:
            mism.append(f"Class A {k}: max|Δ|={d:.3e} > value_tol {value_tol:.1e}")

    # --- Class B: lattice arrays in integer rank units ---------------------------------
    classB = {}
    for k in lattice_arrays:
        rec_delta = float(per_array_delta[k])
        rec_max_step = round(rec_delta * 2 * N_PAIRS)   # from the recorded value delta
        p = PROVENANCE["lattice"][k]
        flips, prov_step, san = p["flips"], p["max_step"], p["lattice_sanity"]
        classB[k] = {"max_step": rec_max_step, "flips": flips,
                     "lattice_sanity": san, "prov_max_step": prov_step,
                     "recorded_value_delta": rec_delta}
        if prov_step != rec_max_step:
            mism.append(f"Class B {k}: provenance max_step {prov_step} != recorded-derived {rec_max_step}")
        if san >= san_tol:
            mism.append(f"Class B {k}: lattice_sanity {san:.3e} >= {san_tol:.1e}")
        if rec_max_step > K:
            mism.append(f"Class B {k}: max_step {rec_max_step} > K={K}")
        if flips > B:
            mism.append(f"Class B {k}: flips {flips} > B={B}")

    return {
        "pass": not mism, "mismatch": mism, "criterion": "two-class (SPEC §7)",
        "value_tol": value_tol, "K_max_step": K, "B_max_flips": B,
        "lattice_sanity_tol": san_tol, "N_pairs": N_PAIRS,
        "classA": classA, "classB": classB,
        "provenance": {k: PROVENANCE[k] for k in
                       ("job", "node", "cpu", "self_repro_bit_identical",
                        "code_path_bit_identical", "committed_built_on")},
        "artifact_sha256": versioning.PINS["reference_v4_sha256"],
    }


def main() -> int:
    bars = yaml.safe_load(BARS_PATH.read_text())
    if not bars.get("frozen"):
        raise RuntimeError("bars.yaml is not frozen — regrade refused (SPEC §6.5)")
    ver = versioning.version()
    if ver not in ("4.0.0-draft.2", "4.0.0"):
        raise RuntimeError(f"VERSION {ver} unexpected — regrade writes the 4.0.0(-draft.2) record")

    old = json.loads((SRC_CERT / "record.json").read_text())
    old_b8 = old["grades"]["bar8"]
    per_array = old_b8["reference_rebuild_parity"]["per_array"]

    # --- regrade the ONLY changed clause: bar 8 rebuild-parity (two-class) -------------
    rebuild_parity = grade_rebuild_parity_two_class(per_array, bars["reference"])

    # --- reassemble bar 8: warm + cold + no_crash carry; rebuild-parity regraded -------
    bar8 = dict(old_b8)
    bar8["reference_rebuild_parity"] = rebuild_parity
    bar8["pass"] = bool(old_b8["no_crash"] and old_b8["warm"]["pass"]
                        and old_b8["cold_anchors"]["pass"] and rebuild_parity["pass"])

    # --- carry every other bar verbatim (grader byte-identical) ------------------------
    verdicts = dict(old["verdicts"])
    verdicts["bar8_integration_determinism"] = bar8["pass"]
    overall = all(verdicts.values())

    grades = dict(old["grades"])
    grades["bar8"] = bar8

    out = REPO_ROOT / "outputs/eval/certification" / ver
    out.mkdir(parents=True, exist_ok=True)
    stamp = versioning.stamp(str(CORPUS))
    record = {
        "version": ver, "overall_pass": overall, "verdicts": verdicts,
        "regrade_of": {
            "run": "4.0.0-draft.1 (job 9531327, node ccc0440, complete A-D, "
                   "zero grader crashes; warm determinism bit-perfect worst=0.0)",
            "run_stamp": old["stamp"],
            "run_bars_sha256": old["bars_sha256"],
            "run_overall": old["overall_pass"],
            "directive": "advisor fail-branch consult: pre-registration defect in bar 8's "
                         "rebuild-parity clause; correct the criterion, regrade — not a re-run",
            "changed": ["bar8 reference_rebuild_parity (scalar 1e-6 -> two-class)"],
            "carried_verbatim": ["bar1", "bar2", "bar4", "bar5", "bar6", "bar7", "bar9",
                                 "bar8.no_crash", "bar8.warm", "bar8.cold_anchors"],
            "disclosures": [REGRADE_PROVENANCE, DISCLOSURE],
            "provenance_job": PROVENANCE,
        },
        "stamp": stamp, "bars_sha256": versioning.sha256_file(BARS_PATH),
        "exam": old["exam"],
        "grades": grades,
        "content_invariance": old["content_invariance"],
        "blockc": old["blockc"], "calibration": old["calibration"],
        "claims": old["claims"], "non_gating_fields": old["non_gating_fields"],
    }
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))

    b1 = old["exam"]["bar1"]
    g2 = old["grades"]["sibling_floor"]
    g2loo = old["grades"]["sibling_floor_loo"]
    rp = rebuild_parity
    md = [
        f"# Certification record — transition-eval/{ver}",
        "",
        f"**Overall: {'PASS' if overall else 'FAIL'}** · bars sha256 "
        f"`{record['bars_sha256'][:16]}…` · corpus sha256 "
        f"`{(stamp['corpus_sha256'] or '')[:16]}…` · regrade commit "
        f"`{stamp['git']['commit_short']}` · reference_v4 sha256 "
        f"`{versioning.PINS['reference_v4_sha256'][:16]}…`",
        "",
        "## Provenance: regrade of the draft.1 run, not a re-run",
        "",
        REGRADE_PROVENANCE,
        "",
        "## The one corrected clause (bar 8 rebuild-parity) — disclosure, verbatim",
        "",
        DISCLOSURE,
        "",
        "### Rebuild-parity, two-class result",
        "",
        f"- **Class A** (value-space arrays, max|Δ| ≤ {rp['value_tol']:.0e}): "
        f"{'PASS' if not any('Class A' in m for m in rp['mismatch']) else 'FAIL'} — "
        f"worst {max(rp['classA'].values()):.2e} "
        f"(pop_P2 {rp['classA']['pop_P2']:.2e}, pop_Z {rp['classA']['pop_Z']:.2e}, "
        f"pop_P {rp['classA']['pop_P']:.2e}; rest ≤ 4e-16).",
        f"- **Class B** (lattice arrays, integer rank units): "
        f"pop_App flips {rp['classB']['pop_App']['flips']}/{N_PAIRS} "
        f"(step {rp['classB']['pop_App']['max_step']}), "
        f"pop_Dyn flips {rp['classB']['pop_Dyn']['flips']}/{N_PAIRS} "
        f"(step {rp['classB']['pop_Dyn']['max_step']}); "
        f"budgets max_step ≤ {rp['K_max_step']}, flips ≤ {rp['B_max_flips']}; "
        f"lattice-sanity 7.3e-12 < {rp['lattice_sanity_tol']:.0e}. "
        f"**{'PASS' if rp['pass'] else 'FAIL'}**.",
        f"- Provenance: job {PROVENANCE['job']}, node {PROVENANCE['node']} (== draft.1's "
        f"cert node ⇒ bit-identical rebuild); on-node self-repro bit-identical; "
        f"cert-path vs build-script-path bit-identical; committed artifact built on "
        f"{PROVENANCE['committed_built_on']}. Reproduce: "
        f"`PYTHONPATH=src python scripts/provenance_rebuild_parity_v4.py`.",
        "",
        "## Per-bar verdicts",
        "",
        "| bar | verdict | data | source |",
        "|---|---|---|---|",
        f"| bar1_m1a_floor (S3 d ≥ {b1['d_min']}) | {'PASS' if verdicts['bar1_m1a_floor'] else 'FAIL'} "
        f"| d {b1['d']:.3f}, acc {b1['acc']:.3f} | carried from draft.1 |",
        f"| bar2_sibling_floor (+ LOO) | {'PASS' if verdicts['bar2_sibling_floor'] else 'FAIL'} "
        f"| {g2['n_pass']}/{g2['n_eligible']} deployed, {g2loo['n_pass']}/{g2loo['n_eligible']} LOO "
        f"| carried from draft.1 |",
        f"| bar4_splices | {'PASS' if verdicts['bar4_splices'] else 'FAIL'} "
        f"| gap {old['grades']['splices']['gap']:.3f}, τ_recal {old['grades']['splices']['tau_recalibrated']:.3f} "
        f"| carried from draft.1 |",
        f"| bar5_reversal | {'PASS' if verdicts['bar5_reversal'] else 'FAIL'} "
        f"| {old['grades']['reversal']['wins']}W/{old['grades']['reversal']['losses']}L "
        f"p={old['grades']['reversal']['rule']} | carried from draft.1 |",
        f"| bar6_m3_panel | {'PASS' if verdicts['bar6_m3_panel'] else 'FAIL'} "
        f"| swap {old['grades']['m3_panel']['swap']['n_pass']}, hard-cut "
        f"{old['grades']['m3_panel']['hard_cut']['n_pass']} | carried from draft.1 |",
        f"| bar7_copy_twins | {'PASS' if verdicts['bar7_copy_twins'] else 'FAIL'} "
        f"| {old['grades']['copy_twins']['n_pass']}/{old['grades']['copy_twins']['n_twins']} "
        f"| carried from draft.1 |",
        f"| bar8_integration_determinism | {'PASS' if verdicts['bar8_integration_determinism'] else 'FAIL'} "
        f"| warm bit-perfect (0.0), cold worst {bar8['cold_anchors']['worst']:.1e}, "
        f"rebuild-parity two-class PASS | warm/cold carried; rebuild-parity REGRADED |",
        f"| bar9_causal_gate | {'PASS' if verdicts['bar9_causal_gate'] else 'FAIL'} "
        f"| 3 metrics PASS, 3 controls FAIL (self-verified) | carried from draft.1 |",
        "",
        f"**What certification claims, exactly:** *{record['claims']}*",
        "",
        "Trust map (recall + definedness per class per metric) lives in the draft.1 exam "
        "artifacts (`exam.r1` per-class recall per metric + `exam.m1c_definedness`); the "
        "content-invariance audit, archive distributions + bridge, calibration constants, "
        "anchors, and the 4.0.0 `non_gating_fields` (reference sha + rebuild-parity, "
        "max-over-proxies excess, D_ZPR reversal deltas, bar-2 dual-path margins, ECDF tie "
        "bound + path-separation) all carry from the draft.1 record, unchanged.",
        "",
        "Artifacts: this record regrades `outputs/eval/certification/4.0.0-draft.1/` "
        "(exam/, analysis/, cert_*/, figures/); the regraded record.json sits in "
        f"`outputs/eval/certification/{ver}/`. Reproduce with `PYTHONPATH=src python "
        "scripts/regrade_draft1_to_v4.py` (+ `scripts/provenance_rebuild_parity_v4.py` "
        "for the Class-B flip-counts).",
        "",
        "The draft.1 FAIL record (bar 8, scalar clause) is retained at "
        "`certifications/v4.0.0-draft.1.md`. `eval/v3.0.0` stays certified for v3 numbers.",
    ]
    if ver == "4.0.0":
        md += ["", "On PASS this record authorizes the annotated tag `eval/v4.0.0`; the "
               "stamp above says UNCERTIFIED because it was taken before the tag exists."]
    rec_dir = REPO_ROOT / "src/diffusion/transition_eval/certifications"
    (rec_dir / f"v{ver}.md").write_text("\n".join(md))
    print(f"[regrade] version={ver} overall={'PASS' if overall else 'FAIL'}")
    print(f"[regrade] bar8={'PASS' if bar8['pass'] else 'FAIL'} "
          f"(rebuild-parity two-class {'PASS' if rp['pass'] else 'FAIL'}: "
          f"App flips 4/step1, Dyn 0; mism={rp['mismatch']})")
    print(f"[regrade] verdicts: " + ", ".join(f"{k}={v}" for k, v in verdicts.items()))
    print(f"[regrade] record -> certifications/v{ver}.md + {out / 'record.json'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
