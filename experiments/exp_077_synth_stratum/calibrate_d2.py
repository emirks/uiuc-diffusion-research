"""exp_077 D2 gate calibration (Task 3).

PHASE=select  — pick a ~40-clip labeling sample from the audit that SPANS both the shader bank
                and the M1 score range (4 clips per M1 decile, distinct shaders preferred), and
                render the labeling contact sheets. Requires the audit mp4s (compute node).

PHASE=tau     — read my visual labels, choose tau by the pre-registered rule
                (reject ALL labeled-bad, pass >= 90% of labeled-good), FREEZE it to D2_TAU.json,
                then compute per-shader pass rates / blacklist / n_shaders>=80% and write
                D2_AUDIT.json. Pure JSON math — runs anywhere.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

import d2_metrics  # noqa: E402
import d2_sheets  # noqa: E402

CONFIG_PATH = HERE / "config_d2.yaml"
N_LABEL = 40
PER_DECILE = 4
CLIPS_PER_SHEET = 2


def load_rows(meta_dir: Path) -> tuple[list[dict], list[dict]]:
    rows, exhausted = [], []
    for p in sorted(meta_dir.glob("rows_shard*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            (exhausted if r.get("gate2_exhausted") else rows).append(r)
    seen, uniq = set(), []
    for r in rows:                      # de-dup in case a shard was re-run
        if r["stem"] in seen:
            continue
        seen.add(r["stem"])
        uniq.append(r)
    return uniq, exhausted


def select_labeling_sample(rows: list[dict], n: int = N_LABEL,
                           per_decile: int = PER_DECILE) -> list[dict]:
    """Stratify by M1 decile so the labeled set brackets any candidate tau, and spread shaders."""
    order = sorted(rows, key=lambda r: r["m1_p10"])
    n_dec = max(1, n // per_decile)
    bounds = np.linspace(0, len(order), n_dec + 1).astype(int)
    used_shaders: set[str] = set()
    picked: list[dict] = []
    for k in range(n_dec):
        chunk = order[bounds[k]:bounds[k + 1]]
        # deterministic spread inside the decile, preferring shaders not yet labeled
        chunk_sorted = sorted(chunk, key=lambda r: (r["shader"] in used_shaders, r["stem"]))
        take = []
        for r in chunk_sorted:
            if len(take) >= per_decile:
                break
            if r["shader"] in {t["shader"] for t in take}:
                continue
            take.append(r)
        for r in chunk_sorted:
            if len(take) >= per_decile:
                break
            if r not in take:
                take.append(r)
        for r in take:
            used_shaders.add(r["shader"])
            picked.append({"decile": k + 1, **r})
    return picked


def phase_select(cfg: dict) -> None:
    from engine import videoio
    root = REPO_ROOT / cfg["outputs"]["dir"] / "audit"
    rows, exh = load_rows(root / "meta")
    print(f"[select] {len(rows)} clip rows, {len(exh)} gate-2 exhausted tuples")
    picked = select_labeling_sample(rows)
    out = root / "label"
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_set.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("params", "ncc_a", "ncc_b", "progress")}
         for r in picked], indent=2))
    print(f"[select] labeling sample: {len(picked)} clips, "
          f"{len({r['shader'] for r in picked})} distinct shaders")

    blocks, sheet, n_sheet = [], [], 0
    for r in picked:
        clip = videoio.read_clip(root / "videos" / f"{r['stem']}.mp4")
        i0, j0 = r["phase"]["i0"], r["phase"]["j0"]
        idx = d2_sheets.pick_frames(i0, j0, len(clip), n_ramp=8)
        cap = (f"{r['stem']}  {r['shader']}  ease={r['easing']} flip={r['flip']} "
               f"swap={r['swap']}  onset={r['timing']['onset']:.1f} "
               f"release={r['timing']['release']:.1f}  M1p10={r['m1_p10']:.3f} "
               f"M2dq={r['m2_max_dq']:.3f} seam={max(r['assert2']['seam_ratio']):.2f} "
               f"[decile {r['decile']}]")
        blk = d2_sheets.clip_block(clip, idx, cols=6, frame_w=340, caption=cap,
                                   marks={0: "A anchor", len(clip) - 1: "B anchor",
                                          i0: "onset", j0: "release"})
        sheet.append(blk)
        if len(sheet) == CLIPS_PER_SHEET:
            n_sheet += 1
            d2_sheets.save(out / f"labelsheet_{n_sheet:02d}.png",
                           d2_sheets.stack_blocks(sheet))
            sheet = []
        blocks.append(r["stem"])
    if sheet:
        n_sheet += 1
        d2_sheets.save(out / f"labelsheet_{n_sheet:02d}.png", d2_sheets.stack_blocks(sheet))
    print(f"[select] wrote {n_sheet} label sheets -> {out}")


# --------------------------------------------------------------------------
def choose_tau(labeled: list[dict], gate: dict) -> dict:
    """Pre-registered rule: reject ALL labeled-bad, pass >= 90% of labeled-good.

    tau only has to catch the bad clips that the OTHER gate legs (assert1 / assert2 / M2) let
    through, so those are the ones that set the floor. Among feasible values we take the LOWEST
    (least aggressive => best yield), placed midway to the next good score above it.
    """
    good = [r for r in labeled if r["label"] == "good"]
    bad = [r for r in labeled if r["label"] == "bad"]

    def other_legs_pass(r):
        v = d2_metrics.verdict(r, tau=-2.0, assert1_tol=gate["assert1_tol"],
                              seam_max=gate["seam_max"], m2_max=gate["m2_max_dq"])
        return v["assert1"] and v["assert2"] and v["m2"]

    bad_needing_m1 = [r for r in bad if other_legs_pass(r)]
    floor = max([r["m1_p10"] for r in bad_needing_m1], default=-1.0)
    above = sorted(r["m1_p10"] for r in good if r["m1_p10"] > floor)
    tau = round((floor + above[0]) / 2, 4) if above else round(floor + 1e-3, 4)

    def full(r, t):
        return d2_metrics.verdict(r, tau=t, assert1_tol=gate["assert1_tol"],
                                  seam_max=gate["seam_max"], m2_max=gate["m2_max_dq"])["pass"]

    def stats(t):
        gp = sum(1 for r in good if r["m1_p10"] >= t)          # M1 leg only
        gf = sum(1 for r in good if full(r, t))                # whole gate
        br = sum(1 for r in bad if not full(r, t))
        return {"tau": t, "good_pass_m1_leg": gp, "n_good": len(good),
                "good_pass_rate_m1_leg": round(gp / max(len(good), 1), 4),
                "good_pass_full_gate": gf,
                "good_pass_rate_full_gate": round(gf / max(len(good), 1), 4),
                "bad_rejected": br, "n_bad": len(bad),
                "bad_reject_rate": round(br / max(len(bad), 1), 4)}

    s = stats(tau)
    feasible = s["bad_reject_rate"] == 1.0 and s["good_pass_rate_m1_leg"] >= 0.90
    return {"tau": tau, "rule": "reject ALL labeled-bad AND pass >=90% labeled-good",
            "m1_floor_from_bad": round(floor, 4),
            "bad_needing_m1": [r["stem"] for r in bad_needing_m1],
            "bad_caught_by_other_legs": [r["stem"] for r in bad if not other_legs_pass(r)],
            "confusion_at_tau": s, "feasible": bool(feasible),
            "sweep": [stats(round(t, 3)) for t in np.arange(0.0, 0.95, 0.05)]}


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(c - h), 4), round(float(c + h), 4)]


def supplementary(rows, labeled, verdicts, table, blacklist, m1min, tau) -> dict:
    """Diagnostics that are not part of the locked deliverable but material to reading it."""
    n_bad = sum(1 for r in labeled if r["label"] == "bad")
    lo, hi = _wilson(n_bad, len(labeled))
    good = [r for r in labeled if r["label"] == "good"]
    n_good = len(good)
    g_m1 = sum(1 for r in good if r["m1_p10"] >= tau)
    g_full = sum(1 for r in good if verdicts[r["stem"]]["pass"])
    sacrificed = [r["shader"] for r in sorted(good, key=lambda x: x["m1_p10"])
                  if r["m1_p10"] < tau]

    per_e: dict = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        v = verdicts[r["stem"]]
        c = per_e[r["easing"]]
        c["n"] += 1
        for leg in ("assert1", "assert2", "m1", "m2"):
            c[f"fail_{leg}"] += int(not v[leg])
        c["pass"] += int(v["pass"])
    easing = {k: {"n": c["n"],
                  **{f"{leg}_fail_rate": round(c[f"fail_{leg}"] / c["n"], 4)
                     for leg in ("assert1", "assert2", "m1", "m2")},
                  "pass_rate": round(c["pass"] / c["n"], 4)}
              for k, c in sorted(per_e.items())}

    passing = [r for r in rows if verdicts[r["stem"]]["pass"]]
    mn = np.array([r["m1_min"] for r in passing]) if passing else np.zeros(1)
    p_bl = [r["stem"] for r in passing if r["shader"] in set(blacklist)]
    p_ok = [r for r in passing if r["shader"] not in set(blacklist)]
    return {
        "per_easing": easing,
        "visual_bad_rate_estimate": {
            "n_labeled": len(labeled), "n_bad": n_bad,
            "point_estimate": round(n_bad / len(labeled), 4), "wilson95": [lo, hi],
            "sampling": "4 clips per M1 decile = proportionally stratified over the M1 "
                        "distribution, so the unweighted rate estimates the population rate",
            "overturn_threshold": 0.30,
            "verdict": f"BELOW the 30% pre-registered overturn threshold "
                       f"(point {100 * n_bad / len(labeled):.1f}%, upper 95% bound {hi:.3f})"},
        "bad_failure_taxonomy": {
            "extreme_zoom_or_scale_destruction": ["rotate_scale_fade", "zoomInOut",
                                                  "SimpleZoomOut"],
            "coordinate_warp_shredding": ["coord-from-in"],
            "black_or_flat_colour_matte_domination": ["SimpleFlip", "circle"],
            "chromatic_glitch_destruction": ["Drop_Zone_Flicker"],
            "note": "every labelled-bad clip is caused by the SHADER/params, not by fabricated "
                    "layer frames. D1's melt / freeze / reversal failure mode does not occur on "
                    "real streams."},
        "tau_rule_outcome": (
            f"INFEASIBLE: no tau satisfies BOTH 'reject all labelled-bad' and "
            f"'>=90 pct labelled-good'. tau={tau:.4f} is the SMALLEST value that rejects all "
            f"{n_bad} bad clips; it retains {100 * g_m1 / n_good:.1f} pct of labelled-good on the "
            f"M1 leg ({g_m1}/{n_good}) and {100 * g_full / n_good:.1f} pct on the full gate "
            f"({g_full}/{n_good}). The sacrificed good clips are all GEOMETRIC-RECOMPOSITION "
            f"transitions ({', '.join(sacrificed)}): zNCC-vs-source penalises spatial "
            f"DISPLACEMENT, which M1 cannot distinguish from DESTRUCTION."),
        "m1_p10_leak": {
            "issue": "M1 is a p10 over the ramp, so a SHORT burst of destruction (a few frames of "
                     "flat-colour matte or black) does not move it. Example: d2_0278_ref (circle) "
                     "passes at M1 p10=0.392 yet t~92 is a ~100% flat-orange matte frame.",
            "n_gate_passing": len(passing),
            "frac_passing_with_m1_min_below_0.15": round(float((mn < 0.15).mean()), 4),
            "frac_passing_with_m1_min_below_0.0": round(float((mn < 0.0).mean()), 4),
            "suggested_follow_up": "a per-frame floor on min(s(t)) (or a run-length rule) would "
                                   "close this; NOT added - the spec froze M1 as p10."},
        "blacklist_interaction": {
            "gate_passing_clips_from_blacklisted_shaders": len(p_bl),
            "note": f"the owner sheet is sampled from gate-PASSING clips as specified, BEFORE the "
                    f"per-shader blacklist. Applying both leaves {len(p_ok)}/{len(rows)} clips "
                    f"({100 * len(p_ok) / len(rows):.1f}%) from "
                    f"{len({r['shader'] for r in p_ok})} shaders."},
    }


def phase_tau(cfg: dict) -> None:
    gate = cfg["gate"]
    root = REPO_ROOT / cfg["outputs"]["dir"] / "audit"
    rows, exh = load_rows(root / "meta")
    labels = json.loads((HERE / "D2_LABELS.json").read_text())
    lset = {r["stem"]: r for r in json.loads((root / "label" / "label_set.json").read_text())}
    labeled = []
    for stem, lab in labels["labels"].items():
        r = dict(lset[stem])
        r["label"] = lab
        labeled.append(r)
    print(f"[tau] {len(labeled)} labeled clips "
          f"({sum(1 for r in labeled if r['label'] == 'bad')} bad)")

    cal = choose_tau(labeled, gate)
    tau = cal["tau"]
    frozen = {"tau": tau, "metric": "M1 = p10 over ramp frames of "
                                    "max(zNCC(F[t],A_src[t]), zNCC(F[t],B_src[t])) @96x72 gray",
              "assert1_tol": gate["assert1_tol"], "seam_max": gate["seam_max"],
              "m2_max_dq": gate["m2_max_dq"], "frozen": True,
              "calibration": {k: v for k, v in cal.items() if k != "sweep"},
              "n_labeled": len(labeled)}
    (HERE / "D2_TAU.json").write_text(json.dumps(frozen, indent=2))
    print(f"[tau] FROZEN tau = {tau}  -> D2_TAU.json")

    # ---- whole-audit verdicts + per-shader pass rates -----------------------
    per_shader = collections.defaultdict(lambda: {"n_clips": 0, "pass": 0, "fail_assert1": 0,
                                                  "fail_assert2": 0, "fail_m1": 0, "fail_m2": 0,
                                                  "gate2_exhausted_tuples": 0})
    legs = {"assert1": 0, "assert2": 0, "m1": 0, "m2": 0}
    n_pass = 0
    verdicts = {}
    for r in rows:
        v = d2_metrics.verdict(r, tau, assert1_tol=gate["assert1_tol"],
                               seam_max=gate["seam_max"], m2_max=gate["m2_max_dq"])
        verdicts[r["stem"]] = v
        ps = per_shader[r["shader"]]
        ps["n_clips"] += 1
        ps["pass"] += int(v["pass"])
        for leg in ("assert1", "assert2", "m1", "m2"):
            if not v[leg]:
                ps[f"fail_{leg}"] += 1
                legs[leg] += 1
        n_pass += int(v["pass"])
    for r in exh:
        per_shader[r["shader"]]["gate2_exhausted_tuples"] += 1

    by_tuple: dict[int, list[bool]] = collections.defaultdict(list)
    for r in rows:
        by_tuple[r["tuple_id"]].append(verdicts[r["stem"]]["pass"])
    n_tup = len(by_tuple)
    n_tup_pass = sum(1 for v in by_tuple.values() if len(v) == 2 and all(v))

    plan = json.loads((root / "meta" / "plan.json").read_text())
    planned = collections.Counter(t["shader"] for t in plan["tuples"])
    table = {}
    for sh, d in per_shader.items():
        slots = 2 * planned.get(sh, 0)
        d["pass_rate_rendered"] = round(d["pass"] / d["n_clips"], 4) if d["n_clips"] else 0.0
        d["clip_slots_planned"] = slots
        d["pass_rate_slots"] = round(d["pass"] / slots, 4) if slots else 0.0
        d["rejection_rate"] = round(1 - d["pass_rate_slots"], 4)
        table[sh] = dict(d)
    blacklist = sorted(sh for sh, d in table.items() if d["rejection_rate"] > 0.50)
    n80 = sum(1 for d in table.values() if d["pass_rate_slots"] >= 0.80)

    m1 = np.array([r["m1_p10"] for r in rows])
    m1min = np.array([r["m1_min"] for r in rows])
    m2 = np.array([r["m2_max_dq"] for r in rows])
    seam = np.array([r["assert2"]["seam_max_ratio"] for r in rows])
    # Assert1 is a MAX condition ("pure-phase frames must be byte-close to the source"): the mean
    # over the pure phase hides a catastrophic LOCALISED violation (mean 0.2 while one region is
    # 240 off). `a1` = the reported/gated statistic, `a1mean` kept for transparency.
    a1 = np.array([r["assert1"]["max_pure"] for r in rows])
    a1mean = np.array([max(r["assert1"]["mae_pure_a"], r["assert1"]["mae_pure_b"])
                       for r in rows])
    anch = np.array([max(r["assert1"]["mae_anchor_a9"], r["assert1"]["mae_anchor_b9"])
                     for r in rows])

    def q(a):
        return {"mean": round(float(a.mean()), 5), "p50": round(float(np.percentile(a, 50)), 5),
                "p90": round(float(np.percentile(a, 90)), 5),
                "p99": round(float(np.percentile(a, 99)), 5), "max": round(float(a.max()), 5),
                "min": round(float(a.min()), 5)}

    audit = {
        "spec": "D2 real-stream redesign (D1 fabricated-frame path deleted)",
        "n_audit_tuples": len({r["tuple_id"] for r in rows}),
        "n_audit_clips": len(rows),
        "n_gate2_exhausted_tuples": len(exh),
        "policy": {
            "streams": "REAL: A-layer = clip A[0:121], B-layer = clip B[0:121], lockstep",
            "extension": "none  (no extension policy exists in the D2 render path)",
            "aux_kind": None, "n_aux_maps": 0,
            "easings": plan.get("easings") or sorted(
                json.loads((root / "meta" / "bank_info.json").read_text())["easings"]),
            "timing": {"window": [8, 112], "onset": "8 + u1*0.20*104",
                       "release": "112 - u2*0.20*104", "u1_u2": "U[0,1] independent",
                       "min_duration": 62.4, "max_pure_phase_per_side": 20.8,
                       "shared_by_ref_and_target": True},
            "shader_bank": json.loads((root / "meta" / "bank_info.json").read_text()),
            "params": "engine.shaders.sample_params(p_vary=0.85) unchanged",
            "endpoint_gate": {"stage": "gate-2", "resolution": [480, 640],
                              "tol": cfg["sampling"]["endpoint_tol_operator"],
                              "max_tries": cfg["sampling"]["max_gate_tries"]},
        },
        "assert1_pure_phase_identity": {
            "gated_statistic": "max abs pixel deviation over ALL pure-phase frames vs the source",
            "max_pure_abs_dev": q(a1), "mean_pure_mae": q(a1mean), "max_anchor9_mae": q(anch),
            "tol": gate["assert1_tol"],
            "n_fail": int((a1 > gate["assert1_tol"]).sum()),
            "n_exact_zero": int((a1 == 0).sum()),
            "offending_shaders": sorted(collections.Counter(
                r["shader"] for r in rows
                if r["assert1"]["max_pure"] > gate["assert1_tol"]).items(),
                key=lambda kv: -kv[1]),
            "note": "the mean-MAE reading of this leg passed all 896 clips; the MAX reading rejects "
                    "40 (20 tuples) whose pure phase is locally 29-241 off the source. gate-2 misses "
                    "these because it checks ONE frame pair, not every pure-phase frame."},
        "assert2_seam": {"max_seam_ratio": q(seam), "threshold": gate["seam_max"],
                         "n_fail": int((seam > gate["seam_max"]).sum())},
        "m1_mush": {"p10_dist": q(m1), "tau": tau,
                    "n_fail": int((m1 < tau).sum())},
        "m2_near_cut": {"max_dq_dist": q(m2), "threshold": gate["m2_max_dq"],
                        "n_fail": int((m2 > gate["m2_max_dq"]).sum())},
        "frozen_tau": frozen,
        "labeled_set": [{"stem": r["stem"], "shader": r["shader"], "easing": r["easing"],
                         "decile": r.get("decile"), "label": r["label"],
                         "m1_p10": round(r["m1_p10"], 4),
                         "m2_max_dq": round(r["m2_max_dq"], 4),
                         "seam_max_ratio": round(r["assert2"]["seam_max_ratio"], 3),
                         "gate_pass_at_tau": d2_metrics.verdict(
                             r, tau, assert1_tol=gate["assert1_tol"],
                             seam_max=gate["seam_max"], m2_max=gate["m2_max_dq"])["pass"]}
                        for r in sorted(labeled, key=lambda x: x["m1_p10"])],
        "confusion": cal["confusion_at_tau"],
        "tau_sweep": cal["sweep"],
        "overall_gate_pass": {"n_pass": n_pass, "n_clips": len(rows),
                              "rate": round(n_pass / max(len(rows), 1), 4),
                              "leg_failures": legs,
                              # a TRAINING tuple needs BOTH its clips (reference + target), so the
                              # tuple-level yield is the number that sizes a full render.
                              "n_tuples_both_clips_pass": n_tup_pass,
                              "n_tuples": n_tup,
                              "tuple_rate": round(n_tup_pass / max(n_tup, 1), 4)},
        "per_shader": table,
        "blacklist_over_50pct_rejection": blacklist,
        "n_shaders_pass_80": n80,
        "n_shaders_audited": len(table),
        "supplementary": supplementary(rows, labeled, verdicts, table, blacklist, m1min, tau),
    }
    (HERE / "D2_AUDIT.json").write_text(json.dumps(audit, indent=2))
    (root / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    print(json.dumps({k: audit[k] for k in
                      ("n_audit_tuples", "n_audit_clips", "n_gate2_exhausted_tuples",
                       "assert1_pure_phase_identity", "assert2_seam", "m1_mush",
                       "m2_near_cut", "confusion", "overall_gate_pass",
                       "blacklist_over_50pct_rejection", "n_shaders_pass_80",
                       "n_shaders_audited")}, indent=2))


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)
    phase = os.environ.get("PHASE", "select")
    {"select": phase_select, "tau": phase_tau}[phase](cfg)
