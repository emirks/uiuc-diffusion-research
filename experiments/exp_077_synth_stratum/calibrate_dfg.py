"""exp_077 D2-FULL — CALIBRATE the degenerate-frame gate (DFG). PRE-COMMITTED, before rendering.

    PHASE=features python calibrate_dfg.py     # BATCH ONLY (re-renders clips, reads mp4)
    PHASE=grid     python calibrate_dfg.py     # pure post-process, runs anywhere

PHASE=features
--------------
Recomputes each already-graded clip EXACTLY as it was produced — the recorded operator
(shader / params / easing / flip / swap) + the recorded timing, re-rendered on the REAL source
streams — so the DFG features are measured on RAW uint8 frames, the same substrate the gate
sees at render time. (The mp4 on disk is also decoded for the graded clips and the raw-vs-mp4
feature delta is reported, so the codec is never a hidden confound.)

Clip sets:
  audit       all 896 clips of the 448-tuple D2 audit  (`audit/meta/rows_shard*.jsonl`)
              -> carries the 40 BASELINE labels (D2_LABELS.json) and the ROUND-5 bottom-10
                 (floor_bottom10.json + D2_POLICY_FINAL.json::floor_labels), and doubles as the
                 projection set for the expected DFG rejection rate / per-shader rates.
  firstchunk  the 64 clips of the KILLED param-clamp chunk (`d2full_firstchunk/meta/...`)
              -> its BAD clips are EXTRA POSITIVES (a black frame is a black frame even if the
                 clamp induced it). The clamped clips themselves are discarded.

PHASE=grid
----------
Grid: theta_flat in {0.02, 0.03, 0.05} x theta_black in {0.04, 0.06, 0.08}
      (theta_white = 1 - theta_black, symmetric) x K in {2, 3, 4, 6}.
The saturated-wash test is considered ONLY if the winning sat-disabled config fails to catch the
saturated-flat-colour positives (the pure-green fadecolor clips) — per the task spec.

ACCEPTANCE BAR (declared before any number was looked at; NOT softenable):
  recall >= 5/7 on the baseline BAD set, 5/5 on the round-5 BAD set,
  false positives <= 2 of the 38 GOOD clips, and StaticFade must pass 5/5.
If nothing meets it, the ESCAPE triggers: no detector ships.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

import d2_metrics  # noqa: E402
import dfg  # noqa: E402
import streams_real as sr  # noqa: E402

CONFIG_PATH = HERE / "config_d2full.yaml"
FEAT_SUB = "dfg_calib"
log = logging.getLogger("exp077.dfg")

# ---- the declared bar ----------------------------------------------------
BAR = {"baseline_bad_recall_min": 5, "baseline_bad_n": 7,
       "round5_bad_recall_min": 5, "round5_bad_n": 5,
       "max_false_positives": 2, "good_n": 38, "staticfade_pass_required": 5}

GRID_FLAT = (0.02, 0.03, 0.05)
GRID_BLACK = (0.04, 0.06, 0.08)
GRID_K = (2, 3, 4, 6)
GRID_SAT = (0.35, 0.45, 0.55)
GRID_SAT_MULT = (2.0, 3.0)


# ==========================================================================
# labels
# ==========================================================================
def load_labels() -> dict:
    """stem -> {"set": ..., "label": "bad"|"good", "note": ...} for every GRADED clip."""
    out: dict[str, dict] = {}
    lab = json.loads((HERE / "D2_LABELS.json").read_text())
    for stem, v in lab["labels"].items():
        out[stem] = {"set": "baseline", "label": v,
                     "note": lab["why_bad"].get(stem, lab["borderline_calls"].get(stem, ""))}

    pol = json.loads((HERE / "D2_POLICY_FINAL.json").read_text())
    bottom10 = json.loads((HERE / "floor_bottom10.json").read_text())
    r5_bad = {e.split("/")[0] for e in pol["floor_labels"]["bad"]}
    assert len(r5_bad) == 5, r5_bad
    for stem in bottom10:
        assert stem not in out, f"round-5 stem {stem} collides with a baseline label"
        out[stem] = {"set": "round5", "label": "bad" if stem in r5_bad else "good",
                     "note": "round-5 bottom-10 by m1_min"}

    fc = json.loads((HERE / "D2_FIRSTCHUNK_VISUAL.json").read_text())
    for tid, (cls, shader, note) in fc["targets"].items():
        out[f"{tid}_tgt"] = {"set": "clamped", "label": {"BAD": "bad"}.get(cls, cls.lower()),
                             "grade": cls, "note": note}
    for stem, (cls, shader, note) in fc["references_examined"].items():
        out[stem] = {"set": "clamped", "label": {"BAD": "bad"}.get(cls, cls.lower()),
                     "grade": cls, "note": note}
    return out


# ==========================================================================
# PHASE = features
# ==========================================================================
def _rows_audit(root: Path) -> list[dict]:
    rows = []
    for f in sorted((root / "audit" / "meta").glob("rows_shard*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append({
                    "stem": r["stem"], "set": "audit", "role": r["role"],
                    "shader": r["shader"], "easing": r["easing"], "flip": r["flip"],
                    "swap": r["swap"], "params": r["params"], "timing": r["timing"],
                    "phase": r["phase"], "A": r["A"], "B": r["B"],
                    "gate_metrics": {"assert1": r["assert1"], "assert2": r["assert2"],
                                     "m1_p10": r["m1_p10"], "m1_min": r["m1_min"],
                                     "m2_max_dq": r["m2_max_dq"]},
                    "mp4": root / "audit" / "videos" / f"{r['stem']}.mp4",
                })
    return rows


def _rows_firstchunk(root: Path) -> list[dict]:
    rows = []
    for f in sorted((root / "d2full_firstchunk" / "meta").glob("tuples_shard*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            for role, stem_key, pair_key in (("target", "target_stem", "target_pair"),
                                             ("reference", "reference_stem", "ref_pair")):
                c = t["clips"][role]
                rows.append({
                    "stem": t[stem_key], "set": "firstchunk", "role": role,
                    "shader": t["shader"], "easing": t["easing"], "flip": t["flip"],
                    "swap": t["swap"], "params": t["params"], "timing": t["timing"],
                    "phase": t["phase"], "A": t[pair_key]["A"], "B": t[pair_key]["B"],
                    "gate_metrics": {"assert1": c["assert1"], "assert2": c["assert2"],
                                     "m1_p10": c["m1_p10"], "m1_min": c["m1_min"],
                                     "m2_max_dq": c["m2_max_dq"]},
                    "mp4": root / "d2full_firstchunk" / "videos" / f"{t[stem_key]}.mp4",
                })
    return rows


def phase_features() -> None:
    import render_d2 as rd2
    from engine import operators, videoio
    from engine.glrunner import GLRunner

    cfg = load_config(CONFIG_PATH)
    inf, gt = cfg["inference"], cfg["gate"]
    H, W, T, K = inf["height"], inf["width"], inf["num_frames"], inf["anchor_frames"]
    root = REPO_ROOT / cfg["outputs"]["dir"]
    out_dir = root / FEAT_SUB
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / "DFG_FEATURES.jsonl"

    labels = load_labels()
    rows = _rows_audit(root) + _rows_firstchunk(root)
    log.info("clips: %d (audit %d + firstchunk %d) | graded labels: %d", len(rows),
             sum(r["set"] == "audit" for r in rows), sum(r["set"] == "firstchunk" for r in rows),
             len(labels))
    missing = sorted(set(labels) - {r["stem"] for r in rows})
    assert not missing, f"graded stems with no metadata row: {missing}"

    done = set()
    if feat_path.exists():
        for line in feat_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["stem"])
        log.info("resuming: %d clips already featured", len(done))
    rows = [r for r in rows if r["stem"] not in done]
    rows.sort(key=lambda r: (r["A"], r["B"], r["stem"]))     # maximise the source-clip cache

    runner = GLRunner(W, H)
    log.info("GL: %s", runner.renderer_name())
    plan = json.loads((HERE / "d2full_plan.json").read_text())
    bank, _ = sr.d2_shader_bank(runner, Path(cfg["model"]["shader_bank"]),
                                tol=cfg["sampling"]["endpoint_tol"],
                                holdout=plan["holdout_shaders"])
    cache = rd2.ClipCache(REPO_ROOT / cfg["inputs"]["clips_dir"], 120)
    gray_src: dict[str, np.ndarray] = {}
    f = open(feat_path, "a", buffering=1)
    t0 = time.time()

    for n, r in enumerate(rows):
        if r["shader"] not in bank:
            log.warning("%s: shader %s not in gate-1 bank — skipped", r["stem"], r["shader"])
            continue
        a = cache.get(r["A"], f"clips/{r['A']}.mp4")
        b = cache.get(r["B"], f"clips/{r['B']}.mp4")
        for cid, arr in ((r["A"], a), (r["B"], b)):
            if cid not in gray_src:
                gray_src[cid] = d2_metrics.to_small_gray(arr)
        op = operators.Operator(op_id=f"recon_{r['stem']}", shader=r["shader"],
                                params=r["params"], easing=r["easing"], flip=r["flip"],
                                swap=r["swap"], extension=sr.EXTENSION, aux_kind=None, aux_seed=0)
        p = sr.progress_ramp(T, K, op.easing, r["timing"]["onset"], r["timing"]["release"])
        clip = sr.render_real(runner, bank, op, a, b, p)
        i0, j0 = r["phase"]["i0"], r["phase"]["j0"]
        g = d2_metrics.to_small_gray(clip)
        ft = dfg.features(clip, i0, j0, gray=g, gray_a=gray_src[r["A"]],
                          gray_b=gray_src[r["B"]], need_sat=True)

        rec = {k: r[k] for k in ("stem", "set", "role", "shader", "easing", "flip", "swap",
                                 "A", "B", "phase")}
        rec["params"] = r["params"]
        rec["timing"] = {k: r["timing"][k] for k in ("onset", "release", "duration")}
        rec["verdict_frozen"] = d2_metrics.verdict(
            r["gate_metrics"], gt["tau"], assert1_tol=gt["assert1_tol"],
            seam_max=gt["seam_max"], m2_max=gt["m2_max_dq"])
        rec["m1_p10"] = r["gate_metrics"]["m1_p10"]
        rec["m1_min"] = r["gate_metrics"]["m1_min"]
        rec["label"] = labels.get(r["stem"])
        rec["features"] = ft

        # ---- graded clips: quantify the RAW-vs-mp4 gap so the codec is not a hidden confound
        if r["stem"] in labels and Path(r["mp4"]).exists():
            dec = videoio.read_clip(Path(r["mp4"]))
            gd = d2_metrics.to_small_gray(dec)
            fd = dfg.features(dec, i0, j0, gray=gd, gray_a=gray_src[r["A"]],
                              gray_b=gray_src[r["B"]], need_sat=True)
            rec["mp4_delta"] = {
                "recon_vs_mp4_mae": round(float(np.abs(clip.astype(np.float32)
                                                       - dec.astype(np.float32)).mean()), 4),
                "dL_max": round(float(np.abs(np.array(ft["L"]) - np.array(fd["L"])).max()), 5),
                "dS_max": round(float(np.abs(np.array(ft["S"]) - np.array(fd["S"])).max()), 5),
                "dsat_max": round(float(np.abs(np.array(ft["sat"])
                                               - np.array(fd["sat"])).max()), 5)}
            rec["features_mp4"] = fd

        f.write(json.dumps(rec) + "\n")
        if (n + 1) % 50 == 0 or n == len(rows) - 1:
            log.info("%d/%d  %.1f min  cache h/m=%d/%d", n + 1, len(rows),
                     (time.time() - t0) / 60, cache.hits, cache.misses)
    f.close()
    log.info("[features] -> %s", feat_path)


# ==========================================================================
# PHASE = grid
# ==========================================================================
def _score_config(recs: dict, cfg: dict) -> dict:
    """Apply one DFG config to every graded clip + the projection set."""
    res = {s: dfg.evaluate(r["features"], cfg) for s, r in recs.items()}
    groups: dict[str, list[str]] = defaultdict(list)
    for s, r in recs.items():
        lab = r["label"]
        if lab is None:
            continue
        if lab["set"] == "baseline":
            groups["baseline_" + lab["label"]].append(s)
        elif lab["set"] == "round5":
            groups["round5_" + lab["label"]].append(s)
        elif lab["set"] == "clamped" and lab["label"] == "bad":
            groups["clamped_bad"].append(s)
        elif lab["set"] == "clamped":
            groups["clamped_" + lab["label"]].append(s)

    def rec(k):
        return sum(res[s]["reject"] for s in groups.get(k, []))

    good = groups["baseline_good"] + groups["round5_good"]
    fp_stems = [s for s in good if res[s]["reject"]]
    sf = [s for s in groups["round5_good"] if recs[s]["shader"] == "StaticFade"]
    out = {
        "config": {k: cfg.get(k, dfg.DEFAULT_CONFIG[k]) for k in dfg.DEFAULT_CONFIG},
        "baseline_bad_recall": rec("baseline_bad"), "baseline_bad_n": len(groups["baseline_bad"]),
        "round5_bad_recall": rec("round5_bad"), "round5_bad_n": len(groups["round5_bad"]),
        "clamped_bad_recall": rec("clamped_bad"), "clamped_bad_n": len(groups["clamped_bad"]),
        "clamped_marginal_flagged": rec("clamped_marginal"),
        "clamped_marginal_n": len(groups["clamped_marginal"]),
        "clamped_ok_flagged": rec("clamped_ok"), "clamped_ok_n": len(groups["clamped_ok"]),
        "good_n": len(good), "false_positives": len(fp_stems), "fp_stems": fp_stems,
        "staticfade_n": len(sf), "staticfade_pass": sum(not res[s]["reject"] for s in sf),
        "missed_baseline_bad": [s for s in groups["baseline_bad"] if not res[s]["reject"]],
        "missed_round5_bad": [s for s in groups["round5_bad"] if not res[s]["reject"]],
        "missed_clamped_bad": [s for s in groups["clamped_bad"] if not res[s]["reject"]],
    }
    out["total_bad_recall"] = (out["baseline_bad_recall"] + out["round5_bad_recall"]
                              + out["clamped_bad_recall"])
    out["total_bad_n"] = out["baseline_bad_n"] + out["round5_bad_n"] + out["clamped_bad_n"]
    out["meets_bar"] = bool(
        out["baseline_bad_recall"] >= BAR["baseline_bad_recall_min"]
        and out["round5_bad_recall"] >= BAR["round5_bad_recall_min"]
        and out["false_positives"] <= BAR["max_false_positives"]
        and out["staticfade_pass"] == out["staticfade_n"] == BAR["staticfade_pass_required"])
    return out, res


def _projection(recs: dict, res: dict, keep_shaders: set, keep_easings: set) -> dict:
    """Expected DFG rejection rate among clips that ALREADY pass the frozen gate and are drawn
    from the D2-FULL sampling distribution (keep_shaders x keep_easings)."""
    sel = [s for s, r in recs.items()
           if r["set"] == "audit" and r["verdict_frozen"]["pass"]
           and r["shader"] in keep_shaders and r["easing"] in keep_easings]
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s in sel:
        per[recs[s]["shader"]][0] += 1
        per[recs[s]["shader"]][1] += int(res[s]["reject"])
    n_rej = sum(res[s]["reject"] for s in sel)
    return {
        "n_gate_passing_in_distribution": len(sel), "n_dfg_rejected": n_rej,
        "dfg_reject_rate": round(n_rej / max(len(sel), 1), 4),
        "per_shader": {k: {"n": v[0], "rejected": v[1], "rate": round(v[1] / v[0], 4)}
                       for k, v in sorted(per.items(), key=lambda kv: -kv[1][1] / kv[1][0])},
        "shaders_over_60pct_min20": sorted(k for k, v in per.items()
                                          if v[0] >= 20 and v[1] / v[0] > 0.60),
        "shaders_over_60pct_any_n": sorted(k for k, v in per.items() if v[1] / v[0] > 0.60),
    }


def phase_grid() -> None:
    cfg_y = load_config(CONFIG_PATH)
    root = REPO_ROOT / cfg_y["outputs"]["dir"]
    feat_path = root / FEAT_SUB / "DFG_FEATURES.jsonl"
    recs = {}
    for line in feat_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["stem"]] = r
    graded = {s: r for s, r in recs.items() if r["label"] is not None}
    log.info("features: %d clips (%d graded)", len(recs), len(graded))

    plan = json.loads((HERE / "d2full_plan.json").read_text())
    keep_shaders, keep_easings = set(plan["keep_shaders"]), set(plan["keep_easings"])

    # ---- the declared grid, saturated-wash test DISABLED --------------------
    table = []
    for flat, black, k in product(GRID_FLAT, GRID_BLACK, GRID_K):
        c = {"theta_flat": flat, "theta_black": black, "theta_white": round(1 - black, 4),
             "K": k, "theta_sat": None, "sat_flat_mult": 2.5}
        sc, _ = _score_config(recs, c)
        table.append(sc)

    def rank_key(sc):
        c = sc["config"]
        # PRE-COMMITTED: maximise total BAD recall; tie-break LARGER K, then fewer FPs, then the
        # most conservative thresholds (smallest theta_flat, then smallest theta_black), then
        # sat-disabled before sat-enabled.
        return (-sc["total_bad_recall"], -c["K"], sc["false_positives"], c["theta_flat"],
                c["theta_black"], c["theta_sat"] is not None)

    feasible = sorted([s for s in table if s["meets_bar"]], key=rank_key)
    result = {
        "bar": BAR,
        "grid": {"theta_flat": list(GRID_FLAT), "theta_black": list(GRID_BLACK),
                 "theta_white": "1 - theta_black (symmetric)", "K": list(GRID_K)},
        "selection_rule": ("max total BAD recall (baseline + round5 + clamped) subject to the "
                           "bar; tie-break larger K, then fewer FPs, then smaller theta_flat, "
                           "then smaller theta_black, then sat-disabled"),
        "n_graded": len(graded), "n_configs_evaluated": len(table),
        "n_feasible": len(feasible),
        "table_sat_disabled": sorted(table, key=rank_key),
    }

    if not feasible:
        result["outcome"] = "ESCAPE"
        result["ships"] = False
        result["note"] = ("No grid config meets the declared bar. Per the pre-committed escape: "
                          "no detector ships; render UNCLAMPED at baseline and document the "
                          "residual.")
        (HERE / "DFG_CALIB.json").write_text(json.dumps(result, indent=1))
        log.error("ESCAPE: no feasible config")
        _print_table(result, recs, None)
        return

    best = feasible[0]

    # ---- saturated-wash test: added ONLY if the winner misses a saturated-flat positive -----
    sat_positives = [s for s, r in graded.items()
                     if r["label"]["label"] == "bad" and "saturated" in (r["label"].get("note")
                                                                        or "").lower()]
    missed_sat = [s for s in sat_positives if s in best["missed_clamped_bad"]
                  or s in best["missed_baseline_bad"] or s in best["missed_round5_bad"]]
    result["saturated_positives"] = sat_positives
    result["saturated_positives_missed_by_best_sat_disabled"] = missed_sat
    sat_table = []
    if missed_sat:
        for flat, black, k, ts, mult in product(GRID_FLAT, GRID_BLACK, GRID_K,
                                                GRID_SAT, GRID_SAT_MULT):
            c = {"theta_flat": flat, "theta_black": black, "theta_white": round(1 - black, 4),
                 "K": k, "theta_sat": ts, "sat_flat_mult": mult}
            sc, _ = _score_config(recs, c)
            sat_table.append(sc)
        sat_feasible = sorted([s for s in sat_table if s["meets_bar"]], key=rank_key)
        result["n_sat_configs_evaluated"] = len(sat_table)
        result["table_sat_enabled_top20"] = sorted(sat_table, key=rank_key)[:20]
        if sat_feasible and sat_feasible[0]["total_bad_recall"] > best["total_bad_recall"]:
            result["sat_decision"] = ("ENABLED — strictly improves total BAD recall "
                                      f"{best['total_bad_recall']} -> "
                                      f"{sat_feasible[0]['total_bad_recall']} within the bar")
            best = sat_feasible[0]
        else:
            result["sat_decision"] = ("DISABLED — no feasible sat-enabled config improves total "
                                      "BAD recall over the sat-disabled winner")
    else:
        result["sat_decision"] = ("DISABLED — the flatness test already catches every "
                                  "saturated-flat-colour positive")

    chosen = best["config"]
    sc, res = _score_config(recs, chosen)
    result.update({"outcome": "SHIP", "ships": True, "chosen": chosen, "chosen_scores": sc,
                   "projection": _projection(recs, res, keep_shaders, keep_easings)})
    result["per_clip"] = [{
        "stem": s, "set": r["label"]["set"], "label": r["label"]["label"],
        "grade": r["label"].get("grade", r["label"]["label"].upper()),
        "shader": r["shader"], "easing": r["easing"],
        "window": [r["features"]["i0"], r["features"]["j0"]], "n_window": r["features"]["n_window"],
        "L_min": res[s]["worst"]["L_min"], "L_max": res[s]["worst"]["L_max"],
        "S_min": res[s]["worst"]["S_min"], "sat_max": res[s]["worst"]["sat_max"],
        "m1_p10": r["m1_p10"], "m1_min": r["m1_min"],
        "n_flag": res[s]["n_flag"], "by_test": res[s]["by_test"],
        "flagged": res[s]["reject"], "note": (r["label"].get("note") or "")[:90],
        "mp4_delta": r.get("mp4_delta"),
    } for s, r in sorted(graded.items(), key=lambda kv: (kv[1]["label"]["set"],
                                                         kv[1]["label"]["label"], kv[0]))]
    dl = [c["mp4_delta"] for c in result["per_clip"] if c["mp4_delta"]]
    if dl:
        result["raw_vs_mp4"] = {
            "n": len(dl),
            "recon_vs_mp4_mae_max": max(d["recon_vs_mp4_mae"] for d in dl),
            "dL_max": max(d["dL_max"] for d in dl), "dS_max": max(d["dS_max"] for d in dl),
            "dsat_max": max(d["dsat_max"] for d in dl)}
    (HERE / "DFG_CALIB.json").write_text(json.dumps(result, indent=1))
    log.info("[grid] -> %s", HERE / "DFG_CALIB.json")
    _print_table(result, recs, res)


def _print_table(result: dict, recs: dict, res: dict | None) -> None:
    print("\n=== DFG grid (sat disabled), ranked ===")
    print(f"{'flat':>5} {'blk':>5} {'wht':>5} {'K':>2} | {'base':>5} {'r5':>4} {'clmp':>5} "
          f"{'FP':>3} {'SF':>4} | bar")
    for s in result["table_sat_disabled"]:
        c = s["config"]
        print(f"{c['theta_flat']:>5.2f} {c['theta_black']:>5.2f} {c['theta_white']:>5.2f} "
              f"{c['K']:>2} | {s['baseline_bad_recall']}/{s['baseline_bad_n']:<3} "
              f"{s['round5_bad_recall']}/{s['round5_bad_n']:<2} "
              f"{s['clamped_bad_recall']}/{s['clamped_bad_n']:<3} "
              f"{s['false_positives']:>3} {s['staticfade_pass']}/{s['staticfade_n']:<2} | "
              f"{'PASS' if s['meets_bar'] else '-'}")
    if not result.get("ships"):
        print("\nOUTCOME: ESCAPE — no detector ships.")
        return
    print(f"\nCHOSEN: {json.dumps(result['chosen'])}")
    print(f"sat_decision: {result['sat_decision']}")
    print(json.dumps({k: v for k, v in result["chosen_scores"].items() if k != "config"}, indent=1))
    print(f"\nprojection: {json.dumps({k: v for k, v in result['projection'].items() if k != 'per_shader'}, indent=1)}")
    print(f"raw_vs_mp4: {json.dumps(result.get('raw_vs_mp4'))}")
    print("\n=== per-clip calibration table ===")
    print(f"{'stem':<16}{'set':<11}{'lab':<5}{'shader':<20}{'win':<11}{'Lmin':>7}{'Lmax':>7}"
          f"{'Smin':>7}{'satmx':>7}{'m1min':>8}{'nflg':>5}  {'tests':<22}FLAG")
    for c in result["per_clip"]:
        bt = ",".join(f"{k}:{v}" for k, v in c["by_test"].items() if v)
        print(f"{c['stem']:<16}{c['set']:<11}{c['grade'][:4]:<5}{c['shader'][:19]:<20}"
              f"{str(c['window']):<11}{c['L_min']:>7.3f}{c['L_max']:>7.3f}{c['S_min']:>7.3f}"
              f"{c['sat_max']:>7.3f}{c['m1_min']:>8.3f}{c['n_flag']:>5}  {bt:<22}"
              f"{'YES' if c['flagged'] else '.'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    ph = os.environ.get("PHASE", "grid")
    if ph == "features":
        phase_features()
    elif ph == "grid":
        phase_grid()
    else:
        sys.exit(f"unknown PHASE={ph}")
