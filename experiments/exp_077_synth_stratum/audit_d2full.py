"""exp_077 D2-FULL STAGE 2b (render audit, fan-in) — aggregate every rendered shard into
D2_BUILD_AUDIT.json. Pure CPU, no GL, idempotent, safe to re-run at any time (also useful for
watching a partial render).

Emits exactly the numbers the build must be judged on: tuple/pair counts, the ops-per-pair
histogram (must be all 8), distinct-shaders-per-pair minimum (must be >= 6), per-shader and
per-easing allocation, realized overdraw + attempt stats, gate pass stats + per-shader gate
rejection rates, max pure-phase MAE (must be 0), the m1_min_flag count, and the aux/extension
invariants.

    python audit_d2full.py [--sub d2full]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUDIT = HERE / "D2_BUILD_AUDIT.json"


def merge_audit(update: dict, path: Path = AUDIT) -> None:
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(update)
    path.write_text(json.dumps(cur, indent=2))


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
    return round(xs[k], 5)


def load(run: Path):
    tuples = [json.loads(l) for f in sorted((run / "meta").glob("tuples_shard*.jsonl"))
              for l in f.read_text().splitlines() if l.strip()]
    attempts = [json.loads(l) for f in sorted((run / "meta").glob("attempts_shard*.jsonl"))
                for l in f.read_text().splitlines() if l.strip()]
    stats = [json.loads(f.read_text()) for f in sorted((run / "meta").glob("stats_shard*.json"))]
    clamps = [json.loads(l) for f in sorted((run / "meta").glob("clamp_shard*.jsonl"))
              for l in f.read_text().splitlines() if l.strip()]
    return tuples, attempts, stats, clamps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=None)
    ap.add_argument("--out", default=None, help="override audit json path")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config_d2full.yaml").read_text())
    sub = args.sub or cfg["outputs"]["subdir"]
    run = REPO_ROOT / cfg["outputs"]["dir"] / sub
    out_path = Path(args.out) if args.out else AUDIT
    plan = json.loads((HERE / "d2full_plan.json").read_text())
    d2, gt = cfg["d2"], cfg["gate"]

    tuples, attempts, stats, clamps = load(run)
    if not tuples:
        sys.exit(f"[audit] no accepted tuples under {run}")
    n_expected = d2["n_target_pairs"] * d2["ops_per_target"]

    # ---- structure ----
    by_pair = defaultdict(list)
    for t in tuples:
        by_pair[t["target_index"]].append(t)
    ops_hist = Counter(len(v) for v in by_pair.values())
    distinct_hist = Counter(len({x["shader"] for x in v}) for v in by_pair.values())
    per_shader = Counter(t["shader"] for t in tuples)
    per_easing = Counter(t["easing"] for t in tuples)
    per_flip = Counter(t["flip"] for t in tuples)
    n_swap = sum(1 for t in tuples if t["swap"])
    n_shader_swapped = sum(1 for t in tuples if t.get("shader_swapped"))

    # ---- clip-level metrics over the delivered dataset ----
    clip_rows = [c for t in tuples for c in (t["clips"]["target"], t["clips"]["reference"])]
    max_pure = max(c["assert1"]["max_pure"] for c in clip_rows)
    mae_pure = max(max(c["assert1"]["mae_pure_a"], c["assert1"]["mae_pure_b"]) for c in clip_rows)
    max_anchor = max(max(c["assert1"]["mae_anchor_a9"], c["assert1"]["mae_anchor_b9"])
                     for c in clip_rows)
    seams = [c["assert2"]["seam_max_ratio"] for c in clip_rows]
    m1s = [c["m1_p10"] for c in clip_rows]
    m2s = [c["m2_max_dq"] for c in clip_rows]
    n_flag_clips = sum(1 for c in clip_rows if c["m1_min_flag"])
    n_flag_tuples = sum(1 for t in tuples if t["m1_min_flag"])

    # ---- rejection-sampling economics ----
    att_counts = [t["attempts"] for t in tuples]
    n_renders = sum(1 for a in attempts if "target" in a) + \
        sum(1 for a in attempts if "reference" in a)
    n_clips = 2 * len(tuples)
    rendered_stats = [s for s in stats]
    # gate legs, over every rendered-and-scored clip (accepted + rejected)
    leg_fail = Counter()
    n_scored = n_scored_pass = 0
    per_shader_att = Counter()
    per_shader_rej = Counter()
    for a in attempts:
        for role in ("target", "reference"):
            c = a.get(role)
            if not c:
                continue
            n_scored += 1
            v = c["verdict"]
            if v["pass"]:
                n_scored_pass += 1
            else:
                for leg in ("assert1", "assert2", "m1", "m2"):
                    if not v[leg]:
                        leg_fail[leg] += 1
        sh = a.get("shader")
        if sh and ("target" in a or a.get("gate2_exhausted")):
            per_shader_att[sh] += 1
            if not a.get("accepted"):
                per_shader_rej[sh] += 1
    gate_rej_rate = {s: round(per_shader_rej[s] / per_shader_att[s], 4)
                     for s in sorted(per_shader_att) if per_shader_att[s]}
    worst_rej = sorted(gate_rej_rate.items(), key=lambda kv: -kv[1])[:15]

    # ---- clamp events over the delivered dataset ----
    cl_rule, cl_param, cl_shader, cl_default = Counter(), Counter(), Counter(), Counter()
    for row in clamps:
        for e in row["events"]:
            cl_rule[e["rule"]] += 1
            cl_param[f"{e['shader']}.{e['param']}"] += 1
            cl_shader[e["shader"]] += 1
            if e.get("was_default"):
                cl_default[f"{e['shader']}.{e['param']}"] += 1
    acc_clamp = Counter()
    for t in tuples:
        for e in t.get("clamp_events", []) or []:
            acc_clamp[f"{e['shader']}.{e['param']}"] += 1

    u1 = [t["timing"]["u1"] for t in tuples]
    u2 = [t["timing"]["u2"] for t in tuples]

    section = {
        "render_dir": str(run),
        "n_tuples": len(tuples),
        "n_tuples_expected": n_expected,
        "complete": len(tuples) == n_expected,
        "n_target_pairs": len(by_pair),
        "n_clips": n_clips,
        "n_shards_reported": len(stats),
        "ops_per_pair_hist": dict(sorted(ops_hist.items())),
        "all_pairs_have_exactly_8_ops": set(ops_hist) == {8} and len(by_pair) == d2["n_target_pairs"],
        "distinct_shaders_per_pair_hist": dict(sorted(distinct_hist.items())),
        "distinct_shaders_per_pair_min": min(distinct_hist) if distinct_hist else None,
        "distinct_shaders_min_ge_6": (min(distinct_hist) >= d2["min_distinct_shaders"]
                                      if distinct_hist else False),
        "per_shader_allocation": dict(sorted(per_shader.items())),
        "per_shader_min_max": [min(per_shader.values()), max(per_shader.values())],
        "n_shaders_used": len(per_shader),
        "per_easing_allocation": dict(sorted(per_easing.items())),
        "per_flip_allocation": dict(sorted(per_flip.items())),
        "swap_fraction": round(n_swap / len(tuples), 4),
        "n_slots_shader_swapped_by_rejection": n_shader_swapped,
        # invariants
        "aux_kind_all_null": all(t["aux_kind"] is None for t in tuples),
        "extension_all_none": all(t["extension"] == "none" for t in tuples),
        "easings_within_keep7": sorted(set(per_easing)) == sorted(plan["keep_easings"]) or
                                set(per_easing) <= set(plan["keep_easings"]),
        "no_blacklisted_shader_used": not (set(per_shader) & set(plan["blacklist"])),
        "no_holdout_shader_used": not (set(per_shader) & set(plan["holdout_shaders"])),
        # assert1 / gate distributions on the delivered clips
        "max_pure_phase_MAX_abs_diff": round(max_pure, 6),
        "max_pure_phase_MAE": round(mae_pure, 6),
        "max_anchor9_MAE": round(max_anchor, 6),
        "pure_phase_identity_exact": max_pure == 0.0,
        "seam_max_ratio_dist": {"p50": q(seams, .5), "p90": q(seams, .9), "max": round(max(seams), 4)},
        "m1_p10_dist": {"min": round(min(m1s), 4), "p10": q(m1s, .1), "p50": q(m1s, .5)},
        "m2_max_dq_dist": {"p50": q(m2s, .5), "p90": q(m2s, .9), "max": round(max(m2s), 4)},
        "tau_used": gt["tau"],
        "n_clips_m1_min_flag": n_flag_clips,
        "n_tuples_m1_min_flag": n_flag_tuples,
        "m1_min_flag_threshold": gt["m1_min_flag_threshold"],
        "m1_min_flag_is_gating": False,
        # rejection sampling economics
        "attempts_per_slot": {"mean": round(st.mean(att_counts), 3), "p50": q(att_counts, .5),
                              "p90": q(att_counts, .9), "max": max(att_counts)},
        "attempts_hist": dict(sorted(Counter(att_counts).items())),
        "n_renders_total": n_renders,
        "realized_overdraw": round(n_renders / n_clips, 4),
        "overdraw_ceiling": d2["overdraw_ceiling"],
        "within_overdraw_ceiling": (n_renders / n_clips) <= d2["overdraw_ceiling"],
        "gate_pass_stats": {"n_clips_scored": n_scored, "n_pass": n_scored_pass,
                            "clip_pass_rate": round(n_scored_pass / n_scored, 4) if n_scored else None,
                            "leg_failures": dict(leg_fail)},
        "per_shader_gate_rejection_rate": gate_rej_rate,
        "worst_15_shaders_by_gate_rejection": [{"shader": s, "rej_rate": r} for s, r in worst_rej],
        "n_slots_exhausted": sum(s.get("n_slots_exhausted", 0) for s in rendered_stats),
        "n_timing_redraws": sum(s.get("n_timing_redraws", 0) for s in rendered_stats),
        "n_gate2_exhausted_draws": sum(s.get("n_gate2_exhausted_draws", 0) for s in rendered_stats),
        "render_wall_min_per_shard": {s["shard"]: s.get("wall_min") for s in rendered_stats},
        # timing law (should stay ~U[0,1] because timing is fixed per slot, not resampled)
        "timing_u1_mean_sd": [round(st.mean(u1), 4), round(st.pstdev(u1), 4)],
        "timing_u2_mean_sd": [round(st.mean(u2), 4), round(st.pstdev(u2), 4)],
        "onset_min_max": [round(min(t["timing"]["onset"] for t in tuples), 3),
                          round(max(t["timing"]["onset"] for t in tuples), 3)],
        "release_min_max": [round(min(t["timing"]["release"] for t in tuples), 3),
                            round(max(t["timing"]["release"] for t in tuples), 3)],
        # clamp diagnostics
        "param_clamp_active": all(t.get("param_clamp", False) for t in tuples),
        "clamp_events_total_all_draws": sum(cl_rule.values()),
        "clamp_events_by_rule": dict(cl_rule.most_common()),
        "clamp_top20_params": dict(cl_param.most_common(20)),
        "clamp_top15_shaders": dict(cl_shader.most_common(15)),
        "clamp_events_moving_a_default": dict(cl_default.most_common(10)),
        "clamp_events_on_ACCEPTED_operators_top15": dict(acc_clamp.most_common(15)),
        "n_accepted_tuples_with_a_clamped_param": sum(
            1 for t in tuples if t.get("clamp_events")),
    }
    merge_audit({"stage2_render": section}, out_path)
    brief = {k: section[k] for k in (
        "n_tuples", "complete", "n_target_pairs", "ops_per_pair_hist",
        "all_pairs_have_exactly_8_ops", "distinct_shaders_per_pair_min",
        "per_shader_min_max", "per_easing_allocation", "max_pure_phase_MAX_abs_diff",
        "pure_phase_identity_exact", "n_clips_m1_min_flag", "realized_overdraw",
        "gate_pass_stats", "n_slots_exhausted", "aux_kind_all_null", "extension_all_none",
        "param_clamp_active", "clamp_events_by_rule")}
    print(json.dumps(brief, indent=2))
    print(f"[audit] -> {out_path}")


if __name__ == "__main__":
    main()
