"""exp_077 D2-FULL STAGE 1 (planning) — the 3,072-slot plan for the FINAL synthetic dataset.

Deterministic, pure CPU, NO GL (the shader bank is taken verbatim from D2_POLICY_FINAL.json's
`keep_shaders`, which was itself derived from the gate-1-validated 448-tuple audit, so no
re-validation is needed here; the renderer re-runs gate-1 and hard-asserts the 72 are present).

LOCKED STRUCTURE (identical to the validated D1 plan, minus aux):
* 384 distinct TARGET pairs + 768 distinct REF pairs, cross-clip (A, B) drawn from the tightened
  bank (227 clips). A tuple's ref pair is CONTENT-DISJOINT from its target pair. Each ref pair is
  reused exactly 4x (768*4 = 3072) and the 8 refs of a target pair are distinct.
* Each TARGET pair gets exactly 8 operator slots spanning 8 DISTINCT shaders (>= 6 required),
  frequency-balanced over the 72 keep_shaders (=> 42 or 43 slots per shader).
* Everything continuous (params, easing, flip, swap, timing) is rejection-sampled per slot at
  render time against the frozen gate — the plan only fixes endpoints + the starting shader.

    python plan_d2_full.py            # writes d2full_plan.json + D2_BUILD_AUDIT.json stages 0-1
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.exp_utils import load_config  # noqa: E402

CONFIG_PATH = HERE / "config_d2full.yaml"
AUDIT = HERE / "D2_BUILD_AUDIT.json"
PLAN = HERE / "d2full_plan.json"

ENGINE_EASINGS = ["in_cubic", "in_expo", "in_out_cubic", "in_out_sine", "linear", "mid_hold",
                  "out_cubic", "out_expo", "smootherstep", "smoothstep", "snap_early", "snap_late"]


def merge_audit(update: dict) -> None:
    cur = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
    cur.update(update)
    AUDIT.write_text(json.dumps(cur, indent=2))


def build_pairs(clip_ids, n_target, n_ref, rng):
    """Sample n_target + n_ref DISTINCT cross-clip ORDERED pairs (A != B). (from plan_d1.py)"""
    seen, pool = set(), []
    need = n_target + n_ref
    tries = 0
    while len(pool) < need and tries < need * 100:
        a, b = rng.sample(clip_ids, 2)
        if (a, b) not in seen:
            seen.add((a, b))
            pool.append((a, b))
        tries += 1
    if len(pool) < need:
        raise RuntimeError(f"could not sample {need} distinct pairs (got {len(pool)})")
    rng.shuffle(pool)
    return pool[:n_target], pool[n_target:need]


def assign_refs(target_pairs, ref_pairs, reuse, ops_per_target, rng, restarts=40):
    """Each target gets `ops_per_target` DISTINCT content-disjoint ref pairs; each ref pair used
    exactly `reuse` times. Greedy highest-remaining-capacity with restarts. (from plan_d1.py)"""
    for attempt in range(restarts):
        cap = {r: reuse for r in ref_pairs}
        order = list(range(len(target_pairs)))
        rng.shuffle(order)
        assignment, ok = {}, True
        for ti in order:
            a, b = target_pairs[ti]
            blocked = {a, b}
            cands = [r for r in ref_pairs
                     if cap[r] > 0 and r[0] not in blocked and r[1] not in blocked]
            if len(cands) < ops_per_target:
                ok = False
                break
            rng.shuffle(cands)
            cands.sort(key=lambda r: cap[r], reverse=True)
            chosen = cands[:ops_per_target]
            for r in chosen:
                cap[r] -= 1
            assignment[ti] = chosen
        if ok and all(v == 0 for v in cap.values()):
            return assignment, attempt
    raise RuntimeError(f"ref assignment failed after {restarts} restarts")


def allocate_shaders(n_target, ops_per_target, avail, rng):
    """Per target: `ops_per_target` DISTINCT shaders, globally frequency-balanced."""
    used = {s: 0 for s in avail}
    per_target = []
    for _ in range(n_target):
        pool = sorted(avail, key=lambda s: (used[s], rng.random()))
        picked = pool[:ops_per_target]
        for s in picked:
            used[s] += 1
        rng.shuffle(picked)
        per_target.append(picked)
    return per_target, used


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    d2 = cfg["d2"]
    seed = cfg["runtime"]["seed"]
    rng = random.Random(f"{seed}-d2full-plan")

    pol = json.loads((REPO_ROOT / cfg["inputs"]["policy"]).read_text())
    keep_shaders = sorted(pol["keep_shaders"])
    blacklist = sorted(pol["blacklist"])
    keep_easings = sorted(pol["keep_easings"])
    drop_easings = sorted(pol["drop_easings"])
    holdout = sorted(json.loads((REPO_ROOT / cfg["inputs"]["d1_audit"]).read_text())
                     ["stage1_plan"]["holdout_shader_list"])

    # ---- policy seatbelts (fail loudly rather than silently shipping the wrong bank) ----
    assert len(keep_shaders) == pol["n_keep"] == 72, len(keep_shaders)
    assert len(blacklist) == pol["n_blacklist"] == 40, len(blacklist)
    assert not (set(keep_shaders) & set(blacklist)), "keep/blacklist overlap"
    assert not (set(keep_shaders) & set(holdout)), "a holdout shader is in keep_shaders"
    assert not (set(blacklist) & set(holdout)), "a holdout shader is blacklisted"
    assert len(keep_easings) == 7 and len(drop_easings) == 3
    assert sorted(keep_easings + drop_easings) == sorted(
        set(ENGINE_EASINGS) - {"snap_early", "snap_late"}), "easing partition mismatch"
    bank_dir = Path(cfg["model"]["shader_bank"])
    missing = [s for s in keep_shaders if not (bank_dir / f"{s}.glsl").exists()]
    assert not missing, f"keep_shaders missing from the bank dir: {missing}"
    print(f"[plan] policy OK: keep={len(keep_shaders)} blacklist={len(blacklist)} "
          f"easings={keep_easings} holdout={len(holdout)}")

    bank_json = json.loads((REPO_ROOT / cfg["inputs"]["bank_tightened"]).read_text())
    clip_ids = [c["clip_id"] for c in bank_json["clips"]]
    mp4_of = {c["clip_id"]: c["mp4"] for c in bank_json["clips"]}
    print(f"[plan] tightened bank: {len(clip_ids)} clips")

    target_pairs, ref_pairs = build_pairs(clip_ids, d2["n_target_pairs"], d2["n_ref_pairs"], rng)
    assignment, attempts = assign_refs(target_pairs, ref_pairs, d2["ref_reuse"],
                                       d2["ops_per_target"], rng)
    print(f"[plan] target_pairs={len(target_pairs)} ref_pairs={len(ref_pairs)} "
          f"(assignment restart {attempts}, each ref used exactly {d2['ref_reuse']}x)")

    per_target_shaders, shader_used = allocate_shaders(
        d2["n_target_pairs"], d2["ops_per_target"], keep_shaders, rng)

    targets = []
    tid = 0
    distinct_hist: Counter = Counter()
    ops_hist: Counter = Counter()
    clip_use_t = Counter()
    ref_use = Counter()
    for ti in range(d2["n_target_pairs"]):
        tp, refs, shs = target_pairs[ti], assignment[ti], per_target_shaders[ti]
        assert len(refs) == len(shs) == d2["ops_per_target"]
        assert len(set(refs)) == d2["ops_per_target"], "duplicate ref pair on one target"
        clip_use_t[tp[0]] += 1
        clip_use_t[tp[1]] += 1
        slots = []
        for k in range(d2["ops_per_target"]):
            rp = refs[k]
            assert not ({tp[0], tp[1]} & {rp[0], rp[1]}), "ref/target share a clip"
            ref_use[rp] += 1
            slots.append({"tuple_id": tid, "slot": k, "shader": shs[k],
                          "ref_pair": {"A": rp[0], "B": rp[1],
                                       "A_mp4": mp4_of[rp[0]], "B_mp4": mp4_of[rp[1]]}})
            tid += 1
        assert len(set(shs)) >= d2["min_distinct_shaders"], f"target {ti}: {len(set(shs))} distinct"
        distinct_hist[len(set(shs))] += 1
        ops_hist[len(slots)] += 1
        targets.append({"target_index": ti,
                        "target_pair": {"A": tp[0], "B": tp[1],
                                        "A_mp4": mp4_of[tp[0]], "B_mp4": mp4_of[tp[1]]},
                        "slots": slots})

    assert tid == d2["n_target_pairs"] * d2["ops_per_target"] == 3072, tid
    assert set(ref_use.values()) == {d2["ref_reuse"]}, "ref reuse not uniform"
    print(f"[plan] {tid} slots | ops/target hist={dict(ops_hist)} "
          f"distinct-shader hist={dict(distinct_hist)}")
    print(f"[plan] per-shader planned: min={min(shader_used.values())} "
          f"max={max(shader_used.values())} over {len(shader_used)} shaders")

    plan = {
        "seed": seed, "spec": d2,
        "keep_shaders": keep_shaders, "blacklist": blacklist,
        "keep_easings": keep_easings, "drop_easings": drop_easings,
        "holdout_shaders": holdout,
        "gate": cfg["gate"], "n_tuples": tid, "targets": targets,
    }
    PLAN.write_text(json.dumps(plan, indent=2))
    print(f"[plan] wrote {PLAN} ({PLAN.stat().st_size/1e6:.1f} MB)")

    merge_audit({
        "spec": ("D2-FINAL: real streams (extension=none), ZERO aux maps, 7 kept easings, "
                 "72 keep_shaders (blacklist applied at SAMPLING time), frozen gate "
                 "tau=0.2543 / max_pure<=0.5 / seam<=2.0 / dq<=0.5, per-slot rejection sampling"),
        "policy_frozen": {
            "tau": cfg["gate"]["tau"], "assert1_tol": cfg["gate"]["assert1_tol"],
            "seam_max": cfg["gate"]["seam_max"], "m2_max_dq": cfg["gate"]["m2_max_dq"],
            "m1_min_flag_threshold": cfg["gate"]["m1_min_flag_threshold"],
            "floor_leg": "ABSENT by decision (non-gating metadata flag m1_min < -0.05 instead)",
            "keep_easings": keep_easings, "drop_easings": drop_easings,
            "n_keep_shaders": len(keep_shaders), "n_blacklist_shaders": len(blacklist),
            "blacklist_applied_at": "sampling time (blacklisted shaders are never drawn)",
            "holdout_shaders": holdout, "aux_kind": None, "extension": "none",
            "timing": "window [8,112]; onset=8+u1*0.20*104, release=112-u2*0.20*104, u1,u2~U[0,1] indep",
        },
        "stage1_plan": {
            "tightened_bank_count": len(clip_ids),
            "n_target_pairs": d2["n_target_pairs"], "ops_per_target": d2["ops_per_target"],
            "n_ref_pairs": d2["n_ref_pairs"], "ref_reuse": d2["ref_reuse"],
            "ref_reuse_uniform": True,
            "n_tuples_planned": tid,
            "ops_per_target_pair_hist_planned": dict(ops_hist),
            "distinct_shaders_per_target_hist_planned": dict(distinct_hist),
            "per_shader_allocation_planned": dict(sorted(shader_used.items())),
            "per_shader_planned_min_max": [min(shader_used.values()), max(shader_used.values())],
            "clip_usage_as_target_min_max_mean": [
                min(clip_use_t.get(c, 0) for c in clip_ids),
                max(clip_use_t.get(c, 0) for c in clip_ids),
                round(sum(clip_use_t.values()) / len(clip_ids), 2)],
            "ref_assignment_restarts": attempts,
        },
    })
    print("[plan] D2_BUILD_AUDIT.json updated (spec, policy_frozen, stage1_plan)")


if __name__ == "__main__":
    main()
