"""exp_077 D1 STAGE 1 (planning) — build the full 3,072-tuple plan (fable round-3 locked spec).

Deterministic, CPU. Decides everything EXCEPT the continuous operator params (those are
rejection-sampled per tuple at render time against the real endpoint frames). Writes
`d1_plan.json` (3072 fully-addressed tuple slots) + the held-out-shader list + the planned
per-shader / per-aux-family allocation into `D1_BUILD_AUDIT.json`.

LOCKED SPEC
-----------
* ENDPOINTS: 384 distinct TARGET pairs + 768 distinct REF pairs, cross-clip A/B (start9 from
  clip A, end9 from clip B) drawn from the tightened bank. A tuple's ref pair and target pair
  are CONTENT-DISJOINT (share no clip). Each ref pair reused exactly 4x (768*4 = 3072).
* OPERATORS: 3,072 unique draws. HOLD OUT 8 shaders entirely. 50% carry an aux-map, stratified
  over 7 map families x {luma, displacement}. Per-operator endpoint-identity rejection sampling
  happens at render time (MANDATORY) — the plan only fixes shader + aux family.
* TUPLES: each TARGET pair under exactly 8 operators = 4 aux + 4 non-aux (mixed, 50% aux),
  spanning >= 6 distinct shaders (2 aux shaders + 4 distinct non-aux = 6, guaranteed by build).

NOTE ON THE AUX FRACTION (flagged for the advisor): the gl-transitions engine has exactly TWO
aux-capable shaders (luma, displacement). Honoring the explicit, bolded "50% carry an aux-map"
therefore forces luma+displacement to ~768 draws each — which is irreconcilable with the
parenthetical "~25 each over 122 shaders" (that would put aux at ~1.6%). This build honors the
50%-aux design (the medium-bearing families are the token-branch's key citizens, and the
effective-K audit showed aux families carry the incremental appearance diversity). The TRUE
per-shader allocation is reported in the audit so the tension is fully transparent.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

from engine import maps, operators, shaders  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402

CONFIG_PATH = HERE / "config_d1.yaml"
AUDIT = HERE / "D1_BUILD_AUDIT.json"
PLAN = HERE / "d1_plan.json"


def merge_audit(update: dict) -> None:
    cur = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
    cur.update(update)
    AUDIT.write_text(json.dumps(cur, indent=2))


def build_pairs(clip_ids, n_target, n_ref, rng):
    """Sample n_target + n_ref DISTINCT cross-clip ordered pairs (A != B), pools disjoint."""
    seen = set()
    pool = []
    # cap attempts generously; the space is |clips|*(|clips|-1) >> needed
    need = n_target + n_ref
    tries = 0
    while len(pool) < need and tries < need * 100:
        a, b = rng.sample(clip_ids, 2)
        key = (a, b)
        if key not in seen:
            seen.add(key)
            pool.append(key)
        tries += 1
    if len(pool) < need:
        raise RuntimeError(f"could not sample {need} distinct pairs (got {len(pool)})")
    rng.shuffle(pool)
    return pool[:n_target], pool[n_target:n_target + n_ref]


def assign_refs(target_pairs, ref_pairs, reuse, ops_per_target, rng, restarts=40):
    """Assign `ops_per_target` ref pairs to each target: content-disjoint, distinct per target,
    each ref pair used exactly `reuse` times. Greedy highest-remaining-capacity with restarts."""
    for attempt in range(restarts):
        cap = {r: reuse for r in ref_pairs}
        order = list(range(len(target_pairs)))
        rng.shuffle(order)
        assignment = {}
        ok = True
        for ti in order:
            a, b = target_pairs[ti]
            blocked = {a, b}
            cands = [r for r in ref_pairs
                     if cap[r] > 0 and r[0] not in blocked and r[1] not in blocked]
            if len(cands) < ops_per_target:
                ok = False
                break
            # prefer highest remaining capacity (spreads usage, avoids end-game starvation)
            rng.shuffle(cands)
            cands.sort(key=lambda r: cap[r], reverse=True)
            chosen = cands[:ops_per_target]
            for r in chosen:
                cap[r] -= 1
            assignment[ti] = chosen
        if ok and all(v == 0 for v in cap.values()):
            return assignment, attempt
    raise RuntimeError(f"ref assignment failed after {restarts} restarts")


def allocate_operators(n_target, aux_per_target, aux_shaders, non_aux_avail, rng):
    """Per target: aux_per_target aux slots (balanced over aux_shaders x 7 families) +
    (8-aux_per_target) DISTINCT non-aux shaders (frequency-balanced). Returns per-target
    lists of dicts {shader, aux_kind|None}."""
    fams = list(maps.MAP_KINDS)  # 7 families
    non_aux_per_target = 8 - aux_per_target

    # --- aux: build a globally family-balanced pool per aux shader, split evenly per target ---
    # each target gets aux_per_target/len(aux_shaders) slots per aux shader (2 luma + 2 disp)
    per_shader = aux_per_target // len(aux_shaders)
    assert per_shader * len(aux_shaders) == aux_per_target, "aux_per_target must split evenly over aux shaders"
    fam_pool = {s: [] for s in aux_shaders}
    for s in aux_shaders:
        total = n_target * per_shader
        # round-robin families -> balanced counts
        seq = [fams[i % len(fams)] for i in range(total)]
        rng.shuffle(seq)
        fam_pool[s] = seq

    # --- non-aux: frequency-balanced distinct shaders per target ---
    used = {s: 0 for s in non_aux_avail}
    per_target = []
    for t in range(n_target):
        slots = []
        for s in aux_shaders:
            for _ in range(per_shader):
                slots.append({"shader": s, "aux_kind": fam_pool[s].pop()})
        # pick non_aux_per_target least-used DISTINCT non-aux shaders
        pool = sorted(non_aux_avail, key=lambda s: (used[s], rng.random()))
        picked = pool[:non_aux_per_target]
        for s in picked:
            used[s] += 1
            slots.append({"shader": s, "aux_kind": None})
        rng.shuffle(slots)
        per_target.append(slots)
    return per_target, used


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    d1 = cfg["d1"]
    seed = cfg["runtime"]["seed"]
    rng = random.Random(seed)

    # ---- tightened bank ----
    bank_json = json.loads((REPO_ROOT / cfg["inputs"]["bank_tightened"]).read_text())
    clip_ids = [c["clip_id"] for c in bank_json["clips"]]
    mp4_of = {c["clip_id"]: c["mp4"] for c in bank_json["clips"]}
    print(f"[plan] tightened bank: {len(clip_ids)} clips")

    # ---- usable shader bank (same GL gate the render uses -> held-out are guaranteed usable) ----
    bank_dir = Path(cfg["model"]["shader_bank"])
    bank = shaders.load_bank(bank_dir)
    runner = GLRunner(cfg["inference"]["width"], cfg["inference"]["height"])
    print(f"[plan] GL: {runner.renderer_name()} | parsed {len(bank)} shaders")
    usable, _ = operators.validate_bank(runner, bank, tol=cfg["sampling"]["endpoint_tol"])
    usable_names = sorted(usable)
    aux_shaders = sorted(s for s in usable_names if s in shaders.AUX_SAMPLER_SHADERS)
    non_aux_usable = sorted(s for s in usable_names if s not in shaders.AUX_SAMPLER_SHADERS)
    print(f"[plan] usable={len(usable_names)}  aux={aux_shaders}  non_aux={len(non_aux_usable)}")
    if sorted(aux_shaders) != ["displacement", "luma"]:
        raise RuntimeError(f"expected aux shaders [displacement, luma], got {aux_shaders}")

    # ---- hold out 8 non-aux shaders entirely ----
    holdout = sorted(rng.sample(non_aux_usable, d1["n_holdout_shaders"]))
    non_aux_avail = sorted(s for s in non_aux_usable if s not in holdout)
    print(f"[plan] held out {len(holdout)} shaders: {holdout}")
    print(f"[plan] non-aux available for draws: {len(non_aux_avail)}")

    # ---- endpoint pairs ----
    target_pairs, ref_pairs = build_pairs(clip_ids, d1["n_target_pairs"], d1["n_ref_pairs"], rng)
    print(f"[plan] target_pairs={len(target_pairs)}  ref_pairs={len(ref_pairs)}")

    # ---- ref assignment (content-disjoint, 4x reuse) ----
    assignment, attempts = assign_refs(
        target_pairs, ref_pairs, d1["ref_reuse"], d1["ops_per_target"], rng)
    print(f"[plan] ref assignment OK (restart {attempts}); each ref used exactly {d1['ref_reuse']}x")

    # ---- operator allocation ----
    per_target_ops, non_aux_used = allocate_operators(
        d1["n_target_pairs"], d1["aux_per_target"], aux_shaders, non_aux_avail, rng)

    # ---- emit tuples ----
    tuples = []
    tid = 0
    per_shader_alloc: dict[str, int] = {}
    per_family_alloc: dict[str, int] = {}          # keyed "shader/family"
    ops_per_target_hist: dict[int, int] = {}
    distinct_shaders_hist: dict[int, int] = {}
    clip_use = {c: 0 for c in clip_ids}
    for ti in range(d1["n_target_pairs"]):
        tp = target_pairs[ti]
        refs = assignment[ti]
        ops = per_target_ops[ti]
        assert len(refs) == len(ops) == d1["ops_per_target"]
        shaders_here = set()
        n_aux_here = 0
        clip_use[tp[0]] += 1
        clip_use[tp[1]] += 1
        for slot in range(d1["ops_per_target"]):
            op = ops[slot]
            rp = refs[slot]
            # content-disjoint hard check
            assert not ({tp[0], tp[1]} & {rp[0], rp[1]}), "ref/target share a clip"
            shaders_here.add(op["shader"])
            per_shader_alloc[op["shader"]] = per_shader_alloc.get(op["shader"], 0) + 1
            if op["aux_kind"]:
                n_aux_here += 1
                key = f"{op['shader']}/{op['aux_kind']}"
                per_family_alloc[key] = per_family_alloc.get(key, 0) + 1
            tuples.append({
                "tuple_id": tid,
                "target_index": ti,
                "target_pair": {"A": tp[0], "B": tp[1], "A_mp4": mp4_of[tp[0]], "B_mp4": mp4_of[tp[1]]},
                "ref_pair": {"A": rp[0], "B": rp[1], "A_mp4": mp4_of[rp[0]], "B_mp4": mp4_of[rp[1]]},
                "shader": op["shader"],
                "aux_kind": op["aux_kind"],
            })
            tid += 1
        # per-target asserts (locked spec)
        assert len(ops) == 8, f"target {ti}: not 8 operators"
        assert len(shaders_here) >= d1["min_distinct_shaders"], \
            f"target {ti}: only {len(shaders_here)} distinct shaders (< {d1['min_distinct_shaders']})"
        assert 0 < n_aux_here < 8, f"target {ti}: not mixed aux/no-aux (aux={n_aux_here})"
        ops_per_target_hist[8] = ops_per_target_hist.get(8, 0) + 1
        distinct_shaders_hist[len(shaders_here)] = distinct_shaders_hist.get(len(shaders_here), 0) + 1

    assert tid == d1["n_target_pairs"] * d1["ops_per_target"], f"expected 3072 tuples, got {tid}"
    n_aux = sum(1 for t in tuples if t["aux_kind"])
    print(f"[plan] {tid} tuples  aux={n_aux} ({100*n_aux/tid:.1f}%)  distinct-shader hist={distinct_shaders_hist}")

    plan = {
        "seed": seed,
        "spec": d1,
        "usable_shaders": usable_names,
        "aux_shaders": aux_shaders,
        "holdout_shaders": holdout,
        "non_aux_available": non_aux_avail,
        "n_tuples": tid,
        "tuples": tuples,
    }
    PLAN.write_text(json.dumps(plan, indent=2))
    print(f"[plan] wrote {PLAN}")

    merge_audit({
        "stage1_plan": {
            "tightened_bank_count": len(clip_ids),
            "n_target_pairs": d1["n_target_pairs"],
            "n_ref_pairs": d1["n_ref_pairs"],
            "ref_reuse": d1["ref_reuse"],
            "n_tuples": tid,
            "usable_shader_count": len(usable_names),
            "aux_shaders": aux_shaders,
            "holdout_shader_list": holdout,
            "n_non_aux_available": len(non_aux_avail),
            "aux_fraction_planned": round(n_aux / tid, 4),
            "per_shader_allocation_planned": dict(sorted(per_shader_alloc.items())),
            "per_aux_family_allocation_planned": dict(sorted(per_family_alloc.items())),
            "ops_per_target_pair_hist": ops_per_target_hist,
            "distinct_shaders_per_target_hist": distinct_shaders_hist,
            "clip_usage_as_target_min_max_mean": [
                min(clip_use.values()), max(clip_use.values()),
                round(sum(clip_use.values()) / len(clip_use), 2)],
            "aux_fraction_note": (
                "engine has ONLY 2 aux-capable shaders (luma, displacement); honoring the bolded "
                "'50% carry aux' forces them to ~768 draws each, irreconcilable with '~25 each over "
                "122 shaders'. 50%-aux design honored (medium-bearing families = key citizens). "
                "True per-shader allocation reported above; tension flagged."),
        }
    })
    print("[plan] audit updated")


if __name__ == "__main__":
    main()
