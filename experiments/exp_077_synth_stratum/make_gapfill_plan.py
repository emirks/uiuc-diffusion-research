"""exp_077 D2-FULL — GAP-FILL plan for the slots whose retry ladder EXHAUSTED.

WHY (measured, 2026-07-24): 4 of 3,072 slots exhausted all 25 attempts (5 shaders x 5
param/timing redraws). The blocker is NOT the shader or the parameters — it is the SLOT'S
REFERENCE PAIR, which the plan fixes and the ladder never redraws:

  * 3 of the 4 share ONE reference pair whose clip has a PERFECTLY STATIC pure phase. Assert2 is
    a RATIO (handoff step / that bucket's own mean frame delta) and the denominator collapses to
    its 1e-3 floor, so the seam ratio runs 1.8-1560 against a <= 2.0 gate, and the degenerate q(t)
    also pins m2 at ~0.64 against a <= 0.5 gate. Unpassable for any shader.
  * the 4th sits just over the seam threshold (median 2.47) on a fast-motion reference.

The LOCKED spec says "384 target pairs x **exactly 8** operators" but only "768 ref pairs
content-disjoint + reused **~4x**". This plan therefore spends the slack the spec marks with the
tilde to protect the invariant it states exactly: it re-runs ONLY the exhausted slots, with a
SUBSTITUTED reference pair, and changes nothing else — same frozen gate, same tau, same 72-shader
bank, same easings, same ladder, same timing law.

The substitute is drawn from the plan's own reference-pair pool under the plan's own constraints:
content-disjoint from the target pair, not already used by that target, and preferring the
LEAST-used pairs so the reuse distribution stays as close to 4x as possible.

    python make_gapfill_plan.py        # -> d2full_gapfill_plan.json + the sbatch line
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = REPO_ROOT / "outputs/videos/exp_077_synth_stratum_d2/d2full"


def main() -> None:
    plan = json.loads((HERE / "d2full_plan.json").read_text())
    accepted: dict[int, dict] = {}
    for f in sorted((RUN / "meta").glob("tuples_shard*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                accepted[t["tuple_id"]] = t

    # reference-pair usage over the DELIVERED tuples (so the substitute keeps reuse near 4x)
    use = Counter((t["ref_pair"]["A"], t["ref_pair"]["B"]) for t in accepted.values())
    pool = {(s["ref_pair"]["A"], s["ref_pair"]["B"]): s["ref_pair"]
            for t in plan["targets"] for s in t["slots"]}

    # BAN every reference pair that already proved unpassable (it exhausted some slot's ladder).
    # Without this the "least-used first" rule hands the pathological pair straight back out,
    # because every slot it broke left it under-used.
    banned = {(s["ref_pair"]["A"], s["ref_pair"]["B"]) for t in plan["targets"] for s in t["slots"]
              if s["tuple_id"] not in accepted}

    targets = []
    rows = []
    for tgt in plan["targets"]:
        missing = [s for s in tgt["slots"] if s["tuple_id"] not in accepted]
        if not missing:
            continue
        tp = tgt["target_pair"]
        used_by_target = {(s["ref_pair"]["A"], s["ref_pair"]["B"]) for s in tgt["slots"]}
        # shaders this target pair already delivers -> the swap ladder must not duplicate them
        exclude = sorted({accepted[s["tuple_id"]]["shader"] for s in tgt["slots"]
                          if s["tuple_id"] in accepted})
        rng = random.Random(f"{plan['seed']}-gapfill-{tgt['target_index']}")
        slots = []
        for s in missing:
            bad = (s["ref_pair"]["A"], s["ref_pair"]["B"])
            cand = [k for k in pool
                    if k not in used_by_target                      # distinct within the target
                    and k not in banned                             # never proved unpassable
                    and not ({tp["A"], tp["B"]} & set(k))]          # content-disjoint
            assert cand, f"no content-disjoint substitute for target {tgt['target_index']}"
            lo = min(use[k] for k in cand)
            cand = sorted(k for k in cand if use[k] == lo)          # least-used first
            new = cand[rng.randrange(len(cand))]
            use[new] += 1
            used_by_target.add(new)
            slots.append({"tuple_id": s["tuple_id"], "slot": s["slot"], "shader": s["shader"],
                          "ref_pair": pool[new]})
            rows.append({"tuple_id": s["tuple_id"], "target_index": tgt["target_index"],
                         "slot": s["slot"], "planned_shader": s["shader"],
                         "ref_pair_exhausted": list(bad), "ref_pair_substituted": list(new),
                         "substitute_prior_uses": lo,
                         "target_shaders_excluded_from_the_ladder": exclude})
        targets.append({"target_index": tgt["target_index"], "target_pair": tp,
                        "exclude_shaders": exclude, "slots": slots})

    out = dict(plan)
    out["targets"] = targets
    out["n_tuples"] = sum(len(t["slots"]) for t in targets)
    out["spec"] = dict(plan["spec"], gapfill=True,
                       note=("GAP-FILL: only the slots whose ladder exhausted, with a substituted "
                             "content-disjoint reference pair. Gate/tau/bank/easings/ladder/timing "
                             "law all unchanged."))
    (HERE / "d2full_gapfill_plan.json").write_text(json.dumps(out, indent=1))
    (HERE / "D2_GAPFILL.json").write_text(json.dumps({
        "why": ("4 of 3,072 slots exhausted the 25-attempt ladder. The blocker is the slot's "
                "REFERENCE PAIR, which the ladder never redraws: 3 share one reference whose clip "
                "has a perfectly static pure phase, collapsing the assert2 ratio denominator to "
                "its 1e-3 floor (seam 1.8-1560 vs a <=2.0 gate) and pinning m2 at ~0.64 vs <=0.5."),
        "action": ("re-run ONLY those slots with a substituted content-disjoint reference pair, "
                   "drawn least-used-first from the plan's own pool. Spends the 'reused ~4x' slack "
                   "to protect the exact invariant '384 target pairs x exactly 8 operators'. "
                   "Frozen gate, tau, 72-shader bank, easings, retry ladder and timing law are "
                   "UNCHANGED; no recalibration."),
        "n_slots": len(rows), "slots": rows}, indent=1))
    print(json.dumps(rows, indent=1))
    print(f"\n[gapfill] {len(rows)} slots over {len(targets)} target pairs "
          f"-> d2full_gapfill_plan.json")
    print("  sbatch --export=ALL,SHARD=0,NSHARDS=1,SHARD_TAG=GF,"
          "PLANFILE=d2full_gapfill_plan.json job_render_d2full.sbatch")


if __name__ == "__main__":
    main()
