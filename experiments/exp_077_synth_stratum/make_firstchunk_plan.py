"""exp_077 D2-FULL — the FIRST-CHUNK AUDIT plan (pre-committed clamp check, 2026-07-24 ruling).

64 clips (4 target pairs x 8 slots x {target, reference}) rendered through the IDENTICAL code path
and the IDENTICAL frozen gate as the mass build, but with the three known SURVIVING offenders
(EdgeTransition, fadecolor, ColourDistance) deliberately OVERSAMPLED to 12/32 slots (37.5%, vs
~4% under uniform allocation) so the clamp is stress-tested where it matters.

Renders into its own tree (OUTSUB=d2full_firstchunk) — it cannot contaminate the dataset, and the
mass build's uniform 72-shader allocation is NOT reweighted.

    python make_firstchunk_plan.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFENDERS = ["EdgeTransition", "fadecolor", "ColourDistance"]


def main() -> None:
    plan = json.loads((HERE / "d2full_plan.json").read_text())
    # borrow 4 real target pairs (with their real, content-disjoint ref pairs) from the mass plan
    targets = [json.loads(json.dumps(t)) for t in plan["targets"][:4]]
    # targets 0-1: heavy offender load (2x each offender + 2 as planned) ; targets 2-3: as planned
    over = OFFENDERS * 2
    for t in targets[:2]:
        for k, slot in enumerate(t["slots"]):
            if k < len(over):
                slot["shader"] = over[k]
    n_off = sum(1 for t in targets for s in t["slots"] if s["shader"] in OFFENDERS)
    for i, t in enumerate(targets):
        t["target_index"] = i
        for k, s in enumerate(t["slots"]):
            s["tuple_id"] = i * 8 + k
            s["slot"] = k
    out = dict(plan)
    out["targets"] = targets
    out["n_tuples"] = 32
    out["firstchunk"] = {"purpose": "pre-committed clamp check (baseline visual bad rate 17.5%)",
                         "oversampled_shaders": OFFENDERS,
                         "offender_slots": n_off, "total_slots": 32,
                         "offender_share": round(n_off / 32, 4)}
    path = HERE / "d2full_firstchunk_plan.json"
    path.write_text(json.dumps(out, indent=2))
    shs = [s["shader"] for t in targets for s in t["slots"]]
    print(f"[firstchunk] {len(shs)} slots -> {path.name}")
    print(f"[firstchunk] offenders {n_off}/32 = {100*n_off/32:.1f}%  shaders={sorted(set(shs))}")


if __name__ == "__main__":
    main()
