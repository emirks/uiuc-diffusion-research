"""S2b — gate-rejection rate BY BANK (A5 SYNTHESIS RULING 4, pre-registered >10 pp flag).

The union pool merges two provenances, and the advisor pre-registered one specific risk:
HumanVid clips are person-only and comparatively low-texture, so they could systematically
fail the M1 "mush" gate (`m1_p10 >= tau`) and quietly bias the delivered stratum back toward
synth content. The bar: **flag if the between-bank differential exceeds 10 percentage
points.** This script measures it; it does not adjudicate.

Three views, all computed from the append-only ops log (`meta/ops_shard*.jsonl`) joined to the
frozen plan's per-pair bank labels:

  1. ENDPOINT-SLOT view (the headline). Every render attempt puts two endpoint clips on screen.
     An attempt contributes one slot to each of its two endpoints' banks, and a rejected
     attempt contributes a rejected slot to both. rate(bank) = rejected slots / total slots.
     This is the measure that answers "do person clips fail the gates more often".
  2. PAIR-TYPE view. Rejection rate per {synth_synth, cross, humanvid_humanvid}. Cleaner
     causally (no shared attribution), and it is the bucket the 25/50/25 quota controls.
  3. PER-GATE breakdown of both views (gate2 / assert1 / assert2 / m1 / m2), because the
     pre-registered concern names M1 specifically.

Also reports the REALISED per-op bank mix from the accepted manifest (the planner asserts no
bank-pure op on the PRIMARY 10 pairs; the renderer may substitute spares, so the delivered
blocks are re-checked here).

    python bank_rejection_audit.py            # -> BANK_REJECTION_AUDIT.json
"""

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROOT = REPO_ROOT / "outputs/videos/ctt_v2_s2_humanvid/full"
DIFFERENTIAL_FLAG_PP = 10.0        # pre-registered, A5 SYNTHESIS RULING 4
GATES = ("gate2", "assert1", "assert2", "m1", "m2")


def rate(rej: int, tot: int) -> float | None:
    return round(100.0 * rej / tot, 3) if tot else None


def main() -> None:
    plan = json.loads((HERE / "PLAN_S2_UNION.json").read_text())
    pair_bank = {p["pair_id"]: (p["A_bank"], p["B_bank"]) for p in plan["pairs"]}
    pair_type = {p["pair_id"]: p["bank_type"] for p in plan["pairs"]}

    clips, ops = [], []
    for f in sorted(glob.glob(str(ROOT / "meta/clips_shard*.jsonl"))):
        clips += [json.loads(l) for l in open(f) if l.strip()]
    for f in sorted(glob.glob(str(ROOT / "meta/ops_shard*.jsonl"))):
        ops += [json.loads(l) for l in open(f) if l.strip()]

    # ---- tallies ------------------------------------------------------------------------
    # slot[bank] / slot_rej[bank]; type[t] / type_rej[t]; and the same split per failed gate.
    slot = collections.Counter()
    slot_rej = collections.Counter()
    slot_gate = collections.defaultdict(collections.Counter)      # gate -> bank -> n
    tslot = collections.Counter()
    tslot_rej = collections.Counter()
    tslot_gate = collections.defaultdict(collections.Counter)     # gate -> type -> n

    def add(pid: int, rejected_gates: list[str] | None) -> None:
        ba, bb = pair_bank[pid]
        t = pair_type[pid]
        for b in (ba, bb):
            slot[b] += 1
        tslot[t] += 1
        if rejected_gates is None:
            return
        for b in (ba, bb):
            slot_rej[b] += 1
            for g in rejected_gates:
                slot_gate[g][b] += 1
        tslot_rej[t] += 1
        for g in rejected_gates:
            tslot_gate[g][t] += 1

    for c in clips:                                   # accepted attempts
        add(c["pair_id"], None)
    for o in ops:                                     # rejected attempts
        for r in o["rejects"]:
            g = ["gate2"] if r["stage"] == "gate2" else list(r.get("failed", []))
            add(r["pair_id"], g or ["unknown"])

    banks = ["synth", "humanvid"]
    by_bank = {b: {"attempt_slots": slot[b], "rejected_slots": slot_rej[b],
                   "rejection_rate_pct": rate(slot_rej[b], slot[b]),
                   "per_gate_rate_pct": {g: rate(slot_gate[g][b], slot[b]) for g in GATES}}
               for b in banks}
    types = ["synth_synth", "cross", "humanvid_humanvid"]
    by_type = {t: {"attempts": tslot[t], "rejected": tslot_rej[t],
                   "rejection_rate_pct": rate(tslot_rej[t], tslot[t]),
                   "per_gate_rate_pct": {g: rate(tslot_gate[g][t], tslot[t]) for g in GATES}}
               for t in types}

    diffs = {}
    rs, rh = by_bank["synth"]["rejection_rate_pct"], by_bank["humanvid"]["rejection_rate_pct"]
    diffs["overall_pp"] = round(abs(rh - rs), 3) if None not in (rs, rh) else None
    for g in GATES:
        a, b = by_bank["synth"]["per_gate_rate_pct"][g], by_bank["humanvid"]["per_gate_rate_pct"][g]
        diffs[f"{g}_pp"] = round(abs(b - a), 3) if None not in (a, b) else None
    flagged = sorted(k for k, v in diffs.items() if v is not None and v > DIFFERENTIAL_FLAG_PP)

    # ---- realised per-op bank mix (the no-bank-pure invariant, post-render) ---------------
    by_op = collections.defaultdict(list)
    for c in clips:
        by_op[c["op_index"]].append(c)
    mixes, pure = {}, []
    for oi, rows in by_op.items():
        cnt = collections.Counter()
        for r in rows:
            ba, bb = pair_bank[r["pair_id"]]
            cnt[ba] += 1
            cnt[bb] += 1
        mixes[oi] = dict(cnt)
        if len(cnt) == 1:
            pure.append(oi)
    minority = sorted(min(m.values()) if len(m) > 1 else 0 for m in mixes.values())
    delivered_type = collections.Counter(pair_type[c["pair_id"]] for c in clips)
    n = len(clips) or 1

    out = {
        "stratum": "S2b",
        "authority": "A5 SYNTHESIS RULING 4 — gate-rejection-by-bank differential, "
                     f"pre-registered flag at >{DIFFERENTIAL_FLAG_PP} pp",
        "n_clips_accepted": len(clips),
        "n_attempts": sum(tslot.values()),
        "by_endpoint_bank": by_bank,
        "by_pair_type": by_type,
        "differential_pp": diffs,
        "differential_flag_bar_pp": DIFFERENTIAL_FLAG_PP,
        "flagged": flagged,
        "verdict": "NO FLAG" if not flagged else f"FLAGGED: {flagged}",
        "delivered_clip_bank_mix": {
            t: {"clips": delivered_type[t], "pct": round(100.0 * delivered_type[t] / n, 2)}
            for t in types},
        "delivered_endpoint_slots_by_bank": {
            b: sum(1 for c in clips for x in pair_bank[c["pair_id"]] if x == b) for b in banks},
        "realised_per_op_bank_mix": {
            "n_ops": len(mixes),
            "bank_pure_ops": pure,
            "minority_bank_endpoints_min": minority[0] if minority else None,
            "minority_bank_endpoints_median": minority[len(minority) // 2] if minority else None,
            "minority_bank_endpoints_max": minority[-1] if minority else None,
            "histogram": dict(sorted(collections.Counter(minority).items()))},
    }
    (HERE / "BANK_REJECTION_AUDIT.json").write_text(json.dumps(out, indent=1))

    print(f"attempts {out['n_attempts']} | accepted clips {len(clips)}")
    print("-- endpoint-slot view (headline) --")
    for b in banks:
        v = by_bank[b]
        print(f"  {b:<9} slots {v['attempt_slots']:>6} | rejected {v['rejected_slots']:>5} | "
              f"rate {v['rejection_rate_pct']}% | " +
              " ".join(f"{g}={v['per_gate_rate_pct'][g]}" for g in GATES))
    print("-- pair-type view --")
    for t in types:
        v = by_type[t]
        print(f"  {t:<18} attempts {v['attempts']:>6} | rejected {v['rejected']:>5} | "
              f"rate {v['rejection_rate_pct']}% | " +
              " ".join(f"{g}={v['per_gate_rate_pct'][g]}" for g in GATES))
    print(f"differential (|humanvid - synth|, pp): {diffs}")
    print(f"bar {DIFFERENTIAL_FLAG_PP} pp -> {out['verdict']}")
    print(f"delivered clip mix: " + " ".join(
        f"{t} {out['delivered_clip_bank_mix'][t]['pct']}%" for t in types))
    print(f"realised per-op bank mix: bank-pure ops {len(pure)} | minority endpoints "
          f"min/med/max {out['realised_per_op_bank_mix']['minority_bank_endpoints_min']}/"
          f"{out['realised_per_op_bank_mix']['minority_bank_endpoints_median']}/"
          f"{out['realised_per_op_bank_mix']['minority_bank_endpoints_max']}")


if __name__ == "__main__":
    main()
