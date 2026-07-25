#!/usr/bin/env python
"""Merge the visual-review verdicts into the {clip_id: accept|reason} map `tighten_v2 --stage
finalize` consumes, and check that every candidate was actually looked at.

The owner's condition on expansion was that every candidate is "seen one by one visually". 139
candidates were reviewed: sheets 00-01 by me directly, sheets 02-11 by four parallel reviewers
working from an identical written rubric. This merges those files, asserts full coverage with no
duplicates, and cross-checks the verdicts against the independently-measured motion statistic so
that a "static" call is backed by a number rather than an impression.
"""

import json
import os
import sys

REPO = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research"
V2 = os.path.join(REPO, "data/processed/ctt_v2_strata/endpoints_v2")
WORK = os.path.join(V2, "_work")
VD = os.path.join(WORK, "verdicts")


def main() -> None:
    q = json.load(open(os.path.join(WORK, "REVIEW_QUEUE.json")))
    expected = list(q["review_queue"])

    files = [os.path.join(WORK, "verdicts_operator_sheets00_01.json")]
    files += [os.path.join(VD, f) for f in sorted(os.listdir(VD)) if f.endswith(".json")]

    verdicts: dict = {}
    provenance: dict = {}
    dupes: list = []
    for f in files:
        who = "operator" if "operator" in os.path.basename(f) else os.path.basename(f)[:-5]
        d = json.load(open(f))
        for k, rows in d.items():
            if not k.startswith("sheet"):
                continue
            for r in rows:
                cid = r["clip_id"]
                if cid in verdicts:
                    dupes.append(cid)
                    continue
                verdicts[cid] = ("accept" if r["verdict"] == "accept"
                                 else r.get("reason", "reject"))
                provenance[cid] = who

    # Labels were transcribed by eye off a rendered contact sheet, so a digit can be misread.
    # Repair ONLY where the intended id is unambiguous: a single queue entry within edit
    # distance 2 that nothing else has claimed. Anything ambiguous stays missing and blocks.
    def lev(a, b):
        if abs(len(a) - len(b)) > 2:
            return 99
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    repairs = {}
    for cid in [c for c in verdicts if c not in set(expected)]:
        cands = [e for e in expected if e not in verdicts and lev(cid, e) <= 2]
        if len(cands) == 1:
            repairs[cid] = cands[0]
    for wrong, right in repairs.items():
        verdicts[right] = verdicts.pop(wrong)
        provenance[right] = provenance.pop(wrong)
        print(f"[agg] repaired transcription slip: {wrong!r} -> {right!r}")

    missing = [c for c in expected if c not in verdicts]
    extra = [c for c in verdicts if c not in expected]
    print(f"[agg] reviewed {len(verdicts)} of {len(expected)} candidates | "
          f"missing {len(missing)} | not-in-queue {len(extra)} | duplicate rows {len(dupes)}")
    if extra:
        print(f"[agg] clip_ids not in the queue (likely transcription slips): {extra[:10]}")
    if missing:
        print(f"[agg] MISSING (never reviewed): {missing[:10]}")

    # drop any id that is not in the queue, then re-check coverage
    verdicts = {k: v for k, v in verdicts.items() if k in set(expected)}
    still_missing = [c for c in expected if c not in verdicts]
    if still_missing:
        json.dump(still_missing, open(os.path.join(WORK, "UNREVIEWED.json"), "w"), indent=1)
        sys.exit(f"[agg] {len(still_missing)} candidates have no verdict — wrote UNREVIEWED.json. "
                 f"Every candidate must be seen before finalize.")

    # cross-check "static" calls against the measured motion statistic
    motion = json.load(open(os.path.join(WORK, "MOTION.json")))
    floor = json.load(open(os.path.join(WORK, "MOTION_FLOOR.json")))
    stat = [(c, motion.get(c)) for c, v in verdicts.items() if v == "static"]
    print(f"[agg] motion cross-check — clips called 'static' and their measured motion "
          f"(pool p05 reference {floor.get('floor')}):")
    for c, m in sorted(stat, key=lambda t: (t[1] is None, t[1])):
        print(f"      {c:<46} {m}")

    acc = [c for c, v in verdicts.items() if v == "accept"]
    reasons: dict = {}
    for c, v in verdicts.items():
        if v != "accept":
            reasons[v] = reasons.get(v, 0) + 1
    print(f"[agg] ACCEPTED {len(acc)} / {len(verdicts)}  ({len(acc)/len(verdicts):.0%})")
    print(f"[agg] reject reasons: {json.dumps(dict(sorted(reasons.items())))}")

    out = os.path.join(WORK, "VISUAL_VERDICTS.json")
    json.dump(verdicts, open(out, "w"), indent=1)
    json.dump({"provenance": provenance,
               "note": "sheets 00-01 reviewed by the operator directly; 02-11 by four parallel "
                       "reviewers under an identical written rubric",
               "n_reviewed": len(verdicts), "n_accepted": len(acc),
               "reject_reasons": reasons},
              open(os.path.join(WORK, "VISUAL_REVIEW_PROVENANCE.json"), "w"), indent=1)
    print(f"[agg] -> {out}")


if __name__ == "__main__":
    main()
