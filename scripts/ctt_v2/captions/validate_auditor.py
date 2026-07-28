#!/usr/bin/env python
"""Validation-before-use for the Layer-2 auditor (advisor A13 step 3).

A8 confirmed `gemini-3.5-flash` as auditor on a specific measurement -- the wrong-clip
mismatch control returning 100.0% on 391 pairings.  A13: "that validation belongs to the
MODEL, not the PROCEDURE", so a new auditor must earn the same certificate before it may
gate anything.  Two lanes, both pre-committed:

  mismatch     n >= 200 wrong-clip pairings; bar `inaccurate=YES` >= 95%.
               A description is audited against a DIFFERENT clip's video.  An auditor that
               cannot see the mismatch cannot see a real inaccuracy either.
  calibration  re-audit 150-200 round-2 descriptions that 3.5-flash already audited, and
               report leak/inaccurate agreement + the flag-rate delta.  This is what makes
               the instrument change INTERPRETABLE rather than merely recorded: any
               marginal first-pass miss later is read against this delta.
  meter        a small metered probe -- exact `usageMetadata` per call, no bars -- so the
               cost of a lane can be MEASURED before it is committed to.  A12 set a hard
               50 TL reserve and forbade beginning a spend whose worst case breaches it;
               pro-tier token price was unobservable to A12, so it is measured here.

Every lane runs through the PRODUCTION audit path (`generate_descriptions.audit_one` +
`validate_audit_verdict`).  Validating an instrument through a parallel re-implementation
would certify code that the mass run does not use.

An unusable verdict is counted as an explicit ERROR, never as a pass and never as a flag,
and a lane whose error count is non-zero CANNOT return PASS.

    PY=$LAB/envs/diffusion/bin/python
    source $LAB/secrets/gemini_transition.env
    $PY validate_auditor.py meter       --n 10  --out <dir>
    $PY validate_auditor.py mismatch    --n 200 --out <dir>
    $PY validate_auditor.py calibration --n 200 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import generate_descriptions as gd  # noqa: E402
from caption_common import STRIPS_INDEX  # noqa: E402

PILOT = HERE / "pilot_m3"
ROUND2 = PILOT / "round2"
ROUND3 = PILOT / "round3"

#: A12's conservative ceiling, used only to turn measured tokens into a TL figure.
#: Deliberately an UPPER bound (the 895 TL invoice also covered ~760 unarchived calls).
TL_PER_MTOK_FLASH = 371.0


# ======================================================================================
# helpers
# ======================================================================================
def load_records(d: Path) -> dict:
    return json.loads((d / "records.json").read_text())


def accepted_rows(recs: dict) -> list:
    return [v for v in recs.values() if v.get("description")]


def derange_by_role(rows: list, shift: int = 7) -> list:
    """A8's exact derangement: within each role, pair row i with row i+7 (wrapping).

    Kept identical to the procedure 3.5-flash was certified under, so the two
    certificates are comparable -- that comparability is the whole point of re-running
    it rather than inventing a new control.
    """
    byrole: dict[str, list] = {}
    for v in rows:
        byrole.setdefault(v["role"], []).append(v)
    jobs = []
    for role, vs in sorted(byrole.items()):
        shifted = vs[shift:] + vs[:shift]
        for v, other in zip(vs, shifted):
            if v["clip_id"] == other["clip_id"]:
                continue
            jobs.append((v, other, role))
    return jobs


def usage_of(rec: dict) -> dict:
    u = ((rec.get("raw_response") or {}).get("usageMetadata")) or {}
    return {"prompt": u.get("promptTokenCount") or 0,
            "out": u.get("candidatesTokenCount") or 0,
            "think": u.get("thoughtsTokenCount") or 0,
            "total": u.get("totalTokenCount") or 0}


def summarise_tokens(recs: list) -> dict:
    tot = {"prompt": 0, "out": 0, "think": 0, "total": 0}
    n = 0
    for r in recs:
        u = usage_of(r)
        if u["total"]:
            n += 1
            for k in tot:
                tot[k] += u[k]
    per = {k: round(v / n, 1) for k, v in tot.items()} if n else {}
    return {"n_with_usage": n, "totals": tot, "per_call": per}


def run_pool(jobs, workers: int, label: str):
    """Run audits concurrently.  Returns (records, verdicts, errors).

    `errors` are recorded, never silently dropped: a lane with errors cannot PASS.
    """
    out = []
    t0 = time.time()
    gd._stop.clear()

    def one(j):
        v, other, role = j
        video = json.loads(STRIPS_INDEX.read_text())[other["clip_id"]][f"{role}_video"] \
            if isinstance(other, dict) else other
        return v, gd.audit_one(v["clip_id"], role, v["description"], video)

    idx = json.loads(STRIPS_INDEX.read_text())

    def one_job(j):
        row, video_clip, role = j
        video = idx[video_clip][f"{role}_video"]
        rec = gd.audit_one(row["clip_id"], role, row["description"], video)
        rec["audited_against_clip"] = video_clip
        rec["description"] = row["description"]
        try:
            rec["_verdict_ok"] = gd.validate_audit_verdict(rec)
        except gd.AuditError as e:
            rec["_verdict_ok"] = None
            rec["_audit_error"] = str(e)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, rec in enumerate(ex.map(one_job, jobs), 1):
            out.append(rec)
            if i % 50 == 0:
                print(f"  [{label}] {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    print(f"  [{label}] done {len(out)} in {time.time()-t0:.0f}s", flush=True)
    return out


def archive(path: Path, recs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def pin_block() -> dict:
    # Ruling citations carry the FILENAME, never the bare number: the advisor namespace
    # collided FOUR times (A10, A11, A12, A13 each have two files), so "per A13" is
    # ambiguous.  See advisors/LEDGER.md.
    return {"model": gd.AUDIT_MODEL, "temperature": gd.AUDIT_TEMPERATURE,
            "thinking_level": gd.AUDIT_THINKING_LEVEL,
            "max_output_tokens": gd.AUDIT_MAX_TOKENS,
            "authority": ("advisors/A14_RECONCILIATION_VERBATIM.md Q2 -- auditor pin, "
                          "conditional on the step-2 matched-side bars")}


def cost_block(tok: dict) -> dict:
    """Report the measured tokens, and what they cost at flash rates and at pro multiples.

    The multiplier is NOT assumed: it is reported as a band (A12's 4-8x) so the reader can
    see the decision's sensitivity to the one number nobody has measured yet.
    """
    total = tok["totals"]["total"]
    at_flash = total * TL_PER_MTOK_FLASH / 1e6
    return {"tokens_total": total,
            "TL_at_flash_rate_371_per_M": round(at_flash, 2),
            "TL_if_pro_is_4x_flash": round(at_flash * 4, 2),
            "TL_if_pro_is_8x_flash": round(at_flash * 8, 2),
            "note": ("A12 §2: pro token price is 'unobservable under prepaid TL and typically "
                     "4-8x flash'. The band is reported rather than a point estimate.")}


# ======================================================================================
# lanes
# ======================================================================================
def cmd_meter(a):
    """Metered probe: real audit calls, exact usage, no bars.  Cost discovery only."""
    out = Path(a.out)
    rows = accepted_rows(load_records(ROUND3 if a.store == "round3" else ROUND2))
    jobs = [(r, r["clip_id"], r["role"]) for r in rows[:a.n]]   # MATCHED (correct clip)
    print(f"[meter] {len(jobs)} matched audits | auditor={gd.AUDIT_MODEL} "
          f"thinking={gd.AUDIT_THINKING_LEVEL} cap={gd.AUDIT_MAX_TOKENS}")
    recs = run_pool(jobs, a.workers, "meter")
    archive(out / "raw_meter_responses.jsonl", recs)
    tok = summarise_tokens(recs)
    errs = [r["_audit_error"] for r in recs if r.get("_audit_error")]
    res = {"lane": "meter", "auditor_pin": pin_block(), "n": len(recs),
           "n_errors": len(errs), "errors": errs[:10],
           "model_version_echo": sorted({r.get("model_version_echo") for r in recs}),
           "tokens": tok, "cost": cost_block(tok),
           "per_description_tokens": tok["per_call"].get("total")}
    (out / "AUDITOR_METER.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def cmd_mismatch(a):
    """A8's wrong-clip control, re-run for the new model.  Bar: inaccurate=YES >= 95%."""
    out = Path(a.out)
    store = ROUND3 if a.store == "round3" else ROUND2
    rows = accepted_rows(load_records(store))
    pairs = derange_by_role(rows)
    if a.n:
        # Truncate EVENLY across roles.  `derange_by_role` emits A-role then B-role, so a
        # plain [:n] on a 399-pair list would have made a "200-pair" control almost
        # entirely A-role -- a control that silently stops covering half the store is the
        # same failure class as the verdict bug this campaign just fixed.
        byrole: dict[str, list] = {}
        for v, other, role in pairs:
            byrole.setdefault(role, []).append((v, other, role))
        per = a.n // max(1, len(byrole))
        pairs = [p for role in sorted(byrole) for p in byrole[role][:per]]
    jobs = [(v, other["clip_id"], role) for v, other, role in pairs]
    print(f"[mismatch] {len(jobs)} wrong-clip pairings from {store.name} | "
          f"auditor={gd.AUDIT_MODEL}")
    recs = run_pool(jobs, a.workers, "mismatch")
    archive(out / "raw_mismatch_responses.jsonl", recs)

    ok = [r for r in recs if r.get("_verdict_ok")]
    errs = [r for r in recs if r.get("_audit_error")]
    n = len(ok)
    inacc = sum(1 for r in ok if r["_verdict_ok"]["inaccurate"] == "YES")
    leak = sum(1 for r in ok if r["_verdict_ok"]["leak"] == "YES")
    flag = sum(1 for r in ok if r["_verdict_ok"]["inaccurate"] == "YES"
               or r["_verdict_ok"]["leak"] == "YES")
    tok = summarise_tokens(recs)
    pct = round(100 * inacc / n, 2) if n else 0.0
    res = {"lane": "mismatch_control", "auditor_pin": pin_block(),
           "store": store.name, "derangement": "A8 exact: within-role shift of 7",
           "n_requested": len(jobs), "n_usable_verdicts": n, "n_errors": len(errs),
           "errors": [r["_audit_error"] for r in errs][:10],
           "inaccurate_yes": inacc, "leak_yes": leak, "flagged_any": flag,
           "inaccurate_pct": pct,
           "flagged_any_pct": round(100 * flag / n, 2) if n else 0.0,
           "bar": "inaccurate=YES >= 95% (A13 step 3.1)",
           "n_meets_min_200": len(jobs) >= 200,
           "model_version_echo": sorted({r.get("model_version_echo") for r in recs}),
           "tokens": tok, "cost": cost_block(tok),
           "verdict": ("PASS" if (n and pct >= 95.0 and not errs and len(jobs) >= 200)
                       else "FAIL")}
    if errs:
        res["verdict_note"] = ("errors present -> cannot PASS: an unverified measurement "
                               "cannot gate (standing rule)")
    (out / "AUDITOR_MISMATCH_CONTROL.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "tokens"}, indent=1))
    return res


def cmd_calibration(a):
    """Cross-auditor calibration against 3.5-flash's archived verdicts on round 2."""
    out = Path(a.out)
    recs2 = load_records(ROUND2)
    # the round-2 records carry the ACCEPTED attempt's 3.5-flash verdict in `audit`
    rows = [v for v in accepted_rows(recs2) if isinstance(v.get("audit"), dict)
            and v["audit"].get("leak") in ("YES", "NO")]
    rows = rows[:a.n]
    jobs = [(r, r["clip_id"], r["role"]) for r in rows]           # MATCHED, correct clip
    print(f"[calibration] re-auditing {len(jobs)} round-2 descriptions "
          f"(3.5-flash verdicts on file) | auditor={gd.AUDIT_MODEL}")
    recs = run_pool(jobs, a.workers, "calibration")
    archive(out / "raw_calibration_responses.jsonl", recs)

    old_by = {f"{r['clip_id']}|{r['role']}": r["audit"] for r in rows}
    agree = {"leak": 0, "inaccurate": 0}
    both = 0
    old_flags = {"leak": 0, "inaccurate": 0}
    new_flags = {"leak": 0, "inaccurate": 0}
    disagreements = []
    errs = [r["_audit_error"] for r in recs if r.get("_audit_error")]
    for r in recs:
        v = r.get("_verdict_ok")
        if not v:
            continue
        key = f"{r['clip_id']}|{r['role']}"
        old = old_by.get(key) or {}
        if old.get("leak") not in ("YES", "NO"):
            continue
        both += 1
        for f in ("leak", "inaccurate"):
            o, nw = old.get(f), v.get(f)
            if o == nw:
                agree[f] += 1
            else:
                disagreements.append({"key": key, "field": f, "flash_3_5": o,
                                      "pro_3_1": nw,
                                      "description": (r.get("description") or "")[:110]})
            if o == "YES":
                old_flags[f] += 1
            if nw == "YES":
                new_flags[f] += 1
    tok = summarise_tokens(recs)
    res = {"lane": "cross_auditor_calibration", "auditor_pin": pin_block(),
           "reference_auditor": "gemini-3.5-flash (round-2 archived verdicts)",
           "n_requested": len(jobs), "n_compared": both, "n_errors": len(errs),
           "errors": errs[:10],
           "n_meets_min_150": len(jobs) >= 150,
           "agreement_pct": {f: (round(100 * agree[f] / both, 2) if both else None)
                             for f in ("leak", "inaccurate")},
           "flag_rate_pct": {
               f: {"flash_3_5": round(100 * old_flags[f] / both, 2) if both else None,
                   "pro_3_1": round(100 * new_flags[f] / both, 2) if both else None,
                   "delta_pp": round(100 * (new_flags[f] - old_flags[f]) / both, 2)
                   if both else None}
               for f in ("leak", "inaccurate")},
           "disagreements": disagreements[:40],
           "n_disagreements": len(disagreements),
           "model_version_echo": sorted({r.get("model_version_echo") for r in recs}),
           "tokens": tok, "cost": cost_block(tok),
           "interpretation": ("A13: this delta is the bridge for reading any marginal "
                              "first-pass miss on the mass store. It is a calibration "
                              "report, not a pass/fail gate.")}
    (out / "AUDITOR_CALIBRATION.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("tokens", "disagreements")}, indent=1))
    return res


def cmd_matched(a):
    """A14 EXECUTION STEP 2 -- the keystone.  One run, three obligations.

    Re-audit the round-2 IN-GRID descriptions (the ones A12 pre-registered for reuse)
    under the candidate auditor, on the production audit path, against the CORRECT clip.

    Why this one run is load-bearing three times over (A14 Q2):
      (a) MATCHED-SIDE CONTROL.  The 220/220 mismatch certificate is ONE-SIDED: an auditor
          that flags everything also scores 220/220.  `gemini-3-flash-preview` has no
          measured matched-side false-positive rate (3.5-flash 5.75%, flash-lite 2.0%).
          This measures it.  PRE-COMMITTED BARS, fixed before the run:
          matched `inaccurate=YES` <= 10%, `errors == 0`.
      (b) CROSS-AUDITOR CALIBRATION.  Agreement + flag-rate delta against the archived
          3.5-flash verdicts -- the comparability bridge for reading a marginal
          first-pass miss later (no trend may be claimed across an auditor change).
      (c) AUDIT-PROVENANCE UNIFICATION, which is what legitimises reusing these rows
          inside a store whose remainder is audited by the pinned auditor.

    TWO ARMS, and they are kept strictly apart:
      arm `accepted`  -- the reused descriptions.  THIS ARM CARRIES THE BARS.  Note the
                        archived reference verdicts here are all NO *by construction*
                        (a row was accepted only because 3.5-flash said NO), so its
                        "agreement" is not independent evidence -- it IS the matched-side
                        false-positive rate.  Reported honestly as such.
      arm `flagged`   -- round-2 attempts 3.5-flash actually flagged.  REPORTED, NEVER
                        BARRED: it is the only positive-side agreement available, and
                        folding it into a pre-committed bar's denominator would move the
                        bar after the fact.
    """
    out = Path(a.out)
    grid = {tuple(x) for x in json.loads(Path(a.grid).read_text())}
    recs2 = load_records(ROUND2)

    accepted, flagged = [], []
    for v in recs2.values():
        if (v["clip_id"], v["role"]) not in grid:
            continue
        if v.get("description") and isinstance(v.get("audit"), dict) \
                and v["audit"].get("leak") in ("YES", "NO"):
            accepted.append(v)
        for h in v.get("history", []):
            old = h.get("audit")
            if isinstance(old, dict) and h.get("description") and \
                    (old.get("leak") == "YES" or old.get("inaccurate") == "YES"):
                flagged.append({"clip_id": v["clip_id"], "role": v["role"],
                                "description": h["description"], "audit": old})
    if a.n:
        accepted = accepted[:a.n]

    print(f"[matched] grid={a.grid} ({len(grid)} pairs) | reused in-grid accepted="
          f"{len(accepted)} | in-grid 3.5-flash-flagged attempts={len(flagged)} | "
          f"auditor={gd.AUDIT_MODEL} thinking={gd.AUDIT_THINKING_LEVEL}")

    def audit_arm(rows, label):
        jobs = [(r, r["clip_id"], r["role"]) for r in rows]      # MATCHED = correct clip
        return run_pool(jobs, a.workers, label) if jobs else []

    rec_acc = audit_arm(accepted, "matched/accepted")
    rec_flg = audit_arm(flagged, "matched/flagged")
    archive(out / "raw_matched_accepted_responses.jsonl", rec_acc)
    archive(out / "raw_matched_flagged_responses.jsonl", rec_flg)

    def score(rows, refs, label):
        ref_by = {f'{r["clip_id"]}|{r["role"]}': r["audit"] for r in refs}
        ok = [r for r in rows if r.get("_verdict_ok")]
        errs = [r["_audit_error"] for r in rows if r.get("_audit_error")]
        n = len(ok)
        agree = {"leak": 0, "inaccurate": 0}
        new_f = {"leak": 0, "inaccurate": 0}
        old_f = {"leak": 0, "inaccurate": 0}
        disagreements = []
        for r in ok:
            key = f'{r["clip_id"]}|{r["role"]}'
            old, new = ref_by.get(key) or {}, r["_verdict_ok"]
            for f in ("leak", "inaccurate"):
                if new.get(f) == "YES":
                    new_f[f] += 1
                if old.get(f) == "YES":
                    old_f[f] += 1
                if old.get(f) == new.get(f):
                    agree[f] += 1
                else:
                    disagreements.append({
                        "key": key, "field": f, "ref_3_5_flash": old.get(f),
                        "new_auditor": new.get(f),
                        "errors": (new.get("errors") or [])[:3],
                        "description": (r.get("description") or "")[:130]})
        pct = (lambda k: round(100 * k / n, 2) if n else None)
        return {
            "arm": label, "n_requested": len(rows), "n_usable_verdicts": n,
            "n_errors": len(errs), "errors": errs[:10],
            "new_flag_pct": {f: pct(new_f[f]) for f in new_f},
            "ref_flag_pct": {f: pct(old_f[f]) for f in old_f},
            "delta_pp": {f: (round(100 * (new_f[f] - old_f[f]) / n, 2) if n else None)
                         for f in new_f},
            "agreement_pct": {f: pct(agree[f]) for f in agree},
            "n_disagreements": len(disagreements), "disagreements": disagreements[:60],
            "model_version_echo": sorted({r.get("model_version_echo") for r in rows}),
            "tokens": summarise_tokens(rows),
        }

    acc = score(rec_acc, accepted, "accepted_reused_in_grid")
    flg = score(rec_flg, flagged, "attempts_flagged_by_3_5_flash")
    acc["reference_note"] = (
        "The 3.5-flash reference verdicts on this arm are NO for every field BY "
        "CONSTRUCTION -- a row is in this arm only because 3.5-flash accepted it. "
        "So `agreement_pct` here is 100 - new_flag_pct and is NOT independent evidence; "
        "the informative quantity is new_flag_pct.inaccurate = the MATCHED-SIDE "
        "FALSE-POSITIVE RATE, which is precisely the gap A14 Q2 says the one-sided "
        "220/220 mismatch control leaves open.")
    flg["reference_note"] = (
        "REPORTED, NOT BARRED. The only positive-side agreement available: rows 3.5-flash "
        "flagged and round 2 therefore regenerated. Folding these into the bar's "
        "denominator would move a pre-committed bar after the fact.")

    matched_flag = acc["new_flag_pct"]["inaccurate"]
    bars = {"matched_inaccurate_yes_max_pct": 10.0, "errors_max": 0,
            "pre_committed": ("A14 EXECUTION STEP 2, fixed before the run "
                              "(advisors/A14_RECONCILIATION_VERBATIM.md Q2)"),
            "on_fail": ("fall back to gemini-3.5-flash-lite and repeat this same "
                        "re-audit -- pre-committed, no new consultation. If BOTH fail, "
                        "STOP: no auditor has earned the certificate.")}
    reasons = []
    if acc["n_usable_verdicts"] == 0:
        reasons.append("no usable verdicts -- nothing was measured")
    if acc["n_errors"]:
        reasons.append(f'{acc["n_errors"]} audit errors (bar: 0)')
    if matched_flag is None or matched_flag > 10.0:
        reasons.append(f"matched inaccurate=YES {matched_flag}% > 10% bar")
    verdict = "PASS" if not reasons else "FAIL"

    res = {"lane": "matched_side_control_and_calibration",
           "authority": "A14 execution step 2 (advisors/A14_RECONCILIATION_VERBATIM.md)",
           "discharges": ["matched-side false-positive control (the one-sided gap)",
                          "cross-auditor calibration vs archived 3.5-flash verdicts",
                          "audit-provenance unification of the reused in-grid rows"],
           "auditor_pin": pin_block(), "grid": a.grid, "grid_pairs": len(grid),
           "reference_auditor": "gemini-3.5-flash (round-2 archived verdicts)",
           "bars": bars, "matched_inaccurate_pct": matched_flag,
           "arms": {"accepted": acc, "flagged": flg},
           "fail_reasons": reasons, "verdict": verdict}
    (out / "AUDITOR_MATCHED_CONTROL.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1))
    for k, arm in res["arms"].items():
        print(f'  [{k}] n={arm["n_usable_verdicts"]} errors={arm["n_errors"]} '
              f'new_flag={arm["new_flag_pct"]} ref_flag={arm["ref_flag_pct"]} '
              f'agree={arm["agreement_pct"]}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("meter", "mismatch", "calibration", "matched"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--store", choices=("round2", "round3"), default="round3")
    ap.add_argument("--grid", default="outputs/ctt_v2/captions/mass_pairs.json",
                    help="the pinned (clip, role) grid; `matched` restricts to it")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"meter": cmd_meter, "mismatch": cmd_mismatch, "calibration": cmd_calibration,
     "matched": cmd_matched}[a.cmd](a)


if __name__ == "__main__":
    main()
