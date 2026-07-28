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
    return {"model": gd.AUDIT_MODEL, "temperature": gd.AUDIT_TEMPERATURE,
            "thinking_level": gd.AUDIT_THINKING_LEVEL,
            "max_output_tokens": gd.AUDIT_MAX_TOKENS,
            "authority": "A13 Q1 pinned auditor config"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("meter", "mismatch", "calibration"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--store", choices=("round2", "round3"), default="round3")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"meter": cmd_meter, "mismatch": cmd_mismatch, "calibration": cmd_calibration}[a.cmd](a)


if __name__ == "__main__":
    main()
