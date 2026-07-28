#!/usr/bin/env python
"""A8's residual loop: OPERATOR MANUAL REWRITE of descriptions the generator could not land.

A8 §4 pins `unresolved inaccurate in the final store = 0` as HARD, via A4's loop:
"regenerate once with fresh N; residual -> operator manual rewrite, re-audited, logged."
This is that last step, and it is deliberately a separate, auditable script rather than a
flag on the generator: a hand-written string must never enter the store on a path that
could be mistaken for a generated one.

Every rewrite is:
  * validated MECHANICALLY -- `format_violations` + Tier-1 (hard) and Tier-2 (recorded),
    the same functions the generator uses; no bypass, no lenient mode;
  * RE-AUDITED through the production `audit_one` + `validate_audit_verdict` path under
    the pinned auditor, with the raw response archived (an audit failure is a hard stop,
    never a silent accept -- the §21/§23.1 defect class);
  * LOGGED with the operator's edit rationale and the full pre-rewrite history, so the
    provenance of every hand-written byte is legible in the store.

Hand-editing to satisfy a *distributional gate* stays BANNED (A13/§23.7: the be-verb
hand-edit ban). This script exists only to fix content errors the auditor identified and
mechanical format violations -- never to move gate 8a/8b.

Usage
-----
  $PY manual_rewrite.py --rewrites rewrites.json --source-store <dir> --out <dir>

`rewrites.json`: [{"clip_id": ..., "role": "A"|"B", "description": ..., "reason": ...}]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_descriptions as G  # noqa: E402
from caption_common import LeakFilter, format_violations, word_count  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites", required=True)
    ap.add_argument("--source-store", required=True,
                    help="store dir holding the MANUAL_REWRITE_QUEUE records being fixed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rewrites = json.loads(Path(a.rewrites).read_text())
    src_recs = json.loads((Path(a.source_store) / "records.json").read_text())
    index = json.loads(G.STRIPS_INDEX.read_text())
    lf = LeakFilter()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    audit_archive = (out / "raw_audit_responses.jsonl").open("w")

    records, descriptions = {}, {}
    for rw in rewrites:
        clip, role, desc = rw["clip_id"], rw["role"], rw["description"]
        key = f"{clip}|{role}"
        entry = index[clip]
        video = entry[f"{role}_video"]

        # ---- mechanical validation: identical functions, no bypass ----------
        fmt = format_violations(desc, role)
        t1 = lf.tier1(desc)
        t2 = lf.tier2(desc)
        if fmt or t1:
            raise SystemExit(
                f"{key}: operator rewrite FAILS mechanical validation -- "
                f"format={fmt} tier1={t1}. Fix the rewrite; the store does not take it.")

        # ---- re-audit on the production path -------------------------------
        arec = G.audit_one(clip, role, desc, video)
        audit_archive.write(json.dumps(arec) + "\n")
        audit_archive.flush()
        v = G.validate_audit_verdict(arec)   # raises AuditError on any unusable verdict
        if v.get("leak") == "YES" or v.get("inaccurate") == "YES":
            raise SystemExit(
                f"{key}: operator rewrite STILL flagged by the auditor: {json.dumps(v)}\n"
                f"  text: {desc!r}\n"
                f"Escalate rather than loosen: a hand-written description the auditor "
                f"still rejects is not a store-ready description.")

        prior = src_recs.get(key, {})
        records[key] = {
            "clip_id": clip, "role": role, "bank": entry["bank"],
            "description": desc,
            "accepted_on_attempt": "operator_manual_rewrite",
            "provenance": "OPERATOR MANUAL REWRITE (A8 s4 residual loop)",
            "operator_reason": rw["reason"],
            "words": word_count(desc), "tier2": t2, "audit": v,
            "pre_rewrite_history": prior.get("history"),
            "pre_rewrite_status": prior.get("status"),
        }
        descriptions.setdefault(clip, {})[role] = desc
        print(f"[ok] {key}: audited clean ({word_count(desc)} words, tier2={t2})")
        print(f"     reason: {rw['reason']}")

    audit_archive.close()
    (out / "records.json").write_text(json.dumps(records, indent=1))
    (out / "descriptions.json").write_text(json.dumps(descriptions, indent=1))
    (out / "run_meta.json").write_text(json.dumps({
        "generator_model": "OPERATOR (hand-written)",
        "prompt_variant": "v2",
        "prompt_variant_note": (
            "The rewrite preserves the v2 register of the store it joins. No prompt was "
            "used; the operator edited the last generated attempt minimally to remove the "
            "specific violation the mechanical filters / auditor named."),
        "audit_enabled": True,
        "auditor_model": G.AUDIT_MODEL,
        "auditor_thinking_level": G.AUDIT_THINKING_LEVEL,
        "audit_temperature": G.AUDIT_TEMPERATURE,
        "n_pairs": len(records),
        "authority": "A8 s4: 'residual -> operator manual rewrite, re-audited, logged'",
        "hand_edit_ban_note": (
            "A13/DOSSIER s23.7: hand-editing to move a distributional gate (the be-verb "
            "tell) remains BANNED. These edits fix auditor-named content errors and "
            "mechanical audio-word format violations only."),
    }, indent=1))
    print(f"\n{len(records)} manual rewrite(s) accepted -> {out}")


if __name__ == "__main__":
    main()
