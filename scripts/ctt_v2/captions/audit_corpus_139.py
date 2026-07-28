#!/usr/bin/env python
"""The S0 corpus-139 Layer-2 audit (A4's requirement; A8 s6 / A13-a / A14 Q4: BEFORE assembly).

WHAT IS BEING ASKED.  The corpus caption grammar is

    "{start desc} sksz. {end desc}"        (or "{start desc} sksz." when one-sided)

so the captions describe the clip's ENDPOINTS and the ` sksz.` trigger carries the
transition itself.  The audit question is therefore NOT "is the caption right" -- it is
whether an *endpoint* description leaks the *transition effect* that the trigger is
supposed to be the sole carrier of.  A description of a person standing in a garage is
fine; a description that says the garage is "about to erupt" is a leak, because it turns
the caption into a soft trigger token and trains the text-routing recipe A4 Finding 2
banned.

Layer-2 = the same production auditor, the same AUDIT_QUESTION, the same
`validate_audit_verdict` hard-error path as the new store.  Each of the 139 captions is
split on ` sksz.` and each side is audited against ITS OWN byte-pure 9-frame anchor
(A-side vs frames 0-8, B-side vs frames 112-120), so a verdict is always about the window
the description actually claims to describe.

 THESE CAPTIONS ARE CERTIFIED AND THIS SCRIPT IS STRICTLY READ-ONLY.
   It writes a report and nothing else.  There is deliberately no --fix, no rewrite path
   and no store output.  A8 s6 and A14 Q4: "any hit escalates to owner, never silently
   edited."  A hit is reported VERBATIM for the owner to adjudicate; this script does not
   judge, drop, or repair one.

Usage
-----
  $PY audit_corpus_139.py --out <report.json> [--workers 8] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_descriptions as G  # noqa: E402
from caption_common import CORPUS_CAPTIONS  # noqa: E402

ANCHORS = (Path(__file__).resolve().parents[3]
           / "data/processed/corpus_anchors/corpus_anchors_index.json")


def split_caption(caption: str):
    """-> (a_desc, b_desc|None). Mirrors caption_common.load_corpus_descriptions exactly."""
    parts = caption.strip().split(" sksz.")
    a = parts[0].strip().rstrip(".").strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    b = rest.strip().rstrip(".").strip() if rest else None
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = json.loads(CORPUS_CAPTIONS.read_text())
    anchors = json.loads(ANCHORS.read_text())
    if a.limit:
        rows = rows[: a.limit]

    # ---- build the audit units, asserting anchor coverage POSITIVELY ---------
    units = []
    for r in rows:
        clip = Path(r["video"]).stem
        if clip not in anchors:
            raise SystemExit(
                f"caption for {r['video']!r} has no anchor entry (clip id {clip!r}). "
                f"Refusing to audit a subset silently: an absent anchor means the audit "
                f"would under-cover the certified corpus.")
        a_desc, b_desc = split_caption(r["caption"])
        if not a_desc:
            raise SystemExit(f"{clip}: empty A-side description parsed from caption")
        units.append((clip, "A", a_desc))
        if b_desc:
            units.append((clip, "B", b_desc))

    n_two_sided = sum(1 for _, role, _ in units if role == "B")
    print(f"{len(rows)} certified captions -> {len(units)} audit units "
          f"({len(rows)} A-side + {n_two_sided} B-side); auditor={G.AUDIT_MODEL} "
          f"thinkingLevel={G.AUDIT_THINKING_LEVEL}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = out.with_name(out.stem + "_raw_responses.jsonl").open("w")
    lock = threading.Lock()
    results, errors = [], []

    def one(u):
        clip, role, desc = u
        video = anchors[clip][f"{role}_video"]
        arec = G.audit_one(clip, role, desc, video)
        with lock:
            archive.write(json.dumps(arec) + "\n")
            archive.flush()
        v = G.validate_audit_verdict(arec)   # hard-errors on any unusable verdict
        return {"clip_id": clip, "role": role, "description": desc,
                "anchor_video": video, "verdict": v,
                "model_version_echo": arec.get("model_version_echo")}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, u): u for u in units}
        for i, (f, u) in enumerate(futs.items(), 1):
            try:
                results.append(f.result())
            except Exception as e:            # noqa: BLE001
                errors.append({"unit": list(u), "error": f"{type(e).__name__}: {e}"})
            if i % 50 == 0:
                print(f"  {i}/{len(units)}")
    archive.close()

    # ---- A11 positive-presence: "zero hits" may only PASS if we positively
    #      parsed a verdict for EVERY unit.  A shrunken denominator is a FAIL.
    leaks = [r for r in results if str(r["verdict"].get("leak", "")).upper() == "YES"]
    inacc = [r for r in results if str(r["verdict"].get("inaccurate", "")).upper() == "YES"]
    complete = (len(results) == len(units)) and not errors

    report = {
        "what": "S0 corpus-139 Layer-2 audit (endpoint descriptions vs byte-pure 9-frame anchors)",
        "authority": ("A4 (audit required before assembly); A8 s6; A14 Q4 -- hits ESCALATE "
                      "to owner, captions are CERTIFIED and stay byte-identical"),
        "read_only": True,
        "captions_source": str(CORPUS_CAPTIONS),
        "anchors_index": str(ANCHORS),
        "auditor": {"model": G.AUDIT_MODEL, "thinking_level": G.AUDIT_THINKING_LEVEL,
                    "temperature": G.AUDIT_TEMPERATURE,
                    "question": G.AUDIT_QUESTION,
                    "model_version_echo": sorted({r["model_version_echo"]
                                                  for r in results if r["model_version_echo"]})},
        "counts": {"captions": len(rows), "audit_units": len(units),
                   "a_side": len(rows), "b_side": n_two_sided,
                   "verdicts_parsed": len(results), "errors": len(errors)},
        "positive_presence_control": {
            "rule": ("A11: an absence-assert may only PASS if the instrument positively "
                     "produced a verdict for every unit; a shrunken denominator is a FAIL, "
                     "never a zero-hits PASS."),
            "verdicts_parsed_equals_units": len(results) == len(units),
            "errors_zero": not errors,
            "satisfied": complete,
        },
        "result": {
            "leak_YES": len(leaks),
            "inaccurate_YES": len(inacc),
            "verdict": ("CLEAN" if complete and not leaks and not inacc else
                        "INSTRUMENT-INCOMPLETE" if not complete else "HITS -- ESCALATE"),
        },
        "hits_verbatim": {"leak": leaks, "inaccurate": inacc},
        "errors": errors,
        "escalation_note": ("Any hit above is reported verbatim and is the OWNER's call. "
                            "This script did not and must not edit a certified caption."),
    }
    out.write_text(json.dumps(report, indent=1))

    print(f"\nleak=YES {len(leaks)} | inaccurate=YES {len(inacc)} | errors {len(errors)}")
    print(f"positive-presence control satisfied: {complete}")
    print(f"VERDICT: {report['result']['verdict']}")
    for h in leaks + inacc:
        tag = "LEAK" if h in leaks else "INACCURATE"
        print(f"  [{tag}] {h['clip_id']}|{h['role']}: {h['description']!r}")
        print(f"          verdict: {json.dumps(h['verdict'])}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
