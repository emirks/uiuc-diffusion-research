#!/usr/bin/env python
"""Build the Tier-2 REVIEW queue as a ready-to-work artifact -- A4 Q3 Layer-1 Tier-2.

A4's adjudication question is fixed: *is this describing visible scene content, or
an effect/change?*  Scene content passes (a house on fire is a house on fire).

This emits (a) a JSON queue with one record per flagged description, each carrying
the matched span in context and an empty `verdict` field for the reviewer, and
(b) a TSV for eyeballing in a spreadsheet.  It also emits the 23 Tier-2 flags from
the CERTIFIED corpus as calibration rows (`source: corpus_calibration`, verdict
pre-filled `SCENE_CONTENT`) -- those are known-good by construction, so they show
the reviewer what a pass looks like before they touch a single new row.

No API calls.  Usage:
  PY=/projects/illinois/eng/cs/jrehg/users/emirkisa/envs/diffusion/bin/python
  $PY tier2_queue.py --store <dir-with-records.json> [--store <dir2> ...] \
      --out pilot_m3/tier2_queue.json --tsv pilot_m3/tier2_queue.tsv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_common import LeakFilter, load_corpus_descriptions  # noqa: E402

ADJUDICATION_QUESTION = (
    "Is this describing visible scene content, or an effect/change? "
    "Scene content passes (a house on fire is a house on fire)."
)
VERDICT_VALUES = ["SCENE_CONTENT (keep)", "EFFECT_OR_CHANGE (regenerate)", "UNSURE (escalate)"]


def context_for(text: str, flag: str, width: int = 44) -> str:
    """Return the flagged token with surrounding context, or '' if not locatable."""
    token = flag.split(":", 1)[1] if ":" in flag else flag
    m = re.search(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"\w*", text, re.I)
    if not m:
        return ""
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return ("…" if a else "") + text[a:m.start()] + "«" + text[m.start():m.end()] + "»" \
        + text[m.end():b] + ("…" if b < len(text) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--no-corpus-calibration", action="store_true")
    a = ap.parse_args()

    lf = LeakFilter()
    rows = []

    if not a.no_corpus_calibration:
        cA, cB = load_corpus_descriptions()
        for role, descs in (("A", cA), ("B", cB)):
            for i, d in enumerate(descs):
                flags = lf.tier2(d)
                if flags:
                    rows.append({
                        "source": "corpus_calibration", "store": "certified_139",
                        "clip_id": f"corpus_{role}_{i}", "role": role, "bank": "corpus",
                        "description": d, "tier2_flags": flags,
                        "contexts": {f: context_for(d, f) for f in flags},
                        "verdict": "SCENE_CONTENT (keep)",
                        "note": "CERTIFIED text — known-good by construction; calibration only, "
                                "never edit.",
                    })

    for store in a.store:
        recs = json.loads((Path(store) / "records.json").read_text())
        label = Path(store).name
        for key, v in sorted(recs.items()):
            if not v.get("description") or not v.get("tier2"):
                continue
            d = v["description"]
            rows.append({
                "source": "new", "store": label,
                "clip_id": v["clip_id"], "role": v["role"], "bank": v["bank"],
                "description": d, "tier2_flags": v["tier2"],
                "contexts": {f: context_for(d, f) for f in v["tier2"]},
                "verdict": "", "note": "",
            })

    n_new = sum(1 for r in rows if r["source"] == "new")
    n_cal = len(rows) - n_new
    from collections import Counter
    flag_counts = Counter(f for r in rows if r["source"] == "new" for f in r["tier2_flags"])

    out = {
        "created": "2026-07-28",
        "adjudication_question": ADJUDICATION_QUESTION,
        "verdict_values": VERDICT_VALUES,
        "rule": "A4 Q3 Tier-2 is a REVIEW flag, not a rejection. Tier-1 is the only "
                "auto-reject tier. Filling `verdict` is the whole job.",
        "calibration_note": "The certified 139 corpus captions trip Tier-2 at 13.5% (23/171) on "
                            "exactly the same kind of legitimate content (glow, water, earth, box, "
                            "fire, cloud). Those rows are included with verdict pre-filled so the "
                            "reviewer can calibrate before judging new text.",
        "counts": {"new_flagged": n_new, "corpus_calibration": n_cal, "total": len(rows),
                   "new_flag_histogram": dict(flag_counts.most_common())},
        "queue": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}: {n_new} new flagged + {n_cal} calibration = {len(rows)} rows")
    print("new flag histogram:", dict(flag_counts.most_common()))

    if a.tsv:
        lines = ["source\tstore\tclip_id\trole\tbank\tflags\tcontext\tdescription\tverdict"]
        for r in rows:
            ctx = " | ".join(v for v in r["contexts"].values() if v)
            lines.append("\t".join([
                r["source"], r["store"], r["clip_id"], r["role"], r["bank"],
                ",".join(r["tier2_flags"]), ctx.replace("\t", " "),
                r["description"].replace("\t", " "), r["verdict"],
            ]))
        Path(a.tsv).write_text("\n".join(lines) + "\n")
        print(f"wrote {a.tsv}")


if __name__ == "__main__":
    main()
