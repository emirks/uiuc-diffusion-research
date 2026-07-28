#!/usr/bin/env python3
"""Assemble S4's TRAINING captions from its stored descriptions, and validate every one.

The stores hold bare descriptions (trailing period stripped, A4 Q1 convention).  What the
inventory's `caption` field must contain is the ASSEMBLED caption the trainer sees:

    one-sided:  "{A-description}. sksz."          <- S4
    two-sided:  "{A-description}. sksz. {B-description}."

S4 is one-sided, so there is no suffix sentence.  Every output is checked with
`root_common.caption_violations` — the same function the assembler applies — so a caption that
would fail at assembly fails here instead, where it is cheap to fix.

Usage:
    python scripts/ctt_v2/captions/assemble_s4_captions.py \
        --out outputs/ctt_v2/captions/S4_CAPTIONS_ASSEMBLED.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/ctt_v2"))
import root_common as rc  # noqa: E402

STORE = REPO / "outputs/ctt_v2/captions/S4_CAPTION_STORE.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    store = json.loads(STORE.read_text())
    desc = store["descriptions"]

    out: dict[str, str] = {}
    for key, text in desc.items():
        clip, _, role = key.rpartition("|")
        if role != "A":
            raise SystemExit(f"S4 is one-sided; unexpected role in key {key!r}")
        out[clip] = f"{text}.{rc.TRIGGER_SENTENCE}"

    filt = rc.leak_filter()
    bad = {c: v for c, v in ((c, rc.caption_violations(t, filt)) for c, t in out.items()) if v}
    if bad:
        raise SystemExit(f"[assemble] {len(bad)} assembled caption(s) violate RULING 9, "
                         f"first 5:\n  " + "\n  ".join(f"{c}: {v}" for c, v in
                                                       list(bad.items())[:5]))

    p = Path(args.out)
    if not p.is_absolute():
        p = REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(sorted(out.items())), indent=1))
    h = hashlib.sha256(json.dumps(dict(sorted(out.items())), sort_keys=True).encode()).hexdigest()
    print(f"[ok] {p.relative_to(REPO)}: {len(out)} assembled captions, 0 RULING 9 violations")
    print(f"[ok] leak filter: {filt.source}")
    print(f"[ok] content_hash sha256:{h[:16]}  (from store {store['content_hash'][7:23]})")
    print(f"[ok] example: {out[sorted(out)[0]]}")


if __name__ == "__main__":
    main()
