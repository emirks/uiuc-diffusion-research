#!/usr/bin/env python3
"""Persist each synthetic stratum's ASSEMBLED training caption into its inventory.

S2a/S2b carry `caption: None` (7,961 and 7,990 clips): their captions are assembled from the
locked (clip, role) store at encode-input time and were never written back, so `assemble_root`
stops with "inventory has no caption". S1 and S4 already carry theirs (from `build_s1_spec.py`
and the S4 store); S0 is `kind == "corpus"` and carries the certified text.

The caption is taken from `build_encode_inputs.assembled_for()` — **the same function whose output
was encoded** — rather than re-assembled here. That matters: the `conditions` path attached by
`attach_conditions.py` points at a CONTENT-ADDRESSED embed keyed by sha256 of the caption text, so
a caption re-derived by any second implementation could drift from the tensor the trainer actually
consumes, and nothing downstream would notice.

Consistency is then PROVEN, not assumed: for every clip, sha256(attached caption)[:16] must equal
the hash recorded in `<st>_clip_to_hash.json`, i.e. the basename of the embed the inventory points
at. A mismatch is a hard stop.

    python scripts/ctt_v2/attach_captions.py            # report
    python scripts/ctt_v2/attach_captions.py --write
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/ctt_v2"))
import root_common as rc  # noqa: E402

IN = REPO / "outputs/ctt_v2/conditions_inputs"
INV = REPO / "outputs/ctt_v2/inventories"
LOCKED = REPO / "outputs/ctt_v2/captions/CAPTION_STORE.json"
S4_STORE = REPO / "outputs/ctt_v2/captions/S4_CAPTION_STORE.json"


def _bei():
    """Load the encode-input builder by path — it is the authority for caption assembly."""
    p = REPO / "scripts/ctt_v2/captions/build_encode_inputs.py"
    spec = importlib.util.spec_from_file_location("_ctt_bei", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    bei = _bei()
    locked = json.loads(LOCKED.read_text())["descriptions"]
    s4 = json.loads(S4_STORE.read_text())["descriptions"]
    filt = rc.leak_filter()
    rc_ = 0

    for st in ("S1", "S2a", "S2b", "S4"):
        inv_p = INV / f"{st}.json"
        inv = json.loads(inv_p.read_text())
        need = [s for s, c in inv["clips"].items() if not c.get("caption")]
        c2h = json.loads((IN / f"{st}_clip_to_hash.json").read_text())
        caps = bei.assembled_for(st, locked, s4)

        # consistency: the caption must hash to the embed the inventory already points at
        bad_hash, bad_leak, absent = [], [], []
        for stem in inv["clips"]:
            cap = caps.get(stem)
            if cap is None:
                absent.append(stem)
                continue
            h = hashlib.sha256(cap.encode()).hexdigest()[:16]
            if h != c2h.get(stem):
                bad_hash.append(f"{stem}: caption sha {h} != mapped {c2h.get(stem)}")
            if rc.caption_violations(cap, filt):
                bad_leak.append(stem)

        ok = not bad_hash and not bad_leak and not absent
        print(f"[{st}] {len(inv['clips']):6,} clips | missing caption {len(need):6,} "
              f"| hash mismatch {len(bad_hash):4} | RULING 9 {len(bad_leak):4} "
              f"| not assembled {len(absent):4} | {'OK' if ok else 'PROBLEM'}")
        for lst, lbl in ((bad_hash, "hash"), (bad_leak, "leak"), (absent, "absent")):
            if lst:
                print(f"      {lbl}, first 3: {lst[:3]}")
        if not ok:
            rc_ = 1
            continue
        if not need:
            print("      nothing to fill (already present and hash-consistent)")
            continue

        if args.write:
            for stem in need:
                inv["clips"][stem]["caption"] = caps[stem]
            inv.setdefault("provenance", {})["captions_attached"] = {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "by": "scripts/ctt_v2/attach_captions.py",
                "source": "build_encode_inputs.assembled_for() -- the SAME function whose output "
                          "was Gemma-encoded, so the stored caption cannot drift from the "
                          "content-addressed embed the `conditions` path resolves to",
                "n_filled": len(need),
                "consistency_proof": "sha256(caption)[:16] == the clip's hash in "
                                     f"{st}_clip_to_hash.json, verified for all "
                                     f"{len(inv['clips'])} clips",
                "ruling9_violations": 0,
            }
            rc.assert_no_worktree_paths(inv, f"{st} inventory")
            inv_p.write_text(json.dumps(inv, indent=1))
            print(f"      [ok] filled {len(need):,} captions -> {inv_p.relative_to(REPO)}")
    if not args.write:
        print("\n(report only — pass --write to apply)")
    return rc_


if __name__ == "__main__":
    raise SystemExit(main())
