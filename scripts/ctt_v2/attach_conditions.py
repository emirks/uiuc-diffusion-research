#!/usr/bin/env python3
"""Attach the content-addressed `conditions` path to every synthetic stratum's inventory.

Why this exists as its own step
-------------------------------
`build_inventories._attach` fills `latents` / `cond_clean` / `conditions` from CLI path templates
whose substitution context is `{clip} {group} {class} {shader} {A} {B}`. The conditions tree is
**content-addressed** — keyed by the sha256 of the caption text, not by any of those — so no
template over that context can name it, and all four synthetic strata were left with
`conditions: None` (S0 is unaffected: it is `kind == "corpus"` and carries the certified
`eval_ladder/dataset/conditions/` paths directly).

The path shape is NOT invented here. `conditions_inputs/ENCODE_INPUTS_MANIFEST.json` already
declares, per stratum:

    conditions_path_template : outputs/ctt_v2/conditions/by_caption/{caption_sha16}.pt
    clip_to_caption_hash     : outputs/ctt_v2/conditions_inputs/<st>_clip_to_hash.json

so this reads the declaration and the map rather than restating either.

This is ADDITIVE and does not touch clip sets, groups, endpoints or captions — only a field that
is currently null. Rosters and encodes stay keyed to exactly the same stems. Every target is
verified to exist before anything is written, and a stratum is written only if it is complete.

    python scripts/ctt_v2/attach_conditions.py            # report
    python scripts/ctt_v2/attach_conditions.py --write
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/ctt_v2"))
import root_common as rc  # noqa: E402

IN = REPO / "outputs/ctt_v2/conditions_inputs"
INV = REPO / "outputs/ctt_v2/inventories"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    man = json.loads((IN / "ENCODE_INPUTS_MANIFEST.json").read_text())
    rc_ = 0
    for st, rec in man["strata"].items():
        inv_p = INV / f"{st}.json"
        inv = json.loads(inv_p.read_text())
        if inv.get("kind") == "corpus":
            print(f"[{st}] kind=corpus — carries certified conditions paths, skipped")
            continue

        tpl = rec["conditions_path_template"]
        c2h = json.loads((REPO / rec["clip_to_caption_hash"]).read_text())

        filled, missing_map, missing_file, already = 0, [], [], 0
        updates: dict[str, str] = {}
        for stem in inv["clips"]:
            h = c2h.get(stem)
            if h is None:
                missing_map.append(stem)
                continue
            p = REPO / tpl.format(caption_sha16=h)
            if not p.exists():
                missing_file.append(f"{stem} -> {h}")
                continue
            cur = inv["clips"][stem].get("conditions")
            new = rc.canonical_source(p)
            if cur == new:
                already += 1
            else:
                updates[stem] = new
                filled += 1

        ok = not missing_map and not missing_file
        print(f"[{st}] {len(inv['clips']):6,} clips | to fill {filled:6,} | already {already:5,} "
              f"| no hash {len(missing_map):4} | embed absent {len(missing_file):4} "
              f"| {'OK' if ok else 'INCOMPLETE'}")
        if missing_map:
            print(f"      no clip_to_hash entry, first 3: {missing_map[:3]}")
        if missing_file:
            print(f"      embed file absent, first 3: {missing_file[:3]}")
        if not ok:
            rc_ = 1
            continue

        if args.write and updates:
            for stem, v in updates.items():
                inv["clips"][stem]["conditions"] = v
            inv.setdefault("provenance", {})["conditions_attached"] = {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "by": "scripts/ctt_v2/attach_conditions.py",
                "authority": "content-addressed conditions tree; path shape read from "
                             "conditions_inputs/ENCODE_INPUTS_MANIFEST.json, never restated",
                "template": tpl,
                "clip_to_caption_hash": rec["clip_to_caption_hash"],
                "n_distinct_embeds": rec["n_distinct_captions"],
                "n_clips_pointed": len(c2h),
                "dedup_factor": rec["dedup_factor"],
                "note": "ADDITIVE: only the null `conditions` field was filled. Clip sets, "
                        "groups, endpoints and captions are untouched, so rosters and encodes "
                        "remain keyed to identical stems.",
            }
            rc.assert_no_worktree_paths(inv, f"{st} inventory")
            inv_p.write_text(json.dumps(inv, indent=1))
            print(f"      [ok] wrote {filled:,} conditions paths -> {inv_p.relative_to(REPO)}")
    if not args.write:
        print("\n(report only — pass --write to apply)")
    return rc_


if __name__ == "__main__":
    raise SystemExit(main())
