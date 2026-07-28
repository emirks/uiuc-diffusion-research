#!/usr/bin/env python
"""CTT v2 — the CAPTION-STORE assert (A11 item 5, machine check 2).

A11 item 5 keeps the role-scoping and rejects the whole-clip drop, but requires the rule to
be MACHINE-CHECKED rather than merely written down:

    `descriptions.json` contains NO A-role description for `openvid_T1MiFx98l3g_0_50to156`
    — presence is a hard FAIL — and DOES contain its B-role description (guards against an
    over-broad skip silently dropping the legitimate role).

Both halves matter and they fail in opposite directions:

* an EXCLUDED (clip, role) description present  => the blank-white A-anchor got captioned,
  which is the defect the adjudication exists to remove;
* the COMPLEMENTARY role missing               => the skip was over-broad and quietly cost
  the campaign 10 perfectly good rendered S2b clips, which is the outcome A11 explicitly
  rejected ("Whole-clip drop REJECTED").

The exclusion list is DERIVED from `POOL_DROPS_M3_ADJUDICATION.json` through
`root_common.load_caption_store_exclusions()` — the same single source the caption pipeline
skips on (`generate_descriptions.apply_role_scoped_exclusions`) and the same one
`assert_root.py`'s A12 enforces against the assembled root.  There is no hand-kept list
anywhere in the chain, so the three consumers cannot drift apart.

An ABSENT adjudication file is a hard FAIL, never an empty exclusion: the content pool was
deliberately left byte-unchanged, so that file is the only carrier of the instruction and a
silently-vacuous exclusion is exactly the defect class this campaign keeps meeting.

    python scripts/ctt_v2/captions/assert_caption_store.py --store <descriptions.json>

Exit code 0 = every check passed.  Anything else is a hard failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import root_common as rc  # noqa: E402

#: A store keyed `{clip_id: {"A": ..., "B": ...}}` (the production shape written by
#: `generate_descriptions.run()`).  Both roles are expected for a two-sided clip.
ALL_ROLES = ("A", "B")


def check_store(store: dict, pool_drops: Path | None = None) -> list[dict]:
    """Return [{name, ok, detail, offenders}] — every record is a HARD check."""
    role, clip_level, prov = rc.load_caption_store_exclusions(pool_drops)
    out: list[dict] = []

    if prov.get("error"):
        return [{"name": "S1_adjudication_present", "ok": False,
                 "detail": f"{prov['file']}: {prov['error']}", "offenders": []}]
    out.append({"name": "S1_adjudication_present", "ok": True,
                "detail": f"{prov['file']} sha {prov['sha256'][:12]} — "
                          f"{sum(len(v) for v in role.values())} role-scoped + "
                          f"{len(clip_level)} clip-level exclusions on record",
                "offenders": []})

    # ---- excluded (clip, role) descriptions must be ABSENT -----------------------------
    present = []
    for clip, roles in sorted(role.items()):
        for r in sorted(roles):
            if r in (store.get(clip) or {}):
                present.append(f"{clip}:{r} — EXCLUDED description is present in the store")
    for clip in sorted(clip_level):
        if clip in store:
            present.append(f"{clip}:* — clip-level-excluded clip is present in the store")
    out.append({"name": "S2_excluded_descriptions_absent", "ok": not present,
                "detail": "no excluded (clip, role) description is in the store"
                          if not present else f"{len(present)} excluded descriptions present",
                "offenders": present})

    # ---- the COMPLEMENTARY roles must still be THERE ------------------------------------
    # Only checked for role-scoped clips: a clip-level exclusion legitimately removes all
    # roles, and a clip the store was never asked to cover is not this assert's business —
    # so the guard fires only once the clip appears in the store at all.
    missing = []
    for clip, roles in sorted(role.items()):
        if clip not in store:
            missing.append(f"{clip} — role-scoped clip is absent from the store ENTIRELY; "
                           f"the skip was over-broad (A11 item 5 kept its {'/'.join(sorted(set(ALL_ROLES) - set(roles)))}-role)")
            continue
        for r in sorted(set(ALL_ROLES) - set(roles)):
            if r not in store[clip]:
                missing.append(f"{clip}:{r} — legitimate role is MISSING; the skip was "
                               f"over-broad (only {'/'.join(sorted(roles))} is excluded)")
    out.append({"name": "S3_complementary_roles_present", "ok": not missing,
                "detail": "every role-scoped clip still carries its legitimate role(s)"
                          if not missing else f"{len(missing)} legitimate roles missing",
                "offenders": missing})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="path to descriptions.json")
    ap.add_argument("--pool-drops", help="override POOL_DROPS_M3_ADJUDICATION.json")
    ap.add_argument("--report", help="write a JSON report here")
    args = ap.parse_args()

    p = Path(args.store)
    if not p.exists():
        print(f"[FAIL] caption store {p} is absent")
        return 2
    store = json.loads(p.read_text())

    results = check_store(store, Path(args.pool_drops) if args.pool_drops else None)
    for rec in results:
        print(f"[{'PASS' if rec['ok'] else 'FAIL'}] {rec['name']}: {rec['detail']}")
        for o in rec["offenders"][:10]:
            print(f"        - {o}")

    failed = [r["name"] for r in results if not r["ok"]]
    if args.report:
        rc.write_json(args.report, {"store": str(p), "n_clips": len(store),
                                    "failed": failed, "results": results})
    if failed:
        print(f"\n[caption-store] HARD FAILURES: {failed}")
        return 1
    print(f"\n[caption-store] all {len(results)} checks passed over {len(store)} clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
