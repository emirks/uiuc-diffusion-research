#!/usr/bin/env python
"""Pinned grid(s) -> the (clip, role) description list for the CTT v2 mass caption run.

A4 Q7.3: caption only the clips the pinned grids ACTUALLY use.  A description is
per-(clip, role), never per-sample, so a clip used as the A endpoint of 40 rows still
gets exactly one A-role description; the same clip used as a B endpoint elsewhere gets
one B-role description as well.  Roles are anchored to frames, not to grid position:

    A-role = frames 0-8      (the clip's opening anchor)
    B-role = frames 112-120  (the clip's closing anchor)

Caption grammar (DOSSIER):  one-sided `{S1}. sksz.`  /  two-sided `{S1}. sksz. {S2}.`
so a one-sided row consumes only its A endpoint's A-role description, and a two-sided
row consumes the A endpoint's A-role plus the B endpoint's B-role.

TWO grid schemas are understood, and an unrecognised file is a HARD STOP rather than a
silently-empty contribution:

  * S2 union plan   -- `pairs: [{A, B, ...}]`          (exp_082 PLAN_S2_UNION.json)
  * S1 grid         -- `rows: [{endpoint_a, endpoint_b, sided, ...}]`  (S1_GRID.json)

THE ROLE-SCOPED EXCLUSION IS DERIVED, NEVER HAND-KEPT.  It comes from
`data/processed/ctt_v2_strata/POOL_DROPS_M3_ADJUDICATION.json` through the SAME loader
`generate_descriptions.py` and `assert_root.py` use (`root_common.load_caption_store_
exclusions`), so the three consumption channels cannot drift.  An absent adjudication
file is a hard stop: a silently-vacuous exclusion is exactly the defect the machine
check exists to catch.

Usage
-----
  PY=/projects/illinois/eng/cs/jrehg/users/emirkisa/envs/diffusion/bin/python
  $PY build_mass_pair_list.py --plan <PLAN_S2_UNION.json> [<S1_GRID.json> ...] \
      --out <dir>/mass_pairs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caption_common import STRIPS_INDEX  # noqa: E402
from root_common import load_caption_store_exclusions, sha256_file  # noqa: E402


def extract(plan_path: Path):
    """-> (list[(clip_id, role, why)], schema_name, counts).  Unknown schema = hard stop."""
    rec = json.loads(plan_path.read_text())
    want: list[tuple[str, str, str]] = []

    if isinstance(rec, dict) and isinstance(rec.get("pairs"), list):
        for p in rec["pairs"]:
            a, b = p.get("A"), p.get("B")
            if not a or not b:
                raise SystemExit(f"{plan_path}: pair {p.get('pair_id')} missing A or B")
            want.append((a, "A", "S2:A-endpoint"))
            want.append((b, "B", "S2:B-endpoint"))
        return want, "s2_union_pairs", {"rows": len(rec["pairs"])}

    if isinstance(rec, dict) and isinstance(rec.get("rows"), list):
        one = two = 0
        for r in rec["rows"]:
            a = r.get("endpoint_a")
            if not a:
                raise SystemExit(f"{plan_path}: row {r.get('row_id')} has no endpoint_a")
            want.append((a, "A", "S1:endpoint_a"))
            b, sided = r.get("endpoint_b"), r.get("sided")
            if sided == "two":
                if not b:
                    raise SystemExit(
                        f"{plan_path}: row {r.get('row_id')} is two-sided but has no "
                        f"endpoint_b — the `{{S1}}. sksz. {{S2}}.` caption cannot be built")
                want.append((b, "B", "S1:endpoint_b"))
                two += 1
            else:
                one += 1
        return want, "s1_grid_rows", {"rows": len(rec["rows"]),
                                      "one_sided": one, "two_sided": two}

    raise SystemExit(
        f"{plan_path}: unrecognised grid schema (no `pairs` and no `rows` list). Refusing "
        f"to contribute zero pairs silently — add the schema to extract() explicitly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", nargs="+", required=True,
                    help="one or more PINNED grid files (S2 union plan and/or S1 grid)")
    ap.add_argument("--exclusions", default=None,
                    help="POOL_DROPS_M3_ADJUDICATION.json (default: root_common.POOL_DROPS)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    index = json.loads(STRIPS_INDEX.read_text())

    # ---- gather ---------------------------------------------------------
    seen: dict[tuple[str, str], set[str]] = {}
    provenance = []
    for pth in a.plan:
        p = Path(pth)
        if not p.exists():
            raise SystemExit(f"--plan {p}: does not exist")
        want, schema, counts = extract(p)
        for clip, role, why in want:
            seen.setdefault((clip, role), set()).add(why)
        provenance.append({"file": str(p), "sha256": sha256_file(p), "schema": schema,
                           "counts": counts, "clip_role_requests": len(want),
                           "distinct_clip_roles": len({(c, r) for c, r, _ in want})})
        print(f"[plan] {p.name}: schema={schema} {counts} -> {len(want)} requests, "
              f"{len({(c, r) for c, r, _ in want})} distinct (clip, role)")

    # ---- media must exist ----------------------------------------------
    missing = sorted({c for c, r in seen if c not in index})
    if missing:
        raise SystemExit(f"{len(missing)} grid clip(s) absent from the caption-strip index "
                         f"{STRIPS_INDEX}: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    no_anchor = sorted(f"{c}|{r}" for c, r in seen if not index[c].get(f"{r}_video"))
    if no_anchor:
        raise SystemExit(f"{len(no_anchor)} (clip, role) have no anchor video: {no_anchor[:10]}")

    # ---- DERIVED role-scoped exclusions --------------------------------
    role_x, clip_x, prov = load_caption_store_exclusions(
        Path(a.exclusions) if a.exclusions else None)
    if prov.get("error"):
        raise SystemExit(f"[exclusions] {prov['file']}: {prov['error']}")

    kept, skipped = [], []
    for (clip, role) in sorted(seen):
        if clip in clip_x or role in role_x.get(clip, ()):
            skipped.append([clip, role])
        else:
            kept.append([clip, role])

    print(f"[exclusions] {prov['file']} sha {prov['sha256'][:12]} "
          f"({prov.get('verdict')}): skipped {len(skipped)} -> {skipped}")

    # An exclusion that matches nothing is a silent no-op — say so loudly.
    for clip, roles in role_x.items():
        for r in roles:
            if [clip, r] not in skipped:
                print(f"[exclusions] NOTE: role-scoped exclusion {clip}|{r} matched no "
                      f"requested pair (the grids do not use it in that role).")

    # ---- write ----------------------------------------------------------
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, indent=1))

    by_role = {"A": sum(1 for c, r in kept if r == "A"), "B": sum(1 for c, r in kept if r == "B")}
    by_bank: dict[str, int] = {}
    for c, r in kept:
        by_bank[index[c]["bank"]] = by_bank.get(index[c]["bank"], 0) + 1
    manifest = {
        "built_from": provenance,
        "strips_index": str(STRIPS_INDEX),
        "exclusions": {**prov, "skipped_pairs": skipped},
        "counts": {"distinct_clip_roles_requested": len(seen), "excluded": len(skipped),
                   "descriptions_to_generate": len(kept), "by_role": by_role,
                   "by_bank": by_bank,
                   "distinct_clips": len({c for c, _ in kept})},
        "out": str(out),
    }
    mpath = out.with_name(out.stem + "_manifest.json")
    mpath.write_text(json.dumps(manifest, indent=1))

    print(f"\n{len(kept)} descriptions to generate "
          f"(A {by_role['A']} / B {by_role['B']}; {len({c for c, _ in kept})} distinct clips)")
    print(f"  by bank: {by_bank}")
    print(f"  -> {out}\n  -> {mpath}")


if __name__ == "__main__":
    main()
