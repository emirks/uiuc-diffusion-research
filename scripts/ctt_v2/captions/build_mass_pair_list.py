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

THREE sources are understood, and an unrecognised file is a HARD STOP rather than a
silently-empty contribution:

  * S2 union plan   -- `pairs: [{A, B, ...}]`          (exp_082 PLAN_S2_UNION.json)
  * S1 grid         -- `rows: [{endpoint_a, endpoint_b, sided, ...}]`  (S1_GRID.json)
  * S2a rendered    -- JSONL, one record per rendered clip, keys `A` / `B`
                      (outputs/videos/ctt_v2_s2/full/meta/clips_shard*.jsonl), via
                      `--s2a-meta`

⚠ THE S2a SCHEMA TRAP (this is why `--s2a-meta` is a separate, asserted code path).
S2a's endpoints live in its *rendered metadata*, not in a plan file, and that metadata
names them **`A` / `B`** -- NOT `endpoint_a` / `endpoint_b`.  A strict `endpoint_a`
lookup over these shards returns an EMPTY set, and the resulting bug does not look like
a bug: it looks like the reassuring statement "S2a needs no descriptions."  That failure
mode cost this campaign 36 absent (clip, role) pairs, 26 of them at risk of never being
generated at all.

Per A11's Derived-Constant Rule + the ANSI/positive-presence rule, the fix is not "read
the right key" -- it is to make the absence claim UNFALSIFIABLE-BY-SILENCE:

  1. POSITIVE PRESENCE: every record must positively carry non-empty `A` and `B`; a
     record missing either is a hard instrument failure, never a skipped row.
  2. NON-EMPTY: the extracted S2a set must be non-empty.
  3. DERIVED CONSTANT: it must equal `S2A_EXPECTED_CLIP_ROLES`, independently recomputed
     here from the shards; a mismatch raises SPEC-CONSTANT-MISMATCH, which ESCALATES and
     never executes a fallback branch (the spec/reality disagreement is ambiguous).
  4. TRAP WITNESS: the vacuous `endpoint_a` lookup is executed on purpose and reported,
     so the record shows the empty set is a property of the schema, not of the need.

S4 is NOT a source here and must never become one: this list is the S0/S1/S2 caption
requirement only.  S4 carries its own (deferred) caption lane.

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
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caption_common import STRIPS_INDEX  # noqa: E402
from root_common import load_caption_store_exclusions, sha256_file  # noqa: E402

#: A11 Derived-Constant Rule: pinned literal, INDEPENDENTLY RECOMPUTED below from the
#: shards themselves.  Derivation: |{(rec.A, "A")} u {(rec.B, "B")}| over every record of
#: outputs/videos/ctt_v2_s2/full/meta/clips_shard*.jsonl (7,990 rendered S2a clips, the
#: DOSSIER §17.1-corrected production count) = 454.  Mismatch => SPEC-CONSTANT-MISMATCH.
S2A_EXPECTED_CLIP_ROLES = 454


class SpecConstantMismatch(SystemExit):
    """A11: a spec/reality disagreement. ESCALATES; never selects a fallback branch."""


def extract_s2a(paths: list[Path]):
    """S2a rendered metadata (JSONL, keys `A`/`B`) -> the same (clip, role, why) shape.

    Every guard in the module docstring's 1-4 is enforced here.  Returns
    (want, counts, per_shard_shas, trap_witness).
    """
    want: list[tuple[str, str, str]] = []
    n_records = 0
    n_with_endpoint_a = 0          # the TRAP WITNESS -- expected 0
    per_shard: list[dict] = []

    for p in sorted(paths):
        rows = 0
        for lineno, line in enumerate(p.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows += 1
            n_records += 1
            # (1) POSITIVE PRESENCE -- a missing key is instrument failure, not a skip.
            a, b = rec.get("A"), rec.get("B")
            if not a or not b:
                raise SystemExit(
                    f"{p}:{lineno}: S2a record {rec.get('stem')!r} is missing `A` and/or "
                    f"`B`. Refusing to skip it: an S2a record without endpoints means the "
                    f"render metadata schema changed, and a skipped row here is exactly "
                    f"the silent under-count this loader exists to prevent.")
            if rec.get("endpoint_a"):
                n_with_endpoint_a += 1
            want.append((a, "A", "S2a:A-endpoint"))
            want.append((b, "B", "S2a:B-endpoint"))
        per_shard.append({"file": str(p), "sha256": sha256_file(p), "records": rows})

    distinct = {(c, r) for c, r, _ in want}

    # (2) NON-EMPTY.
    if not distinct:
        raise SystemExit(
            "S2a contributed ZERO (clip, role) pairs. This is the schema trap, not a "
            "real absence -- S2a's rendered metadata uses `A`/`B`. Refusing to proceed.")

    # (3) DERIVED CONSTANT -- escalate, never fall back.
    if len(distinct) != S2A_EXPECTED_CLIP_ROLES:
        raise SpecConstantMismatch(
            f"SPEC-CONSTANT-MISMATCH (A11 Derived-Constant Rule): S2a yielded "
            f"{len(distinct)} distinct (clip, role) pairs from {n_records} records across "
            f"{len(per_shard)} shard(s); the pinned derivation says "
            f"{S2A_EXPECTED_CLIP_ROLES}. This is ambiguous between changed data and a "
            f"changed spec, so NO fallback branch runs. ESCALATE to advisor review: "
            f"re-derive the constant, or explain the data change, before this gate runs.")

    trap = {
        "records_with_endpoint_a_key": n_with_endpoint_a,
        "strict_endpoint_a_lookup_would_yield": 0 if n_with_endpoint_a == 0 else None,
        "note": ("Executed on purpose. S2a's schema exposes endpoints as `A`/`B`, so a "
                 "strict `endpoint_a` lookup yields an EMPTY set and misreads as "
                 "'S2a needs nothing'. This witness records that the empty set is a "
                 "property of the schema, not of the requirement."),
    }
    counts = {"records": n_records, "shards": len(per_shard)}
    print(f"[plan] S2a rendered meta: {len(per_shard)} shard(s), {n_records} records -> "
          f"{len(want)} requests, {len(distinct)} distinct (clip, role) "
          f"[derived-constant {S2A_EXPECTED_CLIP_ROLES} OK]")
    print(f"[plan] S2a TRAP WITNESS: records carrying an `endpoint_a` key = "
          f"{n_with_endpoint_a} (a strict endpoint_a loader would have contributed 0)")
    return want, counts, per_shard, trap


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
    ap.add_argument("--s2a-meta", nargs="+", required=True,
                    help="S2a rendered-metadata JSONL shards (keys `A`/`B`), e.g. "
                         "outputs/videos/ctt_v2_s2/full/meta/clips_shard*.jsonl. REQUIRED: "
                         "S2a's endpoints exist nowhere else, and omitting them silently "
                         "under-counts the requirement by 36 (clip, role) pairs.")
    ap.add_argument("--exclusions", default=None,
                    help="POOL_DROPS_M3_ADJUDICATION.json (default: root_common.POOL_DROPS)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    index = json.loads(STRIPS_INDEX.read_text())

    # ---- gather ---------------------------------------------------------
    seen: dict[tuple[str, str], set[str]] = {}
    provenance = []

    # --- source 3: S2a rendered metadata (the previously-MISSING source) ---
    s2a_paths = [Path(p) for p in a.s2a_meta]
    for p in s2a_paths:
        if not p.exists():
            raise SystemExit(f"--s2a-meta {p}: does not exist")
    s2a_want, s2a_counts, s2a_shards, s2a_trap = extract_s2a(s2a_paths)
    for clip, role, why in s2a_want:
        seen.setdefault((clip, role), set()).add(why)
    # One stable sha for the whole S2a source: sha256 over the sorted per-shard shas.
    s2a_rollup = hashlib.sha256(
        "\n".join(f"{Path(s['file']).name} {s['sha256']}" for s in s2a_shards).encode()
    ).hexdigest()
    provenance.append({
        "file": f"{len(s2a_shards)} shard(s): {Path(s2a_shards[0]['file']).parent}"
                f"/clips_shard*.jsonl",
        "sha256": s2a_rollup,
        "sha256_kind": "rollup: sha256 over sorted '<shard name> <shard sha256>' lines",
        "schema": "s2a_rendered_meta_jsonl",
        "counts": s2a_counts,
        "clip_role_requests": len(s2a_want),
        "distinct_clip_roles": len({(c, r) for c, r, _ in s2a_want}),
        "derived_constant": {"name": "S2A_EXPECTED_CLIP_ROLES",
                             "value": S2A_EXPECTED_CLIP_ROLES, "recomputed": True},
        "trap_witness": s2a_trap,
        "shards": s2a_shards,
    })

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
    # ---- all THREE sources must have contributed -------------------------
    schemas = {p["schema"] for p in provenance}
    required = {"s2a_rendered_meta_jsonl", "s2_union_pairs", "s1_grid_rows"}
    if not required <= schemas:
        raise SystemExit(
            f"the requirement needs all three sources; missing {sorted(required - schemas)}. "
            f"Got {sorted(schemas)}. A two-source list under-counts and is not the "
            f"requirement.")
    for p in provenance:
        if p["distinct_clip_roles"] == 0:
            raise SystemExit(f"source {p['file']} contributed ZERO pairs -- refusing a "
                             f"silently-vacuous source (see the S2a schema trap).")

    by_source = {p["schema"]: p["distinct_clip_roles"] for p in provenance}

    manifest = {
        "built_from": provenance,
        "source_shas": {p["schema"]: p["sha256"] for p in provenance},
        "distinct_clip_roles_by_source": by_source,
        "union_is_the_requirement": len(seen),
        "s4": {"status": "OUT OF SCOPE -- deferred by owner",
               "sources_read": "S2a + S2b + S1 only; no S4 plan is read by this builder",
               "s4_pairs_in_this_list": 0},
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
