"""CTT v2 — DERIVE the inline-OOD **demo** pool (never hand-edit it).

Authority, all three by filename (the advisor namespace has four collisions, so a bare
number is ambiguous — see `misc/ctt_v2_final/advisors/LEDGER.md`):

  misc/ctt_v2_final/advisors/A10_pool_drops_VERBATIM.md
      the STANDING campaign-wide exclusion, and its rule: *defects are dispositioned at the
      unit of consumption*, not of storage.
  misc/ctt_v2_final/advisors/A16_29_orphaned_s2a_clips_VERBATIM.md  (+ A17 affirmation)
      applied that rule to the rendered S2a consumption units.
  misc/ctt_v2_final/advisors/A18_28plus1_and_ood_demo_VERBATIM.md   decision 2
      applies it to the one consumption lane that was never enumerated as one: the inline-OOD
      **demo** pool.  80 -> 79.

WHAT THIS DOES AND — MORE IMPORTANTLY — WHAT IT MUST NOT DO
-----------------------------------------------------------
    demo_pool = PREREG_inline_ood_ops_s2a.json:clip_ids  MINUS  G
    G         = clip_ids  n  {stems consuming the excluded (clip, A) pair}

**The OP SET IS UNTOUCHED.**  `FilmBurn_bce3e2cb2d` remains one of the 8 pre-registered
inline-OOD ops; 9 of its 10 clips remain demo-eligible.  Replacing an op because of a
measured pixel property of one of its clips is exactly the post-draw curation
`A11_seven_open_items_VERBATIM.md` item 1 forbids, and the seed-42 draw's legitimacy rests on
that prohibition.  This script therefore **never** reads, ranks or re-draws `op_ids`.

Why the exclusion reaches this lane at all — it is NOT weight or adjudication contamination:
inline validation is generation at a training pause, so **no gradient flows from a demo**, and
per `A2_training_eval_VERBATIM.md` Q3 **inline scores never gate anything**.  The single live
channel is a **FALSE KILL**: A2 Q3's health-only kill criterion fires on "mechanically
degenerate inline output (constant frames) at two consecutive checks", and `s2_0818_c03`'s
first 18 frames are flat white (YMIN=YMAX=231) — as a *fixed* demo it could reproduce
constant-frame output at every check and kill a healthy run.

Per A2 Q3, the two FIXED OOD demo pairings are drawn from this pool and their clip ids are
recorded **before training launch**.

    python scripts/ctt_v2/derive_ood_demo_pool.py \
        --out misc/ctt_v2_final/DERIVED_inline_ood_demo_pool.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import root_common as rc  # noqa: E402

REPO = rc.REPO_ROOT
PREREG = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/misc/ctt_v2_final/"
              "PREREG_inline_ood_ops_s2a.json")
META_GLOB = "outputs/videos/ctt_v2_s2/full/meta/clips_shard*.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prereg", default=str(PREREG))
    ap.add_argument("--meta-glob", default=META_GLOB)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # ---- source 1: the standing exclusion, DERIVED from the adjudication sidecar ---------
    # never a literal clip name here; `root_common` raises rather than degrading to {}.
    excl = {(c, r) for c, roles in rc.ROLE_EXCLUSIONS.items() for r in roles}
    if not excl:
        raise SystemExit("[ood-demo] ROLE_EXCLUSIONS is VACUOUS — instrument failure, not "
                         "'nothing to exclude' (A16 keyed-join rule item 1)")

    # ---- source 2: the pre-registration (READ ONLY — the op set is never touched) --------
    prereg = json.loads(Path(args.prereg).read_text())
    clip_ids = list(prereg["clip_ids"])
    if len(clip_ids) != len(set(clip_ids)):
        raise SystemExit("[ood-demo] PREREG clip_ids are not distinct")

    # ---- the keyed join, with its universe ENUMERATED and both roles reported ------------
    files = sorted(glob.glob(args.meta_glob if Path(args.meta_glob).is_absolute()
                             else str(REPO / args.meta_glob)))
    if not files:
        raise SystemExit(f"[ood-demo] no meta shards matched {args.meta_glob!r}")
    n_rows = 0
    hits_as_a: dict[str, list[str]] = {}
    hits_as_b: dict[str, list[str]] = {}
    control_key = None          # positive control: a (clip, A) key KNOWN to be present
    control_hits: list[str] = []
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            n_rows += 1
            a, b = r["A"], r["B"]
            if control_key is None and not rc.role_excluded(a, "A"):
                control_key = a         # first non-excluded A endpoint we see
            if a == control_key:
                control_hits.append(r["stem"])
            if rc.role_excluded(a, "A"):
                hits_as_a.setdefault(f"{a}:A", []).append(r["stem"])
            if rc.role_excluded(b, "B"):
                hits_as_b.setdefault(f"{b}:B", []).append(r["stem"])

    if not control_hits:
        raise SystemExit("[ood-demo] the POSITIVE CONTROL returned nothing — the join path is "
                         "broken, so its zeros are not information (A16 keyed-join rule item 2)")

    consumers = sorted({s for v in hits_as_a.values() for s in v}
                       | {s for v in hits_as_b.values() for s in v})
    G = sorted(set(clip_ids) & set(consumers))
    pool = sorted(set(clip_ids) - set(G))

    out = {
        "what": "the inline-OOD DEMO pool — derived, never hand-edited",
        "authority": [
            "misc/ctt_v2_final/advisors/A10_pool_drops_VERBATIM.md "
            "(STANDING exclusion; defects are dispositioned at the unit of CONSUMPTION)",
            "misc/ctt_v2_final/advisors/A16_29_orphaned_s2a_clips_VERBATIM.md "
            "(RULING OF RECORD for the rendered-S2a consumption units)",
            "misc/ctt_v2_final/advisors/A17_29clip_affirmation_VERBATIM.md "
            "(INDEPENDENT AFFIRMATION)",
            "misc/ctt_v2_final/advisors/A18_28plus1_and_ood_demo_VERBATIM.md decision 2 "
            "(this lane: 80 -> 79, as a mechanical application of the standing exclusion)",
        ],
        "derivation": "clip_ids MINUS (clip_ids n consumers of the role-excluded (clip, role) "
                      "pair). Computed from two artefacts that pre-date any measurement: "
                      "ROLE_EXCLUSIONS (via POOL_DROPS_M3_ADJUDICATION.json) and "
                      "PREREG_inline_ood_ops_s2a.json. NOTHING here inspects a demo for "
                      "suitability, and no clip is chosen or rejected on a measured property.",
        "op_set_untouched": {
            "n_ops": len(prereg["op_ids"]),
            "op_ids": list(prereg["op_ids"]),
            "rule": "A11_seven_open_items_VERBATIM.md item 1 — the seed-42 op draw is NEVER "
                    "post-filtered. FilmBurn_bce3e2cb2d REMAINS an inline-OOD op; only one of "
                    "its ten clips leaves the DEMO lane. A re-draw would be the violation.",
        },
        "harm_model": {
            "weight_contamination": "none — inline validation is generation at a training "
                                    "pause; no gradient flows from a demo",
            "adjudication_contamination": "none by construction — A2_training_eval_VERBATIM.md "
                                          "Q3: inline scores never gate anything",
            "the_one_live_channel": "FALSE KILL — A2 Q3's health-only criterion kills on "
                                    "'mechanically degenerate inline output (constant frames) "
                                    "at two consecutive checks'; a flat-white fixed demo could "
                                    "reproduce constant-frame output at every check and kill a "
                                    "healthy run",
        },
        "sources": {
            "prereg": str(args.prereg),
            "prereg_sha256": rc.sha256_file(Path(args.prereg)),
            "sidecar": str(rc.POOL_DROPS),
            "sidecar_sha256": rc.sha256_file(rc.POOL_DROPS),
            "meta_glob": args.meta_glob,
            "n_shards": len(files),
        },
        "universe_enumerated": {
            "s2a_meta_rows_scanned": n_rows,
            "standing_role_exclusions": {c: list(r) for c, r in rc.ROLE_EXCLUSIONS.items()},
            "hits_as_A": {k: len(v) for k, v in sorted(hits_as_a.items())},
            "hits_as_B": {k: len(v) for k, v in sorted(hits_as_b.items())},
            "n_consumers_total": len(consumers),
            "positive_control": {
                "key": f"{control_key}:A",
                "n_hits": len(control_hits),
                "why": "a (clip, role) key KNOWN to be present, returned non-empty through the "
                       "IDENTICAL join path — so the zeros above are real zeros",
            },
        },
        "result": {
            "n_prereg_clip_ids": len(clip_ids),
            "G_excluded_from_the_demo_lane": G,
            "n_demo_pool": len(pool),
            "demo_pool": pool,
        },
        "next_action": "A2_training_eval_VERBATIM.md Q3 — the two FIXED OOD demo pairings are "
                       "drawn from this pool and their clip ids RECORDED BEFORE TRAINING LAUNCH.",
        "derived_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    rc.write_json(Path(args.out), out)
    print(f"[ood-demo] {n_rows} S2a meta rows scanned; "
          f"consumers of the excluded pair: {len(consumers)} "
          f"(as A {sum(len(v) for v in hits_as_a.values())}, "
          f"as B {sum(len(v) for v in hits_as_b.values())})")
    print(f"[ood-demo] positive control {control_key}:A -> {len(control_hits)} hits")
    print(f"[ood-demo] G = clip_ids n consumers = {G}")
    print(f"[ood-demo] DEMO POOL = {len(clip_ids)} - {len(G)} = {len(pool)} clips "
          f"-> {args.out}")
    print("[ood-demo] op set UNTOUCHED: "
          f"{len(prereg['op_ids'])} inline-OOD ops, FilmBurn_bce3e2cb2d still among them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
