"""CTT v2 — build a per-stratum inventory (the input contract of `assemble_root.py`).

An inventory is a pure description of WHAT a stratum contributes: its groups (the unit the
ring-offset pairing runs inside), the clips in each group, each clip's four source tensors
and its caption, and the content endpoints the clip was built from.  It contains no mix
weights, no exclusions and no pairing — those are `assemble_root.py`'s job.

    {
      "schema": "ctt_v2_stratum_inventory/1",
      "stratum": "S2a",
      "kind": "synthetic_op",
      "endpoint_disjointness": true,        # A3-F5b applies to this stratum
      "groups": {"<op_id>": {"class": null, "shader": "BookFlip", "sided": "two",
                             "clips": ["s2_0000_c00", ...]}},
      "clips":  {"<stem>": {"group": ..., "latents": ..., "cond_clean": ..., "conditions": ...,
                            "caption": "...", "endpoints": ["A", "B"]}},
      "provenance": {...}
    }

Three builders, one contract:
  s0     — the real corpus stratum, from `eval_ladder/train/inventory.json` + the frozen split
  s2meta — any stratum rendered by `experiments/exp_082_s2_humanvid/render_s2.py`
           (S2a today, S2b when it lands — identical `meta/clips_shard*.jsonl` schema)
  spec   — generic: a hand-written/generated group spec (S1, S4 when they land)

Usage
-----
    python scripts/ctt_v2/build_inventories.py s0 --out <path>
    python scripts/ctt_v2/build_inventories.py s2meta --stratum S2a \
        --meta-glob 'outputs/videos/ctt_v2_s2/full/meta/clips_shard*.jsonl' \
        --latents '<dir>/{clip}.pt' --cond-clean '<dir>/{clip}.pt' \
        --conditions '<dir>/{clip}.pt' --captions <json> --caption-key '{A}|{B}' --out <path>
    python scripts/ctt_v2/build_inventories.py spec --spec <json> --out <path> [source flags]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import root_common as rc  # noqa: E402

REPO = rc.REPO_ROOT
S0_INVENTORY = REPO / "eval_ladder/train/inventory.json"
S0_CAPTIONS = REPO / "eval_ladder/dataset/captions/dataset_captions.json"
S0_CONDITIONS = REPO / "eval_ladder/dataset/conditions"
S0_COND_CLEAN = REPO / "eval_ladder/dataset/cond_clean"


# --------------------------------------------------------------------------------------
def _fmt(tpl: str, **kw) -> str:
    try:
        return tpl.format(**kw)
    except KeyError as exc:
        raise SystemExit(f"template {tpl!r} uses unknown placeholder {exc}") from None


def _attach(inv: dict, args, require: bool) -> None:
    """Fill latents / cond_clean / conditions / caption from CLI templates.

    CAPTION LOOKUP IS (clip, role)-KEYED, HAS NO CROSS-ROLE FALLBACK (A10 `enforced_at[0]`),
    AND HARD-FAILS ON EVERY MISSING KEY IT HAS NOT BEEN GIVEN A DISPOSITION FOR.  Three
    rules, in this order:

    1. **A consumption hit carried by the STANDING `ROLE_EXCLUSIONS` is DROPPED-AND-RECORDED,
       not a crash.**  This used to `SystemExit` with *"fix the grid or the render, do not
       degrade"* — a crash written when a violation implied an upstream leak.  The
       disposition that message demanded has since been ruled, twice, on primary evidence:
       `misc/ctt_v2_final/advisors/A16_29_orphaned_s2a_clips_VERBATIM.md` (RULING OF RECORD)
       and `misc/ctt_v2_final/advisors/A17_29clip_affirmation_VERBATIM.md` (independent
       affirmation, same disposition, reached before it found A16).  Neither grid-fix nor
       re-render: **drop**.  The defect is in the pixels, not the caption gap — frames 0–17
       of the affected clips are flat white (YMIN=YMAX=231, probed directly) — so no caption,
       not even a truthful one, repairs them.  The drop set is computed here as
       `ROLE_EXCLUSIONS ∩ this inventory's consumed (clip, role) sources`; a hand-kept stem
       list would recreate the `INTENDED_WEIGHTS_PCT` landmine class.
       The entries stay in the inventory with `caption = None` and are removed at ASSEMBLY,
       where `assemble_root.apply_exclusions` records each one in
       `ROOT_MANIFEST.json:drops` under `role_scoped_caption_exclusion` /
       `role_scoped_prefix_condition` — that manifest record is the closing evidence the
       ruling asks for, which is why the drop is *recorded here* and *executed there*
       rather than silently vanishing from this file.
    2. **A missing NON-EXCLUDED key is still a hard crash** — unchanged.  A key that no
       standing exclusion accounts for is instrument failure, never a skipped row.
    3. **No cross-role fallback, ever.**  Nothing here ever consults another role or a
       clip-level key, and a dropped clip that *does* carry a description in the store is a
       hard error: it means something upstream invented the fallback A10 forbids.
    """
    caps = json.loads(Path(args.captions).read_text()) if args.captions else None
    if isinstance(caps, list):  # [{video, caption}] shape
        caps = {r["video"]: r["caption"] for r in caps}

    # ---- pass 1: record consumed (clip, role) sources and DERIVE the drop set -----------
    # `ROLE_EXCLUSIONS ∩ rendered meta`, at build time.  If the standing exclusion set is
    # empty this loop is vacuous and every missing key falls through to the pass-2 crash —
    # a silently-degraded sidecar therefore cannot buy a clip an exemption (and
    # `root_common` now raises rather than degrading in the first place).
    excluded: dict[str, list[str]] = {}
    for stem, c in inv["clips"].items():
        g = inv["groups"][c["group"]]
        # the (clip, role) descriptions this clip's caption will consume — recorded on the
        # entry so the root asserts do not have to re-derive them
        srcs = rc.caption_sources(c, g.get("sided", "two"), inv.get("kind", "synthetic_op"))
        c["caption_sources"] = [list(x) for x in srcs]
        reasons = []
        hits = sorted({f"{cl}:{ro}" for cl, ro in srcs if rc.role_excluded(cl, ro)})
        if hits:  # same reason vocabulary as assemble_root.apply_exclusions
            reasons.append("role_scoped_caption_exclusion:" + ",".join(hits))
        eps = list(c.get("endpoints") or [])
        if eps and rc.role_excluded(eps[0], "A"):
            # ...and the CONDITIONING channel of the same exclusion: the prefix anchor is
            # the A endpoint's frames 0-8 (A10 `enforced_at[2]`).
            reasons.append(f"role_scoped_prefix_condition:{eps[0]}:A")
        if reasons:
            excluded[stem] = reasons

    # ---- pass 2: resolve source templates and captions ---------------------------------
    missing, fallback_evidence = [], []
    for stem, c in inv["clips"].items():
        g = inv["groups"][c["group"]]
        eps = list(c["endpoints"])
        ctx = {"clip": stem, "group": c["group"], "class": g.get("class") or "",
               "shader": g.get("shader") or "",
               "A": (eps + ["", ""])[0], "B": (eps + ["", ""])[1]}
        for key, tpl in (("latents", args.latents), ("cond_clean", args.cond_clean),
                         ("conditions", args.conditions)):
            if tpl:
                p = Path(_fmt(tpl, **ctx))
                if not p.is_absolute():
                    p = REPO / p
                c[key] = str(p)
                if require and not p.exists():
                    missing.append(f"{key}:{p}")
        if caps is None:
            continue
        k = _fmt(args.caption_key, **ctx)
        if stem in excluded:
            # DROPPED (rule 1).  Its description does not exist BY ADJUDICATION, so its
            # absence is the exclusion working, not a missing source.  `caption` stays None
            # and no other key is consulted.
            if k in caps:
                fallback_evidence.append(f"{stem} ({k!r}): {excluded[stem]}")
            continue
        if k not in caps:
            missing.append(f"caption:{k}")
        else:
            c["caption"] = caps[k]

    if fallback_evidence:
        raise SystemExit(
            f"[inventory] {len(fallback_evidence)} ROLE-EXCLUDED clip(s) CARRY A CAPTION in "
            f"{args.captions} — a description exists for a (clip, role) that A10 withheld "
            f"({rc.POOL_DROPS.name}). Something upstream invented the cross-role fallback "
            f"A10 forbids: it would caption a blank-screen anchor with content it does not "
            f"show. Fix the caption source; do not relax this check:\n  "
            + "\n  ".join(fallback_evidence[:10]))
    if excluded:
        inv["role_scoped_exclusion_drops"] = {
            "authority": [
                "misc/ctt_v2_final/advisors/A16_29_orphaned_s2a_clips_VERBATIM.md "
                "(RULING OF RECORD — drop at assembly, derived, recorded)",
                "misc/ctt_v2_final/advisors/A17_29clip_affirmation_VERBATIM.md "
                "(INDEPENDENT AFFIRMATION — same disposition on independent evidence)",
            ],
            "derivation": "ROLE_EXCLUSIONS n this inventory's consumed (clip, role) "
                          "sources, computed at build time — never a hand-kept stem list",
            "standing_role_exclusions": {c: list(r) for c, r in rc.ROLE_EXCLUSIONS.items()},
            "sidecar": str(rc.POOL_DROPS),
            "n_clips": len(excluded),
            "clips": {s: excluded[s] for s in sorted(excluded)},
            "disposition": "kept in this file with caption=null; REMOVED AT ASSEMBLY by "
                           "assemble_root.apply_exclusions, which records each one in "
                           "ROOT_MANIFEST.json:drops under the same two reasons. A clip "
                           "whose whole GROUP is dropped first (holdout shader / inline-OOD "
                           "op / zs class) is removed by that group drop instead, and so "
                           "appears in `drops[*].groups`, not in `drops[*].clips`.",
        }
        print(f"[inventory] {len(excluded)} clip(s) consume a ROLE-EXCLUDED (clip, role) "
              f"description (A10; {rc.POOL_DROPS.name}) -> DROPPED AT ASSEMBLY per "
              f"A16_29_orphaned_s2a_clips_VERBATIM.md + A17_29clip_affirmation_VERBATIM.md; "
              f"recorded in `role_scoped_exclusion_drops`. No cross-role fallback was used.")
        for s in sorted(excluded)[:5]:
            print(f"        drop: {s}: {excluded[s]}")
    if missing:
        raise SystemExit(f"[inventory] {len(missing)} missing sources, first 10:\n  "
                         + "\n  ".join(missing[:10]))


def _finish(inv: dict, out: Path) -> None:
    n_pairs = sum(len(rc.ring_pairs(sorted(g["clips"]))) for g in inv["groups"].values())
    inv["counts"] = {
        "groups": len(inv["groups"]),
        "clips": len(inv["clips"]),
        "pairs_if_unfiltered": n_pairs,
    }
    rc.write_json(out, inv)
    print(f"[inventory] {inv['stratum']}: {len(inv['groups'])} groups, {len(inv['clips'])} clips, "
          f"{n_pairs} pairs (before exclusions) -> {out}")


# --------------------------------------------------------------------------------------
def build_s0(args) -> dict:
    prompts = rc._prompts()
    inv_src = rc.read_json(S0_INVENTORY)
    sided = prompts.sidedness()
    caps = {r["video"]: r["caption"] for r in rc.read_json(S0_CAPTIONS)}

    # class membership comes from the frozen split via prompts.clip_class(), never the name
    clips = sorted(inv_src["clips"]["ic_gen"])
    groups: dict[str, dict] = {}
    entries: dict[str, dict] = {}
    for clip in clips:
        cls = prompts.clip_class(clip)
        groups.setdefault(cls, {"class": cls, "shader": None,
                                "sided": sided[cls], "clips": []})["clips"].append(clip)
        lat = rc.resolve_repo_rel(inv_src["latents"][clip])
        entries[clip] = {
            "group": cls,
            "latents": str(lat),
            "cond_clean": str(S0_COND_CLEAN / cls / f"{clip}.pt"),
            "conditions": str(S0_CONDITIONS / cls / f"{clip}.pt"),
            "caption": caps[f"{cls}/{clip}.mp4"],
            "endpoints": [clip],
        }
    for g in groups.values():
        g["clips"] = sorted(g["clips"])

    missing = [f"{k}:{v[k]}" for v in entries.values() for k in
               ("latents", "cond_clean", "conditions") if not Path(v[k]).exists()]
    if missing and not args.no_require_sources:
        raise SystemExit(f"[inventory] S0: {len(missing)} missing sources, first 10:\n  "
                         + "\n  ".join(missing[:10]))
    return {
        "schema": rc.INVENTORY_SCHEMA,
        "stratum": "S0",
        "kind": "corpus",
        "endpoint_disjointness": False,   # S0 IS the eval endpoint bank by design (seen cells)
        "groups": dict(sorted(groups.items())),
        "clips": dict(sorted(entries.items())),
        "provenance": {
            "roster": str(S0_INVENTORY.relative_to(REPO)),
            "split": str(rc.SPLIT_PATH.relative_to(REPO)),
            "captions": str(S0_CAPTIONS),
            "captions_sha256": rc.sha256_file(S0_CAPTIONS),
            "sidedness": "eval_ladder/prompts.py:sidedness() (corpus_manifest x class_axes_v2)",
        },
    }


def build_s2meta(args) -> dict:
    files = sorted(glob.glob(args.meta_glob if Path(args.meta_glob).is_absolute()
                             else str(REPO / args.meta_glob)))
    if not files:
        raise SystemExit(f"[inventory] no meta shards matched {args.meta_glob!r}")
    groups: dict[str, dict] = {}
    entries: dict[str, dict] = {}
    for f in files:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            gid = r["op_id"]
            groups.setdefault(gid, {"class": None, "shader": r["shader"],
                                    "sided": args.sided, "clips": []})["clips"].append(r["stem"])
            entries[r["stem"]] = {"group": gid, "latents": None, "cond_clean": None,
                                  "conditions": None, "caption": None,
                                  "endpoints": [r["A"], r["B"]]}
    for g in groups.values():
        g["clips"] = sorted(g["clips"])
    inv = {
        "schema": rc.INVENTORY_SCHEMA,
        "stratum": args.stratum,
        "kind": "synthetic_op",
        "endpoint_disjointness": True,
        "groups": dict(sorted(groups.items())),
        "clips": dict(sorted(entries.items())),
        "provenance": {"meta_glob": args.meta_glob, "n_shards": len(files),
                       "sided": args.sided,
                       "sided_authority": "DOSSIER 1.5 — S2 is a true A->B pair, hence two-sided"},
    }
    _attach(inv, args, require=not args.no_require_sources)
    return inv


def build_spec(args) -> dict:
    spec = rc.read_json(args.spec)
    groups = {gid: {"class": g.get("class"), "shader": g.get("shader"),
                    "sided": g.get("sided", args.sided), "clips": sorted(g["clips"])}
              for gid, g in spec["groups"].items()}
    endpoints = spec.get("endpoints", {})
    entries = {}
    for gid, g in groups.items():
        for clip in g["clips"]:
            entries[clip] = {"group": gid, "latents": None, "cond_clean": None,
                             "conditions": None, "caption": None,
                             "endpoints": list(endpoints.get(clip, []))}
    inv = {
        "schema": rc.INVENTORY_SCHEMA,
        "stratum": spec.get("stratum", args.stratum),
        "kind": spec.get("kind", "synthetic_op"),
        "endpoint_disjointness": bool(spec.get("endpoint_disjointness", True)),
        "groups": dict(sorted(groups.items())),
        "clips": dict(sorted(entries.items())),
        "provenance": {"spec": str(args.spec), "spec_sha256": rc.sha256_file(args.spec)}
        | spec.get("provenance", {}),
    }
    _attach(inv, args, require=not args.no_require_sources)
    return inv


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_stratum=True):
        if need_stratum:
            p.add_argument("--stratum", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--latents", help="path template, placeholders {clip}{group}{class}{A}{B}")
        p.add_argument("--cond-clean", help="path template")
        p.add_argument("--conditions", help="path template")
        p.add_argument("--captions", help="json: {key: caption} or [{video, caption}]")
        p.add_argument("--caption-key", default="{clip}", help="caption-store key template")
        p.add_argument("--sided", default="two", choices=["one", "two"])
        p.add_argument("--no-require-sources", action="store_true",
                       help="allow absent source tensors (inventory becomes non-assemblable)")

    p0 = sub.add_parser("s0", help="the real corpus stratum")
    p0.add_argument("--out", required=True)
    p0.add_argument("--no-require-sources", action="store_true")

    p1 = sub.add_parser("s2meta", help="a stratum rendered by render_s2.py")
    common(p1)
    p1.add_argument("--meta-glob", required=True)

    p2 = sub.add_parser("spec", help="generic group spec (S1 / S4)")
    common(p2, need_stratum=False)
    p2.add_argument("--stratum", default=None)
    p2.add_argument("--spec", required=True)

    args = ap.parse_args()
    inv = {"s0": build_s0, "s2meta": build_s2meta, "spec": build_spec}[args.cmd](args)
    _finish(inv, Path(args.out))


if __name__ == "__main__":
    main()
