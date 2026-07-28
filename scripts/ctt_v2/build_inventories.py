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


def _attach(inv: dict, args, require: bool) -> dict:
    """Fill latents / cond_clean / conditions / caption from CLI templates.

    CAPTION LOOKUP IS (clip, role)-KEYED, HARD-FAILS ON MISSING, AND HAS NO CROSS-ROLE
    FALLBACK (A10 `enforced_at[0]`).  A missing key is an error, never a fallback to the
    other role or to a clip-level key.

    **A16 — role-scoped consumption hits are DROPPED AND RECORDED, not a crash.**
    This function used to `SystemExit` on any consumption of a role-excluded (clip, role)
    with *"fix the grid or the render, do not degrade"*.  That crash was written when a
    violation implied an upstream leak; A16 supplied the disposition its own message
    demanded — **neither grid-fix nor re-render: drop.**  29 already-rendered, gate-accepted
    S2a rows consume `openvid_T1MiFx98l3g_0_50to156` as their A endpoint, whose A-anchor is
    verified flat white (frames 0-17, YMIN=YMAX=231), so no truthful caption exists for it.
    They are dropped here, with the SAME reasons vocabulary the assembler uses
    (`role_scoped_caption_exclusion` / `role_scoped_prefix_condition`), and the record is
    propagated into `ROOT_MANIFEST.json`'s drop record by `assemble_root.py`.

    What is still a HARD CRASH, deliberately:
      * a missing caption key for a clip that is **not** carried by a standing exclusion —
        that is the converse defect (a description that should exist and does not), and it
        must never degrade into a silent drop;
      * a role-excluded description that is actually PRESENT in the store — the only way a
        cross-role fallback could ever succeed, so it is refused at the door;
      * a vacuous caption join, and any lookup key whose SHAPE disagrees with the store
        (A16 items 1 and 4).

    The drop set is DERIVED as `ROLE_EXCLUSIONS ∩ (what this stratum actually consumes)`.
    A hand-kept list of 29 stems would recreate the `INTENDED_WEIGHTS_PCT` landmine class:
    a recorded constant that silently stops matching reality.
    """
    # The drop set is derived from the standing exclusion, so a VACUOUS exclusion would
    # silently drop nothing and certify the result (A16 item 1; see require_role_exclusions).
    rc.require_role_exclusions("build_inventories._attach")
    caps = json.loads(Path(args.captions).read_text()) if args.captions else None
    if isinstance(caps, list):  # [{video, caption}] shape
        caps = {r["video"]: r["caption"] for r in caps}
    store = None
    if caps is not None:
        # A16 item 4 — key shape is validated against the store's self-declaration (here the
        # `--caption-key` template IS the declaration) and against the store's own keys,
        # before any result is interpreted.  A wrong template used to produce N "missing"
        # entries; now it names the shape mismatch.
        store = rc.KeyedStore(caps, name=f"caption store {args.captions}",
                              keying=f"'{args.caption_key}'")
    missing, drops, present_excluded = [], [], []
    wanted_keys = []
    for stem, c in inv["clips"].items():
        g = inv["groups"][c["group"]]
        eps = list(c["endpoints"])
        ctx = {"clip": stem, "group": c["group"], "class": g.get("class") or "",
               "shader": g.get("shader") or "",
               "A": (eps + ["", ""])[0], "B": (eps + ["", ""])[1]}
        # the (clip, role) descriptions this clip's caption will consume — recorded on the
        # entry so the root asserts do not have to re-derive them
        srcs = rc.caption_sources(c, g.get("sided", "two"), inv.get("kind", "synthetic_op"))
        c["caption_sources"] = [list(x) for x in srcs]

        # ---- A16: the standing exclusion, intersected with what this clip consumes -------
        reasons = []
        hits = [f"{cl}:{role}" for cl, role in srcs if rc.role_excluded(cl, role)]
        if hits:
            reasons.append("role_scoped_caption_exclusion:" + ",".join(sorted(set(hits))))
        if eps and rc.role_excluded(eps[0], "A"):
            reasons.append(f"role_scoped_prefix_condition:{eps[0]}:A")
        if reasons:
            # NO CROSS-ROLE FALLBACK, and this is where a fabricated one would show up: if a
            # caption EXISTS for a clip whose (clip, role) description is role-excluded, then
            # something upstream substituted text — the B-role description, or another clip's
            # — for nine frames of blank white.  That is the exact failure A10's exclusion
            # exists to prevent, so it is a hard crash, never a quiet consumption.
            if store is not None and store.has(_fmt(args.caption_key, **ctx)):
                present_excluded.append(
                    f"{stem}: consumes role-excluded {sorted(set(hits))} yet the store HAS a "
                    f"caption under key {_fmt(args.caption_key, **ctx)!r}")
            drops.append({"clip": stem, "group": c["group"], "stratum": inv["stratum"],
                          "reasons": reasons, "dropped_at": "inventory_build",
                          "authority": "A16 (drop) on A10 (role-scoped exclusion)"})
            continue

        for key, tpl in (("latents", args.latents), ("cond_clean", args.cond_clean),
                         ("conditions", args.conditions)):
            if tpl:
                p = Path(_fmt(tpl, **ctx))
                if not p.is_absolute():
                    p = REPO / p
                # REPO is `parents[1]` of this file, so building from a worktree would bake
                # that ephemeral checkout into every recorded source path.
                c[key] = rc.canonical_source(p) if p.exists() else str(p)
                if require and not p.exists():
                    missing.append(f"{key}:{p}")
        if store is not None:
            k = _fmt(args.caption_key, **ctx)
            wanted_keys.append(k)
            try:
                c["caption"] = store.require(k)
            except KeyError:
                # NOT carried by a standing exclusion => the converse defect => CRASH
                missing.append(f"caption:{k}")
    if present_excluded:
        raise SystemExit(
            f"[inventory] {len(present_excluded)} clip(s) consume a role-EXCLUDED "
            f"(clip, role) description (A10; {rc.POOL_DROPS.name}) AND have a caption in the "
            f"store — a cross-role fallback was invented upstream. Refusing to consume it; "
            f"fix the composer, do not degrade:\n  " + "\n  ".join(present_excluded[:10]))
    if missing:
        raise SystemExit(f"[inventory] {len(missing)} missing sources, first 10:\n  "
                         + "\n  ".join(missing[:10]))
    if store is not None:
        # A16 item 1 — an empty join result is a failure, never information
        store.join_nonvacuous(wanted_keys, name=f"{inv['stratum']} captions x store")

    # ---- remove the dropped stems, and record the derivation -----------------------------
    dropped_stems = {d["clip"] for d in drops}
    for stem in dropped_stems:
        inv["clips"].pop(stem, None)
    for g in inv["groups"].values():
        g["clips"] = [s for s in g["clips"] if s not in dropped_stems]
    consumed = {(cl, role) for c in inv["clips"].values()
                for cl, role in (c.get("caption_sources") or [])}
    hit_pairs: dict[str, int] = {}
    for d in drops:
        for r in d["reasons"]:
            if r.startswith("role_scoped_caption_exclusion:"):
                for h in r.split(":", 1)[1].split(","):
                    hit_pairs[h] = hit_pairs.get(h, 0) + 1
    rec = {
        "authority": "A16 §Q1 (drop the 29 at consumption, derived, recorded) on A10's "
                     "consumption-unit rule; reasons vocabulary shared with assemble_root",
        "derivation": "ROLE_EXCLUSIONS INTERSECT this stratum's consumed (clip, role) pairs, "
                      "computed at build time — never a hand-kept stem list",
        "role_exclusions_scanned": {c: list(r) for c, r in sorted(rc.ROLE_EXCLUSIONS.items())},
        "n_dropped": len(drops),
        "excluded_pairs_hit": dict(sorted(hit_pairs.items())),
        "surviving_consumed_pairs": len(consumed),
        "dropped_clips": sorted(drops, key=lambda d: d["clip"]),
    }
    inv["build_drops"] = rec
    if drops:
        print(f"[inventory] {inv['stratum']}: DROPPED {len(drops)} clip(s) at build time on "
              f"the standing A10 role exclusion (A16 disposition): "
              f"{rec['excluded_pairs_hit']}")
        for d in sorted(drops, key=lambda d: d["clip"])[:5]:
            print(f"        drop {d['clip']}: {'; '.join(d['reasons'])}")
    return rec


def _finish(inv: dict, out: Path) -> None:
    n_pairs = sum(len(rc.ring_pairs(sorted(g["clips"]))) for g in inv["groups"].values())
    bd = inv.setdefault("build_drops", {"n_dropped": 0, "dropped_clips": [],
                                        "excluded_pairs_hit": {},
                                        "derivation": "no caption/role attachment step ran"})
    inv["counts"] = {
        "groups": len(inv["groups"]),
        "clips": len(inv["clips"]),
        "pairs_if_unfiltered": n_pairs,
        # A16 — clips removed at BUILD time by the standing role-scoped exclusion.  Recorded
        # here so `clips` is never read as "everything that was rendered".
        "dropped_at_build": bd["n_dropped"],
        "clips_before_build_drops": len(inv["clips"]) + bd["n_dropped"],
    }
    # An inventory is READ AGAIN at assembly time, possibly after the worktree is gone.
    rc.assert_no_worktree_paths(inv, f"{inv['stratum']} inventory")
    rc.write_json(out, inv)
    print(f"[inventory] {inv['stratum']}: {len(inv['groups'])} groups, {len(inv['clips'])} clips, "
          f"{n_pairs} pairs (before exclusions) -> {out}"
          + (f"  [{bd['n_dropped']} dropped at build on the A10 role exclusion]"
             if bd["n_dropped"] else ""))


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
            "latents": rc.canonical_source(lat),
            "cond_clean": rc.canonical_source(S0_COND_CLEAN / cls / f"{clip}.pt"),
            "conditions": rc.canonical_source(S0_CONDITIONS / cls / f"{clip}.pt"),
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
    #: An explicit per-clip `caption_sources` overrides the derivation in
    #: `root_common.caption_sources`. S1's s0cf layer needs it: its endpoint is a certified S0
    #: corpus clip, so it draws on the certified 139 and NOT on the per-(clip, role) store —
    #: the same `[]` that `kind == "corpus"` returns for S0 itself. Without this passthrough the
    #: derivation would name a (clip, role) key that the store legitimately does not contain.
    spec_cap_srcs = spec.get("caption_sources") or {}
    entries = {}
    for gid, g in groups.items():
        for clip in g["clips"]:
            entries[clip] = {"group": gid, "latents": None, "cond_clean": None,
                             "conditions": None, "caption": None,
                             "endpoints": list(endpoints.get(clip, []))}
            if clip in spec_cap_srcs:
                entries[clip]["caption_sources"] = [list(x) for x in spec_cap_srcs[clip]]
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
