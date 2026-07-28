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

    CAPTION LOOKUP IS (clip, role)-KEYED, HARD-FAILS ON MISSING, AND HAS NO CROSS-ROLE
    FALLBACK (A10 `enforced_at[0]`).  Two guards, both hard:

    * requesting a role-excluded (clip, role) description raises immediately — a request
      for `openvid_T1MiFx98l3g_0_50to156`'s A-role description means the clip leaked into
      A-role upstream, and that must crash rather than silently substitute the healthy
      B-role text and describe the wrong nine frames;
    * a missing key is an error, never a fallback to the other role or to a clip-level key.
    """
    caps = json.loads(Path(args.captions).read_text()) if args.captions else None
    if isinstance(caps, list):  # [{video, caption}] shape
        caps = {r["video"]: r["caption"] for r in caps}
    missing, role_violations = [], []
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
        # the (clip, role) descriptions this clip's caption will consume — recorded on the
        # entry so the root asserts do not have to re-derive them
        srcs = rc.caption_sources(c, g.get("sided", "two"), inv.get("kind", "synthetic_op"))
        c["caption_sources"] = [list(x) for x in srcs]
        for clip_id, role in srcs:
            if rc.role_excluded(clip_id, role):
                role_violations.append(f"{stem}: needs the role-{role} description of "
                                       f"{clip_id!r}, which is role-excluded by A10")
        if caps is not None:
            k = _fmt(args.caption_key, **ctx)
            if k not in caps:
                missing.append(f"caption:{k}")
            else:
                c["caption"] = caps[k]
    if role_violations:
        raise SystemExit(
            f"[inventory] {len(role_violations)} clips consume a ROLE-EXCLUDED (clip, role) "
            f"description (A10; {rc.POOL_DROPS.name}). No cross-role fallback exists — fix "
            f"the grid or the render, do not degrade:\n  " + "\n  ".join(role_violations[:10]))
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
