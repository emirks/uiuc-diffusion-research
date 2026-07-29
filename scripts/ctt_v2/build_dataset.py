#!/usr/bin/env python3
"""Build the CTT v2 SSOT dataset directory — list-based, no symlink root.

This replaces `assemble_root.py`'s physical tree (one symlink per sample x 5 dirs;
2,021,295 links at its peak) with ONE directory that IS the dataset:

    datasets/ctt_v2/
      README.md  VERSION  MANIFEST.json
      samples.jsonl          # one JSON row per base pair: id, stratum, group, caption_key,
                             # shape, and the 5 store paths RELATIVE to this directory
      mix.json               # ruled stratum weights + the derived per-stratum trainer weights
      captions.json          # caption_key -> caption text (provenance/debug)
      encodes/S*/            # latents + cond_clean + the actual mp4 clips (moved in;
                             # S0's are COPIED in from eval_ladder/exp_064, which other
                             # campaigns own and which are therefore never moved)
      conditions/            # by_caption/ content-addressed text embeds (moved in)
                             # s0_corpus/ per-clip S0 embeds (copied in)
      masks/                 # the (shape, sidedness) mask store (copied in)
      inventories/  docs/    # the consumed inventories + DATASET.md + owner reject list

The mix is NOT materialised here. Every base pair appears exactly once; the weights in
mix.json are realised by the trainer's `StratifiedEpochSampler` (ltx_trainer/datasets.py),
which logs per-stratum consumed counts and fails closed at train start — that assert
replaces the off-disk countability the physical root provided (old A3).

Enumeration, exclusions, pairing, slugging and caption keying are IMPORTED from
`assemble_root` / `root_common`, not re-implemented: the physical root's consistent
state is translated, never re-derived.

    # dry run (prints the full plan, writes nothing):
    python3 scripts/ctt_v2/build_dataset.py --expect S0=385,S1=3675,S2a=22731,S2b=23577,S4=6000
    # real build:
    python3 scripts/ctt_v2/build_dataset.py --expect ... --execute
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import assemble_root as ar
import root_common as rc

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "outputs/ctt_v2/strata_manifest.json"
DEFAULT_OUT = REPO / "datasets/ctt_v2"
OLD_SHAPE_CACHE = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/misc/ctt_v2_final/artefacts/retired_roots/ctt_v2_mix/_shape_cache.json")
OLD_MASK_STORE = REPO / "outputs/ctt_v2/masks/_mask_store"

#: Stores that MOVE into the dataset (same filesystem: an instant rename). A relative
#: compat symlink is left at the old location so every recorded path — old root symlinks,
#: viewer mounts, manifests — keeps resolving. (old_path, dataset-relative destination).
MOVES = [
    (REPO / "outputs/ctt_v2/encodes", "encodes"),
    (REPO / "outputs/ctt_v2/conditions/by_caption", "conditions/by_caption"),
]

#: S0's stores are COPIED in, never moved: the originals belong to other campaigns
#: (eval_ladder, exp_058/062/064) and its 139 latents alone span FOUR historical roots.
#: So S0 paths are not prefix-mapped — each file lands at a canonical destination derived
#: from its `<group>/<clip>.pt` tail:
S0_DEST = {"latents": "encodes/S0/latents", "reference_latents": "encodes/S0/latents",
           "cond_clean_latents": "encodes/S0/cond_clean", "conditions": "conditions/s0_corpus"}

DOC_COPIES = [
    (REPO / "data/DATASET.md", "docs/DATASET.md"),
    (REPO / "outputs/viewers/s1_label/rejects.json", "docs/s1_owner_rejects.json"),
]


def parse_expect(spec: str) -> dict[str, int]:
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = int(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--version", default="2.1.0",
                    help="dataset version. 2.0.0 was the retired physical-root form; "
                         "2.1.0 is the list-based form after the S1 owner reject pass.")
    ap.add_argument("--expect", required=True,
                    help="hard-asserted per-stratum base pair counts, e.g. S0=385,S1=3675,...")
    ap.add_argument("--execute", action="store_true",
                    help="perform moves/copies and write the dataset (default: dry run)")
    args = ap.parse_args()

    t0 = time.time()
    out_root = Path(args.out)
    expect = parse_expect(args.expect)

    # ---- enumerate: same code path as the physical root -------------------------------
    man = rc.read_json(args.manifest)
    if man.get("schema") != rc.STRATA_MANIFEST_SCHEMA:
        raise SystemExit(f"bad manifest schema: {man.get('schema')!r}")
    present = [s for s, c in man["strata"].items() if c.get("present")]
    absent = sorted(s for s in man["strata"] if s not in present)
    if absent:
        raise SystemExit(f"[dataset] this builder stamps the FULL dataset; absent strata "
                         f"{absent} need assemble_root's absent-branch logic instead")

    invs = {}
    for s in present:
        inv = rc.read_json(man["strata"][s]["inventory"])
        if inv.get("schema") != rc.INVENTORY_SCHEMA or inv["stratum"] != s:
            raise SystemExit(f"[dataset] bad inventory for {s}")
        invs[s] = inv

    rc.require_role_exclusions("build_dataset")
    ex = rc.load_exclusions(None)
    if not ex.inline_ood_ops:
        raise SystemExit("[dataset] the 8 pre-registered inline-OOD ops did not load")

    kept_groups, samples = {}, {}
    for s, inv in invs.items():
        kept_groups[s], _ = ar.apply_exclusions(inv, ex)
        samples[s] = ar.build_samples(inv, kept_groups[s], int(man["pairing"]["max_refs_per_target"]))
        print(f"[dataset] {s}: {len(kept_groups[s])}/{len(inv['groups'])} groups kept, "
              f"{len(samples[s])} base pairs")

    base_counts = {s: len(samples[s]) for s in present}
    if base_counts != expect:
        raise SystemExit(f"[dataset] HARD STOP: base pair counts {base_counts} != expected {expect}")

    # ---- weights: ruled aggregates, S2 split derived pro-rata (A12) -------------------
    stratum_weights = {n: float(v) for n, v in man["stratum_weights_pct"].items()}
    groups = {k: tuple(v) for k, v in (man.get("prorata_groups") or rc.PRORATA_GROUPS).items()}
    if groups != {k: tuple(v) for k, v in rc.PRORATA_GROUPS.items()}:
        raise SystemExit(f"[dataset] prorata_groups {groups} disagree with root_common")
    if abs(sum(stratum_weights.values()) - 100.0) > 1e-9:
        raise SystemExit(f"[dataset] ruled weights sum to {sum(stratum_weights.values())}, not 100")
    trainer_weights, split = rc.expand_prorata_weights(stratum_weights, base_counts, groups)
    print("[dataset] trainer weights: "
          + ", ".join(f"{s} {w:.6f}" for s, w in sorted(trainer_weights.items())))

    # ---- slugs: same function, same collision guards as assembly ----------------------
    for s in present:
        for scope, gids in (("assembled", kept_groups[s]), ("inventory", invs[s]["groups"])):
            _, collisions = rc.slug_map(gids)
            if collisions:
                raise SystemExit(f"[dataset] {s}: slug collision over {scope} groups: {collisions}")

    # ---- resolve every row's 5 store files on the CURRENT paths -----------------------
    # Shapes come from the retired root's cache (keys are realpath|size|mtime; mv/cp -p
    # preserve size+mtime, so entries stay valid via the alias pass below).
    build_cache = out_root / ".shape_cache.json"
    if not build_cache.exists() and OLD_SHAPE_CACHE.exists():
        build_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OLD_SHAPE_CACHE, build_cache)
    shapes = ar.ShapeCache(build_cache)

    dataset_real = os.path.realpath(out_root)
    prefix_map = [(os.path.realpath(src), dst) for src, dst in MOVES]
    s0_copies: dict[str, str] = {}

    def rewrite(abs_path: str, source: str | None = None, stratum: str | None = None) -> str:
        """Old absolute realpath -> dataset-relative path. Unmatched roots are a hard error."""
        if abs_path.startswith(dataset_real + os.sep):
            return os.path.relpath(abs_path, dataset_real)
        if stratum == "S0":
            group, name = abs_path.rsplit("/", 2)[1:]
            dst = f"{S0_DEST[source]}/{group}/{name}"
            prior = s0_copies.setdefault(abs_path, dst)
            if prior != dst:
                raise SystemExit(f"[dataset] S0 source maps to two destinations: {abs_path}")
            clash = [s for s, d in s0_copies.items() if d == dst and s != abs_path]
            if clash:
                raise SystemExit(f"[dataset] S0 destination collision at {dst}: "
                                 f"{[abs_path] + clash}")
            return dst
        for src_real, dst_rel in prefix_map:
            if abs_path.startswith(src_real + os.sep):
                return dst_rel + abs_path[len(src_real):]
        raise SystemExit(f"[dataset] file outside every declared store root: {abs_path}\n"
                         f"declared roots: {[p for p, _ in prefix_map]}")

    rows, captions = [], {}
    mask_names: set[str] = set()
    for s in present:
        inv = invs[s]
        kind = inv.get("kind", "synthetic_op")
        for smp in samples[s]:
            tgt, ref = inv["clips"][smp["target"]], inv["clips"][smp["reference"]]
            for k in ("latents", "cond_clean", "conditions"):
                if not tgt.get(k):
                    raise SystemExit(f"[dataset] {s}/{smp['target']}: inventory has no {k}")
            if not ref.get("latents"):
                raise SystemExit(f"[dataset] {s}/{smp['reference']}: no reference latents")
            cap = tgt.get("caption")
            if not cap:
                raise SystemExit(f"[dataset] {s}/{smp['target']}: inventory has no caption")

            src = {"latents": os.path.realpath(tgt["latents"]),
                   "reference_latents": os.path.realpath(ref["latents"]),
                   "cond_clean_latents": os.path.realpath(tgt["cond_clean"]),
                   "conditions": os.path.realpath(tgt["conditions"])}
            f, h, w = shapes.get(Path(tgt["latents"]))
            mname = rc.mask_store_name(f, h, w, smp["sided"])
            mask_names.add(mname)

            paths = {k: rewrite(v, source=k, stratum=s) for k, v in src.items()}
            paths["masks"] = f"masks/{mname}"

            ckey = rc.sha256_text(cap)[:16]
            captions[ckey] = cap
            gslug = rc.slug_group(smp["group"])
            rows.append({"id": f"{s}/{gslug}/{smp['target']}__ref_{smp['reference']}",
                         "stratum": s, "group": smp["group"], "group_slug": gslug,
                         "target": smp["target"], "reference": smp["reference"],
                         "sided": smp["sided"], "caption_key": ckey, "shape": [f, h, w],
                         "paths": paths,
                         "endpoints": tgt.get("endpoints") or [],
                         "caption_sources": [list(x) for x in rc.caption_sources(
                             tgt, smp["sided"], kind)]})

    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("[dataset] duplicate row ids")
    for r in rows:
        for p in r["paths"].values():
            if p.startswith(("/", "..")) or "/../" in p:
                raise SystemExit(f"[dataset] non-relative path escaped: {p}")

    # keep the shape cache valid at the files' NEW realpaths (mv/cp -p preserve size+mtime)
    for key in list(shapes.data):
        real, size, mtime = key.rsplit("|", 2)
        try:
            alias = f"{os.path.join(dataset_real, rewrite(real))}|{size}|{mtime}"
        except SystemExit:
            continue
        if alias not in shapes.data:
            shapes.data[alias] = shapes.data[key]
            shapes.dirty = True

    n_files = len({p for r in rows for p in r["paths"].values()})
    print(f"[dataset] {len(rows)} rows, {n_files} distinct files, {len(captions)} captions, "
          f"masks {sorted(mask_names)}")

    missing_masks = [m for m in sorted(mask_names) if not (OLD_MASK_STORE / m).exists()]
    if missing_masks:
        raise SystemExit(f"[dataset] masks absent from the verified store: {missing_masks}")

    if not args.execute:
        print(f"[dataset] DRY RUN — would move {[str(s) for s, _ in MOVES]}, copy "
              f"{len(s0_copies)} S0 files + {len(mask_names)} masks, write {out_root} "
              f"({time.time() - t0:.1f}s)")
        return

    # ---- execute: moves (instant renames) + compat symlinks ---------------------------
    out_root.mkdir(parents=True, exist_ok=True)
    for src, dst_rel in MOVES:
        dst = out_root / dst_rel
        if src.is_symlink():
            if os.path.realpath(src) != os.path.realpath(dst):
                raise SystemExit(f"[dataset] {src} is a symlink but not to {dst}")
            print(f"[dataset] already moved: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
        rel_target = os.path.relpath(dst, src.parent)
        os.symlink(rel_target, src)
        print(f"[dataset] moved {src} -> {dst} (compat symlink left behind)")

    for src_abs, dst_rel in sorted(s0_copies.items()):
        dst = out_root / dst_rel
        if dst.exists() and dst.stat().st_size == os.stat(src_abs).st_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_abs, dst)
    print(f"[dataset] copied {len(s0_copies)} S0 store files (originals untouched)")

    (out_root / "masks").mkdir(exist_ok=True)
    for m in sorted(mask_names):
        dst = out_root / "masks" / m
        if not dst.exists():
            shutil.copy2(OLD_MASK_STORE / m, dst)
    shapes.save()

    # ---- verify: every distinct referenced file exists at its NEW path ----------------
    distinct = sorted({p for r in rows for p in r["paths"].values()})
    with ThreadPoolExecutor(max_workers=16) as pool:
        missing = [p for p in pool.map(
            lambda p: None if (out_root / p).is_file() else p, distinct) if p]
    if missing:
        raise SystemExit(f"[dataset] {len(missing)}/{len(distinct)} files MISSING after "
                         f"build, e.g. {missing[:5]}")
    print(f"[dataset] verified: {len(distinct)} distinct files all present")

    # ---- records -----------------------------------------------------------------------
    (out_root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc.write_json(out_root / "captions.json", captions)
    rc.write_json(out_root / "mix.json", {
        "schema": "ctt_v2_mix/2",
        "authority": man.get("mix_contract_authority"),
        "stratum_weights_pct": stratum_weights,
        "prorata_groups": {k: list(v) for k, v in groups.items()},
        "base_counts": base_counts,
        "prorata_split": split,
        "trainer_weights_pct": trainer_weights,
        "note": "trainer_weights_pct is what the training config's data.stratum_weights_pct "
                "consumes; the S2 aggregate is split pro-rata to base_counts. The mix is "
                "realised by ltx_trainer's StratifiedEpochSampler — to change it, edit the "
                "TRAINING CONFIG; the dataset never needs rebuilding.",
    })

    inv_dir = out_root / "inventories"
    inv_dir.mkdir(exist_ok=True)
    inv_shas = {}
    for s in present:
        src = Path(man["strata"][s]["inventory"])
        shutil.copy2(src, inv_dir / f"{s}.json")
        inv_shas[s] = rc.sha256_file(src)

    (out_root / "docs").mkdir(exist_ok=True)
    for src, dst_rel in DOC_COPIES:
        if src.exists():
            shutil.copy2(src, out_root / dst_rel)
    shutil.copy2(args.manifest, out_root / "docs/strata_manifest.json")

    (out_root / "VERSION").write_text(args.version + "\n")
    rc.write_json(out_root / "MANIFEST.json", {
        "schema": "ctt_v2_dataset/1",
        "version": args.version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "builder": "scripts/ctt_v2/build_dataset.py",
        "seed": man.get("seed", rc.SEED),
        "pairing": {"rule": rc.PAIRING_RULE,
                    "max_refs_per_target": int(man["pairing"]["max_refs_per_target"])},
        "counts": {"rows": len(rows), "per_stratum": base_counts,
                   "distinct_files": len(distinct), "captions": len(captions)},
        "mix_mode": "sampler",
        "trainer_weights_pct": trainer_weights,
        "sha256": {"samples.jsonl": rc.sha256_file(out_root / "samples.jsonl"),
                   "captions.json": rc.sha256_file(out_root / "captions.json"),
                   "mix.json": rc.sha256_file(out_root / "mix.json")},
        "sources": {"strata_manifest": {"path": str(args.manifest),
                                        "sha256": rc.sha256_file(args.manifest)},
                    "inventories_sha256": inv_shas},
        "s1_owner_reject_pass": (invs["S1"].get("provenance") or {}).get("owner_reject_pass"),
        "moved_stores": [{"from": str(s), "to": d, "compat_symlink": True} for s, d in MOVES],
        "copied_s0_files": len(s0_copies),
        "verify": {"distinct_files_stated": len(distinct), "missing": 0,
                   "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
    })

    print(f"[dataset] WROTE {out_root} v{args.version} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
