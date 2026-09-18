#!/usr/bin/env python3
"""store_features — the ONE feature-store CLI (contract v2, clause 10).

Subcommands:
  coverage   variant x namespace have/of matrix; --md writes FEATURES_COVERAGE.md
  fsck       orphan / stale / mixed-host / manifest checks; exit 1 on any of those
  sha256sums write videos/SHA256SUMS (skips unchanged by size+mtime)
  migrate    legacy lookup -> validate np.load -> HARD LINK -> sidecar -> manifest
             -> meta block; per-ns hit/miss table
  extract    fill misses only (extractor registry keyed by namespace); GPU

Targets (shared by every subcommand):
  --gens GLOB...   variant dirs (store/gens/<arm>/<variant>); videos under videos/
  --corpus         data/processed/transitions_std121/*/*.mp4 (grouped per class)
  --conds          eval_ladder/conds/*.mp4

Records per feature: host (socket.gethostname()), code_sha (git HEAD short sha,
+dirty if the two feature-store files are modified), created (ISO-8601 UTC).

No GPU is required for coverage/fsck/sha256sums/migrate. Only ``extract`` loads a
backbone (imported lazily from diffusion.feature_extractors).
"""

from __future__ import annotations

import argparse
import glob
import json
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.feature_store import (  # noqa: E402
    MIGRATABLE, NAMESPACES, FeatureStore, legacy_filenames, legacy_key,
    sha256_file)

# default legacy cache dirs searched, in order, before any --from extras
DEFAULT_LEGACY_DIRS = ("misc/refvfx_baseline/probe/cache", "outputs/eval/cache")
CORPUS_GLOB = "data/processed/transitions_std121/*/*.mp4"
CONDS_DIR = "eval_ladder/conds"
COVERAGE_MD = "store/FEATURES_COVERAGE.md"


# --- provenance --------------------------------------------------------------
def code_sha() -> str:
    """git HEAD short sha; ``+dirty`` when either feature-store file is modified."""
    try:
        head = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
    watch = ["src/diffusion/feature_store.py", "scripts/store_features.py"]
    try:
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", *watch],
            text=True, stderr=subprocess.DEVNULL).strip()
        if dirty:
            head += "+dirty"
    except Exception:
        pass
    return head


# --- target resolution -------------------------------------------------------
def _expand(pattern: str) -> list[Path]:
    hits = glob.glob(pattern) or glob.glob(str(REPO_ROOT / pattern))
    return sorted(Path(p) for p in hits)


def resolve_targets(args) -> list[dict]:
    """-> [{label, video_dir, variant_dir|None}] — one entry per group of mp4s."""
    targets: list[dict] = []
    for pat in getattr(args, "gens", None) or []:
        for vdir in _expand(pat):
            if not vdir.is_dir():
                continue
            label = "/".join(vdir.parts[-3:])          # gens/<arm>/<variant>
            targets.append({"label": label, "video_dir": vdir / "videos",
                            "variant_dir": vdir})
    if getattr(args, "corpus", False):
        classes = sorted({p.parent for p in _expand(CORPUS_GLOB)})
        for cdir in classes:
            targets.append({"label": f"corpus/{cdir.name}", "video_dir": cdir,
                            "variant_dir": None})
    if getattr(args, "conds", False):
        conds = REPO_ROOT / CONDS_DIR
        if conds.is_dir():
            targets.append({"label": "conds", "video_dir": conds,
                            "variant_dir": None})
    return targets


# --- pure workers (importable / unit-testable) -------------------------------
def migrate_videos(store: FeatureStore, videos, search_dirs, *, host, sha,
                   dry_run=False) -> dict:
    """Migrate the three legacy namespaces for a list of videos.

    Returns {ns: {"hit": n, "miss": n, "linked": n, "bad": [..]}}. A dry run
    only counts hits/misses (no np.load, no link). A real run validates each hit
    with ``np.load`` (materializing the primary array), then hard-links it and
    writes the sidecar. Videos already present for a namespace are counted as
    hits (``linked`` unchanged)."""
    import numpy as np
    from diffusion.feature_store import NS_ARRAYS
    stats = {ns: {"hit": 0, "miss": 0, "linked": 0, "bad": []} for ns in MIGRATABLE}
    for v in videos:
        try:
            key = legacy_key(v)
        except FileNotFoundError:
            for ns in MIGRATABLE:
                stats[ns]["miss"] += 1
            continue
        vsha = None                                         # hash the video once, on demand
        for ns in MIGRATABLE:
            if store.has(v, ns):
                stats[ns]["hit"] += 1
                continue
            found = None
            for name in legacy_filenames(ns, key):
                for d in search_dirs:
                    p = Path(d) / name
                    if p.exists():
                        found = p
                        break
                if found:
                    break
            if found is None:
                stats[ns]["miss"] += 1
                continue
            stats[ns]["hit"] += 1
            if dry_run:
                continue
            try:
                z = np.load(found)
                prim = next((a for a in NS_ARRAYS[ns] if a in z.files), None)
                if prim is None:
                    raise ValueError(f"missing array {NS_ARRAYS[ns]} in {found.name}")
                _ = z[prim].shape                       # force a read (truncation)
            except Exception as e:                       # noqa: BLE001
                stats[ns]["miss"] += 1
                stats[ns]["hit"] -= 1
                stats[ns]["bad"].append(f"{v.name}:{ns}:{type(e).__name__}:{e}")
                continue
            try:
                rel = str(found.resolve().relative_to(store.root))
            except ValueError:
                rel = str(found.resolve())
            if vsha is None:
                vsha = sha256_file(v)
            # migrated: extraction host unknown -> sidecar host=null; the node
            # running migrate is recorded as migrated_by_host.
            store.put(v, ns, None, {"migrated_by_host": host, "code_sha": sha,
                                    "video_sha256": vsha, "origin": f"migrated:{rel}"},
                      link_from=found)
            stats[ns]["linked"] += 1
    return stats


def extract_videos(store: FeatureStore, ns: str, videos, extractor, *, host, sha,
                   dry_run=False) -> dict:
    """Fill misses of one namespace. ``extractor`` is an object with
    ``extract(video) -> dict[str, np.ndarray]`` (already built / loaded). A dry
    run counts what WOULD be filled. Returns {"have","miss","filled","bad"}."""
    out = {"have": 0, "miss": 0, "filled": 0, "bad": []}
    for v in videos:
        if store.has(v, ns):
            out["have"] += 1
            continue
        out["miss"] += 1
        if dry_run:
            continue
        try:
            arrays = extractor.extract(v)
            store.put(v, ns, arrays, {"host": host, "code_sha": sha,
                                      "origin": "extracted"})
            out["filled"] += 1
        except Exception as e:                           # noqa: BLE001
            out["bad"].append(f"{v.name}:{type(e).__name__}:{e}")
    return out


def write_sha256sums(store: FeatureStore, video_dir: Path) -> dict:
    """Write ``<video_dir>/SHA256SUMS`` (standard sha256sum text format), reusing
    hashes for mp4s whose size+mtime are unchanged (cached in an ignored
    ``.SHA256SUMS.stat.json``). Returns {"n","computed","reused"}."""
    video_dir = Path(video_dir)
    stat_p = video_dir / ".SHA256SUMS.stat.json"
    try:
        cache = json.loads(stat_p.read_text())
    except Exception:
        cache = {}
    lines, new_cache = [], {}
    computed = reused = 0
    for v in store.iter_videos(video_dir):
        st = v.stat()
        c = cache.get(v.name)
        if c and c.get("size") == st.st_size and c.get("mtime_ns") == st.st_mtime_ns:
            sha = c["sha256"]
            reused += 1
        else:
            sha = sha256_file(v)
            computed += 1
        lines.append(f"{sha}  {v.name}")
        new_cache[v.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                             "sha256": sha}
    (video_dir / "SHA256SUMS").write_text("\n".join(lines) + ("\n" if lines else ""))
    stat_p.write_text(json.dumps(new_cache, indent=0))
    return {"n": len(lines), "computed": computed, "reused": reused}


def _short(ns: str) -> str:
    return ns.split("@", 1)[0]


# --- subcommands -------------------------------------------------------------
def cmd_coverage(args, store):
    targets = resolve_targets(args)
    rows = []
    for t in targets:
        cov = store.coverage(t["video_dir"])
        rows.append((t["label"], cov))
    cols = [_short(ns) for ns in NAMESPACES]
    w0 = max([len("target")] + [len(r[0]) for r in rows], default=6)
    header = "  ".join([f"{'target':<{w0}}"] + [f"{c:>11}" for c in cols])
    print(header)
    for label, cov in rows:
        cells = [f"{cov[ns]['have']}/{cov[ns]['of']}" for ns in NAMESPACES]
        print("  ".join([f"{label:<{w0}}"] + [f"{c:>11}" for c in cells]))
    if getattr(args, "md", False):
        md = ["# Feature coverage (generated by `store_features.py coverage --md`)",
              "", "| target | " + " | ".join(cols) + " |",
              "|---|" + "|".join(["---"] * len(cols)) + "|"]
        for label, cov in rows:
            cells = [f"{cov[ns]['have']}/{cov[ns]['of']}" for ns in NAMESPACES]
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        (REPO_ROOT / COVERAGE_MD).write_text("\n".join(md) + "\n")
        print(f"[md] wrote {COVERAGE_MD} ({len(rows)} targets)")
    return 0


def cmd_fsck(args, store):
    targets = resolve_targets(args)
    bad = False
    for t in targets:
        if getattr(args, "rebuild_manifest", False):
            n = store.rebuild_manifest(t["video_dir"])
            print(f"[manifest] {t['label']}: rebuilt {n} records")
        rep = store.fsck(t["video_dir"], rehash=getattr(args, "rehash", False))
        flags = []
        for k in ("orphan_npz", "orphan_json", "stale", "no_video"):
            if rep[k]:
                flags.append(f"{k}={len(rep[k])}")
        if rep["mixed_hosts"]:
            flags.append(f"mixed_hosts={list(rep['mixed_hosts'])}")
        if rep["legacy_ns"]:
            flags.append(f"legacy={[_short(ns) for ns in rep['legacy_ns']]}")
        if rep["manifest_drift"]:
            flags.append("manifest_drift")
        status = "OK" if rep["ok"] and not rep["manifest_drift"] else \
                 ("WARN" if rep["ok"] else "FAIL")
        print(f"[{status}] {t['label']}: {rep['n_features']} features over "
              f"{rep['n_videos']} videos" + (f" | {', '.join(flags)}" if flags else ""))
        for k in ("orphan_npz", "orphan_json", "stale"):
            for item in rep[k][:20]:
                print(f"    {k}: {item}")
        if not rep["ok"]:
            bad = True
    return 1 if bad else 0


def cmd_sha256sums(args, store):
    for t in resolve_targets(args):
        r = write_sha256sums(store, t["video_dir"])
        print(f"[sha256sums] {t['label']}: {r['n']} files "
              f"({r['computed']} computed, {r['reused']} reused) -> "
              f"{t['video_dir']}/SHA256SUMS")
    return 0


def _refresh(store, t):
    """Post-write bookkeeping: rebuild manifest, refresh gen meta features block."""
    store.rebuild_manifest(t["video_dir"])
    vd = t.get("variant_dir")
    if vd is not None and (vd / "meta.yaml").exists():
        store.write_meta_block(vd)


def cmd_migrate(args, store):
    targets = resolve_targets(args)
    search_dirs = [REPO_ROOT / d for d in DEFAULT_LEGACY_DIRS]
    search_dirs += [Path(d) for d in (args.from_dirs or [])]
    search_dirs = [d for d in search_dirs if d.exists()]
    host, sha = socket.gethostname(), code_sha()
    print(f"[migrate] dirs: {', '.join(str(d) for d in search_dirs)}"
          f"{'  (DRY RUN)' if args.dry_run else ''}")
    grand = {ns: {"hit": 0, "miss": 0, "linked": 0} for ns in MIGRATABLE}
    for t in targets:
        vids = store.iter_videos(t["video_dir"])
        stats = migrate_videos(store, vids, search_dirs, host=host, sha=sha,
                               dry_run=args.dry_run)
        cells = [f"{_short(ns)} {stats[ns]['hit']}/{len(vids)}" for ns in MIGRATABLE]
        print(f"  {t['label']:<48} " + "  ".join(cells))
        for ns in MIGRATABLE:
            for k in ("hit", "miss", "linked"):
                grand[ns][k] += stats[ns][k]
            for b in stats[ns]["bad"]:
                print(f"    BAD: {b}")
        if not args.dry_run:
            _refresh(store, t)
    print("  " + "-" * 60)
    for ns in MIGRATABLE:
        g = grand[ns]
        print(f"  TOTAL {_short(ns):<12} hit={g['hit']} miss={g['miss']} "
              f"linked={g['linked']}")
    return 0


def cmd_extract(args, store):
    targets = resolve_targets(args)
    ns = args.ns
    if ns not in NAMESPACES:
        print(f"unknown namespace {ns!r}; known: {', '.join(NAMESPACES)}")
        return 2
    all_videos = []
    for t in targets:
        all_videos += [(t, v) for v in store.iter_videos(t["video_dir"])]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        all_videos = all_videos[i::n]
    host, sha = socket.gethostname(), code_sha()
    to_fill = [v for (t, v) in all_videos if not store.has(v, ns)]
    print(f"[extract] {ns}: {len(all_videos)} videos, {len(to_fill)} to fill"
          f"{'  (DRY RUN)' if args.dry_run else ''}")
    if args.dry_run or not to_fill:
        return 0
    from diffusion.feature_extractors import REGISTRY
    extractor = REGISTRY[ns](args.device)               # loads backbone once (GPU)
    touched = {}
    for t, v in all_videos:
        r = extract_videos(store, ns, [v], extractor, host=host, sha=sha)
        if r["filled"]:
            touched[t["label"]] = t
        for b in r["bad"]:
            print(f"    BAD: {b}")
    for t in touched.values():
        _refresh(store, t)
    print(f"[extract] {ns}: filled {len(to_fill)} (touched {len(touched)} targets)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_targets(p):
        p.add_argument("--gens", nargs="+", metavar="GLOB",
                       help="variant dirs (store/gens/<arm>/<variant>)")
        p.add_argument("--corpus", action="store_true")
        p.add_argument("--conds", action="store_true")

    p = sub.add_parser("coverage"); add_targets(p)
    p.add_argument("--md", action="store_true", help=f"write {COVERAGE_MD}")

    p = sub.add_parser("fsck"); add_targets(p)
    p.add_argument("--rehash", action="store_true")
    p.add_argument("--rebuild-manifest", dest="rebuild_manifest", action="store_true")

    p = sub.add_parser("sha256sums"); add_targets(p)

    p = sub.add_parser("migrate"); add_targets(p)
    p.add_argument("--from", dest="from_dirs", nargs="+", metavar="DIR",
                   help="extra legacy cache dirs (searched after the defaults)")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("extract"); add_targets(p)
    p.add_argument("ns", help="namespace to fill")
    p.add_argument("--shard", metavar="i/n", help="0-based shard index / count")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cuda")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    store = FeatureStore(REPO_ROOT)
    return {"coverage": cmd_coverage, "fsck": cmd_fsck,
            "sha256sums": cmd_sha256sums, "migrate": cmd_migrate,
            "extract": cmd_extract}[args.cmd](args, store)


if __name__ == "__main__":
    sys.exit(main())
