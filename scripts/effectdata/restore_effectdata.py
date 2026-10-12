#!/usr/bin/env python3
"""Exact restore of the EffectData files we downloaded from HF, at the pinned revision.

The Videos/*.zip (765 GiB) were deleted on 2026-10-12 to free space; everything else in
data/raw/effectdata/ was kept. This script puts them back byte-identically:

  python scripts/effectdata/restore_effectdata.py              # restore Videos/*.zip
  python scripts/effectdata/restore_effectdata.py --what all   # + example_preview/, annotations.json, effect_names_list.csv
  python scripts/effectdata/restore_effectdata.py --names Accretion_Eye_Vortices,Air_Bending   # a few effects
  python scripts/effectdata/restore_effectdata.py --check      # only report what is missing / wrong

Source of truth = restore_manifest.json (beside this script, tracked in git): repo, pinned commit,
and path/size/sha256 of every file (Videos sha256 = HF LFS oid; small files hashed locally at freeze).
Downloads come from https://huggingface.co/datasets/<repo>/resolve/<commit>/<path> (plain HTTPS,
no xet: avoids the Lustre .lock hang). One sequential process (owner rule for login-node downloads);
resumes .part files with HTTP Range; verifies size + sha256 before an atomic rename into place.
Files already present with the right size are skipped (add --rehash to verify their sha256 too).

`freeze` rebuilds the manifest from HF + the local copy (used once, 2026-10-12):
  python scripts/effectdata/restore_effectdata.py freeze
"""
import argparse, hashlib, json, os, sys, time, urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "restore_manifest.json"
DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "effectdata"
REPO = "ysy31415926/EffectData"
CHUNK = 8 << 20


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def freeze():
    from huggingface_hub import HfApi
    api = HfApi()
    rev = api.dataset_info(REPO).sha
    files = []
    for it in api.list_repo_tree(REPO, repo_type="dataset", revision=rev, recursive=True, expand=True):
        if getattr(it, "size", None) is None:
            continue
        top = it.path.split("/")[0]
        if top == "Videos" and it.path.endswith(".zip"):
            files.append({"path": it.path, "size": it.size, "sha256": it.lfs.sha256})
        elif top == "example_preview" or it.path in ("annotations.json", "effect_names_list.csv"):
            local = DATA / it.path
            if not local.is_file() or local.stat().st_size != it.size:
                sys.exit(f"local copy of {it.path} missing or size differs from HF; refusing to freeze")
            files.append({"path": it.path, "size": it.size, "sha256": sha256_file(local)})
    files.sort(key=lambda f: f["path"])
    json.dump({"repo": REPO, "revision": rev, "frozen": time.strftime("%F %T"),
               "dest": "data/raw/effectdata", "n": len(files),
               "total_bytes": sum(f["size"] for f in files), "files": files},
              open(MANIFEST, "w"), indent=0)
    print(f"[ok] {MANIFEST}: {len(files)} files @ {rev}")


def fetch(sess, url, dst, size, sha):
    part = dst.with_name(dst.name + ".part")
    for attempt in range(1, 6):
        start = part.stat().st_size if part.exists() else 0
        if start > size:
            part.unlink(); start = 0
        hdr = {"Range": f"bytes={start}-"} if start else {}
        try:
            with sess.get(url, headers=hdr, stream=True, timeout=(30, 300)) as r:
                if start and r.status_code != 206:     # server ignored Range -> start over
                    start = 0
                r.raise_for_status()
                with open(part, "ab" if start else "wb") as f:
                    for b in r.iter_content(CHUNK):
                        f.write(b)
            if part.stat().st_size != size:
                raise IOError(f"size {part.stat().st_size} != {size}")
            if sha256_file(part) != sha:
                part.unlink()
                raise IOError("sha256 mismatch (part removed)")
            os.replace(part, dst)
            return
        except Exception as e:                       # noqa: BLE001 - retry any transport error
            print(f"  retry {attempt}/5 {dst.name}: {e}", flush=True)
            time.sleep(min(60, 5 * attempt))
    raise SystemExit(f"FAILED after retries: {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="restore", choices=["restore", "freeze"])
    ap.add_argument("--what", default="videos", choices=["videos", "all"])
    ap.add_argument("--names", help="comma-separated effect names (Videos/<name>.zip) to restore")
    ap.add_argument("--dest", default=str(DATA), help="target root (default: data/raw/effectdata)")
    ap.add_argument("--check", action="store_true", help="report missing/wrong files, download nothing")
    ap.add_argument("--rehash", action="store_true", help="also sha256-verify files already present")
    a = ap.parse_args()
    if a.cmd == "freeze":
        return freeze()

    m = json.load(open(MANIFEST))
    files = [f for f in m["files"] if a.what == "all" or f["path"].startswith("Videos/")]
    if a.names:
        want = {f"Videos/{n.strip()}.zip" for n in a.names.split(",")}
        files = [f for f in files if f["path"] in want]
        if len(files) != len(want):
            sys.exit(f"unknown names: {sorted(want - {f['path'] for f in files})}")
    dest = Path(a.dest)
    todo = []
    for f in files:
        p = dest / f["path"]
        ok = p.is_file() and p.stat().st_size == f["size"] and (not a.rehash or sha256_file(p) == f["sha256"])
        if not ok:
            todo.append(f)
    need = sum(f["size"] for f in todo)
    print(f"{m['repo']} @ {m['revision'][:12]}: {len(files)} selected, {len(todo)} to fetch "
          f"({need / 2**30:,.1f} GiB) -> {dest}", flush=True)
    if a.check or not todo:
        return
    import requests
    sess = requests.Session()
    tok = os.environ.get("HF_TOKEN")
    if tok:
        sess.headers["Authorization"] = f"Bearer {tok}"
    base = f"https://huggingface.co/datasets/{m['repo']}/resolve/{m['revision']}/"
    t0, done = time.time(), 0
    for i, f in enumerate(todo, 1):
        p = dest / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        fetch(sess, base + urllib.parse.quote(f["path"], safe="/"), p, f["size"], f["sha256"])
        done += f["size"]
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {done / 2**30:,.1f}/{need / 2**30:,.1f} GiB  "
                  f"{done / 2**20 / max(el, 1):,.0f} MiB/s", flush=True)
    print("[ok] restore complete, all files size+sha256 verified")


if __name__ == "__main__":
    main()
