#!/usr/bin/env python3
"""Build the static ctt_v2 corpora viewer (refVFX + VFXMaster).

The original viewer needed a server for two reasons: it grouped 150k samples by
axis at startup, and it sliced video members out of WebDataset tars per request.
Neither needs a process once the static server honours Range (viewerctl httpd
does). This script precomputes the grouping into range-addressable files, so the
page fetches only the bytes it is about to show:

    meta.json              axes: [label, count, start] per subset — the sidebar
    ids_<sub>_<axis>.bin   uint32 row ids, grouped, in the same order as meta
    rows_<sub>.jsonl       one slim record per row
    rowoff_<sub>.bin       uint32 byte offsets into rows_<sub>.jsonl (n+1 entries)
    shards_* / vfx_*       symlinks to the tars and the VFXMaster tree

Media never gets copied: the page range-fetches [offset, offset+len) out of the
shard tar and hands the bytes to <video> as a blob.

    python3 scripts/viewers/build_ctt_v2_corpora.py
"""
from __future__ import annotations

import array
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RAW = REPO / "data/raw"
IDX = RAW / "refvfx/_viewer_index"
OUT = REPO / "outputs/viewers/ctt_v2_corpora"

SHARD_DIR = {"code": RAW / "refvfx/data/code_based_edits",
             "lora": RAW / "refvfx/data/I2V_LoRA"}
VFX_H264 = RAW / "vfxmaster/extracted_h264/data"
VFX_ORIG = RAW / "vfxmaster/extracted/data"

AXES = {"code": ["effect", "spatial", "temporal", "content", "browse"],
        "lora": ["effect", "content", "browse"],
        "vfx": ["class", "browse"]}
AXIS_KEY = {"effect": "et", "spatial": "sfam", "temporal": "tfam",
            "content": "fp", "class": "cls"}
MAX_LABELS = 5000          # sidebar caps at this; the count is reported when it bites


def link(name: str, target: Path) -> None:
    p = OUT / name
    if not target.exists():
        print(f"  {name}: MISSING TARGET {target}")
        return
    rel = os.path.relpath(target, OUT)
    if p.is_symlink():
        if os.readlink(p) == rel:
            return
        p.unlink()
    elif p.exists():
        print(f"  {name}: real file in the way, skipped")
        return
    p.symlink_to(rel)


def slim(subset: str, r: dict) -> dict:
    """Only what a card renders — the index carries build-time fields too."""
    if subset == "vfx":
        rel = r["path"]
        root = 0 if (VFX_H264 / rel).exists() else 1
        return {"p": rel, "r": root, "cls": r["cls"], "pr": r.get("cap")}
    out = {"sh": r["sh"], "m": r["m"], "et": r.get("et"), "pr": r.get("pr"),
           "ori": r.get("ori"), "mt": r.get("mt"), "fp": r.get("fp")}
    if r.get("sfam"):
        out["sfam"] = r["sfam"]
    if r.get("tfam"):
        out["tfam"] = r["tfam"]
    return {k: v for k, v in out.items() if v is not None}


def build_subset(subset: str, meta: dict) -> None:
    src = IDX / f"{subset}.jsonl.gz"
    if not src.exists():
        print(f"[{subset}] SKIP — {src} missing (run build_index.py)")
        meta[subset] = {"n": 0, "axes": {}, "axis_order": AXES[subset]}
        return

    rows = [json.loads(line) for line in gzip.open(src, "rt")]
    n = len(rows)

    # rows_<sub>.jsonl + uint32 offset table (n+1 entries, so len = off[i+1]-off[i])
    offs = array.array("I", [0])
    with open(OUT / f"rows_{subset}.jsonl", "wb") as fh:
        for r in rows:
            fh.write(json.dumps(slim(subset, r), separators=(",", ":")).encode() + b"\n")
            offs.append(fh.tell())
    (OUT / f"rowoff_{subset}.bin").write_bytes(offs.tobytes())

    axes_meta = {}
    for axis in AXES[subset]:
        if axis == "browse":
            continue
        key = AXIS_KEY[axis]
        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            v = r.get(key)
            if v:
                groups[v].append(i)
        if axis == "content":
            # only groups that realise the counterfactual (same content, >1 operator)
            groups = {k: v for k, v in groups.items() if len(v) > 1}
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        dropped = max(0, len(ordered) - MAX_LABELS)
        ordered = ordered[:MAX_LABELS]

        ids = array.array("I")
        entries = []
        for label, members in ordered:
            entries.append([label, len(members), len(ids)])
            ids.extend(members)
        (OUT / f"ids_{subset}_{axis}.bin").write_bytes(ids.tobytes())
        axes_meta[axis] = entries
        note = f", {dropped} labels beyond the {MAX_LABELS} cap dropped" if dropped else ""
        print(f"[{subset}] {axis}: {len(entries)} groups, {len(ids)} memberships{note}")

    meta[subset] = {"n": n, "axes": axes_meta, "axis_order": AXES[subset]}
    if subset in SHARD_DIR:
        meta[subset]["shards"] = sorted(p.name for p in SHARD_DIR[subset].glob("shard-*.tar"))
    print(f"[{subset}] {n} rows, rows file "
          f"{(OUT / f'rows_{subset}.jsonl').stat().st_size / 1e6:.1f} MB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    for subset in ("code", "lora", "vfx"):
        build_subset(subset, meta)

    link("shards_code", SHARD_DIR["code"])
    link("shards_lora", SHARD_DIR["lora"])
    link("vfx_h264", VFX_H264)
    link("vfx_orig", VFX_ORIG)
    # The page itself is source, so it is tracked under scripts/ and linked in here.
    link("index.html", REPO / "scripts/viewers/ctt_v2_corpora.html")

    (OUT / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"\n[done] {OUT.relative_to(REPO)} — meta.json "
          f"{(OUT / 'meta.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
