"""ctt_v2 dataset viewer — one-time indexer.

Walks the refVFX WebDataset shards (tar headers only — the 375 GB of video bytes are never
read, except a 64 KB fingerprint of each input) and the VFXMaster extracted tree, and writes
compact JSONL indexes the viewer server loads at startup:

    data/raw/refvfx/_viewer_index/code.jsonl.gz     one line per sample:
        {k, sh, m: {in|out|mask: [offset, size]}, et, sf, tf_, sfam, tfam, mt, ori, fp, pr}
    data/raw/refvfx/_viewer_index/lora.jsonl.gz     same, minus spatial/temporal split
    data/raw/refvfx/_viewer_index/vfx.jsonl.gz      {cls, path, cap}

The `fp` content fingerprint is (input size, crc32 of first 64 KB) — if the dataset re-uses
base videos byte-identically across effects, grouping by fp yields the same-content x
many-operators diagonal. Whether that axis is real is REPORTED at the end, not assumed.

Run:  nice -n 19 python scripts/ctt_v2/dataset_viewer/build_index.py [--workers 4]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RAW = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/data/raw")
IDX = RAW / "refvfx/_viewer_index"

ROLE = {"input_image_or_video": "in", "output_video": "out", "mask_or_output_conditioning": "mask"}
_DIGIT = re.compile(r"\d")
# parameter-name tokens that trail an effect family once numbers are cut
_PARAM_WORDS = {"amount", "size", "angle", "strength", "speed", "frame", "frames", "num",
                "segments", "softness", "start", "end", "center", "radius", "scale", "level",
                "intensity", "factor", "width", "height", "count", "sigma", "threshold"}


def family(s: str) -> str:
    toks = []
    for t in s.split("_"):
        if _DIGIT.search(t):
            break
        toks.append(t)
    while toks and toks[-1] in _PARAM_WORDS:
        toks.pop()
    return "_".join(toks) or s


def split_effect(et: str) -> tuple[str, str]:
    """'effect_<spatial>_temporal_<temporal>_mask_...' -> (spatial, temporal)."""
    body = et[len("effect_"):] if et.startswith("effect_") else et
    sp, _, rest = body.partition("_temporal_")
    tp = rest.split("_mask_")[0] if rest else ""
    return sp, tp


def index_shard(args: tuple[str, int, str]) -> list[dict]:
    subset, shard_no, path = args
    samples: dict[str, dict] = {}
    with tarfile.open(path) as tf:
        for m in tf:
            key, _, rest = m.name.partition(".")
            role = ROLE.get(rest.rsplit(".", 1)[0] if rest.endswith((".mp4", ".png")) else rest[:-5]
                            if rest.endswith(".json") else rest)
            s = samples.setdefault(key, {"k": key, "sh": shard_no, "m": {}})
            if rest == "json":
                s["_json"] = (m.offset_data, m.size)
            elif role:
                ext = rest.rsplit(".", 1)[-1]
                s["m"][role] = [m.offset_data, m.size, ext]
    out = []
    with open(path, "rb") as fh:
        for key in sorted(samples):
            s = samples[key]
            off, size = s.pop("_json", (None, None))
            if off is None or "out" not in s["m"]:
                continue
            fh.seek(off)
            meta = json.loads(fh.read(size))
            et = meta.get("effect_type") or ""
            s["et"] = et
            s["mt"] = meta.get("mask_type")
            s["ori"] = meta.get("orientation")
            s["pr"] = meta.get("prompt") or ""
            if subset == "code":
                sp, tp = split_effect(et)
                s["sf"], s["tf_"] = sp, tp
                s["sfam"], s["tfam"] = family(sp), family(tp)
            if "in" in s["m"]:
                ioff, isize = s["m"]["in"][0], s["m"]["in"][1]
                fh.seek(ioff)
                s["fp"] = f"{isize}:{zlib.crc32(fh.read(min(isize, 65536))):08x}"
            out.append(s)
    return out


def build_refvfx(subset: str, workers: int) -> list[dict]:
    shards = sorted((RAW / "refvfx/data" / ("code_based_edits" if subset == "code" else "I2V_LoRA"))
                    .glob("shard-*.tar"))
    jobs = [(subset, i, str(p)) for i, p in enumerate(shards)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, part in enumerate(ex.map(index_shard, jobs)):
            rows.extend(part)
            print(f"[index:{subset}] shard {i}: {len(part)} samples (total {len(rows)})", flush=True)
    return rows


def build_vfx() -> list[dict]:
    root = RAW / "vfxmaster/extracted/data"
    info = json.load(open(RAW / "vfxmaster/info.json"))
    known = {}
    for r in info:
        known[r["video_path"]] = {"cls": r["class"], "path": r["video_path"],
                                  "cap": r.get("video_caption") or ""}
    rows, uncat = [], 0
    for p in sorted(root.rglob("*.mp4")):
        rel = str(p.relative_to(root))
        if rel in known:
            rows.append(known.pop(rel))
        else:
            rows.append({"cls": p.parent.name, "path": rel, "cap": ""})
            uncat += 1
    print(f"[index:vfx] {len(rows)} on-disk mp4s ({uncat} not in info.json; "
          f"{len(known)} info rows with no file)")
    return rows


def write(name: str, rows: list[dict]) -> None:
    IDX.mkdir(parents=True, exist_ok=True)
    with gzip.open(IDX / f"{name}.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"[index] wrote {name}: {len(rows)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--subsets", nargs="*", default=["vfx", "lora", "code"])
    args = ap.parse_args()

    for subset in args.subsets:
        rows = build_vfx() if subset == "vfx" else build_refvfx(subset, args.workers)
        write(subset, rows)
        if subset in ("code", "lora"):
            fps = Counter(r["fp"] for r in rows if "fp" in r)
            multi = sum(1 for c in fps.values() if c > 1)
            ops_per_fp = Counter()
            by_fp: dict[str, set] = {}
            for r in rows:
                if "fp" in r:
                    by_fp.setdefault(r["fp"], set()).add(r["et"])
            cross = sum(1 for s in by_fp.values() if len(s) > 1)
            print(f"[axis-check:{subset}] {len(fps)} distinct contents; {multi} reused; "
                  f"{cross} appear under >1 effect -> content axis "
                  f"{'REAL' if cross > 50 else 'DEGENERATE'}")
        if subset == "code":
            print(f"[axis-check:code] effects={len({r['et'] for r in rows})} "
                  f"spatial_fams={len({r['sfam'] for r in rows})} "
                  f"temporal_fams={len({r['tfam'] for r in rows})}")


if __name__ == "__main__":
    main()
