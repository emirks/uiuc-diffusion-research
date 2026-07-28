#!/usr/bin/env python
"""Analyze the REAL (Pexels) portion of the HumanVid dataset from its manifest alone.

HumanVid's real subset is distributed as URL lists + per-frame camera annotations
(Google Drive folder 1UGEkOKXYX9BGUFz0ao6lOGXkZjQGoJcZ), NOT as media -- the authors
state "we cannot redistribute them".

Every Pexels CDN URL encodes resolution and fps in its filename:
    https://videos.pexels.com/video-files/<id>/<id>-hd_<W>_<H>_<fps>fps.mp4
and each camera annotation file has exactly one line per frame. Together these give
resolution / fps / frame-count / duration for all ~19k real clips with ZERO downloads,
which matters because the Pexels ToS forbids scripted bulk collection for ML.

Usage:
    python scripts/analyze_humanvid_real.py \
        --manifest-dir /tmp/hv_probe/gdrive \
        --out data/manifests/humanvid_real
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import Counter
from pathlib import Path

# <id>-<tier>_<W>_<H>_<fps>fps.mp4  (tier is sd/hd/uhd)
URL_RE = re.compile(
    r"video-files/(?P<id>\d+)/(?P<fid>\d+)-(?P<tier>[a-z]+)_(?P<w>\d+)_(?P<h>\d+)_(?P<fps>\d+)fps\.mp4"
)

# Our endpoint contract.
TARGET_W, TARGET_H = 480, 640
TARGET_FRAMES = 121
TARGET_FPS = 24
TARGET_SECONDS = TARGET_FRAMES / TARGET_FPS  # 5.0417 s
TARGET_AR = TARGET_W / TARGET_H  # 0.75 portrait


def parse_manifest(path: Path) -> tuple[list[dict], list[str]]:
    """Parse one url list. Returns (records, unparseable_lines)."""
    records, bad = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = URL_RE.search(line)
        if not m:
            bad.append(line)
            continue
        w, h = int(m["w"]), int(m["h"])
        records.append(
            {
                "id": m["id"],
                "url": line,
                "tier": m["tier"],
                "width": w,
                "height": h,
                "fps": int(m["fps"]),
                "aspect": w / h,
                "orientation": "portrait" if h > w else ("square" if h == w else "landscape"),
            }
        )
    return records, bad


def frame_counts(cam_dir: Path) -> dict[str, int]:
    """video id -> frame count (one camera pose line per frame)."""
    out = {}
    if not cam_dir.is_dir():
        return out
    for f in cam_dir.glob("*.txt"):
        n = 0
        with f.open("rb") as fh:
            for n, _ in enumerate(fh, 1):
                pass
        out[f.stem] = n
    return out


def max_crop(rec: dict) -> tuple[float, float]:
    """Largest 0.75-AR (portrait) window that fits inside the source frame."""
    w, h = rec["width"], rec["height"]
    if (w / h) > TARGET_AR:  # source wider than target -> full height, narrow width
        return h * TARGET_AR, float(h)
    return float(w), w / TARGET_AR  # source taller/narrower -> full width, short height


def portrait_crop_ok(rec: dict) -> bool:
    """Can a 480x640 portrait window be cropped from this frame without upscaling?"""
    cw, ch = max_crop(rec)
    return cw >= TARGET_W and ch >= TARGET_H


def crop_retention(rec: dict) -> float:
    """Fraction of source frame AREA kept by the largest 0.75-AR crop.

    This is the number that actually matters: it says how much framing context is
    discarded, i.e. how likely a subject-aware portrait crop is to cut the subject.
    """
    cw, ch = max_crop(rec)
    return (cw * ch) / (rec["width"] * rec["height"])


def enough_frames(rec: dict) -> bool:
    """Does the clip hold >= 5.04 s of content (one 121f @24fps endpoint after resample)?"""
    n = rec.get("frames")
    if not n:
        return False
    return (n / rec["fps"]) >= TARGET_SECONDS


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def summarize(recs: list[dict]) -> dict:
    n = len(recs)
    with_frames = [r for r in recs if r.get("frames")]
    durs = sorted(r["frames"] / r["fps"] for r in with_frames)

    def q(p: float) -> float:
        if not durs:
            return 0.0
        return round(durs[min(int(p * len(durs)), len(durs) - 1)], 2)

    crop_ok = [r for r in recs if portrait_crop_ok(r)]
    long_ok = [r for r in with_frames if enough_frames(r)]
    both = [r for r in with_frames if portrait_crop_ok(r) and enough_frames(r)]
    ret = sorted(crop_retention(r) for r in recs)

    # How many 121f@24fps endpoints could a clip yield (HumanVid itself segments >10 s)?
    total_endpoints = sum(int((r["frames"] / r["fps"]) // TARGET_SECONDS) for r in both)

    return {
        "n_clips": n,
        "n_with_camera_annotation": len(with_frames),
        "orientation": dict(Counter(r["orientation"] for r in recs).most_common()),
        "fps": dict(Counter(r["fps"] for r in recs).most_common()),
        "tier": dict(Counter(r["tier"] for r in recs).most_common()),
        "top_resolutions": dict(
            Counter(f"{r['width']}x{r['height']}" for r in recs).most_common(12)
        ),
        "duration_s": {
            "min": round(durs[0], 2) if durs else 0,
            "p10": q(0.10),
            "median": q(0.50),
            "p90": q(0.90),
            "max": round(durs[-1], 2) if durs else 0,
            "mean": round(statistics.fmean(durs), 2) if durs else 0,
            "total_hours": round(sum(durs) / 3600, 1) if durs else 0,
        },
        "endpoint_fitness": {
            "portrait_crop_no_upscale": {"n": len(crop_ok), "pct": pct(len(crop_ok), n)},
            "duration_ge_5.04s": {
                "n": len(long_ok),
                "pct": pct(len(long_ok), len(with_frames)),
            },
            "both": {"n": len(both), "pct": pct(len(both), n)},
            "max_121f_endpoints_if_segmented": total_endpoints,
            "crop_area_retained": {
                "median": round(statistics.median(ret), 3) if ret else 0,
                "mean": round(statistics.fmean(ret), 3) if ret else 0,
                "note": "fraction of source frame area surviving the largest 0.75-AR portrait crop",
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    md = args.manifest_dir
    args.out.mkdir(parents=True, exist_ok=True)

    cam_root = md / "cam" / "camera_release"
    all_recs: list[dict] = []
    per_split = {}

    for split, fname in (
        ("horizontal", "pexels-horizontal-urls-new.txt"),
        ("vertical", "pexels-vertical-urls-new.txt"),
    ):
        recs, bad = parse_manifest(md / fname)
        fc = frame_counts(cam_root / split)
        hits = 0
        for r in recs:
            r["split"] = split
            if r["id"] in fc:
                r["frames"] = fc[r["id"]]
                hits += 1
        per_split[split] = {
            "unparseable_urls": len(bad),
            "camera_files_on_disk": len(fc),
            "urls_matched_to_camera": hits,
            **summarize(recs),
        }
        all_recs.extend(recs)

    report = {
        "source": {
            "dataset": "HumanVid (NeurIPS D&B 2024), real/Internet portion",
            "platform": "Pexels.com",
            "manifest": "Google Drive folder 1UGEkOKXYX9BGUFz0ao6lOGXkZjQGoJcZ",
            "media_redistributed": False,
            "note": "authors: 'The pexels video data is collected from the Internet and we cannot redistribute them.'",
        },
        "endpoint_contract": {
            "width": TARGET_W,
            "height": TARGET_H,
            "frames": TARGET_FRAMES,
            "fps": TARGET_FPS,
            "seconds": round(TARGET_SECONDS, 4),
        },
        "overall": summarize(all_recs),
        "per_split": per_split,
    }

    (args.out / "fitness_report.json").write_text(json.dumps(report, indent=2))

    # Compact per-clip index (metadata only -- no media). Gzipped: 4.7 MB -> 0.2 MB.
    with gzip.open(args.out / "clips.jsonl.gz", "wt") as fh:
        for r in sorted(all_recs, key=lambda x: (x["split"], x["id"])):
            fh.write(json.dumps(r) + "\n")

    print(json.dumps(report["overall"], indent=2))
    print(f"\nwrote {args.out/'fitness_report.json'} and {args.out/'clips.jsonl.gz'}")


if __name__ == "__main__":
    main()
