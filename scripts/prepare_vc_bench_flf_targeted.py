#!/usr/bin/env python3
"""
Preprocess a specific subset of vc-bench raw videos: extract first/last clips.

Identical logic to prepare_vc_bench_flf.py, but instead of scanning ALL videos
in the raw directory, it processes only the filenames listed in --targets (one
raw filename per line, or passed directly as positional arguments).

Reads:  data/raw/vc-bench-hf/  (flat directory; filenames contain backslashes)
Writes: data/processed/vc-bench-flf/  (same layout as prepare_vc_bench_flf.py)

Example:
    python scripts/prepare_vc_bench_flf_targeted.py \\
        --raw-dir data/raw/vc-bench-hf \\
        "Animals\\animal\\animal_1508067_3840x2160.mp4" \\
        "Weather\\sunset\\sunset_1621682_1920x1080.mp4"
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image


REPO_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR    = REPO_ROOT / "data" / "raw" / "vc-bench-hf"
OUT_DIR    = REPO_ROOT / "data" / "processed" / "vc-bench-flf"
CLIP_SIZES = [2, 4, 8, 16, 24, 36, 48]


def make_video_id(filename: str) -> str:
    stem  = filename[:-4] if filename.endswith(".mp4") else filename
    clean = stem.replace("\\", "_")
    clean = re.sub(r"\s*&\s*", "_", clean)
    clean = clean.replace(" ", "_")
    clean = re.sub(r"_+", "_", clean).strip("_")
    sha1  = hashlib.sha1(filename.encode()).hexdigest()[:10]
    return f"{clean}_{sha1}"


def _fraction(fps: float) -> Fraction:
    return Fraction(fps).limit_denominator(100_000)


def write_clip(frames: list[np.ndarray], fps: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w   = frames[0].shape[:2]
    w_enc  = w - (w % 2)
    h_enc  = h - (h % 2)
    ctr    = av.open(str(out_path), mode="w")
    stream = ctr.add_stream("libx264", rate=_fraction(fps))
    stream.width   = w_enc
    stream.height  = h_enc
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    for arr in frames:
        frame = av.VideoFrame.from_ndarray(arr[:h_enc, :w_enc], format="rgb24")
        frame = frame.reformat(format="yuv420p")
        for pkt in stream.encode(frame):
            ctr.mux(pkt)
    for pkt in stream.encode():
        ctr.mux(pkt)
    ctr.close()


def write_png(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(out_path))


def process_video(video_path: Path, out_dir: Path, clip_sizes: list[int], dry_run: bool) -> dict:
    filename = video_path.name
    video_id = make_video_id(filename)

    frames_dir = out_dir / "first_last_frames" / video_id
    first_png  = frames_dir / "first.png"
    last_png   = frames_dir / "last.png"

    need_frames   = not first_png.exists() or not last_png.exists()
    missing_clips = [
        n for n in clip_sizes
        if not (out_dir / f"first_last_clips_{n}" / video_id / "first.mp4").exists()
        or not (out_dir / f"first_last_clips_{n}" / video_id / "last.mp4").exists()
    ]

    if not need_frames and not missing_clips:
        return {"video_id": video_id, "source": filename, "status": "skipped_exists"}

    if dry_run:
        return {"video_id": video_id, "source": filename, "status": "dry_run"}

    max_n     = max(missing_clips) if missing_clips else 1
    first_buf: list[np.ndarray] = []
    last_buf  = collections.deque(maxlen=max_n)
    fps = 30.0
    width = height = total_frames = 0

    try:
        container = av.open(str(video_path))
        stream    = container.streams.video[0]
        fps       = float(stream.average_rate or 30)
        width     = stream.width
        height    = stream.height
        for packet in container.demux(stream):
            for frame in packet.decode():
                arr = frame.to_ndarray(format="rgb24")
                total_frames += 1
                if len(first_buf) < max_n:
                    first_buf.append(arr)
                last_buf.append(arr)
        container.close()
    except Exception:
        return {"video_id": video_id, "source": filename, "status": "error",
                "error": traceback.format_exc(limit=3)}

    if total_frames == 0:
        return {"video_id": video_id, "source": filename, "status": "error",
                "error": "no decodable frames"}

    last_list = list(last_buf)

    if need_frames:
        try:
            write_png(first_buf[0],  first_png)
            write_png(last_list[-1], last_png)
        except Exception as exc:
            return {"video_id": video_id, "source": filename, "status": "error",
                    "error": f"frame write failed: {exc}"}

    outputs: dict = {}
    for n in missing_clips:
        clip_dir  = out_dir / f"first_last_clips_{n}" / video_id
        available = min(n, total_frames)
        try:
            write_clip(first_buf[:available],  fps, clip_dir / "first.mp4")
            write_clip(last_list[-available:], fps, clip_dir / "last.mp4")
            outputs[f"first_last_clips_{n}"] = {
                "first": str(clip_dir / "first.mp4"),
                "last":  str(clip_dir / "last.mp4"),
                "actual_frames": available,
            }
        except Exception as exc:
            return {"video_id": video_id, "source": filename, "status": "error",
                    "error": f"clip write failed n={n}: {exc}\n{traceback.format_exc(limit=3)}"}

    return {
        "video_id": video_id, "source": filename, "status": "ok",
        "fps": fps, "width": width, "height": height,
        "total_frames": total_frames, "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*",
                        help="Raw filenames to process (e.g. 'Animals\\\\animal\\\\animal_123.mp4'). "
                             "If omitted, reads one filename per line from stdin.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir

    if not raw_dir.exists():
        sys.exit(f"ERROR: raw-dir not found: {raw_dir}")

    filenames = args.targets if args.targets else [l.strip() for l in sys.stdin if l.strip()]
    if not filenames:
        sys.exit("ERROR: no target filenames provided.")

    video_paths = []
    for fname in filenames:
        vp = raw_dir / fname
        if not vp.exists():
            print(f"[warn] not found: {vp}", file=sys.stderr)
        else:
            video_paths.append(vp)

    if not video_paths:
        sys.exit("ERROR: none of the target files were found in raw-dir.")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    print(f"Targeting {len(video_paths)} video(s) in {raw_dir}")
    print(f"Output → {out_dir}")
    if args.dry_run:
        print("DRY RUN – no files will be written.")
    print()

    t0 = time.time()
    counts: collections.Counter = collections.Counter()
    with open(meta_path, "a") as meta_fh:
        for vp in video_paths:
            print(f"  Processing {vp.name} …")
            result = process_video(vp, out_dir, CLIP_SIZES, args.dry_run)
            counts[result["status"]] += 1
            meta_fh.write(json.dumps(result) + "\n")
            meta_fh.flush()
            print(f"    → {result['status']}  video_id={result['video_id']}")

    print(f"\nDone in {time.time()-t0:.1f}s  ok={counts['ok']}  "
          f"skip={counts['skipped_exists']}  err={counts['error']}")


if __name__ == "__main__":
    main()
