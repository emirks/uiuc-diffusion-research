#!/usr/bin/env python3
"""
Preprocess vc-bench-data: extract first/last frames and clips from every video.

Reads:  data/raw/vc-bench-data/  (flat directory; filenames contain backslashes)

Writes: data/processed/vc-bench-flf/
    first_last_frames/{video_id}/first.png  last.png
    first_last_clips_2/{video_id}/first.mp4   last.mp4
    first_last_clips_4/{video_id}/first.mp4   last.mp4
    first_last_clips_8/{video_id}/first.mp4   last.mp4
    first_last_clips_16/{video_id}/first.mp4  last.mp4
    first_last_clips_24/{video_id}/first.mp4  last.mp4
    first_last_clips_36/{video_id}/first.mp4  last.mp4
    first_last_clips_48/{video_id}/first.mp4  last.mp4
    metadata.jsonl

Video-ID convention (matches the existing broken-bak naming):
    stem = filename without .mp4
         → replace backslashes with underscores
         → replace " & " with "_"
         → replace remaining spaces with "_"
    hash = first 10 chars of SHA-1 of the raw filename (including .mp4)
    video_id = f"{stem}_{hash}"

Run:
    python scripts/prepare_vc_bench_flf.py [--dry-run] [--workers N]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR   = REPO_ROOT / "data" / "raw" / "vc-bench-data"
OUT_DIR   = REPO_ROOT / "data" / "processed" / "vc-bench-flf"

CLIP_SIZES = [2, 4, 8, 16, 24, 36, 48]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_video_id(filename: str) -> str:
    """Derive a clean, filesystem-safe video_id from the raw filename."""
    stem = filename[:-4] if filename.endswith(".mp4") else filename
    clean = stem.replace("\\", "_")
    clean = re.sub(r"\s*&\s*", "_", clean)
    clean = clean.replace(" ", "_")
    clean = re.sub(r"_+", "_", clean).strip("_")
    sha1 = hashlib.sha1(filename.encode()).hexdigest()[:10]
    return f"{clean}_{sha1}"


def write_clip(frames: list[np.ndarray], fps: float, out_path: Path) -> None:
    """Write a list of HxWx3 RGB uint8 numpy arrays as an MP4 clip."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    # Ensure dimensions are even (required by yuv420p)
    w_enc = w - (w % 2)
    h_enc = h - (h % 2)
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction_from_float(fps))
    stream.width  = w_enc
    stream.height = h_enc
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    for arr in frames:
        frame = av.VideoFrame.from_ndarray(arr[:h_enc, :w_enc], format="rgb24")
        frame = frame.reformat(format="yuv420p")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def Fraction_from_float(fps: float):
    """Convert fps float to a Fraction for PyAV stream rate."""
    from fractions import Fraction
    return Fraction(fps).limit_denominator(100_000)


def write_png(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(out_path))


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(args: tuple) -> dict:
    """
    Process a single video file.
    Returns a metadata dict with at least keys: video_id, status.

    Skips any individual output (frame PNGs or per-size clips) that already
    exists on disk, so previously completed sizes are never re-written.
    """
    video_path, out_dir, clip_sizes, dry_run = args
    filename  = video_path.name          # e.g. "Animals\cat\cat_123_1920x1080.mp4"
    video_id  = make_video_id(filename)

    frames_dir = out_dir / "first_last_frames" / video_id
    first_png  = frames_dir / "first.png"
    last_png   = frames_dir / "last.png"

    # Determine what is actually missing
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

    # ---- read video in a single pass ----
    # Only need to buffer as many frames as the largest *missing* clip size.
    max_n     = max(missing_clips) if missing_clips else 1
    first_buf = []          # stores first `max_n` frames (ndarray HxWxC uint8)
    last_buf  = collections.deque(maxlen=max_n)
    fps       = 30.0
    width = height = 0
    total_frames = 0

    try:
        container = av.open(str(video_path))
        stream    = container.streams.video[0]
        fps       = float(stream.average_rate or 30)
        width     = stream.width
        height    = stream.height

        for packet in container.demux(stream):
            for frame in packet.decode():
                arr = frame.to_ndarray(format="rgb24")  # HxWx3 uint8
                total_frames += 1
                if len(first_buf) < max_n:
                    first_buf.append(arr)
                last_buf.append(arr)

        container.close()
    except Exception as exc:
        return {
            "video_id":   video_id,
            "source":     filename,
            "status":     "error",
            "error":      traceback.format_exc(limit=3),
        }

    if total_frames == 0:
        return {"video_id": video_id, "source": filename, "status": "error",
                "error": "no decodable frames"}

    last_list = list(last_buf)  # last `min(total_frames, max_n)` frames

    # ---- save first / last frames (PNG) – only if missing ----
    if need_frames:
        try:
            write_png(first_buf[0],  first_png)
            write_png(last_list[-1], last_png)
        except Exception as exc:
            return {"video_id": video_id, "source": filename, "status": "error",
                    "error": f"frame write failed: {exc}"}

    # ---- save clips for each missing clip size ----
    outputs: dict[str, dict] = {}
    for n in missing_clips:
        key = f"first_last_clips_{n}"
        clip_dir  = out_dir / key / video_id
        available = min(n, total_frames)

        first_clip = first_buf[:available]
        # last N from the rolling buffer (may overlap with first when video is short)
        last_clip  = last_list[-available:]

        try:
            write_clip(first_clip, fps, clip_dir / "first.mp4")
            write_clip(last_clip,  fps, clip_dir / "last.mp4")
            outputs[key] = {
                "first": str(clip_dir / "first.mp4"),
                "last":  str(clip_dir / "last.mp4"),
                "actual_frames": available,
            }
        except Exception as exc:
            return {"video_id": video_id, "source": filename, "status": "error",
                    "error": f"clip write failed for n={n}: {exc}\n{traceback.format_exc(limit=3)}"}

    return {
        "video_id":     video_id,
        "source":       filename,
        "status":       "ok",
        "fps":          fps,
        "width":        width,
        "height":       height,
        "total_frames": total_frames,
        "outputs":      outputs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan videos and check what would be done, no file writes.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4).")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Root directory of vc-bench-data (default: %(default)s).")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Output root directory (default: %(default)s).")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir

    if not raw_dir.exists():
        sys.exit(f"ERROR: raw-dir not found: {raw_dir}")

    videos = sorted(raw_dir.glob("*.mp4"))
    if not videos:
        sys.exit(f"ERROR: No .mp4 files found in {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    print(f"Found {len(videos)} videos in {raw_dir}")
    print(f"Output → {out_dir}")
    print(f"Clip sizes: {CLIP_SIZES} frames")
    if args.dry_run:
        print("DRY RUN – no files will be written.")
    print()

    tasks = [(v, out_dir, CLIP_SIZES, args.dry_run) for v in videos]

    counts = collections.Counter()
    t0 = time.time()

    with open(meta_path, "a") as meta_fh:
        if args.workers <= 1:
            # Sequential – easier to debug
            for task in tqdm(tasks, desc="Processing"):
                result = process_video(task)
                counts[result["status"]] += 1
                meta_fh.write(json.dumps(result) + "\n")
                meta_fh.flush()
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(process_video, t): t[0].name for t in tasks}
                with tqdm(total=len(futures), desc="Processing") as pbar:
                    for fut in as_completed(futures):
                        result = fut.result()
                        counts[result["status"]] += 1
                        meta_fh.write(json.dumps(result) + "\n")
                        meta_fh.flush()
                        pbar.update(1)
                        pbar.set_postfix(ok=counts["ok"],
                                         skip=counts["skipped_exists"],
                                         err=counts["error"])

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  ok:      {counts['ok']}")
    print(f"  skipped: {counts['skipped_exists']}")
    print(f"  dry_run: {counts['dry_run']}")
    print(f"  errors:  {counts['error']}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
