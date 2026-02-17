#!/usr/bin/env python3
from __future__ import annotations

"""
Run with something like:

cd /workspace/diffusion-research
python scripts/prepare_vcbench_flf_data.py \
  --local_scratch_dir /tmp/vcbench_scratch \
  --workers 4 \
  --quality_mode near_lossless
"""


import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_lengths(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("No clip lengths provided")
    if any(v <= 0 for v in vals):
        raise ValueError("Clip lengths must be positive")
    return sorted(set(vals))


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return s or "video"


def video_id_for(path: Path, raw_root: Path) -> str:
    rel = str(path.relative_to(raw_root))
    stem = sanitize_name(path.stem)
    h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{h}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def dlog(debug: bool, msg: str) -> None:
    if debug:
        print(msg)


def run_cmd(cmd: list[str], *, debug: bool, label: str) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip().splitlines()[-3:]
        detail = " | ".join(tail) if tail else "no stderr"
        raise RuntimeError(f"{label} failed: {detail}") from e


def build_encode_args(quality_mode: str, encode_preset: str, encode_crf: int) -> list[str]:
    if quality_mode == "lossless":
        return ["-c:v", "libx264", "-preset", encode_preset, "-crf", "0", "-qp", "0"]
    return ["-c:v", "libx264", "-preset", encode_preset, "-crf", str(encode_crf), "-pix_fmt", "yuv420p"]


def ffprobe_video_info(video_path: Path, *, debug: bool) -> tuple[int, float]:
    # Fast probe first (no full decode).
    fast_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    fast_out = subprocess.run(fast_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fast_data = json.loads(fast_out.stdout)
    streams = fast_data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream")

    s0 = streams[0]
    fps_str = s0.get("avg_frame_rate") or s0.get("r_frame_rate") or "0/1"
    num, den = fps_str.split("/")
    fps = float(num) / float(den)
    if fps <= 0:
        fps = 16.0

    nb = s0.get("nb_frames")
    if nb not in (None, "N/A"):
        dlog(debug, f"[probe] fast nb_frames used: {video_path.name} -> {nb} frames @ {fps:.3f} fps")
        return int(nb), fps

    # Fallback: exact count (slow, requires full decode).
    dlog(debug, f"[probe] fast probe missing nb_frames, fallback to full decode count: {video_path.name}")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream")
    s0 = streams[0]
    nb = s0.get("nb_read_frames")
    if nb in (None, "N/A"):
        raise RuntimeError("Could not read frame count (nb_read_frames)")
    total_frames = int(nb)
    dlog(debug, f"[probe] full decode count used: {video_path.name} -> {total_frames} frames @ {fps:.3f} fps")

    return total_frames, fps


def extract_all_outputs_one_pass(
    video_path: Path,
    out_root: Path,
    vid: str,
    lengths: list[int],
    total_frames: int,
    fps: float,
    overwrite: bool,
    *,
    debug: bool,
    quality_mode: str,
    encode_preset: str,
    encode_crf: int,
) -> None:
    ff_dir = out_root / "first_last_frames" / vid
    ensure_dir(ff_dir)
    for n in lengths:
        ensure_dir(out_root / f"first_last_clips_{n}" / vid)

    split_labels = [f"v{i}" for i in range(2 + 2 * len(lengths))]
    split_expr = "[0:v]split=" + str(len(split_labels)) + "".join(f"[{x}]" for x in split_labels)
    parts = [split_expr]

    last_idx = total_frames - 1
    parts.append(f"[{split_labels[0]}]select='eq(n\\,0)'[first_frame]")
    parts.append(f"[{split_labels[1]}]select='eq(n\\,{last_idx})'[last_frame]")

    base = 2
    for i, n in enumerate(lengths):
        first_in = split_labels[base + 2 * i]
        last_in = split_labels[base + 2 * i + 1]
        first_out = f"first_{n}"
        last_out = f"last_{n}"

        first_vf = f"[{first_in}]trim=start_frame=0:end_frame={n},setpts=PTS-STARTPTS,fps={fps:.6f}"
        if total_frames < n:
            pad_seconds = (n - total_frames) / max(fps, 1e-6)
            first_vf += f",tpad=stop_mode=clone:stop_duration={pad_seconds:.6f}"
        parts.append(first_vf + f"[{first_out}]")

        last_start = max(0, total_frames - n)
        last_end = last_start + n
        last_vf = f"[{last_in}]trim=start_frame={last_start}:end_frame={last_end},setpts=PTS-STARTPTS,fps={fps:.6f}"
        available = total_frames - last_start
        if available < n:
            pad_seconds = (n - available) / max(fps, 1e-6)
            last_vf += f",tpad=stop_mode=clone:stop_duration={pad_seconds:.6f}"
        parts.append(last_vf + f"[{last_out}]")

    filter_complex = ";".join(parts)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")
    cmd += ["-i", str(video_path), "-filter_complex", filter_complex]

    # First/last frame PNG outputs.
    cmd += ["-map", "[first_frame]", "-frames:v", "1", str(ff_dir / "first.png")]
    cmd += ["-map", "[last_frame]", "-frames:v", "1", str(ff_dir / "last.png")]

    encode_args = build_encode_args(quality_mode, encode_preset, encode_crf)
    for n in lengths:
        cdir = out_root / f"first_last_clips_{n}" / vid
        cmd += ["-map", f"[first_{n}]", "-frames:v", str(n), "-map_metadata", "0"] + encode_args + [str(cdir / "first.mp4")]
        cmd += ["-map", f"[last_{n}]", "-frames:v", str(n), "-map_metadata", "0"] + encode_args + [str(cdir / "last.mp4")]

    dlog(debug, f"[extract] one-pass ffmpeg for {video_path.name}")
    run_cmd(cmd, debug=debug, label=f"extract_one_pass:{video_path.name}")


def expected_outputs_exist(out_root: Path, vid: str, lengths: Iterable[int]) -> bool:
    ff = out_root / "first_last_frames" / vid
    if not (ff / "first.png").exists() or not (ff / "last.png").exists():
        return False
    for n in lengths:
        cdir = out_root / f"first_last_clips_{n}" / vid
        if not (cdir / "first.mp4").exists() or not (cdir / "last.mp4").exists():
            return False
    return True


def process_video(
    video_path: Path,
    raw_root: Path,
    out_root: Path,
    lengths: list[int],
    overwrite: bool,
    verbose: bool,
    debug: bool,
    local_scratch_dir: str | None,
    keep_scratch: bool,
    quality_mode: str,
    encode_preset: str,
    encode_crf: int,
) -> dict:
    vid = video_id_for(video_path, raw_root)
    rel = str(video_path.relative_to(raw_root))

    if not overwrite and expected_outputs_exist(out_root, vid, lengths):
        return {"video_id": vid, "source": rel, "status": "skipped_exists"}

    if verbose:
        print(f"[video] start: {rel}")

    # Optional: work on pod-local scratch to reduce network storage I/O overhead.
    scratch_job_root: Path | None = None
    work_video = video_path
    work_out_root = out_root
    if local_scratch_dir:
        scratch_base = Path(local_scratch_dir).resolve()
        scratch_job_root = scratch_base / "vcbench_flf_work" / vid
        ensure_dir(scratch_job_root)
        work_video = scratch_job_root / f"source{video_path.suffix.lower()}"
        shutil.copy2(video_path, work_video)
        work_out_root = scratch_job_root / "out"
        dlog(debug, f"[scratch] copied source -> {work_video}")

    total_frames, fps = ffprobe_video_info(work_video, debug=debug)
    if total_frames <= 0:
        raise RuntimeError("Video has zero frames")

    ff_dir = work_out_root / "first_last_frames" / vid
    extract_all_outputs_one_pass(
        work_video,
        work_out_root,
        vid,
        lengths,
        total_frames,
        fps,
        overwrite=True,
        debug=debug,
        quality_mode=quality_mode,
        encode_preset=encode_preset,
        encode_crf=encode_crf,
    )
    if verbose:
        print(f"[video] saved frames: {rel}")

    for n in lengths:
        if verbose:
            print(f"[video] saved clips_{n}: {rel}")

    # Sync final artifacts back to network storage once per output file.
    if scratch_job_root is not None:
        final_ff = out_root / "first_last_frames" / vid
        ensure_dir(final_ff)
        shutil.copy2(ff_dir / "first.png", final_ff / "first.png")
        shutil.copy2(ff_dir / "last.png", final_ff / "last.png")
        for n in lengths:
            src_cdir = work_out_root / f"first_last_clips_{n}" / vid
            dst_cdir = out_root / f"first_last_clips_{n}" / vid
            ensure_dir(dst_cdir)
            shutil.copy2(src_cdir / "first.mp4", dst_cdir / "first.mp4")
            shutil.copy2(src_cdir / "last.mp4", dst_cdir / "last.mp4")
        dlog(debug, f"[scratch] synced outputs -> {out_root}")
        if not keep_scratch:
            shutil.rmtree(scratch_job_root, ignore_errors=True)
            dlog(debug, f"[scratch] cleaned -> {scratch_job_root}")

    if verbose:
        print(f"[video] done: {rel}")

    return {
        "video_id": vid,
        "source": rel,
        "status": "ok",
        "fps": fps,
        "total_frames": total_frames,
    }


def find_videos(raw_root: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    vids: list[Path] = []
    for p in raw_root.rglob("*"):
        if not p.is_file():
            continue
        if "/.cache/" in p.as_posix():
            continue
        if p.suffix.lower() in exts:
            vids.append(p)
    vids.sort()
    return vids


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare VC-Bench first/last frames and clips from all videos")
    ap.add_argument("--raw_root", type=str, default="data/raw/vc-bench-data")
    ap.add_argument("--out_root", type=str, default="data/processed/vc-bench-flf")
    ap.add_argument("--lengths", type=str, default="24,36,48")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="If >0, process only first N videos")
    ap.add_argument("--workers", type=int, default=1, help="Parallel video workers (use >1 to speed up)")
    ap.add_argument("--quiet", action="store_true", help="Reduce per-video progress logs")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logs for probe/fallback/command failures")
    ap.add_argument(
        "--local_scratch_dir",
        type=str,
        default="",
        help="Optional pod-local directory for per-video processing (example: /tmp/vcbench_scratch).",
    )
    ap.add_argument("--keep_scratch", action="store_true", help="Keep scratch job folders for debugging.")
    ap.add_argument(
        "--quality_mode",
        type=str,
        default="near_lossless",
        choices=["lossless", "near_lossless", "balanced"],
        help="Clip encoding profile. near_lossless is usually best speed/quality tradeoff.",
    )
    ap.add_argument("--encode_preset", type=str, default="", help="Override x264 preset for clips.")
    ap.add_argument("--encode_crf", type=int, default=-1, help="Override CRF for non-lossless modes.")
    args = ap.parse_args()

    root = repo_root()
    raw_root = (root / args.raw_root).resolve()
    out_root = (root / args.out_root).resolve()
    lengths = parse_lengths(args.lengths)

    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    ensure_dir(out_root)
    videos = find_videos(raw_root)
    if args.limit > 0:
        videos = videos[: args.limit]

    print(f"[info] raw_root: {raw_root}")
    print(f"[info] out_root: {out_root}")
    print(f"[info] lengths: {lengths}")
    print(f"[info] videos found: {len(videos)}")

    records: list[dict] = []
    errors = 0
    workers = max(1, int(args.workers))

    verbose = not args.quiet
    debug = bool(args.debug)
    local_scratch_dir = args.local_scratch_dir.strip() or None

    quality_mode = str(args.quality_mode)
    default_preset = {"lossless": "ultrafast", "near_lossless": "medium", "balanced": "fast"}[quality_mode]
    default_crf = {"lossless": 0, "near_lossless": 12, "balanced": 18}[quality_mode]
    encode_preset = (args.encode_preset or default_preset).strip()
    encode_crf = int(args.encode_crf) if int(args.encode_crf) >= 0 else default_crf

    if local_scratch_dir:
        ensure_dir(Path(local_scratch_dir).resolve())
        print(f"[info] local_scratch_dir: {Path(local_scratch_dir).resolve()}")
    print(f"[info] quality_mode: {quality_mode}, preset: {encode_preset}, crf: {encode_crf}")

    if workers == 1:
        for i, vp in enumerate(videos, start=1):
            try:
                rec = process_video(
                    vp,
                    raw_root,
                    out_root,
                    lengths,
                    overwrite=args.overwrite,
                    verbose=verbose,
                    debug=debug,
                    local_scratch_dir=local_scratch_dir,
                    keep_scratch=bool(args.keep_scratch),
                    quality_mode=quality_mode,
                    encode_preset=encode_preset,
                    encode_crf=encode_crf,
                )
                records.append(rec)
                if i % 20 == 0 or rec.get("status") != "ok":
                    print(f"[{i}/{len(videos)}] {rec['status']}: {rec['source']}")
            except Exception as e:
                errors += 1
                rec = {
                    "video_id": video_id_for(vp, raw_root),
                    "source": str(vp.relative_to(raw_root)),
                    "status": "error",
                    "error": str(e),
                }
                records.append(rec)
                print(f"[{i}/{len(videos)}] error: {rec['source']} :: {e}")
    else:
        try:
            executor_cls = ProcessPoolExecutor
            ex_test = executor_cls(max_workers=workers)
            ex_test.shutdown(wait=True)
        except Exception:
            # Some sandboxes disallow multiprocessing semaphores; thread pool still parallelizes ffmpeg subprocesses.
            print("[warn] ProcessPool unavailable here; falling back to ThreadPoolExecutor.")
            executor_cls = ThreadPoolExecutor

        with executor_cls(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    process_video,
                    vp,
                    raw_root,
                    out_root,
                    lengths,
                    args.overwrite,
                    verbose,
                    debug,
                    local_scratch_dir,
                    bool(args.keep_scratch),
                    quality_mode,
                    encode_preset,
                    encode_crf,
                ): vp
                for vp in videos
            }
            done = 0
            for fut in as_completed(futs):
                vp = futs[fut]
                done += 1
                try:
                    rec = fut.result()
                except Exception as e:
                    errors += 1
                    rec = {
                        "video_id": video_id_for(vp, raw_root),
                        "source": str(vp.relative_to(raw_root)),
                        "status": "error",
                        "error": str(e),
                    }
                records.append(rec)
                if done % 20 == 0 or rec.get("status") != "ok":
                    print(f"[{done}/{len(videos)}] {rec['status']}: {rec['source']}")

    meta_path = out_root / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in records if r.get("status") == "ok")
    skipped = sum(1 for r in records if r.get("status") == "skipped_exists")
    print("[done]")
    print(f"[summary] ok={ok}, skipped={skipped}, errors={errors}, total={len(records)}")
    print(f"[summary] metadata={meta_path}")


if __name__ == "__main__":
    main()
