#!/usr/bin/env python3
"""splice_r1.py — after R1, cut the END anchor from each R1 output and write R2/R3 registries.

For one --seed:
  for every prompt x this seed:
    * locate the R1 output mp4  (out/r1/<arm>/probe__<pid>__<endpoint>__s<seed>.mp4)
    * verify it decodes to exactly 121 frames
    * clip id = probe_<pid>_s<seed>
    * conds/<clip>_start9.mp4  := byte copy of conds/<endpoint>_start9.mp4   (SAME start anchor as R1)
    * conds/<clip>_end9.mp4    := cut_windows(R1 output) frames 112..120     (R1's own last 9 frames)
  write reg/r2_s<seed>.jsonl (full prompt) and reg/r3_s<seed>.jsonl (neutral prompt), sided two
  append rows to reg/splice_manifest.csv

Idempotent: existing windows are kept (start copy is re-verified by sha; end cut is skipped by
cut_windows if present); the per-seed registries are rewritten in full each run.

    python splice_r1.py --seed 42 [PROMPTS_JSONL] [--r1-out out/r1]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import _probe_common as C

sys.path.insert(0, str(C.EVAL_LADDER))
import encode_conditioning as ec  # noqa: E402


def _ensure_libx264_ffmpeg() -> str:
    """cut_windows encodes with `-c:v libx264`, but the DeltaAI spack ffmpeg on PATH is built
    WITHOUT libx264 (only h264_v4l2m2m). The ltx2 env ships imageio-ffmpeg's static binary, which
    DOES have libx264 — the same encoder the existing conds/*_start9.mp4 were made with (h264).
    Point cut_windows at a libx264-capable ffmpeg so the frame selection stays byte-identical to
    the repo recipe. No change to encode_conditioning (the shared contract)."""
    import subprocess

    def has_x264(exe: str) -> bool:
        try:
            return "libx264" in subprocess.run(
                [exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        except Exception:
            return False

    if ec._FFMPEG and has_x264(ec._FFMPEG):
        return ec._FFMPEG
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if has_x264(exe):
            ec._FFMPEG = exe
            print(f"[splice] using imageio-ffmpeg (libx264): {exe}")
            return exe
    except Exception as e:
        print(f"[splice] imageio-ffmpeg unavailable: {e}")
    raise SystemExit("[splice] no libx264-capable ffmpeg found — cannot cut end windows")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def frame_count(path: Path) -> int:
    """Packet count of the video stream (fast: container walk, no decode)."""
    ffprobe = shutil.which("ffprobe") or str(Path.home() / ".local/bin/ffprobe")
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="?", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--r1-out", default=str(C.OUT / "r1"), help="R1 --out-root")
    args = ap.parse_args()

    _ensure_libx264_ffmpeg()
    seed = args.seed
    ppath = Path(args.prompts) if args.prompts else C.default_prompts()
    prompts = C.load_prompts(ppath)
    r1_root = Path(args.r1_out)

    r2_rows, r3_rows, manifest = [], [], []
    errors = []
    for p in prompts:
        pid = p["prompt_id"]
        ep = p["endpoint"]
        r1_mp4 = C.out_mp4("r1", C.r1_item_id(p), seed, out_root=r1_root)
        if not r1_mp4.exists():
            errors.append(f"{pid}: R1 output missing {r1_mp4}")
            continue
        nfr = frame_count(r1_mp4)
        if nfr != ec.STD_FRAMES:
            errors.append(f"{pid}: R1 output {r1_mp4.name} has {nfr} frames (want {ec.STD_FRAMES})")
            continue

        clip = C.probe_clip(pid, seed)
        src_start = C.CONDS / f"{ep}_start9.mp4"
        dst_start = C.CONDS / f"{clip}_start9.mp4"
        dst_end = C.CONDS / f"{clip}_end9.mp4"

        # start window: byte copy of the real endpoint start (same anchor R1 saw)
        if not dst_start.exists():
            shutil.copyfile(src_start, dst_start)
        start_sha_match = sha256(dst_start) == sha256(src_start)
        if not start_sha_match:
            errors.append(f"{pid}: start window copy sha mismatch for {clip}")

        # end window: cut frames 112..120 from the R1 output (start9 already present -> skipped)
        ec.cut_windows(r1_mp4, clip, C.CONDS)
        if not dst_end.exists():
            errors.append(f"{pid}: end window not produced for {clip}")
            continue

        r2_rows.append(C.build_r2r3_row(p, seed, "R2"))
        r3_rows.append(C.build_r2r3_row(p, seed, "R3"))
        manifest.append(dict(
            prompt_id=pid, seed=seed, real_endpoint=ep, probe_clip=clip,
            r1_path=str(r1_mp4.relative_to(C.REPO_ROOT)),
            start_window=str(dst_start.relative_to(C.REPO_ROOT)),
            end_window=str(dst_end.relative_to(C.REPO_ROOT)),
            r1_frames=nfr, start_sha_match=start_sha_match,
            start_sha=sha256(dst_start)[:16],
        ))

    C.write_jsonl(C.REG / f"r2_s{seed}.jsonl", r2_rows)
    C.write_jsonl(C.REG / f"r3_s{seed}.jsonl", r3_rows)

    # append/merge manifest (idempotent by (prompt_id, seed))
    man_path = C.REG / "splice_manifest.csv"
    fields = ["prompt_id", "seed", "real_endpoint", "probe_clip", "r1_path",
              "start_window", "end_window", "r1_frames", "start_sha_match", "start_sha"]
    existing = {}
    if man_path.exists():
        for row in csv.DictReader(open(man_path)):
            existing[(row["prompt_id"], row["seed"])] = row
    for m in manifest:
        existing[(m["prompt_id"], str(m["seed"]))] = m
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in sorted(existing, key=lambda t: (t[1], t[0])):
            w.writerow({k2: existing[k].get(k2, "") for k2 in fields})

    print(f"[splice] seed={seed}: wrote {len(r2_rows)} R2 + {len(r3_rows)} R3 rows; "
          f"manifest now {len(existing)} (prompt,seed) entries")
    if errors:
        print(f"[splice] {len(errors)} PROBLEM(S):")
        for e in errors:
            print("   -", e)
        raise SystemExit(1)
    print("[splice] OK")


if __name__ == "__main__":
    main()
