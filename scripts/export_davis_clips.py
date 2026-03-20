"""Export DAVIS sequences as MP4 clips at three lengths.

For each sequence listed in SEQUENCES, writes three MP4s into
data/processed/DAVIS/<sequence>/:
    1s.mp4   – first 24 frames  (1 second at 24 fps)
    2s.mp4   – first 48 frames  (2 seconds; clamped to available frames)
    full.mp4 – all frames

Uses ffmpeg via subprocess (image2 → libx264, yuv420p, crf 18).

Usage (from repo root):
    python scripts/export_davis_clips.py
"""
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_BASE  = REPO_ROOT / "data/raw/DAVIS/JPEGImages/480p"
OUT_BASE  = REPO_ROOT / "data/processed/DAVIS"
FPS       = 24

SEQUENCES = [
    "camel",
    "horsejump-low",
    "walking",
    "tuk-tuk",
    "paragliding-launch",
    "boxing-fisheye",
    "gold-fish",
    "night-race",
]

CLIPS = [
    ("1s",   FPS * 1),   # 24 frames
    ("2s",   FPS * 2),   # 48 frames (clamped per sequence)
    ("full", None),      # all frames
]


def ffmpeg_encode(
    input_pattern: str,
    out_path: pathlib.Path,
    max_frames: int | None,
) -> None:
    """Run ffmpeg to encode a JPEG sequence to an MP4."""
    cmd = [
        "ffmpeg",
        "-y",                            # overwrite without asking
        "-framerate", str(FPS),
        "-i", input_pattern,
    ]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg failed for {out_path.name}:", file=sys.stderr)
        print(result.stderr[-800:], file=sys.stderr)
        sys.exit(1)


def main() -> None:
    for seq in SEQUENCES:
        src_dir = RAW_BASE / seq
        if not src_dir.is_dir():
            print(f"[skip] {seq} — not found at {src_dir}")
            continue

        total_frames = sum(1 for f in src_dir.iterdir() if f.suffix == ".jpg")
        out_dir = OUT_BASE / seq
        out_dir.mkdir(parents=True, exist_ok=True)

        # ffmpeg image2 demuxer expects a printf-style pattern.
        # DAVIS frames are named 00000.jpg … so %05d.jpg works.
        pattern = str(src_dir / "%05d.jpg")

        print(f"\n[{seq}]  {total_frames} frames  →  {out_dir.relative_to(REPO_ROOT)}")

        for label, max_frames in CLIPS:
            clamped = max_frames
            if max_frames is not None and max_frames > total_frames:
                clamped = total_frames
                print(f"  {label}: requested {max_frames} frames but only {total_frames} available — using {clamped}")
            else:
                display = clamped if clamped is not None else total_frames
                print(f"  {label}: {display} frames")

            out_path = out_dir / f"{label}.mp4"
            ffmpeg_encode(pattern, out_path, clamped)
            size_kb = out_path.stat().st_size // 1024
            print(f"    → {out_path.name}  ({size_kb} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
