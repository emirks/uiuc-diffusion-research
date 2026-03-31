"""Export DAVIS sequences as MP4 clips at three lengths.

For each sequence referenced in CLASS_PAIRS, writes three MP4s into
data/processed/DAVIS/<sequence>/:
    1s.mp4   – 24 frames  (default: first 24; overridable per sequence)
    2s.mp4   – 48 frames  (default: first 48; overridable per sequence)
    full.mp4 – all frames (always the full sequence; no override)

Per-sequence intervals can be set in CLIP_OVERRIDES:
    (start_frame, end_frame_inclusive)
    Both are 0-indexed. Negative values count from the end of the sequence.
    None for end_frame means "start + default label length" (clamped to total).

Example:
    "mallard-fly": {
        "1s": (20, 43),   # frames 20–43 inclusive (24 frames)
        "2s": (8,  55),   # frames 8–55 inclusive  (48 frames)
    }
    "motocross-bumps": {
        "1s": (-24, None),  # last 24 frames
        "2s": (-48, None),  # last 48 frames
    }

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

# Class pairs used in the LTX2 C2V evaluation.
# Each tuple is (start_sequence, end_sequence).
CLASS_PAIRS = {
    "1_similar_context_similar_category_similar_movement": [
        ("blackswan", "mallard-water"),
    ],
    "2_similar_context_similar_category_different_movement": [
        ("paragliding-launch", "paragliding"),
        ("motocross-bumps", "motocross-jump"),
        # complex movement
        ("mallard-fly", "mallard-water"),
    ],
    "3_similar_context_different_category_similar_movement": [
        # null
    ],
    "4_similar_context_different_category_different_movement": [
        # null
    ],
    "5_different_context_similar_category_similar_movement": [
        ("car-roundabout", "bus"),
        ("car-turn", "car-shadow"),
        ("lucia", "hike"),
        # complex movement
        ("breakdance-flare", "breakdance"),
    ],
    "6_different_context_similar_category_different_movement": [
        ("longboard", "kite-surf"),
    ],
    "7_different_context_different_category_similar_movement": [
        # optical flow study did not give meaningful results
    ],
    "8_different_context_different_category_different_movement": [
        ("blackswan", "boat"),
    ],
}

# Deduplicated list of all sequences referenced in any pair.
SEQUENCES = sorted({
    seq
    for pairs in CLASS_PAIRS.values()
    for (a, b) in pairs
    for seq in (a, b)
})

# Default clip lengths in frames for the "1s" and "2s" labels.
CLIPS = [
    ("1s",   FPS * 1),   # 24 frames by default
    ("2s",   FPS * 2),   # 48 frames by default
    ("full", None),      # all frames — no override supported
]

# Per-sequence, per-label frame-window overrides.
# Format: { "sequence-name": { "label": (start_frame, end_frame_inclusive) } }
# - start_frame: 0-indexed; negative counts from end of sequence.
# - end_frame_inclusive: 0-indexed inclusive; None → start + default_label_frames
#   (clamped to total). Negative counts from end of sequence.
# "full" cannot be overridden here — it always exports all frames.
CLIP_OVERRIDES: dict[str, dict[str, tuple[int | None, int | None]]] = {
    # Take 1s and 2s clips from the END of the motocross-bumps sequence.
    "motocross-bumps": {
        "1s": (-24, None),   # last 24 frames (frames 36–59)
        "2s": (-48, None),   # last 48 frames (frames 12–59)
    },
    # Custom windows for mallard-fly.
    "mallard-fly": {
        "1s": (20, 43),   # frames 20–43 inclusive (24 frames)
        "2s": (8,  55),   # frames 8–55  inclusive (48 frames)
    },
}


def resolve_interval(
    start: int | None,
    end_inclusive: int | None,
    default_count: int | None,
    total: int,
) -> tuple[int, int]:
    """Resolve (start_frame, end_frame_exclusive) from override spec + total frames.

    Args:
        start:         Override start frame (0-indexed, negative = from end). None → 0.
        end_inclusive: Override end frame inclusive (0-indexed, negative = from end).
                       None → start + default_count (clamped to total).
        default_count: Default number of frames for this label (e.g. 24 for "1s").
                       Used only when end_inclusive is None.
        total:         Total frames available in the sequence.

    Returns:
        (start_frame, end_frame_exclusive) both clamped to [0, total].
    """
    s = 0 if start is None else (start if start >= 0 else total + start)
    s = max(0, min(s, total))

    if end_inclusive is not None:
        e = end_inclusive if end_inclusive >= 0 else total + end_inclusive
        e_excl = e + 1
    elif default_count is not None:
        e_excl = s + default_count
    else:
        e_excl = total  # full

    e_excl = max(s, min(e_excl, total))
    return s, e_excl


def ffmpeg_encode(
    src_dir: pathlib.Path,
    out_path: pathlib.Path,
    start_frame: int,
    num_frames: int,
) -> None:
    """Encode a contiguous range of DAVIS JPEG frames to MP4.

    DAVIS frames are 00000.jpg … so we pass -start_number <start_frame>
    to the image2 demuxer and -frames:v <num_frames> to limit output length.
    """
    pattern = str(src_dir / "%05d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(FPS),
        "-start_number", str(start_frame),
        "-i", pattern,
        "-frames:v", str(num_frames),
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

        overrides = CLIP_OVERRIDES.get(seq, {})

        print(f"\n[{seq}]  {total_frames} frames  →  {out_dir.relative_to(REPO_ROOT)}")

        for label, default_count in CLIPS:
            if label == "full":
                # Always export all frames, no override.
                start, end_excl = 0, total_frames
            else:
                ov = overrides.get(label)
                if ov is not None:
                    start, end_excl = resolve_interval(ov[0], ov[1], default_count, total_frames)
                else:
                    start, end_excl = resolve_interval(None, None, default_count, total_frames)

            count = end_excl - start
            suffix = ""
            if label != "full" and default_count is not None and count < default_count:
                suffix = f"  ⚠ only {count} frames available (wanted {default_count})"
            interval_str = f"frames {start}–{end_excl - 1}" if label != "full" else f"frames 0–{total_frames - 1}"
            print(f"  {label}: {count} frames  [{interval_str}]{suffix}")

            out_path = out_dir / f"{label}.mp4"
            ffmpeg_encode(src_dir, out_path, start, count)
            size_kb = out_path.stat().st_size // 1024
            print(f"    → {out_path.name}  ({size_kb} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
