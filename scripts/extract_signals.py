"""Extract visual signals from video or image inputs.

Usage examples
--------------
# Run all signals on a single video:
    python scripts/extract_signals.py --input outputs/videos/exp_012.../video.mp4

# Run only depth and optical flow:
    python scripts/extract_signals.py \\
        --input outputs/videos/exp_012.../video.mp4 \\
        --signals depth_dav2 flow_raft

# Flat directory — every mp4 at the top level:
    python scripts/extract_signals.py \\
        --input outputs/videos/exp_012_wan21_c2v_anchor_sweep/run_0003/ \\
        --signals all

# Nested directory (vc-bench style: clip_dir/first.mp4 + last.mp4) — use --recursive:
    python scripts/extract_signals.py \\
        --input data/processed/vc-bench-flf/first_last_clips_24/ \\
        --recursive \\
        --signals depth_dav2 flow_raft pose_yolo

# List available signals:
    python scripts/extract_signals.py --list

Outputs
-------
For each video the results land in:

    outputs/signals/<relative-path-from-input-root>/<clip_name>/<video_stem>/
        <signal_slug>/
            raw/<key>/frame_0000.npy …
            viz/frame_0000.png …
            viz.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── repo root on sys.path so we can import diffusion.signals without install ──
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _flush_cuda() -> None:
    """Release CUDA memory between extractors so models don't stack up in VRAM."""
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

from diffusion.signals import SIGNAL_REGISTRY
from diffusion.signals.io import read_video_frames, read_image, save_signal_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _collect_inputs(path: Path, recursive: bool = False) -> list[Path]:
    """Return a sorted list of video/image files to process.

    Non-recursive (default): scans only the top level of a directory.
    Recursive (--recursive):  walks all subdirectories — handles nested layouts
                              like vc-bench where each clip folder holds first.mp4
                              and last.mp4.
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        if recursive:
            candidates = sorted(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES | _IMAGE_SUFFIXES
            )
        else:
            candidates = sorted(
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES | _IMAGE_SUFFIXES
            )
        if not candidates:
            hint = " (try --recursive for nested layouts)" if not recursive else ""
            raise SystemExit(f"No video/image files found under {path}{hint}")
        return candidates
    raise SystemExit(f"Input path does not exist: {path}")


def _load_input(path: Path) -> tuple[list, float]:
    """Return (frames_rgb_list, fps)."""
    if path.suffix.lower() in _VIDEO_SUFFIXES:
        return read_video_frames(path)
    # Single image → list of length 1
    return [read_image(path)], 1.0


def _output_root_for(input_path: Path, input_root: Path, out_base: Path | None) -> Path:
    """Compute per-file output directory, preserving nested structure.

    For a flat file:        outputs/signals/<stem>/
    For a nested file:      outputs/signals/<clip_dir>/<stem>/
      e.g. first_last_clips_24/clip_foo/first.mp4
           → outputs/signals/first_last_clips_24/clip_foo/first/
    """
    base = out_base if out_base else _REPO_ROOT / "outputs" / "signals"
    try:
        rel = input_path.relative_to(input_root)
        # Replace the filename with a stem-only dir component
        return base / rel.parent / input_path.stem
    except ValueError:
        return base / input_path.stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract visual signals (depth, flow, pose, …) from video/images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path,
                        help="Video file, image file, or directory containing them.")
    parser.add_argument("--signals", "-s", nargs="+", default=["all"],
                        metavar="SLUG",
                        help="Signal slugs to run (default: all). "
                             "E.g. --signals depth_dav2 flow_raft")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        metavar="DIR",
                        help="Output root directory. "
                             "Defaults to outputs/signals/<video_stem>/")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device (default: cuda)")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS for output viz videos. Defaults to input FPS.")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Walk subdirectories when --input is a directory. "
                             "Required for nested layouts like vc-bench "
                             "(clip_dir/first.mp4, clip_dir/last.mp4).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a signal if its output directory already exists.")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available signal slugs and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print("\nAvailable signals:")
        print("-" * 50)
        for slug, cls in SIGNAL_REGISTRY.items():
            print(f"  {slug:<22}  {cls.__doc__.splitlines()[0].strip()}")
        print()
        return

    if args.input is None:
        print("Error: --input is required (use --list to see available signals).")
        sys.exit(1)

    # Resolve which signals to run
    if args.signals == ["all"] or "all" in args.signals:
        selected_slugs = list(SIGNAL_REGISTRY.keys())
    else:
        unknown = [s for s in args.signals if s not in SIGNAL_REGISTRY]
        if unknown:
            print(f"Unknown signal(s): {unknown}")
            print(f"Available: {list(SIGNAL_REGISTRY.keys())}")
            sys.exit(1)
        selected_slugs = args.signals

    input_root = args.input.resolve() if args.input.is_dir() else args.input.resolve().parent
    inputs = _collect_inputs(args.input, recursive=args.recursive)

    print(f"\nInputs  : {len(inputs)} file(s)")
    print(f"Signals : {selected_slugs}")
    print(f"Device  : {args.device}")
    print()

    for input_path in inputs:
        input_path = input_path.resolve()
        print(f"── {input_path.relative_to(input_root)}")

        out_root = _output_root_for(input_path, input_root, args.output)
        out_root.mkdir(parents=True, exist_ok=True)

        try:
            frames, native_fps = _load_input(input_path)
        except Exception as exc:
            print(f"  ✗ Failed to read {input_path.name}: {exc}")
            continue

        fps = args.fps if args.fps else native_fps
        print(f"  {len(frames)} frames @ {fps:.1f} fps → {out_root}")

        for slug in selected_slugs:
            signal_dir = out_root / slug
            if args.skip_existing and signal_dir.exists():
                print(f"  ↷ {slug:<30}  (skipped — already exists)")
                continue

            cls = SIGNAL_REGISTRY[slug]
            extractor = cls(device=args.device)

            # Flow and tracking need ≥2 frames; skip single images gracefully
            if extractor.input_type == "video" and len(frames) < 2:
                print(f"  ↷ {slug:<30}  (skipped — needs ≥2 frames)")
                continue

            t0 = time.perf_counter()
            try:
                result  = extractor(frames)
                viz     = extractor.visualize(frames, result)
                save_signal_result(result, slug, out_root, viz, fps=fps)
            except Exception as exc:
                print(f"  ✗ {slug}: {exc}")
                continue
            finally:
                # Release model weights and free CUDA memory before next extractor
                del extractor
                _flush_cuda()
            elapsed = time.perf_counter() - t0
            print(f"     ({elapsed:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
