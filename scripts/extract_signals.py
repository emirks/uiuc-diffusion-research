"""Extract visual signals from 6 vc-bench-flf source clips.

Inputs
------
Six clips from data/processed/vc-bench-flf/first_last_clips_24 — the raw
input data used by exp_012.  Chosen to span three activity categories
(action, conversation, dancing) and two clip positions (first / last):

  action_3106432   / first  — exp_012 anchor pair A, 2562×1440
  action_1402988   / last   — exp_012 anchor pair B, 1920×1080
  action_1581362   / first  — exp_012 anchor pair C, 2562×1440
  conversation_3044864 / first — 3840×2160
  dancing_2795733  / first  — 3840×2160
  action_853720    / last   — 1920×1080

Outputs
-------
outputs/signals/vc-bench-flf_first_last_clips_24/run_XXXX/<clip_dir>/<first_or_last>/<signal_slug>/
    raw/<key>/frame_0000.npy …
    viz/frame_0000.png …
    viz.mp4

Run directories are auto-incremented (run_0001, run_0002, …) via next_run_dir.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Hardcoded config — edit here to change behaviour
# ---------------------------------------------------------------------------

_CLIP_DIR = _REPO_ROOT / "data/processed/vc-bench-flf/first_last_clips_24"

INPUTS: list[Path] = [
    # action — exp_012 pair A (2562×1440)
    _CLIP_DIR / "Actions_Activities_action_action_3106432_2562x1440_e99d05828f" / "first.mp4",
    _CLIP_DIR / "Actions_Activities_action_action_1402988_1920x1080_199e157427" / "last.mp4",
    # action — exp_012 pair C (2562×1440)
    _CLIP_DIR / "Actions_Activities_action_action_1581362_2562x1440_b27b9c451a" / "first.mp4",
    # conversation (3840×2160)
    _CLIP_DIR / "Actions_Activities_conversation_conversation_3044864_3840x2160_03d3231485" / "first.mp4",
    # dancing (3840×2160)
    _CLIP_DIR / "Actions_Activities_dancing_dancing_2795733_3840x2160_a76be01eb1" / "first.mp4",
    # action (1920×1080)
    _CLIP_DIR / "Actions_Activities_action_action_853720_1920x1080_dea415b561" / "last.mp4",
]

# Signals to run — "all" expands to every registered extractor
SIGNALS: list[str] | str = "all"

# Base output dir — next_run_dir will create run_0001 / run_0002 / … inside
OUTPUT_BASE: Path = _REPO_ROOT / "outputs" / "signals" / "vc-bench-flf_first_last_clips_24"

DEVICE: str = "cuda"

# Set to True to skip a signal whose output directory already exists
SKIP_EXISTING: bool = True


# ---------------------------------------------------------------------------

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
from diffusion.signals.io import read_video_frames, save_signal_result
from diffusion.exp_utils import next_run_dir


def _resolve_signals() -> list[str]:
    if SIGNALS == "all" or SIGNALS == ["all"] or "all" in SIGNALS:
        return list(SIGNAL_REGISTRY.keys())
    unknown = [s for s in SIGNALS if s not in SIGNAL_REGISTRY]
    if unknown:
        print(f"Unknown signal(s): {unknown}")
        print(f"Available: {list(SIGNAL_REGISTRY.keys())}")
        sys.exit(1)
    return list(SIGNALS)


def main() -> None:
    selected_slugs = _resolve_signals()

    missing = [p for p in INPUTS if not p.exists()]
    if missing:
        print("Missing input files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = next_run_dir(OUTPUT_BASE)

    print(f"\nRun     : {run_id}  →  {run_dir}")
    print(f"Inputs  : {len(INPUTS)} video(s)")
    print(f"Signals : {selected_slugs}")
    print(f"Device  : {DEVICE}")
    print()

    # Pre-load all frames once so we're not re-reading videos inside the signal loop
    loaded: list[tuple[Path, list, float, Path]] = []
    for input_path in INPUTS:
        rel = input_path.relative_to(_CLIP_DIR)
        out_dir = run_dir / rel.parent / input_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            frames, native_fps = read_video_frames(input_path)
        except Exception as exc:
            print(f"✗ Failed to read {rel}: {exc}")
            continue
        shutil.copy2(input_path, out_dir / input_path.name)
        print(f"  loaded  {rel}  ({len(frames)} frames @ {native_fps:.1f} fps)")
        loaded.append((input_path, frames, native_fps, out_dir))

    print()

    # Outer loop: one model load per signal, runs across all inputs, then unloads
    for slug in selected_slugs:
        all_skip = all(
            SKIP_EXISTING and (out_dir / slug).exists()
            for _, _, _, out_dir in loaded
        )
        if all_skip:
            print(f"── {slug:<30}  (all inputs already done, skipped)")
            continue

        print(f"── {slug}")
        cls = SIGNAL_REGISTRY[slug]
        extractor = cls(device=DEVICE)

        for input_path, frames, native_fps, out_dir in loaded:
            rel = input_path.relative_to(_CLIP_DIR)
            signal_dir = out_dir / slug

            if SKIP_EXISTING and signal_dir.exists():
                print(f"  ↷ {rel}  (already exists)")
                continue

            if extractor.input_type == "video" and len(frames) < 2:
                print(f"  ↷ {rel}  (skipped — needs ≥2 frames)")
                continue

            t0 = time.perf_counter()
            try:
                result = extractor(frames)
                viz    = extractor.visualize(frames, result)
                save_signal_result(result, slug, out_dir, viz, fps=native_fps)
                print(f"  ✓ {rel}  ({time.perf_counter() - t0:.1f}s)")
            except Exception as exc:
                print(f"  ✗ {rel}: {exc}")

        del extractor
        _flush_cuda()
        print()

    print(f"\nDone.  [{run_id}  →  {run_dir}]")


if __name__ == "__main__":
    main()
