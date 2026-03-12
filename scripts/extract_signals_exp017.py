"""Extract visual signals from exp_017 start/end clip pairs.

Reads sample definitions from:
    experiments/exp_017_ltx2_c2v_category_sweep/config.yaml

For each sample in the config, runs all registered signals on both the
start_clip and the end_clip.

Output layout
-------------
outputs/signals/exp_017_ltx2_c2v_category_sweep/
  run_XXXX/
    <sample_id>/
      start_clip.mp4          ← copy of source start clip
      end_clip.mp4            ← copy of source end clip
      start/
        <signal_slug>/
          raw/<key>/frame_0000.npy …
          viz/frame_0000.png …
          viz.mp4
      end/
        <signal_slug>/
          raw/<key>/frame_0000.npy …
          viz/frame_0000.png …
          viz.mp4

Run directories are auto-incremented (run_0001, run_0002, …) via next_run_dir.
"""
from __future__ import annotations

import dataclasses
import shutil
import sys
import time
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_EXP_CONFIG = _REPO_ROOT / "experiments" / "exp_017_ltx2_c2v_category_sweep" / "config.yaml"

# Signals to run — "all" expands to every registered extractor
SIGNALS: list[str] | str = "all"

# Base output dir — next_run_dir will create run_0001 / run_0002 / … inside
OUTPUT_BASE: Path = _REPO_ROOT / "outputs" / "signals" / "exp_017_ltx2_c2v_category_sweep"

# Optional: set to an existing run dir to fill in missing signals instead of
# starting a new run.  e.g. _REPO_ROOT / "outputs/signals/.../run_0003"
TARGET_RUN_DIR: Path | None = _REPO_ROOT / "outputs" / "signals" / "exp_017_ltx2_c2v_category_sweep" / "run_0003"

DEVICE: str = "cuda"

# Set to True to skip a (sample, role, signal) whose output dir already exists
SKIP_EXISTING: bool = True


# ---------------------------------------------------------------------------

@dataclasses.dataclass
class InputSpec:
    sample_id: str
    role: str       # "start" or "end"
    path: Path


def _load_inputs() -> list[InputSpec]:
    with open(_EXP_CONFIG) as f:
        cfg = yaml.safe_load(f)

    specs: list[InputSpec] = []
    for sample in cfg["samples"]:
        sid = sample["sample_id"]
        specs.append(InputSpec(
            sample_id=sid,
            role="start",
            path=_REPO_ROOT / sample["start_clip"],
        ))
        specs.append(InputSpec(
            sample_id=sid,
            role="end",
            path=_REPO_ROOT / sample["end_clip"],
        ))
    return specs


def _flush_cuda() -> None:
    """Release CUDA memory between extractors so models don't stack up in VRAM.

    Order: synchronize (finish pending kernels) → delete refs → gc → empty_cache
    so the driver can actually reclaim freed memory.
    """
    try:
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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
    all_specs = _load_inputs()

    missing = [s for s in all_specs if not s.path.exists()]
    if missing:
        print("Missing input files:")
        for s in missing:
            print(f"  [{s.sample_id}/{s.role}]  {s.path}")
        sys.exit(1)

    if TARGET_RUN_DIR is not None:
        run_dir = TARGET_RUN_DIR.resolve()
        run_id = run_dir.name
        print(f"  (targeting existing run dir)")
    else:
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        run_id, run_dir = next_run_dir(OUTPUT_BASE)

    print(f"\nRun     : {run_id}  →  {run_dir}")
    print(f"Inputs  : {len(all_specs)} clips  ({len(all_specs) // 2} samples × start+end)")
    print(f"Signals : {selected_slugs}")
    print(f"Device  : {DEVICE}")
    print()

    # ------------------------------------------------------------------
    # Pre-load all frames once; copy source clips into output tree
    # ------------------------------------------------------------------
    # loaded entry: (spec, frames, native_fps, signal_out_root)
    # signal_out_root = run_dir / sample_id / role
    loaded: list[tuple[InputSpec, list, float, Path]] = []

    for spec in all_specs:
        sample_dir = run_dir / spec.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # copy source clip to sample_dir as start_clip.mp4 / end_clip.mp4
        dest_clip = sample_dir / f"{spec.role}_clip.mp4"
        if not dest_clip.exists():
            shutil.copy2(spec.path, dest_clip)

        signal_root = sample_dir / spec.role
        signal_root.mkdir(parents=True, exist_ok=True)

        try:
            frames, native_fps = read_video_frames(spec.path)
        except Exception as exc:
            print(f"  ✗ Failed to read {spec.sample_id}/{spec.role}: {exc}")
            continue

        print(f"  loaded  {spec.sample_id:<35} {spec.role}  "
              f"({len(frames)} frames @ {native_fps:.1f} fps)")
        loaded.append((spec, frames, native_fps, signal_root))

    print()

    # ------------------------------------------------------------------
    # Outer loop: one model load per signal, run across all inputs, unload
    # ------------------------------------------------------------------
    # VRAM: only one extractor in memory at a time; lazy load on first __call__;
    #       after all inputs we del extractor and _flush_cuda() so the next
    #       signal starts with a clean GPU.  RAM: all clip frames kept so each
    #       video is read once and reused for every signal (no re-decode).
    for slug in selected_slugs:
        all_skip = all(
            SKIP_EXISTING and (signal_root / slug).exists()
            for _, _, _, signal_root in loaded
        )
        if all_skip:
            print(f"── {slug:<30}  (all inputs already done, skipped)")
            continue

        print(f"── {slug}")
        cls = SIGNAL_REGISTRY[slug]
        extractor = cls(device=DEVICE)

        for spec, frames, native_fps, signal_root in loaded:
            label = f"{spec.sample_id}/{spec.role}"
            signal_dir = signal_root / slug

            if SKIP_EXISTING and signal_dir.exists():
                print(f"  ↷  {label}  (already exists)")
                continue

            if extractor.input_type == "video" and len(frames) < 2:
                print(f"  ↷  {label}  (skipped — needs ≥2 frames)")
                continue

            t0 = time.perf_counter()
            try:
                result = extractor(frames)
                viz    = extractor.visualize(frames, result)
                save_signal_result(result, slug, signal_root, viz, fps=native_fps)
                print(f"  ✓  {label:<45}  ({time.perf_counter() - t0:.1f}s)")
            except Exception as exc:
                print(f"  ✗  {label}: {exc}")

        del extractor
        _flush_cuda()
        print()

    print(f"\nDone.  [{run_id}  →  {run_dir}]")


if __name__ == "__main__":
    main()
