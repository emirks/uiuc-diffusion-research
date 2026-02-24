"""Shared utilities for experiment run scripts.

Handles common boilerplate so each run.py can focus on the experiment-specific
pipeline call.  Import as:

    from diffusion.exp_utils import load_config, next_run_dir, resolve_resolution, load_clip_from_mp4
"""
from __future__ import annotations

import math
import pathlib
from typing import Optional

import PIL.Image
import torchvision.io as tio
import yaml


def load_config(config_path: pathlib.Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def next_run_dir(out_dir: pathlib.Path) -> tuple[str, pathlib.Path]:
    """Return (run_id, run_dir) for the next sequential run under out_dir."""
    existing = []
    for p in out_dir.glob("run_*"):
        if p.is_dir():
            try:
                existing.append(int(p.name.split("_", 1)[1]))
            except Exception:
                pass
    nxt = (max(existing) + 1) if existing else 1
    run_id  = f"run_{nxt:04d}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def snap_to_grid(value: int, mod: int) -> int:
    return value // mod * mod


def target_resolution(
    pil_frame: PIL.Image.Image, max_area: int, mod_value: int
) -> tuple[int, int]:
    """(height, width) preserving aspect ratio so H*W ≈ max_area, snapped to mod_value."""
    aspect = pil_frame.height / pil_frame.width
    height = snap_to_grid(round(math.sqrt(max_area * aspect)), mod_value)
    width  = snap_to_grid(round(math.sqrt(max_area / aspect)), mod_value)
    return height, width


def resolve_resolution(
    inference_cfg: dict,
    mod_value: int,
    ref_image: Optional[PIL.Image.Image] = None,
) -> tuple[int, int]:
    """Resolve (height, width) from the inference section of a config dict.

    Two modes:
    - Explicit: config has 'height' and 'width' — returned directly.
    - max_area: config has 'max_area' — computed from ref_image aspect ratio,
      snapped to mod_value (= vae_scale_factor_spatial * patch_size).
    """
    if "height" in inference_cfg and "width" in inference_cfg:
        return inference_cfg["height"], inference_cfg["width"]
    if "max_area" in inference_cfg:
        if ref_image is None:
            raise ValueError("ref_image is required when config uses 'max_area'")
        return target_resolution(ref_image, inference_cfg["max_area"], mod_value)
    raise ValueError("inference config must contain 'height'/'width' or 'max_area'")


def load_clip_from_mp4(path: pathlib.Path, anchor_frames: int) -> list[PIL.Image.Image]:
    """Load the first anchor_frames frames of an mp4 as a list of PIL images.

    Uses torchvision.io.read_video (output_format='THWC') returning a
    (T, H, W, 3) uint8 tensor at native resolution.  Resizing is intentionally
    deferred to the pipeline's VideoProcessor.preprocess_video.
    """
    frames_t, _, _ = tio.read_video(str(path), output_format="THWC", pts_unit="sec")
    if len(frames_t) < anchor_frames:
        raise ValueError(
            f"{path.name} has {len(frames_t)} frames but anchor_frames={anchor_frames}"
        )
    return [PIL.Image.fromarray(frames_t[i].numpy()) for i in range(anchor_frames)]
