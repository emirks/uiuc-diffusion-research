"""Shared utilities for experiment run scripts.

Handles common boilerplate so each run.py can focus on the experiment-specific
pipeline call.  Import as:

    from diffusion.exp_utils import (
        load_config, next_run_dir, resolve_path, resolve_resolution,
        load_clip_from_mp4, compute_num_frames, TeeLogger,
    )
"""
from __future__ import annotations

import math
import pathlib
import sys
from typing import IO, Optional

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


def resolve_path(path: str) -> str:
    """Expand user and resolve to an absolute path string (e.g. for model paths in config)."""
    return str(pathlib.Path(path).expanduser().resolve())


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


def compute_num_frames(anchor_frames: int, target_middle: int = 24) -> int:
    """Return the smallest num_frames = 2*anchor_frames + middle satisfying
    the VAE temporal constraint (num_frames - 1) % 4 == 0, with middle >= target_middle.

    Example results with target_middle=24:
      af=1  → num_frames=29  (middle=27)
      af=2  → num_frames=29  (middle=25)
      af=4  → num_frames=33  (middle=25)
      af=8  → num_frames=41  (middle=25)
      af=16 → num_frames=57  (middle=25)
      af=24 → num_frames=73  (middle=25)
    """
    raw = 2 * anchor_frames + target_middle
    remainder = (raw - 1) % 4
    if remainder != 0:
        raw += 4 - remainder
    return raw


class TeeLogger:
    """Context manager that tees sys.stdout to both the terminal and a log file.

    stderr is left untouched so tqdm/diffusers progress bars render correctly
    in the terminal while all print() output is captured to the log file.

    Usage::

        run_id, run_dir = next_run_dir(out_dir)
        with TeeLogger(run_dir / "run.log"):
            ... # all print() calls go to terminal + run.log
    """

    def __init__(self, log_path: pathlib.Path) -> None:
        self._log_path = log_path
        self._file: Optional[IO[str]] = None
        self._orig: Optional[IO[str]] = None

    # --- file-like interface so sys.stdout = self works ---
    def write(self, data: str) -> None:
        self._orig.write(data)      # type: ignore[union-attr]
        self._file.write(data)      # type: ignore[union-attr]

    def flush(self) -> None:
        self._orig.flush()          # type: ignore[union-attr]
        self._file.flush()          # type: ignore[union-attr]

    def __enter__(self) -> "TeeLogger":
        self._file = open(self._log_path, "w", buffering=1, encoding="utf-8")
        self._orig = sys.stdout
        sys.stdout = self           # type: ignore[assignment]
        return self

    def __exit__(self, *args) -> None:
        sys.stdout = self._orig     # type: ignore[assignment]
        self._file.close()          # type: ignore[union-attr]


def image_dir_to_tmp_mp4(
    image_dir: pathlib.Path,
    num_frames: int,
    out_path: pathlib.Path,
    fps: int = 24,
    from_end: bool = False,
) -> None:
    """Write a short MP4 clip from JPEG frames in a DAVIS-style image directory.

    Frames are selected in sorted filename order.  Pass ``from_end=True`` to
    take the *last* ``num_frames`` (use this for the end-clip of a sequence).

    Args:
        image_dir:  Directory containing numbered JPEG frames (e.g. 00000.jpg).
        num_frames: Number of frames to encode.
        out_path:   Destination MP4 path; parent directory must already exist.
        fps:        Frame rate written into the MP4 container.
        from_end:   Take the last ``num_frames`` instead of the first.
    """
    import numpy as np
    import torch

    frames = sorted(image_dir.glob("*.jpg"))
    if len(frames) < num_frames:
        raise ValueError(
            f"{image_dir.name} has {len(frames)} frames but num_frames={num_frames}"
        )
    selected = frames[-num_frames:] if from_end else frames[:num_frames]
    pil_frames = [PIL.Image.open(p).convert("RGB") for p in selected]
    tensor = torch.from_numpy(
        np.stack([np.array(f) for f in pil_frames], axis=0)
    )  # (T, H, W, 3) uint8
    tio.write_video(str(out_path), tensor, fps=fps)


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
