#!/usr/bin/env python3
"""Experiment 005 — Wan 2.1 clip-to-clip video connecting.

Loads two 24-frame anchor clips (start + end), runs the hard-constraint
denoising pipeline, and saves the resulting 72-frame video to outputs/.

Usage (from repo root):
    python experiments/exp_005_wan21_clip_connect/run.py

Set dry_run: false in config.yaml before running the full inference.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import AutoencoderKLWan, WanTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from torchvision.transforms.functional import to_tensor
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusion.wan_clip_connect import (
    WanVideoConnectingPipeline,
    normalize_clip_to_neg_one_to_one,
)


# ── Repository helpers ────────────────────────────────────────────────────────

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config at {path} did not parse as a dict.")
    return cfg


def resolve(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (root / q)


def pick_dtype(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from: {list(mapping)}")
    return mapping[name]


def next_run_dir(out_dir: Path) -> tuple[str, Path]:
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


# ── Clip loading ──────────────────────────────────────────────────────────────

def center_crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    src_ar = src_w / src_h
    tgt_ar = target_w / target_h
    if src_ar > tgt_ar:
        new_w = int(round(src_h * tgt_ar))
        left  = (src_w - new_w) // 2
        box   = (left, 0, left + new_w, src_h)
    else:
        new_h = int(round(src_w / tgt_ar))
        top   = (src_h - new_h) // 2
        box   = (0, top, src_w, top + new_h)
    return img.crop(box)


def load_clip_tensor(
    path: Path,
    num_frames: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Load an mp4, take the first num_frames, resize to (H, W).

    Returns a float32 tensor of shape (1, 3, num_frames, H, W) in [-1, 1].
    """
    import imageio.v3 as iio

    # Read all frames as (F, H, W, C) uint8.
    raw = iio.imread(str(path), plugin="pyav")
    if len(raw) < num_frames:
        raise ValueError(
            f"Clip {path} has only {len(raw)} frames, but {num_frames} are required."
        )

    frames: list[torch.Tensor] = []
    for idx in range(num_frames):
        img = Image.fromarray(raw[idx]).convert("RGB")
        img = center_crop_to_aspect(img, target_w=target_w, target_h=target_h)
        img = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
        # to_tensor converts PIL (H, W, C) uint8 → float32 (C, H, W) in [0, 1].
        frames.append(to_tensor(img))

    clip = torch.stack(frames, dim=1)   # (3, F, H, W)
    clip = clip.unsqueeze(0)            # (1, 3, F, H, W)
    return normalize_clip_to_neg_one_to_one(clip)


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def check_dependencies() -> None:
    required = ["imageio", "imageio_ffmpeg", "torchvision"]
    missing  = [n for n in required if importlib.util.find_spec(n) is None]
    if missing:
        raise SystemExit(
            "Missing dependencies: " + ", ".join(missing) + "\n"
            "Install with: pip install " + " ".join(missing)
        )


def check_frame_count(path: Path, required: int) -> None:
    import imageio.v3 as iio
    frames = iio.imread(str(path), plugin="pyav")
    n = len(frames)
    if n < required:
        raise SystemExit(
            f"Clip {path} has {n} frames but {required} are required. "
            "Check anchor_frames in config.yaml."
        )


def preflight_export_check() -> None:
    """Verify export_to_video works before spending time on inference."""
    dummy = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(4)]
    with tempfile.TemporaryDirectory(prefix="exp005_preflight_") as td:
        out = Path(td) / "check.mp4"
        export_to_video(dummy, str(out), fps=8)
        if not out.exists() or out.stat().st_size == 0:
            raise SystemExit("export_to_video sanity check failed.")


def validate_resolution(h: int, w: int) -> None:
    if h % 16 != 0 or w % 16 != 0:
        raise SystemExit(
            f"height={h} and width={w} must both be divisible by 16. "
            f"Try {h - h % 16}x{w - w % 16} or {h - h % 16 + 16}x{w - w % 16 + 16}."
        )


def validate_frame_count(num_frames: int, anchor_frames: int, temporal_compression: int = 4) -> None:
    if (num_frames - 1) % temporal_compression != 0:
        raise SystemExit(
            f"num_frames={num_frames} does not satisfy (F-1) % {temporal_compression} == 0. "
            "Valid values: 5, 9, 13, ..., 69, 73, 77, ..."
        )
    latent_total  = (num_frames    - 1) // temporal_compression + 1
    latent_anchor = (anchor_frames - 1) // temporal_compression + 1
    if latent_total <= 2 * latent_anchor:
        raise SystemExit(
            f"num_frames={num_frames} leaves no room for the middle section. "
            f"Need latent_total > 2 * latent_anchor ({2 * latent_anchor}). "
            f"Currently latent_total={latent_total}."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    check_dependencies()

    root = repo_root()
    cfg  = load_cfg(Path(__file__).with_name("config.yaml"))

    # Resolve paths.
    start_path  = resolve(root, cfg["inputs"]["start_clip"])
    end_path    = resolve(root, cfg["inputs"]["end_clip"])
    out_dir     = resolve(root, cfg["outputs"]["root_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    target_h      = int(cfg["inference"]["height"])
    target_w      = int(cfg["inference"]["width"])
    num_frames    = int(cfg["inference"]["num_frames"])
    anchor_frames = int(cfg["inference"]["anchor_frames"])

    # Input validation.
    validate_resolution(target_h, target_w)
    validate_frame_count(num_frames, anchor_frames)

    if not start_path.exists():
        raise FileNotFoundError(f"start_clip not found: {start_path}")
    if not end_path.exists():
        raise FileNotFoundError(f"end_clip not found: {end_path}")

    print(f"[info] start_clip : {start_path}")
    print(f"[info] end_clip   : {end_path}")
    print(f"[info] output_dir : {out_dir}")
    print(f"[info] resolution : {target_w}x{target_h}  frames: {num_frames}  anchor: {anchor_frames}")

    if bool(cfg["runtime"].get("dry_run", True)):
        print("[dry-run] inputs and config look valid. Set dry_run: false to run inference.")
        return

    preflight_export_check()
    check_frame_count(start_path, anchor_frames)
    check_frame_count(end_path,   anchor_frames)

    # Device / dtype.
    cuda_available   = torch.cuda.is_available()
    requested_device = str(cfg["runtime"].get("device", "auto")).lower()
    use_cuda         = cuda_available and requested_device != "cpu"
    device_str       = "cuda" if use_cuda else "cpu"

    transformer_dtype = pick_dtype(cfg["model"].get("transformer_dtype",  "bfloat16"))
    text_enc_dtype    = pick_dtype(cfg["model"].get("text_encoder_dtype", "bfloat16"))
    vae_dtype         = pick_dtype(cfg["model"].get("vae_dtype",          "float32"))
    repo_id           = cfg["model"]["repo_id"]

    print(f"[runtime] device           : {device_str}")
    print(f"[runtime] transformer_dtype: {transformer_dtype}")
    print(f"[runtime] text_encoder_dtype: {text_enc_dtype}")
    print(f"[runtime] vae_dtype        : {vae_dtype}")
    if cuda_available:
        name     = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[runtime] gpu              : {name} ({total_gb:.1f} GiB)")

    # Load model components individually so each can use the right dtype.
    print(f"[info] loading components from {repo_id} …")
    tokenizer    = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        repo_id, subfolder="text_encoder", torch_dtype=text_enc_dtype
    )
    transformer  = WanTransformer3DModel.from_pretrained(
        repo_id, subfolder="transformer", torch_dtype=transformer_dtype
    )
    scheduler    = FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo_id, subfolder="scheduler"
    )
    vae          = AutoencoderKLWan.from_pretrained(
        repo_id, subfolder="vae", torch_dtype=vae_dtype
    )

    pipe = WanVideoConnectingPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        vae=vae,
    )

    if use_cuda:
        if bool(cfg["runtime"].get("enable_model_cpu_offload", False)):
            print("[runtime] mode: model_cpu_offload")
            pipe.enable_model_cpu_offload()
        else:
            print("[runtime] mode: full model on GPU")
            pipe = pipe.to(device_str)
    else:
        pipe = pipe.to("cpu")

    # Load and preprocess anchor clips.
    print(f"[preprocess] loading clips  ({anchor_frames} frames → {target_w}x{target_h})")
    start_clip = load_clip_tensor(start_path, anchor_frames, target_h, target_w)
    end_clip   = load_clip_tensor(end_path,   anchor_frames, target_h, target_w)
    print(f"[preprocess] start_clip shape: {tuple(start_clip.shape)}")
    print(f"[preprocess] end_clip   shape: {tuple(end_clip.shape)}")

    # Generator (for deterministic output).
    seed      = int(cfg["runtime"]["seed"])
    gen_device = device_str if use_cuda else "cpu"
    generator  = torch.Generator(device=gen_device).manual_seed(seed)
    print(f"[runtime] seed: {seed}")

    # Run inference.
    run_id, run_dir = next_run_dir(out_dir)
    print(f"[run] id : {run_id}")
    print(f"[run] dir: {run_dir}")
    print("[info] running clip-to-clip inference …")

    enable_vae_tiling = bool(cfg["inference"].get("enable_vae_tiling", True))
    result = pipe(
        start_clip=start_clip,
        end_clip=end_clip,
        prompt=cfg["inputs"]["prompt"],
        negative_prompt=cfg["inputs"].get("negative_prompt"),
        num_frames=num_frames,
        height=target_h,
        width=target_w,
        anchor_frames=anchor_frames,
        num_inference_steps=int(cfg["inference"]["num_inference_steps"]),
        guidance_scale=float(cfg["inference"]["guidance_scale"]),
        generator=generator,
        enable_vae_tiling=enable_vae_tiling,
    )
    frames = result.frames[0]

    # Save output.
    fps        = int(cfg["outputs"].get("fps", 16))
    video_path = run_dir / "sample.mp4"
    export_to_video(frames, str(video_path), fps=fps)

    snapshot_path = run_dir / "config_snapshot.yaml"
    with snapshot_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    record = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "video_path": str(video_path),
        "start_clip": str(start_path),
        "end_clip": str(end_path),
        "seed": seed,
        "device": device_str,
        "transformer_dtype": str(transformer_dtype),
        "vae_dtype": str(vae_dtype),
        "config": cfg,
    }
    metadata_path = out_dir / "runs_metadata.jsonl"
    with metadata_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[saved] {video_path}")
    print(f"[saved] {snapshot_path}")
    print(f"[saved] {metadata_path}")
    print("[done]")


if __name__ == "__main__":
    main()
