#!/usr/bin/env python3
"""Experiment 006 — Wan 2.1 1.3B clip-to-clip video connecting (prompt grid).

Runs a grid of N clip-pairs × M prompts and saves one video per combination.
Clip pairs include a "cross-clip" entry where start and end come from different
source videos.

Usage (from repo root):
    python experiments/exp_006_wan21_clip_connect_1_3b/run.py

Set dry_run: false in config.yaml before running full inference.
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
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
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
    """Load the first num_frames from an mp4, resize to (H, W).

    Returns float32 (1, 3, F, H, W) in [-1, 1].
    """
    import imageio.v3 as iio

    raw = iio.imread(str(path), plugin="pyav")
    if len(raw) < num_frames:
        raise ValueError(
            f"Clip {path} has only {len(raw)} frames; {num_frames} required."
        )
    frames: list[torch.Tensor] = []
    for idx in range(num_frames):
        img = Image.fromarray(raw[idx]).convert("RGB")
        img = center_crop_to_aspect(img, target_w=target_w, target_h=target_h)
        img = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
        frames.append(to_tensor(img))
    clip = torch.stack(frames, dim=1).unsqueeze(0)  # (1, 3, F, H, W)
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
    n = len(iio.imread(str(path), plugin="pyav"))
    if n < required:
        raise SystemExit(
            f"Clip {path} has {n} frames but {required} are required. "
            "Check anchor_frames in config.yaml."
        )


def preflight_export_check() -> None:
    dummy = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(4)]
    with tempfile.TemporaryDirectory(prefix="exp006_preflight_") as td:
        out = Path(td) / "check.mp4"
        export_to_video(dummy, str(out), fps=8)
        if not out.exists() or out.stat().st_size == 0:
            raise SystemExit("export_to_video sanity check failed.")


def validate_resolution(h: int, w: int) -> None:
    if h % 16 != 0 or w % 16 != 0:
        raise SystemExit(
            f"height={h} and width={w} must both be divisible by 16."
        )


def validate_frame_count(num_frames: int, anchor_frames: int, tc: int = 4) -> None:
    if (num_frames - 1) % tc != 0:
        raise SystemExit(
            f"num_frames={num_frames} must satisfy (F-1) % {tc} == 0."
        )
    if (num_frames - 1) // tc + 1 <= 2 * ((anchor_frames - 1) // tc + 1):
        raise SystemExit(
            "num_frames leaves no room for the generated middle section."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    check_dependencies()

    root = repo_root()
    cfg  = load_cfg(Path(__file__).with_name("config.yaml"))

    out_dir = resolve(root, cfg["outputs"]["root_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    target_h      = int(cfg["inference"]["height"])
    target_w      = int(cfg["inference"]["width"])
    num_frames    = int(cfg["inference"]["num_frames"])
    anchor_frames = int(cfg["inference"]["anchor_frames"])

    validate_resolution(target_h, target_w)
    validate_frame_count(num_frames, anchor_frames)

    # Parse grid inputs.
    raw_pairs = cfg["inputs"]["clip_pairs"]
    prompts   = list(cfg["inputs"]["prompts"])
    if not raw_pairs:
        raise SystemExit("config.inputs.clip_pairs must contain at least one entry.")
    if not prompts:
        raise SystemExit("config.inputs.prompts must contain at least one entry.")

    clip_pairs = []
    for entry in raw_pairs:
        pair_id    = entry["id"]
        start_path = resolve(root, entry["start_clip"])
        end_path   = resolve(root, entry["end_clip"])
        if not start_path.exists():
            raise FileNotFoundError(f"[pair={pair_id}] start_clip not found: {start_path}")
        if not end_path.exists():
            raise FileNotFoundError(f"[pair={pair_id}] end_clip not found: {end_path}")
        clip_pairs.append((pair_id, start_path, end_path))

    total_runs = len(clip_pairs) * len(prompts)
    print(f"[info] clip_pairs : {[p[0] for p in clip_pairs]}")
    print(f"[info] prompts    : {len(prompts)}")
    print(f"[info] total_runs : {total_runs}")
    print(f"[info] resolution : {target_w}x{target_h}  frames: {num_frames}  anchor: {anchor_frames}")
    print(f"[info] model      : Wan 2.1 T2V 1.3B  ({cfg['model']['repo_id']})")

    if bool(cfg["runtime"].get("dry_run", True)):
        for pair_id, start_path, end_path in clip_pairs:
            print(f"[dry-run] pair={pair_id}  start={start_path.name}  end={end_path.name}")
        print("[dry-run] Set dry_run: false in config.yaml to run inference.")
        return

    preflight_export_check()
    for pair_id, start_path, end_path in clip_pairs:
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

    print(f"[runtime] device            : {device_str}")
    print(f"[runtime] transformer_dtype : {transformer_dtype}")
    print(f"[runtime] text_encoder_dtype: {text_enc_dtype}")
    print(f"[runtime] vae_dtype         : {vae_dtype}")
    if cuda_available:
        name     = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[runtime] gpu               : {name} ({total_gb:.1f} GiB)")

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

    enable_vae_tiling = bool(cfg["inference"].get("enable_vae_tiling", True))
    base_seed         = int(cfg["runtime"]["seed"])
    fps               = int(cfg["outputs"].get("fps", 16))
    negative_prompt   = cfg["inputs"].get("negative_prompt")
    gen_device        = device_str if use_cuda else "cpu"

    # All 9 videos go into one single run directory.
    run_id, run_dir = next_run_dir(out_dir)
    print(f"[run] id : {run_id}")
    print(f"[run] dir: {run_dir}")

    # Save config snapshot once for the whole run.
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    records = []
    run_counter = 0
    for pair_id, start_path, end_path in clip_pairs:
        print(f"\n[pair] {pair_id}  start={start_path.name}  end={end_path.name}")

        print(f"[preprocess] loading clips ({anchor_frames} frames → {target_w}x{target_h})")
        start_clip = load_clip_tensor(start_path, anchor_frames, target_h, target_w)
        end_clip   = load_clip_tensor(end_path,   anchor_frames, target_h, target_w)
        print(f"[preprocess] start={tuple(start_clip.shape)}  end={tuple(end_clip.shape)}")

        for prompt_idx, prompt in enumerate(prompts):
            run_counter += 1
            run_seed  = base_seed + run_counter - 1
            generator = torch.Generator(device=gen_device).manual_seed(run_seed)

            # Descriptive filename: {pair_id}__prompt_{N}.mp4
            video_name = f"{pair_id}__prompt_{prompt_idx:02d}.mp4"
            video_path = run_dir / video_name

            print(
                f"[run] ({run_counter}/{total_runs})  "
                f"pair={pair_id}  prompt={prompt_idx}  seed={run_seed}  → {video_name}"
            )

            result = pipe(
                start_clip=start_clip,
                end_clip=end_clip,
                prompt=prompt,
                negative_prompt=negative_prompt,
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
            export_to_video(frames, str(video_path), fps=fps)

            records.append({
                "video_file": video_name,
                "pair_id": pair_id,
                "start_clip": str(start_path),
                "end_clip": str(end_path),
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": run_seed,
            })
            print(f"[saved] {video_path}")

    # Write a single manifest with every video and its metadata.
    manifest_path = run_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Also append a summary entry to the experiment-level log.
    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "total_videos": run_counter,
        "videos": [r["video_file"] for r in records],
        "device": device_str,
        "model_repo": repo_id,
        "transformer_dtype": str(transformer_dtype),
        "vae_dtype": str(vae_dtype),
        "config": cfg,
    }
    with (out_dir / "runs_metadata.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\n[done] {run_counter} videos in {run_dir}")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
