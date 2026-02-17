#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Invalid config: {path}")
    return cfg


def resolve_path(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (root / q)


def pick_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[name]


def main() -> None:
    root = repo_root()
    cfg = load_cfg(Path(__file__).with_name("config.yaml"))

    start_path = resolve_path(root, cfg["inputs"]["start_frame"])
    end_path = resolve_path(root, cfg["inputs"]["end_frame"])
    out_dir = resolve_path(root, cfg["outputs"]["root_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not start_path.exists() or not end_path.exists():
        raise FileNotFoundError(
            "Missing start/end frame.\n"
            f"start: {start_path}\n"
            f"end:   {end_path}\n"
            "Update config.yaml paths first."
        )

    if bool(cfg["runtime"].get("dry_run", True)):
        print("[dry-run] inputs/config look valid.")
        print(f"[dry-run] start: {start_path}")
        print(f"[dry-run] end:   {end_path}")
        print(f"[dry-run] out:   {out_dir}")
        return

    dtype = pick_dtype(cfg["model"]["torch_dtype"])
    use_cuda = torch.cuda.is_available() and cfg["runtime"].get("device", "auto") != "cpu"

    print(f"[info] loading: {cfg['model']['repo_id']}")
    pipe = WanImageToVideoPipeline.from_pretrained(
        cfg["model"]["repo_id"],
        torch_dtype=dtype,
        use_safetensors=bool(cfg["model"].get("use_safetensors", True)),
    )

    if use_cuda:
        if bool(cfg["runtime"].get("enable_model_cpu_offload", True)):
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    start_img = Image.open(start_path).convert("RGB")
    end_img = Image.open(end_path).convert("RGB")

    gen_device = "cuda" if use_cuda else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(int(cfg["runtime"]["seed"]))

    print("[info] running FLF2V inference...")
    output = pipe(
        prompt=cfg["inputs"]["prompt"],
        negative_prompt=cfg["inputs"].get("negative_prompt"),
        image=start_img,
        last_image=end_img,
        height=int(cfg["inference"]["height"]),
        width=int(cfg["inference"]["width"]),
        num_frames=int(cfg["inference"]["num_frames"]),
        guidance_scale=float(cfg["inference"]["guidance_scale"]),
        num_inference_steps=int(cfg["inference"]["num_inference_steps"]),
        generator=generator,
    )

    frames = output.frames[0]

    # Save mp4 starter output.
    video_path = out_dir / "sample.mp4"
    export_to_video(frames, str(video_path), fps=int(cfg["outputs"].get("fps", 16)))
    print(f"[saved] {video_path}")


if __name__ == "__main__":
    main()
