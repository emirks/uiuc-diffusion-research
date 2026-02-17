#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image

try:
    import yaml
except ImportError as e:
    raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from e

try:
    from diffusers import DiffusionPipeline
except ImportError as e:
    raise SystemExit("Missing dependency: diffusers. Install with `pip install diffusers`.") from e


@dataclass(frozen=True)
class RunContext:
    root: Path
    cfg: Dict[str, Any]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a dict at {path}")
    return cfg


def pick_device(name: str) -> str:
    name = name.lower()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name in {"cuda", "cpu"}:
        if name == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return name
    raise ValueError(f"Unsupported device: {name}")


def to_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def resolve_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: torch.Tensor | float, dot_threshold: float = 0.9995) -> torch.Tensor:
    """Spherical linear interpolation for latent guidance scaffolding."""
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=v0.device, dtype=v0.dtype)

    v0_norm = torch.linalg.vector_norm(v0, dim=-1, keepdim=True).clamp_min(1e-8)
    v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True).clamp_min(1e-8)
    u0 = v0 / v0_norm
    u1 = v1 / v1_norm

    dot = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    use_lerp = torch.abs(dot) > dot_threshold

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp_min(1e-8)
    theta_t = theta_0 * t
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    slerp_out = s0 * v0 + s1 * v1

    lerp_out = (1.0 - t) * v0 + t * v1
    return torch.where(use_lerp, lerp_out, slerp_out)


def build_call_kwargs(pipe: DiffusionPipeline, cfg: Dict[str, Any], start_img: Image.Image, end_img: Image.Image) -> Dict[str, Any]:
    call_sig = inspect.signature(pipe.__call__)
    allowed = set(call_sig.parameters.keys())

    inf = cfg["inference"]
    inp = cfg["inputs"]

    candidates: Dict[str, Any] = {
        "prompt": inp["prompt"],
        "negative_prompt": inp.get("negative_prompt"),
        "num_frames": int(inf["num_frames"]),
        "num_inference_steps": int(inf["num_inference_steps"]),
        "guidance_scale": float(inf["guidance_scale"]),
        "height": int(inf["height"]),
        "width": int(inf["width"]),
        "generator": torch.Generator(device=pipe.device).manual_seed(int(cfg["runtime"]["seed"])),
        # Different FLF2V pipelines expose different names.
        "image": start_img,
        "start_image": start_img,
        "first_frame": start_img,
        "image_start": start_img,
        "end_image": end_img,
        "last_frame": end_img,
        "image_end": end_img,
    }

    kwargs = {k: v for k, v in candidates.items() if k in allowed and v is not None}
    if "prompt" not in kwargs:
        raise RuntimeError("Pipeline __call__ does not accept `prompt`; adjust build_call_kwargs for this pipeline API.")
    return kwargs


def extract_frames(output: Any) -> list[Image.Image]:
    # Most diffusers video pipelines expose `frames` as list[list[PIL.Image]] for batch outputs.
    if hasattr(output, "frames"):
        frames = output.frames
        if isinstance(frames, list) and frames:
            first = frames[0]
            if isinstance(first, list):
                return first
            return frames
    raise RuntimeError("Unable to extract frames from pipeline output. Inspect output type and update extract_frames().")


def save_frames(frames: list[Image.Image], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for i, img in enumerate(frames):
        img.save(out_dir / f"frame_{i:04d}.png")


def run(ctx: RunContext) -> None:
    cfg = ctx.cfg
    root = ctx.root

    start_path = resolve_path(root, cfg["inputs"]["start_frame"])
    end_path = resolve_path(root, cfg["inputs"]["end_frame"])

    if not start_path.exists() or not end_path.exists():
        raise FileNotFoundError(
            "Start/end frame not found. Expected files:\n"
            f"- {start_path}\n"
            f"- {end_path}\n"
            "Set inputs.start_frame and inputs.end_frame in config.yaml."
        )

    out_root = resolve_path(root, cfg["outputs"]["root_dir"]) / cfg["experiment_name"]
    ensure_dir(out_root)

    manifest = {
        "experiment_name": cfg["experiment_name"],
        "repo_id": cfg["model"]["repo_id"],
        "start_frame": str(start_path),
        "end_frame": str(end_path),
        "dry_run": bool(cfg["runtime"].get("dry_run", True)),
    }

    with (out_root / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if bool(cfg["runtime"].get("dry_run", True)):
        print(f"[dry-run] config + input validation passed. Manifest written to: {out_root / 'run_manifest.json'}")
        return

    device = pick_device(cfg["runtime"]["device"])
    dtype = to_dtype(cfg["model"]["torch_dtype"])

    print(f"[info] loading pipeline: {cfg['model']['repo_id']}")
    pipe = DiffusionPipeline.from_pretrained(
        cfg["model"]["repo_id"],
        torch_dtype=dtype,
        use_safetensors=bool(cfg["model"].get("use_safetensors", True)),
    )

    if bool(cfg["runtime"].get("enable_model_cpu_offload", False)) and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    start_img = Image.open(start_path).convert("RGB")
    end_img = Image.open(end_path).convert("RGB")

    kwargs = build_call_kwargs(pipe, cfg, start_img, end_img)
    print(f"[info] pipeline call kwargs: {sorted(kwargs.keys())}")

    with torch.no_grad():
        output = pipe(**kwargs)

    frames = extract_frames(output)
    if bool(cfg["outputs"].get("save_frames", True)):
        save_frames(frames, out_root / "frames")

    print(f"[saved] {len(frames)} frames to: {out_root / 'frames'}")


if __name__ == "__main__":
    cfg_path = Path(__file__).with_name("config.yaml")
    context = RunContext(root=repo_root(), cfg=load_config(cfg_path))
    run(context)
