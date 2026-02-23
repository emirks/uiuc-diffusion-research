"""exp_011 — FLF2V equivalence check for WanVideoConnectPipeline.

Runs WanVideoConnectPipeline with anchor_frames=1 using the exact same PNG
inputs, model, and inference settings as exp_009.  Output should be visually
identical to exp_009 if the C2V pipeline correctly degenerates to FLF2V
when given single-frame anchors.

Conditioning equivalence with FLF2V (anchor_frames=1):
  video_condition : [first_frame | zeros(num_frames-2) | last_frame]
  mask            : 1 at pixel-frame 0 and pixel-frame (num_frames-1), 0 elsewhere
  CLIP embed      : encode_image([start_clip[0], end_clip[0]])
All three are structurally identical to what WanImageToVideoPipeline produces.
"""
import pathlib
import numpy as np
import yaml
import torch
import PIL.Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

from diffusion.pipeline_wan_c2v import WanVideoConnectPipeline

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def next_run_dir(out_dir: pathlib.Path) -> tuple[str, pathlib.Path]:
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


def target_resolution(pil_frame: PIL.Image.Image, max_area: int, mod_value: int) -> tuple[int, int]:
    """Compute (height, width) preserving aspect ratio so H*W ≈ max_area,
    snapped to mod_value.  Identical to exp_009's aspect_ratio_resize logic."""
    aspect = pil_frame.height / pil_frame.width
    height = snap_to_grid(round(np.sqrt(max_area * aspect)), mod_value)
    width  = snap_to_grid(round(np.sqrt(max_area / aspect)), mod_value)
    return height, width


def main() -> None:
    cfg = load_config()

    # ── Resolve paths ──────────────────────────────────────────────────────────
    first_path    = REPO_ROOT / cfg["inputs"]["first_frame"]
    last_path     = REPO_ROOT / cfg["inputs"]["last_frame"]
    anchor_frames = cfg["inputs"]["anchor_frames"]
    out_dir       = REPO_ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] first_frame   : {first_path}")
    print(f"[info] last_frame    : {last_path}")
    print(f"[info] anchor_frames : {anchor_frames}")

    # ── Runtime ────────────────────────────────────────────────────────────────
    device = cfg["runtime"]["device"]
    dtype  = torch.bfloat16 if cfg["runtime"]["dtype"] == "bfloat16" else torch.float32
    seed   = cfg["runtime"]["seed"]
    repo   = cfg["model"]["repo_id"]

    print(f"[runtime] device : {device}  dtype : {dtype}")
    print(f"[info] loading pipeline from {repo} …")

    # ── Load pipeline ──────────────────────────────────────────────────────────
    vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
    image_encoder = CLIPVisionModel.from_pretrained(repo, subfolder="image_encoder", torch_dtype=torch.float32)
    pipe = WanVideoConnectPipeline.from_pretrained(
        repo, vae=vae, image_encoder=image_encoder, torch_dtype=dtype,
    )
    if cfg["runtime"].get("cpu_offload", True):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.vae.enable_tiling()

    # ── Load PNG frames (identical inputs to exp_009) ─────────────────────────
    first_frame = load_image(str(first_path))
    last_frame  = load_image(str(last_path))

    # Compute target resolution from first frame — same formula as exp_009.
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height, width = target_resolution(first_frame, cfg["inference"]["max_area"], mod_value)
    print(f"[info] resolution    : {width}x{height}  (mod_value={mod_value})")

    # Wrap as single-frame lists; preprocess_video inside the pipeline handles
    # resize + normalisation to (B, C, 1, H, W) float32 in [-1, 1].
    start_clip = [first_frame]
    end_clip   = [last_frame]

    # ── Generate ───────────────────────────────────────────────────────────────
    generator = torch.Generator(device=device).manual_seed(seed)

    print("[info] running inference …")
    output = pipe(
        start_clip=start_clip,
        end_clip=end_clip,
        anchor_frames=anchor_frames,
        prompt=cfg["inputs"]["prompt"],
        negative_prompt=cfg["inputs"]["negative_prompt"],
        height=height,
        width=width,
        num_frames=cfg["inference"]["num_frames"],
        num_inference_steps=cfg["inference"]["num_inference_steps"],
        guidance_scale=cfg["inference"]["guidance_scale"],
        generator=generator,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    run_id, run_dir = next_run_dir(out_dir)
    video_name = (
        f"s{seed}"
        f"_af{anchor_frames}"
        f"_nf{cfg['inference']['num_frames']}"
        f"_steps{cfg['inference']['num_inference_steps']}"
        f"_cfg{cfg['inference']['guidance_scale']}.mp4"
    )
    video_path = run_dir / video_name

    export_to_video(output.frames[0], str(video_path), fps=cfg["outputs"]["fps"])

    with (run_dir / "config_snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    print(f"[done] {run_id}  →  {video_path}")


if __name__ == "__main__":
    main()
