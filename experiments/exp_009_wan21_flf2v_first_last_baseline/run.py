"""exp_009 — Wan 2.1 FLF2V first-to-last frame video generation.

Uses WanImageToVideoPipeline with the FLF2V-14B-720P model, which was
specifically trained for first+last frame conditioning.
"""
import pathlib
import numpy as np
import yaml
import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

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


def aspect_ratio_resize(image, pipe, max_area):
    """Resize image preserving aspect ratio so total pixels ≈ max_area,
    snapping to the model's spatial patch grid."""
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width  = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image  = image.resize((width, height))
    return image, height, width


def center_crop_resize(image, height, width):
    resize_ratio = max(width / image.width, height / image.height)
    width  = round(image.width  * resize_ratio)
    height = round(image.height * resize_ratio)
    image  = TF.center_crop(image, [width, height])
    return image, height, width


def main() -> None:
    cfg = load_config()

    # ── Resolve paths ──────────────────────────────────────────────────────────
    first_path = REPO_ROOT / cfg["inputs"]["first_frame"]
    last_path  = REPO_ROOT / cfg["inputs"]["last_frame"]
    out_dir    = REPO_ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] first_frame : {first_path}")
    print(f"[info] last_frame  : {last_path}")

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
    pipe = WanImageToVideoPipeline.from_pretrained(
        repo, vae=vae, image_encoder=image_encoder, torch_dtype=dtype,
    )
    if cfg["runtime"].get("cpu_offload", True):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.vae.enable_tiling()

    # ── Prepare images ────────────────────────────────────────────────────────
    first_frame = load_image(str(first_path))
    last_frame  = load_image(str(last_path))

    first_frame, height, width = aspect_ratio_resize(first_frame, pipe, cfg["inference"]["max_area"])
    if last_frame.size != first_frame.size:
        last_frame, _, _ = center_crop_resize(last_frame, height, width)

    print(f"[info] resolution  : {width}x{height}")

    # ── Generate ───────────────────────────────────────────────────────────────
    generator = torch.Generator(device=device).manual_seed(seed)

    print("[info] running inference …")
    output = pipe(
        image=first_frame,
        last_image=last_frame,
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
    video_name = f"s{seed}_steps{cfg['inference']['num_inference_steps']}_cfg{cfg['inference']['guidance_scale']}.mp4"
    video_path = run_dir / video_name

    export_to_video(output.frames[0], str(video_path), fps=cfg["outputs"]["fps"])

    with (run_dir / "config_snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    print(f"[done] {run_id}  →  {video_path}")


if __name__ == "__main__":
    main()
