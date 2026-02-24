"""exp_010 — Wan 2.1 C2V clip-to-clip video connecting baseline.

Uses WanVideoConnectPipeline with the FLF2V-14B-720P model.
start_clip and end_clip anchor the first/last anchor_frames of the output video.

Clips are passed as List[PIL.Image] directly to the pipeline.
VideoProcessor.preprocess_video handles resize + normalisation internally,
producing (1, C, T, H, W) float32 in [-1, 1] range as expected by prepare_latents.
"""
import pathlib
import yaml
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel

from diffusion.pipeline_wan_c2v import WanVideoConnectPipeline
from diffusion.exp_utils import load_config, next_run_dir, resolve_resolution, load_clip_from_mp4

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)

    # ── Resolve paths ──────────────────────────────────────────────────────────
    start_path    = REPO_ROOT / cfg["inputs"]["start_clip"]
    end_path      = REPO_ROOT / cfg["inputs"]["end_clip"]
    anchor_frames = cfg["inputs"]["anchor_frames"]
    out_dir       = REPO_ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] start_clip    : {start_path}")
    print(f"[info] end_clip      : {end_path}")
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

    # ── Load clips as List[PIL.Image] at native resolution ────────────────────
    # Resizing is handled by VideoProcessor.preprocess_video inside the pipeline.
    start_frames = load_clip_from_mp4(start_path, anchor_frames)
    end_frames   = load_clip_from_mp4(end_path,   anchor_frames)

    # ── Compute target resolution ──────────────────────────────────────────────
    # ref_image = first frame of start clip (used when config specifies max_area).
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height, width = resolve_resolution(cfg["inference"], mod_value, start_frames[0])
    print(f"[info] resolution    : {width}x{height}  (mod_value={mod_value})")

    # ── Generate ───────────────────────────────────────────────────────────────
    generator = torch.Generator(device=device).manual_seed(seed)

    print("[info] running inference …")
    output = pipe(
        start_clip=start_frames,
        end_clip=end_frames,
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
