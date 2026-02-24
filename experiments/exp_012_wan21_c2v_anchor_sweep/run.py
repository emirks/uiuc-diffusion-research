"""exp_012 — Wan 2.1 C2V anchor-frames sweep across 4 video pairs.

Runs WanVideoConnectPipeline for every combination of:
  - 4 pairs: 3 self (first→last) + 1 cross (3106432/first → 1581362/first)
  - 6 anchor_frames values: [1, 2, 4, 8, 16, 24]

All clips loaded from first_last_clips_24 (24-frame clips support all anchor
values; load_clip_from_mp4 takes the first N frames, so e.g. N=1 from a
24-frame clip is identical to loading from a 2-frame clip).

anchor_frames=1 degenerates to FLF2V conditioning (single-frame anchors),
matching exp_011's mechanism but loading from MP4 instead of PNG.

Pipeline is loaded once and reused across all 24 runs.
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

    clip_dir      = REPO_ROOT / cfg["inputs"]["clip_dir"]
    pairs         = cfg["inputs"]["pairs"]
    af_list       = cfg["inputs"]["anchor_frames_list"]
    out_dir       = REPO_ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg["runtime"]["device"]
    dtype  = torch.bfloat16 if cfg["runtime"]["dtype"] == "bfloat16" else torch.float32
    seed   = cfg["runtime"]["seed"]
    repo   = cfg["model"]["repo_id"]

    print(f"[runtime] device : {device}  dtype : {dtype}")
    print(f"[info] pairs         : {len(pairs)}")
    print(f"[info] anchor_frames : {af_list}")
    print(f"[info] total runs    : {len(pairs) * len(af_list)}")
    print(f"[info] loading pipeline from {repo} …")

    # ── Load pipeline once ─────────────────────────────────────────────────────
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

    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

    # ── Single run directory for all 24 outputs ────────────────────────────────
    run_id, run_dir = next_run_dir(out_dir)
    print(f"[info] run_dir       : {run_dir}")

    # ── Sweep ──────────────────────────────────────────────────────────────────
    total = len(pairs) * len(af_list)
    done  = 0

    for pair in pairs:
        start_path = clip_dir / pair["start_clip"]
        end_path   = clip_dir / pair["end_clip"]
        pair_name  = pair["name"]

        for anchor_frames in af_list:
            done += 1
            print(f"\n[{done}/{total}] pair={pair_name}  anchor_frames={anchor_frames}")

            # Load exactly anchor_frames frames from the 24-frame clips.
            start_frames = load_clip_from_mp4(start_path, anchor_frames)
            end_frames   = load_clip_from_mp4(end_path,   anchor_frames)

            # Resolution from first frame of start clip.
            height, width = resolve_resolution(cfg["inference"], mod_value, start_frames[0])
            print(f"[info] resolution : {width}x{height}")

            generator = torch.Generator(device=device).manual_seed(seed)

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

            video_name = (
                f"{pair_name}"
                f"_af{anchor_frames}"
                f"_s{seed}"
                f"_nf{cfg['inference']['num_frames']}"
                f"_steps{cfg['inference']['num_inference_steps']}"
                f"_cfg{cfg['inference']['guidance_scale']}.mp4"
            )
            video_path = run_dir / video_name
            export_to_video(output.frames[0], str(video_path), fps=cfg["outputs"]["fps"])
            print(f"[saved] {video_name}")

    with (run_dir / "config_snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    print(f"\n[done] {run_id}  →  {run_dir}  ({total} videos)")


if __name__ == "__main__":
    main()
