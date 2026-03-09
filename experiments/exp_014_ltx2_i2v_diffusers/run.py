"""exp_014 — LTX-2 image-to-video, two-stage diffusers pipeline.

Mirrors the official diffusers example for LTX-2:
  https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2

Two-stage flow
──────────────
Stage 1 : LTX2ImageToVideoPipeline  (full 19B model, 40 steps, CFG=4)
          → video_latent + audio_latent  (output_type="latent")

Upsampler: LTX2LatentUpsamplePipeline  (spatial ×2 in latent space)
          → upscaled_video_latent

Stage 2 : same LTX2ImageToVideoPipeline with distilled LoRA (3 steps, CFG=1)
          → final video (np) + audio (np)

Memory controls:
  - enable_sequential_cpu_offload: streams each sub-module on/off GPU
  - vae.enable_tiling():           tiles VAE decode to avoid OOM

To run:
  source /workspace/miniforge3/etc/profile.d/conda.sh
  conda activate /workspace/envs/diff
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python experiments/exp_014_ltx2_i2v_diffusers/run.py
"""
import os
import pathlib
import yaml
import torch

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
    run_id = f"run_{nxt:04d}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def main() -> None:
    cfg = load_config()

    # ── HuggingFace cache ────────────────────────────────────────────────────
    hf_cache = cfg["model"].get("hf_cache_dir")
    if hf_cache:
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache)

    # Deferred imports so env vars are picked up before HF tokenizer/model init
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers import LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
    from diffusers.pipelines.ltx2.export_utils import encode_video
    from diffusers.utils import load_image

    # ── Paths ─────────────────────────────────────────────────────────────────
    image_path = str(REPO_ROOT / cfg["inputs"]["image"])
    out_dir    = REPO_ROOT / cfg["outputs"]["dir"]
    repo_id    = cfg["model"]["repo_id"]

    print(f"[info] image    : {image_path}")
    print(f"[info] repo_id  : {repo_id}")

    # ── dtype & device ────────────────────────────────────────────────────────
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[cfg["model"]["torch_dtype"]]
    device = cfg["runtime"]["device"]
    seed   = cfg["runtime"]["seed"]

    inf = cfg["inference"]
    mem = cfg["memory"]

    # ── Stage 1: load I2V pipeline ────────────────────────────────────────────
    use_fp8 = cfg["model"].get("use_fp8", False)

    if use_fp8:
        from huggingface_hub import hf_hub_download
        fp8_weight = cfg["model"].get("fp8_weight_name", "ltx-2-19b-dev-fp8.safetensors")
        print(f"[info] downloading FP8 transformer weights ({fp8_weight}) …")
        fp8_path = hf_hub_download(
            repo_id=repo_id,
            filename=fp8_weight,
            cache_dir=hf_cache or None,
        )
        print(f"[info] loading FP8 transformer from {fp8_path} …")
        transformer = LTX2VideoTransformer3DModel.from_single_file(
            fp8_path,
            torch_dtype=torch.float8_e4m3fn,
        )
        print("[info] loading LTX2ImageToVideoPipeline (FP8 transformer) …")
        pipe = LTX2ImageToVideoPipeline.from_pretrained(
            repo_id,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )
    else:
        print("[info] loading LTX2ImageToVideoPipeline …")
        pipe = LTX2ImageToVideoPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype)

    if mem.get("enable_sequential_cpu_offload", False):
        print("[info] enabling sequential CPU offload …")
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe.to(device)

    image     = load_image(image_path)
    generator = torch.Generator(device=device).manual_seed(seed)

    print(
        f"[info] Stage 1  seed={seed}  {inf['width']}x{inf['height']}  "
        f"frames={inf['num_frames']}  steps={inf['stage1_num_inference_steps']} …"
    )
    video_latent, audio_latent = pipe(
        image=image,
        prompt=cfg["inputs"]["prompt"],
        negative_prompt=cfg["inputs"]["negative_prompt"],
        height=inf["height"],
        width=inf["width"],
        num_frames=inf["num_frames"],
        frame_rate=inf["frame_rate"],
        num_inference_steps=inf["stage1_num_inference_steps"],
        sigmas=None,
        guidance_scale=inf["stage1_guidance_scale"],
        generator=generator,
        output_type="latent",
        return_dict=False,
    )

    # ── Latent upsampler (spatial ×2) ─────────────────────────────────────────
    print("[info] loading LTX2LatentUpsamplerModel …")
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        repo_id,
        subfolder="latent_upsampler",
        torch_dtype=torch_dtype,
    )
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
    upsample_pipe.enable_model_cpu_offload(device=device)

    print("[info] upsampling latents …")
    upscaled_video_latent = upsample_pipe(
        latents=video_latent,
        output_type="latent",
        return_dict=False,
    )[0]

    # ── Stage 2: distilled LoRA refinement ───────────────────────────────────
    print("[info] loading Stage 2 distilled LoRA …")
    pipe.load_lora_weights(
        repo_id,
        adapter_name="stage_2_distilled",
        weight_name=inf["stage2_lora_weight"],
    )
    pipe.set_adapters("stage_2_distilled", inf["stage2_lora_strength"])

    if mem.get("enable_vae_tiling", False):
        print("[info] enabling VAE tiling …")
        pipe.vae.enable_tiling()

    # Swap to the scheduler config required for Stage 2 distilled sigmas
    new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_dynamic_shifting=False,
        shift_terminal=None,
    )
    pipe.scheduler = new_scheduler

    print(f"[info] Stage 2  steps={inf['stage2_num_inference_steps']}  guidance={inf['stage2_guidance_scale']} …")
    video, audio = pipe(
        latents=upscaled_video_latent,
        audio_latents=audio_latent,
        prompt=cfg["inputs"]["prompt"],
        negative_prompt=cfg["inputs"]["negative_prompt"],
        num_inference_steps=inf["stage2_num_inference_steps"],
        noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
        sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
        guidance_scale=inf["stage2_guidance_scale"],
        generator=generator,
        output_type="np",
        return_dict=False,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    run_id, run_dir = next_run_dir(out_dir)
    video_path = run_dir / f"s{seed}_steps{inf['stage1_num_inference_steps']}+{inf['stage2_num_inference_steps']}.mp4"

    encode_video(
        video[0],
        fps=inf["frame_rate"],
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=str(video_path),
    )

    with (run_dir / "config_snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[done] {run_id}  →  {video_path}")


if __name__ == "__main__":
    main()
