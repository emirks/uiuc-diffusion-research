"""exp_020 — LTX-2 C2V using the official diffusers LTX2ConditionPipeline.

Migrates exp_016 from the vendored KeyframeInterpolationPipeline to the
HuggingFace diffusers LTX2ConditionPipeline + LTX2LatentUpsamplePipeline.

Key API differences vs exp_016
-------------------------------
- Conditioning: LTX2VideoCondition(frames=list[PIL], index=<latent_idx>, strength)
  instead of ClipConditioningInput(path, frame_idx_px, strength, K).
- Index coordinates: latent frame index (not pixel frame offset).
  end_clip_index = latent_num_frames - clip_latent_frames
                 = (num_frames-1)//8+1  -  (num_clip_frames-1)//8+1
- Frame loading: DAVIS JPEG dirs read directly as PIL lists — no temp MP4.
- Stage-2 conditioning: same conditions object re-passed at full resolution.
  The pipeline resizes frames internally to the requested height/width.
- Memory: enable_sequential_cpu_offload() replaces StateDictRegistry;
  pipeline is created once and called in a loop over samples.

How to run (from repo root, LTX-2 venv with diffusers installed):
    source src/LTX-2/.venv/bin/activate
    pip install "diffusers>=0.33.0" accelerate   # one-time

    export HF_HOME=/workspace/cache/huggingface
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python experiments/exp_020_ltx2_c2v_diffusers/run.py
"""
from __future__ import annotations

import argparse
import glob
import logging
import pathlib
import sys
import time

import torch
import torchvision.io as tio
import yaml
from PIL import Image

# diffusers imports (requires `pip install diffusers>=0.33.0 accelerate`)
from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

from diffusion.exp_utils import load_config, next_run_dir, TeeLogger

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"

# LTX-2 VAE causal temporal downscale factor.
LTX_TEMPORAL_SCALE = 8

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame / clip helpers
# ---------------------------------------------------------------------------

def load_frames_from_dir(dir_path: str | pathlib.Path, num_frames: int, from_end: bool = False) -> list[Image.Image]:
    """Return up to *num_frames* PIL RGB frames from a DAVIS JPEG directory.

    Args:
        dir_path:   Directory containing sequentially named JPEG files.
        num_frames: How many frames to load.
        from_end:   If True, take the *last* num_frames; otherwise the *first*.
    """
    jpgs = sorted(glob.glob(str(pathlib.Path(dir_path) / "*.jpg")))
    if not jpgs:
        raise FileNotFoundError(f"No .jpg files found in {dir_path}")
    jpgs = jpgs[-num_frames:] if from_end else jpgs[:num_frames]
    return [Image.open(p).convert("RGB") for p in jpgs]


def load_frames_from_mp4(mp4_path: str | pathlib.Path, num_frames: int, from_end: bool = False) -> list[Image.Image]:
    """Return up to *num_frames* PIL RGB frames decoded from an MP4 file.

    Args:
        mp4_path:   Path to the MP4 file.
        num_frames: How many frames to return.
        from_end:   If True, take the *last* num_frames; otherwise the *first*.
    """
    # read_video returns (video, audio, info); video is uint8 THWC tensor.
    video, _, _ = tio.read_video(str(mp4_path), pts_unit="sec", output_format="THWC")
    if video.shape[0] == 0:
        raise ValueError(f"No frames decoded from {mp4_path}")
    frames = video[-num_frames:] if from_end else video[:num_frames]
    return [Image.fromarray(f.numpy()) for f in frames]


def load_clip_frames(
    sample: dict,
    repo_root: pathlib.Path,
    num_clip_frames: int,
    start_key_mp4: str = "start_clip",
    end_key_mp4: str   = "end_clip",
    start_key_dir: str = "start_images",
    end_key_dir: str   = "end_images",
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Load start and end conditioning frames from a sample config dict.

    Supports two input formats (matching exp_016 conventions):
      - ``start_clip`` / ``end_clip``: paths to MP4 files.
        start uses first *num_clip_frames* frames; end uses last *num_clip_frames*.
      - ``start_images`` / ``end_images``: DAVIS-style JPEG directories.
        start uses first frames; end uses last frames.
    """
    if start_key_mp4 in sample and end_key_mp4 in sample:
        start = load_frames_from_mp4(repo_root / sample[start_key_mp4], num_clip_frames, from_end=False)
        end   = load_frames_from_mp4(repo_root / sample[end_key_mp4],   num_clip_frames, from_end=True)
    else:
        start = load_frames_from_dir(repo_root / sample[start_key_dir], num_clip_frames, from_end=False)
        end   = load_frames_from_dir(repo_root / sample[end_key_dir],   num_clip_frames, from_end=True)
    return start, end


def compute_end_clip_latent_index(num_frames: int, num_clip_frames: int) -> int:
    """Return the LTX latent frame index where the end clip's first latent starts.

    Ensures the clip's last latent token aligns with the output's last latent
    frame, mirroring the pixel-offset calculation from exp_016 but expressed
    in latent coordinates (as required by LTX2VideoCondition.index).

    Example: num_frames=97 (13 lat), num_clip_frames=25 (4 lat) → index=9.
    """
    latent_num_frames  = (num_frames      - 1) // LTX_TEMPORAL_SCALE + 1
    clip_latent_frames = (num_clip_frames  - 1) // LTX_TEMPORAL_SCALE + 1
    return latent_num_frames - clip_latent_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg     = load_config(args.config)
    out_dir = REPO_ROOT / cfg["outputs"]["dir"]
    run_id, run_dir = next_run_dir(out_dir)

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
            force=True,
        )

        print(f"[info] run_dir : {run_dir}")
        print(f"[info] samples : {len(cfg['samples'])}")

        # ── Model parameters ────────────────────────────────────────────────
        model_id           = cfg["model"]["model_id"]
        distilled_lora_str = cfg["model"]["distilled_lora_strength"]

        # ── Inference parameters ─────────────────────────────────────────────
        num_frames      = cfg["inference"]["num_frames"]
        frame_rate      = float(cfg["inference"]["frame_rate"])
        height          = cfg["inference"]["height"]
        width           = cfg["inference"]["width"]
        stage1_steps    = cfg["inference"]["stage1_num_inference_steps"]
        seed            = cfg["runtime"]["seed"]
        num_clip_frames = cfg["inputs"]["num_clip_frames"]
        start_strength  = cfg["inputs"]["start_clip_strength"]
        end_strength    = cfg["inputs"]["end_clip_strength"]
        negative_prompt = cfg["inputs"]["negative_prompt"].strip()
        use_cross_ts    = cfg["inference"]["use_cross_timestep"]

        # Stage 1 half-resolution
        h1, w1 = height // 2, width // 2

        # Latent frame index for the end clip conditioning
        end_clip_index = compute_end_clip_latent_index(num_frames, num_clip_frames)
        log.info(
            "Clip conditioning: K=%d px  end_clip_latent_index=%d  "
            "(latent_num_frames=%d  clip_latent_frames=%d)",
            num_clip_frames,
            end_clip_index,
            (num_frames - 1) // LTX_TEMPORAL_SCALE + 1,
            (num_clip_frames - 1) // LTX_TEMPORAL_SCALE + 1,
        )

        # ── Load pipeline ─────────────────────────────────────────────────────
        log.info("Loading LTX2ConditionPipeline from %s …", model_id)
        t0 = time.perf_counter()
        pipe = LTX2ConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        # Sequential CPU offload: keeps all model components in CPU RAM between
        # pipeline calls; only the active component is on GPU at any time.
        # This replaces the StateDictRegistry pattern from exp_016.
        pipe.enable_sequential_cpu_offload(device="cuda")
        pipe.vae.enable_tiling()
        log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)

        # Pre-load distilled LoRA (Stage 2). We will enable/disable it per stage.
        log.info("Loading distilled LoRA from %s (subfolder weight) …", model_id)
        pipe.load_lora_weights(
            model_id,
            weight_name="ltx-2-19b-distilled-lora-384.safetensors",
            adapter_name="stage_2_distilled",
        )
        pipe.set_adapters("stage_2_distilled", adapter_weights=distilled_lora_str)

        # Stage-2 scheduler: non-dynamic shifting (required for distilled sigmas)
        stage2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )

        # Load spatial upsampler from HF Hub.
        log.info("Loading spatial upsampler from %s/latent_upsampler …", model_id)
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            model_id,
            subfolder="latent_upsampler",
            torch_dtype=torch.bfloat16,
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
        upsample_pipe.enable_model_cpu_offload(device="cuda")

        # ── Per-sample inference loop ──────────────────────────────────────────
        summary: list[dict] = []

        for idx, sample in enumerate(cfg["samples"]):
            sample_id  = sample["sample_id"]
            prompt     = sample["prompt"].strip()
            sample_dir = run_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            log.info("─── Sample %d/%d  id=%s ───", idx + 1, len(cfg["samples"]), sample_id)
            start_src = sample.get("start_clip") or sample.get("start_images", "")
            end_src   = sample.get("end_clip")   or sample.get("end_images",   "")
            print(f"[info] start : {start_src}")
            print(f"[info] end   : {end_src}")
            print(f"[info] prompt: {prompt[:80].strip()}…")

            log.info("Loading start/end clip frames (%d px frames each) …", num_clip_frames)
            start_frames, end_frames = load_clip_frames(sample, REPO_ROOT, num_clip_frames)

            # Build LTX2VideoCondition objects.
            # index=0           → start clip placed at latent frame 0 (pixel frame 0)
            # index=end_clip_index → end clip placed so its last latent aligns
            #                       with the output's last latent frame.
            # The pipeline encodes the PIL list through the VAE and positions the
            # resulting latent tokens at `index * H_lat * W_lat` in the sequence.
            conditions = [
                LTX2VideoCondition(frames=start_frames, index=0,              strength=start_strength),
                LTX2VideoCondition(frames=end_frames,   index=end_clip_index, strength=end_strength),
            ]

            generator = torch.Generator(device="cuda").manual_seed(seed)

            log.info(
                "Stage 1: %dx%d  frames=%d  steps=%d  seed=%d",
                h1, w1, num_frames, stage1_steps, seed,
            )
            t_infer = time.perf_counter()

            # Stage 1 — base model, standard scheduler
            pipe.disable_lora()
            video_latent, audio_latent = pipe(
                conditions=conditions,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=h1,
                width=w1,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=stage1_steps,
                guidance_scale=cfg["inference"]["guidance_scale"],
                stg_scale=cfg["inference"]["stg_scale"],
                modality_scale=cfg["inference"]["modality_scale"],
                guidance_rescale=cfg["inference"]["guidance_rescale"],
                audio_guidance_scale=cfg["inference"]["audio_guidance_scale"],
                audio_stg_scale=cfg["inference"]["audio_stg_scale"],
                audio_modality_scale=cfg["inference"]["audio_modality_scale"],
                audio_guidance_rescale=cfg["inference"]["audio_guidance_rescale"],
                spatio_temporal_guidance_blocks=cfg["inference"]["stg_blocks"],
                use_cross_timestep=use_cross_ts,
                generator=generator,
                output_type="latent",
                return_dict=False,
            )
            log.info("Stage 1 done in %.1fs.", time.perf_counter() - t_infer)

            # Upsample: Stage-1 latent → ×2 spatial resolution
            log.info("Upsampling latent …")
            t_up = time.perf_counter()
            upscaled_latent = upsample_pipe(
                latents=video_latent,
                output_type="latent",
                return_dict=False,
            )[0]
            log.info("Upsample done in %.1fs.", time.perf_counter() - t_up)

            # Stage 2 — distilled LoRA, distilled sigmas, full resolution.
            # Re-passing conditions anchors the clips at full (512×768) resolution,
            # matching the exp_016 approach where Stage 2 also re-conditions.
            pipe.enable_lora()
            pipe.scheduler = stage2_scheduler
            log.info("Stage 2: %dx%d  steps=%d  seed=%d", height, width, 3, seed)
            t_s2 = time.perf_counter()
            video, audio = pipe(
                latents=upscaled_latent,
                audio_latents=audio_latent,
                conditions=conditions,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=3,
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                guidance_scale=1.0,
                use_cross_timestep=use_cross_ts,
                generator=generator,
                output_type="np",
                return_dict=False,
            )
            elapsed = time.perf_counter() - t_infer
            log.info("Stage 2 done in %.1fs. Total inference: %.1fs.", time.perf_counter() - t_s2, elapsed)

            # Encode to MP4
            video_path = sample_dir / f"s{seed}_K{num_clip_frames}_steps{stage1_steps}.mp4"
            log.info("Encoding video → %s", video_path)
            encode_video(
                video[0],
                fps=int(frame_rate),
                audio=audio[0].float().cpu(),
                audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
                output_path=str(video_path),
            )

            # Per-sample config snapshot
            per_sample_cfg = {
                "sample_id":  sample_id,
                "prompt":     prompt,
                "start_src":  start_src,
                "end_src":    end_src,
                "clip_conditioning": {
                    "num_clip_frames":  num_clip_frames,
                    "start_index":      0,
                    "end_index":        end_clip_index,
                    "start_strength":   start_strength,
                    "end_strength":     end_strength,
                },
                "inference":  cfg["inference"],
                "runtime":    cfg["runtime"],
                "output":     str(video_path),
                "elapsed_s":  round(elapsed, 1),
            }
            with (sample_dir / "config_snapshot.yaml").open("w") as f:
                yaml.safe_dump(per_sample_cfg, f, sort_keys=False, allow_unicode=True)

            summary.append({"sample_id": sample_id, "video": str(video_path), "elapsed_s": round(elapsed, 1)})
            log.info("Saved %s  (%.1fs)", video_path.name, elapsed)

        # ── Run-level artefacts ───────────────────────────────────────────────
        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"run_id": run_id, "samples": summary}, f, sort_keys=False, allow_unicode=True)

        total = sum(s["elapsed_s"] for s in summary)
        log.info("All %d samples done.  Total inference time: %.1fs", len(summary), total)
        for s in summary:
            print(f"[done] {s['sample_id']}  →  {s['video']}")


if __name__ == "__main__":
    main()
