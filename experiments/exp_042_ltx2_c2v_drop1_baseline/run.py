"""exp_042 — Production C2V baseline with drop1 (two-stage, audio muxed).

Stock LTX-2 two-stage production path, mirroring exp_020:
    Stage 1 (base, non-distilled, CFG=3.2)
        → Latent spatial upsample ×2
            → Stage 2 (distilled LoRA, 3 steps, CFG=1.0)
                → encode_video with model-generated audio

The single deviation from a vanilla pipeline call is a monkey-patch on
`apply_visual_conditioning` that drops the FIRST latent frame of the
end-clip anchor in `cmask`, `clean_latents`, and the initial `latents`.
The patch is a no-op when `conditions=None` (Stage 2 case).

Both Stage 1 and Stage 2 outputs are saved per sample so the base-model
latent and the upscaled distilled refinement are visible side-by-side.

How to run:
    source /workspace/cache/pod_init.sh
    conda activate /workspace/envs/diff
    cd /workspace/diffusion-research
    python experiments/exp_042_ltx2_c2v_drop1_baseline/run.py
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
import types

import numpy as np
import torch
import yaml

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

from diffusion.exp_utils import (
    load_config,
    next_run_dir,
    resolve_resolution,
    load_clip_from_mp4,
    TeeLogger,
)

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"
DEVICE         = "cuda:0"

log = logging.getLogger(__name__)


def _install_drop1_patch(pipe: LTX2ConditionPipeline, log_state: dict) -> None:
    """Patch `pipe.apply_visual_conditioning` so the first latent frame of
    the end-clip anchor is zeroed in cmask, clean_latents, and seed
    latents. End anchor = condition with the largest latent index. When
    `condition_indices` is empty (Stage 2 with `conditions=None`) the
    patch is a no-op.
    """
    orig = type(pipe).apply_visual_conditioning

    def _patched(
        self,
        latents,
        conditioning_mask,
        condition_latents,
        condition_strengths,
        condition_indices,
        latent_height,
        latent_width,
    ):
        latents, cmask, clean = orig(
            self,
            latents,
            conditioning_mask,
            condition_latents,
            condition_strengths,
            condition_indices,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        if not condition_indices:
            return latents, cmask, clean
        end_latent_idx = max(int(i) for i in condition_indices)
        tpf  = latent_height * latent_width
        d_s  = end_latent_idx * tpf
        d_e  = d_s + tpf
        before = int((cmask > 0).sum())
        latents[:, d_s:d_e] = 0.0
        cmask[:,   d_s:d_e] = 0.0
        clean[:,   d_s:d_e] = 0.0
        after  = int((cmask > 0).sum())
        log_state.setdefault("by_sample", []).append({
            "end_latent_idx": end_latent_idx,
            "tpf":            tpf,
            "before":         before,
            "after":          after,
        })
        log.info(
            "drop1 (end_latent_idx=%d, tpf=%d): cmask active tokens %d → %d",
            end_latent_idx, tpf, before, after,
        )
        return latents, cmask, clean

    pipe.apply_visual_conditioning = types.MethodType(_patched, pipe)


def _decode_video_latents(pipe: LTX2ConditionPipeline, latent_5d: torch.Tensor) -> np.ndarray:
    """Decode an unpacked + denormalized 5D video latent to a (F, H, W, 3)
    uint8 numpy array — the same path the pipeline takes for output_type='np'
    with default decode_timestep=0 (i.e. no decode-time renoise)."""
    latents = latent_5d.to(pipe.vae.dtype)
    timestep = None
    if pipe.vae.config.timestep_conditioning:
        timestep = torch.tensor([0.0], device=latents.device, dtype=latents.dtype)
    video = pipe.vae.decode(latents, timestep, return_dict=False)[0]
    video = pipe.video_processor.postprocess_video(video, output_type="np")
    return (np.clip(video[0], 0.0, 1.0) * 255).astype(np.uint8)


def _decode_audio_latents(pipe: LTX2ConditionPipeline, audio_latent: torch.Tensor) -> torch.Tensor:
    """Decode an unpacked + denormalized audio latent to a waveform via
    audio_vae → vocoder, same path as output_type='np' in the pipeline."""
    audio_latents = audio_latent.to(pipe.audio_vae.dtype)
    mel = pipe.audio_vae.decode(audio_latents, return_dict=False)[0]
    return pipe.vocoder(mel)


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
        log.info("run_dir : %s", run_dir)
        log.info("samples : %d", len(cfg["samples"]))

        model_id        = cfg["model"]["model_id"]
        lora_strength   = float(cfg["model"]["distilled_lora_strength"])
        num_frames      = int(cfg["inference"]["num_frames"])
        frame_rate      = float(cfg["inference"]["frame_rate"])
        n_steps         = int(cfg["inference"]["num_inference_steps"])
        gscale          = float(cfg["inference"]["guidance_scale"])
        seed            = int(cfg["runtime"]["seed"])
        num_clip_frames = int(cfg["inputs"]["num_clip_frames"])
        start_strength  = float(cfg["inputs"]["start_clip_strength"])
        end_strength    = float(cfg["inputs"]["end_clip_strength"])
        prompt          = cfg["inputs"]["prompt"].strip()
        negative_prompt = cfg["inputs"]["negative_prompt"].strip()

        log.info("Loading LTX2ConditionPipeline …")
        t0 = time.perf_counter()
        pipe = LTX2ConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=DEVICE)
        pipe.vae.enable_tiling()
        stage1_scheduler = pipe.scheduler

        log.info("Loading distilled LoRA (Stage 2) …")
        pipe.load_lora_weights(
            model_id,
            adapter_name="stage_2_distilled",
            weight_name="ltx-2-19b-distilled-lora-384.safetensors",
        )
        pipe.set_adapters("stage_2_distilled", lora_strength)
        stage2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None,
        )

        log.info("Loading spatial upsampler …")
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            model_id, subfolder="latent_upsampler", torch_dtype=torch.bfloat16,
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
        upsample_pipe.enable_model_cpu_offload(device=DEVICE)
        log.info("Pipeline + LoRA + upsampler ready in %.1fs.", time.perf_counter() - t0)

        ts      = pipe.vae_temporal_compression_ratio
        n_lat   = (num_frames      - 1) // ts + 1
        k_lat   = (num_clip_frames - 1) // ts + 1
        end_idx = n_lat - k_lat
        log.info("end_clip_index = %d  (n_lat=%d, k_lat=%d)", end_idx, n_lat, k_lat)

        log_state: dict = {}
        _install_drop1_patch(pipe, log_state)

        mod_value     = int(pipe.vae_spatial_compression_ratio * pipe.transformer_spatial_patch_size)
        audio_sr      = int(pipe.vocoder.config.output_sampling_rate)
        summary: list[dict] = []

        for sample in cfg["samples"]:
            sample_id  = sample["sample_id"]
            sample_dir = run_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            log.info("─── %s ───", sample_id)

            clip_path   = REPO_ROOT / sample["clip"]
            full_frames = load_clip_from_mp4(clip_path, num_frames)
            ref_image   = full_frames[0]
            height, width = resolve_resolution(cfg["inference"], mod_value, ref_image)
            src_W, src_H  = ref_image.size
            log.info(
                "Source: %s  src_HW=%dx%d  → Stage1 HxW=%dx%d  Stage2 HxW=%dx%d",
                clip_path.name, src_H, src_W, height, width, height * 2, width * 2,
            )

            start_frames = full_frames[:num_clip_frames]
            end_frames   = full_frames[-num_clip_frames:]
            conditions = [
                LTX2VideoCondition(frames=start_frames, index=0,       strength=start_strength),
                LTX2VideoCondition(frames=end_frames,   index=end_idx, strength=end_strength),
            ]

            t_sample = time.perf_counter()

            # ── Stage 1 — base, non-distilled, CFG=gscale ────────────────────
            pipe.scheduler = stage1_scheduler
            pipe.disable_lora()
            log.info("Stage 1: %dx%d  %d steps  CFG=%.2f", height, width, n_steps, gscale)
            t_s1 = time.perf_counter()
            gen_s1 = torch.Generator(device=DEVICE).manual_seed(seed)
            video_latent, audio_latent = pipe(
                conditions=conditions,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=n_steps,
                sigmas=None,
                guidance_scale=gscale,
                generator=gen_s1,
                output_type="latent",
                return_dict=False,
            )
            log.info("Stage 1 done in %.1fs.", time.perf_counter() - t_s1)

            # Decode + save Stage 1 output (base resolution, real audio).
            t_dec = time.perf_counter()
            stage1_video = _decode_video_latents(pipe, video_latent)
            stage1_audio = _decode_audio_latents(pipe, audio_latent)
            log.info("Stage 1 decode in %.1fs.", time.perf_counter() - t_dec)

            stub = f"seed{seed}_steps{n_steps}_cfg{gscale:g}_drop1".replace(".", "p")
            stage1_path = sample_dir / f"stage1_{stub}.mp4"
            encode_video(
                stage1_video,
                fps=int(frame_rate),
                audio=stage1_audio[0].float().cpu(),
                audio_sample_rate=audio_sr,
                output_path=str(stage1_path),
            )
            log.info("Saved %s", stage1_path.name)

            # ── Spatial upsample ×2 on Stage-1 latent ────────────────────────
            t_up = time.perf_counter()
            upscaled_latent = upsample_pipe(
                latents=video_latent, output_type="latent", return_dict=False,
            )[0]
            log.info("Upsample done in %.1fs.", time.perf_counter() - t_up)

            # ── Stage 2 — distilled LoRA, 3 steps, CFG=1.0 ───────────────────
            pipe.scheduler = stage2_scheduler
            pipe.enable_lora()
            log.info("Stage 2: %dx%d  3 steps  distilled", height * 2, width * 2)
            t_s2 = time.perf_counter()
            gen_s2 = torch.Generator(device=DEVICE).manual_seed(seed)
            stage2_video, stage2_audio = pipe(
                latents=upscaled_latent,
                audio_latents=audio_latent,
                prompt=prompt,
                width=width * 2,
                height=height * 2,
                num_frames=num_frames,
                num_inference_steps=3,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                generator=gen_s2,
                guidance_scale=1.0,
                output_type="np",
                return_dict=False,
            )
            log.info("Stage 2 done in %.1fs.", time.perf_counter() - t_s2)

            stage2_path = sample_dir / f"stage2_{stub}.mp4"
            encode_video(
                stage2_video[0],
                fps=int(frame_rate),
                audio=stage2_audio[0].float().cpu(),
                audio_sample_rate=audio_sr,
                output_path=str(stage2_path),
            )
            log.info(
                "Saved %s  (sample total %.1fs)",
                stage2_path.name, time.perf_counter() - t_sample,
            )

            summary.append({
                "sample_id":          sample_id,
                "stage1_HxW":         [int(height),     int(width)],
                "stage2_HxW":         [int(height * 2), int(width * 2)],
                "seed":               seed,
                "stage1_steps":       n_steps,
                "stage1_cfg":         gscale,
                "stage2_steps":       3,
                "stage2_cfg":         1.0,
                "lora_strength":      lora_strength,
                "stage1_video":       f"{sample_id}/{stage1_path.name}",
                "stage2_video":       f"{sample_id}/{stage2_path.name}",
            })
            torch.cuda.empty_cache()

        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"run_id": run_id, "samples": summary}, f, sort_keys=False)
        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "drop1_log.yaml").open("w") as f:
            yaml.safe_dump(log_state, f, sort_keys=False)

        log.info("[done] %s → %s  (%d samples)", run_id, run_dir, len(summary))


if __name__ == "__main__":
    main()
