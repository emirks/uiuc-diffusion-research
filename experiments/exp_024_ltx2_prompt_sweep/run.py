"""exp_024 — LTX-2 prompt optimization sweep.

Measures how prompt variation alone moves transition quality, given fixed
source/target conditioning clips from the DAVIS subset.  Everything fixed
except prompt text (6 categories × 10 pairs = 60 runs).

Forked from exp_020.  Key parameter changes vs exp_020:
  - num_frames: 193  (≈8 s at 24 fps, was 121/5 s)
  - guidance_scale: 3.2  (was 4.0)
  - Config structure: sample["prompt_variants"] list instead of single prompt.
  - enhance_prompt: not supported by LTX2ConditionPipeline (Diffusers) — omitted.

How to run:
    source /workspace/miniforge3/etc/profile.d/conda.sh
    conda activate /workspace/envs/diff
    cd /workspace/diffusion-research
    python experiments/exp_024_ltx2_prompt_sweep/run.py
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

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

from diffusion.exp_utils import load_config, next_run_dir, TeeLogger

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"

LTX_TEMPORAL_SCALE = 8
DEVICE             = "cuda:0"

log = logging.getLogger(__name__)


# ── Frame / clip helpers ──────────────────────────────────────────────────────

def load_frames_from_mp4(path: str | pathlib.Path, n: int, from_end: bool = False) -> list[Image.Image]:
    video, _, _ = tio.read_video(str(path), pts_unit="sec", output_format="THWC")
    if video.shape[0] == 0:
        raise ValueError(f"No frames decoded from {path}")
    frames = video[-n:] if from_end else video[:n]
    return [Image.fromarray(f.numpy()) for f in frames]


def load_frames_from_dir(path: str | pathlib.Path, n: int, from_end: bool = False) -> list[Image.Image]:
    jpgs = sorted(glob.glob(str(pathlib.Path(path) / "*.jpg")))
    if not jpgs:
        raise FileNotFoundError(f"No .jpg files in {path}")
    jpgs = jpgs[-n:] if from_end else jpgs[:n]
    return [Image.open(p).convert("RGB") for p in jpgs]


def load_clip_frames(sample: dict, repo_root: pathlib.Path, n: int) -> tuple[list[Image.Image], list[Image.Image]]:
    if "start_clip" in sample:
        start = load_frames_from_mp4(repo_root / sample["start_clip"], n, from_end=False)
        end   = load_frames_from_mp4(repo_root / sample["end_clip"],   n, from_end=True)
    else:
        start = load_frames_from_dir(repo_root / sample["start_images"], n, from_end=False)
        end   = load_frames_from_dir(repo_root / sample["end_images"],   n, from_end=True)
    return start, end


def end_clip_index(num_frames: int, num_clip_frames: int) -> int:
    """Latent index where the end clip's first token is placed (aligns last latent)."""
    n_lat = (num_frames      - 1) // LTX_TEMPORAL_SCALE + 1
    k_lat = (num_clip_frames - 1) // LTX_TEMPORAL_SCALE + 1
    return n_lat - k_lat


# ── Main ─────────────────────────────────────────────────────────────────────

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
        total_variants = sum(len(s["prompt_variants"]) for s in cfg["samples"])
        print(f"[info] total runs (sample × category): {total_variants}")

        model_id        = cfg["model"]["model_id"]
        lora_strength   = cfg["model"]["distilled_lora_strength"]
        num_frames      = cfg["inference"]["num_frames"]
        frame_rate      = float(cfg["inference"]["frame_rate"])
        height          = cfg["inference"]["height"]
        width           = cfg["inference"]["width"]
        num_steps       = cfg["inference"]["num_inference_steps"]
        guidance_scale  = cfg["inference"]["guidance_scale"]
        run_stage2      = cfg["inference"].get("run_stage2", True)
        seed            = cfg["runtime"]["seed"]
        num_clip_frames = cfg["inputs"]["num_clip_frames"]
        start_strength  = cfg["inputs"]["start_clip_strength"]
        end_strength    = cfg["inputs"]["end_clip_strength"]
        negative_prompt = cfg["inputs"]["negative_prompt"].strip()

        end_idx = end_clip_index(num_frames, num_clip_frames)
        log.info(
            "Clip index: start=0  end=%d  (num_frames=%d → %d lat, clip=%d px → %d lat)",
            end_idx, num_frames,
            (num_frames - 1) // LTX_TEMPORAL_SCALE + 1,
            num_clip_frames,
            (num_clip_frames - 1) // LTX_TEMPORAL_SCALE + 1,
        )

        # ── Load pipeline ────────────────────────────────────────────────────
        log.info("Loading LTX2ConditionPipeline from %s …", model_id)
        t0   = time.perf_counter()
        pipe = LTX2ConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=DEVICE)
        pipe.vae.enable_tiling()
        log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)

        stage1_scheduler = pipe.scheduler

        if run_stage2:
            log.info("Loading distilled LoRA (Stage 2) …")
            pipe.load_lora_weights(
                model_id,
                adapter_name="stage_2_distilled",
                weight_name="ltx-2-19b-distilled-lora-384.safetensors",
            )
            pipe.set_adapters("stage_2_distilled", lora_strength)

            stage2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
            )

            log.info("Loading spatial upsampler …")
            latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                model_id, subfolder="latent_upsampler", torch_dtype=torch.bfloat16
            )
            upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
            upsample_pipe.enable_model_cpu_offload(device=DEVICE)
        else:
            log.info("run_stage2=false — skipping LoRA, upsampler, and Stage 2 load.")

        # ── Per-sample, per-variant loop ──────────────────────────────────────
        summary: list[dict] = []
        run_count = 0

        for s_idx, sample in enumerate(cfg["samples"]):
            sample_id = sample["sample_id"]
            start_src = sample.get("start_clip") or sample.get("start_images", "")
            end_src   = sample.get("end_clip")   or sample.get("end_images",   "")

            log.info("═══ Sample %d/%d  id=%s ═══", s_idx + 1, len(cfg["samples"]), sample_id)
            print(f"[info] start  : {start_src}")
            print(f"[info] end    : {end_src}")

            start_frames, end_frames = load_clip_frames(sample, REPO_ROOT, num_clip_frames)

            conditions_base = [
                LTX2VideoCondition(frames=start_frames, index=0,       strength=start_strength),
                LTX2VideoCondition(frames=end_frames,   index=end_idx, strength=end_strength),
            ]

            for v_idx, variant in enumerate(sample["prompt_variants"]):
                category = variant["category"]
                prompt   = variant["text"].strip()
                run_count += 1

                variant_dir = run_dir / sample_id / category
                variant_dir.mkdir(parents=True, exist_ok=True)

                log.info(
                    "─── [%d/%d] sample=%s  cat=%s ───",
                    run_count, total_variants, sample_id, category,
                )
                prompt_preview = (prompt[:80] + "…") if len(prompt) > 80 else (f'"{prompt}"' if prompt else "(empty)")
                print(f"[info] prompt : {prompt_preview}")

                generator = torch.Generator(device=DEVICE).manual_seed(seed)
                t_infer   = time.perf_counter()

                # ── Stage 1 ───────────────────────────────────────────────────
                pipe.scheduler = stage1_scheduler
                pipe.disable_lora()
                log.info("Stage 1: %dx%d  %d steps  guidance=%.1f", height, width, num_steps, guidance_scale)

                stage1_kwargs = dict(
                    conditions=conditions_base,
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    num_inference_steps=num_steps,
                    sigmas=None,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="latent",
                    return_dict=False,
                )
                # negative_prompt only has effect when CFG is active (guidance_scale > 1).
                if guidance_scale > 1.0:
                    stage1_kwargs["negative_prompt"] = negative_prompt

                video_latent, audio_latent = pipe(**stage1_kwargs)
                log.info("Stage 1 done in %.1fs.", time.perf_counter() - t_infer)

                # ── Save Stage 1 preview (before upsampling) ─────────────────
                t_s1_save = time.perf_counter()
                latents_s1 = video_latent.to(device=DEVICE, dtype=pipe.vae.dtype)
                if pipe.vae.config.timestep_conditioning:
                    decode_ts = torch.tensor([0.0], device=DEVICE, dtype=latents_s1.dtype)
                else:
                    decode_ts = None
                stage1_frames = pipe.vae.decode(latents_s1, decode_ts, return_dict=False)[0]
                stage1_np = pipe.video_processor.postprocess_video(stage1_frames, output_type="np")
                audio_sr   = pipe.vocoder.config.output_sampling_rate
                silent_audio = torch.zeros(2, int(num_frames / frame_rate * audio_sr))
                stage1_path = variant_dir / f"s{seed}_cat{category}_steps{num_steps}_stage1.mp4"
                encode_video(
                    stage1_np[0],
                    fps=int(frame_rate),
                    audio=silent_audio,
                    audio_sample_rate=audio_sr,
                    output_path=str(stage1_path),
                )
                log.info("Stage 1 preview saved in %.1fs.", time.perf_counter() - t_s1_save)

                # ── Upsample ×2 + Stage 2 (skipped when run_stage2=false) ────
                if run_stage2:
                    t_up = time.perf_counter()
                    upscaled_latent = upsample_pipe(
                        latents=video_latent, output_type="latent", return_dict=False
                    )[0]
                    log.info("Upsample done in %.1fs.", time.perf_counter() - t_up)

                    pipe.scheduler = stage2_scheduler
                    pipe.enable_lora()
                    log.info("Stage 2: 3 steps  guidance=1.0  %dx%d", width * 2, height * 2)
                    t_s2 = time.perf_counter()

                    video, audio = pipe(
                        latents=upscaled_latent,
                        audio_latents=audio_latent,
                        prompt=prompt,
                        width=width * 2,
                        height=height * 2,
                        num_frames=num_frames,
                        num_inference_steps=3,
                        noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                        sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                        generator=generator,
                        guidance_scale=1.0,
                        output_type="np",
                        return_dict=False,
                    )
                    elapsed = time.perf_counter() - t_infer
                    log.info("Stage 2 done in %.1fs.  Total: %.1fs.", time.perf_counter() - t_s2, elapsed)

                    video_path = variant_dir / f"s{seed}_cat{category}_steps{num_steps}.mp4"
                    encode_video(
                        video[0],
                        fps=int(frame_rate),
                        audio=audio[0].float().cpu(),
                        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
                        output_path=str(video_path),
                    )
                    log.info("Saved %s", video_path)
                else:
                    elapsed = time.perf_counter() - t_infer
                    video_path = stage1_path
                    log.info("Stage 2 skipped.  Total: %.1fs.", elapsed)

                with (variant_dir / "config_snapshot.yaml").open("w") as f:
                    yaml.safe_dump({
                        "sample_id":  sample_id,
                        "category":   category,
                        "prompt":     prompt,
                        "start_src":  start_src,
                        "end_src":    end_src,
                        "clip_conditioning": {
                            "num_clip_frames": num_clip_frames,
                            "start_index":     0,
                            "end_index":       end_idx,
                            "start_strength":  start_strength,
                            "end_strength":    end_strength,
                        },
                        "inference":  cfg["inference"],
                        "runtime":    cfg["runtime"],
                        "output_stage1": str(stage1_path),
                        "output":        str(video_path),
                        "elapsed_s":     round(elapsed, 1),
                    }, f, sort_keys=False, allow_unicode=True)

                summary.append({
                    "sample_id":     sample_id,
                    "category":      category,
                    "video_stage1":  str(stage1_path),
                    "video":         str(video_path),
                    "elapsed_s":     round(elapsed, 1),
                })

        # ── Run-level artefacts ───────────────────────────────────────────────
        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"run_id": run_id, "runs": summary}, f, sort_keys=False, allow_unicode=True)

        total_time = sum(s["elapsed_s"] for s in summary)
        log.info("All %d runs done.  Total inference: %.1fs (%.1f min)", len(summary), total_time, total_time / 60)
        for s in summary:
            print(f"[done] {s['sample_id']}  cat={s['category']}  →  {s['video']}")


if __name__ == "__main__":
    main()
