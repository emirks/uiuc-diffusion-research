"""exp_020 — LTX-2 C2V — minimal VC aligned with diffusers *Condition Pipeline Generation*.

Canonical doc (same page, two sections combined):
  https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md

  • **#condition-pipeline-generation** — `LTX2ConditionPipeline`, `LTX2VideoCondition`,
    `pipe.vae.enable_tiling()` right after CPU offload, two-stage FLF2V-style Stage 2
    kwargs (`width=width*2`, `height=height*2`, `num_inference_steps=3`,
    `sigmas=STAGE_2_DISTILLED_SIGMA_VALUES`, `generator=…`, `guidance_scale=1.0`).
    VC here = two *video* conditions (start clip @ 0, end clip @ end_idx), not FLF2V images.

  • **#two-stages-generation** (same file) — `Lightricks/LTX-2` base + Stage 2 distilled
    LoRA from Hub, Stage 1 non-distilled (`sigmas=None`, `guidance_scale=4.0`), and
    `noise_scale` on Stage 2 as in that T2V snippet.  At `guidance_scale=1.0`, CFG is off
    (`do_classifier_free_guidance` is False), so `negative_prompt` on Stage 2 has no effect
    — we omit it there; Stage 1 still passes it with CFG.

Setup order: Condition Pipeline block (offload → tiling) → Two-stages block (load LoRA,
stage-2 scheduler) → latent upsampler — see comments in `main()`.

Stage 2 spatial: `width=width*2`, `height=height*2` are required for `LTX2ConditionPipeline`
(mask built from pixel `height`/`width`; not inferred from latents alone).

Conditioning index for end clip (multi-frame):
  end_clip_index = latent_num_frames - clip_latent_frames  (121 px → 16 lat, 25 px clip → 4 lat → index 12)
  index=-1 would trim a multi-frame clip to one latent — do not use for VC.

Memory: `enable_model_cpu_offload` (whole components), not `enable_sequential_cpu_offload`.

How to run:
    source /workspace/miniforge3/etc/profile.d/conda.sh
    conda activate /workspace/envs/diff
    cd /workspace/diffusion-research
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

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

from diffusion.exp_utils import load_config, next_run_dir, TeeLogger

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"

LTX_TEMPORAL_SCALE = 8   # VAE causal temporal downscale factor
DEVICE             = "cuda:0"

log = logging.getLogger(__name__)


# ── Frame / clip helpers ──────────────────────────────────────────────────────

def load_frames_from_mp4(path: str | pathlib.Path, n: int, from_end: bool = False) -> list[Image.Image]:
    """Decode up to *n* RGB frames from an MP4 (first or last *n*)."""
    video, _, _ = tio.read_video(str(path), pts_unit="sec", output_format="THWC")
    if video.shape[0] == 0:
        raise ValueError(f"No frames decoded from {path}")
    frames = video[-n:] if from_end else video[:n]
    return [Image.fromarray(f.numpy()) for f in frames]


def load_frames_from_dir(path: str | pathlib.Path, n: int, from_end: bool = False) -> list[Image.Image]:
    """Load up to *n* RGB frames from a JPEG directory (first or last *n*)."""
    jpgs = sorted(glob.glob(str(pathlib.Path(path) / "*.jpg")))
    if not jpgs:
        raise FileNotFoundError(f"No .jpg files in {path}")
    jpgs = jpgs[-n:] if from_end else jpgs[:n]
    return [Image.open(p).convert("RGB") for p in jpgs]


def load_clip_frames(sample: dict, repo_root: pathlib.Path, n: int) -> tuple[list[Image.Image], list[Image.Image]]:
    """Dispatch frame loading by input format (MP4 or JPEG dir)."""
    if "start_clip" in sample:
        start = load_frames_from_mp4(repo_root / sample["start_clip"], n, from_end=False)
        end   = load_frames_from_mp4(repo_root / sample["end_clip"],   n, from_end=True)
    else:
        start = load_frames_from_dir(repo_root / sample["start_images"], n, from_end=False)
        end   = load_frames_from_dir(repo_root / sample["end_images"],   n, from_end=True)
    return start, end


def end_clip_index(num_frames: int, num_clip_frames: int) -> int:
    """Latent index where the end clip's first latent token should be placed.

    Aligns the clip's last latent with the output's last latent frame so
    the end conditioning covers the final temporal region of the video.
    index=-1 must NOT be used for multi-frame clips (trims to 1 latent).
    """
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

        model_id        = cfg["model"]["model_id"]
        lora_strength   = cfg["model"]["distilled_lora_strength"]
        num_frames      = cfg["inference"]["num_frames"]
        frame_rate      = float(cfg["inference"]["frame_rate"])
        height          = cfg["inference"]["height"]
        width           = cfg["inference"]["width"]
        num_steps       = cfg["inference"]["num_inference_steps"]
        guidance_scale  = cfg["inference"]["guidance_scale"]
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

        # ── Condition Pipeline Generation (ltx2.md): load → offload → tiling ──
        # https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md#condition-pipeline-generation
        # Doc uses enable_sequential_cpu_offload; we use enable_model_cpu_offload (whole components).
        log.info("Loading LTX2ConditionPipeline from %s …", model_id)
        t0   = time.perf_counter()
        pipe = LTX2ConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        # Whole-component offload (see DiffusionPipeline.enable_model_cpu_offload docstring).
        pipe.enable_model_cpu_offload(device=DEVICE)
        pipe.vae.enable_tiling()
        log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)

        # Keep the default stage-1 scheduler so it can be restored each sample.
        stage1_scheduler = pipe.scheduler

        # ── Two-stages Generation (same doc page): LoRA + Stage-2 scheduler ──
        # https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md#two-stages-generation
        # FLF2V Condition example uses a fully-distilled checkpoint without extra LoRA; for
        # Lightricks/LTX-2 we attach Stage-2 distilled LoRA here (T2V snippet weight_name).
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

        # Spatial upsampler (×2 each spatial dim, latent → latent) — matches T2V two-stage snippet.
        log.info("Loading spatial upsampler …")
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            model_id, subfolder="latent_upsampler", torch_dtype=torch.bfloat16
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
        upsample_pipe.enable_model_cpu_offload(device=DEVICE)

        # ── Per-sample loop ───────────────────────────────────────────────────
        summary: list[dict] = []

        for idx, sample in enumerate(cfg["samples"]):
            sample_id  = sample["sample_id"]
            prompt     = sample["prompt"].strip()
            sample_dir = run_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            start_src = sample.get("start_clip") or sample.get("start_images", "")
            end_src   = sample.get("end_clip")   or sample.get("end_images",   "")
            log.info("─── Sample %d/%d  id=%s ───", idx + 1, len(cfg["samples"]), sample_id)
            print(f"[info] start  : {start_src}")
            print(f"[info] end    : {end_src}")
            print(f"[info] prompt : {prompt[:80]}…")

            start_frames, end_frames = load_clip_frames(sample, REPO_ROOT, num_clip_frames)

            # LTX2VideoCondition — the core diffusers API for visual conditioning.
            # frames: list[PIL.Image] → encoded by the pipeline VAE internally.
            # index:  latent frame index where this clip's first token is placed.
            # strength=1.0: fully clean anchor (no denoising applied to this region).
            conditions = [
                LTX2VideoCondition(frames=start_frames, index=0,       strength=start_strength),
                LTX2VideoCondition(frames=end_frames,   index=end_idx, strength=end_strength),
            ]

            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            t_infer   = time.perf_counter()

            # ── Stage 1: base model, non-distilled ───────────────────────────
            # Restore stage-1 scheduler (may have been swapped by previous sample).
            pipe.scheduler = stage1_scheduler
            pipe.disable_lora()
            log.info("Stage 1: %dx%d  %d steps  guidance=%.1f", height, width, num_steps, guidance_scale)

            # Stage 1: Two-stages Generation defaults (non-distilled CFG), not the distilled 8-step FLF2V Stage 1.
            video_latent, audio_latent = pipe(
                conditions=conditions,
                prompt=prompt,
                negative_prompt=negative_prompt,
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
            log.info("Stage 1 done in %.1fs.", time.perf_counter() - t_infer)

            # ── Upsample ×2 spatial (latent → latent) ────────────────────────
            t_up = time.perf_counter()
            upscaled_latent = upsample_pipe(
                latents=video_latent, output_type="latent", return_dict=False
            )[0]
            log.info("Upsample done in %.1fs.", time.perf_counter() - t_up)

            # ── Stage 2: distilled LoRA, 3 steps ─────────────────────────────
            # Condition Pipeline FLF2V + Two-stages T2V: noise_scale from T2V (renoise); no negative_prompt
            # at guidance_scale=1.0 (CFG disabled — see pipeline do_classifier_free_guidance).
            pipe.scheduler = stage2_scheduler
            pipe.enable_lora()
            log.info("Stage 2: 3 steps  guidance=1.0  %dx%d (distilled LoRA)", width * 2, height * 2)
            t_s2 = time.perf_counter()

            video, audio = pipe(
                latents=upscaled_latent,
                audio_latents=audio_latent,
                prompt=prompt,
                width=width * 2,
                height=height * 2,
                num_frames=num_frames,  # doc examples omit (default 121); explicit if config changes
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

            # ── Save ─────────────────────────────────────────────────────────
            video_path = sample_dir / f"s{seed}_K{num_clip_frames}_steps{num_steps}.mp4"
            encode_video(
                video[0],
                fps=int(frame_rate),
                audio=audio[0].float().cpu(),
                audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
                output_path=str(video_path),
            )
            log.info("Saved %s", video_path)

            with (sample_dir / "config_snapshot.yaml").open("w") as f:
                yaml.safe_dump({
                    "sample_id":  sample_id,
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
                    "output":     str(video_path),
                    "elapsed_s":  round(elapsed, 1),
                }, f, sort_keys=False, allow_unicode=True)

            summary.append({"sample_id": sample_id, "video": str(video_path), "elapsed_s": round(elapsed, 1)})

        # ── Run-level artefacts ───────────────────────────────────────────────
        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"run_id": run_id, "samples": summary}, f, sort_keys=False, allow_unicode=True)

        total = sum(s["elapsed_s"] for s in summary)
        log.info("All %d samples done.  Total inference: %.1fs", len(summary), total)
        for s in summary:
            print(f"[done] {s['sample_id']}  →  {s['video']}")


if __name__ == "__main__":
    main()
