"""exp_018 — LTX-2 C2V with SLERP-guided stage-1 latent initialisation.

Forked from exp_016 (sequential batch evaluation).  The only differences:
  - Reads ``inputs.slerp_blend_alpha`` from config.
  - Creates a ``SlerpInit`` strategy and passes it as ``latent_init_fn`` to the
    pipeline, which biases the middle temporal frames toward a SLERP interpolation
    between the last VAE frame of the start clip and the first VAE frame of the
    end clip before denoising begins.
  - Output filename includes the blend alpha: s{seed}_K{K}_a{alpha}_steps{steps}.mp4

Everything else (pipeline loading, sample loop, outputs) is identical to exp_016.

To run (from repo root, LTX-2 venv):
    cd /workspace/diffusion-research
    source src/LTX-2/.venv/bin/activate

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python experiments/exp_018_ltx2_c2v_s2_slerp_init/run.py
"""
import argparse
import logging
import pathlib
import shutil
import sys
import time

import torch
import yaml

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.loader.registry import StateDictRegistry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines.utils.args import ClipConditioningInput
from ltx_pipelines.utils.media_io import encode_video

from diffusion.exp_utils import load_config, next_run_dir, resolve_path, TeeLogger, image_dir_to_tmp_mp4
from diffusion.ltx_latent_init import SlerpInit

REPO_ROOT        = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG   = pathlib.Path(__file__).parent / "config.yaml"

# LTX-2 video VAE temporal downscale factor.
LTX_TEMPORAL_SCALE = 8

log = logging.getLogger(__name__)


def compute_clip_frame_idx(num_output_frames: int, num_clip_frames: int) -> tuple[int, int]:
    """Return (frame_idx_start, frame_idx_end) for clip conditioning.

    frame_idx_start is always 0.
    frame_idx_end = num_output_frames - num_clip_frames aligns the end clip's
    last VAE latent token with the output video's last latent token.
    """
    return 0, num_output_frames - num_clip_frames


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG,
                        help="Path to config YAML (default: config.yaml)")
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

        print(f"[info] run_dir      : {run_dir}")
        print(f"[info] samples      : {len(cfg['samples'])}")

        checkpoint_path     = resolve_path(cfg["model"]["checkpoint_path"])
        distilled_lora_path = resolve_path(cfg["model"]["distilled_lora_path"])
        spatial_upsampler   = resolve_path(cfg["model"]["spatial_upsampler_path"])
        gemma_root          = resolve_path(cfg["model"]["gemma_root"])

        distilled_lora = [
            LoraPathStrengthAndSDOps(
                distilled_lora_path,
                cfg["model"]["distilled_lora_strength"],
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

        registry = StateDictRegistry()

        log.info("Loading KeyframeInterpolationPipeline (bf16, no quantization, StateDictRegistry)…")
        t0 = time.perf_counter()
        pipeline = KeyframeInterpolationPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler,
            gemma_root=gemma_root,
            loras=[],
            quantization=None,
            registry=registry,
        )
        log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)

        video_guider_params = MultiModalGuiderParams(
            cfg_scale=3.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )

        num_frames       = cfg["inference"]["num_frames"]
        num_clip_frames  = cfg["inputs"]["num_clip_frames"]
        strength_start   = cfg["inputs"]["start_clip_strength"]
        strength_end     = cfg["inputs"]["end_clip_strength"]
        slerp_blend_alpha = float(cfg["inputs"]["slerp_blend_alpha"])
        frame_rate       = float(cfg["inference"]["frame_rate"])
        seed             = cfg["runtime"]["seed"]
        steps            = cfg["inference"]["num_inference_steps"]

        frame_idx_start, frame_idx_end = compute_clip_frame_idx(num_frames, num_clip_frames)
        tiling_config       = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        # Build the SLERP strategy once; it is stateless between samples.
        latent_init_fn = SlerpInit(blend_alpha=slerp_blend_alpha)

        log.info(
            "Clip conditioning: K=%d  frame_idx_start=%d  frame_idx_end=%d",
            num_clip_frames, frame_idx_start, frame_idx_end,
        )
        log.info("SLERP init: blend_alpha=%.3f", slerp_blend_alpha)

        summary: list[dict] = []

        for idx, sample in enumerate(cfg["samples"]):
            sample_id  = sample["sample_id"]
            prompt     = sample["prompt"]
            sample_dir = run_dir / sample_id

            # Resolve input clip paths.
            clips_in_sample_dir = False
            if "start_images" in sample and "end_images" in sample:
                sample_dir.mkdir(parents=True, exist_ok=True)
                start_path = str(sample_dir / "start_clip.mp4")
                end_path   = str(sample_dir / "end_clip.mp4")
                log.info("Building start clip from images: %s", sample["start_images"])
                image_dir_to_tmp_mp4(
                    REPO_ROOT / sample["start_images"], num_clip_frames,
                    pathlib.Path(start_path), fps=int(frame_rate), from_end=False,
                )
                log.info("Building end clip from images: %s", sample["end_images"])
                image_dir_to_tmp_mp4(
                    REPO_ROOT / sample["end_images"], num_clip_frames,
                    pathlib.Path(end_path), fps=int(frame_rate), from_end=True,
                )
                clips_in_sample_dir = True
            elif "start_clip" in sample and "end_clip" in sample:
                start_path = str(REPO_ROOT / sample["start_clip"])
                end_path   = str(REPO_ROOT / sample["end_clip"])
            else:
                clip_dir   = REPO_ROOT / sample["clip_dir"]
                start_path = str(clip_dir / "first.mp4")
                end_path   = str(clip_dir / "last.mp4")

            log.info(
                "─── Sample %d/%d  id=%s ───",
                idx + 1, len(cfg["samples"]), sample_id,
            )
            print(f"[info] start_clip : {start_path}")
            print(f"[info] end_clip   : {end_path}")
            print(f"[info] prompt     : {prompt[:80].strip()}…")

            clips = [
                ClipConditioningInput(start_path, frame_idx_start, strength_start, num_clip_frames),
                ClipConditioningInput(end_path,   frame_idx_end,   strength_end,   num_clip_frames),
            ]

            log.info(
                "Running inference  seed=%s  %sx%s  frames=%s  steps=%s  slerp_alpha=%.3f",
                seed,
                cfg["inference"]["width"],
                cfg["inference"]["height"],
                num_frames,
                steps,
                slerp_blend_alpha,
            )
            t_infer = time.perf_counter()
            video, audio = pipeline(
                prompt=prompt,
                negative_prompt=cfg["inputs"]["negative_prompt"],
                seed=seed,
                height=cfg["inference"]["height"],
                width=cfg["inference"]["width"],
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=steps,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=[],
                clips=clips,
                tiling_config=tiling_config,
                latent_init_fn=latent_init_fn,
            )
            elapsed = time.perf_counter() - t_infer
            log.info("Inference done in %.1fs.", elapsed)

            sample_dir.mkdir(parents=True, exist_ok=True)

            if not clips_in_sample_dir:
                shutil.copy2(start_path, sample_dir / "start_clip.mp4")
                shutil.copy2(end_path,   sample_dir / "end_clip.mp4")
                log.info("Copied input clips → %s/", sample_dir.name)
            else:
                log.info("Input clips already in %s/", sample_dir.name)

            # Include blend_alpha in filename so runs with different alphas coexist.
            alpha_tag  = f"a{slerp_blend_alpha:.2f}".replace(".", "p")
            video_path = sample_dir / f"s{seed}_K{num_clip_frames}_{alpha_tag}_steps{steps}.mp4"

            log.info("Encoding video → %s", video_path)
            encode_video(
                video=video,
                fps=int(frame_rate),
                audio=audio,
                output_path=str(video_path),
                video_chunks_number=video_chunks_number,
            )

            per_sample_cfg = {
                "sample_id":         sample_id,
                "prompt":            prompt,
                "start_clip_copied": str(sample_dir / "start_clip.mp4"),
                "end_clip_copied":   str(sample_dir / "end_clip.mp4"),
                "clip_conditioning": {
                    "num_clip_frames":  num_clip_frames,
                    "frame_idx_start":  frame_idx_start,
                    "frame_idx_end":    frame_idx_end,
                    "start_strength":   strength_start,
                    "end_strength":     strength_end,
                },
                "slerp_init": {
                    "blend_alpha": slerp_blend_alpha,
                },
                "inference": cfg["inference"],
                "runtime":   cfg["runtime"],
                "output":    str(video_path),
                "elapsed_s": round(elapsed, 1),
            }
            if "clip_dir" in sample:
                per_sample_cfg["clip_dir"] = sample["clip_dir"]
            if "start_clip" in sample:
                per_sample_cfg["start_clip"] = sample["start_clip"]
            if "end_clip" in sample:
                per_sample_cfg["end_clip"] = sample["end_clip"]
            if "start_images" in sample:
                per_sample_cfg["start_images"] = sample["start_images"]
            if "end_images" in sample:
                per_sample_cfg["end_images"] = sample["end_images"]
            with (sample_dir / "config_snapshot.yaml").open("w") as f:
                yaml.safe_dump(per_sample_cfg, f, sort_keys=False, allow_unicode=True)

            summary.append({"sample_id": sample_id, "video": str(video_path), "elapsed_s": round(elapsed, 1)})
            log.info("Saved %s  (%.1fs)", video_path.name, elapsed)

        # Run-level artefacts
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
