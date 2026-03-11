"""exp_017 — LTX-2 C2V category sweep: 8 samples across 8 categories.

Forked from exp_015 (single-sample C2V baseline).  Introduces two pair types to compare:

  same_video  – both clips come from the same source video (first 24 px frames
                and last 24 px frames).  Directly comparable to exp_015/016.

  cross_video – start clip and end clip come from *different* source videos
                (and different VC-Bench categories), forcing the model to
                synthesise a plausible bridge between unrelated scenes.

8 samples total (4 same-video, 4 cross-video) spanning 8 distinct categories:
  action, conversation, dancing, animal, horse+forest, sunset+coast,
  music+architecture, car+talk.

Each sample specifies its own explicit start_clip / end_clip paths in
config.yaml, rather than a shared clip_dir, so that cross-video pairs are
expressed cleanly.

Output structure (per sample_id directory):
    outputs/videos/exp_017_ltx2_c2v_category_sweep/run_NNNN/
        run.log
        summary.yaml
        config_snapshot.yaml
        {sample_id}/
            start_clip.mp4          ← copy of input start clip
            end_clip.mp4            ← copy of input end clip
            s{seed}_K{K}_steps{steps}.mp4
            config_snapshot.yaml

To run (from repo root, LTX-2 venv):
    cd /workspace/diffusion-research
    source src/LTX-2/.venv/bin/activate

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python experiments/exp_017_ltx2_c2v_category_sweep/run.py
"""
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

from diffusion.exp_utils import load_config, next_run_dir, resolve_path, TeeLogger

REPO_ROOT   = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"

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
    cfg     = load_config(CONFIG_PATH)
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

        samples = cfg["samples"]
        print(f"[info] run_dir      : {run_dir}")
        print(f"[info] total samples: {len(samples)}")
        same  = sum(1 for s in samples if s["pair_type"] == "same_video")
        cross = sum(1 for s in samples if s["pair_type"] == "cross_video")
        print(f"[info] same_video   : {same}   cross_video: {cross}")

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

        # StateDictRegistry keeps all model state dicts in CPU RAM between
        # pipeline() calls so that samples 2..N skip disk I/O entirely
        # (CPU RAM → GPU only).  Trade-off: ~65 GB CPU RAM needed.
        # Without it every sample re-reads the 19B checkpoint and Gemma 12B
        # from disk, doubling/tripling total wall-clock time.
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

        num_frames      = cfg["inference"]["num_frames"]
        num_clip_frames = cfg["inputs"]["num_clip_frames"]
        strength_start  = cfg["inputs"]["start_clip_strength"]
        strength_end    = cfg["inputs"]["end_clip_strength"]
        frame_rate      = float(cfg["inference"]["frame_rate"])
        seed            = cfg["runtime"]["seed"]
        steps           = cfg["inference"]["num_inference_steps"]

        frame_idx_start, frame_idx_end = compute_clip_frame_idx(num_frames, num_clip_frames)
        tiling_config       = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        log.info(
            "Clip conditioning: K=%d  frame_idx_start=%d  frame_idx_end=%d",
            num_clip_frames, frame_idx_start, frame_idx_end,
        )

        summary: list[dict] = []

        for idx, sample in enumerate(samples):
            sample_id = sample["sample_id"]
            pair_type = sample["pair_type"]
            category  = sample["category"]
            prompt    = sample["prompt"]

            start_path = str(REPO_ROOT / sample["start_clip"])
            end_path   = str(REPO_ROOT / sample["end_clip"])

            log.info(
                "─── Sample %d/%d  id=%s  type=%s  cat=%s ───",
                idx + 1, len(samples), sample_id, pair_type, category,
            )
            print(f"[info] start_clip : {start_path}")
            print(f"[info] end_clip   : {end_path}")
            print(f"[info] prompt     : {prompt[:80].strip()}…")

            clips = [
                ClipConditioningInput(start_path, frame_idx_start, strength_start, num_clip_frames),
                ClipConditioningInput(end_path,   frame_idx_end,   strength_end,   num_clip_frames),
            ]

            log.info(
                "Running inference  seed=%s  %sx%s  frames=%s  steps=%s",
                seed,
                cfg["inference"]["width"],
                cfg["inference"]["height"],
                num_frames,
                steps,
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
            )
            elapsed = time.perf_counter() - t_infer
            log.info("Inference done in %.1fs.", elapsed)

            sample_dir = run_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Copy input clips so the output directory is self-contained.
            shutil.copy2(start_path, sample_dir / "start_clip.mp4")
            shutil.copy2(end_path,   sample_dir / "end_clip.mp4")
            log.info("Copied input clips → %s/", sample_dir.name)

            video_path = sample_dir / f"s{seed}_K{num_clip_frames}_steps{steps}.mp4"

            log.info("Encoding video → %s", video_path)
            encode_video(
                video=video,
                fps=int(frame_rate),
                audio=audio,
                output_path=str(video_path),
                video_chunks_number=video_chunks_number,
            )

            per_sample_cfg = {
                "sample_id": sample_id,
                "pair_type": pair_type,
                "category":  category,
                "start_clip":        sample["start_clip"],
                "end_clip":          sample["end_clip"],
                "start_clip_copied": str(sample_dir / "start_clip.mp4"),
                "end_clip_copied":   str(sample_dir / "end_clip.mp4"),
                "prompt":     prompt,
                "clip_conditioning": {
                    "num_clip_frames": num_clip_frames,
                    "frame_idx_start": frame_idx_start,
                    "frame_idx_end":   frame_idx_end,
                    "start_strength":  strength_start,
                    "end_strength":    strength_end,
                },
                "inference": cfg["inference"],
                "runtime":   cfg["runtime"],
                "output":    str(video_path),
                "elapsed_s": round(elapsed, 1),
            }
            with (sample_dir / "config_snapshot.yaml").open("w") as f:
                yaml.safe_dump(per_sample_cfg, f, sort_keys=False, allow_unicode=True)

            summary.append({
                "sample_id": sample_id,
                "pair_type": pair_type,
                "category":  category,
                "video":     str(video_path),
                "elapsed_s": round(elapsed, 1),
            })
            log.info("Saved %s  (%.1fs)", video_path.name, elapsed)

        # Run-level artefacts
        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"run_id": run_id, "samples": summary}, f,
                           sort_keys=False, allow_unicode=True)

        total = sum(s["elapsed_s"] for s in summary)
        log.info("All %d samples done.  Total inference time: %.1fs", len(summary), total)

        print("\n[summary]")
        for s in summary:
            print(f"  [{s['pair_type']:12s}] {s['sample_id']:35s}  {s['elapsed_s']:.1f}s  →  {s['video']}")


if __name__ == "__main__":
    main()
