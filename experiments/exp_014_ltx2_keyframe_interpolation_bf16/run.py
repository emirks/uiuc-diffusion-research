"""exp_014 — LTX-2 KeyframeInterpolationPipeline, full bfloat16 model.

Identical to exp_013 but uses the full-precision bf16 checkpoint with no
quantization. Eliminates all FP8/Triton issues entirely.

To run (from repo root, LTX-2 venv):
    PYTORCH_ALLOC_CONF=expandable_segments:True \\
    python experiments/exp_014_ltx2_keyframe_interpolation_bf16/run.py
"""
import logging
import pathlib
import yaml

import torch
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

from diffusion.exp_utils import load_config, next_run_dir, resolve_path

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"
log = logging.getLogger(__name__)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(CONFIG_PATH)

    first_path = str(REPO_ROOT / cfg["inputs"]["first_frame"])
    last_path  = str(REPO_ROOT / cfg["inputs"]["last_frame"])
    out_dir    = REPO_ROOT / cfg["outputs"]["dir"]

    print(f"[info] first_frame : {first_path}")
    print(f"[info] last_frame  : {last_path}")

    checkpoint_path     = resolve_path(cfg["model"]["checkpoint_path"])
    distilled_lora_path  = resolve_path(cfg["model"]["distilled_lora_path"])
    spatial_upsampler   = resolve_path(cfg["model"]["spatial_upsampler_path"])
    gemma_root          = resolve_path(cfg["model"]["gemma_root"])

    distilled_lora = [
        LoraPathStrengthAndSDOps(
            distilled_lora_path,
            cfg["model"]["distilled_lora_strength"],
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]

    log.info("Loading KeyframeInterpolationPipeline (bf16, no quantization)…")
    pipeline = KeyframeInterpolationPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=spatial_upsampler,
        gemma_root=gemma_root,
        loras=[],
        quantization=None,   # full bf16 — no FP8, no Triton
    )
    log.info("Pipeline loaded.")

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

    num_frames = cfg["inference"]["num_frames"]
    images = [
        ImageConditioningInput(first_path, 0, cfg["inputs"]["first_frame_strength"]),
        ImageConditioningInput(last_path, num_frames - 1, cfg["inputs"]["last_frame_strength"]),
    ]

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    frame_rate = float(cfg["inference"]["frame_rate"])
    log.info(
        "Running inference  seed=%s  %sx%s  frames=%s  steps=%s",
        cfg["runtime"]["seed"],
        cfg["inference"]["width"],
        cfg["inference"]["height"],
        num_frames,
        cfg["inference"]["num_inference_steps"],
    )

    video, audio = pipeline(
        prompt=cfg["inputs"]["prompt"],
        negative_prompt=cfg["inputs"]["negative_prompt"],
        seed=cfg["runtime"]["seed"],
        height=cfg["inference"]["height"],
        width=cfg["inference"]["width"],
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=cfg["inference"]["num_inference_steps"],
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=images,
        tiling_config=tiling_config,
    )

    run_id, run_dir = next_run_dir(out_dir)
    seed  = cfg["runtime"]["seed"]
    steps = cfg["inference"]["num_inference_steps"]
    video_path = run_dir / f"s{seed}_steps{steps}.mp4"

    log.info("Encoding video to %s", video_path)
    encode_video(
        video=video,
        fps=int(frame_rate),
        audio=audio,
        output_path=str(video_path),
        video_chunks_number=video_chunks_number,
    )

    with (run_dir / "config_snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    log.info("Done %s  →  %s", run_id, video_path)


if __name__ == "__main__":
    main()
