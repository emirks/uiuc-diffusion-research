"""exp_015 — LTX-2 Clip-to-Video (C2V), strategy 1: clip guiding latents.

Forked from exp_014 (KeyframeInterpolationPipeline, bf16).  Instead of
conditioning on a single first and last *frame*, this experiment conditions on
full *clips* — the first K frames of a start clip and the last K frames of an
end clip — and generates the connecting video in between.

Conditioning mechanism (LTX-2 temporal scale = 8):
  Each clip (K=24 pixel frames) encodes to 3 VAE latent tokens.
  frame_idx for the start clip = 0  →  tokens at output positions [0,8,16].
  frame_idx for the end clip   = num_frames - K = 73  →  tokens at [73,81,89].
  After the causal-fix applied to the output latent, these align exactly with
  output latent tokens 0-2 (start) and 10-12 (end).  Latent tokens 3-9
  (≈ pixel frames 17–73) are generated freely by the diffusion model.

To run (from repo root, LTX-2 venv):
    cd /workspace/diffusion-research
    source src/LTX-2/.venv/bin/activate

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python experiments/exp_015_ltx2_c2v_s1_clip_guiding/run.py
"""
import logging
import pathlib
import sys
import yaml

import torch
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines.utils.args import ClipConditioningInput
from ltx_pipelines.utils.media_io import encode_video

from diffusion.exp_utils import load_config, next_run_dir, resolve_path, TeeLogger

REPO_ROOT   = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"

# LTX-2 video VAE temporal downscale factor (hardcoded constant, see
# ltx_core.types.SpatioTemporalScaleFactors.default()).
LTX_TEMPORAL_SCALE = 8

log = logging.getLogger(__name__)


def compute_clip_frame_idx(num_output_frames: int, num_clip_frames: int) -> tuple[int, int]:
    """Return (frame_idx_start, frame_idx_end) for clip conditioning.

    ``frame_idx_start`` is always 0.

    ``frame_idx_end`` is chosen so that the end clip's VAE latent tokens fall
    at exactly the same pixel-space temporal positions as the last
    ``num_lat_clip`` tokens of the output video latent (accounting for the
    causal temporal-scale of the LTX-2 video VAE).

    With temporal_scale T=8 and K=num_clip_frames pixel frames:
        num_lat_clip = (K - 1) // T + 1
        The clip's latent tokens land at local positions [0, T, 2T, ...,
        (num_lat_clip-1)*T] when causal_fix=False (frame_idx != 0).
        For the last token to align with the output's last latent position we
        need:   frame_idx_end + (num_lat_clip - 1) * T == num_output_frames - 1
        i.e.:   frame_idx_end = num_output_frames - 1 - (num_lat_clip - 1) * T

    This simplifies to ``num_output_frames - num_clip_frames`` because:
        (num_lat_clip - 1) * T = ((K-1)//T) * T  which equals K-1 when K-1 is
        divisible by T (e.g. K=9,17,25) but equals K-T+T-1=K-1 when rounded.
    For K=24 (23//8=2 → num_lat_clip=3):
        frame_idx_end = 96 - 2*8 = 80... but direct formula gives 97-24=73.

    Wait — the correct derivation uses the pixel coordinate grid BEFORE the
    causal-fix.  Without causal_fix the clip token at latent index i occupies
    the pixel range [i*T, (i+1)*T].  The output token at latent index j (with
    causal_fix applied) occupies [(j*T - (T-1)).clamp(0), (j+1)*T - (T-1)].
    Setting the clip's last range equal to the output's last range:
        frame_idx + (num_lat_clip-1)*T  ==  (F_lat_out - 1)*T - (T-1)
    where F_lat_out = (num_output_frames - 1) // T + 1.
    For num_output_frames=97, T=8: F_lat_out=13, (12*8 - 7) = 89.
    For K=24: num_lat_clip=3, frame_idx_end = 89 - 16 = 73.
    And 73 == 97 - 24 == num_output_frames - K. ✓

    So the closed-form is: frame_idx_end = num_output_frames - num_clip_frames.
    """
    frame_idx_start = 0
    frame_idx_end = num_output_frames - num_clip_frames
    return frame_idx_start, frame_idx_end


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
            stream=sys.stdout,  # sys.stdout IS TeeLogger here → all logs go to run.log
            force=True,
        )

        start_path = str(REPO_ROOT / cfg["inputs"]["start_clip"])
        end_path   = str(REPO_ROOT / cfg["inputs"]["end_clip"])

        print(f"[info] run_dir         : {run_dir}")
        print(f"[info] start_clip      : {start_path}")
        print(f"[info] end_clip        : {end_path}")

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

        log.info("Loading KeyframeInterpolationPipeline (C2V, bf16, no quantization)…")
        pipeline = KeyframeInterpolationPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler,
            gemma_root=gemma_root,
            loras=[],
            quantization=None,
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

        num_frames      = cfg["inference"]["num_frames"]
        num_clip_frames = cfg["inputs"]["num_clip_frames"]
        strength_start  = cfg["inputs"]["start_clip_strength"]
        strength_end    = cfg["inputs"]["end_clip_strength"]

        frame_idx_start, frame_idx_end = compute_clip_frame_idx(num_frames, num_clip_frames)

        log.info(
            "Clip conditioning: K=%d  frame_idx_start=%d  frame_idx_end=%d",
            num_clip_frames, frame_idx_start, frame_idx_end,
        )

        clips = [
            ClipConditioningInput(start_path, frame_idx_start, strength_start, num_clip_frames),
            ClipConditioningInput(end_path,   frame_idx_end,   strength_end,   num_clip_frames),
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
            images=[],   # not used; clip conditioning is passed via `clips`
            clips=clips,
            tiling_config=tiling_config,
        )

        seed  = cfg["runtime"]["seed"]
        steps = cfg["inference"]["num_inference_steps"]
        video_path = run_dir / f"s{seed}_K{num_clip_frames}_steps{steps}.mp4"

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
