"""exp_013 — LTX-2 KeyframeInterpolationPipeline first-to-last frame.

Uses KeyframeInterpolationPipeline to interpolate between the first and last
frames of an action clip from vc-bench-flf, producing a two-stage video.

To run:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /workspace/diffusion-research/experiments/exp_013_ltx2_keyframe_interpolation/run.py

We have to set pytorch_cuda_alloc_conf

"""
import logging
import pathlib
import yaml

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"
log = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve(path: str) -> str:
    return str(pathlib.Path(path).expanduser().resolve())


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

@torch.inference_mode()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()

    # ── Paths ─────────────────────────────────────────────────────────────────
    first_path = str(REPO_ROOT / cfg["inputs"]["first_frame"])
    last_path  = str(REPO_ROOT / cfg["inputs"]["last_frame"])
    out_dir    = REPO_ROOT / cfg["outputs"]["dir"]

    print(f"[info] first_frame : {first_path}")
    print(f"[info] last_frame  : {last_path}")

    # ── Model paths ───────────────────────────────────────────────────────────
    checkpoint_path      = resolve(cfg["model"]["checkpoint_path"])
    distilled_lora_path  = resolve(cfg["model"]["distilled_lora_path"])
    spatial_upsampler    = resolve(cfg["model"]["spatial_upsampler_path"])
    gemma_root           = resolve(cfg["model"]["gemma_root"])

    distilled_lora = [
        LoraPathStrengthAndSDOps(
            distilled_lora_path,
            cfg["model"]["distilled_lora_strength"],
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # LTX inference path has little internal logging: loaders do not log; you get
    # tqdm progress for denoising steps and media_io "Video saved to ..." at the end.
    log.info("Loading KeyframeInterpolationPipeline (checkpoint + LoRA + upsampler + Gemma)…")
    pipeline = KeyframeInterpolationPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=spatial_upsampler,
        gemma_root=gemma_root,
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
    )
    log.info("Pipeline loaded.")

    # ── Guider params (defaults from ltx-pipelines docs) ─────────────────────
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

    # ── Image conditioning ────────────────────────────────────────────────────
    num_frames = cfg["inference"]["num_frames"]
    images = [
        ImageConditioningInput(first_path, 0, cfg["inputs"]["first_frame_strength"]),
        ImageConditioningInput(last_path, num_frames - 1, cfg["inputs"]["last_frame_strength"]),
    ]

    # ── Tiling ────────────────────────────────────────────────────────────────
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    # ── Run ───────────────────────────────────────────────────────────────────
    log.info(
        "Running inference  seed=%s  %sx%s  frames=%s  steps=%s",
        cfg["runtime"]["seed"],
        cfg["inference"]["width"],
        cfg["inference"]["height"],
        num_frames,
        cfg["inference"]["num_inference_steps"],
    )
    frame_rate = float(cfg["inference"]["frame_rate"])
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

    # ── Save ──────────────────────────────────────────────────────────────────
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