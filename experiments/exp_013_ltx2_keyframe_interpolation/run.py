"""exp_013 — LTX-2 KeyframeInterpolationPipeline first-to-last frame.

Uses KeyframeInterpolationPipeline to interpolate between the first and last
frames of an action clip from vc-bench-flf, producing a two-stage video.
"""
import pathlib
import yaml
import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


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


def main() -> None:
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
    print("[info] loading KeyframeInterpolationPipeline …")
    pipeline = KeyframeInterpolationPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=spatial_upsampler,
        gemma_root=gemma_root,
        loras=[],
    )

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
        (first_path, 0,              cfg["inputs"]["first_frame_strength"]),
        (last_path,  num_frames - 1, cfg["inputs"]["last_frame_strength"]),
    ]

    # ── Tiling ────────────────────────────────────────────────────────────────
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"[info] running inference  seed={cfg['runtime']['seed']}  "
          f"{cfg['inference']['width']}x{cfg['inference']['height']}  "
          f"frames={num_frames}  steps={cfg['inference']['num_inference_steps']} …")

    video, audio = pipeline(
        prompt=cfg["inputs"]["prompt"],
        negative_prompt=cfg["inputs"]["negative_prompt"],
        seed=cfg["runtime"]["seed"],
        height=cfg["inference"]["height"],
        width=cfg["inference"]["width"],
        num_frames=num_frames,
        frame_rate=cfg["inference"]["frame_rate"],
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

    encode_video(
        video=video,
        fps=cfg["inference"]["frame_rate"],
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=str(video_path),
        video_chunks_number=video_chunks_number,
    )

    with (run_dir / "config_snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[done] {run_id}  →  {video_path}")


if __name__ == "__main__":
    main()
