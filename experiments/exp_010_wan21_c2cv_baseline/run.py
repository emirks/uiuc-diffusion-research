"""exp_010 — Wan 2.1 C2V clip-to-clip video connecting baseline.

Uses WanVideoConnectPipeline with the FLF2V-14B-720P model.
start_clip and end_clip anchor the first/last anchor_frames of the output video.

Clips are passed as List[PIL.Image] directly to the pipeline.
VideoProcessor.preprocess_video handles resize + normalisation internally,
producing (1, C, T, H, W) float32 in [-1, 1] range as expected by prepare_latents.
"""
import pathlib
import numpy as np
import yaml
import torch
import PIL.Image
import torchvision.io as tio
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel

from diffusion.pipeline_wan_c2v import WanVideoConnectPipeline

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def next_run_dir(out_dir: pathlib.Path) -> tuple[str, pathlib.Path]:
    existing = []
    for p in out_dir.glob("run_*"):
        if p.is_dir():
            try:
                existing.append(int(p.name.split("_", 1)[1]))
            except Exception:
                pass
    nxt = (max(existing) + 1) if existing else 1
    run_id  = f"run_{nxt:04d}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def snap_to_grid(value: int, mod: int) -> int:
    """Floor-snap value to the nearest multiple of mod."""
    return value // mod * mod


def target_resolution(pil_frame: PIL.Image.Image, max_area: int, mod_value: int) -> tuple[int, int]:
    """Compute (height, width) preserving aspect ratio so H*W ≈ max_area,
    snapped to mod_value (spatial patch grid requirement)."""
    aspect = pil_frame.height / pil_frame.width
    height = snap_to_grid(round(np.sqrt(max_area * aspect)), mod_value)
    width  = snap_to_grid(round(np.sqrt(max_area / aspect)), mod_value)
    return height, width


def load_clip_from_mp4(path: pathlib.Path, anchor_frames: int) -> list[PIL.Image.Image]:
    """Load the first anchor_frames frames of an mp4 as a list of PIL images.

    Uses torchvision.io.read_video (output_format='THWC') which returns a
    (T, H, W, 3) uint8 tensor at the video's native resolution.
    Resizing to the target resolution is intentionally deferred to the pipeline
    (VideoProcessor.preprocess_video), keeping this function pure I/O.
    """
    frames_t, _, _ = tio.read_video(str(path), output_format="THWC", pts_unit="sec")
    # frames_t: (T, H, W, 3) uint8
    if len(frames_t) < anchor_frames:
        raise ValueError(
            f"{path.name} has {len(frames_t)} frames but anchor_frames={anchor_frames}"
        )
    return [PIL.Image.fromarray(frames_t[i].numpy()) for i in range(anchor_frames)]


def main() -> None:
    cfg = load_config()

    # ── Resolve paths ──────────────────────────────────────────────────────────
    start_path    = REPO_ROOT / cfg["inputs"]["start_clip"]
    end_path      = REPO_ROOT / cfg["inputs"]["end_clip"]
    anchor_frames = cfg["inputs"]["anchor_frames"]
    out_dir       = REPO_ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] start_clip    : {start_path}")
    print(f"[info] end_clip      : {end_path}")
    print(f"[info] anchor_frames : {anchor_frames}")

    # ── Runtime ────────────────────────────────────────────────────────────────
    device = cfg["runtime"]["device"]
    dtype  = torch.bfloat16 if cfg["runtime"]["dtype"] == "bfloat16" else torch.float32
    seed   = cfg["runtime"]["seed"]
    repo   = cfg["model"]["repo_id"]

    print(f"[runtime] device : {device}  dtype : {dtype}")
    print(f"[info] loading pipeline from {repo} …")

    # ── Load pipeline ──────────────────────────────────────────────────────────
    vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
    image_encoder = CLIPVisionModel.from_pretrained(repo, subfolder="image_encoder", torch_dtype=torch.float32)
    pipe = WanVideoConnectPipeline.from_pretrained(
        repo, vae=vae, image_encoder=image_encoder, torch_dtype=dtype,
    )
    if cfg["runtime"].get("cpu_offload", True):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.vae.enable_tiling()

    # ── Load clips as List[PIL.Image] at native resolution ────────────────────
    # Resizing is handled by VideoProcessor.preprocess_video inside the pipeline.
    start_frames = load_clip_from_mp4(start_path, anchor_frames)
    end_frames   = load_clip_from_mp4(end_path,   anchor_frames)

    # ── Compute target resolution from the first frame of the start clip ──────
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height, width = target_resolution(start_frames[0], cfg["inference"]["max_area"], mod_value)
    print(f"[info] resolution    : {width}x{height}  (mod_value={mod_value})")

    # ── Generate ───────────────────────────────────────────────────────────────
    generator = torch.Generator(device=device).manual_seed(seed)

    print("[info] running inference …")
    output = pipe(
        start_clip=start_frames,
        end_clip=end_frames,
        anchor_frames=anchor_frames,
        prompt=cfg["inputs"]["prompt"],
        negative_prompt=cfg["inputs"]["negative_prompt"],
        height=height,
        width=width,
        num_frames=cfg["inference"]["num_frames"],
        num_inference_steps=cfg["inference"]["num_inference_steps"],
        guidance_scale=cfg["inference"]["guidance_scale"],
        generator=generator,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    run_id, run_dir = next_run_dir(out_dir)
    video_name = (
        f"s{seed}"
        f"_af{anchor_frames}"
        f"_nf{cfg['inference']['num_frames']}"
        f"_steps{cfg['inference']['num_inference_steps']}"
        f"_cfg{cfg['inference']['guidance_scale']}.mp4"
    )
    video_path = run_dir / video_name

    export_to_video(output.frames[0], str(video_path), fps=cfg["outputs"]["fps"])

    with (run_dir / "config_snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    print(f"[done] {run_id}  →  {video_path}")


if __name__ == "__main__":
    main()
