"""exp_023 — VAE latent linear interpolation (dissolve-cause probe).

Hypothesis under test
---------------------
Exp-020 (LTX-2 C2V with rectified flow) shows a dissolve / cross-fade in the
transition region between the start and end clips.  Rectified flow is trained to
travel in a *straight line* from noise to data in latent space.  If the model
"goes straight" from the start-clip latent to the end-clip latent, that straight
path decoded frame-by-frame is exactly a VAE linear interpolation.

This experiment isolates that mechanism: for the *same* samples as exp-020, we
**skip diffusion entirely** and instead

    1. Encode the start clip with the LTX-2 VAE  → start latents (T_clip lat frames)
    2. Encode the end   clip with the LTX-2 VAE  → end   latents (T_clip lat frames)
    3. Build a T_total-frame latent volume by linearly interpolating along the
       temporal axis from start latents to end latents.
    4. Decode the interpolated latent directly with the VAE.
    5. Save as MP4 (no audio — no diffusion run means no audio latents).

If the decoded video shows the same dissolve artefact, the straight-path nature
of rectified flow is a sufficient explanation.  If not, the dissolve must arise
from something else (text conditioning, noise initialisation, etc.).

Interpolation scheme
--------------------
Let T_clip = (num_clip_frames - 1) // 8 + 1  (= 4 for 25 px frames)
    T_total = (num_frames      - 1) // 8 + 1  (= 16 for 121 px frames)

For each output latent frame t ∈ [0, T_total):
    alpha   = t / (T_total - 1)               # 0 → 1 linearly
    s_idx   = min(t, T_clip - 1)              # clamp into start latent range [0, T_clip)
    e_idx   = max(t - (T_total - T_clip), 0)  # clamp into end   latent range [0, T_clip)
    out[t]  = (1 - alpha) * start_lat[s_idx] + alpha * end_lat[e_idx]

This ensures:
  • t = 0         → pure start_lat[0]         (start of start clip)
  • t = T_total-1 → pure end_lat[T_clip-1]   (end of end clip)
  • middle frames → smooth linear cross-blend

Implementation note
-------------------
Only the LTX-2 VAE is loaded (AutoencoderKLLTX2Video subfolder="vae").
The 19B transformer, text encoder, audio models, etc. are never loaded.
A VideoProcessor is constructed with the same vae_scale_factor=32 the pipeline uses.

How to run:
    source /workspace/miniforge3/etc/profile.d/conda.sh
    conda activate /workspace/envs/diff
    cd /workspace/diffusion-research
    python experiments/exp_023_vae_latent_lerp/run.py
"""
from __future__ import annotations

import argparse
import glob
import logging
import pathlib
import sys
import time

import numpy as np
import torch
import torchvision.io as tio
import yaml
from PIL import Image

from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.video_processor import VideoProcessor

from diffusion.exp_utils import load_config, next_run_dir, TeeLogger

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"

LTX_TEMPORAL_SCALE    = 8   # VAE causal temporal downscale factor
LTX_SPATIAL_SCALE     = 32  # VAE spatial downscale factor (matches pipeline default)
DEVICE                = "cuda:0"

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


# ── VAE helpers ───────────────────────────────────────────────────────────────

@torch.inference_mode()
def encode_clip(
    vae: AutoencoderKLLTX2Video,
    video_processor: VideoProcessor,
    frames: list[Image.Image],
    height: int,
    width: int,
) -> torch.Tensor:
    """Encode a list of PIL frames → raw VAE latents [1, C, T_lat, H_lat, W_lat].

    Preprocessing: resize/crop to (height, width) and normalize to [-1, 1].
    Encoding: deterministic (mode / argmax of the posterior distribution).
    Returns raw latents — no scaling_factor applied — ready for vae.decode().
    """
    # preprocess_video: list[PIL] → [B=1, C, T, H, W]  in [-1, 1]
    pixel_tensor = video_processor.preprocess_video(
        frames, height=height, width=width, resize_mode="crop"
    ).to(dtype=vae.dtype, device=vae.device)

    encoder_out = vae.encode(pixel_tensor)
    if hasattr(encoder_out, "latent_dist"):
        latents = encoder_out.latent_dist.mode()   # deterministic
    else:
        latents = encoder_out.latents
    return latents  # [1, C, T_lat, H_lat, W_lat]


@torch.inference_mode()
def lerp_latents(
    start_lat: torch.Tensor,
    end_lat: torch.Tensor,
    T_total: int,
) -> torch.Tensor:
    """Build a T_total-frame latent by linearly interpolating start → end.

    start_lat / end_lat shape: [1, C, T_clip, H_lat, W_lat]

    Mapping for frame t ∈ [0, T_total):
        alpha   = t / (T_total - 1)
        s_idx   = min(t, T_clip - 1)
        e_idx   = max(t - (T_total - T_clip), 0)
        out[t]  = (1 - alpha) * start_lat[:, :, s_idx] + alpha * end_lat[:, :, e_idx]

    Boundary behaviour:
      - t = 0            → alpha=0,   pure start_lat[0]
      - t = T_total-1    → alpha=1,   pure end_lat[-1]
      - start region     → start clip frames indexed directly, end contribution growing from 0
      - end region       → end clip frames indexed directly, start contribution shrinking to 0
      - transition zone  → blends start_lat[-1] ↔ end_lat[0]
    """
    T_clip = start_lat.shape[2]
    assert end_lat.shape[2] == T_clip, "start and end must have the same number of latent frames"

    out = torch.zeros(
        start_lat.shape[0], start_lat.shape[1], T_total,
        start_lat.shape[3], start_lat.shape[4],
        dtype=start_lat.dtype, device=start_lat.device,
    )
    for t in range(T_total):
        alpha = t / (T_total - 1)
        s_idx = min(t, T_clip - 1)
        e_idx = max(t - (T_total - T_clip), 0)
        out[:, :, t] = (1.0 - alpha) * start_lat[:, :, s_idx] + alpha * end_lat[:, :, e_idx]

    return out  # [1, C, T_total, H_lat, W_lat]


@torch.inference_mode()
def decode_latents(
    vae: AutoencoderKLLTX2Video,
    latents: torch.Tensor,
) -> np.ndarray:
    """Decode raw latents → numpy video [T, H, W, C] in [0, 1].

    decode_timestep=0.0 → treat as perfectly clean latents (no noise mixed in).
    """
    latents_fp = latents.to(dtype=vae.dtype, device=vae.device)

    if getattr(vae.config, "timestep_conditioning", False):
        timestep = torch.tensor([0.0] * latents_fp.shape[0], device=vae.device, dtype=latents_fp.dtype)
    else:
        timestep = None

    decoded = vae.decode(latents_fp, timestep, return_dict=False)[0]
    # decoded: [B, C, T, H, W] in [-1, 1]
    # → clamp, rescale, permute → [B, T, H, W, C] in [0, 1]
    decoded = (decoded.float().clamp(-1.0, 1.0) + 1.0) / 2.0
    decoded = decoded.permute(0, 2, 3, 4, 1).cpu().numpy()  # [B, T, H, W, C]
    return decoded


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
        num_frames      = cfg["inference"]["num_frames"]
        frame_rate      = float(cfg["inference"]["frame_rate"])
        height          = cfg["inference"]["height"]
        width           = cfg["inference"]["width"]
        seed            = cfg["runtime"]["seed"]
        num_clip_frames = cfg["inputs"]["num_clip_frames"]

        T_total = (num_frames      - 1) // LTX_TEMPORAL_SCALE + 1  # 16 for 121 frames
        T_clip  = (num_clip_frames - 1) // LTX_TEMPORAL_SCALE + 1  # 4  for 25  frames
        log.info(
            "Latent dims: T_total=%d  T_clip=%d  (num_frames=%d  num_clip_frames=%d)",
            T_total, T_clip, num_frames, num_clip_frames,
        )

        # ── Load only the VAE (no transformer / text encoder / audio models) ──
        # Loading the full LTX2ConditionPipeline is unnecessary — only the VAE and
        # a VideoProcessor are required for encode → lerp → decode.
        log.info("Loading VAE from %s (subfolder=vae) …", model_id)
        t0  = time.perf_counter()
        vae = AutoencoderKLLTX2Video.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        ).to(DEVICE)
        vae.enable_tiling()
        log.info("VAE loaded in %.1fs.", time.perf_counter() - t0)

        # VideoProcessor: same configuration as LTX2ConditionPipeline uses.
        video_processor = VideoProcessor(vae_scale_factor=LTX_SPATIAL_SCALE, resample="bilinear")

        # ── Per-sample loop ───────────────────────────────────────────────────
        summary: list[dict] = []

        for idx, sample in enumerate(cfg["samples"]):
            sample_id  = sample["sample_id"]
            sample_dir = run_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            start_src = sample.get("start_clip") or sample.get("start_images", "")
            end_src   = sample.get("end_clip")   or sample.get("end_images",   "")
            log.info("─── Sample %d/%d  id=%s ───", idx + 1, len(cfg["samples"]), sample_id)
            print(f"[info] start  : {start_src}")
            print(f"[info] end    : {end_src}")

            t_infer = time.perf_counter()

            start_frames, end_frames = load_clip_frames(sample, REPO_ROOT, num_clip_frames)

            # ── Step 1 & 2: Encode both clips ─────────────────────────────────
            log.info("Encoding start clip …")
            start_lat = encode_clip(vae, video_processor, start_frames, height, width)
            log.info("  start_lat  shape : %s  dtype=%s", tuple(start_lat.shape), start_lat.dtype)

            log.info("Encoding end clip …")
            end_lat = encode_clip(vae, video_processor, end_frames, height, width)
            log.info("  end_lat    shape : %s  dtype=%s", tuple(end_lat.shape), end_lat.dtype)

            # ── Step 3: Linear interpolation in latent space ──────────────────
            log.info("Interpolating: %d clip lat frames → %d total lat frames …", T_clip, T_total)
            interp_lat = lerp_latents(start_lat, end_lat, T_total)
            log.info("  interp_lat shape : %s", tuple(interp_lat.shape))

            # ── Step 4: Decode ────────────────────────────────────────────────
            log.info("Decoding interpolated latents …")
            t_dec = time.perf_counter()
            video_np = decode_latents(vae, interp_lat)  # [1, T, H, W, C] in [0, 1]
            log.info("  VAE decode done in %.1fs.", time.perf_counter() - t_dec)

            elapsed = time.perf_counter() - t_infer
            log.info("Sample done in %.1fs.", elapsed)

            # ── Step 5: Save ──────────────────────────────────────────────────
            video_path = sample_dir / f"s{seed}_K{num_clip_frames}_lerp.mp4"
            encode_video(
                video_np[0],
                fps=int(frame_rate),
                audio=None,
                audio_sample_rate=None,
                output_path=str(video_path),
            )
            log.info("Saved %s", video_path)

            with (sample_dir / "config_snapshot.yaml").open("w") as f:
                yaml.safe_dump({
                    "sample_id":     sample_id,
                    "start_src":     start_src,
                    "end_src":       end_src,
                    "method":        "vae_latent_linear_interpolation",
                    "T_total":       T_total,
                    "T_clip":        T_clip,
                    "num_frames":    num_frames,
                    "num_clip_frames": num_clip_frames,
                    "height":        height,
                    "width":         width,
                    "runtime":       cfg["runtime"],
                    "output":        str(video_path),
                    "elapsed_s":     round(elapsed, 1),
                }, f, sort_keys=False, allow_unicode=True)

            summary.append({"sample_id": sample_id, "video": str(video_path), "elapsed_s": round(elapsed, 1)})

        # ── Run-level artefacts ───────────────────────────────────────────────
        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"run_id": run_id, "samples": summary}, f, sort_keys=False, allow_unicode=True)

        total = sum(s["elapsed_s"] for s in summary)
        log.info("All %d samples done.  Total: %.1fs", len(summary), total)
        for s in summary:
            print(f"[done] {s['sample_id']}  →  {s['video']}")


if __name__ == "__main__":
    main()
