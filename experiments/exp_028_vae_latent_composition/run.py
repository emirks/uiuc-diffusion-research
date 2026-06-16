"""exp_028 — VAE latent composition with held boundaries + pixel-anchored bridge.

Fork of exp_023. Same VAE-only mechanism, but two fixes:

  (1) The start- and end-clip latents are *held pure* in their respective
      regions of the output timeline. exp_023's blend started at t=1
      because its alpha ramped across the entire timeline; here, alpha is
      zero everywhere except the middle.

  (2) The middle region can be filled either by a latent-space lerp
      between start_lat[-1] and end_lat[0] (`hold_lerp_hold`), or by a
      lerp between *single-frame VAE encodings* of last_start_frame and
      first_end_frame (`hold_bridge_hold`). The latter avoids the
      semantic mismatch in `hold_lerp_hold` (where one endpoint is a
      motion latent and the other is a key-frame latent), and avoids the
      double-counting of `first_end_pixel` that the earlier pixel-bridge
      version had (where the bridge clip ended on `first_end_pixel` and
      `end_lat[0]` immediately re-encoded the same pixel).

The script sweeps `num_frames_sweep` (e.g. [121, 89, 65]) per sample to
visualise how bridge size (M = T_total - 2*T_clip) affects perceived
dissolve speed and stability.

Three modes are  per sample for visual comparison:
  - naive            : exp_023 baseline (whole-timeline lerp)
  - hold_lerp_hold   : pure boundaries + latent-space lerp in middle
  - hold_bridge_hold : pure boundaries + VAE-encoded pixel cross-fade

How to run:
    source /workspace/miniforge3/etc/profile.d/conda.sh
    conda activate /workspace/envs/diff
    cd /workspace/diffusion-research
    python experiments/exp_028_vae_latent_composition/run.py
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

from diffusion.exp_utils iwrittenmport load_config, next_run_dir, TeeLogger

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"

LTX_TEMPORAL_SCALE = 8
LTX_SPATIAL_SCALE  = 32
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


# ── VAE encode / decode ───────────────────────────────────────────────────────

@torch.inference_mode()
def encode_clip(
    vae: AutoencoderKLLTX2Video,
    video_processor: VideoProcessor,
    frames: list[Image.Image],
    height: int,
    width: int,
) -> torch.Tensor:
    """Encode PIL frames → raw latents [1, C, T_lat, H_lat, W_lat] (deterministic)."""
    pixel_tensor = video_processor.preprocess_video(
        frames, height=height, width=width, resize_mode="crop"
    ).to(dtype=vae.dtype, device=vae.device)
    encoder_out = vae.encode(pixel_tensor)
    if hasattr(encoder_out, "latent_dist"):
        latents = encoder_out.latent_dist.mode()
    else:
        latents = encoder_out.latents
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKLLTX2Video, latents: torch.Tensor) -> np.ndarray:
    """Decode raw latents → numpy video [B, T, H, W, 3] in [0, 1]."""
    latents_fp = latents.to(dtype=vae.dtype, device=vae.device)
    if getattr(vae.config, "timestep_conditioning", False):
        timestep = torch.tensor([0.0] * latents_fp.shape[0], device=vae.device, dtype=latents_fp.dtype)
    else:
        timestep = None
    decoded = vae.decode(latents_fp, timestep, return_dict=False)[0]
    decoded = (decoded.float().clamp(-1.0, 1.0) + 1.0) / 2.0
    decoded = decoded.permute(0, 2, 3, 4, 1).cpu().numpy()
    return decoded


# ── Composition modes ─────────────────────────────────────────────────────────

@torch.inference_mode()
def compose_naive(start_lat: torch.Tensor, end_lat: torch.Tensor, T_total: int) -> torch.Tensor:
    """exp_023 baseline: alpha ramps across the entire timeline.

    For each output position t:
        alpha = t / (T_total - 1)
        s_idx = min(t, T_clip - 1)
        e_idx = max(t - (T_total - T_clip), 0)
        out[t] = (1 - alpha) * start_lat[s_idx] + alpha * end_lat[e_idx]
    """
    T_clip = start_lat.shape[2]
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
    return out


@torch.inference_mode()
def compose_hold_lerp_hold(start_lat: torch.Tensor, end_lat: torch.Tensor, T_total: int) -> torch.Tensor:
    """Pure start_lat in head, pure end_lat in tail, latent-space lerp in middle.

    Layout (T_clip = start/end latent frame count, M = T_total - 2*T_clip):
        [0 .. T_clip)              = start_lat[0 .. T_clip)
        [T_clip .. T_total-T_clip) = lerp(start_lat[-1], end_lat[0], M+1 anchors)
        [T_total-T_clip .. T_total) = end_lat[0 .. T_clip)

    The middle lerp uses M+1 anchor points spanning *exclusively* between
    start_lat[-1] and end_lat[0] — endpoints excluded so we don't duplicate
    the held frames.
    """
    T_clip = start_lat.shape[2]
    M = T_total - 2 * T_clip
    if M < 0:
        raise ValueError(
            f"hold_lerp_hold requires T_total >= 2*T_clip; got T_total={T_total}, T_clip={T_clip}"
        )

    out = torch.zeros(
        start_lat.shape[0], start_lat.shape[1], T_total,
        start_lat.shape[3], start_lat.shape[4],
        dtype=start_lat.dtype, device=start_lat.device,
    )

    # Head: hold start clip pure.
    out[:, :, :T_clip] = start_lat
    # Tail: hold end clip pure.
    out[:, :, T_total - T_clip:] = end_lat

    # Middle: lerp strictly between start_lat[-1] and end_lat[0]. We choose
    # the M middle positions so alphas are (1, 2, ..., M) / (M+1) — strictly
    # interior, never reaching the endpoints.
    if M > 0:
        a = start_lat[:, :, -1:]   # [1, C, 1, H, W]
        b = end_lat[:, :, :1]      # [1, C, 1, H, W]
        for j in range(M):
            alpha = (j + 1) / (M + 1)
            out[:, :, T_clip + j] = ((1.0 - alpha) * a + alpha * b)[:, :, 0]
    return out


@torch.inference_mode()
def compose_hold_bridge_hold(
    start_lat:        torch.Tensor,
    end_lat:          torch.Tensor,
    T_total:          int,
    start_frames:     list[Image.Image],
    end_frames:       list[Image.Image],
    vae:              AutoencoderKLLTX2Video,
    video_processor:  VideoProcessor,
    height:           int,
    width:            int,
) -> tuple[torch.Tensor, dict]:
    """Pure start_lat + pure end_lat + a *single-frame-encoded lerp* in the middle.

    The previous pixel-cross-fade variant double-anchored ``first_end_pixel``:
    it appeared both at the tail of the bridge clip and at ``end_lat[0]``,
    so two adjacent latent slots described the same source pixel and the
    second-clip onset came in twice. This version drops that overlap.

    Mechanism:
      * VAE-encode ``last_start_frame``  as a 1-frame clip → key-frame
        latent ``A`` of shape ``(1, C, 1, H, W)``.
      * VAE-encode ``first_end_frame``   as a 1-frame clip → key-frame
        latent ``B`` of shape ``(1, C, 1, H, W)``.
      * Lerp between ``A`` and ``B`` at M strictly-interior alphas
        ``(1..M)/(M+1)`` to fill the M middle slots.

    Why single-frame encodings (rather than reusing ``start_lat[-1]`` /
    ``end_lat[0]``)? ``start_lat[-1]`` is a *motion* latent encoding pixel
    frames 17-24 of the start clip; ``end_lat[0]`` is a *key-frame* latent
    encoding only pixel frame 0 of the end clip. Lerping between two
    different semantic types of latent is what produced exp_023's middle
    flicker. Re-encoding each endpoint as a 1-frame clip puts both anchors
    on the same surface of the VAE manifold (both are key-frame latents),
    so the lerp stays inside a single semantic regime.

    Decoder behaviour to keep in mind: the LTX-2 VAE treats latent slot 0
    as a key-frame (decodes to 1 pixel) and slots 1..15 as motion (decode
    to 8 pixels each). The M middle slots sit in motion positions, so the
    decoder will *interpret* these single-frame-encoded latents as motion
    and produce 8 pixel frames per slot — typically a near-static "hold"
    of whatever the latent encodes. The user explicitly wants this: the
    bridge will look "stuck" because the only true single-frame anchor in
    the decoded volume is position 0 (``start_lat[0]``).

    Returns (composed_latents, diagnostics_dict).
    """
    T_clip = start_lat.shape[2]
    M = T_total - 2 * T_clip
    if M <= 0:
        raise ValueError(
            f"hold_bridge_hold requires T_total > 2*T_clip; got T_total={T_total}, T_clip={T_clip}"
        )

    last_start_lat = encode_clip(vae, video_processor, [start_frames[-1]], height, width)
    first_end_lat  = encode_clip(vae, video_processor, [end_frames[0]],   height, width)
    if last_start_lat.shape[2] != 1 or first_end_lat.shape[2] != 1:
        raise RuntimeError(
            "Expected single-frame encodings to produce 1 latent each; got "
            f"last_start_lat={tuple(last_start_lat.shape)}, "
            f"first_end_lat={tuple(first_end_lat.shape)}"
        )
    A = last_start_lat[:, :, 0:1]
    B = first_end_lat [:, :, 0:1]
    log.info("  single-frame anchors encoded: A=%s  B=%s  → lerping %d middle latents",
             tuple(A.shape), tuple(B.shape), M)

    out = torch.zeros(
        start_lat.shape[0], start_lat.shape[1], T_total,
        start_lat.shape[3], start_lat.shape[4],
        dtype=start_lat.dtype, device=start_lat.device,
    )
    out[:, :, :T_clip]            = start_lat
    out[:, :, T_total - T_clip:]  = end_lat
    for j in range(M):
        alpha = (j + 1) / (M + 1)
        out[:, :, T_clip + j] = ((1.0 - alpha) * A + alpha * B)[:, :, 0]

    diag = {
        "M_latent_frames":      int(M),
        "anchor_A_shape":       tuple(A.shape),
        "anchor_B_shape":       tuple(B.shape),
        "lerp_alphas":          [round((j + 1) / (M + 1), 4) for j in range(M)],
    }
    return out, diag


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
        frame_rate      = float(cfg["inference"]["frame_rate"])
        height          = cfg["inference"]["height"]
        width           = cfg["inference"]["width"]
        seed            = cfg["runtime"]["seed"]
        num_clip_frames = cfg["inputs"]["num_clip_frames"]
        modes           = list(cfg["composition"]["modes"])

        if "num_frames_sweep" in cfg["inference"]:
            num_frames_sweep = list(cfg["inference"]["num_frames_sweep"])
        else:
            num_frames_sweep = [int(cfg["inference"]["num_frames"])]

        T_clip = (num_clip_frames - 1) // LTX_TEMPORAL_SCALE + 1
        log.info(
            "T_clip=%d  (num_clip_frames=%d) | num_frames_sweep=%s | modes=%s",
            T_clip, num_clip_frames, num_frames_sweep, modes,
        )

        # Pre-validate that every length × mode combo is feasible.
        for nf in num_frames_sweep:
            T_total = (nf - 1) // LTX_TEMPORAL_SCALE + 1
            M       = T_total - 2 * T_clip
            log.info("  length %d → T_total=%d  middle_M=%d", nf, T_total, M)
            if "hold_bridge_hold" in modes and M <= 0:
                raise ValueError(
                    f"hold_bridge_hold needs M>=1 (T_total>2*T_clip); num_frames={nf} → "
                    f"T_total={T_total}, T_clip={T_clip}, M={M}. Pick a larger num_frames."
                )

        # ── Load VAE (and only the VAE) ───────────────────────────────────────
        log.info("Loading VAE from %s (subfolder=vae) …", model_id)
        t0  = time.perf_counter()
        vae = AutoencoderKLLTX2Video.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        ).to(DEVICE)
        vae.enable_tiling()
        log.info("VAE loaded in %.1fs.", time.perf_counter() - t0)

        video_processor = VideoProcessor(vae_scale_factor=LTX_SPATIAL_SCALE, resample="bilinear")

        # ── Per-sample loop ───────────────────────────────────────────────────
        summary: list[dict] = []
        for idx, sample in enumerate(cfg["samples"]):
            sample_id  = sample["sample_id"]
            sample_dir = run_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            start_src = sample.get("start_clip") or sample.get("start_images", "")
            end_src   = sample.get("end_clip")   or sample.get("end_images",   "")
            log.info("─── Sample %d/%d  id=%s  difficulty=%s ───",
                     idx + 1, len(cfg["samples"]), sample_id, sample.get("difficulty", "?"))
            print(f"[info] start : {start_src}")
            print(f"[info] end   : {end_src}")

            t_sample = time.perf_counter()

            # ── Load pixel frames + encode both clips ─────────────────────────
            start_frames, end_frames = load_clip_frames(sample, REPO_ROOT, num_clip_frames)
            log.info("Encoding start clip …")
            start_lat = encode_clip(vae, video_processor, start_frames, height, width)
            log.info("  start_lat shape : %s  dtype=%s", tuple(start_lat.shape), start_lat.dtype)
            log.info("Encoding end clip …")
            end_lat = encode_clip(vae, video_processor, end_frames, height, width)
            log.info("  end_lat   shape : %s  dtype=%s", tuple(end_lat.shape), end_lat.dtype)

            sample_summary: dict = {
                "sample_id":  sample_id,
                "difficulty": sample.get("difficulty"),
                "start_src":  start_src,
                "end_src":    end_src,
                "runs":       {},   # keyed as "N{num_frames}/{mode}"
            }

            # ── Run each length × mode ────────────────────────────────────────
            for nf in num_frames_sweep:
                T_total = (nf - 1) // LTX_TEMPORAL_SCALE + 1
                M       = T_total - 2 * T_clip
                log.info("── length: num_frames=%d  T_total=%d  M=%d ──", nf, T_total, M)

                for mode in modes:
                    t_mode = time.perf_counter()
                    log.info("─ mode: %s ─", mode)
                    if mode == "naive":
                        composed = compose_naive(start_lat, end_lat, T_total)
                        diag = {}
                    elif mode == "hold_lerp_hold":
                        composed = compose_hold_lerp_hold(start_lat, end_lat, T_total)
                        diag = {}
                    elif mode == "hold_bridge_hold":
                        composed, diag = compose_hold_bridge_hold(
                            start_lat, end_lat, T_total,
                            start_frames, end_frames,
                            vae, video_processor, height, width,
                        )
                    else:
                        raise ValueError(f"unknown composition mode: {mode!r}")

                    log.info("  composed shape : %s", tuple(composed.shape))

                    log.info("Decoding …")
                    t_dec = time.perf_counter()
                    video_np = decode_latents(vae, composed)
                    log.info("  decode done in %.1fs.", time.perf_counter() - t_dec)

                    video_path = sample_dir / f"s{seed}_K{num_clip_frames}_N{nf}_M{M}_mode-{mode}.mp4"
                    encode_video(
                        video_np[0],
                        fps=int(frame_rate),
                        audio=None,
                        audio_sample_rate=None,
                        output_path=str(video_path),
                    )
                    elapsed_mode = time.perf_counter() - t_mode
                    log.info("  saved %s  (%.1fs)", video_path, elapsed_mode)

                    sample_summary["runs"][f"N{nf}/{mode}"] = {
                        "num_frames": int(nf),
                        "T_total":    int(T_total),
                        "M":          int(M),
                        "mode":       mode,
                        "video":      str(video_path),
                        "elapsed_s":  round(elapsed_mode, 1),
                        **diag,
                    }

            elapsed_sample = time.perf_counter() - t_sample
            log.info("Sample done in %.1fs.", elapsed_sample)

            with (sample_dir / "config_snapshot.yaml").open("w") as f:
                yaml.safe_dump({
                    "sample_id":         sample_id,
                    "start_src":         start_src,
                    "end_src":           end_src,
                    "method":            "vae_latent_composition",
                    "T_clip":            T_clip,
                    "num_clip_frames":   num_clip_frames,
                    "num_frames_sweep":  num_frames_sweep,
                    "height":            height,
                    "width":             width,
                    "modes_run":         modes,
                    "runs":              sample_summary["runs"],
                    "runtime":           cfg["runtime"],
                    "elapsed_s":         round(elapsed_sample, 1),
                }, f, sort_keys=False, allow_unicode=True)

            summary.append(sample_summary)

        # ── Run-level artefacts ───────────────────────────────────────────────
        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"run_id": run_id, "samples": summary}, f, sort_keys=False, allow_unicode=True)

        log.info("All %d samples done.", len(summary))
        for s in summary:
            for key, info in s["runs"].items():
                print(f"[done] {s['sample_id']:45s}  {key:30s}  →  {info['video']}")


if __name__ == "__main__":
    main()
