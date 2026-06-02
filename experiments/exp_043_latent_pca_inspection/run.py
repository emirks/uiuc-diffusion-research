"""exp_043 — VAE-latent PCA inspection.

Encode a curated set of mp4 clips (shadow_smoke sources, exp_024 DAVIS A_word
generations, exp_041 run_0007 block_out injection outputs) into packed LTX-2
latent tokens, copy the existing exp_033 z1 inverted-noise tensors, and sample
a batch of Gaussian noises at the same shape.  Output is consumed by
`notebooks/exp_043_latent_pca.ipynb`.

All encodes happen at a single resolution (608×608, 121 frames) so the packed
shape is uniformly `[1, 5776, 128]` (16 latent frames × 19×19 spatial × 128
channels), matching the shadow_smoke z0/z1 cached in exp_033.

Per-clip outputs:
  outputs/.../run_NNNN/latents/<group>/<name>.pt   (bfloat16, [1, 5776, 128])
  outputs/.../run_NNNN/manifest.yaml                (group/name/source path)
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import numpy as np
import torch
import torchvision.io as tio
import yaml
from PIL import Image

from diffusers.pipelines.ltx2 import LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents

from diffusion.exp_utils import load_config, next_run_dir, TeeLogger

REPO_ROOT   = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"
DEVICE      = "cuda:0"

log = logging.getLogger(__name__)


def load_frames(path: pathlib.Path, num_frames: int) -> list[Image.Image]:
    video, _, _ = tio.read_video(str(path), pts_unit="sec", output_format="THWC")
    if video.shape[0] == 0:
        raise ValueError(f"No frames in {path}")
    if video.shape[0] < num_frames:
        raise ValueError(f"{path.name}: {video.shape[0]} frames < required {num_frames}")
    src = video[:num_frames]
    return [Image.fromarray(f.numpy()) for f in src]


@torch.inference_mode()
def encode_clip(
    pipe: LTX2ConditionPipeline,
    frames: list[Image.Image],
    height: int,
    width: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """mp4 frames → packed normalized LTX-2 latent tokens [1, N, 128] (bfloat16)."""
    src_tensor = pipe.video_processor.preprocess_video(frames, height=height, width=width)
    src_tensor = src_tensor.to(device=DEVICE, dtype=pipe.vae.dtype)
    latent_5d = retrieve_latents(
        pipe.vae.encode(src_tensor), generator=generator, sample_mode="argmax"
    )
    z = pipe._normalize_latents(
        latent_5d,
        pipe.vae.latents_mean,
        pipe.vae.latents_std,
        pipe.vae.config.scaling_factor,
    )
    z = pipe._pack_latents(
        z,
        pipe.transformer_spatial_patch_size,
        pipe.transformer_temporal_patch_size,
    )
    return z.detach().cpu().to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=CONFIG_PATH)
    args = parser.parse_args()

    cfg     = load_config(args.config)
    out_dir = REPO_ROOT / cfg["outputs"]["dir"]
    run_id, run_dir = next_run_dir(out_dir)
    latents_dir = run_dir / "latents"

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
            force=True,
        )
        log.info("[info] run_dir : %s", run_dir)

        model_id   = cfg["model"]["model_id"]
        height     = int(cfg["inference"]["height"])
        width      = int(cfg["inference"]["width"])
        num_frames = int(cfg["inference"]["num_frames"])
        seed       = int(cfg["runtime"]["seed"])

        log.info("Loading LTX2ConditionPipeline from %s (bf16) …", model_id)
        t0 = time.perf_counter()
        pipe = LTX2ConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=DEVICE)
        pipe.vae.enable_tiling()
        log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        manifest: list[dict] = []

        def save(group: str, name: str, source: str, tensor: torch.Tensor) -> None:
            group_dir = latents_dir / group
            group_dir.mkdir(parents=True, exist_ok=True)
            path = group_dir / f"{name}.pt"
            torch.save(tensor, path)
            manifest.append({
                "group":  group,
                "name":   name,
                "source": source,
                "shape":  list(tensor.shape),
                "dtype":  str(tensor.dtype).replace("torch.", ""),
                "path":   str(path.relative_to(run_dir)),
            })
            log.info("  saved %s/%s  shape=%s  ‖z‖=%.2f",
                     group, name, tuple(tensor.shape), tensor.float().norm().item())

        # ── 1. shadow_smoke source clips → z0 ─────────────────────────────
        ss_cfg = cfg["inputs"]["shadow_smoke"]
        log.info("── encoding shadow_smoke sources (%d) ──", len(ss_cfg["samples"]))
        for entry in ss_cfg["samples"]:
            mp4 = REPO_ROOT / entry["clip"]
            frames = load_frames(mp4, num_frames)
            z = encode_clip(pipe, frames, height, width, generator)
            save("smoke_z0", entry["name"], str(mp4.relative_to(REPO_ROOT)), z)

        # ── 2. existing exp_033 z1 inverted noises → copy ──────────────────
        z1_cfg = cfg["inputs"]["smoke_z1_source"]
        z1_dir = REPO_ROOT / z1_cfg["dir"]
        log.info("── copying exp_033 z1 inverted noises (%d) ──", len(z1_cfg["samples"]))
        for name in z1_cfg["samples"]:
            src_pt = z1_dir / name / "z1.pt"
            z = torch.load(src_pt, map_location="cpu", weights_only=False)
            save("smoke_z1", name, str(src_pt.relative_to(REPO_ROOT)), z)

        # ── 3. exp_024 DAVIS A_word generations → z0 ───────────────────────
        dv_cfg = cfg["inputs"]["davis_generations"]
        dv_dir = REPO_ROOT / dv_cfg["base_dir"]
        log.info("── encoding DAVIS A_word generations (%d) ──", len(dv_cfg["pairs"]))
        for pair in dv_cfg["pairs"]:
            mp4 = dv_dir / dv_cfg["file_template"].format(pair=pair)
            frames = load_frames(mp4, num_frames)
            z = encode_clip(pipe, frames, height, width, generator)
            save("davis_gen_z0", pair, str(mp4.relative_to(REPO_ROOT)), z)

        # ── 4. exp_041 run_0007 injection outputs ──────────────────────────
        ej_cfg = cfg["inputs"]["exp_041_injection"]
        ej_dir = REPO_ROOT / ej_cfg["base_dir"]
        log.info("── encoding exp_041 run_0007 injection outputs (%d variants × %d roles) ──",
                 len(ej_cfg["variants"]), len(ej_cfg["roles"]))
        for variant in ej_cfg["variants"]:
            for role in ej_cfg["roles"]:
                mp4 = ej_dir / variant / f"{role}.mp4"
                frames = load_frames(mp4, num_frames)
                z = encode_clip(pipe, frames, height, width, generator)
                save("exp_041_inject", f"{variant}__{role}",
                     str(mp4.relative_to(REPO_ROOT)), z)

        # ── 5. gaussian noise samples ─────────────────────────────────────
        # Match the dtype/shape of the cached z1 (bfloat16, [1, 5776, 128]).
        n_gauss = int(cfg["inputs"]["gaussian"]["n_samples"])
        log.info("── sampling %d Gaussian noises (N(0,1), bfloat16) ──", n_gauss)
        # Determine shape from a known z1 file if present, else use the
        # standard LTX-2 608×608×121 shape: [1, 5776, 128].
        sample_shape = (1, 5776, 128)
        rng = torch.Generator().manual_seed(seed)
        for i in range(n_gauss):
            z = torch.randn(sample_shape, generator=rng, dtype=torch.float32).to(torch.bfloat16)
            save("gaussian", f"sample_{i:03d}", "randn(0,1)", z)

        # ── manifest ──────────────────────────────────────────────────────
        manifest_path = run_dir / "manifest.yaml"
        with manifest_path.open("w") as f:
            yaml.safe_dump({
                "run_id":     run_id,
                "model_id":   model_id,
                "height":     height,
                "width":      width,
                "num_frames": num_frames,
                "latent_shape": [1, 5776, 128],
                "groups": sorted({m["group"] for m in manifest}),
                "entries": manifest,
            }, f, sort_keys=False)
        log.info("[done] wrote %d latents to %s", len(manifest), latents_dir)
        log.info("[done] manifest → %s", manifest_path)


if __name__ == "__main__":
    main()
