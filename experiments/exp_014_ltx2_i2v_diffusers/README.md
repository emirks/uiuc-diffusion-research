# exp_014_ltx2_i2v_diffusers

## Question

Can the official diffusers `LTX2ImageToVideoPipeline` produce coherent video from a single conditioning image using the `Lightricks/LTX-2` checkpoint, as a clean baseline for the diffusers API?

## Setup

Two-stage pipeline, mirroring the official diffusers docs example:

| Stage | Component | Steps | CFG |
|-------|-----------|-------|-----|
| 1 | `LTX2ImageToVideoPipeline` (19B full model, I2V) | 40 | 4.0 |
| — | `LTX2LatentUpsamplePipeline` (spatial ×2 in latent space) | — | — |
| 2 | same pipe + distilled LoRA (`ltx-2-19b-distilled-lora-384`) | 3 | 1.0 |

- **Model**: `Lightricks/LTX-2` via `from_pretrained` (downloaded to `/workspace/cache/huggingface/hub/`).
- **Conditioning**: first frame of the `Actions_Activities_action_action_1581362` vc-bench-flf clip (same as exp_013).
- **Memory**: `enable_sequential_cpu_offload` + `vae.enable_tiling()` to run within available VRAM.
- **Key params**: 512×768, 121 frames, seed=42.

## Relationship to exp_013

exp_013 used the native `ltx-pipelines` `KeyframeInterpolationPipeline` (first+last frame).
This experiment uses the official diffusers I2V API (first frame only) as a direct baseline.

## How to run

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/envs/diff

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python experiments/exp_014_ltx2_i2v_diffusers/run.py
```

**First run will download `Lightricks/LTX-2` in diffusers format** (~tens of GBs).
The cache is stored under `/workspace/cache/huggingface/hub/`.

## Outputs

All artifacts land in `outputs/videos/exp_014_ltx2_i2v_diffusers/run_XXXX/`:
- `s42_steps40.mp4` — generated video with audio
- `config_snapshot.yaml` — frozen config for reproducibility
