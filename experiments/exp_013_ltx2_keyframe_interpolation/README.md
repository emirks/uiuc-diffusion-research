# exp_013_ltx2_keyframe_interpolation

## Question

Can LTX-2's `KeyframeInterpolationPipeline` produce a plausible video transition
between the first and last frame of an action clip from vc-bench-flf, using its
two-stage diffusion + spatial upsampling?

## Setup

- **Pipeline**: `KeyframeInterpolationPipeline` from `ltx_pipelines.keyframe_interpolation`
  (LTX-2, two-stage: 512×768 → 1024×1536 via spatial upsampler).
- **Conditioning**: first frame at index 0, last frame at index `num_frames-1`,
  both with strength 1.0.  Uses guiding-latent conditioning (additive, not
  replacing), which is what the pipeline is designed for.
- **Input clip**: `Actions_Activities_action_action_1581362_2562x1440_b27b9c451a`
  (same clip used in exp_009/exp_010/exp_011 with Wan 2.1).
- **Inference**: 40 steps, seed 42, 97 frames @ 24 fps.
- **Guider defaults**: video CFG 3.0 / STG 1.0 / rescale 0.7 / modality 3.0;
  audio CFG 7.0 / STG 1.0 / rescale 0.7 / modality 3.0.

## How to run

1. **Download models** to `/workspace/cache/models/`:

```bash
pip install huggingface_hub

mkdir -p /workspace/cache/models/LTX-2

# Main checkpoint (~38 GB fp32; use ltx-2-19b-dev-fp8.safetensors for ~19 GB if VRAM-limited)
huggingface-cli download Lightricks/LTX-2 \
  ltx-2-19b-dev.safetensors \
  --local-dir /workspace/cache/models/LTX-2

# Distilled LoRA
huggingface-cli download Lightricks/LTX-2 \
  ltx-2-19b-distilled-lora-384.safetensors \
  --local-dir /workspace/cache/models/LTX-2

# Spatial upsampler
huggingface-cli download Lightricks/LTX-2 \
  ltx-2-spatial-upscaler-x2-1.0.safetensors \
  --local-dir /workspace/cache/models/LTX-2

# Gemma 3 text encoder (full repo, ~8 GB quantized)
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir /workspace/cache/models/gemma-3-12b-it-qat-q4_0-unquantized
```

2. From `diffusion-research/`:

```bash
# Activate the LTX-2 environment
source src/LTX-2/.venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python experiments/exp_013_ltx2_keyframe_interpolation/run.py
```

Or via the CLI entry-point directly:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m ltx_pipelines.keyframe_interpolation \
  --checkpoint-path /workspace/cache/models/LTX-2/ltx-2-19b-dev.safetensors \
  --distilled-lora /workspace/cache/models/LTX-2/ltx-2-19b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /workspace/cache/models/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root /workspace/cache/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A person performs a dynamic physical action outdoors..." \
  --output-path outputs/videos/exp_013_ltx2_keyframe_interpolation/out.mp4 \
  --image data/processed/vc-bench-flf/first_last_frames/Actions_Activities_action_action_1581362_2562x1440_b27b9c451a/first.png 0 1.0 \
  --image data/processed/vc-bench-flf/first_last_frames/Actions_Activities_action_action_1581362_2562x1440_b27b9c451a/last.png 96 1.0 \
  --seed 42 --height 512 --width 768 --num-frames 97
```

## Outputs

Saved under `outputs/videos/exp_013_ltx2_keyframe_interpolation/run_XXXX/`:
- `s42_steps40.mp4` — generated video (1024×1536 after stage-2 upsampling)
- `config_snapshot.yaml` — exact config at run time
