#!/usr/bin/env bash
# Download LTX-2 distilled LoRA + spatial upsampler, and Gemma text encoder.
# Paths match config.yaml. Use LTX-2 venv for huggingface-cli.
set -e
HF=/workspace/diffusion-research/src/LTX-2/.venv/bin/huggingface-cli
export HF_HOME=/workspace/cache/huggingface

mkdir -p /workspace/cache/huggingface/ltx2_models
mkdir -p /workspace/cache/huggingface/gemma

echo "Downloading LTX-2 distilled LoRA and spatial upsampler..."
$HF download Lightricks/LTX-2 \
  ltx-2-19b-distilled-lora-384.safetensors \
  ltx-2-spatial-upscaler-x2-1.0.safetensors \
  --local-dir /workspace/cache/huggingface/ltx2_models

echo "Downloading Gemma 3 text encoder..."
$HF download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir /workspace/cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized

echo "Done. Config paths:"
echo "  distilled_lora_path:  /workspace/cache/huggingface/ltx2_models/ltx-2-19b-distilled-lora-384.safetensors"
echo "  spatial_upsampler_path: /workspace/cache/huggingface/ltx2_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
echo "  gemma_root: /workspace/cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
