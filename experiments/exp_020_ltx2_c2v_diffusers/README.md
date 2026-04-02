# exp_020 — LTX-2 C2V Diffusers Migration

## Question

Can the official **HuggingFace diffusers** `LTX2ConditionPipeline` reproduce the same
clip-to-video (C2V) results as our vendored `KeyframeInterpolationPipeline` (exp_016)?
What are the practical differences between the two conditioning APIs?

---

## Setup

- **Pipeline**: `LTX2ConditionPipeline` from `diffusers` (official HF library).
- **Conditioning API**: `LTX2VideoCondition(frames=<list[PIL]>, index=<latent_idx>, strength=1.0)`
  — passed as `conditions=[start_cond, end_cond]` to both Stage 1 and Stage 2.
- **Frame loading**: DAVIS JPEG directories loaded directly as PIL frames (no temporary MP4
  needed, unlike exp_016).
- **Samples**: identical 6-pair DAVIS subset from exp_016 — 3 easy + 3 hard pairs.
- **Inference**: 97 frames @ 24 fps (≈ 4 s), 512 × 768, seed 42.
  - Stage 1: 40 steps at 256 × 384 (half resolution).
  - Upsample: `LTX2LatentUpsamplePipeline` × 2.
  - Stage 2: 3 steps at 512 × 768 with distilled LoRA.
- **Guidance** (identical to exp_016):
  - Video: CFG 3.0 / STG 1.0 / modality 3.0 / rescale 0.7 / STG-block [29]
  - Audio: CFG 7.0 / STG 1.0 / modality 3.0 / rescale 0.7

---

## Diffusers vs. our vendored LTX-2 — key differences

| Aspect | exp_016 (vendored `KeyframeInterpolationPipeline`) | exp_020 (diffusers `LTX2ConditionPipeline`) |
|---|---|---|
| Conditioning type | `ClipConditioningInput(path, frame_idx_px, strength, K)` | `LTX2VideoCondition(frames=list[PIL], index=lat_idx, strength)` |
| Coordinate system | **Pixel frame offset** — `frame_idx=0` (start), `frame_idx=N_px − K_px` (end) | **Latent frame index** — `index=0` (start), `index=N_lat − K_lat` (end) |
| Input format | File path → MP4 → decoded internally | `list[PIL.Image]` passed directly |
| Stage-2 conditioning | Re-applied (clips re-encoded at ×2 resolution) | Also re-applied here (same `conditions` list; pipeline resizes internally) |
| State dict caching | Explicit `StateDictRegistry` passed to `ModelLedger` | `enable_sequential_cpu_offload` keeps model on CPU between calls |
| Library | Vendored `src/LTX-2/` (upstream fork) | HuggingFace `diffusers` (pip) |

### End-clip index derivation

```
LTX temporal scale = 8
latent_num_frames   = (num_frames     - 1) // 8 + 1   # 97 → 13
clip_latent_frames  = (num_clip_frames - 1) // 8 + 1   # 25 →  4 (24 → 3)
end_clip_index      = latent_num_frames - clip_latent_frames  # 13 - 4 = 9
```

Diffusers `index=-1` is **not** used for multi-frame end clips — it places only
the very last latent frame as a condition, not the full clip. The correct value
is `latent_num_frames - clip_latent_frames`.

---

## How to run

### 1. Environment setup

```bash
# Activate the LTX-2 venv (has torch 2.9 + CUDA)
source /workspace/diffusion-research/src/LTX-2/.venv/bin/activate

# Install diffusers (one-time); accelerate is needed for cpu_offload
pip install "diffusers>=0.33.0" accelerate
```

### 2. First-run model download

`LTX2ConditionPipeline.from_pretrained("Lightricks/LTX-2")` downloads the
**diffusers-format** model components (~38 GB) into the HF Hub cache the first
time. Set `HF_HOME` to keep everything in the workspace:

```bash
export HF_HOME=/workspace/cache/huggingface
```

> **Note:** The raw safetensors blob already cached from exp_016 is in a different
> format (flat single file) and cannot be reused directly by `from_pretrained`.
> The distilled LoRA and spatial-upsampler safetensors at
> `/workspace/cache/huggingface/ltx2_models/` **are** reusable directly
> via `pipe.load_lora_weights(local_path)`.

### 3. Run

```bash
cd /workspace/diffusion-research
export HF_HOME=/workspace/cache/huggingface
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python experiments/exp_020_ltx2_c2v_diffusers/run.py
```

---

## Outputs

```
outputs/videos/exp_020_ltx2_c2v_diffusers/
└── run_NNNN/
    ├── run.log                        # stdout captured with TeeLogger
    ├── config_snapshot.yaml           # full config at run time
    ├── summary.yaml                   # per-sample elapsed_s + video paths
    └── {sample_id}/
        ├── s{seed}_K{K}_steps{steps}.mp4
        └── config_snapshot.yaml       # per-sample record
```

Frame images are not copied (no source MP4 is created).

---

## Expected interpretation

- With identical guidance settings and the same DAVIS pairs, outputs should be
  visually comparable to exp_016 videos.
- Latent alignment of the end clip is slightly different: our approach anchors
  at pixel offset `N_px − K_px = 72`; diffusers anchors at latent index 9
  (≈ pixel 65). The difference is < 1 latent frame and should be imperceptible.
- The diffusers pipeline uses the same denoising loop internals (Euler flow-
  matching), so generation quality should be equivalent.
