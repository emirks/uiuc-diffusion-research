# Exp 013: Implementation Review + Lightest-Weight Config Roadmap

## 1. Experiment implementation review

### Summary: **Implementation is correct**

Your `run.py` matches the current `KeyframeInterpolationPipeline` API in `src/LTX-2` and should run as-is.

### Checklist

| Item | Status | Notes |
|------|--------|--------|
| **Constructor** | OK | `checkpoint_path`, `distilled_lora` (list of `LoraPathStrengthAndSDOps`), `spatial_upsampler_path`, `gemma_root`, `loras=[]`. No `quantization` passed (optional). |
| **Imports** | OK | `ltx_core.loader` exports `LTXV_LORA_COMFY_RENAMING_MAP` and `LoraPathStrengthAndSDOps`; `ImageConditioningInput` from `ltx_pipelines.utils.args`; `encode_video` from `ltx_pipelines.utils.media_io`. |
| **Image conditioning** | OK | `ImageConditioningInput(path, frame_idx, strength)` — 3 args, `crf` uses default. First at frame 0, last at `num_frames - 1`. |
| **Pipeline call** | OK | All required args present: `prompt`, `negative_prompt`, `seed`, `height`/`width`, `num_frames`, `frame_rate`, `num_inference_steps`, `video_guider_params`, `audio_guider_params`, `images`, `tiling_config`. |
| **Return value** | OK | Pipeline returns `(decoded_video: Iterator[torch.Tensor], decoded_audio: Audio)`. You pass `video` into `encode_video`, which accepts `Iterator[torch.Tensor]`. |
| **encode_video** | OK | Signature `(video, fps, audio, output_path, video_chunks_number)`. You pass `fps=int(frame_rate)`, `video_chunks_number=get_video_chunks_number(...)`. |
| **Tiling** | OK | `TilingConfig.default()` and `get_video_chunks_number(num_frames, tiling_config)` match pipeline usage. |
| **Guider params** | OK | `MultiModalGuiderParams` with `cfg_scale`, `stg_scale`, `rescale_scale`, `modality_scale`, `skip_step`, `stg_blocks` — matches pipeline and README. |

### Optional fix (cosmetic)

- **config.yaml comment**  
  You have: `height: 512  # stage-1 height (stage-2 doubles to 1024)`.  
  In `KeyframeInterpolationPipeline`, `height`/`width` are the **final** (stage-2) resolution; stage 1 uses `height//2`, `width//2`. So with 512×768 you get stage-1 256×384 and stage-2 512×768. Suggested comment:  
  `# final output resolution; stage-1 runs at half (256×384)`.

---

## 2. Lightest-weight configuration roadmap

Goal: **Minimize VRAM and compute** while keeping KeyframeInterpolationPipeline usable. Below is a research-backed roadmap (no code changes applied).

### 2.1 Model choice (KeyframeInterpolationPipeline constraint)

- **KeyframeInterpolationPipeline** uses the **full** checkpoint in stage 1 and the **distilled LoRA** only in stage 2. It does **not** use the distilled *checkpoint* (unlike `DistilledPipeline` / `ICLoraPipeline`).
- So you cannot switch to the distilled checkpoint for this pipeline. You can:
  - Use the **FP8 full checkpoint** to cut transformer VRAM (~40% vs BF16).
  - Enable **FP8 quantization in code** on top of the same checkpoint for additional savings.

### 2.2 Minimal settings (research summary)

| Lever | Lightest / minimal | Effect |
|-------|--------------------|--------|
| **Checkpoint file** | `ltx-2-19b-dev-fp8.safetensors` | Pre-quantized FP8 full model → ~40% less VRAM than BF16. KeyframeInterpolationPipeline expects the **full** checkpoint (stage 1) + distilled **LoRA** (stage 2); do not use the distilled *checkpoint* for this pipeline. |
| **Runtime quantization** | `quantization=QuantizationPolicy.fp8_cast()` in `KeyframeInterpolationPipeline(..., quantization=...)` | ~40% transformer VRAM reduction, works on most GPUs. |
| **FP8 scaled MM** | `QuantizationPolicy.fp8_scaled_mm()` | Best VRAM/speed on Hopper (H100); requires `uv sync --frozen --extra fp8-trtllm`. |
| **Resolution** | Smallest valid: e.g. 384×256 or 512×320 (both div. by 32). Stage 1 then 192×128 or 256×160. | Lower res = much less VRAM and faster. |
| **Frames** | Minimum for 8n+1: e.g. 17 or 33 (≈0.7–1.4 s @ 24 fps). | Fewer frames = less memory and time. |
| **Steps** | Reduce stage-1 steps (e.g. 20–30 instead of 40). README suggests gradient estimation can keep quality with fewer steps. | Fewer steps = less compute. |
| **Environment** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Helps allocation; you already use this. |
| **Tiling** | Keep `TilingConfig.default()` (VAE decode is tiled). | Already optimal for memory. |

### 2.3 VRAM ballpark (from web + README)

- **BF16 full model:** ~32 GB+.
- **FP8 (cast or pre-quantized):** ~40–50% less → ~16–20 GB range.
- **Distilled checkpoint** (where applicable): 2–5 GB less than full at same settings; **not** applicable to KeyframeInterpolationPipeline’s stage-1 (full model).
- **Resolution/frames:** 480p 72 frames ~6–9 GB with FP8; 720p 72 frames ~9 GB FP8. Your 512×768 @ 97 frames will sit between these if you add FP8.

### 2.4 Recommended implementation roadmap (lightest weight)

1. **Enable FP8 quantization in the experiment**
   - In `run.py`, add:
     - `from ltx_core.quantization import QuantizationPolicy`
     - In `KeyframeInterpolationPipeline(..., quantization=QuantizationPolicy.fp8_cast())`.
   - No new deps for `fp8_cast`; works on most GPUs.

2. **Switch checkpoint to FP8 in config**
   - Point `checkpoint_path` to `ltx-2-19b-dev-fp8.safetensors` (or the repo’s FP8 full checkpoint path).
   - Keeps KeyframeInterpolationPipeline’s “full + distilled LoRA” design while reducing size.

3. **Lower resolution and frames for minimal runs**
   - In `config.yaml`: e.g. `height: 320`, `width: 512`, `num_frames: 33` (or 17 for a very short clip).
   - Ensures `(height, width)` divisible by 32 and `num_frames` = 8n+1.

4. **Reduce stage-1 steps**
   - In `config.yaml`: e.g. `num_inference_steps: 24` or `20` and compare quality vs 40.

5. **(Optional) FP8 scaled MM for Hopper**
   - If you have an H100 (or supported GPU) and want maximum savings: install with `uv sync --frozen --extra fp8-trtllm`, then use `QuantizationPolicy.fp8_scaled_mm()` in the pipeline.

6. **Do not** switch KeyframeInterpolationPipeline to the distilled *checkpoint* unless the repo explicitly supports it for this pipeline; the current design is full checkpoint + distilled LoRA in stage 2.

### 2.5 Config snapshot for “minimal” test

Suggested minimal YAML slice (merge into your existing config as needed):

```yaml
model:
  checkpoint_path: "..."   # use ltx-2-19b-dev-fp8.safetensors path
  # ... rest unchanged; add quantization in run.py

inference:
  height: 320
  width: 512
  num_frames: 33
  frame_rate: 24.0
  num_inference_steps: 24
```

Plus in `run.py`: pass `quantization=QuantizationPolicy.fp8_cast()` into `KeyframeInterpolationPipeline`.

---

## 3. Summary

- **Exp 013 implementation:** Correct; only optional fix is the config comment for height/width.
- **Lightest path:** Use FP8 checkpoint + `QuantizationPolicy.fp8_cast()` in the pipeline, then lower resolution (e.g. 320×512), fewer frames (e.g. 33), and fewer steps (e.g. 24) for a minimal-VRAM, minimal-compute test. Implement in the order of the roadmap above.
