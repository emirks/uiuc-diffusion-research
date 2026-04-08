# LTX-2: Pipeline API & Practical Reference

Dense quick-reference built from exp_016 → exp_020 and the official Diffusers / Lightricks docs.  
Canonical Diffusers API: [ltx2 docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2)

---

## 1. Model & repos

| Item | Detail |
| ---- | ------ |
| Paper | [LTX-2 (HF papers)](https://hf.co/papers/2601.03233) — DiT foundation, **joint video + audio**. |
| Hub org | [Lightricks](https://huggingface.co/Lightricks) — checkpoints, LoRA, latent upsampler. |
| Upstream app repo | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) (reference; **exp logic uses Diffusers**, not this package). |
| This lab | Vendored code: `src/LTX-2/` · Diffusers: `pip install diffusers` → `diffusers.pipelines.ltx2`. |

---

## 2. Latent geometry

| Quantity | Value / formula |
| -------- | --------------- |
| Temporal VAE scale | **8** (causal): `F_lat = (F_pix - 1) // 8 + 1` |
| Spatial VAE scale | **÷32** per H/W dim |
| Example | 512×768 → 16×24 lat H×W; 121 px frames → **16** lat frames |
| Packing | Latents → packed tokens `[B, N, C]` for DiT; Stage 2 needs **pixel** `height`/`width` for the conditioning mask. |

See `conditioning.md` for the full patchification and token layout.

---

## 3. Vendored "core" stack (`src/LTX-2`)

| Piece | Role |
| ----- | ---- |
| `KeyframeInterpolationPipeline` | C2V / keyframe task; branches **images** vs `ClipConditioningInput` clips. |
| `ClipConditioningInput` | `(path, frame_idx_px, strength, num_clip_frames)` — multi-frame clip → encoded clip latents → `VideoConditionByKeyframeIndex`. |
| `clip_conditionings_by_adding_guiding_latent` | Loads MP4, caps frames, VAE-encodes clip → temporal anchoring. |
| `CHANGES.md` (vendored) | Documents C2V adaptations vs upstream. |

**Coordinate note:** vendored code sometimes uses pixel end index `N_px - K_px`; Diffusers VC uses **latent** index `N_lat - K_lat` (see §6 below).

---

## 4. Diffusers pipelines (API)

| Class | Use |
| ----- | --- |
| `LTX2Pipeline` | Text-to-video; Stage 2 with `latents=` can **infer** packed shape — **no** `height*2`/`width*2` in doc snippet. |
| `LTX2ConditionPipeline` | **Image/video conditions** at latent indices (`LTX2VideoCondition`). Stage 2 must pass `width=width*2`, `height=height*2` — `prepare_latents` builds the conditioning mask from pixel `height`/`width`/`num_frames`. |
| `LTX2LatentUpsamplePipeline` | **×2 spatial**, latent→latent; shares `vae` with main pipe. |
| `LTX2VideoCondition` | `frames` (PIL / list / video), `index` (latent index; **not** `-1` for full multi-frame end clip — trims to one lat), `strength`. |
| Utils | `STAGE_2_DISTILLED_SIGMA_VALUES` — fixed **3-step** Stage 2 σ schedule; `encode_video` export. |

**Deps:** `load_lora_weights` → `peft` backend · `accelerate` for offload · set `HF_HOME` for cache.

---

## 5. Two-stage production recipe (`Lightricks/LTX-2` Hub)

Doc: [Two-stages Generation](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md#two-stages-generation).

| Stage | What | Typical args |
| ----- | ---- | ------------ |
| **1** | Base transformer, CFG, coherent video at base spatial res | `sigmas=None`, `guidance_scale=4.0`, `num_inference_steps=40`, `num_frames=121` |
| Upsampler | `LTX2LatentUpsampleModel` | ×2 spatial in latent space |
| **2** | Refine upsampled latents | Distilled LoRA + `use_dynamic_shifting=False`, `STAGE_2_DISTILLED_SIGMA_VALUES`, `num_inference_steps=3`, `guidance_scale=1.0`, `noise_scale=σ[0]`, **LoRA on** |

**Why the separate LoRA:** the full base checkpoint ships without baking Stage-2 distillation; `ltx-2-19b-distilled-lora-384.safetensors` adapts the same DiT for short high-res refinement. Repos like `rootonchair/LTX-2-19b-distilled` bake it in — their doc snippets omit `load_lora_weights`.

---

## 6. Video connecting (C2V) — Diffusers vs vendored

| Topic | Rule |
| ----- | ---- |
| End clip **index** | `end_idx = N_lat - K_lat` with `N_lat=(num_frames-1)//8+1`, `K_lat=(num_clip_frames-1)//8+1`. |
| `index=-1` | Valid for **single**-frame "last" condition; **wrong** for multi-frame end clip (collapses to one latent). |
| Stage 2 alignment | Pass explicit `num_frames` if ever ≠ default **121** so mask matches Stage 1. |
| `negative_prompt` @ Stage 2 | `guidance_scale>1` needed for CFG; at `1.0`, `do_classifier_free_guidance` is false — negative unused; can omit. |

---

## 7. Conditioning patterns (quick summary)

| Pattern | API | What happens |
| ------- | --- | ------------ |
| **In-grid** | Diffusers `LTX2VideoCondition` + `prepare_latents` | Condition latents **overwrite** the packed grid at `latent_idx * H_lat * W_lat`. Sequence length unchanged. |
| **Appended guiding latents** | Vendored `VideoConditionByKeyframeIndex` + `clip_conditionings_by_adding_guiding_latent` | Encoded clips **concatenated after** the main tokens. `VideoLatentTools.clear_conditioning` truncates before decode. Positions use **pixel-frame** `frame_idx`. |

**Index semantics:** Diffusers `LTX2VideoCondition.index` is a **latent temporal index**. Vendored `ClipConditioningInput.frame_idx` is a **pixel-space** offset. Do not assume the same integer means the same thing across stacks.

**Mask naming (inverted between stacks):**
- Diffusers: `conditioning_mask = 1` → fully clean; `video_timestep = t × (1 − mask)` so conditioned tokens see ~0 diffusion time.
- `ltx-core`: `denoise_mask = 1 − conditioning_mask` → `0` = fully clean.

Full deep-dive: `conditioning.md`.

---

## 8. Pitfalls & fixes

| Issue | Fix |
| ----- | --- |
| `PEFT backend is required` | `pip install peft` |
| Shape mismatch Stage 2 (e.g. `6144` vs `~20k` packed tokens) | Pass `width*2`, `height*2`; align `num_frames` with Stage 1 / defaults |
| Stage 1 OK, Stage 2 wrong schedule | Restore `pipe.scheduler` to Stage-1 scheduler each sample; call `disable_lora` / `enable_lora` per stage |
| Low VRAM + slow steps | `enable_sequential_cpu_offload` shards layers; `enable_model_cpu_offload` keeps one component on GPU at a time |
| Packed latent shape error in trajectory logging | `[B, N, C]` from scheduler — unpack to `[C, F', H', W']` using `F'×H'×W' = N`; pass `lat_shape=(F',H',W')` to logger `reset()` |
