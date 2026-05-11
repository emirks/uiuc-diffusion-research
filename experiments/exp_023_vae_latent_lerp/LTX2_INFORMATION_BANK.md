# LTX-2 information bank

Compact reference from migration work (exp_016 → exp_020) and official **Diffusers** / **Lightricks** docs.  
Canonical Diffusers API: [ltx2.md](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2) · [source](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md).

---

## 1. Model & repos


| Item              | Detail                                                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| Paper             | [LTX-2 (HF papers)](https://hf.co/papers/2601.03233) — DiT foundation, **joint video + audio**.                      |
| Hub org           | [Lightricks](https://huggingface.co/Lightricks) — checkpoints, LoRA, latent upsampler.                               |
| Upstream app repo | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) (reference; **exp logic uses Diffusers**, not this package). |
| This lab          | Vendored code: `src/LTX-2/` · Diffusers: `pip install diffusers` → `diffusers.pipelines.ltx2`.                       |


---

## 2. Latent geometry (shared)


| Quantity           | Value / formula                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Temporal VAE scale | **8** (causal): `F_lat = (F_pix - 1) // 8 + 1`                                                                         |
| Spatial VAE scale  | **÷32** per H/W pixel dim                                                                                              |
| Example            | 512×768 → 16×24 lat H×W; 121 px frames → **16** lat frames                                                             |
| Packing            | Latents → packed tokens for DiT; Stage 2 needs **pixel** `height`/`width` for **mask** in condition pipeline (see §6). |


Deeper vendored-path notes: `[notes/models/ltx2/conditioning_mechanism.md](../../../notes/models/ltx2/conditioning_mechanism.md)`.

---

## 3. Vendored “core” stack (`src/LTX-2`)


| Piece                                         | Role                                                                                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `KeyframeInterpolationPipeline`               | C2V / keyframe task; branches **images** vs `**ClipConditioningInput`** clips.                                                 |
| `ClipConditioningInput`                       | `(path, frame_idx_px, strength, num_clip_frames)` — multi-frame clip → encoded clip latents → `VideoConditionByKeyframeIndex`. |
| `clip_conditionings_by_adding_guiding_latent` | Loads MP4, caps frames, VAE-encodes clip → temporal anchoring.                                                                 |
| `CHANGES.md` (vendored)                       | Documents C2V adaptations vs upstream.                                                                                         |


**Coordinate note:** vendored notes sometimes use pixel end index `N_px - K_px`; Diffusers VC uses **latent** index `N_lat - K_lat` (see §7; Stage 2 / mask sizing §9).

---

## 4. Diffusers pipelines (API)


| Class                        | Use                                                                                                                                                                                                                                                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LTX2Pipeline`               | Text-to-video; **Stage 2** with `latents=` can **infer** packed shape — **no** `height*2`/`width*2` in doc snippet.                                                                                                                                                                                                           |
| `LTX2ConditionPipeline`      | **Image/video conditions** at latent indices (`LTX2VideoCondition`). **Stage 2** must pass `**width=width*2`, `height=height*2`** — `prepare_latents` builds **conditioning mask** from pixel `height`/`width`/`num_frames`; inference of `latent_*` from 5D latents only feeds **sequence length / shift**, not mask sizing. |
| `LTX2LatentUpsamplePipeline` | **×2 spatial**, latent→latent; shares `vae` with main pipe.                                                                                                                                                                                                                                                                   |
| `LTX2VideoCondition`         | `frames` (PIL / list / video), `index` (latent index; **not** `-1` for full multi-frame end clip — trims to one lat), `strength`.                                                                                                                                                                                             |
| Utils                        | `STAGE_2_DISTILLED_SIGMA_VALUES` — fixed **3-step** Stage 2 σ schedule; `encode_video` export.                                                                                                                                                                                                                                |


**Deps:** `load_lora_weights` → `**peft`** backend · `**accelerate**` for offload · set `HF_HOME` for cache.

---

## 5. Two-stage **production** recipe (`Lightricks/LTX-2` Hub)

Doc: [Two-stages Generation](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md#two-stages-generation).


| Stage     | What                                                              | Typical args                                                                                                                                                                                |
| --------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1**     | Base transformer, **CFG**, coherent video at **base** spatial res | `sigmas=None`, `guidance_scale=4.0`, `num_inference_steps=40`, `num_frames=121`, …                                                                                                          |
| Upsampler | `LTX2LatentUpsampleModel`                                         | ×2 **spatial** in latent space                                                                                                                                                              |
| **2**     | **Refine** upsampled latents                                      | **Distilled LoRA** + scheduler `use_dynamic_shifting=False`, `**STAGE_2_DISTILLED_SIGMA_VALUES`**, `num_inference_steps=3`, `guidance_scale=1.0`, `noise_scale=σ[0]` (renoise), **LoRA on** |


**Why a separate LoRA file on `Lightricks/LTX-2` only:** full **base** checkpoint ships **without** baking Stage-2 distillation into the main weights; `**ltx-2-19b-distilled-lora-384.safetensors`** adapts the same DiT for **short** high-res refinement (detail / sharpness). `**rootonchair/LTX-2-19b-distilled`** (and similar) = **already distilled base** → doc snippets **omit** `load_lora_weights` for that extra file.

---

## 6. Doc sections combined for VC (exp_020)

- **[Condition Pipeline Generation](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md#condition-pipeline-generation):** `conditions=[…]`, offload → `**vae.enable_tiling()`**, FLF2V-style **Stage 2** kwargs (`width*2`, `height*2`, …). VC = **two video clips** as conditions (start `@0`, end `@end_idx`), not FLF2V stills.
- **Same page — Two-stages:** LoRA load, Stage-2 scheduler, `noise_scale` / σ table.
- **Memory:** docs often use `enable_sequential_cpu_offload`; we use `**enable_model_cpu_offload`** (whole submodules per `model_cpu_offload_seq`) for speed on large GPUs.

---

## 7. Video connecting (C2V) — Diffusers vs vendored


| Topic                       | Rule                                                                                                                                |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| End clip **index**          | `end_idx = N_lat - K_lat` with `N_lat=(num_frames-1)//8+1`, `K_lat=(num_clip_frames-1)//8+1`.                                       |
| `index=-1`                  | Valid for **single**-frame “last” condition; **wrong** for multi-frame end clip (collapses to one latent).                          |
| Stage 2 alignment           | Pass explicit `**num_frames`** if ever ≠ default **121** so mask matches Stage 1.                                                   |
| `negative_prompt` @ Stage 2 | `**guidance_scale>1`** needed for CFG; at `**1.0**`, `do_classifier_free_guidance` is false — negative unused; can omit on Stage 2. |


---

## 8. How conditioning is applied (Diffusers vs vendored `ltx-core`)

This section summarizes how **`LTX2ConditionPipeline`** (Diffusers) and the vendored **`KeyframeInterpolationPipeline`** / **`ltx_core.conditioning`** paths differ, and how that relates to **training** (`ltx-trainer`). Same base weights can look “fine” under both styles because the **physics** of the denoising step (clean vs noisy blend, per-token timesteps) matches; the difference is **where** condition tokens live in the sequence.

### Two injection patterns

| Pattern | Typical API | What happens |
| -------- | ----------- | ------------ |
| **In-grid** | Diffusers `LTX2VideoCondition` + `prepare_latents` / `apply_visual_conditioning`; vendored `VideoConditionByLatentIndex` + `image_conditionings_by_replacing_latent` (e.g. `TI2VidTwoStagesPipeline`) | Fixed packed grid for the full video. Condition latents **overwrite** the slice starting at `latent_idx * latent_H * latent_W` (multi-frame conditions extend `num_cond_tokens`). Sequence length unchanged. |
| **Appended “guiding” latents** | Vendored `VideoConditionByKeyframeIndex` + `image_conditionings_by_adding_guiding_latent` / `clip_conditionings_by_adding_guiding_latent` (**`KeyframeInterpolationPipeline` only** among common pipelines) | Encoded keyframes/clips are **patchified and concatenated after** the main video tokens. **`VideoLatentTools.clear_conditioning`** then **truncates** back to the target grid before decode. Positions use **pixel-frame** `frame_idx` (time axis shifted, then ÷ `fps` for RoPE), not a raw latent index. |

**Index semantics:** Diffusers `LTX2VideoCondition.index` is a **latent temporal index** (with negative wrap, e.g. `-1` → last latent frame). Vendored `ImageConditioningInput.frame_idx` / `ClipConditioningInput.frame_idx` are **pixel-space** offsets for aligning clip latents with the output timeline (see upstream `ltx-pipelines` README: replacing vs guiding latents).

### Mask naming (same math)

- **Diffusers** uses a **conditioning mask** in `[0,1]`: strength 1 → fully use **clean** condition in the post-step blend; `video_timestep = t * (1 - mask)` so conditioned tokens see ~0 diffusion time.
- **`ltx-core`** uses **`denoise_mask`**: `denoise_mask = 1 - strength` on conditioned tokens; `post_process_latent` blends so **`denoise_mask = 0`** means “fully clean”. The Diffusers pipeline comment states explicitly: *denoise_mask = 1 − conditioning_mask*.

So the implementations are **not** contradictory—only the **names** are inverses.

### Training alignment (public `ltx-trainer`)

- **`TextToVideoStrategy`:** single sequence of video tokens; with probability `first_frame_conditioning_p`, the **first frame’s** tokens (`: H_lat×W_lat`) are kept clean (timestep 0). That matches **in-grid** conditioning on frame 0, not appended tokens.
- **`VideoToVideoStrategy` (IC-LoRA):** **reference ∥ target** concatenation with loss on target; matches **`VideoConditionByReferenceLatent`** (append reference, then strip)—different task than FLF2V/C2V keyframes.
- **Appended guiding keyframes** are **not** spelled out in those YAML strategies; they are first-class in **`ltx-core`** and **`KeyframeInterpolationPipeline`**, so treat them as supported for inference (likely covered by internal pretraining / robustness), while **in-grid** is the path that matches the published T2V fine-tuning recipe most directly.

### Why both can still work

The DiT uses **3D RoPE** and **per-token timesteps**: conditioned tokens are identified by **position + timestep≈0**, not only by buffer index. Appended guiding tokens still carry correct spatiotemporal coordinates; self-attention is not causal, so extra trailing tokens can still **influence** the main grid before truncation. Large models also tolerate mild train/serve mismatch here.

**Practical takeaway for exp_020:** Diffusers VC is **in-grid** at **latent** indices; vendored C2V uses **appended** clips with **pixel** `frame_idx`. Do not assume the same integer means the same thing across stacks.

---

## 9. Pitfalls & fixes (session log)


| Issue                                                        | Fix                                                                                                                    |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `PEFT backend is required`                                   | `pip install peft`                                                                                                     |
| Shape mismatch Stage 2 (e.g. `6144` vs `~20k` packed tokens) | Pass `**width*2`, `height*2**`; align `**num_frames**` with Stage 1 / defaults                                         |
| Stage 1 OK, Stage 2 wrong schedule                           | Restore `**pipe.scheduler**` to Stage-1 scheduler each sample; `**disable_lora**` / `**enable_lora**` per stage        |
| Low VRAM + slow steps                                        | `enable_sequential_cpu_offload` shards layers; `**enable_model_cpu_offload**` keeps one **component** on GPU at a time |


---

## 10. exp_020 snapshot

- **Config:** `Lightricks/LTX-2`, DAVIS MP4 pairs (same subset as exp_016 `config_davis.yaml`), `num_frames=121`, 512×768 → 1024×1536 after upsampler+Stage 2, LoRA strength **1.0** (doc default), artifacts under `outputs/…/exp_020_ltx2_c2v_diffusers/`.
- **Entry:** `python experiments/exp_020_ltx2_c2v_diffusers/run.py` (conda env with diffusers+accelerate+peft).

---

## 11. Further reading (repo)

- `[notes/models/ltx2/README.md](../../../notes/models/ltx2/README.md)` — index of LTX-2 notes (includes this bank under `information_bank.md`)  
- `[notes/models/ltx2/conditioning_mechanism.md](../../../notes/models/ltx2/conditioning_mechanism.md)` — vendored patchifier, masks, RoPE, exp_014/015  
- `[src/LTX-2/packages/ltx-pipelines/README.md](../../../src/LTX-2/packages/ltx-pipelines/README.md)` — upstream pipeline CLI docs

