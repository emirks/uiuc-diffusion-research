# LTX-2 Conditioning Mechanism

> Notes from experiments exp_014 (keyframe interpolation) and exp_015 (clip-to-video C2V).
> Code lives in `src/LTX-2/packages/ltx-core` and `src/LTX-2/packages/ltx-pipelines`.

---

## 1. The Two Spaces: Pixel and Latent

The LTX-2 video VAE compresses video in all three spatiotemporal dimensions:

```
Pixel space:  [B, 3,    F_pix, H_pix, W_pix]
                         ×8     ×32    ×32
Latent space: [B, 128, F_lat, H_lat, W_lat]
```

Scale factors are defined in `ltx_core.types.SpatioTemporalScaleFactors.default()`:

```python
SpatioTemporalScaleFactors(time=8, height=32, width=32)
```

**Temporal formula** (causal VAE — first frame is stride-1, not stride-8):

```
F_lat = (F_pix - 1) // 8 + 1
```


| Pixel frames      | Latent frames | Example            |
| ----------------- | ------------- | ------------------ |
| 1 (single image)  | 1             |                    |
| 24 (1s at 24fps)  | 3             | `(24-1)//8+1 = 3`  |
| 97 (~4s at 24fps) | 13            | `(97-1)//8+1 = 13` |


**Spatial:** a 512×768 pixel frame → 16×24 latent grid.

**Latent channels:** 128 (compressed color+texture information).

---

## 2. Patchification: Latent Volume → 1D Token Sequence

The transformer does not operate on a 5D tensor. The patchifier flattens the latent into a 1D sequence of tokens via `VideoLatentPatchifier`.

**Code** (`ltx_core/components/patchifiers.py`):

```python
"b c (f p1) (h p2) (w p3)  →  b (f·h·w) (c·p1·p2·p3)"
```

- Temporal patch size `p1 = 1` (no temporal grouping — every latent frame is its own token group)
- Spatial patch size `p2 = p3 = P` (typically P=2, grouping 2×2 latent cells into one token)

**Token count:**

```
N_tokens = F_lat × (H_lat / P) × (W_lat / P)
```

For a 97-frame, 512×768 video (P=2):

- `F_lat=13`, `H_lat=16`, `W_lat=24`
- `N = 13 × 8 × 12 = 1248 tokens`

Each token has dimension `D = 128 × P² = 512`.

---

## 3. Position Assignment: Where Is Each Token in Pixel Space?

Every token carries a **3D bounding box in pixel space**.

**Shape:** `(B, 3, N_tokens, 2)` where:

- Axis 1 = three spatial axes: `(time, height, width)`
- Last dim = `[start, end)` — a range, not a single point

**Why a range?** One latent token represents an entire region of the original video — e.g., 8 pixel frames × 64×64 pixels. The range captures the full extent of that "brick" of video.

**Why pixel space?** Both output tokens and conditioning tokens must share the **same coordinate system** for position embeddings to correctly measure relative distance.

### From Latent Coords to Pixel Coords

**Step 1:** `get_patch_grid_bounds` → integer latent-space bounds `[start, end)` for each token.
For a 3-frame latent: temporal bounds are `[0,1], [1,2], [2,3]`.

**Step 2:** `get_pixel_coords` → multiply by scale factors:

```python
pixel_coords = latent_coords * scale_tensor   # time×8, h×32, w×32
```

Result: `[0,8], [8,16], [16,24]` in pixel-frames.

**Step 3 (causal_fix = True, applied only to output video and start-of-clip cond):**

```python
pixel_coords[:, 0, :] = (pixel_coords[:, 0, :] + 1 - 8).clamp(0)
# i.e. shift ALL temporal coords by -(8-1) = -7
```

**Why the causal fix?** The video VAE is causal: the first latent frame has temporal stride 1 (covers only pixel frame 0), not stride 8. Without correction, the RoPE would think token 0 covers frames `[0,8]` when it actually covers `[0,1]`. The fix shifts all temporal positions by −7 and clamps to 0.

**After causal_fix**, output video token temporal positions for a 97-frame video:


| Latent token | Pixel range (after causal_fix) |
| ------------ | ------------------------------ |
| 0            | [0, 1]                         |
| 1            | [1, 9]                         |
| 2            | [9, 17]                        |
| 3            | [17, 25]                       |
| …            | …                              |
| 10           | [73, 81]                       |
| 11           | [81, 89]                       |
| 12           | [89, 97]                       |


**Step 4:** Temporal dimension is divided by FPS to convert from pixel-frames to seconds:

```python
positions[:, 0, ...] /= fps    # e.g. fps=24
```

Token 12: `[89/24, 97/24] = [3.71s, 4.04s]`.

---

## 4. The LatentState: Everything in One Bundle

After encoding and conditioning, everything lives in a `LatentState` struct
(`ltx_core/types.py`):

```python
@dataclass(frozen=True)
class LatentState:
    latent:        Tensor   # [B, N_total, D]      — noisy+clean token values
    denoise_mask:  Tensor   # [B, N_total, 1]      — 1.0=noisy, 0.0=clean
    positions:     Tensor   # [B, 3, N_total, 2]   — pixel-space bounding boxes
    clean_latent:  Tensor   # [B, N_total, D]      — reference copy before noising
    attention_mask: Tensor | None  # [B, N_total, N_total]
```

For C2V (97-frame output, 2 clips of 24 frames each):

```
.latent        [B, 1248 + 288 + 288, D]
                    output    start   end
                    (noisy)  (clean) (clean)

.denoise_mask  [B, 1824, 1]
                [1.0 × 1248] [0.0 × 288] [0.0 × 288]

.positions     [B, 3, 1824, 2]   — pixel bounding boxes for all tokens

.attention_mask [B, 1824, 1824]  — block structure (see §6)
```

---

## 5. The Three Control Mechanisms

### 5.1 Positions → RoPE (Rotary Position Embeddings)

**Purpose:** Tell the transformer WHERE in (t, y, x) space each token lives.

**Flow:**

```
positions (t_sec, y_px, x_px)
    ↓ normalize by max_pos = [20s, 2048px, 2048px]
fractional positions (t/20, y/2048, x/2048) ∈ [0, ~1]
    ↓ precompute_freqs_cis
3D rotation angles (cos, sin) per token per head
    ↓ apply_rotary_emb
Q and K vectors are rotated before dot-product attention
```

**Key property:** Two tokens at the **same (t, y, x) position** get the same rotation → their Q·K dot product is maximized → they attend to each other strongly. Two tokens far apart in space/time get different rotations → weaker attention.

**Why this matters for conditioning:** When a conditioning token is placed at the exact same temporal range as an output token (via `frame_idx`), they share the same temporal RoPE rotation → the conditioning token exerts maximum positional influence on the output token at that location.

**Code:** `ltx_core/model/transformer/rope.py` → `precompute_freqs_cis`

### 5.1-a Bounding Box → Single Position: The Midpoint

Each token carries a `[start, end)` range per axis — but RoPE needs a single number per axis.
The bridge is `use_middle_indices_grid=True` (default in `LTXModel`), inside `generate_freqs` in `rope.py`:

```python
if use_middle_indices_grid:
    indices_grid = (indices_grid_start + indices_grid_end) / 2.0
```

The bounding box is collapsed to its **midpoint**. That midpoint is then normalized:

```
fractional_pos = midpoint / max_pos[axis]
```

So the "position" of a token is the **center of its pixel-space brick**, normalized to `[0, 1]`.

**Concrete example — 17-pixel-frame video (temporal scale = 4):**


| Latent frame | Pixel range (causal_fix) | Midpoint | Fractional (max_pos=20s, fps=25) |
| ------------ | ------------------------ | -------- | -------------------------------- |
| 0            | [0, 1)                   | 0.5      | 0.5 / 25 / 20 = 0.001            |
| 1            | [1, 5)                   | 3.0      | …                                |
| 2            | [5, 9)                   | 7.0      | …                                |
| 3            | [9, 13)                  | 11.0     | …                                |
| 4            | [13, 17)                 | 15.0     | …                                |


### 5.1-b Image Conditioning vs. Clip Conditioning: Temporal Midpoint Distribution

**Single keyframe at `frame_idx=T`** (e.g. T=0):

- Always 1 latent frame → pixel range `[0,1)` → midpoint `0.5`
- After `+ frame_idx`: midpoint = `T + 0.5`
- All N spatial tokens of this keyframe share the **same temporal midpoint** `T + 0.5`
- They differ only in `(h, w)` midpoints

**Video clip (F_pix frames → F_lat latent frames)** at `frame_idx=T`:

- F_lat latent frames each have their own temporal midpoints, spread across the clip duration
- E.g. 17-frame clip → midpoints `{0.5, 3, 7, 11, 15}` + T
- Different latent frames' tokens have **different temporal midpoints** covering the full clip

**Effect in attention:**

- A conditioning token attends most strongly to the noisy token whose midpoint is closest to its own.
- For a single keyframe: all spatial tokens point to the same moment T → they "pin" the output at time T.
- For a clip: temporal midpoints are distributed across the clip duration → each latent frame pins a different temporal region of the output → richer, motion-aware conditioning.

**No training mismatch:** the clip conditioning tokens' temporal midpoints are identical to the noisy output tokens' temporal midpoints for the same latent frames (both computed the same way with the same causal scaling). So the model sees what it was trained on.

From chat: image vs. clip conditioning via bounding-box midpoints (2026-03-12).

---

### 5.2 Denoise Mask (Controls Noising and Timestep)

**Purpose:** Say which tokens are noisy (output) and which are clean (conditioning).

**Value:** `1.0 = noisy, 0.0 = clean` (a per-token scalar).

For conditioning with `strength=1.0`:

```python
denoise_mask = 1.0 - strength = 0.0
```

**Use 1 — Noising initialization** (`GaussianNoiser`):

```python
noise = randn(shape)
latent = noise * denoise_mask  +  latent * (1 - denoise_mask)
```

- Output tokens (mask=1): replaced entirely with random noise
- Conditioning tokens (mask=0): kept exactly as the clean VAE encoding

**Use 2 — Per-token timestep** (during each denoising step):

```python
timesteps = denoise_mask * current_sigma
```

- Output tokens: timestep = sigma → full denoising signal
- Conditioning tokens: timestep = 0 → "already at noise level 0"

**Use 3 — The X0 formula:**

```python
denoised = latent - velocity * timesteps   # per-token sigma
```

For conditioning tokens: `timesteps=0` → `denoised = latent` → **no update, frozen**.

**Code:** `ltx_core/components/noisers.py`, `ltx_core/utils.py` (`to_denoised`)

---

### 5.3 Attention Mask (Controls Who Can Talk to Whom)

**Purpose:** Prevent different conditioning groups from cross-contaminating each other in self-attention.

**Shape:** `[B, N_total, N_total]`, values in `[0, 1]`.

- `1.0` → full attention allowed
- `0.0` → completely blocked

**Block structure for C2V (2 conditioning groups):**

```
                Output (N)   Cond-Start (M)   Cond-End (M)
Output  (N)  [    1.0     |      1.0       |     1.0     ]
Cond-Start   [    1.0     |      1.0       |     0.0     ]
Cond-End     [    1.0     |      0.0       |     1.0     ]
```

- Output tokens can see everyone (they need information from all conditioning)
- Cond-Start tokens can see output tokens + themselves, but NOT Cond-End
- Cond-End tokens can see output tokens + themselves, but NOT Cond-Start

**Why block cross-conditioning attention?** If Cond-Start could attend to Cond-End, their information would mix. The model was not trained for mixed-conditioning self-attention between separate groups.

**Under the hood:** Converted to additive log-space bias before softmax:

```python
1.0 → log(1.0) = 0.0           # neutral, no effect
0.0 → -float_max               # fully masked
```

Applied as `(B, 1, T, T)` bias added to Q·K scores.

**Code:** `ltx_core/conditioning/mask_utils.py` (`build_attention_mask`)

---

## 6. VideoConditionByKeyframeIndex: The Core Conditioning Operation

**File:** `ltx_core/conditioning/types/keyframe_cond.py`

This is the class that takes an encoded clip/image latent and **appends** it to the existing `LatentState`. It is called once per conditioning item (once for start clip, once for end clip).

```python
class VideoConditionByKeyframeIndex(ConditioningItem):
    keyframes: Tensor   # [B, C, F, H, W] — already VAE-encoded latent
    frame_idx: int      # pixel-frame offset of the clip's first frame in output space
    strength: float     # 1.0 = fully clean, 0.0 = fully noisy
```

### What `apply_to` does, step by step:

```python
# 1. Patchify the conditioning latent into tokens
tokens = patchifier.patchify(self.keyframes)          # [B, M, D]

# 2. Compute local latent-space bounding boxes
latent_coords = patchifier.get_patch_grid_bounds(...)  # [B, 3, M, 2]

# 3. Convert to pixel space + causal_fix if frame_idx==0
positions = get_pixel_coords(latent_coords, scale_factors, causal_fix=...)

# 4. Shift temporal axis by frame_idx (place clip in output timeline)
positions[:, 0, ...] += self.frame_idx

# 5. Convert temporal from pixel-frames to seconds
positions[:, 0, ...] /= fps

# 6. Build denoise_mask (1.0 - strength, per token)
denoise_mask = full(fill_value=1.0 - strength)          # 0.0 if strength=1

# 7. Update attention_mask block structure
new_attention_mask = update_attention_mask(latent_state, ...)

# 8. Append everything to existing LatentState
return LatentState(
    latent        = cat([state.latent, tokens],        dim=1),
    denoise_mask  = cat([state.denoise_mask, denoise_mask], dim=1),
    positions     = cat([state.positions, positions],  dim=2),
    clean_latent  = cat([state.clean_latent, tokens],  dim=1),
    attention_mask = new_attention_mask,
)
```

After `apply_to`, the sequence is `N_output + M` tokens. After both clips, it is `N_output + M_start + M_end`.

**After the denoising loop:**

```python
video_state = video_tools.clear_conditioning(video_state)
# Drops tokens beyond the first N_output — removes all cond tokens
video_state = video_tools.unpatchify(video_state)
# Reshapes [B, N_output, D] → [B, 128, F_lat, H_lat, W_lat]
```

---

## 7. The Two-Stage Pipeline (`KeyframeInterpolationPipeline`)

**File:** `ltx_pipelines/keyframe_interpolation.py`

**Stage 1 — Low resolution generation (half-res: 256×384):**

- Loads the video encoder from `stage_1_model_ledger`
- Builds conditionings at `height//2 × width//2` → clips encoded at lower res → fewer latent cells (8×12 → 4×6 at half-res), so fewer spatial tokens per clip
- Denoises with full transformer + CFG (MultiModalGuider)
- Sigmas: full LTX2Scheduler schedule (40 steps)

**Stage 2 — Upsample + refinement (full-res: 512×768):**

- Upsamples stage-1 latent via `spatial_upsampler` (×2 spatial)
- Builds conditionings AGAIN at full resolution `height × width` → clips re-encoded at full res
- Denoises with distilled LoRA transformer (fewer steps, distilled sigmas)
- Starting point: `initial_video_latent = upscaled_stage1_latent`

**Why re-encode conditioning for each stage?** The conditioning clips must be at the correct spatial resolution for the stage. A clip at 256×384 latent has `4×6=24` spatial patches per temporal token, while at 512×768 it has `8×12=96`. If you used stage-1 conditionings in stage-2, the spatial positions would be wrong (each token would map to the wrong pixel region).

---

## 8. Frame Index Formula: Temporal Alignment

**The key formula** for placing conditioning clips at the right position in the output:

```
frame_idx_start = 0
frame_idx_end   = num_output_frames - num_clip_frames
```

For `num_output_frames=97`, `num_clip_frames=24`: `frame_idx_end = 73`.

**Why this works** (derivation from first principles):

Without causal_fix, clip token local temporal ranges are `[0,8], [8,16], [16,24]`.
With `frame_idx=73` added: `[73,81], [81,89], [89,97]`.

Output tokens 10, 11, 12 have temporal ranges (after causal_fix):

- Token 10: `[80,88] - 7 = [73,81]` ✓
- Token 11: `[88,96] - 7 = [81,89]` ✓
- Token 12: `[96,104] - 7 = [89,97]` ✓

**Single-frame special case:** With `num_clip_frames=1`: `frame_idx_end = 97-1 = 96`. That's exactly what exp_014 uses for the last keyframe.

**Causal_fix rule:**

- `frame_idx == 0` → `causal_fix = True` (start clip, first frame of output)
- `frame_idx != 0` → `causal_fix = False` (end clip, interior clip)

The end clip does NOT get the causal fix because it doesn't start at the beginning of the video; the causal stride-1 correction only applies at the very start of the sequence.

---

## 9. Exp-014: Keyframe Interpolation (Single Frames)

**File:** `experiments/exp_014_ltx2_keyframe_interpolation_bf16/run.py`

**Conditioning:** Two single images — first frame and last frame of target video.

```python
images = [
    ImageConditioningInput(first_path, frame_idx=0,           strength=1.0),
    ImageConditioningInput(last_path,  frame_idx=num_frames-1, strength=1.0),
]
```

**Token counts (97-frame, 512×768):**

- Output: 1248 tokens
- First frame cond: 96 tokens (1 × 8 × 12)
- Last frame cond: 96 tokens
- Total: 1440 tokens

**Position of last frame cond:** `frame_idx=96`, local range `[0,8]` → `[96,104]` in pixel space. Output's last token range `[89,97]`. They overlap in the same temporal neighborhood (RoPE rotation is continuous, not discrete — close positions → strong attention).

**Pipeline:** `KeyframeInterpolationPipeline`, no quantization (full bf16), 40 inference steps.

---

## 10. Exp-015: Clip-to-Video / C2V (Multi-Frame Clips)

**File:** `experiments/exp_015_ltx2_c2v/run.py`

**Conditioning:** Two 24-frame clips encoded as temporal sequences.

```python
clips = [
    ClipConditioningInput(start_clip_path, frame_idx=0,              strength=1.0, num_clip_frames=24),
    ClipConditioningInput(end_clip_path,   frame_idx=num_frames-24,  strength=1.0, num_clip_frames=24),
]
```

`frame_idx_end = 97 - 24 = 73`.

**Token counts (97-frame, 512×768):**

- Output: 1248 tokens
- Start clip cond: 288 tokens (3 × 8 × 12)
- End clip cond: 288 tokens
- Total: 1824 tokens

**Key difference from exp_014:** Each clip is encoded in a **single forward pass** through the video VAE. The causal temporal convolutions see all 24 frames together — later clip frames carry context from earlier ones. This produces richer, motion-aware latent tokens compared to encoding each frame individually.

**Code additions to LTX-2 source (all additive, no breaking changes):**


| File                                      | Change                                                                                                                                |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `ltx_pipelines/utils/args.py`             | Added `ClipConditioningInput(path, frame_idx, strength, num_clip_frames)`                                                             |
| `ltx_pipelines/utils/helpers.py`          | Added `clip_conditionings_by_adding_guiding_latent()`                                                                                 |
| `ltx_pipelines/keyframe_interpolation.py` | Added `clips: list[ClipConditioningInput] | None` to `__call_`_; uses clip helper when provided, falls back to image helper otherwise |


---

## 11. Option Comparison: How to Do C2V


|                      | Option A (individual frames)          | Option C (clip encoding)                  |
| -------------------- | ------------------------------------- | ----------------------------------------- |
| LTX source changes   | 0                                     | +73 lines, 3 files                        |
| Temporal VAE context | ✗ each frame encoded alone            | ✓ full clip in one pass                   |
| Token count          | K × 96 (one per frame)                | 3 × 96 (one per latent time step)         |
| Position alignment   | ✓ each frame at its own frame_idx     | ✓ clip tokens at latent-aligned positions |
| Frame selection      | must pick which K frames to use       | encodes all K frames naturally            |
| Quality              | degrades: VAE has no temporal context | correct behavior                          |


**Option B (subclass / code duplication):** duplicates ~130 lines of pipeline code in the experiment folder; no LTX changes but fragile (diverges if pipeline updates).

**Recommendation:** Option C (current state in exp_015) is best. The additions are genuinely useful library-level abstractions.

---

## 12. C2V Task: Limitations and Future Improvements

### Current limitation

Clip conditioning provides 3 latent tokens per clip covering a ~17-pixel-frame temporal window each. The **exact last pixel frame of the end clip** is "diluted" into the last latent token (which encodes clip frames 16–23 together). The model has no single-pixel-level pin on the exact boundary frame.

### Option D — Clip + Boundary Frame (recommended next experiment = exp_016)

Combine clip conditioning (motion context) with single-frame conditioning (pixel-exact boundary):

```python
clips = [
    ClipConditioningInput(start_clip, frame_idx=0,  strength=1.0, num_clip_frames=24),
    ClipConditioningInput(end_clip,   frame_idx=73, strength=1.0, num_clip_frames=24),
]
images = [
    ImageConditioningInput(first_frame_of_start_clip, frame_idx=0,           strength=1.0),
    ImageConditioningInput(last_frame_of_end_clip,    frame_idx=num_frames-1, strength=1.0),
]
# pass both clips and images, pipeline merges: stage_conds = clip_conds + image_conds
```

This adds 2 × 96 = 192 extra tokens (one per boundary frame), giving both temporal motion context AND pixel-precise boundary anchoring.

**Implementation:** Change the pipeline `if clips / else images` logic to:

```python
stage_conditionings = (
    clip_conditionings_by_adding_guiding_latent(clips, ...) if clips else []
) + (
    image_conditionings_by_adding_guiding_latent(images, ...) if images else []
)
```

### Option F — Reference Video Scaffold (longer term)

1. Construct a scaffold video: start clip frames + neutral/black middle + end clip frames
2. Encode the entire scaffold as a full-length latent
3. Condition with a per-region strength mask: 1.0 at clip positions, 0.0 in the middle
4. Use `VideoConditionByReferenceLatent` for full-resolution temporal context across both clips

This preserves temporal continuity between both clips (they're in the same VAE forward pass) but requires building the scaffold and custom masking logic.

---

## 13. Summary Table: Numbers for 97-Frame, 512×768, K=24


| Quantity                       | Formula                         | Value          |
| ------------------------------ | ------------------------------- | -------------- |
| Temporal scale                 | constant                        | 8              |
| Spatial scale                  | constant                        | 32 × 32        |
| Output latent shape            | `(97-1)//8+1, 512//32, 768//32` | `13 × 16 × 24` |
| Output token count (P=2)       | `13 × 8 × 12`                   | 1248           |
| Clip latent frames             | `(24-1)//8+1`                   | 3              |
| Clip token count (P=2)         | `3 × 8 × 12`                    | 288            |
| End clip frame_idx             | `97 - 24`                       | 73             |
| Total tokens (C2V, exp_015)    | `1248 + 288 + 288`              | 1824           |
| Stage 1 spatial                | `256×384`                       | half-res       |
| Stage 1 clip token count (P=2) | `3 × 4 × 6`                     | 72             |
| Stage 2 = full res             | `512×768`                       | full-res       |
| max_pos for RoPE               | `[20s, 2048px, 2048px]`         | constant       |
| causal_fix shift               | `1 - time_scale = -7`           | constant       |


---

## 14. Mental Model (Simple)

The transformer is a "fill-in-the-blank" machine:

- **Output tokens** = blank slots filled with random noise. The model will fill them in.
- **Conditioning tokens** = reference answers. Frozen. The model uses them as hints.
- **Positions (RoPE)** = seating chart. Conditioning tokens at the same (t, y, x) as output tokens sit right next to them → they whisper most effectively.
- **Denoise mask** = name tag. "Fix me" (1.0) tags get noised and denoised. "Already done" (0.0) tags are untouched through the entire process.
- **Attention mask** = room divider. Different conditioning groups (start clip, end clip) can talk to the output tokens but not to each other.

---

## 15. Key Source Files Reference


| Concept                                                          | File                                                        |
| ---------------------------------------------------------------- | ----------------------------------------------------------- |
| Scale factors, LatentState, VideoLatentShape                     | `ltx_core/types.py`                                         |
| Patchifier, `get_pixel_coords`                                   | `ltx_core/components/patchifiers.py`                        |
| GaussianNoiser                                                   | `ltx_core/components/noisers.py`                            |
| `VideoConditionByKeyframeIndex`                                  | `ltx_core/conditioning/types/keyframe_cond.py`              |
| `VideoConditionByLatentIndex`                                    | `ltx_core/conditioning/types/latent_cond.py`                |
| Attention mask builder                                           | `ltx_core/conditioning/mask_utils.py`                       |
| `VideoLatentTools`, `create_initial_state`, `clear_conditioning` | `ltx_core/tools.py`                                         |
| RoPE, `precompute_freqs_cis`                                     | `ltx_core/model/transformer/rope.py`                        |
| `TransformerArgsPreprocessor`, position embedding prep           | `ltx_core/model/transformer/transformer_args.py`            |
| X0Model, `to_denoised`                                           | `ltx_core/model/transformer/model.py`, `ltx_core/utils.py`  |
| `image_conditionings_by_adding_guiding_latent`                   | `ltx_pipelines/utils/helpers.py`                            |
| `clip_conditionings_by_adding_guiding_latent`                    | `ltx_pipelines/utils/helpers.py`                            |
| `load_video_conditioning`, `load_image_conditioning`             | `ltx_pipelines/utils/media_io.py`                           |
| `ImageConditioningInput`, `ClipConditioningInput`                | `ltx_pipelines/utils/args.py`                               |
| `KeyframeInterpolationPipeline`                                  | `ltx_pipelines/keyframe_interpolation.py`                   |
| Full diagram (keyframe)                                          | `ltx_pipelines/docs/KEYFRAME_INTERPOLATION_TASK_DIAGRAM.md` |


