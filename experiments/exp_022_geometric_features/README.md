# exp_022 — Geometric Feature Extraction

Reads Stage-1 trajectory `.pt` files saved by **exp_021** and computes geometric
features on the latent trajectory to characterise the **dissolve artefact** in
LTX-2 VC generation.

---

## Latent frame → pixel frame mapping

LTX-2 uses a **causal** temporal VAE with scale 8.  
The first latent frame directly encodes pixel 0 (one-to-one).  
Every subsequent latent encodes a **block of 8 pixel frames**.

```
Latent p  │  Pixel frames (0-indexed)  │  Formula
──────────┼────────────────────────────┼──────────────────────────────
   p = 0  │  0                         │  single anchor frame
   p ≥ 1  │  (p−1)×8+1  …  p×8        │  8 pixels per latent
```

Inverse (pixel → latent):  `p = 0` if `px = 0`, else `p = (px − 1) / 8 + 1`  
Latent count for T pixel frames:  `L = (T − 1) // 8 + 1`

At 24 fps, 121 pixel frames → **16 latent frames** ≈ **5 seconds**.

| Latent `p` | Pixel frames (0-idx) | Time range @ 24 fps | Region |
|:----------:|:--------------------:|:-------------------:|:------:|
| 0  | 0        | 0.00 s              | ← start conditioning |
| 1  | 1 – 8    | 0.04 – 0.33 s       | ← start conditioning |
| 2  | 9 – 16   | 0.38 – 0.67 s       | ← start conditioning |
| 3  | 17 – 24  | 0.71 – 1.00 s       | ← start conditioning (last) |
| 4  | 25 – 32  | 1.04 – 1.33 s       | **free middle** |
| 5  | 33 – 40  | 1.38 – 1.67 s       | **free middle** |
| 6  | 41 – 48  | 1.71 – 2.00 s       | **free middle** |
| 7  | 49 – 56  | 2.04 – 2.33 s       | **free middle** |
| 8  | 57 – 64  | 2.38 – 2.67 s       | **free middle** |
| 9  | 65 – 72  | 2.71 – 3.00 s       | **free middle** |
| 10 | 73 – 80  | 3.04 – 3.33 s       | **free middle** |
| 11 | 81 – 88  | 3.38 – 3.67 s       | **free middle** (last) |
| 12 | 89 – 96  | 3.71 – 4.00 s       | ← end conditioning |
| 13 | 97 – 104 | 4.04 – 4.33 s       | ← end conditioning |
| 14 | 105 – 112| 4.38 – 4.67 s       | ← end conditioning |
| 15 | 113 – 120| 4.71 – 5.00 s       | ← end conditioning (last) |

Start conditioning: px 0–24 (25 frames) → latent frames **0–3** (`k_lat = 4`).  
End conditioning: px 89–120 (32 frames, latent 12 starts at px 89) → latent frames **12–15** (`end_idx = 12`).  
**Free middle** = latent frames **4–11** = px 25–88 = **1.04 s – 3.67 s**.

---

## Computed features

All features are functions of `(τ, p)` — denoising step × latent frame.
`z_τ(p)` denotes the spatial latent tensor at step τ and frame p, **flattened to a
vector** over the channel, height, and width dimensions (C × H' × W' = 128 × 16 × 24 = 49 152 elements).

### 2-D feature matrices  `[S, *]`

| Feature | Formula | Shape | What a spike means |
|---------|---------|:-----:|-------------------|
| `norm_z` | `‖z_τ(p)‖₂` | `[S, F']` | Latent energy at that frame / step. Decreases as denoising cleans up noise. |
| `speed_z` | `‖z_τ(p+1) − z_τ(p)‖₂` | `[S, F'−1]` | How much the latent jumps between adjacent frames at step τ. Large = abrupt spatial change. |
| `curvature_z` ★ | `‖z_τ(p+2) − 2·z_τ(p+1) + z_τ(p)‖₂` | `[S, F'−2]` | **Discrete Laplacian** along frame axis — how sharply the trajectory bends. Primary dissolve signal. |
| `angular_z` | `cos(z_τ(p+1)−z_τ(p),  z_τ(p+2)−z_τ(p+1))` | `[S, F'−2]` | Cosine similarity of consecutive frame deltas. Near −1 = direction reversal = dissolve boundary. |
| `pred_mag` | `‖v_θ(z_τ, τ, c)(p)‖₂` | `[S, F']` | Magnitude of the model's velocity prediction per frame. Spike = model exerts most effort there. |
| `pred_curv` | `‖v_θ(p+2) − 2·v_θ(p+1) + v_θ(p)‖₂` | `[S, F'−2]` | Curvature of the prediction field along frame axis. |
| `step_size_z` | `‖z_{τ+1}(p) − z_τ(p)‖₂` | `[S, F']` | How much each frame's latent moves between consecutive denoising steps. Large = frame is "hard" to denoise. |

### 1-D vectors on the final clean latent `z₀`  `[F'−*]`

These are the most diagnostic for dissolve detection, because `z₀` is what the
decoder actually renders.

| Feature | Formula | Shape | Notes |
|---------|---------|:-----:|-------|
| `norm_z0` | `‖z₀(p)‖₂` | `[F']` | Per-frame energy after full denoising. |
| `speed_z0` | `‖z₀(p+1) − z₀(p)‖₂` | `[F'−1]` | Frame jump in the clean latent. |
| `curvature_z0` ★★ | `‖z₀(p+2) − 2·z₀(p+1) + z₀(p)‖₂` | `[F'−2]` | **Primary dissolve signal** on the final output latent. |
| `angular_z0` | `cos(z₀(p+1)−z₀(p),  z₀(p+2)−z₀(p+1))` | `[F'−2]` | **Direction flip signal** on the final output latent. |
| `pred_mag0` | `‖v_θ(z_{τ=final}, τ, c)(p)‖₂` | `[F']` | Prediction magnitude at the last denoising step. |

### Scalar dissolve descriptors (per sample)

| Column | Formula | Meaning |
|--------|---------|---------|
| `pred_global_p` | `argmax_p curvature_z0(p) + 1` | Frame with highest curvature — global over all frames. |
| `pred_global_s` | `pred_global_p × 8 / 24` | Same in seconds. |
| `pred_free_p` | `argmax_{p ∈ [k_lat..end_idx−1]} curvature_z0(p) + 1` | Same but restricted to the **free middle** only — more reliable. |
| `pred_free_s` | `pred_free_p × 8 / 24` | Same in seconds. |
| `dissolve_strength` | `max(curvature_z0) / mean(curvature_z0)` | How "spiky" the curvature profile is. 1.0 = flat (no clear dissolve); 1.5+ = clear peak. |
| `angular_min` | `min_p angular_z0(p)` | Most negative cosine similarity. Closer to −1 = sharper direction reversal. |
| `gt_s` | — | Visual ground truth: second at which the dissolve is visible in the rendered video. |
| `err_global_s` | `|pred_global_s − gt_s|` | Absolute error of the global prediction vs ground truth. |
| `err_free_s` | `|pred_free_s − gt_s|` | Absolute error of the free-middle-restricted prediction. |

---

## What each plot shows

### `{sample}_heatmaps.png` — six (τ × p) heatmaps

**x-axis = latent frame p (0–15)**,  **y-axis = denoising step τ (0 = noisy, top; 39 = clean, bottom)**.  
Cyan dashed = start-conditioning boundary.  Magenta dashed = end-conditioning boundary.

| Panel | Formula | What to look for |
|-------|---------|-----------------|
| `‖z_t(p)‖` | `‖z_τ(p)‖₂` | Bright early rows (noisy) fading to dark (clean). Should be uniform across p at any τ. |
| `‖v_θ(p)‖` | `‖v_θ(p)‖₂` | Persistent bright column in the free middle = model consistently working hardest there. |
| `speed_z` | `‖z_τ(p+1)−z_τ(p)‖₂` | Hot column = stable large jump between those frames across all denoising steps. |
| `step_size_z` | `‖z_{τ+1}(p)−z_τ(p)‖₂` | Hot row at a specific τ = model makes its biggest moves at that step. Hot column = one frame always hard to denoise. |
| `curvature_z` ★ | `‖Δ²_p z_τ‖₂` | **Bright vertical stripe in the free middle across many τ = dissolve crystallising there.** Diffuse or boundary-only = no clean dissolve. |
| `angular_z` | `cos(Δz_τ(p), Δz_τ(p+1))` | Red stripe (< 0) at a fixed p across all τ = persistent direction reversal at that frame. |

---

### `{sample}_dissolve_profile.png` — five 1-D signals on final `z₀`

The most diagnostic plot for each sample.  Three vertical markers are drawn on
every panel:

| Marker | Colour | What it marks |
|--------|--------|--------------|
| **Predicted global** | Yellow solid | `argmax(curvature_z0)` over all frames (may be dominated by the conditioning boundary at p=12). |
| **Predicted free-mid** | Orange dashed | `argmax(curvature_z0)` restricted to free middle p=4–11. Usually more accurate. |
| **Visual ground truth** | Green solid | Second when the dissolve is visible in the rendered video (where annotated). |

| Panel | Formula | Simple reading |
|-------|---------|----------------|
| `curvature_z0` (orange bars) | `‖z₀(p+2)−2z₀(p+1)+z₀(p)‖₂` | Tall bar at p* = dissolve is there. Flat profile = distributed confusion, no single cut. The `strength` value in the title quantifies how spiky the bar chart is. |
| `angular_z0` (blue line) | `cos(Δz₀(p), Δz₀(p+1))` | Deep dip below 0 = latent deltas reverse direction at that frame (hard transition). Red fill marks the reversal zone. |
| `speed_z0` (purple line) | `‖z₀(p+1)−z₀(p)‖₂` | Peak = large spatial jump between adjacent frames in the final video. |
| `pred_mag0` (grey bars) | `‖v_θ(p)‖₂` at last step | Uniform = model worked equally hard everywhere. Spike = one frame dominated the last denoising step. |
| `norm_z0` (green bars) | `‖z₀(p)‖₂` | Per-frame energy. Conditioning injects context via tokens, not by zeroing latents, so values are broadly similar across conditioned and free frames. |

---

### `comparison_dissolve.png` — all 10 samples overlaid

**Top**: normalised `curvature_z0`.  **Bottom**: `angular_z0`.  
Solid lines = coloured by class.  **Dashed coloured ticks = free-mid prediction**.  **Solid coloured ticks = visual GT** (where annotated).

---

### `dissolve_frame_evolution.png` — when does the model commit?

Y-axis = estimated dissolve frame (argmax of `curvature_z[τ]`) vs. denoising step τ.  
Stable early (left) → model commits to its dissolve location while latents are still noisy.  
Stable late (right) → dissolve location is determined in the final clean-up steps.

---

### `curvature_grid.png` — small-multiple curvature heatmaps

All samples at a shared colour scale, sorted by class.  Bright patch = dissolve location.

---

## Validation against visual ground truth

Ground truth annotated by manual inspection of rendered videos (run_0002, seed 42).

GT latent frame is computed using the causal formula: `p = (px−1)/8+1` where `px = round(t×24)`.

| Sample | Class | GT (s) | GT p (causal) | Pred global p / s | Pred free p / s | Err global | Err free | Outcome |
|--------|:-----:|:------:|:-------------:|:-----------------:|:---------------:|:----------:|:--------:|:-------:|
| blackswan → mallard-water | 1 | 1.50 | p≈5.4 | p=12 / 4.00 s | — | 2.50 s | — | smooth |
| mallard-fly → mallard-water | 2 | 2.82 | p≈9.5 | p=12 / 4.00 s | — | 1.18 s | — | smooth |
| motocross-bumps → motocross-jump | 2 | 2.27 | p≈7.5 | p=12 / 4.00 s | — | 1.73 s | — | smooth |
| paragliding-launch → paragliding | 2 | 2.79 | p≈9.4 | p=12 / 4.00 s | — | 1.21 s | — | smooth |
| breakdance-flare → breakdance | 5 | 2.49 | **p≈8.4** | p=8 / 2.67 s | p=8 / 2.67 s | **0.18 s ✓** | **0.18 s ✓** | mid-dissolve |
| car-roundabout → bus | 5 | 2.76 | **p≈9.1** | p=9 / 3.00 s | p=9 / 3.00 s | **0.24 s ✓** | **0.24 s ✓** | mid-dissolve |
| car-turn → car-shadow | 5 | — | — | p=12 / 4.00 s | — | — | — | smooth |
| lucia → hike | 5 | 2.44 | p≈8.3 | p=12 / 4.00 s | — | 1.56 s | — | smooth |
| longboard → kite-surf | 6 | 2.50 | p≈8.4 | p=1 / 0.33 s | — | 2.17 s | — | diffuse |
| blackswan → boat | 8 | 3.48 | p≈11.8 | p=1 / 0.33 s | — | 3.15 s | — | diffuse |

`Pred free` shown where it differs from global (i.e. the global argmax was outside the free middle).

### What the validation reveals

**What works:**  `curvature_z0` correctly identifies the dissolve frame for the two
class-5 "mid-dissolve" samples with errors ≤ 0.24 s (< 2 latent frames).  
These are the **only** samples where the curvature peak in the free middle dominates.

**What fails (and why):**

| Case | Issue | Root cause |
|------|-------|-----------|
| Class 1/2 (smooth) | Global argmax=p=12 (4.00 s), but visual dissolve is at 1.5–2.8 s | The conditioning boundary at p=12 creates the sharpest curvature in the final latent, masking a weaker (but real) peak in the free middle |
| Class 5 lucia/car-turn | Same pattern as class 1/2 | These transitions also hit the boundary effect despite being class-5 |
| Class 6/8 (diffuse) | Global argmax=p=1 (meaningless), actual dissolve at 2.5–3.5 s | Curvature is nearly flat (`strength` ≈ 1.1–1.3); no localised bend; the whole free middle is incoherent |

**Key insight:** The conditioning boundary at p=12 always produces a curvature artefact that can
outcompete the real dissolve signal in the free middle.  The fix is to either:
1. Restrict the search to the free middle (`pred_free`) — already implemented.
2. Subtract the conditioning-boundary baseline from the profile before taking the argmax.
3. Use a different discriminative metric in the free middle (speed spike, angular dip, or pred_mag peak) rather than relying on curvature alone.

**Next step:** For the class-5 smooth + class-6/8 cases, inspect the free-middle curvature profile
directly (bars between p=4 and p=11) and compare to GT to quantify whether the signal is present
but weak, or genuinely absent.
