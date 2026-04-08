# LTX-2: Denoising Schedule & Sigma Shift

---

## Scheduler config (from `Lightricks/LTX-2` Hub)

```json
{
  "use_dynamic_shifting": true,
  "base_shift": 0.95,
  "max_shift": 2.05,
  "base_image_seq_len": 1024,
  "max_image_seq_len": 4096,
  "time_shift_type": "exponential",
  "shift_terminal": 0.1,
  "num_train_timesteps": 1000
}
```

There is **no single fixed shift**. The shift is computed dynamically at inference time via **exponential (log-linear) interpolation** between `base_shift` and `max_shift` based on the number of packed tokens N.

Example — Stage 1 at 512×768 with F'=16, H'=16, W'=24:
- N = 16 × 16 × 24 = **6,144 tokens**
- N > `max_image_seq_len` (4096), so shift is clamped near `max_shift` ≈ **2.05**

---

## Effect of shift > 1 on the step grid

The shift transforms the uniform sigma grid:

```
σ_shifted = (shift × σ) / (1 + (shift − 1) × σ)
```

With shift ≈ 2, the per-step time gap `dt_τ = σ_τ − σ_{τ+1}` is **not uniform**:

| Region | σ range | dt after shift ≈ 2 |
|--------|---------|---------------------|
| Early τ (σ ≈ 1, noisy) | tightly packed | **small** (~0.01–0.02) |
| Middle τ (σ ≈ 0.5) | moderate | moderate (~0.02–0.03) |
| Late τ (σ → 0.1, clean) | sparse | **large** (~0.04–0.05) |

`shift_terminal = 0.1` means denoising stops at σ = 0.1, not 0.

---

## Why step_size is large at the end, not the beginning

```
step_size_z[τ, p] = ‖z_{τ+1}(p) − z_τ(p)‖ = dt_τ × ‖v_θ(z_τ, t_τ, c)‖
```

- **`pred_mag`** (velocity magnitude `‖v_θ‖`) is **large at early τ** — the model predicts strongly toward the target when the latent is noisy.
- **`step_size_z`** (actual latent displacement) is **large at late τ** — because dt is large there, despite the smaller v_θ.

They decouple because dt is non-uniform. Both signals are correct; they measure different things.

---

## Three generation phases (Gagneux et al., 2025)

| Phase | σ range | What happens | Steps with shift > 1 |
|-------|---------|--------------|----------------------|
| **Coarse** | 1 → 0.7 | Global temporal layout, clip boundaries, dissolve location committed | Many small steps |
| **Content** | 0.7 → 0.3 | Semantic content, motion trajectories filled in | Moderate |
| **Cleanup** | 0.3 → 0.1 | Details, texture, temporal coherence refined | Few large steps |

The shift allocates more steps to the coarse phase — critical for video, where temporal structure (where is the dissolve? how does the arc flow?) must be resolved while the latent is still noisy and easy to steer.

---

## Practical implications for trajectory analysis (exp_021 / exp_022)

- **`curvature_z` heatmap bright bands appear early (τ ≈ 0–5):** the model commits to the dissolve location during the coarse phase, then barely changes it.
- **`step_size_z` heatmap nearly black until the final ~5 steps:** those are the few large-dt cleanup steps — large spatial jumps even though v_θ is already small.
- **`pred_mag` heatmap bright at top (early τ):** velocity predictions are large at high noise regardless of schedule; this reflects the model's "effort", not the latent's movement.
