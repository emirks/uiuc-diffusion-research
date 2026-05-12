# exp_026 — LTX-2 Seed × End-Clip Strength Sweep

> **Status: placeholder.** Not implemented yet. Companion to exp_025.

## Question

Two coupled questions on a single diagnostic grid:

1. **Seed variance**: with everything else locked, how much of the "creativity vs dissolve" outcome is just stochastic sampling? If seed alone produces enormously different transitions, single-seed sweeps in earlier experiments are noise.
2. **End-clip strength**: does loosening the endpoint clamp (`end_clip_strength < 1.0`) unlock more creative middle behaviour? The endpoints currently fully overwrite their latent slots; reducing strength lets the model partially denoise them and gain freedom near the boundary.

Together these tell us whether structural conditioning knobs matter more than text.

## Setup

- Same 3-pair diagnostic subset as exp_025 (one easy / one mid / one hard class).
- Positive prompt: `""` (locked, exp_024 winner).
- Negative prompt: TBD — pick the exp_025 winner before running exp_026.
- Fixed: `guidance_scale=3.2`, `num_inference_steps=40`, `num_frames=193`, `num_clip_frames=25`, `start_clip_strength=1.0`.
- **Sweep axes:**
  - `seed` ∈ `{42, 123, 7, 2025}` (4 values)
  - `end_clip_strength` ∈ `{1.0, 0.85, 0.7}` (3 values)

→ 4 seeds × 3 strengths × 3 pairs = **36 runs**.

## How to run

```bash
# placeholder — not yet implemented
python experiments/exp_026_ltx2_seed_endstrength_sweep/run.py
```

## Outputs

`outputs/videos/exp_026_ltx2_seed_endstrength_sweep/run_NNNN/{sample_id}/end{0.X}/s{seed}_end{0.X}_steps40.mp4`

## Analysis

Per pair:
- **Seed-variance estimate**: pairwise visual / metric diff between the 4 seeds at `end_strength=1.0`. Establishes a noise floor.
- **Strength effect**: holding seed fixed, do creative transitions appear as `end_strength` drops?
- **Interaction**: does lowering strength change the *kind* of variance across seeds, or just the magnitude?
