# exp_025 — LTX-2 Negative Prompt Sweep

> **Status: placeholder.** Not implemented yet. Forked from exp_024 design but config/run.py are stubs.

## Question

With empty positive prompt (the winning category from exp_024), is the **negative prompt** the dominant text signal — and is the current negative prompt actively suppressing creative transitions?

Hypothesis: terms like `"frame blending, distortion, temporal artifacts"` in the negative prompt push CFG away from exactly the kind of creative morphing we want between semantically distant endpoints. With an empty positive, CFG only steers *away from* the negative, so the negative is doing all the text-side work.

## Setup

- Same DAVIS clip pairs as exp_024 (start with a 3-pair diagnostic subset: one easy / one mid / one hard class).
- Positive prompt: `""` (locked, the exp_024 winner).
- Fixed: `guidance_scale=3.2`, `num_inference_steps=40`, `seed=42`, `num_frames=193`, `num_clip_frames=25`, `start_clip_strength=1.0`, `end_clip_strength=1.0`.
- **Sweep axis — negative prompt variants:**
  1. `N_empty` — `""`
  2. `N_minimal` — quality-only: `"low quality, jpeg artifacts, text, watermark"`
  3. `N_current` — full exp_024 negative (the baseline being challenged)
  4. `N_anti_dissolve` — explicitly anti-dissolve: `"dissolve, crossfade, fade, static, frozen, lifeless, abrupt cut"`

→ 4 variants × 3 pairs = 12 runs.

## How to run

```bash
# placeholder — not yet implemented
python experiments/exp_025_ltx2_negative_prompt_sweep/run.py
```

## Outputs

`outputs/videos/exp_025_ltx2_negative_prompt_sweep/run_NNNN/{sample_id}/{neg_variant}/s42_neg{variant}_steps40.mp4`
