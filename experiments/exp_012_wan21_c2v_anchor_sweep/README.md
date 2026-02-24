# exp_012_wan21_c2v_anchor_sweep

## Question

How does the number of anchor frames (1 → 24) affect video connecting quality
across different video pairs (self and cross)?

## Setup

- **Model:** `Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers`
- **Pipeline:** `WanVideoConnectPipeline` (`src/diffusion/pipeline_wan_c2v.py`)
- **Pairs (4):**
  - `1581362_self` — Actions_Activities 1581362, first→last
  - `3106432_self` — Actions_Activities 3106432, first→last
  - `1402988_self` — Actions_Activities 1402988, first→last
  - `3106432x1581362_cross` — 3106432/first → 1581362/first
- **Anchor frames (6):** 1, 2, 4, 8, 16, 24
- **Total runs:** 24 (4 pairs × 6 anchor_frames, single pipeline load)
- **Clips:** `first_last_clips_24` for all anchor_frames (first N frames taken)
- **anchor_frames=1** degenerates to FLF2V conditioning (validates exp_011 mechanism)
- **Resolution:** max_area=399360 (≈480×832), aspect-ratio snapped per pair
- **Frames:** 81  (`(81-1) % 4 == 0` ✓, `24×2 = 48 < 81` ✓)
- **Steps:** 15  **CFG:** 5.5  **Seed:** 42

## How to run

```bash
cd /workspace/diffusion-research
python experiments/exp_012_wan21_c2v_anchor_sweep/run.py
```

## Expected outcome

- anchor_frames=1 should match exp_011's FLF2V output (single-frame conditioning).
- Quality / temporal coherence expected to improve as anchor_frames increases.
- Cross pair reveals whether the model can bridge clips from different videos.

## Outputs

`outputs/videos/exp_012_wan21_c2v_anchor_sweep/run_NNNN/`

Filenames encode: `{pair}_af{N}_s42_nf81_steps15_cfg5.5.mp4`
