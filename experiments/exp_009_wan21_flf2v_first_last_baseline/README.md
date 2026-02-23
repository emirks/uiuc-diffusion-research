# Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers

## Question

The baseline implementation for Wan2.1 FLF2V-14B-720P Model

## Setup

- **Model:** `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
- **Pipeline:** `WanImageToVideoPipeline` — `image=first_frame`, `last_image=last_frame`
- **Resolution:** 480×832
- **Frames:** 25
- **Steps:** 15
- **CFG:** 5.0
- **Conditioning:** first.png + last.png from the `1581362_2562x1440` clip pair

## How to run

```bash
cd /workspace/diffusion-research
python experiments/exp_009_wan21_i2v_first_last/run.py
```

## Outputs

`outputs/videos/exp_009_wan21_i2v_first_last/s42_steps15_cfg5.0.mp4`
