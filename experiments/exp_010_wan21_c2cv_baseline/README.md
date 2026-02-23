# exp_010_wan21_c2cv_baseline

## Question

Can we extend the FLF2V inpainting conditioning of Wan 2.1 from single
first/last frames to multi-frame anchor clips, enabling a Video Connecting
task where the generated video seamlessly bridges two given clips?

## Setup

- **Model:** `Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers`
- **Pipeline:** `WanVideoConnectPipeline` (`src/diffusion/pipeline_wan_c2v.py`)
  — derived from the official `WanImageToVideoPipeline` with the following
  changes:
  - `prepare_latents` accepts `start_clip (B,C,N,H,W)` + `end_clip` instead
    of single frames; builds sparse conditioning video
    `[start_clip | zeros(middle) | end_clip]`.
  - Temporal mask has `1`s for the first and last `anchor_frames` pixel-frame
    positions (instead of just frame 0 and frame −1).
  - `__call__` accepts `start_clip`, `end_clip`, `anchor_frames`; uses first
    frame of `start_clip` as CLIP image reference.
- **Baseline config (`anchor_frames=1`):** single images passed as 1-frame
  clips, which reproduces FLF2V exactly — used to validate the new pipeline
  against exp_009.
- **Resolution:** 720p (max_area 921600)
- **Frames:** 25
- **Steps:** 15
- **CFG:** 5.5

## How to run

```bash
cd /workspace/diffusion-research
python experiments/exp_010_wan21_c2cv_baseline/run.py
```

To test with real multi-frame clips, set `anchor_frames > 1` in `config.yaml`
and point `start_clip` / `end_clip` to directories containing sorted PNG
frames (the run script loads the first `anchor_frames` images from each
directory).

## Outputs

`outputs/videos/exp_010_wan21_c2cv_baseline/run_NNNN/s42_af1_steps15_cfg5.5.mp4`

The filename encodes seed (`s`), anchor frames (`af`), steps, and CFG scale
for easy comparison across runs.
