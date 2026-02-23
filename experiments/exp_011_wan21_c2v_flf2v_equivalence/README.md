# exp_011_wan21_c2v_flf2v_equivalence

## Question

Does `WanVideoConnectPipeline` with `anchor_frames=1` produce the same result
as the native FLF2V pipeline (`WanImageToVideoPipeline`)?

## Hypothesis

With a single anchor frame at each end, the C2V conditioning is structurally
identical to FLF2V:

| | FLF2V (exp_009) | C2V af=1 (exp_011) |
|---|---|---|
| `video_condition` | `[first \| zeros(nf-2) \| last]` | `[start_clip \| zeros(nf-2) \| end_clip]` |
| Mask 1s | pixel-frame 0 and nf-1 | pixel-frames 0 and nf-1 |
| CLIP embed | `encode([first, last])` | `encode([start_clip[0], end_clip[0]])` |

All three paths collapse to the same tensors, so outputs should be visually
identical to exp_009.

## Setup

- **Model:** `Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers`
- **Pipeline:** `WanVideoConnectPipeline` (`src/diffusion/pipeline_wan_c2v.py`)
- **Inputs:** same `first.png` / `last.png` as exp_009
- **anchor_frames:** 1
- **Resolution:** 720p (max_area 921600)
- **Frames:** 25 (1 start + 23 middle + 1 end)
- **Steps:** 15
- **CFG:** 5.5
- **Seed:** 42

## How to run

```bash
cd /workspace/diffusion-research
python experiments/exp_011_wan21_c2v_flf2v_equivalence/run.py
```

## Expected outcome

Output video should match exp_009's output visually.  Any divergence would
indicate a bug in C2V's mask construction, CLIP encoding, or preprocessing
path.

## Outputs

`outputs/videos/exp_011_wan21_c2v_flf2v_equivalence/run_NNNN/s42_af1_nf25_steps15_cfg5.5.mp4`
