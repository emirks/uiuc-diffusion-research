# exp_047 — Velocity-guided smoke generation

## Question

exp_046's tempblend (latent splice of donor smoke onto the target scene) reads
as smoke but is a static *assembly* — slight blend seam, weak mid-frame motion.
Can we instead make the **model generate** the smoke: run the production sampler
and steer the free-middle toward the extracted smoke target each step, so the
model synthesizes coherent, seamless, turbulent smoke following the donor's
darkening/dynamics? (The "velocity field" substrate.)

## Setup

Fork of exp_044 (loads cached exp_033 z1, regenerates via Euler). New: builds a
**smoke target** = tempblend(donor real free-middle, target endpoint-interp,
Gaussian window peaked at latent frame ~8), packed. During regen, in the x0
(clean-prediction) domain, the free tokens are pulled toward the smoke target by
`guide_weight g` (`_x0_clamp_velocity`), optionally σ-scheduled
(const|lowsigma|highsigma). Anchors stay pinned to clean_latents as before.
g=0 → plain production regen (baseline). Deployable: target endpoints + one
donor sample, same grid (token alignment).

Per variant logs full + free-middle PSNR/LPIPS and perceptual signals
(lum/tex/sat/tdiff); per clip logs the REAL source signature.

## How to run

```bash
PYTHONPATH=src python experiments/exp_047_smoke_velocity_guide/run.py
```

GPU: regen (40 Euler steps) + decode per variant, ×5 variants ×2 clips.

## Expected outcome

- g>0 should produce model-coherent smoke (real texture, seamless) that follows
  the donor's darkening/dynamics — beating tempblend's seam/static-mid and the
  g=0 baseline (static darkened scene).
- Sweep finds the g (and schedule) that best balances smoke presence vs the
  target scene at onset/offset. Judge by visual + lum/tdiff/sat vs REAL.

## Outputs

`outputs/videos/exp_047_smoke_velocity_guide/run_NNNN/`:
`<sample>/source_video.mp4`, `<sample>/regen_<variant>.mp4`, `summary.yaml`, `run.log`.
