# exp_046 — Perceptual donor-smoke injection

## Question

Success was redefined (user, 2026-06-02) from pixel-PSNR to **perceptual smoke
quality** — pixel-PSNR>18 vs a specific clip is information-limited (exp_045:
donor latent priors ≈ noise floor). exp_044 showed the prompt-only baseline
darkens like smoke but lacks billowing **dynamics** (free-mid tdiff 0.039 vs
real 0.063) and shows color artifacts on hard clips.

**Can we inject a donor sample's REAL smoke (real texture + dynamics) into a
target so the transition perceptually reads as coherent billowing smoke?**

## Setup

Decode-based: assemble target z0-anchors + a **single donor's real free-middle
latent** (pinned) in the free zone (frames 4–12), decode, judge perceptually.
Same-grid donor/target required (token alignment): portrait 22×16 {ss0,2,3,6,8},
landscape 16×22 {ss1,5,7,9}, ss4 (19×19) alone.

Priors: `src` (sanity), `endpoint_interp` (endpoints-only baseline),
`donor:<id>` (pin a donor's real middle), `donorblend:<id>:<a>` (soft blend
donor↔endpoint-interp). Self-injection and grid-mismatched donors auto-skip.

Per output, logs perceptual signals on the free-middle: luminance (darkening),
texture (Laplacian energy), saturation, **tdiff (billowing dynamics)**, plus
LPIPS to source. Per clip, also logs the **REAL** source signature and the
**BASELINE** prompt-only regen (exp_033 `regen_video.mp4`) for direct
comparison. Videos saved for visual judgment (the real perceptual test).

## How to run

```bash
PYTHONPATH=src python experiments/exp_046_smoke_perceptual_inject/run.py
```

Pure VAE decode (no solver). ~9 priors × 4 clips.

## Expected outcome

- Donor-injected middles should show **real billowing dynamics** (tdiff → ~0.06,
  matching real, vs baseline 0.039) and clean smoke (no purple artifacts).
- Visual montages: donor-injected transition should read as coherent smoke
  between the target's endpoints, beating the prompt-only baseline on hard clips.
- `donorblend` tests whether softening helps the onset/dissipation continuity.

## Outputs

`outputs/videos/exp_046_smoke_perceptual_inject/run_NNNN/`:
`<sample>/source_video.mp4`, `<sample>/prior_<name>.mp4`, `summary.yaml`
(per-clip real_sig / baseline_sig / per-prior psig + LPIPS).
