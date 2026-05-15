# exp_032 — LTX-2 RF-Solver inversion on real clips, with TRUE self-conditioning

Iteration **It-1** of the RF-inversion research loop (`notes/rf_inversion_loop.md`).
Fork of exp_030 with **exactly one change** — the conditioning anchor.

## Question

exp_030 collapsed on real `shadow_smoke` clips (`inv_recon_rel ≈ 0.68`, 0/10
pass). exp_031's R0→R5 ladder localized the cause: exp_030 built `clean_latents`
by separately VAE-encoding the first-1s / last-1s **sub-clips**. The LTX-2 video
VAE is causal, so a sub-clip encode ≠ the corresponding slice of the full-clip
encode — the solver was hard-pinned, every step, to anchors that don't match
z₀'s own conditioned positions ("fault #2").

**Does replacing that with true self-conditioning — `clean_latents` = the exact
slices of z₀ — recover perceptually-faithful inversion on the full 10-clip
set?** exp_031 R5 said yes on 3 clips (free_rel 0.68 → ~0.11, 2/3 pass
perceptual thresholds). exp_032 is the full-set confirmation, targeting loop
exit ①.

## Setup

Identical to exp_030 **except** the conditioning anchor:

```yaml
data           : data/processed/transitions/shadow_smoke/*.mp4  (10 clips)
vc conditioning: first 1s + last 1s of THE SAME clip define the MASK geometry;
                 the anchors pinned there are z₀'s OWN conditioned positions
                 (clean_latents = z0_packed.clone()) — TRUE self-conditioning
resolution     : max_area = 393216, per-clip aspect via resolve_resolution
audio context  : encoded silent mel through pipe.audio_vae, fixed across steps
inversion      : 40 RF-Solver midpoint steps, CFG=1
reconstruction : 40 RF-Solver midpoint steps, CFG=1
regeneration   : 40 Euler steps, CFG=4
dual gate      : informational only (logged, no retry)
```

## The one change vs exp_030

In `run.py`, `pipe.apply_visual_conditioning(...)` is still called — but only to
derive the conditioning **mask geometry**. Its `clean_latents` output (the
re-encoded sub-clips) is **discarded**, and replaced with:

```python
clean_latents = z0_packed.clone()   # TRUE self-conditioning
```

The inverter only reads `clean_latents` where the mask is 1, so passing the full
z₀ is equivalent to slicing — and it's *exact*, by construction:
`z0_cond == clean_latents`. Everything else (per-clip resolution, encoded-silent
audio, invert+recon+regen, metrics, gate) is byte-for-byte exp_030.

## How to run

```bash
cd /workspace/diffusion-research
python experiments/exp_032_ltx2_rf_inv_selfcond/run.py
```

Outputs land in `outputs/videos/exp_032_ltx2_rf_inv_selfcond/run_NNNN/`.

## Expected outcome

Pre-registered (see the loop Ledger): **≥6/10 clips pass exit ①** — recon median
PSNR ≥ 28 dB, SSIM ≥ 0.88, LPIPS ≤ 0.10. Decision rule: ≥6/10 → exit ① triggered
(pending regen secondary check); 3–5/10 → partial, derive It-2; <3/10 → the
mismatch wasn't the dominant cause, reopen.

## Outputs

Per sample under `<run_dir>/<sample_id>/`: `z0.pt`, `z1.pt`, `z_t_25/50/75.pt`,
`source_video.mp4`, `recon_video.mp4`, `regen_video.mp4`,
`step_diag_{invert,reconstruct,regenerate}_*.csv`, `inv_meta.yaml`.
Run-level: `summary.yaml`, `config_snapshot.yaml`, `run.log` (with the final
metric table).
