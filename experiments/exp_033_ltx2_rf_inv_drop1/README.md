# exp_033 — LTX-2 RF-Solver inversion with end-clip first-latent-frame dropout

Iteration **It-2** of the RF-inversion research loop (`notes/rf_inversion_loop.md`).
Fork of exp_032 with **exactly one change** — drop the worst-discontinuity
token from the conditioning mask, restore production sub-clip anchors.

## Question

exp_032 cleared the perceptual bar (8/10 pass, recon PSNR median 40.9) using
TRUE self-conditioning (`clean_latents` = exact slices of z₀). But that recipe
is **not deployable for editing**:

- At edit time you only have the two endpoint sub-clips, never z₀.
- z₀-slice anchors also leak middle-frame info into the end-clip position
  (causal VAE), violating sub-clip-isolated boundaries.

The discontinuity between (i) sub-clip-isolated encode (production) and
(ii) the corresponding slice of full-video encode (z₀) is **concentrated at
one position** — the first latent frame of the **end** sub-clip:

| Token position | sub-clip encode | full-clip encode |
|---|---|---|
| start sub-clip, first latent frame | fresh-start anchor | fresh-start anchor ✓ match |
| end sub-clip, first latent frame   | fresh-start anchor | carries causal info from the entire middle ✗ |
| end sub-clip, frames ≥ 2 | causal context = sub-clip own past | causal context = full middle (decays as depth grows) |

**Does masking out that single end-clip first-latent-frame position recover
most of exp_032's quality using deployable (sub-clip-isolated) anchors?**

## Setup

Identical to exp_032 **except** the conditioning prep:

```yaml
data           : data/processed/transitions/shadow_smoke/*.mp4  (10 clips)
vc conditioning: first 1s + last 1s of the same clip define MASK geometry;
                 anchors come from apply_visual_conditioning's re-encoded
                 SUB-CLIPS (production, sub-clip-isolated, deployable).
                 cmask_packed is then zeroed at the first latent frame of
                 the END sub-clip — the solver fills that token freely.
resolution     : max_area = 393216, per-clip aspect via resolve_resolution
audio context  : encoded silent mel through pipe.audio_vae, fixed across steps
inversion      : 40 RF-Solver midpoint steps, CFG=1
reconstruction : 40 RF-Solver midpoint steps, CFG=1
regeneration   : 40 Euler steps, CFG=4
dual gate      : informational only (logged, no retry)
```

## The one change vs exp_032

In `run.py`, after `pipe.apply_visual_conditioning(...)`:

```python
tokens_per_latent_frame = latent_height * latent_width
end_latent_idx = max(int(i) for i in condition_indices)       # N_lat - K_lat
drop_start = end_latent_idx * tokens_per_latent_frame
drop_end   = drop_start + tokens_per_latent_frame             # ONE latent frame
cmask_packed[:,  drop_start:drop_end] = 0.0
clean_latents[:, drop_start:drop_end] = 0.0                   # defensive (mask=0 anyway)
```

Assumes `transformer_spatial_patch_size == transformer_temporal_patch_size == 1`
(LTX-2 19B); asserted at runtime.

Note: `clean_latents` is now `apply_visual_conditioning`'s output (sub-clip
re-encoded production anchors), **not** the `z0_packed.clone()` from exp_032.

## How to run

```bash
cd /workspace/diffusion-research
python experiments/exp_033_ltx2_rf_inv_drop1/run.py
```

Outputs land in `outputs/videos/exp_033_ltx2_rf_inv_drop1/run_NNNN/`.

## Expected outcome / decision rule

Pre-registered against exp_030 (broken baseline, recon PSNR median ~18,
recon_rel median 0.68) and exp_032 (idealized self-cond, recon PSNR median
40.9, 8/10 pass):

| recon PSNR median (vs exp_032's 40.9) | reading |
|---|---|
| ≥ 35 dB, ≥6/10 pass exit ① | Dominant cost was the one position. Dropout is the deployable fix. |
| 25–35 dB | Partial: this position is significant but not sufficient. Move to deeper dropout (drop first 2 / 3 latent frames) or soft-pin warmup. |
| ≤ 25 dB | Mismatch is distributed across the whole sub-clip; one-frame dropout isn't enough. Reframe: production-anchored inversion accepted as-is, accept lower recon, validate via edit quality instead. |

## Outputs

Per sample under `<run_dir>/<sample_id>/`: `z0.pt`, `z1.pt`, `z_t_25/50/75.pt`,
`source_video.mp4`, `recon_video.mp4`, `regen_video.mp4`,
`step_diag_{invert,reconstruct,regenerate}_*.csv`, `inv_meta.yaml`.
Run-level: `summary.yaml`, `config_snapshot.yaml`, `run.log` (with the final
metric table).
