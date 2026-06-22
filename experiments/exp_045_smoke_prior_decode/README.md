# exp_045 — Middle-prior decode feasibility (ceiling for smoke injection)

## Question

exp_044 refuted the CFG hypothesis: the production sampler cannot recover the
free-middle smoke transition from z1 (the recon→regen gap is *solver
self-consistency* — recon's midpoint solver retraces the inversion; the
production Euler sampler diverges and produces only generic prompt-smoke).
So the smoke transition must be **injected**.

Before building generative injection, measure the **ceiling**: how well does
each candidate MIDDLE PRIOR decode, if pinned perfectly? Assemble the latent
with perfect anchors (z0) + a candidate middle prior (free latent frames 4–12),
decode, measure free-middle PSNR vs source. Pure VAE decode, no solver.

A pin/inject method using a given prior can do **no better** than this ceiling.
If a deployable smoke-family prior clears ~18 here → injection is viable. If
not → pinning is dead, need a finer substrate (velocity/attention).

## Setup

Loads cached exp_033 `z0.pt` (no re-encode). For each clip, unpacks to
normalized 5d, replaces the free-middle frames with each prior, denormalizes,
decodes. Priors:

- `src` — recipient's own z0 middle (sanity; ~perfect PSNR).
- `gauss` — rms-matched noise (worst case floor).
- `endpoint_hold` / `endpoint_interp` — generic non-smoke baselines (freeze /
  morph the anchors). These are the "generic interpolation" the model bootstrap
  (exp_035) effectively produced.
- `smoke_bcast_loo` — **deployable**: leave-one-out smoke-family, spatial-
  broadcast channel-state per free frame (resolution-agnostic).
- `smoke_bcast_all` — broadcast upper bound (includes self).
- `smoke_spatial_loo` — **deployable**: leave-one-out smoke-family, spatially
  resolved (same-grid donors only).
- `smoke_bcast_loo_keepspatial` — **deployable**: endpoint-interp spatial
  structure with channel-mean shifted to the shared smoke state.

## How to run

```bash
PYTHONPATH=src python experiments/exp_045_smoke_prior_decode/run.py
```

GPU: ~10 decodes/clip × 10 clips, pure VAE decode (no transformer/solver).

## Expected outcome

- `src` ≈ perfect (validates the assemble/decode path).
- `gauss` ≈ the no-information floor.
- Smoke-family priors > endpoint baselines ⇒ the shared smoke is real & helps.
- If the best deployable smoke prior median > 18 ⇒ pin/inject is viable →
  build generative injection (exp_046). Else ⇒ need finer substrate.

## Outputs

`outputs/videos/exp_045_smoke_prior_decode/run_NNNN/`:
`<sample>/source_video.mp4`, `<sample>/prior_<name>.mp4`, `summary.yaml`
(per-clip per-prior free-mid + full PSNR/SSIM), `config_snapshot.yaml`, `run.log`.
