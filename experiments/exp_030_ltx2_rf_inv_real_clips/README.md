# exp_030 — LTX-2 RF-Solver inversion on real existing clips

Fork of exp_029. We no longer run Stage-1 generation: `z₀` is the VAE
encoding of an existing source video. The rest of the pipeline (invert →
reconstruct → regenerate, with the informational dual gate and metric
table) is unchanged from exp_029.

## Question

Does RF-Solver round-trip work on real video latents the same way it works
on Stage-1-generated latents? Specifically, for the `shadow_smoke`
transition clips:
- **inv_recon**: does midpoint inversion + reverse midpoint reproduce the
  VAE-encoded source latent? (solver self-consistency)
- **inv_regen**: does the production sampler (Euler + CFG=4) reproduce
  the VAE-encoded source latent from `z₁`? (production usability)

## Setup

```yaml
data          : data/processed/transitions/shadow_smoke/*.mp4
                10 clips (9 of 5s, 1 of 10s — we use first 5s of all).
samples       : 10 (one entry per mp4)
vc conditioning : first 1s (24 frames) and last 1s of THE SAME clip
prompt        : "A floating black smoke transition between objects."
                shared across all samples; deliberately minimal so
                visual conditioning dominates.
resolution    : max_area = 512*768 = 393216; per-clip aspect via
                resolve_resolution(ref_image). Portrait clips render
                ~768x512, landscape ~512x768.
audio context : pipe.audio_vae.encode(zeros mel) → fixed across all
                steps (encoded silence, in-distribution).
inversion     : 40 RF-Solver midpoint steps, CFG=1
reconstruction: 40 RF-Solver midpoint steps, CFG=1
regeneration  : 40 Euler            steps, CFG=4
dual gate     : (informational only — logged, no retry)
  inv_recon : latent_rel < 0.10 AND latent_cos > 0.99
  inv_regen : latent_rel < 0.20 AND latent_cos > 0.97
```

## Differences vs exp_029

1. **No Stage-1 generation.** `z₀` comes from `pipe.video_processor.preprocess_video(frames)` → `pipe.vae.encode` → `_pack_latents`. No `pipe(...)` call.
2. **Encoded-silence audio, not zeros, not capture.** `build_silent_audio_context`
   encodes a literal silent mel `torch.zeros((1, 2, T_mel, 64))` through
   `pipe.audio_vae` (deterministic posterior mode), then packs +
   normalizes via `pipe.prepare_audio_latents(latents=..., noise_scale=0)`.
   The result is a single fixed in-distribution audio context used
   unchanged across invert / reconstruct / regenerate. Rationale: zeros
   in transformer input space is OOD; the encoded silent latent lives
   on the audio VAE's manifold and is what the model would see at σ=0
   for any silent training clip. Holding it fixed makes the audio axis a
   constant — the video flow becomes the only variable in the round-trip.
3. **VC conditioning is single-clip.** Both start and end conditioning
   slices come from THE SAME source video (first 24 frames + last 24).
   `LTX2VideoCondition` placement at `index=0` and the standard end
   latent index is unchanged.
4. **Per-sample resolution.** `inference.max_area` + `resolve_resolution(ref_image)`
   handles portrait / landscape / square sources cleanly.
5. **Prompt is shared.** One short generic line in `inputs.prompt`. LTX-2
   responds well to sparse text when the conditioning is strong — this
   is the smoke-transition motif distilled to a single sentence.
6. **No `audio_strategy` knob, no `AudioContextRecorder`, no zeros fallback.**
   One path, no branching.

## Caveat — regen no longer matches production exactly

The production sampler initializes audio noisy and runs an `audio_scheduler`
in parallel with the video loop, so production-time `audio_hidden_states`
EVOLVES from noisy to clean as σ descends. exp_030 holds audio at the
"clean encoded silent" point at every regen step. This is a deliberate
choice — it isolates the video flow, and using the same audio context in
invert keeps the inv_regen comparison clean — but it means `inv_regen rel`
here is not strictly the production round-trip metric. It's the round-trip
under a fixed-silent-audio production-like sampler.

## How to run

```bash
cd /workspace/diffusion-research
python experiments/exp_030_ltx2_rf_inv_real_clips/run.py
```

Outputs land in `outputs/videos/exp_030_ltx2_rf_inv_real_clips/run_NNNN/`.

## Outputs

Per sample under `<run_dir>/<sample_id>/`:

```
z0.pt, z1.pt, z_t_25/50/75.pt    # packed bfloat16 latents
source_video.mp4                  # decode(z0) — the round-tripped source
recon_video.mp4                   # decode(z0_recon)
regen_video.mp4                   # decode(z0_regen)
step_diag_invert_n40.csv          # per-step diagnostics
step_diag_reconstruct_n40.csv
step_diag_regenerate_n40_n40.csv
inv_meta.yaml                     # full metrics + gate status
```

Run-level:

```
summary.yaml         # per-sample compact metric record
config_snapshot.yaml # the resolved config
run.log              # tee'd stdout incl. the final metric table
```

## What success looks like

- `inv_recon rel ≲ 0.02` across all 10 samples (matches exp_029's tight
  CFG=1 self-consistency band).
- `inv_regen rel ≈ 0.25-0.35` is the structural baseline — the CFG=1↔CFG=4
  flow mismatch documented in exp_029. Anything dramatically worse points
  to source-VAE / conditioning issues specific to the real-clip path.
- The final metric table prints in `run.log`; one row per sample.
