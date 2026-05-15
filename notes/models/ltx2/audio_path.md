# LTX-2 audio path — what flows into the transformer

Practical reference for anyone touching audio-side conditioning in LTX-2,
especially in modified pipelines (inversion, feature injection, real-clip
ingestion). Diffusers ≥0.37 / `LTX2ConditionPipeline`.

## TL;DR

- LTX-2 always co-evolves an audio latent alongside the video latent during
  Stage-1 generation. The audio is **never an external input** — the pipeline
  initializes it from noise and denoises it in lockstep.
- The video transformer's audio cross-attention is therefore trained to see
  audio_hidden_states at **the corresponding σ-stage of denoising**, not zeros,
  not pre-encoded mels.
- If you bypass Stage-1 (e.g. inversion on a real silent clip), you need to
  supply an audio context. **Encoded silent mel** is the in-distribution
  choice; zeros is out-of-distribution.
- The audio context fed to the transformer is **packed, normalized** (post-
  `_pack_audio_latents` and `_normalize_audio_latents`) — shape
  `(B, audio_num_frames, latent_channels * latent_mel_bins)`.

## Components (LTX2ConditionPipeline)

| component | role |
|---|---|
| `pipe.audio_vae` (`AutoencoderKLLTX2Audio`) | mel ↔ packed latent |
| `pipe.audio_scheduler` | runs alongside `pipe.scheduler` during Stage-1 |
| `pipe.vocoder` | mel → waveform (decode only) |
| `pipe.audio_sampling_rate`, `pipe.audio_hop_length` | mel-STFT params |
| `pipe.audio_vae_temporal_compression_ratio` | mel time → latent time |
| `pipe.audio_vae_mel_compression_ratio` | mel bins → latent mel bins |

## Audio shape derivation (verbatim from pipeline)

```python
duration_s = num_frames / frame_rate
audio_latents_per_second = (
    pipe.audio_sampling_rate
    / pipe.audio_hop_length
    / float(pipe.audio_vae_temporal_compression_ratio)
)
audio_num_frames = round(duration_s * audio_latents_per_second)
num_mel_bins     = pipe.audio_vae.config.mel_bins          # 64
latent_mel_bins  = num_mel_bins // pipe.audio_vae_mel_compression_ratio
latent_channels  = pipe.audio_vae.config.latent_channels   # 8
```

Encoder input shape:  `(B, in_channels=2, T_mel, num_mel_bins=64)`
Encoder output shape: `(B, latent_channels, audio_num_frames, latent_mel_bins)`
Packed shape fed to transformer: `(B, audio_num_frames, latent_channels * latent_mel_bins)`

`T_mel = audio_num_frames * pipe.audio_vae_temporal_compression_ratio`.

For 5s @ fps=24, sr=16000, hop=160 → `audio_num_frames ≈ 126`.

## At generation time — what the transformer actually sees

`LTX2ConditionPipeline.__call__` initializes audio at `noise_scale = sigmas[0] ≈ 1.0`
(pure N(0,1) in pack-space) and calls `audio_scheduler.step(...)` every iteration.
So at video step σ, the audio context is **at audio's σ_audio of denoising** —
high σ ≈ noise, low σ ≈ clean encoded latent.

Under CFG>1, the batch is `cat([audio_latents]*2)` — cond and uncond halves
share the same audio context.

`audio_timestep` is whatever σ the audio scheduler is at, not necessarily the
video σ (they evolve in parallel but with their own σ schedule).

## `prepare_audio_latents` — the ingestion path

Two modes:

1. `latents=None, noise_scale=1.0` → returns N(0,1) randn in packed space.
   This is what the pipeline uses internally to seed Stage-1 gen. **At
   `noise_scale=0` it still returns randn** (footgun: `_create_noised_state`
   computes `noise_scale·new_randn + (1-noise_scale)·initial_randn` and
   `initial_randn` is ALSO randn — exp_027 was bitten by this).
2. `latents=<4D unpacked latents>, noise_scale=0.0` → packs + normalizes the
   provided latents and returns them clean (deterministic).

Use mode 2 whenever you have a real audio latent to ingest. Mode 1 is for
fresh generation only.

## Bypassing Stage-1 (inversion / real-clip pipelines)

When you skip Stage-1 generation (e.g. exp_030: VAE-encode a real clip into
z₀ directly), there's no audio scheduler running. You must supply an audio
context manually. Options ranked by quality:

| approach | in-distribution? | matches training? | notes |
|---|---|---|---|
| `torch.zeros((1, audio_num_frames, C*M))` | NO | NO | What exp_029 used. Cross-attention sees an OOD vector. |
| `pipe.audio_vae.encode(zeros_mel).mode()` + pack + normalize | YES | partially | "Encoded silent mel" — what exp_030 uses. Fixed across all steps. |
| Same as above + σ-conditional noise per step | YES | YES | Most faithful to training; reproduces audio scheduler's trajectory. Not done yet. |

**Why encoded-silent beats zeros:** the audio VAE never outputs zero latents
on any real input, so the transformer was never trained to see a zero audio
context. Encoded silent mel sits inside the audio VAE's output manifold —
it's the σ=0 endpoint of any silent training clip.

**Why fixed-across-steps is acceptable for inversion:** the test we care
about (z₀ → z₁ → z₀_recon) only needs invert and recon to see the *same*
audio context at corresponding steps. Holding it constant satisfies that.
For regen the fixed-silent context does NOT exactly match production
(production evolves audio noisy→clean), but using the same audio in invert
keeps the comparison clean — the inv_regen gap then reflects video-flow
mismatch only.

## Snippet — build encoded-silent context

```python
def build_silent_audio_context(pipe, num_frames, frame_rate, device, dtype):
    duration_s = num_frames / float(frame_rate)
    audio_latents_per_second = (
        pipe.audio_sampling_rate / pipe.audio_hop_length
        / float(pipe.audio_vae_temporal_compression_ratio)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)
    T_mel        = audio_num_frames * pipe.audio_vae_temporal_compression_ratio
    num_mel_bins = pipe.audio_vae.config.mel_bins
    in_channels  = pipe.audio_vae.config.in_channels   # 2

    silent_mel = torch.zeros((1, in_channels, T_mel, num_mel_bins),
                             dtype=pipe.audio_vae.dtype, device=device)
    latents_4d = pipe.audio_vae.encode(silent_mel).latent_dist.mode()  # deterministic
    return pipe.prepare_audio_latents(
        latents=latents_4d, noise_scale=0.0, device=device, dtype=dtype,
    )
```

Returned tensor goes directly into `transformer(audio_hidden_states=...)`.
For CFG>1, duplicate batch-wise: `torch.cat([ah, ah], dim=0)`.

## AudioContextRecorder (Stage-1 capture path)

When you ARE running Stage-1 gen and want to replay the produced audio
trajectory in a downstream phase, wrap `pipe.transformer.forward` and
snapshot `audio_hidden_states` per call. Under CFG>1 both batch halves are
identical at capture time (pipeline pre-doubles by `cat([audio_latents]*2)`)
— store only the first half. See exp_029 for the working implementation.

This is overkill for any pipeline that doesn't run Stage-1 (exp_030 dropped
it entirely).

## Pitfalls / footguns

- `prepare_audio_latents(noise_scale=0, latents=None)` returns **randn**, not
  zeros. Don't call it expecting "audio off." Use mode 2 with explicit
  zero-valued unpacked latents, or skip the helper and build zeros directly.
- Audio is **always batch-2 under CFG>1** even though cond and uncond audio
  are identical. Forgetting to double leads to a shape mismatch at the
  transformer call.
- `audio_timestep` is a 1-D tensor per batch — `torch.full((bsz,), value)`,
  NOT `(1,)` broadcast. Inversion/recon at CFG=1 → batch 1 → `(1,)`.
- The audio VAE expects mel input of shape `(B, in_channels=2, T_mel, mel_bins)`
  — note the **2 channels** (stereo). Silent mel still needs both channels.
- The audio VAE's `latent_dist` is a `DiagonalGaussianDistribution`; use
  `.mode()` for determinism, never `.sample()` unless you also pin a
  generator and accept the stochasticity.

## References

- Pipeline source: `diffusers/pipelines/ltx2/pipeline_ltx2_condition.py`
- Audio VAE: `diffusers/models/autoencoders/autoencoder_kl_ltx2_audio.py`
- Encoder signature: `LTX2AudioEncoder.forward(hidden_states)` → input shape
  `(B, in_channels, T_mel, num_mel_bins)` (comment in source).
- Used by: exp_029 (capture-and-replay path), exp_030 (encoded-silent path).
