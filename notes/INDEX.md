# Knowledge Base Index

Single entry point for everything learned in this project.
**Update this file whenever a note is added, removed, or significantly changed.**

---

## Quick-scan table

| File | Area | Key subjects |
|------|------|-------------|
| [`models/ltx2/pipeline_api.md`](models/ltx2/pipeline_api.md) | LTX-2 | Diffusers API classes, two-stage recipe, C2V index formulas, CFG cond/uncond batching, conditioning patterns summary, pitfalls & fixes |
| [`models/ltx2/conditioning.md`](models/ltx2/conditioning.md) | LTX-2 | VAE encoding, patchification, RoPE positions, denoise mask, in-grid vs guiding-latent conditioning (deep reference) |
| [`models/ltx2/spatial_locality.md`](models/ltx2/spatial_locality.md) | LTX-2 | P=1 vs P=2 patch geometry; token-to-pixel brick mapping; why global attention still yields localized representations (exp_021+) |
| [`models/ltx2/denoising_schedule.md`](models/ltx2/denoising_schedule.md) | LTX-2 | Dynamic sigma shift, why step_size is large at end, three generation phases (coarse/content/cleanup) |
| [`models/ltx2/audio_path.md`](models/ltx2/audio_path.md) | LTX-2 | What flows into `audio_hidden_states`: shape derivation, Stage-1 audio-scheduler trajectory, encoded-silent vs zeros, `prepare_audio_latents` footguns |
| [`models/wan21/overview.md`](models/wan21/overview.md) | Wan 2.1 | Latent geometry, conditioning channels, C2V setup, VC-Bench benchmark performance, Hub IDs |
| [`theory/transformer_architecture.md`](theory/transformer_architecture.md) | Theory | Token embeddings, self-attention mechanics, MLP blocks, output projection, training vs inference pipeline |
| [`theory/video_diffusion_spaces.md`](theory/video_diffusion_spaces.md) | Theory | The three geometries (video-time, denoising-time, representation), VAE latent vs patch vs transformer hidden space, semantic reasoning |
| [`rf_inversion_loop.md`](rf_inversion_loop.md) | Process | **ACTIVE** autonomous research loop for real-clip RF-Solver inversion — protocol, success/exit criteria, pod-hour budget, live pre-registration Ledger |
| [`dataset/vc_bench.md`](dataset/vc_bench.md) | Dataset | VC-Bench task definition, class taxonomy (1/2/5/6/8), dataset structure, 9 eval metrics (VQS/SECS/TSS), transfer approach |
| [`dataset/vc_bench_exploration.md`](dataset/vc_bench_exploration.md) | Dataset | Interactive tools for browsing VC-Bench (FiftyOne, Gradio, options compared) |
| [`ideas.md`](ideas.md) | Research | Hypothesis / minimal-experiment sketches |

---

## Detail by area

### LTX-2 (`models/ltx2/`)

Four files with clear separation of concerns:

**`pipeline_api.md`** — *Quick lookup reference.*  
Diffusers pipeline classes (`LTX2ConditionPipeline`, `LTX2Pipeline`, `LTX2LatentUpsamplePipeline`, `LTX2VideoCondition`); two-stage production recipe; distilled LoRA role; C2V end-clip latent index formula (`end_idx = N_lat - K_lat`); Stage 2 shape & scheduler alignment; CFG cond/uncond batching (`cat([neg, pos])`, one forward, `chunk(2)`, `s>1` switch, audio shares scale, independent of clip-conditioning `strength`); CPU offload patterns; packed latent unpack fix; pitfalls table.

**`conditioning.md`** — *Deep mechanics reference.*  
Causal VAE geometry (`F_lat = (F_pix-1)//8+1`); patchification & packed token format `[B, N, C]`; 3D RoPE position bounding boxes; denoise mask vs attention mask semantics; in-grid conditioning (Diffusers) vs appended guiding-latent conditioning (vendored `KeyframeInterpolationPipeline`); mask naming inversion between stacks; training strategy alignment notes. **§14-b: the self-conditioning anchor rule for RF-Solver inversion** — pin `clean_latents` to exact slices of z₀, not re-encoded sub-clips (causal VAE makes them differ); validated exp_030→032 (0/10 → 8/10 pass).

**`spatial_locality.md`** — *Token geometry & interpretability.*  
Spatial patch size by variant (LTX-2 19B uses P=1); one-token-one-latent-cell correspondence; 32×32×8-frame pixel bricks; how residual streams and RoPE preserve locality through global self-attention; ties to trajectory analysis.

**`denoising_schedule.md`** — *Scheduler behaviour.*  
Scheduler JSON config (`use_dynamic_shifting=true`, `base_shift=0.95`, `max_shift=2.05`, `time_shift_type=exponential`); dynamic shift formula; non-uniform dt table; why `pred_mag` is large early but `step_size_z` is large late; three generation phases with σ ranges; practical heatmap interpretation for exp_021/022.

**`audio_path.md`** — *Audio cross-attention contract.*  
Shape derivation for `audio_hidden_states` (B, audio_num_frames, latent_channels·latent_mel_bins) from `audio_sampling_rate / audio_hop_length / audio_vae_temporal_compression_ratio` + mel/latent-channel config. Stage-1 audio-scheduler trajectory (noisy at σ=σ_max → clean at σ=0). How to supply audio when bypassing Stage-1 (inversion/real-clip pipelines): encoded-silent mel beats zeros because zeros is OOD for the transformer's cross-attention. `prepare_audio_latents` footguns — `noise_scale=0, latents=None` returns randn, not zeros. Code snippet for `build_silent_audio_context`. Used by exp_029 (capture-and-replay) and exp_030 (encoded-silent).

### Wan 2.1 (`models/wan21/`)

**`overview.md`**  
Temporal VAE scale 4 / spatial scale 8; `num_frames` and `anchor_frames` constraints; 36-channel conditioning split (16 noisy + 4 mask + 16 VAE); mask build order; CLIP embed for 2 frames; VC-Bench ranking; Hub IDs for 1.3B and 14B variants.

### Theory (`theory/`)

**`transformer_architecture.md`** — Foundation notes on how a transformer works: tokenisation, one-hot → embedding, self-attention shapes (Q/K/V projections, output projection), MLP blocks, training vs inference. Prerequisite for understanding the DiT backbone.

**`video_diffusion_spaces.md`** — Critical note on keeping the three latent spaces separate: (1) video-time geometry (frame axis, `Δ_p z`), (2) denoising-time geometry (diffusion step τ, `Δ_τ z`), (3) representation geometry (VAE latent, patch, transformer hidden space). Explains where semantic reasoning is most meaningful and how to correctly formulate VC research hypotheses.

### Dataset (`dataset/`)

**`vc_bench.md`** — Full benchmark reference: task definition (clip-to-video), class taxonomy (1/2/5/6/8 by context/category/motion similarity), dataset stats (1,579 videos, 15 categories, 72 subcategories), 9 evaluation sub-metrics grouped into VQS/SECS/TSS, transfer approach, limitations.

**`vc_bench_exploration.md`** — Research notes comparing tools for interactively browsing the 1,261-video local VC-Bench dataset (FiftyOne, Gradio, custom viewer options).

### Process

**`rf_inversion_loop.md`** — *Live.* The autonomous research loop bridging exp_029
(RF-Solver inversion works on generated latents) → exp_030's goal (real clips,
where it collapsed). Holds the COLD→HOT→COLD iteration protocol, the four exit
conditions (perceptual success / scientific floor / wall / 8-pod-hour budget),
pod lifecycle rules, and the growing Ledger of pre-registered iterations. Read
this before resuming any RF-inversion work.

### Research

**`ideas.md`** — Scratch pad. Format: `Question / Hypothesis / Minimal experiment to falsify`.

---

## Maintenance rules

- **Add a file** → add a row to the quick-scan table + an entry in the relevant area section.
- **Remove a file** → remove both entries and delete the file.
- **Significant content update** → update the file's description in the detail section above.
