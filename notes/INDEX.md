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
| [`rf_inversion_loop.md`](rf_inversion_loop.md) | Process | CLOSED autonomous research loop for real-clip RF-Solver inversion — protocol, success/exit criteria, pod-hour budget, full pre-registration Ledger |
| [`rf_inversion_postmortem.md`](rf_inversion_postmortem.md) | Process | Explanatory complement to the Ledger — narrates the four intervention families, the reasoning behind each, the named cause, paths forward |
| [`dataset/vc_bench.md`](dataset/vc_bench.md) | Dataset | VC-Bench task definition, class taxonomy (1/2/5/6/8), dataset structure, 9 eval metrics (VQS/SECS/TSS), transfer approach |
| [`dataset/vc_bench_exploration.md`](dataset/vc_bench_exploration.md) | Dataset | Interactive tools for browsing VC-Bench (FiftyOne, Gradio, options compared) |
| [`dataset/autotransition.md`](dataset/autotransition.md) | Dataset | AutoTransition (HF, 35k templates, 107 transition names, ~52.9 GB) — JSON schema, split-tar-not-gzip gotcha, partial-download recipe |
| [`eval_harness_v3.md`](eval_harness_v3.md) | Eval | Transition-eval harness v3 — CERTIFIED `eval/v3.0.0`; positioning, plan→infer→score flow, trust map, certification meaning; SPEC.md in-package is authoritative |
| [`ideas.md`](ideas.md) | Research | Hypothesis / minimal-experiment sketches |
| [`exp/exp_043_smoke_manifold.md`](exp/exp_043_smoke_manifold.md) | Experiment | Smoke "transition manifold" hypothesis — Phase-1 diagnostics (M1..M6) on cached exp_043 z0. `v_smoke` direction is real & specific; per-clip PC1 unshared; centroid bulges at t≈8. Phase-2/3 roadmap. |
| [`exp/exp_043_inverted_noise_vs_gaussian.md`](exp/exp_043_inverted_noise_vs_gaussian.md) | Experiment | z1 (RF-inverted noise) vs matched-Gaussian deviation. **HYPOTHESIS REFUTED: z1's free-middle is white Gaussian (kurt +0.08, autocorr +0.02, temporal +0.02); the deviation lives only in the clamped anchors (= z0 slices). The smoke signature is in z0's free-middle (kurt +0.48, cross-clip cos +0.26), not z1's.** Replicated across portrait/landscape/square. Recommend sourcing signature from z0 + late-σ trajectory guidance. |
| [`exp/exp_044_smoke_transition_injection.md`](exp/exp_044_smoke_transition_injection.md) | Experiment | Smoke-transition injection arc (goal: free-middle regen PSNR>18). exp_044 refutes the CFG hypothesis — recon→regen gap is solver self-consistency. exp_045 decode-ceiling (info wall). exp_046/047 perceptual donor recipes. **exp_049: σ-matched recon-trajectory injection RECOVERS recon (ss0 33.26 dB) for z1-rich clips, late-σ carries it (late=all); z1-poor saturate (~12-15); donor deployable only early-window. + @torch.inference_mode() OOM footgun.** |
| [`exp/exp_050_lora_baseline.md`](exp/exp_050_lora_baseline.md) | Experiment | **Standard-LoRA baseline on the 10 shadow-smoke clips via the OFFICIAL trainer (campus cluster, 3 H100 arms, ~1h each). Concept acquired by step 500-1000: subject-wrapping ink-black billow on held-out trigger prompts; zero drift on no-trigger prompts; rank-32 attention-only suffices; i2v_ff05 arm = best default (adds first-frame conditioning for free).** Trainer/resume/queue infra facts + checkpoint ladder saved. |
| [exp/exp_051_c2v_ladder.md](exp/exp_051_c2v_ladder.md) | C2V conditioning-mode ladder (t2v/i2v/c2v LoRA): all arms hold endpoint anchors; c2v = tightest wrap + fastest acquisition (step 250); concept rides caption phrase, not trigger |
| [exp/exp_052_eval_harness.md](exp/exp_052_eval_harness.md) | Transition eval harness v1 (`src/diffusion/transition_eval/`): content-invariant metrics (morph profile / motion fidelity / effect appearance / seams / leakage / rubric judge) + lerp-floor & real-ceiling normalization; VALIDATED by style-discrimination exam — appearance 93% 1-NN (chance 24%), metrics fail on complementary classes; exp_051 ladder reproduced quantitatively |
| [exp/exp_053_eval_harness_v2.md](exp/exp_053_eval_harness_v2.md) | Eval harness v2: core mask SURVIVES its ablation (all-frames M3 0.78 < 0.88 bar); M6 leakage adversarially validated on 12 ground-truth copies (min 0.926 vs honest max 0.78, incl. a mislabel audit that reproduced the z1 dichotomy); trust flags + mean±std + Wilson CIs = standard report; ladder base-vs-LoRA survives PAIRED test (6/6), within-LoRA differences don't at n=6; judge → Gemini native-video (q2/q5 degeneracy fixed); trigger claim rescoped to n=3/cell. **§7 (exp_054) full re-validation: 47-clip/11-style corpus, appearance 0.851 (d 2.22), new styles air_bending 0.75 / firelava 0.83 discriminate on appearance only, jump singleton + flying_cam 0.25 confirmed, trust flags refreshed → ladder_v3; report `outputs/eval/exp_054_full_revalidation/REPORT.md`** |
| [exp/exp_056_ic_lora_transition_transfer.md](exp/exp_056_ic_lora_transition_transfer.md) | **IC-LoRA in-context transition transfer (one adapter, 10 classes, type-blind captions): the adapter reads transition STYLE off the in-context reference and re-applies it to foreign endpoints (cross-class app 0.65 @ leak 0.61, seams negative, endpoints 0.97); the BASE model instead near-copies reference CONTENT (leak 0.95–0.98, seams +2.4..+7.1); unseen-class (jump) transfers motion semantics, not appearance. 46-quadruple suite + harness scores + interactive viewer `outputs/eval/exp_056/viewer`** |
| [exp/exp_057_ic_lora_unseen_eval.md](exp/exp_057_ic_lora_unseen_eval.md) | **Broad unseen-class eval of the frozen exp_056 adapter (14 user-labeled classes, 51 quads): in-context transfer survives unseen classes AND unseen one-sided/vanish structure; strength ordering style-texture > object-semantics > camera-arc; texture-cousins (0.45 raw cross) beat novel textures (0.30) — exp_056's transfer partly rode familiar textures; base twins copy on 11/11 (leak 0.97+). METRIC AUDIT: lerp-floor normalization provably broken for 7/16 (one-sided) styles (floor≥ceiling) — raw×leak + twin deltas carry conclusions; anchors reproduce exp_056 within ±0.04 raw. Viewer `outputs/eval/exp_057/viewer`** |
| [exp/exp_058_ic_lora_diverse_retrain.md](exp/exp_058_ic_lora_diverse_retrain.md) | **Diversified mixed-conditioning IC-LoRA retrain (460 pairs / 32 classes; one-sided PREFIX-ONLY via per-pair masks, two-sided prefix+suffix): paired v1→v2 on 40 identical items raw app +0.046 (24/40); held-out vanish probe gas_transformation IMPROVED (0.41→0.61 in-class, appearance-blind → genuine in-context gain); novel-texture gradient largely closed (cross 0.33→0.42); camera cross-target FLAT (conditioning-conflict confirmed); NEW capability: prefix-only generation works (raw 0.50, base twins copy at leak 0.99/seam +117) but leak rises without the suffix anchor (0.73 vs 0.67); COST: anchors −0.06/−0.10 raw (more conservative two-sided transfer) and seam snaps exactly where suffix conditioning meets prefix-only-trained classes. Mask conditioning proven bit-exact vs prefix+suffix. Viewer `outputs/eval/exp_058/viewer`** |
| [exp/exp_059_inference_benchmark.md](exp/exp_059_inference_benchmark.md) | **LTX-2 inference benchmark on H100 80GB (720p/1080p, 5 s, granular per-section timers): dev two-stage 142/215 s warm; distilled eager 103/116 s (only 6–16 s is denoise — rest is per-call model rebuild); distilled + GPU-resident registry 11.2/23.4 s @ 69–72 GB (dev can't: stage-2 LoRA fusion OOM). compile −6–8 % dev / ≈0 dist (+10 min tax, needs py≤3.13 env `.venv-py312`); disk-cold loads ~50 MB/s (Gemma 400 s, ckpt 860 s)** |

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

**`rf_inversion_loop.md`** — *Closed 2026-05-16 via exit ②.* The autonomous
research loop bridging exp_029 (RF-Solver inversion on generated latents) →
exp_030's goal (real clips, where it collapsed). Holds the COLD→HOT→COLD
iteration protocol, the four exit conditions, pod lifecycle rules, and the
full Ledger of all 9 deployable recipes tested. Read this before resuming any
RF-inversion work.

**`rf_inversion_postmortem.md`** — *Closed-loop explanatory report.* Narrates
what was tried and why: the four intervention families (anchor-quality,
model-bootstrap middle, solver step-escalation, σ-conditional release), the
mechanism behind each, why each one failed, the named cause (free-middle cost
coupled to anchor quality through velocity coupling), and three paths forward
outside §0. Designed for a reader who wasn't in the room.

### Eval

**`eval_harness_v3.md`** — The transition-eval harness map: certified status (`eval/v3.0.0`, 2026-07-14), what certification claims and doesn't, the plan→infer→score flow, trust-map consumption rule, open items before the first model report. Authoritative detail lives in `src/diffusion/transition_eval/SPEC.md` (metrics §3, health assessment §6, change protocol §10); committed records in `src/diffusion/transition_eval/certifications/`.

### Research

**`ideas.md`** — Scratch pad. Format: `Question / Hypothesis / Minimal experiment to falsify`.

---

## Maintenance rules

- **Add a file** → add a row to the quick-scan table + an entry in the relevant area section.
- **Remove a file** → remove both entries and delete the file.
- **Significant content update** → update the file's description in the detail section above.
