# exp_051 — C2V capability ladder: conditioning mode (t2v / i2v / c2v) vs clips-to-video generation

**Status: COMPLETE — 4-rung ladder generated on 3 unseen transitions; c2v conditioning-matched
training wins on anchor continuity and learns fastest. 2026-07-06.**

## What was run

C2V = both endpoint clips given (first 2 + last 1 latent frames = first 9 + last 8 pixel
frames), generate the middle. One new LoRA arm (`c2v`) trained with the official
`flexible`-strategy conditions stacked — `prefix temporal_boundary=2` + `suffix
temporal_boundary=1`, both `probability: 1.0` — everything else identical to exp_050
baseline (rank 32/α32, video-only attention targets, lr 1e-4, 2000 steps, exp_050
`.precomputed` latents + SHDWSMK captions reused verbatim; conditions are applied at train
time, no re-preprocessing). C2V inference for all rungs (base / exp_050 `baseline` /
exp_050 `i2v_ff05` / `c2v`) through the trainer's own `ValidationRunner` used standalone
(`experiments/exp_051_ltx2_lora_c2v_ladder/run_c2v_inference.py`) — identical code path,
seed, and settings across arms; LoRA loaded exactly as the trainer does (PEFT wrap +
`diffusion_model.` prefix strip). Test conditions from a FOREIGN transition family
(earth_wave: the source transition is a rolling dirt wave, endpoints verified clean of it):
ew0 (16:9), ew1 (4:3), ew2 (portrait @ 480×640 per-sample dims). 6 samples/arm =
3 transitions × (trigger / no-trigger prompt).

## Results (visual, frame ladders on all 3 transitions)

- **Ladder ranking**: base = generic frame-filling smoke curtain that never interacts with
  the subject (fire hallucination on ew2). All three LoRA arms transfer the signature
  subject-wrapping ink-black billow to unseen endpoints, overriding the source's dirt-wave
  transition type. `c2v` shows the tightest wrap onset (smoke forms ON the subject), the
  most continuous convergence into the end anchor, and the most stable scene-A hold;
  `t2v` shows mild scene-A drift before transition onset; `i2v_ff05` sits between.
- **Endpoint anchoring is mechanism-robust**: even the pure-t2v LoRA (never saw a
  conditioned token in training) holds both anchors perfectly — `VideoConditionByLatentIndex`
  clean-latent/timestep-0 pinning works regardless of the adapter. Conditioning-matched
  training buys *middle quality* (continuity into anchors), not anchor adherence.
- **c2v learns the concept ~2–4× faster per step**: signature morphology present at
  step 250 (vs 500–1000 for exp_050 t2v). With endpoints pinned clean every step, all
  supervision lands on the transition middle — zero capacity spent reconstructing static
  endpoint content.
- **The trigger token is NOT the concept carrier**: no-trigger prompts (still containing
  the caption phrase "a dense mass of black smoke sweeps across…") produce the full
  morphology in every LoRA arm. Binding lives in the phrase; `SHDWSMK` is at most a mild
  amplifier. Practical: in C2V pipelines, keep the phrase; the token is optional.
- **Specialization cost (drift probe)**: no smoke bleed in any arm, but `c2v`'s
  unconditioned golden-retriever sample shows clear composition shift at fixed seed
  (exp_050 arms left it near-identical to base). The arm never trained an unconditioned
  step (both conditions p=1.0). If one LoRA must also serve plain T2V, train with
  p<1.0 mixes; expect slower concept acquisition in exchange.

## Mechanics worth reusing (validated against trainer code)

- Training prefix/suffix conditions use **latent-frame** units (`temporal_boundary`);
  validation conditions use **pixel-frame** units (`num_frames`, prefix %8==1, suffix
  %8==0). First 2 latents ↔ first 9 px frames; last 1 latent ↔ last 8 px frames (121f
  → 16 latents).
- Multiple intrinsic conditions = independent per-sample Bernoullis, applied
  cumulatively (conditioned tokens: clean latent, timestep 0, excluded from loss).
  A validation sample may carry prefix AND suffix at once (keyframe-style C2V) —
  supported by code, absent from shipped configs.
- **Suffix condition clips must be pre-cut**: the validation runner reads condition
  media from the FRONT (no fps retiming) and the causal VAE's receptive field reaches
  backward — pass a clip of exactly the trailing frames (we used last-9 → 2-latent
  encode, keep last latent via num_frames=8) or earlier content bleeds into the end
  anchor. Prefix is safe by causality (only first N frames encoded).
- Standalone conditioned inference = `ValidationRunner(config, model_path,
  text_encoder_path)` then `.run(transformer, step, output_dir, device, progress)`;
  build the runner BEFORE moving the 19B transformer to GPU (Gemma is freed after
  embedding caching). LoRA load: `get_peft_model` with the exact training-time
  rank/alpha/target_modules, strip `diffusion_model.`, `set_peft_model_state_dict` on
  `get_base_model()`. ltx-pipelines CLIs cannot do multi-frame prefix/suffix (single
  image conditioning only); `KeyframeInterpolationPipeline` is single-frame additive
  guiding — different mechanism.
- W&B: trainer logs validation videos natively (`wandb.enabled` + project); the runner
  accepts `wandb_run` for standalone logging. Project: `creative-transition-transfer`
  (training run `7iptdfyt`, inference runs `exp051_infer_*`).

## Open questions / next steps

- Watch the videos (stills understate motion): confirm c2v's continuity advantage
  temporally; check for any seam at latent boundaries 2 and 15.
- Checkpoint ladder for c2v saved (250…2000) — given step-250 acquisition, an early
  checkpoint may already be the sweet spot (least drift, full concept).
- Head-to-head vs the exp_046/047 injection recipes on identical endpoint pairs — the
  comparison this ladder exists to anchor.
- p=0.5/0.5 prefix/suffix mix arm (all four conditioning combos in one LoRA) if a
  single generalist adapter is wanted.
- Higher-resolution buckets for production texture; quantitative eval still absent.
