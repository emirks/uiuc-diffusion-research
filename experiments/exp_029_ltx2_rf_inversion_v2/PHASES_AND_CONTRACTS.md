# Phase contracts in exp_029

Supplementary note. Captures what every phase passes to the LTX-2
transformer, why, and which invariants downstream code (Step 7
feature injection, Step 8 editing) must respect.

If you only read one paragraph: **invert + reconstruct test solver
self-consistency at CFG=1. Regeneration tests generation-trajectory
recovery at CFG=4. The dual gate makes that distinction explicit.
Audio defaults to literal `torch.zeros` in invert/recon/regen (silent
DAVIS clips → no trajectory to preserve; zeros gives reproducibility
and isolates the video flow). `AudioContextRecorder` runs anyway during
base gen for forensics. See §6 for the alternatives matrix.**

---

## 1. CFG mechanics (refresher)

Classifier-Free Guidance mixes two velocity predictions per step:

```
v_cfg = v_uncond + s · (v_cond − v_uncond)
       = (1 − s) · v_uncond + s · v_cond
```

- `s = 1` → no mixing, equal to `v_cond`. Single transformer call per step.
- `s > 1` → push trajectory toward prompt. Two transformer calls (cond + uncond), batched.

`do_classifier_free_guidance = guidance_scale > 1.0`.

---

## 2. Phase × CFG matrix


| Phase              | Solver   | CFG | Calls / step | Prompt(s) used      | Audio (default `zeros`)                                             | Audio (`capture_and_replay`)  |
| ------------------ | -------- | --- | ------------ | ------------------- | ------------------------------------------------------------------- | ----------------------------- |
| Generation (a)     | Euler    | 4.0 | 1 batched(2) | positive + negative | pipeline init: `noise_scale=σ_max`, stepped (not under our control) | same — captured for forensics |
| Inversion (b)      | Midpoint | 1.0 | 2 (midpoint) | positive only       | `torch.zeros` every step                                            | replay reversed               |
| Reconstruction (c) | Midpoint | 1.0 | 2 (midpoint) | positive only       | `torch.zeros` every step                                            | replay forward                |
| Regeneration (d)   | Euler    | 4.0 | 1 batched(2) | positive + negative | `torch.zeros` every step                                            | replay forward                |


"Calls / step" for midpoint = the two probes (z, σ_curr) and (z_mid, σ_mid).
For Euler+CFG = the batched cond + uncond pair.

---

## 3. The prompt IS used during inversion

CFG=1 does **not** mean "no prompt." `encode_prompt(prompt, negative_prompt=None, do_classifier_free_guidance=False)` returns the positive embedding, which is
fed into the transformer at every step. What CFG=1 removes is the second
uncond pass and the mixing, not the conditioning itself.

If you swapped the positive prompt for an empty string during inversion,
you'd be inverting along an entirely different velocity field (the "null
conditional" flow) and round-trip drift would balloon. Don't do it.

---

## 4. Why CFG=1 for invert + reconstruct (three independent reasons)

### 4a. CFG-mixed velocity is not conservative.

For `s ≠ 1` the mixed field is not the gradient of any potential; the ODE
`dz/dσ = v_cfg` is not exactly reversible. Inversion error grows roughly
quadratically with `s`. Every recent RF-inversion paper benchmarks
inversion at `s = 1` for this reason.

### 4b. NFE budget.

CFG-on inversion doubles transformer calls. 60 NFE → 120 NFE per direction
for *worse* reconstruction. Strictly dominated.

### 4c. Cached trajectories must round-trip.

Step 7 will re-denoise z₁ to land back at z₀. That identity only holds
along a reversible flow.

---

## 5. Why regenerate uses CFG=4 (the test that matters)

The CFG=1 round-trip in (b)+(c) is a **necessary** condition for
inversion to be useful — it tells us the midpoint solver itself is
self-consistent. It is **not sufficient**: z₀ was produced by a CFG=4
flow, so the meaningful claim is that re-running the same CFG=4 flow
from z₁ recovers z₀. That's what `regenerate` does.

If `inv_recon` passes but `inv_regen` fails: the solver is fine, but
CFG=1↔CFG=4 mismatch is hurting us. Mitigation candidates: cache the
CFG=4 trajectory's actual noise pattern, or run inversion under an
approximation of the CFG=4 flow (null-text inversion, etc.).

If both fail: the solver isn't tight enough at this step count.
Escalate to 60 steps; then 3rd-order RK; then revisit.

---

## 6. Audio context

### 6.1 What the pipeline does

LTX-2 is jointly audio + video. `LTX2ConditionPipeline` initializes
audio latents at `noise_scale = sigmas[0] ≈ 1.0` (pure Gaussian) and
runs `audio_scheduler.step(noise_pred_audio, t, audio_latents)` every
iteration. So during base generation the transformer's
`audio_hidden_states` evolves from N(0, I) → "denoised audio that
plausibly accompanies the video."

### 6.2 Where audio enters our control surface

We pass `audio_hidden_states` **directly as a kwarg to
`pipe.transformer(...)`** at the **audio-VAE latent level** (already
normalized). Not raw waveform, not mel spectrogram, not pre-VAE.
Concretely, for shape `[1, audio_channels, audio_num_frames, mel_bins]`:
audio cross-attention's `V·audio_hidden_states` contributes that tensor
(after a learned projection) additively to video tokens.

### 6.3 Default: `torch.zeros`

Three reasons:

1. **DAVIS clips are silent.** No real trajectory exists to invert.
2. **Step 7 reuses `z₁` in a fresh `pipe(...)` call** with its own audio
  init — pinning inversion to gen's specific roll over-fits.
3. **Reproducibility + isolation.** Zero output of `V·0 = 0` removes audio
  as a variable; a failed gate unambiguously points to the video flow.

### 6.4 Why regen uses zeros, not "natural" audio

`inv_regen` is a diagnostic. If invert+recon use zeros and regen uses
natural noisy-stepped audio, a regen failure conflates two causes:
(a) inversion drift, (b) audio-context shift between phases. Holding
audio constant across all three under-our-control phases isolates (a).
The "does z₁ work in a real production `pipe(...)` call?" question is
separate; could be added later as a `regen_natural` diagnostic.

### 6.5 The gen↔others asymmetry we accept

Base gen (a) is `pipe(...)` — uncontrolled — and always uses noisy-stepped
audio. So gen sees noisy audio while invert/recon/regen see zeros. exp_027
evidence: samples 1+2 round-tripped at `rel ≈ 0.017` despite the mismatch.
The video transformer tolerates audio context shifts for silent-clip
training data. Forcing zeros inside `pipe(...)` would drift `z₀` away from
what a real generation produces — strictly worse.

### 6.6 Alternatives considered


| Option               | Audio_hidden_states value                    | Pro                                 | Con                                                     | Status                 |
| -------------------- | -------------------------------------------- | ----------------------------------- | ------------------------------------------------------- | ---------------------- |
| `zeros` (default)    | `torch.zeros(...)`                           | reproducible, V·0=0, isolates video | possibly OOD (model never saw exact-zero audio latents) | **shipping**           |
| encoded silence      | `normalize(audio_vae.encode(zero_waveform))` | in-distribution, deterministic      | non-zero contribution, harder to verify                 | not shipped            |
| `capture_and_replay` | record gen's audio, replay reversed/forward  | strict round-trip identity          | over-fits to one gen-audio roll                         | opt-in (ablation only) |
| random per step      | `randn` each call                            | matches gen's stochasticity         | non-reproducible                                        | rejected               |


### 6.7 What exp_027 actually did (bug)

`prepare_audio_latents(noise_scale=0.0, latents=None)` looks like it
returns zeros but doesn't: `_create_noised_state(latents=randn, 0, _) = 0·new_randn + 1·randn = randn`. exp_027 thus used a fixed-random tensor
that varied between runs (`generator=None`). v2 uses literal `torch.zeros`
— reproducible and truly inactive.

### 6.8 Instrumentation

`AudioContextRecorder` wraps `pipe.transformer.forward` during base gen
and snapshots `audio_hidden_states` per call. Saved as
`audio_record.pt` (~2 MB / sample, bf16). Forensic only — not fed back
into invert/recon/regen unless `audio_strategy: capture_and_replay`.

### 6.9 When to switch defaults

Move to `capture_and_replay` if processing clips with **real audio** the
user wants preserved. Move to encoded silence if `inv_regen` shows
systematic drift attributable to audio OOD-ness (no evidence yet).

---

## 7. Per-token visual conditioning is independent of CFG

This is worth separating because there are *two* conditioning mechanisms
running simultaneously, and they do different things:


| Mechanism                           | What it conditions on             | When active                                            |
| ----------------------------------- | --------------------------------- | ------------------------------------------------------ |
| **Text CFG** (= 1 / = 4 phase-dep)  | Gemma prompt embedding            | All phases; mixing only when CFG > 1                   |
| **Per-token timestep** `t·(1−mask)` | Clip latents at indices 0 and N−K | Generation + inversion + reconstruction + regeneration |
| **x₀-domain clamp** on velocity     | Clip latents at same indices      | All phases                                             |
| **Hard re-clamp** at end of step    | Clip latents at same indices      | All phases (defensive)                                 |


The bottom three (visual conditioning) are always on. They are
controlled by `start_clip_strength` / `end_clip_strength`, **not** by
`guidance_scale`. Setting `start_clip_strength=0.5` would partly relax
the C2V tokens; `guidance_scale=1.0` would not.

---

## 8. Step count alignment (Fix #1)

In exp_027 inversion ran at 30 steps while generation ran at 40. The
LTX-2 σ grid uses dynamic shift (`use_dynamic_shifting=true`,
`base_shift=0.95`, `max_shift=2.05`), and the resulting σ samples are
strongly non-uniform AND step-count-dependent: 30 steps and 40 steps
visit *different* points on [0, 1].

The midpoint solver, applied at a different sample of σ values than
the forward Euler used during generation, is solving a
discretization-mismatched problem. Round-trip error from this source
alone can dominate the rest.

Fix is mechanical: `inversion.num_steps = 40`. Inversion now samples σ
at exactly the points generation visited (in reverse order). Same fix
for retry: 50 → 60 to escalate by 50 % (not exactly 1.5× — see
`config.yaml` if you change it).

---

## 9. Pitfalls (carried over from exp_027 + new)

1. **Do not raise `inversion.guidance_scale` above 1.0 in
  `config.yaml`** expecting a sharper reconstruction. The code logs a
   warning and runs at effective CFG=1 anyway (no second uncond pass
   in the midpoint loop). Use `regeneration.guidance_scale` to test
   CFG > 1 trajectory recovery.
2. **Do not switch `regeneration.solver`** from `euler` without
  updating `regenerate()` in run.py. The current code assumes Euler.
3. `**sample_mode="argmax"**` in the `retrieve_latents` call for C2V
  posterior is **deterministic** (returns posterior mode). Switching
   to `"sample"` would silently make C2V conditioning latents
   stochastic — round-trip identity breaks. The run.py call site has
   an explicit comment.
4. **σ < 1e-4 short-circuit** in `_x0_clamp_velocity` is intentional.
  At the reversed grid's first inversion step the half-step probe
   evaluates at σ_mid = 0.05, below `shift_terminal = 0.1`. The clamp
   formula's `/ σ` term squashes everything to zero there. The hard
   re-clamp at end of step keeps conditioned positions pinned
   regardless. Do not "fix" without understanding why it's there.
5. **bf16 / fp32 boundary**: solver state runs in fp32 for precision;
  tensors cast to bf16 only at the transformer call boundary
   (`_call_transformer`). Stripping the `.float()` at the
   `noise_pred_video.float()` line would silently degrade precision
   on the velocity used by the midpoint update.
6. **Audio CFG batch duplication**: when `regenerate` runs at CFG=4 it
  feeds `cat([audio, audio])` as `audio_hidden_states`. The captured
   trajectory stores *single* batches (first half of generation's
   batched audio, since both halves are identical at capture time).
   `_call_transformer` duplicates the captured tensor when CFG > 1.
   If you bypass that path, the transformer will see a single-batch
   audio context with a 2-batch video — runtime error.

---

## 10. Implications for downstream steps

**Step 7 (feature injection)** must:

- Use the *same* solver, σ grid, CFG, prompt, and seed as exp_029's
regeneration phase when re-denoising z₁.
- Replay the captured `audio_record.pt` trajectory at the same step
indices to match the cross-attention context.
- Validate against the cached z₀ before injecting features.

**Step 8 (editing-time CFG)** can crank `guidance_scale` back up at
*editing* time once feature injection is in place — that's separate from
the inversion's CFG and does not break the cache.

**Swapping prompts at edit time** changes the velocity field. Any feature
injection scheme that assumes "same trajectory minus a swapped prompt"
must explicitly account for that — prompt blending or feature warping at
injection time. Not committed to a specific recipe yet.

---

## 11. References

- RF-Solver paper, Sec. 3.1–3.2, Eq. 9: [https://arxiv.org/abs/2411.04746](https://arxiv.org/abs/2411.04746)
- FireFlow (midpoint variant, CFG-off benchmark §4.1): [https://arxiv.org/abs/2412.07517](https://arxiv.org/abs/2412.07517)
- Null-text inversion (CFG-vs-invertibility analysis): [https://arxiv.org/abs/2211.09794](https://arxiv.org/abs/2211.09794)
- `LTX2ConditionPipeline` denoising loop:
`diffusers/pipelines/ltx2/pipeline_ltx2_condition.py:1339-1398`
- Audio prep + scheduler step:
`pipeline_ltx2_condition.py:1184-1185, 1260-1270, 1402` (audio init
noise_scale, prepare_audio_latents call, audio_scheduler.step)

