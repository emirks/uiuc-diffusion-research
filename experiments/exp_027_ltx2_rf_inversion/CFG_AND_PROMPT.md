# CFG and prompt handling in exp_027

Supplementary technical note for `exp_027_ltx2_rf_inversion`. Captures the
*conditioning* contract used during Stage-1 generation vs. RF-Solver
inversion + reconstruction, why they differ, and what downstream code
(Step 7 feature injection, Step 8 editing) must respect to keep cached
artefacts valid.

If you only read one line: **inversion runs at CFG = 1 with the positive
prompt active and the negative prompt dropped; the "2 calls per step" in
the README are from the midpoint integrator, not from CFG.**

---

## 1. CFG mechanics (refresher)

Classifier-Free Guidance mixes two velocity predictions per step — one
conditioned on the text prompt, one on the *negative* (or null) prompt —
according to a guidance scale `s`:

```
v_cfg = v_uncond + s · (v_cond − v_uncond)
       = (1 − s) · v_uncond + s · v_cond
```

- `s = 1` → no mixing, equal to `v_cond` (the unconditional path is never
  computed in this branch — see Sec. 5).
- `s > 1` → push the trajectory toward the prompt direction. Costs one
  extra transformer forward per scheduler step.

In diffusers terminology, the flag that gates the two-pass behavior is
`do_classifier_free_guidance = guidance_scale > 1.0`. With `s = 1` the
pipeline encodes only the positive prompt and runs one forward per step.

---

## 2. Stage-1 generation pass (the source-of-truth pass)

Configured in `config.yaml`:

```yaml
inference:
  num_inference_steps: 40
  guidance_scale: 4.0          # s = 4
inputs:
  negative_prompt: "blurry, out of focus, …"
```

What happens inside `LTX2ConditionPipeline.__call__`:

1. `encode_prompt(prompt, negative_prompt, do_classifier_free_guidance=True)`
   → returns **two** embeddings: `prompt_embeds` and `neg_prompt_embeds`.
2. Each scheduler step calls the transformer **twice** — once with each
   embedding — and mixes the velocities via the CFG formula above.
3. Total NFE: 40 steps × 2 = **80 transformer forwards**.

This is the canonical LTX-2 inference recipe and exactly mirrors
exp_020. The latent it produces is what we call `z₀` and treat as
ground-truth source for the inversion gate.

---

## 3. RF-Solver inversion + reconstruction pass

Configured in `config.yaml`:

```yaml
inversion:
  num_steps: 30
  guidance_scale: 1.0          # s = 1
  solver: "rf_solver_midpoint_2nd"
```

Inside `RFInverter.prepare_sample` (`run.py:152-176`):

```python
encode_prompt(
    prompt=prompt,                    # ← positive prompt, encoded
    negative_prompt=None,             # ← negative dropped
    do_classifier_free_guidance=False,
    ...
)
```

What happens at each of the 30 inversion (or reconstruction) steps:

1. `_midpoint_step` calls `_call_transformer` **twice** — at `(z, σ_curr)`
   and at `(z_mid, σ_mid)`. Both calls use the **same positive prompt
   embedding** stored on `self.prompt_embeds`. No second uncond pass, no
   CFG mixing.
2. Total NFE per direction: 30 steps × 2 = **60 transformer forwards**.
   Inversion + reconstruction = 120 NFE total. (The "× 2" is from the
   midpoint integrator, not from CFG — easy to confuse the two.)

The same code path is used for inversion and reconstruction; the only
difference is whether the σ grid is reversed.

---

## 4. The prompt IS used during inversion

A common misread of the inversion code: *"It's CFG = 1, so the prompt
must be ignored."* That's wrong. Look at `_call_transformer`
(`run.py:228-246`):

```python
self.transformer(
    hidden_states=z_in,
    encoder_hidden_states=self.prompt_embeds,    # ← positive prompt active
    audio_encoder_hidden_states=self.audio_prompt_embeds,
    encoder_attention_mask=self.prompt_attn_mask,
    ...
)
```

`encoder_hidden_states` is the positive-prompt embedding produced by
`encode_prompt`. The transformer is conditioned on the prompt at every
step, in both inversion and reconstruction.

What `s = 1` removes is **the second (unconditional) forward** and the
**CFG mixing**, not the prompt itself.

Why we need the prompt active during inversion:
- The transformer was trained with text conditioning. The velocity field
  it predicts when given a real prompt is *different* from the field it
  predicts under empty text — they live on different effective manifolds.
- We want to invert *along the same flow* that produced `z₀`. The flow
  that produced `z₀` had `encoder_hidden_states = prompt_embeds`, so
  inversion must use the same.
- An "unconditional" inversion (passing empty embeddings) would drift
  the round-trip away from `z₀` even with a perfect solver.

---

## 5. Why CFG = 1 for inversion (three independent reasons)

### 5a. CFG-mixed velocity is not conservative

For `s ≠ 1`, the mixed field

```
v_cfg(z, σ) = v_uncond(z, σ) + s · (v_cond(z, σ) − v_uncond(z, σ))
```

is generally **not** the gradient of any potential, and the ODE
`dz/dσ = v_cfg` is not exactly reversible. Empirically, the inversion
error grows roughly linearly-to-quadratically with `s`: the higher the
guidance, the worse the round-trip closes. This observation drove the
entire null-text-inversion / prompt-to-prompt line of work — and every
recent RF-inversion paper (RF-Solver §3, FireFlow §4.1) benchmarks
inversion at `s = 1` by design.

For our LPIPS-gated round-trip there is *no upside* to non-unit `s`.

### 5b. NFE budget

Turning CFG on during inversion doubles the per-step cost: 60 NFE per
direction → 120 NFE per direction, 120 → 240 total. We'd pay 2× compute
for *worse* reconstruction. Strictly Pareto-dominated.

### 5c. Cached trajectories must be reusable downstream

Step 7 (feature injection) reuses `z₁` and the σ-checkpoints saved here.
For those caches to be meaningful — i.e. for re-denoising `z₁` to land
back at `z₀` in any future session — the inversion ODE must be
reversible. CFG-on inversion is not, so caching it would mean caching a
trajectory that doesn't actually round-trip. That defeats the purpose
of the cache.

Editing-time CFG (Step 8) is a separate question, applied *after* the
cached trajectory is in hand — it doesn't need to match the inversion's
guidance setting.

---

## 6. Negative prompt: dropped, why

The negative prompt is only meaningful as the "uncond" side of a CFG
mix. Without that mix, there is nothing for it to do — passing it to a
single conditioned forward would just confuse the prompt. So
`encode_prompt(... negative_prompt=None, ...)` drops it. The negative
prompt from `config.yaml` is used **only** by the Stage-1 generation
call.

---

## 7. CFG is independent of LTX-2's visual conditioning

This is worth separating because the experiment has *two* conditioning
mechanisms running simultaneously, and they do different things:

| Mechanism | What it conditions on | When |
|---|---|---|
| **Text CFG** (`s = 1` here) | Gemma prompt embedding | Generation only |
| **Per-token timestep** `t·(1−mask)` | Clip latents at indices 0 and `N − K` | Generation + inversion + reconstruction |
| **x₀-domain clamp** on velocity | Clip latents at same indices | Generation + inversion + reconstruction |
| **Hard re-clamp** at end of step | Clip latents at same indices | Generation + inversion + reconstruction (defensive) |

The bottom three (visual conditioning) are active at all times,
regardless of CFG scale. They live in `_call_transformer` (per-token
timestep), `_x0_clamp_velocity`, and `_midpoint_step` respectively.
Disabling them would let the conditioned positions drift away from the
start/end clip latents during inversion, and the reconstruction would
no longer match `z₀` at the clip endpoints.

CFG controls **text** conditioning strength. Visual conditioning is
controlled by `start_clip_strength` / `end_clip_strength` in
`config.yaml` and is on regardless of `guidance_scale`.

---

## 8. Implications for downstream steps

**Step 7 (feature injection):** Must invert and re-denoise with the
*same* solver, σ grid, and `guidance_scale = 1` to reuse cached `z₁`
and σ-checkpoints. Changing any of these will break the round-trip
identity that exp_027 validated.

**Step 8 (editing-time CFG):** Once feature injection is in place, you
can crank `guidance_scale` back up at *editing* time. Editing typically
denoises from a partially-noised checkpoint (say `z_{σ=0.5}`) toward
the edited latent, and CFG there sharpens the edit toward the new
prompt. This is independent of the inversion's CFG and does not break
the cache — as long as the cache itself was built with `s = 1`.

**Switching prompts at editing time:** The cache was built with the
*original* prompt embedding active. If at editing time you swap to a
new prompt and re-denoise `z₁` with that new prompt, you'll get a new
latent — but the trajectory will follow a different velocity field than
the one cached, so any feature-injection scheme that assumes "the same
trajectory minus a swapped prompt" must account for that. RF-Solver /
FireFlow editing recipes handle this with prompt-blending or feature
warping at injection time; we have not committed to a specific recipe
in this repo yet.

---

## 9. Pitfalls future Claude / future me should not step on

1. **Do not raise `inversion.guidance_scale` above 1.0 in `config.yaml`
   expecting a sharper reconstruction.** `run.py:572-577` warns about
   this but does **not** implement the second uncond pass — the run
   would still execute at effective CFG = 1, the warning is logged, and
   you'd waste a debug session figuring out why the knob has no
   effect. If you genuinely want a CFG-on inversion (we don't —
   see Sec. 5), you must add a second `_call_transformer` call inside
   `_midpoint_step` with a stored `neg_prompt_embeds`, then mix.
2. **Do not regenerate `z₀` with different `gen_steps`, `gen_cfg`, or
   `seed` and try to reuse an old `z₁`.** The cache identity
   `decode(z₀) ≈ decode(denoise(z₁))` is only valid for the exact
   `(prompt, seed, σ-grid, generator state, scheduler config)` that
   produced it. The generation pass is stochastic in the noise sampling
   only; deterministic given seed.
3. **Do not switch the scheduler between inversion and reconstruction.**
   Both must use the same dynamic-shifted σ grid. `_build_sigma_grid`
   uses `copy.deepcopy(stage1_scheduler)` so calling it multiple times
   is safe, but only because it never mutates the original.
4. **bf16 vs. fp32 boundary.** Solver state runs in fp32 for precision;
   only `z_in` and `t` are cast to bf16 at the transformer call
   boundary (`run.py:217-247`), and `noise_pred_video.float()` casts
   back. If you ever route a solver tensor straight into the
   transformer without `.to(t_dtype)`, you'll get
   `RuntimeError: mat1 and mat2 must have the same dtype` from a deep
   linear layer — that bug was hit and fixed during exp_027's first
   end-to-end run.
5. **σ < 1e-4 short-circuit.** The reversed grid's first inversion step
   evaluates `_x0_clamp_velocity` at `σ_mid = 0.05`, *below* the
   scheduler's `shift_terminal = 0.1`. The clamp formula divides by σ
   and squashes everything to zero at σ → 0 — `_x0_clamp_velocity`
   short-circuits when `σ < 1e-4` and the hard re-clamp at the end of
   `_midpoint_step` keeps conditioned positions pinned regardless. Do
   not "fix" this branch without understanding why it's there.
6. **Audio stream is dummy zeros.** `prepare_sample` builds a
   minimum-valid packed audio tensor at `noise_scale = 0`. It's passed
   to the transformer to satisfy the AV-aware forward signature but
   never integrated. If LTX-2's audio path becomes relevant downstream,
   audio inversion is **not** done here.

---

## 10. References

- RF-Solver paper, Sec. 3.1–3.2, Eq. 9: https://arxiv.org/abs/2411.04746
- FireFlow (midpoint variant, CFG-off inversion benchmark §4.1): https://arxiv.org/abs/2412.07517
- Null-text inversion (original CFG-vs-invertibility analysis): https://arxiv.org/abs/2211.09794
- `LTX2ConditionPipeline` (per-token timestep + x₀-domain clamp pattern):
  `diffusers/pipelines/ltx2/pipeline_ltx2_condition.py:1387–1395`
