# exp_029 — LTX-2 RF-Solver Flow Inversion (strict-consistency v2)

Fork of exp_027 that closes six methodology gaps surfaced after the first
end-to-end run. The v2 designation is non-cosmetic: the **dual gate** and
**audio capture-and-replay** change what passing means.

## Why a v2

exp_027 produced 2/3 PASS at the first attempt (mallard, car-roundabout)
and FAIL on blackswan even after escalating to 50 steps. Looking at the
methodology with fresh eyes, six issues turned out to be load-bearing:

| # | Issue (exp_027)                                       | Fix (exp_029)                                                    |
|---|-------------------------------------------------------|------------------------------------------------------------------|
| 1 | Inversion uses 30 steps; generation uses 40 steps     | `inversion.num_steps: 40` — match the σ grid exactly             |
| 2 | CFG=1 round-trip ≠ generation-trajectory recovery     | Documented + new `regenerate` phase (Fix #3)                     |
| 3 | No CFG-on test that z₁ actually re-generates z₀       | New `regenerate` phase — Euler + CFG=gen_cfg from z₁ → z₀_regen  |
| 4 | exp_027's "zeros" was actually randn (bug); no audio instrumentation | True `torch.zeros` default + `AudioContextRecorder` capture for forensics. Capture-and-replay kept as opt-in. |
| 5 | Step logs are scalar `‖z‖`; insufficient for failure debug | Per-step CSV: v_norm raw/clamped, z_norm cond/free split, σ, dt  |
| 6 | `retrieve_latents` argmax determinism implicit       | Explicit comment block in run.py                                 |

## Question

After fixing the six gaps above, does the **dual gate** still hold for the
DAVIS triplet (easy / mid / hard)?

- **inv_recon** (solver self-consistency): `‖z₀ − z₀_recon‖ / ‖z₀‖ < 0.10` AND `cos(z₀, z₀_recon) > 0.99`.
  Tests that **midpoint(invert) ∘ midpoint(reconstruct)** ≈ identity at CFG=1.
- **inv_regen** (generation-trajectory recovery): `‖z₀ − z₀_regen‖ / ‖z₀‖ < 0.20` AND `cos > 0.97`.
  Tests that **Euler(z₁, CFG=4, 40 steps)** ≈ z₀ — i.e., the inverted z₁
  is actually a valid noise endpoint for the generation flow that
  produced z₀.

A sample passes only if **both** are within their thresholds.

## Method

Per sample, four phases run sequentially:

### (a) Stage-1 generation — source of truth (40 steps, CFG=4)
Identical to exp_027 + exp_020 recipe. `AudioContextRecorder` wraps
`pipe.transformer.forward` and snapshots `audio_hidden_states` at each
call into `audio_record.pt` (~2 MB / sample) — pure forensics; not fed
back into invert/recon/regen unless `audio_strategy: capture_and_replay`
is opted in. See `PHASES_AND_CONTRACTS.md` §6.

### (b) Inversion — RF-Solver midpoint 2nd-order (40 steps, CFG=1)
z₀ → z₁ via ascending σ grid (reverse of generation grid). Audio defaults to
`torch.zeros` for every step (see "Audio strategy" below).

### (c) Reconstruction — same midpoint solver (40 steps, CFG=1)
z₁ → z₀_recon via descending σ. Audio = zeros, same as inversion. The pair
(b)+(c) is the exp_027 round-trip test — now numerically tighter due to
the 40-step grid and the true-zeros audio context (no hidden random state).

### (d) Regeneration — Euler + CFG (40 steps, CFG=4) — **new in v2**
z₁ → z₀_regen using the exact pipeline forward path: Euler integration
(matches `pipeline_ltx2_condition.py` line 1339-1398), CFG=4 with positive
+ negative prompt. Audio = zeros, matching invert/reconstruct so we test
the same flow we inverted. This is the pipeline forward Step 7 will
actually run (modulo the real Step 7 also re-rolling audio, which we
empirically know doesn't change video much).

### Audio strategy (default: `zeros`)

**What we do.** Inject `torch.zeros((1, audio_channels, audio_num_frames, mel_bins))`
**directly into the transformer's `audio_hidden_states=` kwarg** for every
call in invert / reconstruct / regen. This is the audio-VAE latent level
(already normalized) — not raw waveform, not mel spectrogram. Audio
cross-attention's `V·zeros = 0` → zero contribution to video tokens.

**Why zeros — three independent reasons.**
1. **DAVIS clips are silent.** No real audio trajectory exists to invert.
2. **Step 7 reuses `z₁` in a fresh `pipe(...)` call** with its own random
   audio init — locking inversion to base-gen's audio roll over-fits.
3. **Reproducibility + isolation.** Zeros is deterministic across runs and
   eliminates audio as a variable, so a failed gate unambiguously points
   to the video flow.

**Why regen also uses zeros, not "natural" audio.** `inv_regen` is a
diagnostic, not a production trace. If invert + recon use zeros and regen
uses noisy-stepped audio, a regen failure could mean (a) inversion drift
OR (b) audio-context shift between phases. Keeping audio consistent across
all three under-our-control phases isolates (a). The "does z₁ work in a
real `pipe(...)` call?" question is separate and could be added later as
a third `regen_natural` diagnostic.

**Asymmetry we accept.** Base gen (a) is `pipe(...)` — uncontrolled — and
always inits audio at `noise_scale = sigmas[0] ≈ 1.0` and steps it. So gen
sees noisy-stepped audio while other phases see zeros. exp_027 shows this
asymmetry is tolerable: samples 1+2 round-tripped at `rel ≈ 0.017`.

**Alternatives we considered but didn't ship.**
- *Encoded silence* (`audio_vae.encode(zero_waveform) → normalize`): more
  in-distribution than zeros but harder to verify and gives non-zero
  cross-attention contributions. Worth trying if zeros turns out to be OOD.
- *Capture-and-replay* (record gen's audio, replay reversed/forward):
  implemented as `audio_strategy: capture_and_replay` for ablation only.
  Wrong default because it ties z₁ to one specific audio roll.
- *Random noise per step*: matches gen's stochasticity but is
  non-reproducible. Pure downside.

**Footgun fixed.** exp_027 called `prepare_audio_latents(noise_scale=0, latents=None)`
expecting zeros; `_create_noised_state` actually returns
`0·new_randn + 1·initial_randn = randn`, so exp_027 used a fixed-random
tensor that varied across runs. v2 uses literal `torch.zeros`.

**Forensics.** `AudioContextRecorder` wraps `pipe.transformer.forward`
during base gen and saves the captured trajectory to `audio_record.pt`
(~2 MB / sample, bf16). Not used unless `audio_strategy: capture_and_replay`
is set in config.

`PHASES_AND_CONTRACTS.md` §6 has the long-form discussion.

### (e) Validate — dual gate (informational)
`MetricSuite` evaluates both pairs:

- `(z₀, z₀_recon)` → `inv_recon` metrics
- `(z₀, z₀_regen)` → `inv_regen` metrics

The dual gate is **logged but does not retry**. The dominant gate-failure
mode (`regen_rel ≫ ceiling`) reflects the CFG=1 invert ↔ CFG=gen regen
flow mismatch by design, not a fixable solver-quality problem. Decoded-
space metrics (PSNR / SSIM / LPIPS / temporal flicker) are reported per-
frame for debugging but do not gate.

## Math (unchanged from exp_027)

RF-Solver midpoint 2nd-order:
```
z(τ+dτ) = z(τ) + dτ · v_θ( z(τ) + (dτ/2)·v_θ(z(τ),τ),  τ + dτ/2 )
```
With per-token timestep `t·(1−mask)` and x₀-domain clamp on velocity at
every transformer call. See [RF-Solver](https://arxiv.org/abs/2411.04746)
Sec. 3, Eq. 9.

Regeneration uses Euler (one transformer call per step, CFG-mixed):
```
z(σ_{i+1}) = z(σ_i) + (σ_{i+1} − σ_i) · v_clamped(z(σ_i), σ_i)
```
This is precisely `LTX2ConditionPipeline.__call__`'s denoising loop.

## Setup

```yaml
samples       : 3 DAVIS pairs (easy / mid / hard)
generation    : 40 steps, CFG=4    (audio: pipeline init noise_scale=1, stepped)
inversion     : 40 midpoint, CFG=1 (audio: torch.zeros)
reconstruction: 40 midpoint, CFG=1 (audio: torch.zeros)
regeneration  : 40 Euler,    CFG=4 (audio: torch.zeros — matches invert/recon)
dual gate     :  (informational only — logged, no retry)
  inv_recon : latent_rel < 0.10  AND  latent_cos > 0.99
  inv_regen : latent_rel < 0.20  AND  latent_cos > 0.97
```

## How to run

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/envs/diff
cd /workspace/diffusion-research
python experiments/exp_029_ltx2_rf_inversion_v2/run.py
```

## Outputs

```
run_dir/
  {sample_id}/
    z0.pt                              # packed clean latent (1, N, 128) bfloat16
    z1.pt                              # packed noise endpoint (1, N, 128) bfloat16
    z0_recon.pt                        # CFG=1 midpoint recon
    z0_regen.pt                        # CFG=gen Euler regen
    z_t_25.pt / z_t_50.pt / z_t_75.pt  # inversion checkpoints
    audio_record.pt                    # captured audio trajectory (Fix #4 forensics)
    source_video.mp4                   # decode(z₀)
    recon_video.mp4                    # decode(z₀_recon)
    regen_video.mp4                    # decode(z₀_regen)
    step_diag_invert_n40.csv           # per-step diagnostics (Fix #5)
    step_diag_reconstruct_n40.csv      # ditto
    step_diag_regenerate_n40_n40.csv   # ditto for regen
    inv_meta.yaml                      # metrics_recon + metrics_regen + dual-gate status
  config_snapshot.yaml
  summary.yaml
  run.log
```

## Step-diagnostics CSV schema

One row per outer step. Empty columns are legitimate (midpoint rows use
all columns; Euler rows leave `sigma_mid`, `v_mid_*`, `z_mid_norm` blank).

| Column              | Meaning |
|---------------------|---------|
| `phase`             | invert / reconstruct / regenerate |
| `step_idx`          | 0-based, within phase |
| `sigma_curr`        | σ before the step |
| `sigma_next`        | σ after the step |
| `sigma_mid`         | midpoint σ (only for midpoint phases) |
| `dtau`              | σ_next − σ_curr (signed) |
| `v_norm_raw`        | ‖v‖ before x₀-clamp at σ_curr |
| `v_norm_clamped`    | ‖v‖ after x₀-clamp at σ_curr |
| `v_mid_norm_raw`    | ‖v_mid‖ raw (midpoint only) |
| `v_mid_norm_clamped`| ‖v_mid‖ clamped (midpoint only) |
| `z_in_norm`         | ‖z‖ at step start |
| `z_mid_norm`        | ‖z_mid‖ (midpoint only) |
| `z_next_norm`       | ‖z‖ at step end |
| `z_cond_norm`       | ‖z[conditioned positions]‖ — should remain ≈ ‖clean_latents‖ |
| `z_free_norm`       | ‖z[free positions]‖ — drives solver error |
| `x0_pred_norm`      | ‖z − v·σ‖ (pre-clamp x₀ estimate) |
| `dt_s`              | wall-clock time for the step |

A failing sample is usually visible as `v_norm_raw` spiking or
`z_free_norm` overshooting the gen-time `z_free_norm` trajectory.

## Deliverable

Per video: dual-gate verdict + cached (z₀, z₁, σ-checkpoints, audio_record).
Both gates must pass for Step 7 (feature injection from cached
trajectories) to be safe to use this sample.

## Sources

- RF-Solver paper: <https://arxiv.org/abs/2411.04746>
- FireFlow (midpoint variant): <https://arxiv.org/abs/2412.07517>
- LTX-2 condition pipeline reference impl:
  `diffusers/pipelines/ltx2/pipeline_ltx2_condition.py` (line 1339-1398 is
  the Euler+CFG+x₀-clamp loop `regenerate` mirrors).
