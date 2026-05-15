# exp_027 — LTX-2 RF-Solver Flow Inversion

Step 6 of the transition-editing pipeline: take a Stage-1 generated latent z₀ and
invert it back to a noise endpoint z₁ such that re-denoising z₁ with the same
solver recovers z₀ tightly. The cached z₁ + diagnostic checkpoints feed
Step 7 (feature injection from known-good transitions).

## Question

Can RF-Solver (2nd-order midpoint, 30 steps) invert an LTX-2 Stage-1 latent
under our C2V conditioning so that the round-trip is healthy across both
**latent space** (z₀ ≈ z₀_recon) and **pixel space** (decode(z₀) ≈
decode(z₀_recon))?

Gate (latent-space, decode-free):

- `‖z₀_recon − z₀‖₂ / ‖z₀‖₂ < 0.10`
- `cos(z₀_recon, z₀) > 0.99`

Reported (decoded-space): per-frame PSNR (dB), SSIM, LPIPS (AlexNet), and
temporal-consistency flicker `|Δsrc(t,t+1) − Δrec(t,t+1)|`. Per-frame arrays
and the worst-frame index are saved so a single bad frame is debuggable
without re-running.

If the latent gate passes → cache (z₀, z₁, σ-checkpoints) and proceed to Step 7.
If it fails → escalate to 50 steps; if still failing, fall back to 3rd-order RK.

## Method

For each sample (3 DAVIS pairs spanning easy/mid/hard):

1. **Generate** Stage-1 latent via the exp_020 recipe
  (40 steps, CFG=4, FlowMatchEulerDiscreteScheduler with dynamic shift,
   C2V conditioning at indices 0 and `N_lat − K_lat`). Output is packed and  
   normalized to give us **z₀ ∈ ℝ^{1×N×128}**.
2. **Invert** z₀ → z₁ with **RF-Solver midpoint 2nd-order**:
  - σ grid: same 30-step dynamic-shifted grid the scheduler would produce
   for generation, but reversed so σ ascends from σ_min → 1.0.
  - At each step σ → σ_next:
    - `v = transformer(z, σ)` then x₀-domain clamp using the C2V mask
    - `z_mid = z + (σ_next − σ)/2 · v`
    - `v_mid = transformer(z_mid, σ_mid)` then x₀-domain clamp
    - `z_next = z + (σ_next − σ) · v_mid`
  - **CFG = 1** (uncond only). 30 steps × 2 transformer calls = **60 NFE**.
  - Per-token timestep `t·(1−mask)`: conditioned tokens see ~0 diffusion time and
  remain clamped to the clip latents throughout inversion (mirrors how
  `LTX2ConditionPipeline` runs generation).
  - Audio stream passed as zeros; not stepped.
3. **Reconstruct** z₁ → z₀_recon with the **same midpoint solver, reversed σ grid**.
4. **Validate**: run the unified `MetricSuite` (`run.py:480`) on both the
  packed latents (`z₀` vs `z₀_recon`) and the decoded videos:
  - **Primary gate (latent-space)** — `latent_rel < 0.10` AND `latent_cos > 0.99`.
    Decode-free: isolates inversion error from VAE-decode loss.
  - **Reported (decoded-space)** — PSNR (dB), SSIM, LPIPS, temporal flicker.
    Each metric ships per-frame array + worst-frame index in `inv_meta.yaml`.
5. **Escalation**: if either latent condition fails, automatically retry
  inversion+reconstruction at 50 steps. The 3rd-order RK switch is a separate
  config (not auto-triggered).

## Math (RF-Solver Eq. 9, midpoint form)

The 2nd-order Taylor expansion of the rectified-flow ODE with the finite-difference
derivative evaluated at the half-step is algebraically equivalent to the midpoint
method:

```
z(τ+dτ) = z(τ) + dτ · v_θ( z(τ) + (dτ/2)·v_θ(z(τ),τ),  τ + dτ/2 )
```

This is what the [RF-Solver](https://arxiv.org/abs/2411.04746) paper Eq. 9 reduces
to, and matches what [FireFlow](https://arxiv.org/abs/2412.07517) calls the
"standard midpoint" baseline (before the velocity-reuse optimization that we
intentionally **do not use here** — it would couple consecutive steps and make
the cached `z_t` checkpoints meaningless for downstream feature injection).

For inversion the time direction is reversed (σ increases). For generation it
descends. The same solver function handles both modes by passing the σ array
in the desired order.

## LTX-2 schedule caveat

Dynamic shift for our packed-token count N≈6144 clamps near `max_shift=2.05`,
which produces a strongly non-uniform σ grid: 30 steps from σ=1→0 look roughly

```
[1.000, 0.997, 0.994, ..., 0.598, 0.499, 0.350, 0.100, 0.000]
```

Most steps live near σ≈1 (the "coarse" phase) and only a handful near σ≈0
("cleanup"). For inversion this gets reversed — the first inversion step is
a big σ=0→0.1 jump, evaluated at σ_mid=0.05 which is *below* the scheduler's
`shift_terminal=0.1`. The transformer was rarely trained at σ<0.1, so the
velocity at that step is somewhat out-of-distribution. We mitigate by:

- **x₀-clamp short-circuit at σ<1e-4**: keep the model's velocity as-is at
effective σ=0 (the divide by σ otherwise squashes *all* components).
- **Hard re-clamp of conditioned positions** at the end of every midpoint
step keeps C2V tokens pinned to the clip latents regardless of velocity
noise.

If reconstruction misses the LPIPS gate, the 50-step retry tightens these
end-of-grid jumps; if still failing, the 3rd-order RK switch (config knob,
not auto-triggered) is the next escalation.

## Setup

```yaml
samples       : 3 DAVIS pairs (easy / mid / hard)
generation    : 40 steps, CFG=4 (Stage 1 only; no upsample, no Stage 2)
inversion     : 30 steps, CFG=1, midpoint 2nd-order, audio = zeros
gate          : latent_rel < 0.10  AND  latent_cos > 0.99
                (auto-retry at 50 steps if either fails)
reported      : PSNR / SSIM / LPIPS / temporal-flicker — per-frame + summary
```

## How to run

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/envs/diff
cd /workspace/diffusion-research
python experiments/exp_027_ltx2_rf_inversion/run.py
```

## Outputs

```
run_dir/
  {sample_id}/
    z0.pt             # packed clean latent (1, N, 128) bfloat16
    z1.pt             # packed noise endpoint  (1, N, 128) bfloat16
    z_t_25.pt         # σ ≈ 0.25 inversion checkpoint
    z_t_50.pt         # σ ≈ 0.50
    z_t_75.pt         # σ ≈ 0.75
    inv_meta.yaml     # prompt, seed, scheduler config dump, σ grid, NFE,
                      # full metrics block (psnr / ssim / lpips / temporal /
                      # latent — each with per-frame array + worst-frame index),
                      # gate pass/fail + thresholds, retry status
    source_video.mp4  # decode(z₀) — the source-of-truth
    recon_video.mp4   # decode(z₀_recon) — the round-tripped reconstruction
  config_snapshot.yaml
  summary.yaml
  run.log
```

## Deliverable

Per video: LPIPS reconstruction number + cached latents on disk. Ready for
Step 7 (feature injection from cached inverted trajectories).

## Sources

- RF-Solver paper (math derivation): [https://arxiv.org/abs/2411.04746](https://arxiv.org/abs/2411.04746) — Sec 3.1–3.2, Eq. 9.
- LTXTricks reference rectified sampler (LTX-1 sibling): [https://github.com/logtd/ComfyUI-LTXTricks/blob/main/nodes/rectified_sampler_nodes.py](https://github.com/logtd/ComfyUI-LTXTricks/blob/main/nodes/rectified_sampler_nodes.py)
- FireFlow (midpoint variant explanation): [https://arxiv.org/abs/2412.07517](https://arxiv.org/abs/2412.07517)
- LTX-2 condition pipeline (per-token timestep + x₀-clamp pattern): `diffusers.pipelines.ltx2.pipeline_ltx2_condition.LTX2ConditionPipeline`

