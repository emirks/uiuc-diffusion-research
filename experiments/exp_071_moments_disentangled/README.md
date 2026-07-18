# exp_071 — Disentangled style channel (moment-bottleneck reference)

## Question
Does replacing the residual reference with its DISENTANGLED moment representation — fixed
noise carrying only the per-(channel,frame) spatial mean+std of the residual, spatial layout
destroyed — retain transition-transfer quality (margin) while making reference-CONTENT import
structurally impossible? I.e., is the transferable style carried by a low-dimensional
temporal statistics program rather than by reference appearance?

Motivation: the residual (exp_067) still retains ~80% of the reference's appearance energy
and visibly imports reference CONTENT on zero-shot (Stage-2 hero_flight validation: city/fist
POV from the reference). Probe B1 (CPU): channel-moment descriptors identify the style class
at leave-one-out top-1 0.528 (chance 0.031) — BETTER than the full residual (0.483) — while
destroying spatial layout 100%. So the style signal survives; this run tests whether the
adapter can perform it.

## Setup
Identical to exp_067 (transition strategy, residual_reference on, γ=1.0922, p_drop 0.1,
λ∥=1, margin off, 5000 steps, inline validation @500 with refswap tripwire) EXCEPT
`reference_mode: moments`. Energy-exact construction (same sigma_s → gamma unchanged) makes
this a strict single-variable ablation vs exp_067: the ONLY difference is spatial-layout
destruction. Trainer branch `transition-strategy` @ f062984 (base 7809842). Training carrier
noise is re-sampled every step (seed=None) — free augmentation, no memorization of the noise.

## Pre-registered bar (disentanglement fable, 2026-07-18; docs/transition_method/PRE_REGISTRATION.md)
ADOPT (as the disentangled primary) iff: tier-C margin ≥ (exp_067 tier-C margin − MDE 0.037)
[NON-INFERIORITY — content-free at equal margin is strictly better science] AND median
content-leak metric < exp_067's AND ID-tier margin within MDE AND seam/copy/cam_dtw guards
pass AND ref-swap tripwire alive (swapped ref → different moment program → output must change).
Superiority on zero-shot is upside, not the bar. The content-removal LADDER raw(ic3) →
residual(exp_067) → moments(this) — zero-shot margin AND leak plotted per rung — is the
paper's disentanglement figure.

## How to run
```bash
sbatch --partition=secondary --account=campusclusterusers --gres=gpu:H100:1 \
    --time=03:59:00 --requeue experiments/exp_071_moments_disentangled/job_train.sbatch
```

## Outputs / eval
`outputs/training/exp_071_moments_disentangled/moments/`. Eval reuses exp_067's fork with
`--reference-mode moments` (generation MUST residualize in moments mode to match training).
Free inference-time mechanism ablation: feed a temporally-shuffled moment program → margin
should drop ≥ MDE (proves the model reads the temporal program, not static stats).
