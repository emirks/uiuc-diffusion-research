# exp_069 — Stage 3b: debiased FM + calibrated anti-collapse margin

## Question
On top of debiased FM (Stage 3a), does the calibrated anti-collapse margin — a hinge
on segment-orthogonal deviation D⊥(x̂) vs (1−m)·D⊥(Z), ω=(1−σ)²-weighted,
identifiability-masked — further improve transition transfer, especially the zero-shot
(tier-C) margin?

Margin survived the pre-registered probe gate (Probe A: D⊥ is a valid collapse detector,
clean axis declerp<hold<lerp<base·P<base·PE<GT≈ic3; base·PE median ≤ GT Q1). Curvature
was KILLED by the same gating and is not included.

## Setup
Identical to exp_068 (λ∥=0.25, residual reference) plus `margin_enabled: true`,
`lambda_margin: 0.3` (retuned ×3 for the shifted sigma sampler; E[ω]≈0.09),
`margin_slack: 0.1`, `margin_start_step: 1000` (ramp from 20% of 5000 steps).
Online guard (pre-registered): σ≤0.5 margin-activation fraction must stay in [0.1, 0.8];
sustained ≈1.0 → indiscriminate-push failure → margin killed, rest kept. Logged as
`train/margin_active_lowsigma`. Trainer branch `transition-strategy` @ 218528b.

## Pre-run gate
The aux:FM grad-norm ratio check (σ∈{0.1,0.3,0.6} must fall in [0.05,0.5]; if above at
low σ, halve λ_margin once and record) runs before launch — see eval/grad_ratio_check.py.

## How to run
Launched at the exp_067 mid-run health gate, parallel with exp_068.
```bash
sbatch --partition=secondary --account=campusclusterusers --gres=gpu:H100:1 \
    --time=03:59:00 --requeue experiments/exp_069_s3b/job_train.sbatch
```

## Outputs / eval
`outputs/training/exp_069_s3b/s3b/`. Eval = tier-C SCREEN + inline; full certified eval
only if Stage-3 screen winner (rule in exp_068 README).
