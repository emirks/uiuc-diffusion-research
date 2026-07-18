# exp_068 — Stage 3a: + directionally debiased flow-matching loss

## Question
On top of the residual-reference generalist (exp_067 Stage 2), does the directionally
debiased FM loss (λ∥=0.25 — down-weight velocity-error along the endpoint axis, which
any lerp solution gets for free) improve transition transfer without regressing?

Also serves as the **λ∥ ablation row** (exp_067 λ∥=1 vs this λ∥=0.25 on otherwise
identical residual-reference training).

## Setup
Identical to exp_067 (transition strategy, residual reference γ=1.0922, p_drop 0.1,
5000 steps, inline validation @500 with refswap tripwire) except `lambda_par: 0.25`.
Margin OFF (that is Stage 3b). Trainer branch `transition-strategy` @ 218528b.

## How to run
Launched at the exp_067 mid-run health gate (~step 2500). Parallel with exp_069 on
the second GPU (H100/H200).
```bash
sbatch --partition=secondary --account=campusclusterusers --gres=gpu:H100:1 \
    --time=03:59:00 --requeue experiments/exp_068_s3a/job_train.sbatch
```

## Outputs / eval
`outputs/training/exp_068_s3a/s3a/`. Eval = tier-C SCREEN (63 videos) + inline
validation; full 165-video certified eval only if this is the Stage-3 screen winner.
Screen rule (pre-registered): winner = higher tier-C zero-shot margin; |Δ|<0.02 = tie
→ prefer 3b iff its σ≤0.5 margin activation stayed in-band, else 3a.
