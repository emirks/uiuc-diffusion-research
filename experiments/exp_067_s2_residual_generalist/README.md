# exp_067 — Stage 2: residual-reference IC-LoRA generalist

## Question
Does replacing the raw in-context reference latents with their residual field
(endpoint-content-subtracted, RMS-normalized to corpus γ=1.0922) improve zero-shot
transfer — the C6 failure of ic3 (held-out-class margin 0.038 vs conditioned-base
0.175) — without giving up ic3's in-class margin parity and non-copy synthesis?

Pre-registered bar (advised campaign, Round-1 verdict, set before any results):
PASS iff zero-shot (tier-C) margin improves over ic3 by ≥ MDE 0.037 AND ID margin
stays within MDE of ic3 AND near-copy stays ~3% AND seam not degraded.
Interpretation guard: a pass means "residual reference moves zero-shot", not
"zero-shot solved" — frame against the conditioned-base anchor 0.175.

## Setup
- Recipe = exp_064 ic3 verbatim (LTX-2 19B dev, LoRA rank32/α32 attn1+attn2+ff,
  lr 2e-4 linear, 5000 steps, bf16, seed 42, same split_v1.1 train band / 403 pairs,
  same mask conditioning) with exactly TWO method deltas:
  1. `training_strategy.name: transition` + `residual_reference: true` (γ=1.0922) —
     references fed as residual fields (trainer branch `transition-strategy`
     @ 218528b on $LAB/LTX-2-official; base 7809842; patch sha in campaign dossier).
  2. reference `probability: 0.9` — p_drop 0.1, token-removal semantics (trains the
     no-reference branch for collapsed-teacher guidance later).
  λ∥=1, margin OFF — this stage isolates the reference representation only.
- Inline validation (lora-train directive): 5 samples every 500 steps, initial
  validation ON, fixed seed: [0] ID two-sided (ref shadow_smoke_3), [1] ID one-sided
  prefix-only, [2] OOD held-out-class reference (hero_flight — live C6 probe),
  [3] control (no trigger/no ref), [4] ref-swap partner of [0] (identical except
  reference = water_element_1). `refswap_pair: [0, 4]` → runner logs pre-decode
  interior latent RMS diff to samples/refswap.jsonl.
  TRIPWIRE (pre-registered): rel_to_gamma < 0.02 at BOTH step 2000 and 2500 →
  REFERENCE-DEAD flag → eyeball videos → if confirmed, halt + diagnose before any
  downstream launch.
- Mid-run gate (~step 2500): ref-swap alive + losses sane → Stage-3 runs
  (exp_068/069) launch the same evening without waiting for this eval.

## How to run
```bash
cd $LAB/diffusion-research
sbatch --partition=HCESC-H100-normal --account=hcesc-h100 --gres=gpu:H100:1 \
    experiments/exp_067_s2_residual_generalist/job_train.sbatch
# (or --partition=secondary --account=campusclusterusers with 2 chained copies)
```

## Outputs
`outputs/training/exp_067_s2_residual_generalist/s2/` — checkpoints every 500,
validation samples + refswap.jsonl, DONE marker. Eval: fork exp_065
`manifest_ic3.json` (adapter → step_05000) for the full 165-video certified abc
eval; scoring via exp_066 against the eval/v3.0.0 checkout.
