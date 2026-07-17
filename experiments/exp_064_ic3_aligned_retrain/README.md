# exp_064 — ic3: split-aligned IC-LoRA generalist retrain

## Question
Does retraining the exp_058 generalist with training data aligned to split_v1.1
(train band only, owner-final sidedness keying) produce a generalist whose
tier-B (trained class, unseen endpoints) and tier-C (held-out class) numbers
are trustworthy — eliminating ic2's endpoint contamination (12/16 "unseen"
eval items were its training clips) and its giant_grab sidedness mis-key?

Pre-registration: `docs/eval_ladder/PLAN.md` Amendment 2 (§A2.4). ic2 is
retained as a labeled comparison arm (contrast C10 = ic3 − ic2).

## Setup
- Recipe = exp_058 verbatim: LTX-2 19B dev, LoRA rank 32/α32 attn+FFN,
  lr 2e-4, 5000 steps, ckpt/500, bf16, seed 42, 480×640×121@24, single
  per-pair mask condition (prefix 2 latent frames always; +final latent frame
  iff two_sided) + reference concat.
- Data: split_v1.1 TRAIN BAND of the 32 non-holdout classes — 151 clips,
  403 pairs (95 twosided / 308 onesided). Holdout verbatim ic2: hero_flight,
  illustration_scene, gas_transformation, raven, hole, seamless, jump.
- Keying: owner-final taxonomy (corpus_manifest 2026-07-16); vs exp_058 flips
  giant_grab → twosided.
- Precompute reuse: 132/151 clips symlink exp_058's `.precomputed_clips`;
  19 fresh (listed in `data/processed/transitions_std121/dataset_exp064_missing.json`).
- Validation (lora-train directive): exp_058's 3 samples (ID two-sided,
  ID one-sided prefix-only, held-out OOD hero_flight) + a CONTROL sample
  (no ICTRANS trigger, no reference → must NOT produce a transition).

## How to run
```bash
cd $LAB/diffusion-research
python experiments/exp_064_ic3_aligned_retrain/build_dataset.py   # manifests (login-safe)
# then on Slurm (self-contained precompute -> link -> train, chain 2 copies):
sbatch [flags] experiments/exp_064_ic3_aligned_retrain/job_train.sbatch
```

## Outputs
`outputs/training/exp_064_ic3_aligned_retrain/ic3/` — checkpoints every 500,
DONE marker at 5000. Inference consumed by exp_065 (generation grid v3).
