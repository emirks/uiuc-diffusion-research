# exp_062 — eval-ladder R2/R3 specialists (11 per-class c2v LoRAs)

## Question

Rungs R2 (held-in endpoints) and R3 (unseen endpoints) of the eval ladder need a
**per-class specialist LoRA**. Does per-class training reproduce the class effect,
and how much does it overfit endpoint content (R2−R3)? This experiment TRAINS the 11
specialists; generation (R2/R3 videos) follows once checkpoints land (PLAN §C1).

## Setup

- **Recipe:** exp_051 c2v verbatim, **sidedness-BLIND** — conditions = prefix
  `temporal_boundary=2` p=1.0 + suffix `temporal_boundary=1` p=1.0 (both always on),
  so supervision falls on the middle 13/16 latent frames. rank 32/α32, video-attention
  targets, lr 1e-4 linear, 2000 steps, bs 1, bf16, ckpt every 250. Base = ltx-2-19b-dev
  + Gemma-3-12B-QAT-q4. See `docs/eval_ladder/PLAN.md` §2 for the one pinned deviation
  from exp_051 (ICTRANS + type-blind captions, for prompt parity across rungs).
- **Roster (11):** shadow, portal, super_fast_run, shadow_smoke, polygon, wireframe,
  animalization, color_rain, gas_transformation, hero_flight, illustration_scene.
  Trained on each class's **split-v1 TRAIN clips** (92 clips total). live_concert is
  excluded (0 test clips). hero_flight is included now — blind conditioning makes it
  immune to the pending sidedness validation.
- **Data:** `build_datasets.py` writes per-class `dataset/manifests/<class>.json`
  (type-blind captions from exp_058 + the 24 freshly captioned held-out-class clips via
  `caption_missing.py`) and `configs/<class>.yaml`. Preprocessing bakes the `ICTRANS`
  trigger and encodes to `dataset/.precomputed/<class>/{latents,conditions}` at
  480×640×121. Each specialist's `data.preprocessed_data_root` points at its own dir.
- **Compute:** `HCESC-H100-secondary` (preemptible; resumable), one array task per class
  (`job_train.sbatch` array 0-10). Each task is self-contained: precompute (idempotent) →
  train (resume-aware, skips DONE / resumes latest ckpt). ~1h19m/task.

## How to run

```bash
cd $LAB/diffusion-research
# 1. captions for held-out classes (login node, idempotent)
python experiments/exp_062_ladder_r2r3_specialists/caption_missing.py
# 2. build per-class manifests + configs
python experiments/exp_062_ladder_r2r3_specialists/build_datasets.py
# 3. submit the 11-way training array, chained twice to survive preemption
J1=$(sbatch --parsable --partition=HCESC-H100-secondary --account=hcesc-h100 --gres=gpu:1 --requeue \
     experiments/exp_062_ladder_r2r3_specialists/job_train.sbatch)
sbatch --dependency=afterany:$J1 --partition=HCESC-H100-secondary --account=hcesc-h100 --gres=gpu:1 --requeue \
     experiments/exp_062_ladder_r2r3_specialists/job_train.sbatch
```

## Outputs

- `outputs/training/exp_062_ladder_r2r3_specialists/<class>/checkpoints/lora_weights_step_00250..02000.safetensors`
  (all steps kept for the R2/R3 checkpoint-sensitivity robustness check), `DONE` marker,
  W&B run under project `eval-ladder-r2r3`.
- `dataset/manifests/<class>.json`, `configs/<class>.yaml`, `dataset/.precomputed/<class>/`
  (latents+conditions, gitignored), `dataset/captions_r2.json`, `dataset/index.json`.
- Slurm logs: `outputs/logs/slurm/exp062_train-<ARRAY>_*.out`.

Generation of R2/R3 videos (both ckpt 250 and 2000, seeds 42/43/44, held-in and unseen
endpoints per `docs/eval_ladder/ladder_items_v1.json`) is deferred to PLAN §C1, run once
checkpoints exist. Scoring stays blocked until the sidedness re-annotation is validated.
