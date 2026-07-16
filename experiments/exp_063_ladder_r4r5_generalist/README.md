# exp_063 — eval-ladder R4/R5 (generalist IC-LoRA generation)

## Question

Rungs R4 (generalist on TRAINED classes, unseen endpoints) and R5 (generalist on
HELD-OUT classes, reference-only zero-shot) of the eval ladder. Uses the existing
exp_058 `ic2` step_05000 adapter — no new training. R3−R4 measures specialist-vs-
generalist interference (contamination-stratified, PLAN §C5); R5 is zero-shot transfer.

## Setup

- **Adapter:** exp_058 `ic2` `lora_weights_step_05000.safetensors` (rank 32/α32,
  attn+FFN targets), run in its **native keyed mode** — suffix condition ON iff the
  row is two_sided. Keying source: R4 = exp_058 training sidedness; R5 = validated
  taxonomy label (PLAN §D5). sha256 pinned in `dataset/ladder_r4r5.json`.
- **Targets:** each class's 2 split-v1 test clips (== exp_061 items → prompt parity,
  and their 9-frame cond cuts already exist in `experiments/exp_061_ladder_r0_r1/dataset/cond`).
  **Reference** = the fixed per-class clip from `docs/eval_ladder/ladder_items_v1.json`
  (R4 refs are ic2-trained clips = native; R5 refs are zero-shot).
- **Coverage:** R4 = 8 classes × 2 clips = 16 targets; R5 = gas_transformation,
  illustration_scene × 2 = 4 targets **now**; hero_flight × 2 = 6 videos **DEFERRED**
  (its two_sided key waits on sidedness validation, PLAN §B1). Seeds 42/43/44 →
  20 items × 3 = **60 videos now**, + 6 hero_flight later.
- **Recipe:** 480×640×121@24, 30 steps, CFG 4.0, STG 1.0 stg_v[29], neg
  "worst quality, inconsistent motion, distorted, jittery" — exp_058/061 contract.
- **Compute:** `HCESC-H100-secondary`, one array task per seed; skip-if-exists
  (requeue-safe). ~2 GPU-h total.

## How to run

```bash
cd $LAB/diffusion-research
python experiments/exp_063_ladder_r4r5_generalist/build_manifests.py   # -> dataset/ladder_r4r5.json
sbatch --partition=HCESC-H100-secondary --account=hcesc-h100 --gres=gpu:1 --requeue \
       experiments/exp_063_ladder_r4r5_generalist/job_infer.sbatch
# after sidedness validation, hero_flight R5 (PLAN §B1):
#   uv run --frozen python .../run_ic_inference.py --seed 42 --include-deferred   (×43,44)
```

## Outputs

- `outputs/videos/exp_063_ladder_r4r5_generalist/{R4,R5}/<rung>__<class>__<clip>__s<seed>.mp4`.
- `dataset/ladder_r4r5.json` — 22 rows (20 active + 2 deferred), adapter sha256,
  per-row reference / prefix_only / endpoint_seen_by_ic2 / sidedness key + source.
- Slurm logs: `outputs/logs/slurm/exp063_r4r5-<ARRAY>_*.out`.

Scoring stays blocked until the sidedness re-annotation is validated (feeds the S mask).
Manifests carry everything `score.py` needs (SPEC §2, `endpoint_seen_by_ic2` for the C5
stratification, `reference_video` = the item's ground-truth clip as in exp_061).
