# exp_050 — Standard LoRA fine-tuning baseline on shadow-smoke transitions (official LTX-2 trainer)

## Question

What does the **absolute baseline of standard LoRA fine-tuning** — the official
LTX-2 trainer at its shipped 80GB-tier defaults — learn from the 10 shadow-smoke
transition clips? Does a vanilla t2v LoRA reproduce the concept (a dense black
smoke mass sweeping across the frame and transforming scene A into scene B) on
held-out prompts with the `SHDWSMK` trigger, without degrading unrelated generation?

## Setup

- **Trainer**: official clean clone `$LAB/LTX-2-official` @ `7809842` (2026-06-17)
  — NOT the pod-modified vendored `src/LTX-2`. Env: `uv sync` workspace.
- **Base model**: `ltx-2-19b-dev.safetensors` (bf16 dev checkpoint, consistent with
  all prior LTX-2 experiments in this repo). Text encoder:
  `google/gemma-3-12b-it-qat-q4_0-unquantized`. Both staged in `$LAB/cache/huggingface/`.
- **Data**: the 10 clips in `data/processed/transitions/shadow_smoke/` (9×5s + 1×10s,
  24fps, no audio). Per-clip captions hand-written from visual inspection
  (`dataset/dataset.json`): scene A → consistent concept phrase ("a dense mass of
  black smoke sweeps across the frame and engulfs...") → scene B. Trigger word
  `SHDWSMK` prepended at preprocessing (`--lora-trigger`).
- **Preprocessing**: 3 aspect-ratio buckets `480x640x121; 640x480x121; 576x576x121`
  (portrait/landscape/square, F=121 ≈ full 5s @ 24fps, dims ÷32, F%8==1),
  `--skip-audio` not needed (no audio streams), `--decode` for visual verification.
- **Variants** (all identical to official `t2v_lora.yaml` defaults unless noted:
  rank 32/alpha 32, lr 1e-4 linear, 2000 steps, bs 1, adamw, bf16, no quantization,
  grad checkpointing, checkpoints every 250):
  | arm | config | delta vs official default |
  |---|---|---|
  | baseline | `config_baseline.yaml` | video-only target modules (clips silent) |
  | i2v_ff05 | `config_i2v_ff05.yaml` | + `first_frame` condition @ p=0.5 (official style-LoRA recommendation; superset usable for T2V and I2V) |
  | rank64_ffn | `config_rank64_ffn.yaml` | + rank/alpha 64, + FFN target modules (official capacity recipe) |
- **Validation during training**: every 250 steps + step 0 (before/after evidence),
  seed 42, 480×640×121 @ 24fps, 30 steps, CFG 4.0, STG 1.0 (`stg_v`);
  prompt 1 = held-out in-style prompt with trigger, prompt 2 = unrelated prompt
  (drift check).
- **Compute**: 1× H100 80GB per arm via Slurm (preemptible/4h-window queues;
  `job_train.sbatch` is resume-aware and chain-safe). Sanity: `job_sanity.sbatch`
  (one-sample 50-step dry run + resume test + full preprocess), per the official
  train-model skill Phase 6.

## How to run

```bash
cd $LAB/diffusion-research
# 1. sanity + preprocess (once)
sbatch --partition=HCESC-H100-secondary --account=hcesc-h100 --gres=gpu:1 --requeue \
    experiments/exp_050_ltx2_lora_shadowsmoke/job_sanity.sbatch
# 2. each arm (chain 2 copies to survive 4h windows / preemption)
J1=$(sbatch --parsable -J exp050_baseline --partition=HCESC-H100-secondary --account=hcesc-h100 \
    --gres=gpu:1 --requeue experiments/exp_050_ltx2_lora_shadowsmoke/job_train.sbatch config_baseline.yaml)
sbatch -J exp050_baseline --dependency=afterany:$J1 --partition=HCESC-H100-secondary --account=hcesc-h100 \
    --gres=gpu:1 --requeue experiments/exp_050_ltx2_lora_shadowsmoke/job_train.sbatch config_baseline.yaml
```

## Expected outcome

Pre-registered: with only 10 clips at 200 epochs, the LoRA should (a) bind the
smoke-transition concept to the trigger/caption phrasing — validation prompt 1
turning from "no transition, static-ish scene" at step 0 into a recognizable
black-smoke scene swap by step 1000–2000; (b) risk of overfitting to the 10
source scenes (memorized subjects reappearing) grows with steps — the 250-step
checkpoint ladder exists to pick the sweet spot; (c) the unrelated prompt 2
should stay stable (attention-only rank-32 LoRA is mild); FFN/rank-64 arm is the
most likely to drift. This experiment establishes the reference point that any
smarter method (injection recipes of exp_044–049, IC-LoRA conditioning) must beat.

## Outputs

- `outputs/training/exp_050_ltx2_lora_shadowsmoke/<arm>/` — checkpoints
  (`lora_weights_step_*.safetensors`), validation samples (`samples/step_*`), logs.
- `dataset/.precomputed/` — latents (3 buckets), text embeddings, decoded
  verification videos.
- Slurm logs: `outputs/logs/slurm/exp050_*.out`.

**Completed 2026-07-04.** Jobs: sanity 9321703; baseline 9321859 (58 min, 1.16 s/step,
peak 48.4 GB); i2v_ff05 9321861 (66 min); rank64_ffn 9321863 (72 min). All arms reached
step 2000 with full checkpoint ladders + 18 validation videos each. **Outcome: concept
acquired by step 500–1000** (subject-wrapping ink-black billow on the held-out trigger
prompt vs generic gray explosion at step 0); no drift on the no-trigger prompt in any arm.
Rank-32 attention-only suffices; `i2v_ff05` is the recommended default checkpoint family
(same quality + first-frame conditioning). Details: CHANGELOG 2026-07-04 and
`notes/exp/exp_050_lora_baseline.md`.
