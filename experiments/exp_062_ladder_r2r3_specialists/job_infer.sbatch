#!/bin/bash
#SBATCH --job-name=exp062_r2r3gen
#SBATCH --array=0-32%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:55:00
#SBATCH --output=outputs/logs/slurm/%x-%A_%a.out
# R2/R3 generation from the trained specialists (PLAN §C1). One task per (class, seed):
# 11 classes × 3 seeds = 33 tasks; each generates R2 (held-in) + R3 (unseen) at BOTH
# ckpt 250 and 2000 = 8 videos/task (264 total). SUBMIT ONLY AFTER exp_062 training —
# chain with --dependency=afterok:<train_arrayjobid>. Skip-if-exists (requeue-safe).
# Usage: sbatch --dependency=afterok:<JID> [partition/account/gres --requeue] job_infer.sbatch

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_062_ladder_r2r3_specialists

CLASSES=(shadow portal super_fast_run shadow_smoke polygon wireframe animalization color_rain gas_transformation hero_flight illustration_scene)
SEEDS=(42 43 44)
CLS=${CLASSES[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
echo "=== class=$CLS seed=$SEED (task $SLURM_ARRAY_TASK_ID) ==="
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --frozen python $EXP/run_c2v_inference.py --class $CLS --seed $SEED
RC=$?
echo "=== EXIT $RC ==="
exit $RC
