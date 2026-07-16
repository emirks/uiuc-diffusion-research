#!/bin/bash
#SBATCH --job-name=exp063_r4r5
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:55:00
#SBATCH --output=outputs/logs/slurm/%x-%A_%a.out
# One array task per seed (42/43/44). Generates R4 (16) + R5 gas/illustration (4)
# = 20 videos/seed via exp_058 ic2 step_05000, native keyed. hero_flight R5 is
# deferred (sidedness pending) — not generated here. Skip-if-exists (requeue-safe).
# Usage: sbatch [partition/account/gres flags --requeue] job_infer.sbatch

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_063_ladder_r4r5_generalist

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "=== seed $SEED (task $SLURM_ARRAY_TASK_ID) ==="
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --frozen python $EXP/run_ic_inference.py --seed $SEED
RC=$?
echo "=== EXIT $RC ==="
exit $RC
