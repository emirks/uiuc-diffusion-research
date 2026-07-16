#!/bin/bash
#SBATCH --job-name=exp063_r4x
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:55:00
#SBATCH --output=outputs/logs/slurm/%x-%A_%a.out
# Amendment 1: R4X generation (generalist twin of R3X, contrast C9). One seed per array
# task (42/43/44); 32 donor rows each => 96 videos. INDEPENDENT of specialist training
# (uses the existing ic2 adapter) — submit now. Skip-if-exists (requeue-safe).
# Usage: sbatch [partition/account/gres --requeue] job_r4x.sbatch

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_063_ladder_r4r5_generalist

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "=== R4X seed=$SEED (task $SLURM_ARRAY_TASK_ID) ==="
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --frozen python $EXP/run_ic_inference.py --seed $SEED --manifest dataset/ladder_r4x.json
RC=$?
echo "=== EXIT $RC ==="
exit $RC
