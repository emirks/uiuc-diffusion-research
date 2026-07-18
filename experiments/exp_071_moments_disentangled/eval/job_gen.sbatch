#!/bin/bash
#SBATCH --job-name=exp071_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:55:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/slurm/%x-%A_%a.out
# exp_067 Stage-2 generation (residual reference ON). 55 rows x 3 seeds.
# array task = seed_idx*CHUNKS + chunk;  CHUNKS=4 -> --array=0-11
#   sbatch --array=0-11 [part/acct/gres] eval/job_gen.sbatch
# Skip-if-exists (requeue-safe). Needs exp_067 training DONE (adapter asserted).
set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_071_moments_disentangled
CHUNKS=${CHUNKS:-4}
SEEDS=(42 43 44)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / CHUNKS))]}
CHUNK=$((SLURM_ARRAY_TASK_ID % CHUNKS))
cd $LAB/diffusion-research && mkdir -p outputs/logs/slurm
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run --frozen python $EXP/eval/run_gen.py \
    --manifest $EXP/eval/manifest_071.json \
    --seed $SEED --chunk $CHUNK --num-chunks $CHUNKS
RC=$?
echo "=== EXIT $RC ==="
exit $RC
