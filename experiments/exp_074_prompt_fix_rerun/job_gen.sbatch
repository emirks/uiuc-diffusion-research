#!/bin/bash
#SBATCH --job-name=exp074_gen
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --gres=gpu:H100:1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:59:00
#SBATCH --output=outputs/logs/slurm/%x-%A_%a.out
# exp_074 corrected-prompt regeneration. MANIFEST via --export; array task =
# (seed, chunk): task = seed_idx*CHUNKS + chunk. Seeds 42/43/44.
#   ic3 C+X (51 rows): --export=ALL,MANIFEST=dataset/manifest_ic3_cx.json,CHUNKS=6 --array=0-17
#   r3x per class (4): --export=ALL,MANIFEST=dataset/manifest_r3x_<cls>.json,CHUNKS=1 --array=0-2
# Skip-if-exists + requeue-safe.

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_074_prompt_fix_rerun
CHUNKS=${CHUNKS:-1}
SEEDS=(42 43 44)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / CHUNKS))]}
CHUNK=$((SLURM_ARRAY_TASK_ID % CHUNKS))
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --frozen python $EXP/run_gen.py \
    --manifest ${MANIFEST:?--export=ALL,MANIFEST=...} \
    --seed $SEED --chunk $CHUNK --num-chunks $CHUNKS
RC=$?
echo "=== EXIT $RC ==="
exit $RC
