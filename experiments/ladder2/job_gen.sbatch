#!/bin/bash
#SBATCH --job-name=ladder2_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=03:30:00
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# ladder2 generation. One job = one arm x one seed x one chunk; skip-if-exists makes it
# requeue-safe on preemptible queues.
#   sbatch --array=0-3 --partition=secondary --account=campusclusterusers --gres=gpu:H100:1 \
#          --export=ALL,ARM=spec_color_rain,SEED=42,PRIORITY=P0,NCHUNKS=4 \
#          experiments/ladder2/job_gen.sbatch
set -eo pipefail
source ~/.bashrc || true
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
ARM="${ARM:?set ARM=<registry arm>}"
SEED="${SEED:?set SEED=42|43}"
NCHUNKS="${NCHUNKS:-1}"
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$DR/outputs/logs/slurm"
echo "[gen] host=$(hostname) arm=$ARM seed=$SEED chunk=$CHUNK/$NCHUNKS $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$TRAINER"
uv run --frozen python "$DR/experiments/ladder2/run_gen.py" \
    --arm "$ARM" --seed "$SEED" --chunk "$CHUNK" --num-chunks "$NCHUNKS" \
    ${PRIORITY:+--priority "$PRIORITY"} ${CELLS:+--cells "$CELLS"} ${STEP:+--step "$STEP"}
echo "[gen] done $(date)"
