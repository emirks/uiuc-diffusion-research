#!/bin/bash
#SBATCH --job-name=ladder2_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=03:30:00
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# Score one ladder2 eval chunk with the v4 instrument (eval-v4-cert worktree — owner
# directive 2026-07-20: v4 is the lane instrument). Shared warm feature cache.
#   sbatch --array=0-7 --partition=secondary --account=campusclusterusers \
#          --gres=gpu:H100:1 experiments/ladder2/job_score.sbatch
set -eo pipefail
source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
conda activate "$LAB/envs/diffusion"
MAIN=$LAB/diffusion-research
WT=$MAIN/.claude/worktrees/eval-v4-cert
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
cd "$MAIN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
PYTHONPATH=$WT/src python -m diffusion.transition_eval.score \
  --manifest "$MAIN/experiments/ladder2/eval/eval_c${CHUNK}.json" \
  --corpus  "$MAIN/data/processed/transitions_std121/corpus_manifest.json" \
  --label   "c${CHUNK}" \
  --out-root "$MAIN/outputs/eval/ladder2" \
  --cache-dir "$MAIN/outputs/eval/cache"
echo "=== EXIT $? ==="
