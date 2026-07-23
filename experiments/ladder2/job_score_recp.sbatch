#!/bin/bash
#SBATCH --job-name=ladder2_recp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# Amendment-1 Pass B: score generations + anchors against the RECIPIENT pool.
source ~/.bashrc || true
set -eo pipefail
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
PY_BIN="$LAB/envs/diffusion/bin/python"
MAIN=$LAB/diffusion-research
WT=$MAIN/.claude/worktrees/eval-v4-cert
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
PASS="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
cd "$MAIN"
PYTHONPATH=$WT/src "$PY_BIN" -m diffusion.transition_eval.score \
  --manifest "$MAIN/experiments/ladder2/eval_recp/eval_c${CHUNK}.json" \
  --corpus  "$MAIN/data/processed/transitions_std121/corpus_manifest.json" \
  --label   "recp_c${CHUNK}_p${PASS}" \
  --out-root "$MAIN/outputs/eval/ladder2_recp" \
  --cache-dir "$MAIN/outputs/eval/cache"
echo "=== EXIT $? ==="
