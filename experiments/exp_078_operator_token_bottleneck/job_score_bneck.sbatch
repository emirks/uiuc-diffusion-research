#!/bin/bash
#SBATCH --job-name=bneck_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# exp_078 — score one B1 eval chunk with the certified v4 instrument.
#
# Runs from the WORKTREE (the ladder2 job_score cd's to the MAIN checkout, which has no b1
# registry rows and no --arms flag). Scores into a PRIVATE dir so ladder2's frozen scores are
# never touched. The instrument itself is the certified eval-v4-cert worktree, unchanged.
#
#   # after `run_eval --mode plan --arms b1 ...` has written eval_bneck/eval_c*.json:
#   sbatch --array=0-7 --partition=HCESC-H100-normal --account=hcesc-h100 --gres=gpu:H100:1 \
#          experiments/exp_078_operator_token_bottleneck/job_score_bneck.sbatch
set -eo pipefail
source ~/.bashrc || true
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
PY_BIN="$LAB/envs/diffusion/bin/python"
WT=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch          # OUR checkout
CERT=$LAB/diffusion-research/.claude/worktrees/eval-v4-cert             # the certified instrument
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
MDIR="$WT/eval_ladder/eval_bneck"
SOUT="$WT/outputs/eval/bneck"
PASS="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

mkdir -p "$SOUT/c${CHUNK}_p${PASS}" "$LAB/diffusion-research/outputs/logs/slurm"
cd "$WT"
echo "[score] host=$(hostname) chunk=$CHUNK manifest=$MDIR/eval_c${CHUNK}.json $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PYTHONPATH="$CERT/src" "$PY_BIN" -m diffusion.transition_eval.score \
    --manifest "$MDIR/eval_c${CHUNK}.json" \
    --out "$SOUT/c${CHUNK}_p${PASS}/items.jsonl"
echo "[score] done $(date)"
