#!/bin/bash
#SBATCH --job-name=bneck_pgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# exp_078 — generate one arm on the frozen proxy_set_v2 cells, both claim seeds (42,43).
# Idempotent (run_gen skips existing videos), so requeue-safe and re-runnable.
#
# ARM  = registry arm name (b1 | ic_gen)
# STEP = checkpoint step override (e.g. 1500 or 5000). For ic_gen this selects the comparator ckpt.
#
#   # chained to fire the moment B1 finishes training:
#   sbatch --dependency=afterok:<b1_jobid> --array=0-3%2 \
#          --partition=HCESC-H100-normal --account=hcesc-h100 --gres=gpu:H100:1 --time=03:00:00 \
#          --export=ALL,ARM=b1,STEP=5000 experiments/exp_078_operator_token_bottleneck/job_proxy_gen.sbatch
set -eo pipefail
source ~/.bashrc || true
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
WT=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch
TRAINER=$LAB/LTX-2-bneck/packages/ltx-trainer   # bneck trainer: run_gen imports operator_encoder
VENV=$LAB/LTX-2-official/.venv
ARM="${ARM:?set ARM=b1|ic_gen}"
STEP="${STEP:?set STEP=<checkpoint step>}"
CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NCHUNKS=${NCHUNKS:-4}
# The five proxy_set_v2 cells. Same-content guards + the two cross claim cells + the liveness probe.
CELLS=G-unseen-cross,G-zs-cross,G-unseen-same,G-zs-same,G-ref-control

export LADDER_OUT_ROOT="$WT/outputs/videos/bneck_proxy"
mkdir -p "$LADDER_OUT_ROOT" "$LAB/diffusion-research/outputs/logs/slurm"
echo "[pgen] host=$(hostname) arm=$ARM step=$STEP chunk=$CHUNK/$NCHUNKS $(date)"
nvidia-smi -L || true
cd "$WT"
for SEED in 42 43; do
  echo "[pgen] --- seed $SEED ---"
  PYTHONPATH="$TRAINER/src" "$VENV/bin/python" eval_ladder/run_gen.py \
      --arm "$ARM" --seed "$SEED" --step "$STEP" --cells "$CELLS" \
      --chunk "$CHUNK" --num-chunks "$NCHUNKS"
done
echo "[pgen] done $(date)"
