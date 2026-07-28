#!/bin/bash
#SBATCH --job-name=bneck_nf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# exp_078 — 0c NOISE FLOOR generation (advisor round-1, item C: "measure it before any bar is applied").
#
# Generates ic_gen@5000 on the frozen proxy_set_v2 cells at seeds 44/45. Paired with the ALREADY
# EXISTING seeds 42/43 videos, the two independent evals give the per-cell SD of donor-% and meanC.
# The proxy promote bar is max(+3pp, 2 x SD); if SD > 3pp the bars are revised BEFORE the first B1
# read, and if the zs-same SD > 2pp that cell goes to 4 seeds.
#
# Writes to a CAMPAIGN-PRIVATE video root — never into the shared ladder2 tree the parallel
# dataset-expansion campaign reads.
#
#   sbatch --array=0-7 --partition=secondary --account=campusclusterusers \
#          --gres=gpu:H100:1 --time=03:00:00 --export=ALL,SEED=44 \
#          experiments/exp_078_operator_token_bottleneck/job_noise_floor.sbatch
set -eo pipefail
source ~/.bashrc || true
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
WT=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch
TRAINER=$LAB/LTX-2-cond-bleed-fix/packages/ltx-trainer   # generation uses the LINEAGE trainer:
                                                         # ic_gen is a plain IC-LoRA, no surgery needed
VENV=$LAB/LTX-2-official/.venv
SEED="${SEED:?set SEED=44 or 45}"
CHUNK=${SLURM_ARRAY_TASK_ID:-0}
NCHUNKS=${NCHUNKS:-8}

export LADDER_OUT_ROOT="$WT/outputs/videos/bneck_nf"
mkdir -p "$LADDER_OUT_ROOT" "$LAB/diffusion-research/outputs/logs/slurm"
echo "[nf] host=$(hostname) seed=$SEED chunk=$CHUNK/$NCHUNKS out=$LADDER_OUT_ROOT $(date)"
nvidia-smi -L || true

cd "$WT"
PYTHONPATH="$TRAINER/src" "$VENV/bin/python" eval_ladder/run_gen.py \
    --arm ic_gen --seed "$SEED" \
    --cells G-unseen-cross,G-zs-cross,G-unseen-same,G-zs-same,G-ref-control \
    --chunk "$CHUNK" --num-chunks "$NCHUNKS"
echo "[nf] done $(date)"
