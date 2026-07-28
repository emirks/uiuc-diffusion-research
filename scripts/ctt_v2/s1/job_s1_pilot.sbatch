#!/bin/bash
#SBATCH --job-name=ctt_v2_s1_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=01:30:00
# No --signal: there is nothing to checkpoint mid-clip, and an untrapped USR1 would kill the
# task 2 minutes EARLY. Resume is skip-if-exists on the per-clip output path instead.
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch/outputs/logs/slurm/%x-%A_%a.out
#
# CTT v2 / S1 PILOT -- 33 clips (3 per specialist x 11 specialists), A5 Ruling 3 / A1b Q1e.
#
# Built to BACKFILL: every HCESC node is at 8/8 GPUs, so this asks for one GPU, 90 minutes and
# modest memory, split into 3 independent array tasks (~11 clips each, ~35 min of real work).
# Any one task fits in almost any gap that opens, and `--requeue` + skip-if-exists means a
# preemption costs the clips in flight, never the batch.
#
# The 33 clips are the pilot's 3 shared probe endpoints x all 11 specialists, so they are a
# STRICT SUBSET of the full 390 -- nothing here is thrown away if the gate passes, and the
# batch holds content constant across the 11 classes, which is exactly what the blind
# 11-way class-ID gate needs (the judge cannot use content as a cue).
#
# Partition chosen by measured estimated start (sbatch --test-only, 2026-07-28 00:35):
#   HCESC-H100-secondary -> 00:36 TODAY   (preempts 2 foreign `scavenger` tasks on our own node)
#   HCESC-L40S-secondary -> 08:51 today
#   HCESC-H200-secondary -> 21:19 today
# H100 also happens to be the only GPU the 19B generation path is validated on (every prior
# ladder2 gen ran gres/gpu:h100=1); L40S's 48 GB has never been tested with this model, so an
# unattended backfill there risks an OOM that wastes the window.
#
#   sbatch --array=0-2 --partition=HCESC-H100-secondary --account=hcesc-h100 \
#          scripts/ctt_v2/s1/job_s1_pilot.sbatch
set -eo pipefail
source ~/.bashrc || true

LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
# the ctt-v2 campaign worktree -- NEVER the main tree, which a parallel campaign is writing
DR=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
GRID=$LAB/misc/ctt_v2_final/S1_GRID.json
NCHUNKS="${NCHUNKS:-3}"
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "$DR/outputs/logs/slurm"
echo "[s1] host=$(hostname) chunk=$CHUNK/$NCHUNKS job=$SLURM_JOB_ID $(date)"
echo "[s1] worktree=$DR branch=$(git -C "$DR" rev-parse --abbrev-ref HEAD)@$(git -C "$DR" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

test -f "$GRID" || { echo "[s1] FATAL: grid missing at $GRID"; exit 2; }

cd "$TRAINER"
uv run --frozen python "$DR/scripts/ctt_v2/s1/run_s1_gen.py" \
    --grid "$GRID" --pilot --chunk "$CHUNK" --num-chunks "$NCHUNKS"
echo "=== EXIT $? ==="
echo "[s1] done $(date)"
