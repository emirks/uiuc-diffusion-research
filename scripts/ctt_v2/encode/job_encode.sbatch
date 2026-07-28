#!/bin/bash
#SBATCH --job-name=ctt2_enc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --exclude=ccc0424
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
#
# CTT v2 — VAE-encode latents/ + cond_clean/ for the new dataset strata (prompt-agnostic,
# therefore NOT blocked on the Gemini caption blocker).  One array task = one shard.
#
#   sbatch --partition=HCESC-L40S-normal --account=hcesc-l40s --gres=gpu:L40S:1 \
#          --array=0-15%8 --export=ALL,GROUP=s2  scripts/ctt_v2/encode/job_encode.sbatch
#   sbatch --partition=HCESC-L40S-normal --account=hcesc-l40s --gres=gpu:L40S:1 \
#          --array=0-3%4  --export=ALL,GROUP=aux scripts/ctt_v2/encode/job_encode.sbatch
#
# Overflow (4 h cap, preempt=REQUEUE — safe here, every write is atomic + skip-if-exists):
#   --partition=secondary --account=campusclusterusers --gres=gpu:H100:1 --time=03:59:00
#   --partition=scavenger --account=campusclusterusers --gres=gpu:L40S:1
#
# NOTE: the shard count is a HARDCODED constant inside encode_strata.py (NSHARDS) and is
# deliberately NOT derived from SLURM_ARRAY_TASK_COUNT — a partial resubmit with a narrower
# --array must NOT re-partition the work.  Re-running any shard is a safe no-op.
# ccc0424 is excluded: broken EGL, it killed six render shards on 2026-07-27.
set -eo pipefail
source ~/.bashrc || true

LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
# 🔴 WAS `$LAB/diffusion-research/.claude/worktrees/bottleneck-branch` — an EPHEMERAL worktree.
# On 2026-07-28 that silently cost a run: `NSHARDS["S1"]` was raised 1 -> 8 in the MAIN checkout
# and the roster was re-staged with main's copy (recording nshards=8), but the job kept executing
# the worktree's stale copy, which still said 1. The array's 7 other tasks each reported
# "S1: shard N is empty (nshards=1) — skip" and exited 0 in ~26 s, so the run LOOKED clean while
# doing an eighth of the intended parallelism. Same defect family as the worktree-prefixed
# inventory paths fixed in b71b2db: a worktree is not a durable location to execute from.
WT=$LAB/diffusion-research
# ABSOLUTE interpreter path — the trainer venv owns ltx_trainer (VAE + video reader).
# A relative interpreter caused an exit-127 failure on 2026-07-27.
PY=$LAB/LTX-2-official/.venv/bin/python
export PYTHONPATH="$LAB/LTX-2-cond-bleed-fix/packages/ltx-trainer/src"

GROUP=${GROUP:-s2}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
echo "[job] host=$(hostname) group=$GROUP shard=$SHARD job=${SLURM_JOB_ID} $(date)"
nvidia-smi -L || true
test -x "$PY" || { echo "[job] FATAL: interpreter $PY not executable"; exit 127; }

"$PY" "$WT/scripts/ctt_v2/encode/encode_strata.py" encode --group "$GROUP" --shard "$SHARD"

echo "[job] group=$GROUP shard=$SHARD done $(date)"
