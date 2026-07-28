#!/bin/bash
#SBATCH --job-name=caption_strips
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch/scripts/ctt_v2/caption_strips/logs/strips_%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch/scripts/ctt_v2/caption_strips/logs/strips_%j.err
#
# Caption anchors (9-frame mp4) + 3x3 JPEG filmstrips for both endpoint banks.
# CPU only.  Runs on `secondary` rather than a login node: ffmpeg at 12-way concurrency
# trips the login node's per-user thread ceiling (`pthread_create() failed`).  Every
# ffmpeg call is `-threads 1`, so --workers is the only concurrency knob.
#
#   sbatch scripts/ctt_v2/caption_strips/job_strips.sbatch
#
set -euo pipefail

REPO=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch
PY=/projects/illinois/eng/cs/jrehg/users/emirkisa/envs/diffusion/bin/python
export PATH="$HOME/.local/bin:$PATH"          # static ffmpeg 7.0.2
export FFMPEG_BIN="$(command -v ffmpeg)"

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "ffmpeg    : $FFMPEG_BIN"
echo "started   : $(date -Is)"

cd "$REPO"
srun "$PY" scripts/ctt_v2/caption_strips/extract_caption_strips.py \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    "$@"

echo "finished  : $(date -Is)"
