#!/bin/bash
#SBATCH --job-name=s2bplan
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch/experiments/exp_082_s2_humanvid/logs/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch/experiments/exp_082_s2_humanvid/logs/%x-%j.out

# S2b grid freeze: build the endpoint-frame cache over the 1,146-clip union pool (~0.6 s/clip
# decode => ~12 min), pre-validate the 800-op roster against gate-2, and freeze
# PLAN_S2_UNION.json. CPU-only (EGL/llvmpipe software GL) — never a GPU.
#
# Memory: the frame cache is 1,146 clips x 2 frames x 640x480x3 = 2.1 GB resident, plus a
# transient 111 MB per decoded clip. 24g is ample; the RENDER job is the one that needs more.

# NOTE: source ~/.bashrc BEFORE `set -u` — /etc/bashrc references an unbound BASHRCSOURCED.
source ~/.bashrc
set -eo pipefail

LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYOPENGL_PLATFORM=egl

cd "$LAB/diffusion-research/.claude/worktrees/bottleneck-branch/experiments/exp_082_s2_humanvid"
"$LAB/envs/diffusion/bin/python" plan_s2.py
echo "=== plan EXIT $? ==="
