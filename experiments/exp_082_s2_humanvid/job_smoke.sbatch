#!/bin/bash
#SBATCH --job-name=s2bsmoke
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --time=00:30:00
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch/experiments/exp_082_s2_humanvid/logs/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch/experiments/exp_082_s2_humanvid/logs/%x-%j.out

# S2b smoke: 3 ops end-to-end with the hard asserts -> outputs/.../smoke/SMOKE.json.
# Same resources as one render shard so the smoke also exercises the real memory profile
# (2.1 GB union endpoint-frame cache + up to 140 x 111 MB clip cache).
# Gate for the mass render: do NOT submit the array unless SMOKE.json says PASS.

# NOTE: source ~/.bashrc BEFORE `set -u` — /etc/bashrc references an unbound BASHRCSOURCED.
source ~/.bashrc
set -eo pipefail

LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYOPENGL_PLATFORM=egl

cd "$LAB/diffusion-research/.claude/worktrees/bottleneck-branch/experiments/exp_082_s2_humanvid"
MODE=smoke "$LAB/envs/diffusion/bin/python" render_s2.py
echo "=== smoke EXIT $? ==="
