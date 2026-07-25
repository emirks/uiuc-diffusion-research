#!/bin/bash
#SBATCH --job-name=s2render
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=03:55:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/experiments/exp_081_s2_stratum/logs/%x-%A_%a.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/experiments/exp_081_s2_stratum/logs/%x-%A_%a.out

# S2 mass render: 800 exact ops x 10 content pairs = 8,000 clips, CPU-only.
#
# NO GPU. The renderer is EGL/llvmpipe software GL — this deliberately never touches the
# H100/H200/L40S pools, which stay free for S3 and for the other campaign's training.
#
# Sizing (measured, from the exp_077 D2 build): 2.14 s per TUPLE of 2 clips => ~1.07 s of pure
# GLSL render per clip, plus mp4 encode + the four gate metrics. At ~1.7x gate overdraw the
# whole stratum is ~13,600 renders — a few core-hours total. 20 shards is far more parallelism
# than the job needs; it exists so each shard finishes well inside the 4 h secondary cap and a
# preemption costs almost nothing.
#
# MEMORY: 32g, paired with clip_cache=140 in config_s2.yaml. The first launch used 24g with a
# 230-clip cache and OOM-killed half the array — 111 MB per decoded clip x 230 = 25.5 GB.
#
# Resumability: each shard appends one JSONL row per accepted clip and one per FINALISED op,
# and skips any op already finalised in its own ops_shard file. A requeue therefore resumes at
# op granularity, losing at most the op in flight. Because an op is only written once it is
# complete (or dropped), a half-filled op can never be mistaken for a shipped one.

# NOTE: source ~/.bashrc BEFORE `set -u` — /etc/bashrc references an unbound BASHRCSOURCED.
source ~/.bashrc
set -eo pipefail

LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYOPENGL_PLATFORM=egl

cd "$LAB/diffusion-research/experiments/exp_081_s2_stratum"
SHARD=${SLURM_ARRAY_TASK_ID:-0} NSHARDS=${SLURM_ARRAY_TASK_COUNT:-20} \
  "$LAB/envs/diffusion/bin/python" render_s2.py
echo "=== shard $SLURM_ARRAY_TASK_ID EXIT $? ==="
