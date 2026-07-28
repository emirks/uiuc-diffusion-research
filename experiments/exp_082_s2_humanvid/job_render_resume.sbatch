#!/bin/bash
#SBATCH --job-name=s2hvrender
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --array=14-19
#SBATCH --exclude=ccc0424
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --time=03:55:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch/experiments/exp_082_s2_humanvid/logs/%x-%A_%a.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch/experiments/exp_082_s2_humanvid/logs/%x-%A_%a.out

# S2b (UNION bank) mass render: 800 ops x 10 contents = 8,000 clips, CPU-only.
# Reads the frozen PLAN_S2_UNION.json; NSHARDS=20 and it MUST stay 20 across any resume
# (sharding is `op_index % nshards == shard`).
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
# MEMORY: 40g, paired with clip_cache=140 in config_s2.yaml. S2a's first launch used 24g with a
# 230-clip cache and OOM-killed half the array — 111 MB per decoded clip x 230 = 25.5 GB; the
# fix was 32g + clip_cache 140 (15.5 GB). S2b adds the union endpoint-frame cache on top:
# 1,146 clips x 2 frames x 640x480x3 = 2.1 GB resident (S2a's was 292 MB / 327 clips), so the
# same 140-clip cache now peaks around 24 GB. 40g is that plus honest headroom.
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

cd "$LAB/diffusion-research/.claude/worktrees/bottleneck-branch/experiments/exp_082_s2_humanvid"
SHARD=${SLURM_ARRAY_TASK_ID:-0} NSHARDS=20 \
  "$LAB/envs/diffusion/bin/python" render_s2.py
echo "=== shard $SLURM_ARRAY_TASK_ID EXIT $? ==="
