#!/bin/bash
#SBATCH --job-name=d1_render
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=03:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# D1 STAGE 1 (render) — CPU procedural rendering (EGL/llvmpipe), JOB ARRAY. Idempotent (skips
# tuples whose clips+meta already exist), so a preemption+requeue resumes per-tuple.
#   sbatch --array=0-23%24 --export=ALL,NSHARDS=24 job_render_d1.sbatch
source ~/.bashrc
set -eo pipefail
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
export GALLIUM_DRIVER=llvmpipe
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SHARD=${SLURM_ARRAY_TASK_ID:-0}
export NSHARDS=${NSHARDS:-1}
mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$LAB/diffusion-research/experiments/exp_077_synth_stratum"
echo "[render] host=$(hostname) shard=$SHARD/$NSHARDS $(date)"
python render_d1.py
echo "[render] shard $SHARD done $(date)"
