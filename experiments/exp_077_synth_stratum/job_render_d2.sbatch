#!/bin/bash
#SBATCH --job-name=d2_render
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=03:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# exp_077 D2 render — CPU-ONLY procedural rendering (moderngl + EGL/llvmpipe). NO --gres: this
# must never occupy a GPU. Resumable: each shard appends to its own JSONL and skips done clips.
#   MODE=sanity: sbatch --export=ALL,MODE=sanity job_render_d2.sbatch
#   MODE=audit : sbatch --array=0-15%16 --export=ALL,MODE=audit,NSHARDS=16 job_render_d2.sbatch
source ~/.bashrc
set -eo pipefail
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
export GALLIUM_DRIVER=llvmpipe
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MODE=${MODE:-audit}
export SHARD=${SLURM_ARRAY_TASK_ID:-0}
export NSHARDS=${NSHARDS:-1}
mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$LAB/diffusion-research/experiments/exp_077_synth_stratum"
echo "[d2] host=$(hostname) mode=$MODE shard=$SHARD/$NSHARDS $(date)"
python render_d2.py
echo "[d2] mode=$MODE shard=$SHARD done $(date)"
