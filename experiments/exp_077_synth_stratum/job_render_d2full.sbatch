#!/bin/bash
#SBATCH --job-name=d2f_render
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
# exp_077 D2-FULL render — CPU-ONLY procedural rendering (moderngl + EGL/llvmpipe). NO --gres:
# this must never occupy a GPU. Resumable per accepted tuple (append-only JSONL), so a
# preemption + requeue continues where it stopped.
#   full :  sbatch --array=0-15%16 --export=ALL,NSHARDS=16 job_render_d2full.sbatch
#   probe:  sbatch --export=ALL,SHARD=0,NSHARDS=1,OUTSUB=d2full_firstchunk,PLANFILE=d2full_firstchunk_plan.json job_render_d2full.sbatch
source ~/.bashrc
set -eo pipefail
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
export GALLIUM_DRIVER=llvmpipe
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SHARD=${SLURM_ARRAY_TASK_ID:-${SHARD:-0}}
export NSHARDS=${NSHARDS:-1}
mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$LAB/diffusion-research/experiments/exp_077_synth_stratum"
echo "[d2f] host=$(hostname) shard=$SHARD/$NSHARDS outsub=${OUTSUB:-default} plan=${PLANFILE:-default} $(date)"
python render_d2_full.py
echo "[d2f] shard $SHARD done $(date)"
