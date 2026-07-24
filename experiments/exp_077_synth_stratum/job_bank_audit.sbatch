#!/bin/bash
#SBATCH --job-name=exp077_audit
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# CPU-only operator-bank audit render + gate stats (TASK 0e part A/B). No GPU.
source ~/.bashrc
set -eo pipefail
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export GALLIUM_DRIVER=llvmpipe
mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$LAB/diffusion-research/experiments/exp_077_synth_stratum"
echo "[audit] host=$(hostname) $(date)"
python bank_audit.py
echo "[audit] done $(date)"
