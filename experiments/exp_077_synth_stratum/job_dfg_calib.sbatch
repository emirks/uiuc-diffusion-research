#!/bin/bash
#SBATCH --job-name=dfg_calib
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# exp_077 DFG calibration, PHASE=features — CPU-ONLY (moderngl + EGL/llvmpipe, PyAV decode).
# NO --gres: this must never occupy a GPU. Resumable (append-only JSONL keyed by stem).
#   sbatch job_dfg_calib.sbatch
source ~/.bashrc
set -eo pipefail
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
export GALLIUM_DRIVER=llvmpipe
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PHASE=features
mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$LAB/diffusion-research/experiments/exp_077_synth_stratum"
echo "[dfg] host=$(hostname) phase=$PHASE $(date)"
python calibrate_dfg.py
echo "[dfg] done $(date)"
