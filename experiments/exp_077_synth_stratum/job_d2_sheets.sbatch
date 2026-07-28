#!/bin/bash
#SBATCH --job-name=d2_sheets
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=00:40:00
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# exp_077 D2 contact sheets. CPU-only; needs a compute node only because PyAV/swscale cannot
# allocate on the login nodes. STEP=select -> labeling sheets ; STEP=owner -> the owner sheet.
#   sbatch --export=ALL,STEP=select job_d2_sheets.sbatch
#   sbatch --export=ALL,STEP=owner  job_d2_sheets.sbatch
source ~/.bashrc
set -eo pipefail
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$LAB/diffusion-research/experiments/exp_077_synth_stratum"
echo "[sheets] host=$(hostname) step=${STEP:-select} $(date)"
if [ "${STEP:-select}" = "owner" ]; then
  python owner_sheet_d2.py
else
  PHASE=select python calibrate_d2.py
fi
echo "[sheets] done $(date)"
