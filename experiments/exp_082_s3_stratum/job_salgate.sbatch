#!/bin/bash
#SBATCH --job-name=s3salgate
#SBATCH --partition=scavenger
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Re-render the 63 labelled pilot operators capturing geometric coverage, and calibrate the S3
# gate. CPU: depth stacks are already cached, so no GPU is needed, and EGL falls back to Mesa
# llvmpipe (measured 0.038 s per layer render at 480x640 — a clip is ~20-30 s). All 8 L40S GPUs
# were held by other users while scavenger had 24 idle nodes, and queue time is the scarce
# resource here, not FLOPs.
source ~/.bashrc
set -eo pipefail
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$LAB/cache/huggingface"
cd "$LAB/diffusion-research/experiments/exp_082_s3_stratum"
python salience_gate.py
