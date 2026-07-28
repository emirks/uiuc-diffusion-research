#!/bin/bash
#SBATCH --job-name=s3pilot
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# S3 PILOT — 63 clips on BANK contents, 9 per camera family, heavy-optics tail forced.
# This is the advisor's mandatory gate: the 1,800-clip render may not start until the
# pre-committed bars in config_s3.yaml pass AND the blind visual audit clears.
# Depth stacks are read from the precomputed cache, so timings here are render cost only.

# NOTE: source ~/.bashrc BEFORE `set -u` — /etc/bashrc references an unbound BASHRCSOURCED.
source ~/.bashrc
set -eo pipefail

module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"

export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HOME="$LAB/cache/huggingface"

cd "$LAB/diffusion-research/experiments/exp_082_s3_stratum"
python run_pilot.py
