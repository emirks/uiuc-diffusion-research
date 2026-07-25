#!/bin/bash
#SBATCH --job-name=s3depth
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Precompute the per-frame stabilised depth stack for all 227 bank clips.
# Depth is a pure function of the clip, so this cache survives any later change to the
# operator grid or the endpoint split. Resumable — an existing .npy is never recomputed.

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
python precompute_depth.py
