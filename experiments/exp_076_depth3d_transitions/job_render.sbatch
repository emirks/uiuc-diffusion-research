#!/bin/bash
#SBATCH --job-name=exp076_d3d
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# CPU-only: the operator renderer uses EGL on Mesa llvmpipe, so no GPU is needed
# and this never competes with training for the lab's H100/H200 pools.

# NOTE: source ~/.bashrc BEFORE `set -u` — /etc/bashrc references an unbound
# BASHRCSOURCED and would abort the job.
source ~/.bashrc
set -eo pipefail

module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"

# Keep every thread pool bounded — llvmpipe + swscale + x264 otherwise each try
# to grab all 128 cores of the node and blow past the per-user thread limit.
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export GALLIUM_DRIVER=llvmpipe
export HF_HOME="$LAB/cache/huggingface"

cd "$LAB/diffusion-research/experiments/exp_076_depth3d_transitions"
python run.py
