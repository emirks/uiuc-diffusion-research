#!/bin/bash
#SBATCH --job-name=exp080_d3d121
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24g
#SBATCH --time=03:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# One L40S: Depth-Anything runs per-frame (121 frames/clip stacks), and on an
# NVIDIA node the EGL context binds libEGL_nvidia so even the GL render is
# hardware. Never touches the H100/H200 claim-generation pools.

# NOTE: source ~/.bashrc BEFORE `set -u` — /etc/bashrc references an unbound
# BASHRCSOURCED and would abort the job.
source ~/.bashrc
set -eo pipefail

module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"

export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HOME="$LAB/cache/huggingface"

cd "$LAB/diffusion-research/.claude/worktrees/bottleneck-branch/experiments/exp_080_depth3d_realstream_121"
python run.py
python build_viewer.py "$(ls -d ../../outputs/videos/exp_080_depth3d_realstream_121/run_* | sort | tail -1)"
