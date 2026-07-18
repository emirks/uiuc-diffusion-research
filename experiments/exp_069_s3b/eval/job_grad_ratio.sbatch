#!/bin/bash
#SBATCH --job-name=exp069_gradratio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=00:40:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Stage-3b pre-launch gate: aux(margin):FM grad-norm ratio check.
set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
cd $LAB/diffusion-research && mkdir -p outputs/logs/slurm
cd $LAB/LTX-2-official/packages/ltx-trainer
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run --frozen python $LAB/diffusion-research/experiments/exp_069_s3b/eval/grad_ratio_check.py
echo "=== EXIT $? ==="
