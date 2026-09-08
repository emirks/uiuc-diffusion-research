#!/bin/bash
#SBATCH --job-name=collapse_post
#SBATCH --account=bgjg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110g
#SBATCH --time=00:30:00
#SBATCH --constraint="taiga"
#SBATCH --output=misc/2026-09-08_collapse_remeasure/logs/post_%j.log
# Covariates (DINO/CLIP endpoint distances, GT ceiling) + analysis; scoring already done in job 3114599.
set -euo pipefail
cd $LAB/diffusion-research
source $LAB/envs-aarch64/activate
export HF_HOME=$LAB/cache/huggingface HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
echo "[job] $(hostname) $(date)"
TORCH_THREADS=20 python misc/2026-09-08_collapse_remeasure/covariates.py
echo "[job] covariates done $(date)"
python misc/2026-09-08_collapse_remeasure/analyze.py > misc/2026-09-08_collapse_remeasure/logs/analyze_$SLURM_JOB_ID.out
echo "[job] analysis done $(date)"
