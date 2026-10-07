#!/bin/bash
#SBATCH --job-name=collapse_remeasure
#SBATCH --account=bgjg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110g
#SBATCH --time=01:00:00
#SBATCH --constraint="taiga"
#SBATCH --output=misc/2026-09-08_collapse_remeasure/logs/score_%j.log
# CPU-only scoring of existing store generations with the revised null-family instrument.
# 1 fractional GPU is the smallest billable unit on DeltaAI (bills 1 SU/h); the GPU is idle.
set -euo pipefail
cd $LAB/diffusion-research
source $LAB/envs-aarch64/activate
export HF_HOME=$LAB/cache/huggingface HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[job] $(hostname) $(date) cpus=$SLURM_CPUS_PER_TASK"
python misc/2026-09-08_collapse_remeasure/score_store.py --nproc 30
echo "[job] scoring done $(date)"
TORCH_THREADS=24 python misc/2026-09-08_collapse_remeasure/covariates.py
echo "[job] covariates done $(date)"
python misc/2026-09-08_collapse_remeasure/analyze.py > misc/2026-09-08_collapse_remeasure/logs/analyze_$SLURM_JOB_ID.out
echo "[job] analysis done $(date)"
