#!/bin/bash
#SBATCH --job-name=exp073_precomp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=00:59:00
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/experiments/exp_073_cond_bleed_fix/logs/slurm/%x-%j.out
# Partition/account passed at submit time, e.g.:
#   sbatch --partition=HCESC-H100-secondary --account=hcesc-h100 --export=ALL,MODE=gate job_precompute.sbatch
set -euo pipefail
source ~/.bashrc
DR=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
WT=/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-cond-bleed-fix
VENV=/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv
mkdir -p "$DR/experiments/exp_073_cond_bleed_fix/logs/slurm"
cd "$DR"
echo "[job] host=$(hostname) mode=${MODE:-gate} $(date)"
nvidia-smi -L || true
PYTHONPATH="$WT/packages/ltx-trainer/src" "$VENV/bin/python" \
    experiments/exp_073_cond_bleed_fix/precompute_cond_clean.py --mode "${MODE:-gate}" --device cuda
echo "[job] done $(date)"
