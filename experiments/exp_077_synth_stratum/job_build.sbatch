#!/bin/bash
#SBATCH --job-name=exp077_build
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=03:59:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# GPU: VAE-encode latents + cond-clean + text embeds + assemble root (TASK 0d steps 3-5).
# Idempotent (process_videos skips existing). Submit with RUN=<render run_dir>.
#   sbatch --export=ALL,RUN=/.../run_0000 job_build.sbatch
source ~/.bashrc || true
set -eo pipefail
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
WT=$LAB/LTX-2-cond-bleed-fix
VENV=$LAB/LTX-2-official/.venv
: "${RUN:?set RUN=<render run_dir>}"
mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$LAB/diffusion-research/experiments/exp_077_synth_stratum"
echo "[build] host=$(hostname) run=$RUN $(date)"
nvidia-smi -L || true
PYTHONPATH="$WT/packages/ltx-trainer/src" "$VENV/bin/python" build_dataset.py --run "$RUN" --device cuda
echo "[build] done $(date)"
