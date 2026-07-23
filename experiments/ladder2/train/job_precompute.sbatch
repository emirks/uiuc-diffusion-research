#!/bin/bash
#SBATCH --job-name=ladder2_precomp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# ladder2 training precompute. Idempotent + requeue-safe (skip-if-exists per clip).
#   sbatch --partition=secondary --account=campusclusterusers --gres=gpu:H100:1 \
#          --export=ALL,MODE=cond-clean experiments/ladder2/train/job_precompute.sbatch
#   MODE=cond-clean  isolation-encode the suffix anchors (token-independent)
#   MODE=text        render + encode the leak-free training captions (needs the token verdict)
set -eo pipefail
source ~/.bashrc || true
DR=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
WT=/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-cond-bleed-fix
VENV=/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv
mkdir -p "$DR/outputs/logs/slurm"
cd "$DR"
echo "[job] host=$(hostname) mode=${MODE:?set MODE=cond-clean|text} $(date)"
nvidia-smi -L || true
PYTHONPATH="$WT/packages/ltx-trainer/src" "$VENV/bin/python" \
    experiments/ladder2/train/precompute.py --mode "$MODE" --device cuda
echo "[job] done $(date)"
