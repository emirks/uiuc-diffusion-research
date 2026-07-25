#!/bin/bash
#SBATCH --job-name=ctt_capbank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# Caption both endpoints of every clip in the 227-clip shared endpoint bank, so S2/S3
# procedural samples can have RENDERED prompts (the bank ships none and the S2/S3 hand-off
# spec produces none).
#
# Deliberately NOT on the HCESC-L40S pool: the S2/S3 delivery agent is rendering there and
# this job has no reason to compete with the thing it is waiting on.
#
#   sbatch --partition=secondary --account=campusclusterusers --gres=gpu:A100:1 \
#          scripts/ctt_v2/mix/job_caption_bank.sbatch
source ~/.bashrc || true
set -eo pipefail
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
R=$DR/.claude/worktrees/bottleneck-branch
PY=$LAB/LTX-2-official/.venv/bin/python

echo "[capbank] host=$(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$R"
# Idempotent: resumes from whatever is already in bank_endpoint_captions.json.
"$PY" scripts/ctt_v2/mix/caption_bank.py --device cuda
echo "[capbank] done $(date)"
