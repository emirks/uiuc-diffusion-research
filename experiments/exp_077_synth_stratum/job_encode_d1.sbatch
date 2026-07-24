#!/bin/bash
#SBATCH --job-name=d1_encode
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# D1 STAGE 2 (encode) — VAE-encode latents + cond_clean per SHARD on an L40S (keeps H100s free).
# Idempotent (process_videos + cond_clean skip existing). JOB ARRAY.
#   sbatch --array=0-5%6 --export=ALL,NSHARDS=6 job_encode_d1.sbatch
source ~/.bashrc || true
set -eo pipefail
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
WT=$LAB/LTX-2-cond-bleed-fix
VENV=$LAB/LTX-2-official/.venv
export SHARD=${SLURM_ARRAY_TASK_ID:-0}
export NSHARDS=${NSHARDS:-1}
export DEVICE=cuda
mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$LAB/diffusion-research/experiments/exp_077_synth_stratum"
echo "[encode] host=$(hostname) shard=$SHARD/$NSHARDS $(date)"
nvidia-smi -L || true
PYTHONPATH="$WT/packages/ltx-trainer/src" "$VENV/bin/python" encode_d1.py
echo "[encode] shard $SHARD done $(date)"
