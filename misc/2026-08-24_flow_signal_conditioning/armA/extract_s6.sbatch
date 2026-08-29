#!/bin/bash
#SBATCH --job-name=armA_s6_extract
#SBATCH --account=bhwp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --constraint=taiga
#SBATCH --array=0-95
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=110g
#SBATCH --time=00:45:00
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-08-24_flow_signal_conditioning/armA/logs/%x-%A_%a.out
set -euo pipefail
source /taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/activate
export HF_HOME=/taiga/illinois/eng/cs/jrehg/users/emirkisa/cache/huggingface
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# node-local clip staging (purged at job end); NEVER the 92%-full taiga
export S6_CLIPS_DIR=/tmp/s6clips_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$S6_CLIPS_DIR"
# stagger python imports within a concurrency wave (~24 GPUs); HARDCODED width 96, never derived
sleep $(( (SLURM_ARRAY_TASK_ID % 24) * 8 ))
cd /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-08-24_flow_signal_conditioning/armA
python armA_extract_s6.py extract --shard ${SLURM_ARRAY_TASK_ID}/96 --device cuda
rm -rf "$S6_CLIPS_DIR" || true
