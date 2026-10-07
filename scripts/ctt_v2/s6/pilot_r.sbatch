#!/bin/bash
#SBATCH --job-name=s6rpilot
#SBATCH --account=bhwp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=110g
#SBATCH --time=00:45:00
#SBATCH --constraint="taiga"
#SBATCH --output=outputs/ctt_v2/encodes/EFFECTDATA_r832/slurm_logs/pilot_r_%j.out
set -uo pipefail
cd /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
source /taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/activate
export HF_HOME=/taiga/illinois/eng/cs/jrehg/users/emirkisa/cache/huggingface
# transient extracted mp4s -> node-local /tmp (3.9 TB, purged at job end), NOT /taiga
export S6_CLIPS_DIR=/tmp/s6clips_r_${SLURM_JOB_ID}
mkdir -p "$S6_CLIPS_DIR"
echo "[pilot_r] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) clips=$S6_CLIPS_DIR"

echo "===== STAGE 1: encode_s6r pilot (6 clips per NATIVE shape = 24, all 4 crop cases) ====="
python scripts/ctt_v2/s6/encode_s6r.py pilot --per-native-shape 6
PE=$?
echo "[pilot_r] encode pilot exit=$PE"

echo "===== STAGE 2: encode_s6r verify on the pilot outputs (spot-check, count skipped) ====="
python scripts/ctt_v2/s6/encode_s6r.py verify --no-count --n-sample 24
VE=$?
echo "[pilot_r] verify exit=$VE"

echo "===== STAGE 3: decode check (VAE-decode + PSNR vs process_videos resize/crop, 4 clips) ====="
python misc/2026-08-30_s6_reshape/pilot_r832/pilot_decode_check.py
DE=$?
echo "[pilot_r] decode check exit=$DE"

echo "[pilot_r] SUMMARY encode=$PE verify=$VE decode=$DE"
