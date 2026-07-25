#!/bin/bash
# ctt_v2 — small external-dataset downloads (also the compute-node-internet canary).
#   1. VFXMaster corpus  (8ruceLi/VFXMaster_datasets, 6.6 GB tar.gz, Apache-2.0,
#      9,209 videos / 241 effect classes, per-video captions)
#   2. refVFX I2V_LoRA subset (maxwelljones14/refVFX_dataset, one 12.3 GB shard,
#      CC-BY-4.0, 6,995 pairs from 43 per-effect LoRAs)
# Idempotent: hf download resumes, extraction is guarded.
#
#SBATCH --job-name=ctt_dl_small
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/data/raw/_dl_logs/%x_%j.out

set -eo pipefail
export LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
source ~/.bashrc || true
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
set -u

DATA="$LAB/diffusion-research/data/raw"
echo "[dl] host=$(hostname)  start=$(date '+%F %T')"

# --- internet canary: fail fast + loud if compute nodes have no outbound route
curl -sf --max-time 30 https://huggingface.co/api/datasets/8ruceLi/VFXMaster_datasets >/dev/null \
  || { echo "[dl] FATAL: no outbound internet from $(hostname)"; exit 42; }
echo "[dl] internet OK"

# --- 1. VFXMaster ---
hf download 8ruceLi/VFXMaster_datasets --repo-type dataset --local-dir "$DATA/vfxmaster"
if [ ! -f "$DATA/vfxmaster/extracted/.done" ]; then
  mkdir -p "$DATA/vfxmaster/extracted"
  tar -xzf "$DATA/vfxmaster/data.tar.gz" -C "$DATA/vfxmaster/extracted"
  touch "$DATA/vfxmaster/extracted/.done"
fi
echo "[dl] vfxmaster: $(find "$DATA/vfxmaster/extracted" -name '*.mp4' | wc -l) mp4s extracted"

# --- 2. refVFX I2V_LoRA shard (+ repo metadata) ---
hf download maxwelljones14/refVFX_dataset --repo-type dataset \
  --include "data/I2V_LoRA/*" "metadata.json" "README.md" \
  --local-dir "$DATA/refvfx"
ls -l "$DATA/refvfx/data/I2V_LoRA/"
echo "[dl] DONE $(date '+%F %T')"
