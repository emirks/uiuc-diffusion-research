#!/bin/bash
# ctt_v2 — refVFX code_based_edits subset: 16 WebDataset shards x ~23.7 GB = 374.7 GB.
# (maxwelljones14/refVFX_dataset, CC-BY-4.0; 136,800 pairs / 2,736 code effect types,
#  ~50 base videos per effect — the same-operator x different-content diagonal.)
#
# hf download is resumable + idempotent, so this script is submitted several times in an
# afterany dependency chain; each run continues where the previous timed out. The final
# run prints SHARDS-COMPLETE when all 16 shards are present at full size.
#
#SBATCH --job-name=ctt_dl_code
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --cpus-per-task=8
#SBATCH --mem=12g
#SBATCH --time=03:59:00
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
curl -sf --max-time 30 https://huggingface.co/api/datasets/maxwelljones14/refVFX_dataset >/dev/null \
  || { echo "[dl] FATAL: no outbound internet from $(hostname)"; exit 42; }

hf download maxwelljones14/refVFX_dataset --repo-type dataset \
  --include "data/code_based_edits/*" \
  --local-dir "$DATA/refvfx"

# verify: 16 shards, each >23e9 bytes (shard-00015 may be smaller; require presence only)
n=$(ls "$DATA/refvfx/data/code_based_edits"/shard-*.tar 2>/dev/null | wc -l)
total=$(du -sb "$DATA/refvfx/data/code_based_edits" | cut -f1)
echo "[dl] shards=$n  bytes=$total"
if [ "$n" -eq 16 ] && [ "$total" -gt 370000000000 ]; then
  echo "[dl] SHARDS-COMPLETE $(date '+%F %T')"
else
  echo "[dl] incomplete — continuation job will resume"
fi
