#!/usr/bin/env bash
#SBATCH --job-name=null_extract
#SBATCH --partition=ghx4
#SBATCH --constraint=taiga
#SBATCH --account=bhwp-dtai-gh
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=24 --mem=110g
#SBATCH --time=01:00:00
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-09-13_null_default/logs/%x_%A_%a.out
#
# Phase B (SPEC 6): DINO-CLS + TRANS(44-ch) + VAE latents for every manifest clip and its
# landmarks -> $LAB/cache/null_default/features/<clip_id>[__LERP|__CUT50|__FREEZE|__LATLERP].npz
# Resumable (--only-missing). The operator wrote this and STOPS; the DRIVER:
#   1. sets --account (bgjg-dtai-gh or bhwp-dtai-gh, pick by FairShare per deltaai-throughput),
#   2. optionally shards: uncomment --array below and set SHARDS to match (est. ~3-4 GPU-h total,
#      so ~2-3 shards keep each task under the 2 h wall; a single shard likely exceeds 2 h),
#   3. sbatch this file.
#
# Single job (default):        sbatch job_extract.sbatch
# Sharded (edit both lines):   #SBATCH --array=0-2   +   SHARDS=3
#SBATCH --array=0-7
set -uo pipefail
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
CAMP=$DR/misc/2026-09-13_null_default
cd "$DR"
source "$LAB/envs-aarch64/activate"
export HF_HOME=$LAB/cache/huggingface HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

SHARDS=${SHARDS:-8}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "=== null_extract host=$(hostname) shard=${SHARD_ID}/${SHARDS} $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "$CAMP/scripts/extract_features.py" \
    --device cuda \
    --shard ${SHARD_ID}/${SHARDS} \
    --extractors dino,trans,vae \
    --only-missing
RC=$?
echo "=== EXIT $RC $(date) ==="
exit $RC
