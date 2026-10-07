#!/bin/bash
#SBATCH --job-name=fs_extract
#SBATCH --account=bgjg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --constraint="taiga"
#SBATCH --requeue
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-09-17_feature_store/logs/%x_%A_%a.out
# Feature-store extraction shard. Env: NS (namespace), POP (population json), NSHARDS.
# Resumable (extract fills only misses); no srun; staggered starts (NCSA); one GPU billed (110g/16 cpus).
set -uo pipefail
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 8 ))
export LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa
source $LAB/envs-aarch64/activate
export HF_HOME=$LAB/cache/huggingface TORCH_HOME=$LAB/cache/torch HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
cd $LAB/diffusion-research
export PYTHONPATH=$PWD/src
echo "[shard] host $(hostname) ns=$NS shard=${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS:-1} pop=$POP start $(date)"
for ns in $NS; do
  python scripts/store_features.py extract "$ns" --population "$POP" --shard "${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS:-1}" --device cuda
  echo "[shard] $ns rc=$? $(date)"
done
echo "[shard] done $(date)"
