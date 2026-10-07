#!/bin/bash
#SBATCH --job-name=fs_lens
#SBATCH --account=bgjg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --constraint="taiga"
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-09-17_feature_store/logs/%x_%A_%a.out
# Competitor-lens pass shard: drives their_metrics/score_batch.py --store over the grid-v3 population.
# Env: NSHARDS (array width; must equal --array=0-$((NSHARDS-1))), EVAL_ID (optional, defaults inside).
# Resumable (skip-if-exists per gen JSON); no srun; staggered starts (NCSA); one GPU billed (110g/16 cpus).
# With the four per-video lens namespaces (clip_b32/videoprism/raft_mag/cotracker3) already extracted,
# the only model loads are the fresh input-start-frame CLIP/VideoPrism embeds. Collect + aesthetic +
# INDEX run AFTER the array (CPU, one shot) — see OP4_REPORT.md.
set -uo pipefail
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 8 ))
export LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa
source $LAB/envs-aarch64/activate
export HF_HOME=$LAB/cache/huggingface TORCH_HOME=$LAB/cache/torch HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
cd $LAB/diffusion-research
export PYTHONPATH=$PWD/src
echo "[shard] host $(hostname) shard=${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS:-1} eval=${EVAL_ID:-<default>} start $(date)"
python scripts/lens_pass_gridv3.py \
  ${EVAL_ID:+--eval-id "$EVAL_ID"} \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NSHARDS:-1}" --device cuda
echo "[shard] rc=$? $(date)"
