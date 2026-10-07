#!/bin/bash
# EffectData gapper screen — cache warm-up + kernel ceilings for the screen classes (GPU, refvfx env, cert worktree).
#SBATCH --job-name=eds_ceil
#SBATCH --partition=ghx4
#SBATCH --constraint=taiga
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=02:30:00
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-09-08_ed_gapper_screen/logs/%x-%A_%a.out
set -uo pipefail
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa; DR=$LAB/diffusion-research; WT=$DR/.claude/worktrees/eval-v4-cert
N=${NSHARDS:?}; TASK=${SLURM_ARRAY_TASK_ID:-0}; sleep $(( TASK * 8 ))
module load python/miniforge3_pytorch/2.10.0; source $LAB/envs-aarch64/refvfx/bin/activate
export PYTHONPATH=$WT/src HF_HOME=$LAB/cache/huggingface TORCH_HOME=$LAB/cache/torch HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $DR; python scripts/ed_screen/warm_ceilings.py --shard $TASK --num-shards $N; rc=$?; echo "[eds_ceil] done rc=$rc $(date -Is)"; exit $rc
