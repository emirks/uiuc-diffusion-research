#!/bin/bash
#SBATCH --job-name=exp072_pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:59:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Score one exp_072 pool-reference chunk with the v4.0.0 instrument (eval-v4-cert
# worktree — owner directive 2026-07-20: v4 is the lane instrument; rows also carry
# the app_ref_v3 bridge field). H100 lane, shared warm cache.
# LABEL via --export=ALL,LABEL=pool_pilot.
set -x
source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
conda activate "$LAB/envs/diffusion"
MAIN=$LAB/diffusion-research
WT=$MAIN/.claude/worktrees/eval-v4-cert
EXP=$MAIN/experiments/exp_072_pool_reference_rescore
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $MAIN
PYTHONPATH=$WT/src python -m diffusion.transition_eval.score \
  --manifest $EXP/dataset/eval_${LABEL:?--export=ALL,LABEL=...}.json \
  --corpus $MAIN/data/processed/transitions_std121/corpus_manifest.json \
  --label $LABEL \
  --out-root $MAIN/outputs/eval/exp_072_pool_v4 \
  --cache-dir $MAIN/outputs/eval/cache
RC=$?
echo "=== EXIT $RC ==="
exit $RC
