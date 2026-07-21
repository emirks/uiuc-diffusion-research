#!/bin/bash
#SBATCH --job-name=exp073_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:59:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/experiments/exp_073_cond_bleed_fix/logs/slurm/%x-%j.out
# Score one exp_073 manifest with the v4 instrument (eval-v4-cert worktree). One pinned
# scorer commit for ALL arms (fix/null/original). LABEL via --export=ALL,LABEL=ic3_two_sided_fix
set -x
source ~/.bashrc || true
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
conda activate "$LAB/envs/diffusion"
MAIN=$LAB/diffusion-research
WT=$MAIN/.claude/worktrees/eval-v4-cert
EXP=$MAIN/experiments/exp_073_cond_bleed_fix
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $MAIN
PYTHONPATH=$WT/src python -m diffusion.transition_eval.score \
  --manifest $EXP/dataset/eval/${LABEL:?--export=ALL,LABEL=...}.json \
  --corpus $MAIN/data/processed/transitions_std121/corpus_manifest.json \
  --label $LABEL \
  --out-root $MAIN/outputs/eval/exp_073_cond_bleed_fix_v4 \
  --cache-dir $MAIN/outputs/eval/cache
RC=$?
echo "=== EXIT $RC ==="
exit $RC
