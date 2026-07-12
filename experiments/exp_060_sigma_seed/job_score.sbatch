#!/bin/bash
#SBATCH --job-name=exp060_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# Score the 60 sigma_seed videos with the CERTIFIED harness (eval/v3.0.0
# worktree, clean tree). PYTHONPATH forces worktree code over the env's
# editable install. Shared warm cache is reused (GPU node; corpus clips are
# symlinks resolving to the same inodes -> file_key hits). Adapter arm only
# (--controls off).
set -x
source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
conda activate "$LAB/envs/diffusion"
MAIN=$LAB/diffusion-research
WT=$MAIN/.claude/worktrees/eval-v3.0.0
EXP=$MAIN/experiments/exp_060_sigma_seed
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $WT
PYTHONPATH=$WT/src python -m diffusion.transition_eval.score \
  --manifest $EXP/dataset/eval_manifest.json \
  --corpus $WT/data/processed/transitions_std121/corpus_manifest.json \
  --label adapter \
  --out-root $MAIN/outputs/eval/sigma_seed \
  --cache-dir $MAIN/outputs/eval/cache \
  --controls off
RC=$?
echo "=== EXIT $RC ==="
exit $RC
