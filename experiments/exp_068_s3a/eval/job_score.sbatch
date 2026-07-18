#!/bin/bash
#SBATCH --job-name=exp068_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:59:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Score one exp_067 Stage-2 eval chunk with the CERTIFIED harness (eval/v3.0.0
# worktree — clean tree at the tag so rows stamp certified). H100-purity
# doctrine: all-H100 lane. LABEL via --export=ALL,LABEL=s2_abc_c0.
set -x
source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
conda activate "$LAB/envs/diffusion"
MAIN=$LAB/diffusion-research
WT=$MAIN/.claude/worktrees/eval-v3.0.0
EXP=$MAIN/experiments/exp_068_s3a
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $MAIN
PYTHONPATH=$WT/src python -m diffusion.transition_eval.score \
  --manifest $EXP/eval/eval_${LABEL:?--export=ALL,LABEL=...}.json \
  --corpus $MAIN/data/processed/transitions_std121/corpus_manifest.json \
  --training $EXP/eval/training_manifest_d.json \
  --label $LABEL \
  --out-root $MAIN/outputs/eval/exp_068_s3a \
  --cache-dir $MAIN/outputs/eval/cache
RC=$?
echo "=== EXIT $RC ==="
exit $RC
