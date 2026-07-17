#!/bin/bash
#SBATCH --job-name=exp066_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:59:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Score one ladder-v3 eval manifest with the CERTIFIED harness.
#   LABEL (required, via --export=ALL,LABEL=base_c0) -> dataset/eval_${LABEL}.json
#   TRAIN (optional) -> --training dataset/${TRAIN}.json (ic3 arms use
#   training_manifest_ic3.json for tier joins + M2c copy-vs-training)
# Code = eval-v3-spec worktree (26e6e72): byte-identical to tag eval/v3.0.0
# (docs-only diff, verified) + amendment records in-tree. Corpus = MAIN's
# corrected manifest (222 clips, amendments 2-3). cwd = MAIN so the manifests'
# relative data/ paths resolve. GPU node only (shared cache discipline).
set -x
source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
conda activate "$LAB/envs/diffusion"
MAIN=$LAB/diffusion-research
WT=$MAIN/.claude/worktrees/eval-v3-spec
EXP=$MAIN/experiments/exp_066_ladder_v3_scoring
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $MAIN
TRAIN_FLAG=""
[ -n "$TRAIN" ] && TRAIN_FLAG="--training $EXP/dataset/${TRAIN}.json"
PYTHONPATH=$WT/src python -m diffusion.transition_eval.score \
  --manifest $EXP/dataset/eval_${LABEL:?--export=ALL,LABEL=...}.json \
  --corpus $MAIN/data/processed/transitions_std121/corpus_manifest.json \
  $TRAIN_FLAG \
  --label $LABEL \
  --out-root $MAIN/outputs/eval/ladder_v3 \
  --cache-dir $MAIN/outputs/eval/cache
RC=$?
echo "=== EXIT $RC ==="
exit $RC
