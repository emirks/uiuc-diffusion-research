#!/bin/bash
# grid v3 — v4 scoring on DeltaAI (canonical recipe of misc/2026-08-19_dualforce_score/score_dualforce.sbatch) with the
# grid-v3 AMENDMENT: --corpus = the 677-clip manifest (styles/pools of the new classes), --reference-corpus = the
# 222-clip manifest the reference_v4 populations are certified on (pin + v4 pack). EffectData-tier arms plan with
# GEN_PREFIX_FRAMES=1 (run_eval.py --mode plan) — the mask must match what the model was given.
#   ARM=dualforce_control_neutral_v3 EVAL=store/evals/0NN_<name>__dai__<date>/<arm> MDIR=<plan dir> sbatch --account=bhwp-dtai-gh misc/2026-09-07_eval_grid_v2/score_v3.sbatch
#SBATCH --job-name=v3_score
#SBATCH --account=bhwp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --constraint=taiga
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=03:00:00
#SBATCH --array=0-15%16
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
set -euo pipefail
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
WT=$DR/.claude/worktrees/eval-v4-cert
MDIR=${MDIR:?plan dir with eval_c*.json (run_eval.py --mode plan)}
OUT=${EVAL:?per-arm eval out dir under store/evals/}
CACHE=$DR/misc/refvfx_baseline/probe/cache
CORPUS=$DR/data/processed/transitions_std121/corpus_manifest.json            # grid v3: 677 clips / 97 classes
REFCORPUS=$DR/data/processed/transitions_std121/corpus_manifest_v1_222.json  # the certified 222-clip pin
TASK=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$OUT" "$DR/outputs/logs/slurm"
sleep $(( (TASK % 16) * 5 ))
module load python/miniforge3_pytorch/2.10.0
source $LAB/envs-aarch64/refvfx/bin/activate
export PYTHONPATH=$WT/src
export HF_HOME=$LAB/cache/huggingface TORCH_HOME=$LAB/cache/torch HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $DR
python -m diffusion.transition_eval.score --manifest $MDIR/eval_c${TASK}.json --corpus $CORPUS --reference-corpus $REFCORPUS \
    --label ${LABEL:-c}${TASK} --out-root $OUT --cache-dir $CACHE
echo "done task=$TASK date=$(date -Is)"
