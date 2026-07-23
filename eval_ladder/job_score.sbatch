#!/bin/bash
#SBATCH --job-name=ladder2_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=03:30:00
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# Score one ladder2 eval chunk with the v4 instrument (eval-v4-cert worktree — owner
# directive 2026-07-20: v4 is the lane instrument). Shared warm feature cache.
#   sbatch --array=0-7 --partition=secondary --account=campusclusterusers \
#          --gres=gpu:H100:1 eval_ladder/job_score.sbatch
# NOTE: source bashrc BEFORE `set -e`, and tolerate its failure. The interactive job-monitoring
# hook in ~/.bashrc shells out to `scontrol` on stale array ids and returns non-zero; under
# `set -eo pipefail` that killed scoring chunks 7 seconds in with an empty log.
source ~/.bashrc || true
set -eo pipefail
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
# Call the env's interpreter DIRECTLY rather than `conda activate`: when several array tasks
# start at once the activation can fail silently and leave system python, which then dies with
# ModuleNotFoundError: torch ~10s in. An absolute interpreter path cannot fail that way.
PY_BIN="$LAB/envs/diffusion/bin/python"
MAIN=$LAB/diffusion-research
WT=$MAIN/.claude/worktrees/eval-v4-cert
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
# Label must be UNIQUE PER PASS: scoring is incremental and re-uses chunk indices, so a bare
# "c$CHUNK" label makes each pass overwrite the previous pass's items.jsonl and silently
# destroy already-scored rows. run_eval report globs */items.jsonl, so extra dirs are free.
PASS="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
# MANIFEST_DIR / SCORE_OUT are env-overridable so side lanes (the convergence diagnostic's
# eval_diag chunks -> outputs/eval/ladder2_diag) reuse this script instead of forking it.
MDIR="${MANIFEST_DIR:-eval_ladder/eval}"
SOUT="${SCORE_OUT:-outputs/eval/ladder2}"
cd "$MAIN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
PYTHONPATH=$WT/src "$PY_BIN" -m diffusion.transition_eval.score \
  --manifest "$MAIN/${MDIR}/eval_c${CHUNK}.json" \
  --corpus  "$MAIN/data/processed/transitions_std121/corpus_manifest.json" \
  --label   "c${CHUNK}_p${PASS}" \
  --out-root "$MAIN/${SOUT}" \
  --cache-dir "$MAIN/outputs/eval/cache"
echo "=== EXIT $? ==="
