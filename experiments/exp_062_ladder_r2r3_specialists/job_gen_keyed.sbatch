#!/bin/bash
#SBATCH --job-name=exp062_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:55:00
#SBATCH --output=outputs/logs/slurm/%x-%A_%a.out
# Amendment 1 keyed generation. MODE (via --export=ALL,MODE=r2r3|r1k|r3x) picks the rung;
# --array is set at submit time to match the class count (classes*3 - 1). One task per
# (class, seed). Skip-if-exists (requeue-safe).
#   r2r3 : keyed R2+R3 (ckpt 250+2000), 10 classes (hero_flight DEFERRED)          -> --array=0-29
#   r1k  : R1K base (no adapter), prefix-only, 9 one_sided classes                  -> --array=0-26
#   r3x  : R3X donor endpoints (ckpt 2000), 8 B8 classes                            -> --array=0-23
#   b1   : hero_flight keyed R2+R3 (deferral lifted: owner validated two_sided
#          2026-07-16; blind==keyed symlink, ckpts 250+2000 on disk)                 -> --array=0-2
# r2r3 + r3x need keyed training (chain with --dependency=afterok:<K2>); r1k is independent.
# Usage: sbatch --array=0-N%W --export=ALL,MODE=<mode> [--dependency=...] [part/acct/gres --requeue] job_gen_keyed.sbatch

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_062_ladder_r2r3_specialists

SEEDS=(42 43 44)
case "$MODE" in
  r2r3) CLASSES=(shadow portal super_fast_run polygon wireframe animalization color_rain gas_transformation illustration_scene shadow_smoke); FLAG="";;
  r1k)  CLASSES=(shadow portal super_fast_run polygon wireframe animalization color_rain gas_transformation illustration_scene); FLAG="--no-adapter";;
  r3x)  CLASSES=(shadow portal polygon wireframe animalization color_rain gas_transformation illustration_scene); FLAG="--r3x";;
  b1)   CLASSES=(hero_flight); FLAG="";;
  r3xext) CLASSES=(hero_flight shadow_smoke super_fast_run); FLAG="--r3x";
          export LADDER_GRID=docs/eval_ladder/ladder_items_v3.json;;      # A2.5 X-extension -> --array=0-8
  *) echo "ERROR: MODE unset/invalid ('$MODE') — pass --export=ALL,MODE=r2r3|r1k|r3x"; exit 2;;
esac
CLS=${CLASSES[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
echo "=== MODE=$MODE class=$CLS seed=$SEED flag='$FLAG' (task $SLURM_ARRAY_TASK_ID) ==="
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --frozen python $EXP/run_c2v_inference.py --class $CLS --seed $SEED $FLAG
RC=$?
echo "=== EXIT $RC ==="
exit $RC
