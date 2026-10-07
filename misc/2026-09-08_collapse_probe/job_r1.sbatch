#!/usr/bin/env bash
#SBATCH --job-name=collapse_r1
#SBATCH --partition=ghx4
#SBATCH --constraint=taiga
#SBATCH --account=bgjg-dtai-gh
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=110g
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-09-08_collapse_probe/logs/%x_%A_%a.out
# R1: start anchor only + full prompt. 40 clips per seed (~35 min @ ~45 s/clip). Seeds 42,43,44.
set -uo pipefail
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
CAMP=$DR/misc/2026-09-08_collapse_probe
mkdir -p "$CAMP/logs" "$CAMP/out/r1"
source "$LAB/envs-aarch64/activate"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8

TID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( 42 + TID ))
sleep $(( TID * 8 ))
echo "=== collapse R1 seed=$SEED host=$(hostname) $(date) ==="
time python "$DR/eval_ladder/run_gen.py" \
    --arm base_cond_neutral \
    --extra-registry "$CAMP/reg/r1.jsonl" \
    --seed "$SEED" --chunk 0 --num-chunks 1 \
    --out-root "$CAMP/out/r1"
RC=$?
echo "=== EXIT $RC $(date) ==="
exit $RC
