#!/usr/bin/env bash
#SBATCH --job-name=collapse_r2r3
#SBATCH --partition=ghx4
#SBATCH --constraint=taiga
#SBATCH --account=bgjg-dtai-gh
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=110g
#SBATCH --time=01:45:00
#SBATCH --array=0-2
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-09-08_collapse_probe/logs/%x_%A_%a.out
# R2+R3: splice this seed's end anchors from R1 output, then generate R2 (full) + R3 (neutral).
# 80 clips per seed (~65 min @ ~45 s/clip). Launch AFTER R1: sbatch --dependency=afterok:<r1_jobid>.
set -uo pipefail
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
CAMP=$DR/misc/2026-09-08_collapse_probe
mkdir -p "$CAMP/logs" "$CAMP/out/r2" "$CAMP/out/r3"
source "$LAB/envs-aarch64/activate"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8

TID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( 42 + TID ))
sleep $(( TID * 8 ))
echo "=== collapse R2R3 seed=$SEED host=$(hostname) $(date) ==="

# 1) splice: cut end anchors + write reg/r2_s$SEED.jsonl, reg/r3_s$SEED.jsonl (idempotent)
echo "--- splice seed=$SEED ---"
python "$CAMP/splice_r1.py" --seed "$SEED" || { echo "SPLICE FAILED"; exit 2; }

# 2) R2 — full prompt, both anchors
echo "--- R2 seed=$SEED ---"
time python "$DR/eval_ladder/run_gen.py" \
    --arm base_cond_neutral \
    --extra-registry "$CAMP/reg/r2_s${SEED}.jsonl" \
    --seed "$SEED" --chunk 0 --num-chunks 1 \
    --out-root "$CAMP/out/r2" || { echo "R2 FAILED"; exit 3; }

# 3) R3 — neutral prompt, same anchors as R2
echo "--- R3 seed=$SEED ---"
time python "$DR/eval_ladder/run_gen.py" \
    --arm base_cond_neutral \
    --extra-registry "$CAMP/reg/r3_s${SEED}.jsonl" \
    --seed "$SEED" --chunk 0 --num-chunks 1 \
    --out-root "$CAMP/out/r3" || { echo "R3 FAILED"; exit 4; }

echo "=== EXIT 0 $(date) ==="
