#!/bin/bash
#SBATCH --job-name=exp065_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:55:00
#SBATCH --output=outputs/logs/slurm/%x-%A_%a.out
# Generation grid v3 (one manifest per model; tiers all at once).
# MANIFEST via --export; array task = (seed, chunk): task = seed_idx*CHUNKS + chunk.
#   base_ext (10 rows):  --export=ALL,MANIFEST=dataset/manifest_base_ext.json,CHUNKS=1 --array=0-2
#   ic3 A/B/C (55 rows): --export=ALL,MANIFEST=dataset/manifest_ic3.json,CHUNKS=4   --array=0-11
#   ic3 X (44 rows):     --export=ALL,MANIFEST=dataset/manifest_ic3_x.json,CHUNKS=3 --array=0-8
# ic3 manifests need exp_064 DONE -> submit with --dependency=afterok:<train tail>
# or rely on the adapter-exists assert (fails fast, requeue-safe).

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_065_ladder_v3_grid
CHUNKS=${CHUNKS:-1}
SEEDS=(42 43 44)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / CHUNKS))]}
CHUNK=$((SLURM_ARRAY_TASK_ID % CHUNKS))
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --frozen python $EXP/run_grid_inference.py \
    --manifest ${MANIFEST:?--export=ALL,MANIFEST=...} \
    --seed $SEED --chunk $CHUNK --num-chunks $CHUNKS
RC=$?
echo "=== EXIT $RC ==="
exit $RC
