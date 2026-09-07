#!/bin/bash
# grid v3 null-operator guidance generation (scripts/grid_v3/gen_dcg_v3.py) on DeltaAI — the DCG sweep's stack
# (src/LTX-2-ctt-v2-train packages, one_way) and deployed config; one array task = (seed, chunk); resumable.
#SBATCH --job-name=v3_dcg
#SBATCH --partition=ghx4
#SBATCH --constraint=taiga
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --requeue
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-09-07_eval_grid_v2/gen/logs/%x-%A_%a.out
set -uo pipefail
source ~/.bashrc || true
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
ARM="${ARM:?}"; REG="${REG:?}"; OUT="${OUT:?}"; NCHUNKS="${NCHUNKS:?}"; W="${W:-6.0}"
ID=${SLURM_ARRAY_TASK_ID:-0}; SEED=$(( 42 + ID / NCHUNKS )); CHUNK=$(( ID % NCHUNKS ))
sleep $(( (ID % 16) * 8 ))
source "$LAB/envs-aarch64/activate"
CTT=$DR/src/LTX-2-ctt-v2-train/packages
export PYTHONPATH=$CTT/ltx-trainer/src:$CTT/ltx-core/src:$CTT/ltx-pipelines/src
export HF_HOME=$LAB/cache/huggingface TOKENIZERS_PARALLELISM=false
export DCG_SCRATCH=/tmp/$USER/dcg_v3_${SLURM_JOB_ID:-0}
mkdir -p "$DR/$OUT/videos" "$DCG_SCRATCH"
cd "$DR"
echo "[v3-dcg] host=$(hostname) arm=$ARM w=$W seed=$SEED chunk=$CHUNK/$NCHUNKS frames=${GEN_FRAMES:-121} prefix=${GEN_PREFIX_FRAMES:-9} $(date -Is)"
python scripts/grid_v3/gen_dcg_v3.py --registry "$REG" --arm "$ARM" --w "$W" --seed "$SEED" --chunk "$CHUNK" --num-chunks "$NCHUNKS" --out-root "$OUT"
rc=$?
echo "[v3-dcg] done $(date -Is)"
exit $rc
