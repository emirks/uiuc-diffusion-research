#!/bin/bash
# grid v3 generation through eval_ladder/run_gen.py on DeltaAI (contract v2: --out-root IS the store gen subentry).
# One array task = (seed, chunk): seed = 42 + id / NCHUNKS, chunk = id % NCHUNKS. Resumable (skip-if-exists).
# STACK selects the inference packages: "official" = the venv's editable LTX-2-official (base arms, ic_gen bidirectional,
# as misc/base_arms/job_gen.sbatch); "ctt" = src/LTX-2-ctt-v2-train packages (one_way IC-LoRA arms, as the DCG sweep and
# the dualforce evals). GEN_FRAMES / GEN_PREFIX_FRAMES pass through to run_gen (EffectData tier: 81 / 1).
#SBATCH --job-name=v3_gen
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
ARM="${ARM:?}"; REG="${REG:?stamped registry}"; OUT="${OUT:?store gen subentry}"; NCHUNKS="${NCHUNKS:?}"
STACK="${STACK:-official}"; RANK="${RANK:-32}"; ALPHA="${ALPHA:-32}"
ID=${SLURM_ARRAY_TASK_ID:-0}; SEED=$(( 42 + ID / NCHUNKS )); CHUNK=$(( ID % NCHUNKS ))
sleep $(( (ID % 16) * 8 ))
source "$LAB/envs-aarch64/activate"
if [ "$STACK" = "ctt" ]; then
  CTT=$DR/src/LTX-2-ctt-v2-train/packages
  export PYTHONPATH=$CTT/ltx-trainer/src:$CTT/ltx-core/src:$CTT/ltx-pipelines/src
fi
export HF_HOME=$LAB/cache/huggingface TOKENIZERS_PARALLELISM=false
mkdir -p "$DR/$OUT/videos"
cd "$DR"
echo "[v3-gen] host=$(hostname) arm=$ARM seed=$SEED chunk=$CHUNK/$NCHUNKS stack=$STACK frames=${GEN_FRAMES:-121} prefix=${GEN_PREFIX_FRAMES:-9} $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python eval_ladder/run_gen.py --arm "$ARM" --seed "$SEED" --chunk "$CHUNK" --num-chunks "$NCHUNKS" \
    --extra-registry "$REG" --out-root "$OUT" --videos-dir --rank "$RANK" --alpha "$ALPHA"
rc=$?
echo "[v3-gen] done $(date -Is)"
exit $rc
