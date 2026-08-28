#!/bin/bash
#SBATCH --job-name=s6cond
#SBATCH --account=bhwp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=110g
#SBATCH --time=00:30:00
#SBATCH --constraint="taiga"
#SBATCH --output=outputs/ctt_v2/encodes/EFFECTDATA/slurm_logs/cond_%j.out
set -uo pipefail
# CTT v2 — Gemma text-conditions encode for S6 (EffectData), DeltaAI-native.
# Content-addressed: 28,644 clips carry only 2,000 distinct captions (14.32x), one embed each,
# written to the SHARED by_caption/ tree keyed by sha256(caption)[:16]. Idempotent (skip existing).
# FORBIDDEN flags (deliberately absent): --lora-trigger (our captions carry ' sksz.' inline ->
# double trigger), --remove-llm-prefixes (would mutate certified text).
cd /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
source /taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/activate
export HF_HOME=/taiga/illinois/eng/cs/jrehg/users/emirkisa/cache/huggingface
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa
W=$LAB/diffusion-research
DS=$W/outputs/ctt_v2/conditions_inputs/S6_captions.json
OUT=$W/outputs/ctt_v2/conditions/by_caption
LOGS=$W/outputs/ctt_v2/conditions/logs
mkdir -p "$OUT" "$LOGS"
ENCODER=$W/src/LTX-2-official/packages/ltx-trainer/scripts/process_captions.py
export PYTHONPATH="$W/src/LTX-2-official/packages/ltx-trainer/src"
MODEL=$LAB/cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors
GEMMA=$LAB/cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "NO GPU"; exit 1; }
N=$(python -c "import json;print(len(json.load(open('$DS'))))")
echo "=============== S6 : $N distinct captions ($(date '+%H:%M:%S')) ==============="
SECONDS=0
python "$ENCODER" "$DS" \
    --output-dir "$OUT" \
    --model-path "$MODEL" \
    --text-encoder-path "$GEMMA" \
    --caption-column caption \
    --media-column media_path \
    --batch-size 8 \
    --device cuda 2>&1 | tee "$LOGS/S6_encode.log"
rc=${PIPESTATUS[0]}
echo "[S6] rc=$rc ${SECONDS}s"
exit $rc
