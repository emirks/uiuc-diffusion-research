#!/bin/bash
#SBATCH --job-name=ctt_v2_cond
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/ctt_v2/conditions/logs/cond_%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/ctt_v2/conditions/logs/cond_%j.err
#
# CTT v2 — Gemma text-conditions encode (advisor A21 Q1).
#
# The `conditions/` tree was recorded as "blocked: S1 media". It is NOT:
# `process_captions.py` reads `media_path` ONLY to name its output and never opens a media
# file, so every stratum whose captions are final can be encoded now.
#
# Embeds are CONTENT-ADDRESSED by the sha256 of their own caption text. A caption is a pure
# function of the endpoints and the shader is unnameable (RULING 9), so clips sharing an
# endpoint pair share a caption verbatim: 18,090 clips carry only 3,270 distinct captions
# (S2a 24.05x, S2b 9.99x). One embed per distinct caption => 3,270 files / ~51.5 GB instead of
# 18,090 / ~285 GB, and many clips point at one embed exactly as S0 already does (139 embeds
# serving 385 samples).
#
# FORBIDDEN FLAGS, deliberately absent:
#   --lora-trigger        PREPENDS a trigger; our captions carry ` sksz.` INLINE already, so
#                         passing it yields two triggers and fails `caption_violations`.
#   --remove-llm-prefixes would mutate certified caption text.
#
# Submit:
#   sbatch --partition=HCESC-L40S-normal --account=hcesc-l40s --gres=gpu:L40S:1 \
#          scripts/ctt_v2/encode/job_encode_conditions.sbatch
set -uo pipefail

LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
W=$LAB/diffusion-research
IN=$W/outputs/ctt_v2/conditions_inputs
OUT=$W/outputs/ctt_v2/conditions/by_caption
LOGS=$W/outputs/ctt_v2/conditions/logs
mkdir -p "$OUT" "$LOGS"

VENV_PY=$LAB/LTX-2-official/.venv/bin/python
ENCODER=$LAB/LTX-2-cond-bleed-fix/packages/ltx-trainer/scripts/process_captions.py
export PYTHONPATH="$LAB/LTX-2-cond-bleed-fix/packages/ltx-trainer/src"
MODEL=$LAB/cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors
GEMMA=$LAB/cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized

STRATA="${STRATA:-S0 S2a S2b S4}"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  strata=$STRATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "NO GPU"; exit 1; }
echo

rc_all=0
for ST in $STRATA; do
  DS=$IN/${ST}_captions.json
  [ -f "$DS" ] || { echo "[$ST] MISSING $DS -- run build_encode_inputs.py --write"; rc_all=1; continue; }
  N=$($VENV_PY -c "import json,sys; print(len(json.load(open('$DS'))))")
  echo "=============== $ST : $N distinct captions ($(date '+%H:%M:%S')) ==============="
  SECONDS=0
  $VENV_PY "$ENCODER" "$DS" \
      --output-dir "$OUT" \
      --model-path "$MODEL" \
      --text-encoder-path "$GEMMA" \
      --caption-column caption \
      --media-column media_path \
      --batch-size 8 \
      --device cuda \
      2>&1 | tee "$LOGS/${ST}_encode.log"
  rc=${PIPESTATUS[0]}
  echo "[$ST] rc=$rc  ${SECONDS}s"
  [ "$rc" -eq 0 ] || rc_all=1
done

echo
echo "=============== verify ==============="
$LAB/envs/diffusion/bin/python $W/scripts/ctt_v2/encode/verify_conditions.py \
    --report "$W/outputs/ctt_v2/conditions/CONDITIONS_REPORT.json"
vrc=$?
echo "[verify] rc=$vrc"
echo "[done] encode rc=$rc_all  verify rc=$vrc"
exit $(( rc_all || vrc ))
