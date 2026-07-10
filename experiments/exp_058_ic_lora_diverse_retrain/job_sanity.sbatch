#!/bin/bash
#SBATCH --job-name=exp058_sanity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# partition/account/gres passed at submit time (queue chosen by live probe)
# Does: full clip preprocess (162 clips, single bucket) -> pair symlink tree
#       + per-pair masks -> 50-step mixed-conditioning sanity train incl.
#       step-0/50 validation (3 samples: two-sided anchor, one-sided
#       prefix-only in-class, held-out prefix-only). Idempotent.

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_058_ic_lora_diverse_retrain
STD=$LAB/diffusion-research/data/processed/transitions_std121
MODEL=$LAB/cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors
GEMMA=$LAB/cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 1. preprocess all training clips (encode-once; pairs are symlinks) ----
uv run --frozen python scripts/process_dataset.py $STD/dataset_exp058.json \
    --resolution-buckets "480x640x121" \
    --model-path $MODEL --text-encoder-path $GEMMA \
    --lora-trigger "ICTRANS" \
    --skip-audio \
    --decode \
    --output-dir $EXP/dataset/.precomputed_clips || { echo "=== EXIT 1 (preprocess) ==="; exit 1; }
NLAT=$(find $EXP/dataset/.precomputed_clips/latents -name '*.pt' | wc -l)
NROWS=$(python3 -c "import json; print(len(json.load(open('$STD/dataset_exp058.json'))))")
echo "clip latents: $NLAT / dataset rows: $NROWS"
[ "$NLAT" -eq "$NROWS" ] || { echo "=== EXIT 1 (preprocess dropped clips: $NLAT != $NROWS) ==="; exit 1; }

# ---- 2. assemble the pair symlink tree + per-pair masks ----
uv run --frozen python $EXP/build_dataset.py --link || { echo "=== EXIT 1 (link) ==="; exit 1; }

# ---- 3. 50-step mixed-conditioning sanity train ----
uv run --frozen python scripts/train.py $EXP/config_ic2_sanity.yaml --disable-progress-bars \
    || { echo "=== EXIT 1 (sanity train) ==="; exit 1; }

echo "=== EXIT 0 ==="
