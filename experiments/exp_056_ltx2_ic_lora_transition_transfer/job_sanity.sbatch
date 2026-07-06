#!/bin/bash
#SBATCH --job-name=exp056_sanity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# partition/account/gres passed at submit time (queue chosen by live probe)
# Does: full clip preprocess (47 clips, single bucket) -> pair symlink tree
#       -> 50-step IC-LoRA sanity train incl. step-0/50 validation (3 samples
#       with reference+prefix+suffix). Idempotent: preprocess resumes, links rebuild.

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_056_ltx2_ic_lora_transition_transfer
STD=$LAB/diffusion-research/data/processed/transitions_std121
MODEL=$LAB/cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors
GEMMA=$LAB/cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 1. preprocess all 47 standardized clips (encode-once; pairs are symlinks) ----
uv run --frozen python scripts/process_dataset.py $STD/dataset.json \
    --resolution-buckets "480x640x121" \
    --model-path $MODEL --text-encoder-path $GEMMA \
    --lora-trigger "ICTRANS" \
    --skip-audio \
    --decode \
    --output-dir $EXP/dataset/.precomputed_clips || { echo "=== EXIT 1 (preprocess) ==="; exit 1; }
echo "clip latents: $(find $EXP/dataset/.precomputed_clips/latents -name '*.pt' | wc -l)"
echo "clip conditions: $(find $EXP/dataset/.precomputed_clips/conditions -name '*.pt' | wc -l)"

# ---- 2. assemble the 131-pair symlink tree ----
python3 $EXP/build_dataset.py --link || { echo "=== EXIT 1 (link) ==="; exit 1; }

# ---- 3. 50-step sanity train (reference+prefix+suffix; validation at step 0 and 50) ----
uv run --frozen python scripts/train.py $EXP/config_ic_sanity.yaml --disable-progress-bars \
    || { echo "=== EXIT 1 (sanity train) ==="; exit 1; }

echo "=== EXIT 0 ==="
