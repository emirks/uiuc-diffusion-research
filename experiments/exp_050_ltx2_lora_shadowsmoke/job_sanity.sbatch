#!/bin/bash
#SBATCH --job-name=exp050_sanity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# partition/account/gres passed at submit time (queue chosen by live probe)

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_050_ltx2_lora_shadowsmoke
MODEL=$LAB/cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors
GEMMA=$LAB/cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 1. one-sample metadata (first portrait clip) ----
mkdir -p $EXP/overfit
python3 - <<'PY'
import json
exp = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/experiments/exp_050_ltx2_lora_shadowsmoke"
data = json.load(open(f"{exp}/dataset/dataset.json"))
json.dump([data[0]], open(f"{exp}/overfit/one.json", "w"), indent=2)
PY

# ---- 2. one-sample preprocess at target resolution ----
uv run --frozen python scripts/process_dataset.py $EXP/overfit/one.json \
    --resolution-buckets "480x640x121" \
    --model-path $MODEL --text-encoder-path $GEMMA \
    --lora-trigger "SHDWSMK" \
    --output-dir $EXP/overfit/.precomputed || { echo "=== EXIT 1 (preprocess) ==="; exit 1; }
ls -la $EXP/overfit/.precomputed/latents/ $EXP/overfit/.precomputed/conditions/

# ---- 3. 50-step sanity train (incl. validation sample at step 50) ----
uv run --frozen python scripts/train.py $EXP/config_sanity.yaml --disable-progress-bars \
    || { echo "=== EXIT 1 (sanity train) ==="; exit 1; }

# ---- 4. resume test: continue from latest checkpoint for 10 more steps ----
python3 - <<'PY'
import yaml
exp = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/experiments/exp_050_ltx2_lora_shadowsmoke"
cfg = yaml.safe_load(open(f"{exp}/config_sanity.yaml"))
cfg["model"]["load_checkpoint"] = f"{exp}/overfit/outputs/sanity"
cfg["optimization"]["steps"] = 60
cfg["validation"]["interval"] = None
cfg["checkpoints"]["interval"] = None
cfg["output_dir"] = f"{exp}/overfit/outputs/sanity_resume"
yaml.dump(cfg, open(f"{exp}/overfit/config_resume.yaml", "w"))
PY
uv run --frozen python scripts/train.py $EXP/overfit/config_resume.yaml --disable-progress-bars \
    || { echo "=== EXIT 1 (resume test) ==="; exit 1; }

# ---- 5. FULL preprocess: all 10 clips, 3 AR buckets, with decode verification ----
uv run --frozen python scripts/process_dataset.py $EXP/dataset/dataset.json \
    --resolution-buckets "480x640x121;640x480x121;576x576x121" \
    --model-path $MODEL --text-encoder-path $GEMMA \
    --lora-trigger "SHDWSMK" \
    --decode \
    --output-dir $EXP/dataset/.precomputed || { echo "=== EXIT 1 (full preprocess) ==="; exit 1; }

echo "latents: $(ls $EXP/dataset/.precomputed/latents/ | wc -l), conditions: $(ls $EXP/dataset/.precomputed/conditions/ | wc -l)"
ls $EXP/dataset/.precomputed/
echo "=== EXIT 0 ==="
