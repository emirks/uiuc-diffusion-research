#!/bin/bash
#SBATCH --job-name=exp073_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/experiments/exp_073_cond_bleed_fix/logs/slurm/%x-%j.out
# Single-block LoRA training (no_resume:true in config -> clean, no resume RNG artifacts).
# Submit with: sbatch --job-name=exp073_<name> --partition=<p> --account=<a> --gres=gpu:H100:1 \
#   --time=<t> --export=ALL,CONFIG=<abs config path> job_train.sbatch
set -eo pipefail
source ~/.bashrc || true
WT=/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-cond-bleed-fix
VENV=/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv
TR="$WT/packages/ltx-trainer"
CONFIG="${CONFIG:?set CONFIG=<abs path to config.yaml>}"
cd "$TR"
echo "[train] host=$(hostname) config=$CONFIG $(date)"
nvidia-smi -L || true
OUT=$("$VENV/bin/python" -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
echo "[train] output_dir=$OUT"
PYTHONPATH="$TR/src" "$VENV/bin/python" scripts/train.py "$CONFIG" --disable-progress-bars
echo "[train] train.py returned $(date)"
# Completion marker: final checkpoint present
STEPS=$("$VENV/bin/python" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['optimization']['steps'])")
FINAL=$(printf "%05d" "$STEPS")
if ls "$OUT/checkpoints/lora_weights_step_$FINAL.safetensors" >/dev/null 2>&1; then
    touch "$OUT/DONE"
    echo "[train] DONE (step $FINAL checkpoint present)"
else
    echo "[train] WARNING: final checkpoint step $FINAL missing"; exit 2
fi
echo "[train] done $(date)"
