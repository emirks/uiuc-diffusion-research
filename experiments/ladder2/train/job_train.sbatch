#!/bin/bash
#SBATCH --job-name=ladder2_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# ladder2 LoRA training. Resume is ENABLED in every config (checkpoints.no_resume: false) so a
# `secondary` preemption + requeue continues from the last checkpoint (interval 250/500 steps).
#   sbatch --job-name=ladder2_spec_color_rain --partition=secondary \
#          --account=campusclusterusers --gres=gpu:H100:1 --time=02:30:00 \
#          --export=ALL,MODEL=spec_color_rain experiments/ladder2/train/job_train.sbatch
set -eo pipefail
source ~/.bashrc || true
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
WT=$LAB/LTX-2-cond-bleed-fix          # branch cond-bleed-fix = plain recipe + isolation-encode fix
VENV=$LAB/LTX-2-official/.venv
TR="$WT/packages/ltx-trainer"
MODEL="${MODEL:?set MODEL=<arm name, e.g. spec_color_rain>}"
CONFIG="$DR/experiments/ladder2/train/configs/$MODEL.yaml"
mkdir -p "$DR/outputs/logs/slurm"
echo "[train] host=$(hostname) model=$MODEL $(date)"
nvidia-smi -L || true

# seatbelt 6: the root's cond_clean tree really is the isolation encode (or absent by design)
PYTHONPATH="$DR/experiments/ladder2" "$VENV/bin/python" - "$MODEL" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/experiments/ladder2")
import encode_conditioning as ec
L2 = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/experiments/ladder2")
# sidedness is decided per SAMPLE inside smoke_assert (the generalist root mixes both)
print(ec.smoke_assert(L2 / "dataset/roots" / sys.argv[1]))
PY

cd "$TR"
OUT=$("$VENV/bin/python" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
STEPS=$("$VENV/bin/python" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['optimization']['steps'])")
# _find_checkpoint RAISES on a path that is neither file nor dir, so the resume target must
# exist even on a first run; an EMPTY dir makes it return None and training starts at step 0.
mkdir -p "$OUT/checkpoints"
echo "[train] output_dir=$OUT steps=$STEPS"
PYTHONPATH="$TR/src" "$VENV/bin/python" scripts/train.py "$CONFIG" --disable-progress-bars
FINAL=$(printf "%05d" "$STEPS")
if ls "$OUT/checkpoints/lora_weights_step_$FINAL.safetensors" >/dev/null 2>&1; then
    touch "$OUT/DONE"; echo "[train] DONE (step $FINAL)"
else
    echo "[train] WARNING: final checkpoint step $FINAL missing"; exit 2
fi
echo "[train] done $(date)"
