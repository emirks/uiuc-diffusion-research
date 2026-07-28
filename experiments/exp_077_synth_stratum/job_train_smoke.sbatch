#!/bin/bash
#SBATCH --job-name=exp077_train
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=03:59:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# GPU: 50-step SMOKE IC-LoRA training on the synthetic root (TASK 0d step 6).
# Submit with RUN=<render run_dir> (make_train_config.py must have written configs/synth_smoke.yaml).
#   sbatch --export=ALL,RUN=/.../run_0000 job_train_smoke.sbatch
source ~/.bashrc || true
set -eo pipefail
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
WT=$LAB/LTX-2-cond-bleed-fix
VENV=$LAB/LTX-2-official/.venv
TR="$WT/packages/ltx-trainer"
: "${RUN:?set RUN=<render run_dir>}"
CONFIG="$DR/experiments/exp_077_synth_stratum/configs/synth_smoke.yaml"
mkdir -p "$DR/outputs/logs/slurm"
echo "[train] host=$(hostname) run=$RUN $(date)"
nvidia-smi -L || true

# seatbelt: the root's cond_clean tree really is the isolation encode (per-sample check)
ROOT="$RUN/dataset/roots/synth_smoke"
PYTHONPATH="$DR/eval_ladder" "$VENV/bin/python" - "$ROOT" <<'PY'
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
cc, lat = root / "cond_clean_latents", root / "latents"
n = 0
for f in sorted(cc.rglob("*.pt")):
    a = torch.load(f, map_location="cpu", weights_only=True)["latents"]
    b = torch.load(lat / f.relative_to(cc), map_location="cpu", weights_only=True)["latents"]
    assert torch.equal(a[:, :-1], b[:, :-1]), f"{f.name}: non-suffix frames differ"
    assert not torch.equal(a[:, -1:], b[:, -1:]), f"{f.name}: suffix anchor NOT corrected"
    n += 1
    if n >= 3:
        break
print(f"[smoke] cond_clean verified on {n} samples (suffix anchor corrected, prefix identical)")
PY

cd "$TR"
OUT=$("$VENV/bin/python" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
STEPS=$("$VENV/bin/python" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['optimization']['steps'])")
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
