#!/bin/bash
#SBATCH --job-name=d2f_train
#SBATCH --partition=HCESC-H100-normal
#SBATCH --account=hcesc-h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# exp_077 D2-FULL STAGE 5 (train) — FROZEN generalist recipe on the D2 combined root.
# ***HELD***: do NOT submit without explicit owner approval. Resume enabled (ckpt/500) so a
# requeue continues from the last checkpoint.
source ~/.bashrc || true
set -eo pipefail
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
WT=$LAB/LTX-2-cond-bleed-fix
VENV=$LAB/LTX-2-official/.venv
TR="$WT/packages/ltx-trainer"
CONFIG="$DR/experiments/exp_077_synth_stratum/configs/d2_gen.yaml"
mkdir -p "$DR/outputs/logs/slurm"
echo "[train] host=$(hostname) $(date)"
nvidia-smi -L || true

# informational cond_clean seatbelt (non-fatal: the real guards ran in encode/assemble)
ROOT="$DR/outputs/videos/exp_077_synth_stratum_d2/d2full/dataset/roots/d2_combined"
"$VENV/bin/python" - "$ROOT" <<'PY' || echo "[train] seatbelt check skipped/soft-failed (non-fatal)"
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
cc, lat = root / "cond_clean_latents", root / "latents"
n = 0
for f in sorted(cc.rglob("*.pt"))[:5]:
    a = torch.load(f, map_location="cpu", weights_only=True)["latents"]
    b = torch.load(lat / f.relative_to(cc), map_location="cpu", weights_only=True)["latents"]
    assert torch.equal(a[:, :-1], b[:, :-1]), f"{f}: non-suffix frames differ"
    n += 1
print(f"[train] cond_clean seatbelt: prefix identical on {n} samples")
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
