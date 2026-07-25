#!/bin/bash
#SBATCH --job-name=ctt081
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# exp_081 — ctt_v2 masked retrain (T6(a) one-way reference attention).
#
# Always runs the PRIVATE trainer worktree $LAB/LTX-2-bneck, which carries the
# one-way mask surgery. ltx_core still resolves through the venv .pth to the
# pristine LTX-2-official checkout — that is deliberate and is why the mask
# primitive lives in ltx-trainer (see reference_attention.py's module docstring).
#
#   sbatch --job-name=ctt081_thr_bidir --partition=HCESC-L40S-normal \
#          --account=hcesc-l40s --gres=gpu:L40S:1 --time=01:00:00 \
#          --export=ALL,CONFIG=thr_bidir \
#          experiments/exp_081_ctt_v2_masked_retrain/job_train.sbatch
set -eo pipefail
source ~/.bashrc || true
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
EXP=$DR/.claude/worktrees/bottleneck-branch/experiments/exp_081_ctt_v2_masked_retrain
VENV=$LAB/LTX-2-official/.venv
WT=$LAB/LTX-2-bneck
TR="$WT/packages/ltx-trainer"

CONFIG="${CONFIG:?set CONFIG=<name in configs/, e.g. thr_bidir>}"
CFG="$EXP/configs/$CONFIG.yaml"
[ -f "$CFG" ] || { echo "[fatal] no config at $CFG"; exit 2; }

mkdir -p "$DR/outputs/logs/slurm"
echo "[train] host=$(hostname) config=$CONFIG $(date)"
echo "[train] trainer commit: $(cd "$WT" && git rev-parse --short HEAD) on $(cd "$WT" && git branch --show-current)"
nvidia-smi -L || true

export WANDB_MODE=offline

# Seatbelt: the trainer pairs sources by IDENTICAL RELATIVE PATH and SILENTLY SKIPS
# any sample missing from one of them. Unequal counts or a dangling link is therefore
# silent data loss, not an error. Fail loudly here instead.
"$VENV/bin/python" - "$CFG" <<'PY'
import sys, yaml
from pathlib import Path
root = Path(yaml.safe_load(open(sys.argv[1]))["data"]["preprocessed_data_root"])
counts, bad = {}, []
for sub in sorted(p for p in root.iterdir() if p.is_dir()):
    files = list(sub.rglob("*.pt"))
    counts[sub.name] = len(files)
    bad += [str(f) for f in files if not f.resolve().is_file()]
if bad:
    raise SystemExit(f"[fatal] {len(bad)} dangling links, e.g. {bad[:3]}")
if len(set(counts.values())) != 1:
    raise SystemExit(f"[fatal] unequal source counts (trainer would silently drop): {counts}")
print(f"[seatbelt] root OK: {counts}")
PY

# Echo the attention topology actually being trained, so the log itself records
# which side of the T6(a) intervention this run is.
"$VENV/bin/python" - "$CFG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for c in cfg["training_strategy"]["video"].get("conditions", []):
    if c.get("type") == "reference":
        print(f"[topology] reference attention = {c.get('attention', 'bidirectional')}")
PY

cd "$TR"
PYTHONPATH="$TR/src" "$VENV/bin/python" scripts/train.py "$CFG"
echo "[train] done $(date)"
