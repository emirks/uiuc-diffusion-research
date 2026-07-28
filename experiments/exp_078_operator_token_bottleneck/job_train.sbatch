#!/bin/bash
#SBATCH --job-name=bneck_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# exp_078 operator-token bottleneck — LoRA training / diagnostics.
#
# TRAINER selects which checkout runs, which is the whole point of the equivalence check:
#   TRAINER=bneck    -> $LAB/LTX-2-bneck            (private worktree, carries the surgery)
#   TRAINER=lineage  -> $LAB/LTX-2-cond-bleed-fix   (unmodified; every prior number came from it)
#
#   sbatch --job-name=bneck_equiv_mine --partition=secondary --account=campusclusterusers \
#          --gres=gpu:H100:1 --time=01:30:00 \
#          --export=ALL,CONFIG=equiv_mine,TRAINER=bneck \
#          experiments/exp_078_operator_token_bottleneck/job_train.sbatch
set -eo pipefail
source ~/.bashrc || true
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
EXP=$DR/.claude/worktrees/bottleneck-branch/experiments/exp_078_operator_token_bottleneck
VENV=$LAB/LTX-2-official/.venv

CONFIG="${CONFIG:?set CONFIG=<name in configs/, e.g. equiv_mine>}"
TRAINER="${TRAINER:?set TRAINER=bneck|lineage}"
case "$TRAINER" in
  bneck)   WT=$LAB/LTX-2-bneck ;;
  lineage) WT=$LAB/LTX-2-cond-bleed-fix ;;
  *) echo "[fatal] TRAINER must be 'bneck' or 'lineage', got '$TRAINER'"; exit 2 ;;
esac
TR="$WT/packages/ltx-trainer"
CFG="$EXP/configs/$CONFIG.yaml"

mkdir -p "$DR/outputs/logs/slurm"
echo "[train] host=$(hostname) config=$CONFIG trainer=$TRAINER ($WT) $(date)"
echo "[train] trainer commit: $(cd "$WT" && git rev-parse --short HEAD) on $(cd "$WT" && git branch --show-current)"
nvidia-smi -L || true

# Keep wandb local: these are diagnostics on a shared cluster, no network dependency wanted.
export WANDB_MODE=offline

# Seatbelt: every training root link must resolve. A dangling link is the exact defect that
# left the SHARED ic_gen root unusable (its conditions/ and cond_clean_latents/ trees are 100%
# broken), and the trainer would silently skip the affected samples rather than fail.
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

cd "$TR"
PYTHONPATH="$TR/src" "$VENV/bin/python" scripts/train.py "$CFG"
echo "[train] done $(date)"
