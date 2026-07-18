#!/bin/bash
#SBATCH --job-name=exp069_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=06:30:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# exp_069_s3b Stage-3: residual-reference IC-LoRA generalist (transition strategy,
# branch transition-strategy @ 218528b on LTX-2-official; base 7809842).
# Reuses exp_064's preprocessed pair dataset read-only. Resume-aware + chain-safe
# (submit 2 copies with --dependency=afterany for preemptible queues).
set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_069_s3b
CONFIG=$EXP/config_s3b.yaml
cd $LAB/diffusion-research && mkdir -p outputs/logs/slurm
cd $TRAINER
git -C $LAB/LTX-2-official log --oneline -1
git -C $LAB/LTX-2-official rev-parse --abbrev-ref HEAD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUTDIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
if [ -f "$OUTDIR/DONE" ]; then echo "already DONE"; echo "=== EXIT 0 ==="; exit 0; fi
mkdir -p $OUTDIR

# ---- train (exp_058/064 resume pattern) ----
RUN_CONFIG=$CONFIG
LATEST=$(python3 - <<PY
import pathlib, re
out = pathlib.Path("$OUTDIR")
cks = sorted(out.rglob("*step_*.safetensors"), key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))
print(cks[-1] if cks else "")
PY
)
if [ -n "$LATEST" ]; then
  echo "resuming from $LATEST"
  python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
cfg["model"]["load_checkpoint"] = "$LATEST"
yaml.dump(cfg, open("$OUTDIR/config_resume.yaml", "w"))
PY
  RUN_CONFIG=$OUTDIR/config_resume.yaml
fi

uv run --frozen python scripts/train.py $RUN_CONFIG --disable-progress-bars
RC=$?

STEPS=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['optimization']['steps'])")
FINAL=$(find $OUTDIR -name "*step_0*${STEPS}.safetensors" 2>/dev/null | head -1)
if [ $RC -eq 0 ] && [ -n "$FINAL" ]; then touch $OUTDIR/DONE; fi
echo "=== EXIT $RC ==="
