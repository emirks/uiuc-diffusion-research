#!/bin/bash
#SBATCH --job-name=exp051_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:55:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Usage: sbatch [-J name] [partition/account/gres flags] job_train.sbatch <config_name.yaml>
# Resume-aware: exits fast if DONE; resumes from latest checkpoint if any exist.
# Chain-safe: submit N copies with --dependency=afterany to survive 4h windows.

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_051_ltx2_lora_c2v_ladder
CONFIG=$EXP/${1:?usage: job_train.sbatch <config.yaml>}
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUTDIR=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
STEPS=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['optimization']['steps'])")

if [ -f "$OUTDIR/DONE" ]; then echo "already DONE"; echo "=== EXIT 0 ==="; exit 0; fi

# Resume if checkpoints exist
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

# Mark DONE if the final-step checkpoint exists (trainer zero-pads step numbers)
PADDED=$(printf '%05d' $STEPS)
FINAL=$(find $OUTDIR \( -name "*step_${STEPS}.safetensors" -o -name "*step_${PADDED}.safetensors" \) | head -1)
if [ $RC -eq 0 ] && [ -n "$FINAL" ]; then
  touch $OUTDIR/DONE
  echo "training complete: $FINAL"
fi
echo "=== EXIT $RC ==="
exit $RC
