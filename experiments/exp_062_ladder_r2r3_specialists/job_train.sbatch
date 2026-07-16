#!/bin/bash
#SBATCH --job-name=exp062_train
#SBATCH --array=0-10%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:55:00
#SBATCH --output=outputs/logs/slurm/%x-%A_%a.out
# One self-contained R2/R3 specialist per array task: precompute (if needed) -> train.
# Resume-aware (skips DONE, resumes latest ckpt) + chain-safe (submit 2 copies with
# --dependency=afterany to survive 4h/preemption windows). Preprocess+train in one job
# (exp_050 pattern) so no cross-array dependency is needed.
# Usage: sbatch [partition/account/gres flags] job_train.sbatch

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_062_ladder_r2r3_specialists
MODEL=$LAB/cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors
GEMMA=$LAB/cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized

CLASSES=(shadow portal super_fast_run shadow_smoke polygon wireframe animalization color_rain gas_transformation hero_flight illustration_scene)
CLS=${CLASSES[$SLURM_ARRAY_TASK_ID]}
echo "=== class: $CLS (task $SLURM_ARRAY_TASK_ID) ==="
MANIFEST=$EXP/dataset/manifests/$CLS.json
CONFIG=$EXP/configs/$CLS.yaml
PRE=$EXP/dataset/.precomputed/$CLS
OUTDIR=$EXP/../../outputs/training/exp_062_ladder_r2r3_specialists/$CLS
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 1. precompute latents+conditions (idempotent; ICTRANS trigger baked in) ----
NEXP=$(python3 -c "import json;print(len(json.load(open('$MANIFEST'))))")
NLAT=$(find $PRE/latents -name '*.pt' 2>/dev/null | wc -l)
if [ "$NLAT" -lt "$NEXP" ]; then
  uv run --frozen python scripts/process_dataset.py $MANIFEST \
      --resolution-buckets "480x640x121" \
      --model-path $MODEL --text-encoder-path $GEMMA \
      --lora-trigger "ICTRANS" \
      --output-dir $PRE || { echo "=== EXIT 1 (preprocess) ==="; exit 1; }
else
  echo "precompute present ($NLAT/$NEXP latents) — skipping"
fi
NLAT=$(find $PRE/latents -name '*.pt' 2>/dev/null | wc -l)
NCOND=$(find $PRE/conditions -name '*.pt' 2>/dev/null | wc -l)
echo "precomputed: latents=$NLAT conditions=$NCOND (expected $NEXP)"
[ "$NLAT" -ge "$NEXP" ] || { echo "=== EXIT 1 (precompute incomplete) ==="; exit 1; }

# ---- 2. train (resume-aware) ----
if [ -f "$OUTDIR/DONE" ]; then echo "already DONE"; echo "=== EXIT 0 ==="; exit 0; fi
STEPS=$(python3 -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['optimization']['steps'])")
RUN_CONFIG=$CONFIG
LATEST=$(python3 - <<PY
import pathlib, re
out = pathlib.Path("$OUTDIR")
cks = sorted(out.rglob("*step_*.safetensors"), key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1))) if out.exists() else []
print(cks[-1] if cks else "")
PY
)
if [ -n "$LATEST" ]; then
  echo "resuming from $LATEST"
  python3 - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
cfg["model"]["load_checkpoint"] = "$LATEST"
yaml.safe_dump(cfg, open("$OUTDIR/config_resume.yaml", "w"), sort_keys=False)
PY
  RUN_CONFIG=$OUTDIR/config_resume.yaml
fi

uv run --frozen python scripts/train.py $RUN_CONFIG --disable-progress-bars
RC=$?
PADDED=$(printf '%05d' $STEPS)
FINAL=$(find $OUTDIR \( -name "*step_${STEPS}.safetensors" -o -name "*step_${PADDED}.safetensors" \) 2>/dev/null | head -1)
if [ $RC -eq 0 ] && [ -n "$FINAL" ]; then
  touch $OUTDIR/DONE
  echo "training complete: $FINAL"
fi
echo "=== EXIT $RC ==="
exit $RC
