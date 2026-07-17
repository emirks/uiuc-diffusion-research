#!/bin/bash
#SBATCH --job-name=exp064_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:55:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# ic3 split-aligned generalist retrain (PLAN Amendment 2 SSA2.4).
# Self-contained: precompute missing clips (19) -> link pair tree -> train.
# Resume-aware (skips DONE, resumes latest ckpt) + chain-safe (submit 2 copies
# with --dependency=afterany). Usage: sbatch [part/acct/gres flags] job_train.sbatch

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_064_ic3_aligned_retrain
STD=$LAB/diffusion-research/data/processed/transitions_std121
CONFIG=$EXP/config_ic3.yaml
MODEL=$LAB/cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors
GEMMA=$LAB/cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUTDIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
if [ -f "$OUTDIR/DONE" ]; then echo "already DONE"; echo "=== EXIT 0 ==="; exit 0; fi

# ---- 1. precompute the missing clips (idempotent) ----
NEED=$(python3 - <<PY
import json, pathlib
rows = json.load(open("$STD/dataset_exp064_missing.json"))
pre = pathlib.Path("$EXP/dataset/.precomputed_clips/latents")
missing = [r for r in rows if not (pre / r["video"].replace(".mp4", ".pt")).exists()]
print(len(missing))
PY
)
if [ "$NEED" -gt 0 ]; then
  uv run --frozen python scripts/process_dataset.py $STD/dataset_exp064_missing.json \
      --resolution-buckets "480x640x121" \
      --model-path $MODEL --text-encoder-path $GEMMA \
      --lora-trigger "ICTRANS" \
      --output-dir $EXP/dataset/.precomputed_clips || { echo "=== EXIT 1 (preprocess) ==="; exit 1; }
else
  echo "precompute present — skipping"
fi

# ---- 2. link pair tree (idempotent, cheap) ----
uv run --frozen python $EXP/build_dataset.py --link || { echo "=== EXIT 1 (link) ==="; exit 1; }

# ---- 3. train (exp_058 resume pattern) ----
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
if [ -n "$FINAL" ]; then touch $OUTDIR/DONE; echo "DONE marker written"; fi
echo "=== EXIT $RC ==="
exit $RC
