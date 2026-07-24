#!/bin/bash
#SBATCH --job-name=e079_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48g
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
# exp_079 SupCon-T — train the standalone operator encoder. NO 19B, no generation: an L40S or any
# small GPU is plenty (advisor authorised L40S/scavenger for the whole encoder track, so this never
# competes with the Hopper generation lanes).
#
# Array index -> (arm, seed):  0-2 = E1 x {42,43,44}   3-5 = ablation x {42,43,44}
#
#   sbatch --array=0-5 --partition=secondary --account=campusclusterusers --gres=gpu:1 \
#          --time=01:00:00 experiments/exp_079_standalone_operator_encoder/job_train.sbatch
set -eo pipefail
source ~/.bashrc || true
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
PY_BIN="$LAB/LTX-2-official/.venv/bin/python"
WT=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch
EXP=$WT/experiments/exp_079_standalone_operator_encoder

ARMS=(E1 E1 E1 ablation ablation ablation)
SEEDS=(42 43 44 42 43 44)

mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$WT"
echo "[e079-train] host=$(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Seatbelt: the manipulation latents must exist (the encode job is the dependency).
N=$(find "$EXP/dataset/manip_latents/train" -name '*.pt' 2>/dev/null | wc -l)
if [ "$N" -lt 100 ]; then
  echo "[fatal] only $N train manip latents — run job_encode.sbatch first"; exit 2
fi
echo "[e079-train] train manip latents: $N"

# Two modes. As a job ARRAY, one (arm,seed) per task. As a PLAIN job, all six run back-to-back in
# one allocation — which is what a saturated queue actually rewards: waiting for one slot beats
# waiting for six. Each run is small (8.7M params, latent-space, no 19B), so the bundle still fits
# comfortably in a couple of hours.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  IDX="$SLURM_ARRAY_TASK_ID"
  export ARM="${ARMS[$IDX]}" SEED="${SEEDS[$IDX]}"
  echo "[e079-train] array task $IDX -> arm=$ARM seed=$SEED"
  "$PY_BIN" "$EXP/train.py"
else
  for IDX in "${!ARMS[@]}"; do
    export ARM="${ARMS[$IDX]}" SEED="${SEEDS[$IDX]}"
    if [ -f "$LAB/diffusion-research/outputs/exp_079/${ARM}_seed${SEED}/encoder.pt" ]; then
      echo "[e079-train] skip $ARM seed=$SEED (already trained)"; continue
    fi
    echo "[e079-train] === run $((IDX+1))/6: arm=$ARM seed=$SEED $(date) ==="
    "$PY_BIN" "$EXP/train.py"
  done
fi
echo "[e079-train] done $(date)"
