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

IDX="${SLURM_ARRAY_TASK_ID:-0}"
ARMS=(E1 E1 E1 ablation ablation ablation)
SEEDS=(42 43 44 42 43 44)
export ARM="${ARMS[$IDX]}"
export SEED="${SEEDS[$IDX]}"

mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$WT"
echo "[e079-train] host=$(hostname) arm=$ARM seed=$SEED $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Seatbelt: the manipulation latents must exist (the encode job is the dependency).
N=$(find "$EXP/dataset/manip_latents/train" -name '*.pt' 2>/dev/null | wc -l)
if [ "$N" -lt 100 ]; then
  echo "[fatal] only $N train manip latents — run job_encode.sbatch first"; exit 2
fi
echo "[e079-train] train manip latents: $N"

"$PY_BIN" "$EXP/train.py"
echo "[e079-train] done $(date)"
