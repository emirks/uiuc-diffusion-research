#!/bin/bash
#SBATCH --job-name=e079_encode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# exp_079 SupCon-T — pixel-space temporal manipulations -> frozen VAE latents.
#
# Resumable (skips existing .pt), so a preemptible partition is safe. No generation, no 19B DiT ->
# an L40S is plenty (advisor authorised L40S for the whole encoder track).
#
#   sbatch --partition=secondary --account=campusclusterusers --time=02:00:00 \
#          experiments/exp_079_standalone_operator_encoder/job_encode.sbatch
#   # or an L40S lane:
#   sbatch --partition=HCESC-L40S-normal --account=hcesc-l40s --gres=gpu:L40S:1 --time=02:00:00 \
#          experiments/exp_079_standalone_operator_encoder/job_encode.sbatch
set -eo pipefail
source ~/.bashrc || true
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
# The VAE stack (ltx_trainer) needs the official trainer venv (py3.14, has rich/torch/ltx_trainer);
# the research 'diffusion' env lacks rich. This is the same venv the coarse encode + trainer use.
PY_BIN="$LAB/LTX-2-official/.venv/bin/python"
WT=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch
EXP=$WT/experiments/exp_079_standalone_operator_encoder

mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$WT"
echo "[e079-encode] host=$(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# split.json must exist (deterministic; commit it). Rebuild if missing.
[ -f "$EXP/split.json" ] || "$PY_BIN" "$EXP/build_split.py"

"$PY_BIN" "$EXP/encode_manipulations.py"
echo "[e079-encode] done $(date)"
