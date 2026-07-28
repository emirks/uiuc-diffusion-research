#!/bin/bash
#SBATCH --job-name=ctt2_enc_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=01:00:00
#SBATCH --exclude=ccc0424
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#
# CTT v2 encode SMOKE TEST — a handful of clips per stratum, then the shape/fps/cond_clean
# verification.  Proves the payload contract BEFORE the full arrays go out.
#   sbatch --partition=HCESC-L40S-normal --account=hcesc-l40s --gres=gpu:L40S:1 job_smoke.sbatch
# S1 runs in full (33 clips) because it is the only stratum that mixes one- and two-sided
# clips, and both cond_clean branches must be exercised.
set -eo pipefail
source ~/.bashrc || true

LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
WT=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch
PY=$LAB/LTX-2-official/.venv/bin/python
export PYTHONPATH="$LAB/LTX-2-cond-bleed-fix/packages/ltx-trainer/src"
S=$WT/scripts/ctt_v2/encode/encode_strata.py

echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID} $(date)"
nvidia-smi -L || true
test -x "$PY" || { echo "[smoke] FATAL: interpreter $PY not executable"; exit 127; }

"$PY" "$S" encode --strata S2a --shard 0 --limit 2
"$PY" "$S" encode --strata S2b --shard 0 --limit 2
"$PY" "$S" encode --strata S1  --shard 0
"$PY" "$S" encode --strata S4  --shard 0 --limit 2

echo; echo "================ VERIFY ================"
"$PY" "$S" verify --no-count --n-sample 3
echo "[smoke] done $(date)"
