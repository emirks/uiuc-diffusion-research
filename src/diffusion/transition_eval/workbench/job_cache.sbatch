#!/bin/bash
#SBATCH --job-name=wb_cache_build
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
#
# The metric workbench's ONE GPU job (OPERATIONS §6 step 3). Front-loads all GPU
# work: dense SEA-RAFT flow for 223 clips, 223 rendered-lerp nulls (DINO), the
# corpus ZCA and the anchor/chord distribution. Everything downstream — every
# descriptor variant, exam, ablation and acceptance test — is then
# CPU-on-cached-arrays.
#
# Submit from the WORKTREE root (--output is resolved against the submit dir):
#   cd $LAB/diffusion-research/.claude/worktrees/metric-workbench
#   sbatch src/diffusion/transition_eval/workbench/job_cache.sbatch
#
# Idempotent and safe to requeue: every artifact is skipped when warm and written
# atomically via a .tmp rename, so a preemption can never leave a half-written
# cache entry behind.

set -o pipefail
source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"

WB="$LAB/diffusion-research/.claude/worktrees/metric-workbench"
cd "$WB"

# PYTHONPATH is critical in a worktree: the env's editable install points at the
# main checkout and would silently shadow this branch's code.
export PYTHONPATH="$WB/src"

# All model weights are pre-staged in $LAB/cache (SEA-RAFT + resnet34 + DINOv2),
# so the job never depends on the compute node having internet.
export HF_HUB_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "--- shared-cache canary BEFORE (must be unchanged after: the certified cache is READ-ONLY) ---"
EVC="$LAB/diffusion-research/.claude/worktrees/eval-v3-spec/outputs/eval/cache"
echo "entries=$(ls "$EVC" | wc -l)  touched_today=$(find "$EVC" -type f -newermt "$(date +%Y-%m-%d)" | wc -l)"

python -u -m diffusion.transition_eval.workbench.build_cache
RC=$?

echo "--- shared-cache canary AFTER ---"
echo "entries=$(ls "$EVC" | wc -l)  touched_today=$(find "$EVC" -type f -newermt "$(date +%Y-%m-%d)" | wc -l)"
du -sh "$WB/outputs/eval/workbench_cache" 2>/dev/null

echo "=== EXIT $RC ==="
exit $RC
