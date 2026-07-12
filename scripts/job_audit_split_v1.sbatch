#!/bin/bash
#SBATCH --job-name=split_v1_audit
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=02:30:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Near-duplicate scan for split v1 (M2a copy score, certified worktree code).
# Submit from repo root: sbatch scripts/job_audit_split_v1.sbatch

source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
cd "$LAB/diffusion-research"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
WT="$LAB/diffusion-research/.claude/worktrees/eval-v3-spec"
PYTHONPATH="$WT/src" python -u scripts/audit_split_v1.py
echo "=== EXIT $? ==="
