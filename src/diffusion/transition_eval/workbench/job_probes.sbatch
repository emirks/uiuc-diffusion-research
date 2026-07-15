#!/bin/bash
#SBATCH --job-name=wb_probes
#SBATCH --partition=HCESC-H100-secondary
#SBATCH --account=hcesc-h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/slurm/%x-%j.out
#
# GPU pass 2: flow for the §3.4 acceptance probes (time-reversed camera-tagged
# clips + injected-trajectory warps of the most static clips). Idempotent, atomic
# writes, requeue-safe.
set -o pipefail
source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
WB="$LAB/diffusion-research/.claude/worktrees/metric-workbench"
cd "$WB"
export PYTHONPATH="$WB/src"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name --format=csv,noheader
python -u -m diffusion.transition_eval.workbench.build_probes
echo "=== EXIT $? ==="
