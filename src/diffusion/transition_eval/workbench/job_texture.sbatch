#!/bin/bash
#SBATCH --job-name=wb_texture
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
#
# CPU-only: per-frame texture for §3.2's low-texture definedness gate. Runs in
# parallel with the GPU cache job (no GPU, no shared-cache writes).
set -o pipefail
source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
WB="$LAB/diffusion-research/.claude/worktrees/metric-workbench"
cd "$WB"
export PYTHONPATH="$WB/src"
python -u -m diffusion.transition_eval.workbench.texture
echo "=== EXIT $? ==="
