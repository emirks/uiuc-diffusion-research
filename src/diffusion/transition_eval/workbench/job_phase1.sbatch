#!/bin/bash
#SBATCH --job-name=wb_phase1_fit
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
#
# CPU-only: fit the 4-param similarity to every cached flow field (223 clips x 120
# pairs) and emit the corpus residual-energy + texture distributions that §3.3's
# epsilon and §3.2's low-texture threshold are calibrated from. Produces NO
# candidate score — calibration inputs only.
set -o pipefail
source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
WB="$LAB/diffusion-research/.claude/worktrees/metric-workbench"
cd "$WB"
export PYTHONPATH="$WB/src"
python -u -m diffusion.transition_eval.workbench.run_workbench fit-motion
echo "=== EXIT $? ==="
