#!/bin/bash
# Certification run (SPEC §6.5) — submit FROM THE REPO ROOT of the checkout
# under certification:  sbatch src/diffusion/transition_eval/certify/job_certify.sbatch
# Requires: bars.yaml frozen; instrument tree clean (the stamp records dirt);
# corpus clips + outputs/ archives resolvable (worktrees: symlinked).
#SBATCH --job-name=eval-v3-certify
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=outputs/eval/certification/slurm-%j.log

set -euo pipefail
source ~/.bashrc                      # $LAB cache redirections
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
MAIN_ROOT="${MAIN_ROOT:-/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research}"

echo "[job] $(date) node=$(hostname) cwd=$PWD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m diffusion.transition_eval.certify.run_certification \
    --corpus data/processed/transitions_std121/corpus_manifest.json \
    --main-root "$MAIN_ROOT"
echo "[job] done $(date)"
