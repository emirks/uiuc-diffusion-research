#!/bin/bash
#SBATCH --job-name=exp053_checks
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# exp_053 checks A/B/C: all-frames M3 exam, motion sanity, adversarial
# near-copy leakage bar. Mostly DINO cache hits + ~26 new videos + LPIPS.

source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
cd "$LAB/diffusion-research"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHONPATH=src python -u experiments/exp_053_eval_harness_v2/run_check.py
RC=$?
echo "=== EXIT $RC ==="
exit $RC
