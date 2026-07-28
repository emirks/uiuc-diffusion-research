#!/bin/bash
#SBATCH --job-name=d2f_assemble
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# exp_077 D2-FULL STAGE 4 (assemble + mix, fan-in) — Gemma text embed once, assemble 3072
# synthetic + real x8, equal-source-count assert, then emit the (UNSUBMITTED) train config.
source ~/.bashrc || true
set -eo pipefail
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
export DEVICE=cuda
mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$LAB/diffusion-research/experiments/exp_077_synth_stratum"
echo "[assemble] host=$(hostname) $(date)"
nvidia-smi -L || true
python assemble_d2full.py
echo "[assemble] emitting the D2 train config (NOT submitting the train)"
python make_d2_train_config.py
echo "[assemble] done $(date)"
