#!/bin/bash
#SBATCH --job-name=d1_assemble
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# D1 STAGE 2 (assemble + mix, fan-in) — text-encode neutral prompt ONCE (Gemma), assemble the
# synthetic samples, mix real x8, equal-count assert, aggregate render audit. Then emit the D1
# train config. Uses the research env (torch + shells out to the trainer venv for Gemma).
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
python assemble_d1.py
echo "[assemble] emitting D1 train config"
python make_d1_train_config.py
echo "[assemble] done $(date)"
