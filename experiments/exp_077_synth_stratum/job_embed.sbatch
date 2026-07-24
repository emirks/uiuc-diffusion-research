#!/bin/bash
#SBATCH --job-name=exp077_embed
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
#SBATCH --error=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# GPU: DINOv2-embed the audit clips + cluster -> effective-K (TASK 0e analysis).
# Submit with RUN=<audit run_dir>.
#   sbatch --export=ALL,RUN=/.../exp_077_synth_stratum_audit/run_0000 job_embed.sbatch
source ~/.bashrc || true
set -eo pipefail
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${RUN:?set RUN=<audit run_dir>}"
mkdir -p "$LAB/diffusion-research/outputs/logs/slurm"
cd "$LAB/diffusion-research/experiments/exp_077_synth_stratum"
echo "[embed] host=$(hostname) run=$RUN $(date)"
nvidia-smi -L || true
python embed_cluster.py --run "$RUN"
echo "[embed] done $(date)"
