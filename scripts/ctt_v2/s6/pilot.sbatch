#!/bin/bash
#SBATCH --job-name=s6pilot
#SBATCH --account=bhwp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=110g
#SBATCH --time=00:45:00
#SBATCH --constraint="taiga"
#SBATCH --output=outputs/ctt_v2/encodes/EFFECTDATA/slurm_logs/pilot_%j.out
set -euo pipefail
cd /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
source /taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/activate
export HF_HOME=/taiga/illinois/eng/cs/jrehg/users/emirkisa/cache/huggingface
# transient extracted mp4s -> node-local /tmp (3.9 TB, purged at job end), NOT /taiga
export S6_CLIPS_DIR=/tmp/s6clips_${SLURM_JOB_ID}
mkdir -p "$S6_CLIPS_DIR"
echo "[pilot] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) clips=$S6_CLIPS_DIR"
# NO srun (single-process single-GPU; srun added only launch failures per deltaai-throughput §6)
python scripts/ctt_v2/s6/encode_s6.py pilot --per-shape 6
echo "[pilot] encode_s6 exit=$?"
