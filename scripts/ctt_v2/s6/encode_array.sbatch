#!/bin/bash
#SBATCH --job-name=s6enc
#SBATCH --account=bhwp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=110g
#SBATCH --time=00:45:00
#SBATCH --constraint="taiga"
#SBATCH --array=0-47
#SBATCH --output=outputs/ctt_v2/encodes/EFFECTDATA/slurm_logs/enc_%A_%a.out
set -euo pipefail
# 4 shapes x NSHARDS_PER_SHAPE(12) = 48 tasks.  task -> (shape, shard)
NSPS=12
SHAPE=$(( SLURM_ARRAY_TASK_ID / NSPS ))
SHARD=$(( SLURM_ARRAY_TASK_ID % NSPS ))
# NCSA asks Python jobs be staggered so simultaneous imports don't hammer /sw (deltaai skill)
sleep $(( SLURM_ARRAY_TASK_ID * 8 ))
cd /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
source /taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/activate
export HF_HOME=/taiga/illinois/eng/cs/jrehg/users/emirkisa/cache/huggingface
export S6_CLIPS_DIR=/tmp/s6clips_${SLURM_JOB_ID}   # node-local, purged at job end
mkdir -p "$S6_CLIPS_DIR"
echo "[enc] host=$(hostname) task=$SLURM_ARRAY_TASK_ID shape=$SHAPE shard=$SHARD clips=$S6_CLIPS_DIR"
# pilot (job 3040828) confirmed the largest shape (1248x704x81) encodes on GH200 WITHOUT tiling.
python scripts/ctt_v2/s6/encode_s6.py encode --shape "$SHAPE" --shard "$SHARD" --nshards "$NSPS"
echo "[enc] task=$SLURM_ARRAY_TASK_ID exit=$?"
