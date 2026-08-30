#!/bin/bash
#SBATCH --job-name=s6renc
#SBATCH --account=bhwp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=110g
#SBATCH --time=00:45:00
#SBATCH --constraint="taiga"
#SBATCH --array=0-47
#SBATCH --output=outputs/ctt_v2/encodes/EFFECTDATA_r832/slurm_logs/enc_r_%A_%a.out
set -euo pipefail
# 2 target shapes x NSHARDS_PER_SHAPE(24) = 48 tasks.  task -> (shape, shard)
NSPS=24
SHAPE=$(( SLURM_ARRAY_TASK_ID / NSPS ))
SHARD=$(( SLURM_ARRAY_TASK_ID % NSPS ))
# NCSA asks Python jobs be staggered so simultaneous imports don't hammer /sw (deltaai skill)
sleep $(( SLURM_ARRAY_TASK_ID * 8 ))
cd /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
source /taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/activate
export HF_HOME=/taiga/illinois/eng/cs/jrehg/users/emirkisa/cache/huggingface
export S6_CLIPS_DIR=/tmp/s6clips_r_${SLURM_JOB_ID}   # node-local, purged at job end
mkdir -p "$S6_CLIPS_DIR"
echo "[enc] host=$(hostname) task=$SLURM_ARRAY_TASK_ID shape=$SHAPE shard=$SHARD clips=$S6_CLIPS_DIR"
# pilot (job 3049192) confirmed both target shapes (832x512x81 / 512x832x81) encode on GH200 WITHOUT tiling.
python scripts/ctt_v2/s6/encode_s6r.py encode --shape "$SHAPE" --shard "$SHARD" --nshards "$NSPS"
echo "[enc] task=$SLURM_ARRAY_TASK_ID exit=$?"
