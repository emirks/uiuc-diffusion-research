#!/bin/bash
#SBATCH --job-name=exp061_r0r1
#SBATCH --partition=HCESC-H100-normal
#SBATCH --account=hcesc-h100
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=05:00:00
#SBATCH --array=0-3%2
#SBATCH --output=outputs/logs/slurm/%x-%A_%a.out
# exp_061 R0/R1 baseline generation. 4 array tasks = {r0,r1} x 2 chunks,
# max 2 GPUs concurrent (courtesy). Skip-if-exists per output mp4 -> resumable
# (resubmit the same array to fill any gaps).
# Submit from repo root: sbatch experiments/exp_061_ladder_r0_r1/job_infer.sbatch
set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_061_ladder_r0_r1

ARMS=(r0 r0 r1 r1)
CHUNKS=(0 1 0 1)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
CHUNK=${CHUNKS[$SLURM_ARRAY_TASK_ID]}

cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run --frozen python $EXP/run.py --arm $ARM --chunk $CHUNK --num-chunks 2 --seeds 42,43,44
RC=$?
echo "=== EXIT $RC ==="
exit $RC
