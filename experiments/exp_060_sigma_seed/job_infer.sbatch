#!/bin/bash
#SBATCH --job-name=exp060_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=03:30:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Usage: sbatch -p HCESC-H100-normal -A hcesc-h100 --gres=gpu:H100:1 \
#          experiments/exp_060_sigma_seed/job_infer.sbatch
# All 12 items x 5 seeds = 60 videos in one model load. Skip-if-exists per
# (item,seed) output mp4 -> requeue / preemption-safe restart.
set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_060_sigma_seed
LORA=$LAB/diffusion-research/outputs/training/exp_056_ltx2_ic_lora_transition_transfer/ic/checkpoints/lora_weights_step_03000.safetensors
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run --frozen python $EXP/run.py --lora "$LORA" --seeds 42,43,44,45,46
RC=$?
echo "=== EXIT $RC ==="
exit $RC
