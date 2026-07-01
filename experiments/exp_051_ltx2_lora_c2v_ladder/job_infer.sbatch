#!/bin/bash
#SBATCH --job-name=exp051_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:55:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Usage: sbatch [flags] job_infer.sbatch <label> [lora_path]
#   e.g. job_infer.sbatch base
#        job_infer.sbatch t2v /path/to/lora_weights_step_02000.safetensors
# Runs the standalone C2V ValidationRunner inference for one ladder arm.

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_051_ltx2_lora_c2v_ladder
LABEL=${1:?usage: job_infer.sbatch <label> [lora_path]}
LORA=$2
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ -n "$LORA" ]; then
  uv run --frozen python $EXP/run_c2v_inference.py --label $LABEL --lora $LORA --wandb
else
  uv run --frozen python $EXP/run_c2v_inference.py --label $LABEL --wandb
fi
RC=$?
echo "=== EXIT $RC ==="
exit $RC
