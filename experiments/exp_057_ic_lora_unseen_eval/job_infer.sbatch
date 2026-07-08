#!/bin/bash
#SBATCH --job-name=exp057_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Usage: sbatch [flags] job_infer.sbatch <label:ic_lora|base> <chunk> <num_chunks> [lora_ckpt]
# Skip-if-exists per quadruple -> requeue-safe.

set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/exp_057_ic_lora_unseen_eval
LABEL=${1:?usage: job_infer.sbatch <label> <chunk> <num_chunks> [lora]}
CHUNK=${2:?chunk}
NCHUNK=${3:?num_chunks}
LORA=${4:-}
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ARGS=(--label "$LABEL" --chunk "$CHUNK" --num-chunks "$NCHUNK")
[ -n "$LORA" ] && ARGS+=(--lora "$LORA")

uv run --frozen python $EXP/run_ic_inference.py "${ARGS[@]}"
RC=$?
echo "=== EXIT $RC ==="
exit $RC
