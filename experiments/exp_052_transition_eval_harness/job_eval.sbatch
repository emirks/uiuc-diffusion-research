#!/bin/bash
#SBATCH --job-name=exp052_eval
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Usage: sbatch experiments/exp_052_transition_eval_harness/job_eval.sbatch [--smoke]
# Phase 1: harness validation exam (style discrimination + lerp floor).
# Phase 2: score the exp_051 C2V ladder through the harness.
# All heavy artifacts are content-hash cached -> reruns skip completed work.

source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
cd "$LAB/diffusion-research"

SMOKE=$1
EXP=experiments/exp_052_transition_eval_harness
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHONPATH=src python -u $EXP/run_validation.py --config $EXP/config.yaml $SMOKE
RC1=$?
echo "=== VALIDATION EXIT $RC1 ==="

PYTHONPATH=src python -u $EXP/run_score.py --config $EXP/config.yaml \
    --manifest $EXP/manifest_exp051.json --label ladder $SMOKE
RC2=$?
echo "=== SCORE EXIT $RC2 ==="

RC=$(( RC1 > RC2 ? RC1 : RC2 ))
echo "=== EXIT $RC ==="
exit $RC
