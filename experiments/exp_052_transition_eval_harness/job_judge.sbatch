#!/bin/bash
#SBATCH --job-name=exp052_judge
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Usage: sbatch job_judge.sbatch <results_run_dir_rel_to_repo>
# Judge runs in the LTX-2-official venv (torch 2.9) — the diffusion conda env's
# torch 2.5 predates transformers' Gemma3 attention-mask requirement.

source ~/.bashrc
cd "$LAB/diffusion-research"
OUT=${1:?usage: job_judge.sbatch <results_run_dir>}
EXP=experiments/exp_052_transition_eval_harness

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHONPATH=src "$LAB/LTX-2-official/.venv/bin/python" -u $EXP/run_judge.py \
    --manifest $EXP/manifest_exp051.json --out "$OUT"
RC=$?
echo "=== EXIT $RC ==="
exit $RC
