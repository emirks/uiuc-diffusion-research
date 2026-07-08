#!/bin/bash
#SBATCH --job-name=exp057_eval
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Usage: sbatch experiments/exp_057_ic_lora_unseen_eval/job_eval.sbatch
# Scores the 51 exp_057 quadruples through the exp_053-v2 harness (singleton-
# guarded fork). Reference features are content-hash cached from exp_052/054.

source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
cd "$LAB/diffusion-research"

EXP=experiments/exp_057_ic_lora_unseen_eval
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHONPATH=src python -u $EXP/run_score_ic.py --config $EXP/eval_config.yaml \
    --manifest $EXP/dataset/manifest_ic.json --label quads
RC=$?
echo "=== EXIT $RC ==="
exit $RC
