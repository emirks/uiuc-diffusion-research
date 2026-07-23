#!/bin/bash
#SBATCH --job-name=ladder2_tokprobe
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --gres=gpu:H100:1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:59:00
#SBATCH --output=outputs/logs/slurm/%x-%A_%a.out
# ladder2 token probe: base model (no adapter), prefix-only, prompt {S1}.{token}. vs {S1}.
# array task = seed index. Seeds 42/43 (noise band). skip-if-exists.
set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/experiments/ladder2/token_probe
SEEDS=(42 43)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run --frozen python $EXP/run_gen.py --manifest manifest_probe.json --seed $SEED --chunk 0 --num-chunks 1
RC=$?
echo "=== EXIT $RC ==="
exit $RC
