#!/bin/bash
#SBATCH --job-name=ladder2_tokprobe_h
#SBATCH --partition=HCESC-H100-normal
#SBATCH --account=hcesc-h100
#SBATCH --gres=gpu:H100:1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# Duplicate of job_probe.sbatch on the lab H100 slot (advisor: race the secondary copy,
# first finisher wins). Runs BOTH seeds sequentially, then the base-accepts-IC-reference
# smoke test (gates trusting any generalist margin). skip-if-exists => harmless overlap
# with the secondary copy.
set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
TRAINER=$LAB/LTX-2-official/packages/ltx-trainer
EXP=$LAB/diffusion-research/eval_ladder/token_probe
cd $TRAINER
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for SEED in 42 43; do
  uv run --frozen python $EXP/run_gen.py --manifest manifest_probe.json --seed $SEED --chunk 0 --num-chunks 1
done
# base-accepts-IC-reference smoke (advisor overturn-risk check)
uv run --frozen python $EXP/run_gen.py --manifest manifest_smoke_ref.json --seed 42 --chunk 0 --num-chunks 1
RC=$?
echo "=== EXIT $RC ==="
exit $RC
