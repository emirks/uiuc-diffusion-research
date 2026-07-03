#!/bin/bash
#SBATCH --job-name=exp053_judge_finish
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out
# CPU-only, API-bound: finish the Gemini judge pass after the free-tier daily
# quota reset (submit with --begin), then merge the summary into a fresh
# ladder_v2 report. 15/24 items are response-cached and skip instantly.

source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
cd "$LAB/diffusion-research"
EXP=experiments/exp_053_eval_harness_v2

PYTHONPATH=src python -u $EXP/run_judge_gemini.py \
    --manifest experiments/exp_052_transition_eval_harness/manifest_exp051.json --label ladder
RC=$?

LATEST=$(ls -d outputs/eval/exp_053/judge_gemini_ladder/run_* | tail -1)
if [ $RC -eq 0 ] && [ -f "$LATEST/judge_summary.json" ]; then
    PYTHONPATH=src python -u $EXP/run_score.py \
        --from-items outputs/eval/exp_052/ladder/run_0001/items.jsonl \
        --judge-summary "$LATEST/judge_summary.json" --label ladder_v2
    RC=$?
fi
echo "=== EXIT $RC ==="
exit $RC
