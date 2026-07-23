#!/bin/bash
# Launch the eval-ladder GPU jobs. Run from a Slurm submit host (cc-login3) —
# NOT from a background Jupyter session (no sbatch there). Resume/skip-if-exists
# make every job safe to re-run. See docs/eval_ladder/PLAN.md §6.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # repo root
command -v sbatch >/dev/null || { echo "ERROR: sbatch not found — run from cc-login3"; exit 1; }
P="--partition=HCESC-H100-secondary --account=hcesc-h100 --gres=gpu:1 --requeue"

echo "== A5: 11 R2/R3 specialist trainings (precompute->train array, chained) =="
J1=$(sbatch --parsable $P experiments/exp_062_ladder_r2r3_specialists/job_train.sbatch)
J2=$(sbatch --parsable --dependency=afterany:"$J1" $P experiments/exp_062_ladder_r2r3_specialists/job_train.sbatch)
echo "   train J1=$J1  J2=$J2 (chain)"

echo "== A6: R4/R5 generation — 60 videos now (hero_flight R5 deferred on sidedness) =="
J3=$(sbatch --parsable $P experiments/exp_063_ladder_r4r5_generalist/job_infer.sbatch)
echo "   r4r5 J3=$J3"

cat <<EOF

Submitted. Monitor:  squeue -u emirkisa
                     tail -f outputs/logs/slurm/exp062_train-*_*.out

AFTER all 11 specialists finish (DONE markers in
outputs/training/exp_062_ladder_r2r3_specialists/<class>/), run C1 (R2/R3 gen):

  sbatch --dependency=afterok:$J2 $P \\
      experiments/exp_062_ladder_r2r3_specialists/job_infer.sbatch

DEFERRED (after sidedness validation) — hero_flight R5, 6 videos:
  cd \$LAB/LTX-2-official/packages/ltx-trainer
  for s in 42 43 44; do uv run --frozen python \\
    \$LAB/diffusion-research/experiments/exp_063_ladder_r4r5_generalist/run_ic_inference.py \\
    --seed \$s --include-deferred; done

Scoring stays BLOCKED until sidedness is validated (feeds the S mask).
EOF
