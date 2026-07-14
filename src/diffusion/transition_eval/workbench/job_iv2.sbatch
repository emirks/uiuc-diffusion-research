#!/bin/bash
#SBATCH --job-name=wb_iv2_probes
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=00:45:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out

# The E1' cycle's ONLY GPU job: embed the IV2 probes (hard cuts + their matched lerps).
# ~74 videos x 121 frames. Everything else this cycle is cached-corpus CPU work.
#
# The two IV2 classes are embedded TOGETHER, on THIS device, in THIS job — the
# certification's own probe features are deliberately not reused, so that no
# device-drift signature can separate cuts from lerps for free (PREREG §P6).
#
# Writes ONLY to $WB_CACHE. The certified shared cache is read through
# bundles.ReadOnlyExtractor, which raises rather than compute.

source ~/.bashrc
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"
cd "$LAB/diffusion-research/.claude/worktrees/metric-workbench"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# canary: the certified shared cache must be byte-identical before and after
SHARED="$LAB/diffusion-research/.claude/worktrees/eval-v3-spec/outputs/eval/cache"
BEFORE=$(ls "$SHARED" | wc -l)
echo "[canary] shared cache entries BEFORE: $BEFORE"

PYTHONPATH=src python -u -m diffusion.transition_eval.workbench.build_iv2
RC=$?

AFTER=$(ls "$SHARED" | wc -l)
TOUCHED=$(find "$SHARED" -newermt '-30 minutes' | wc -l)
echo "[canary] shared cache entries AFTER:  $AFTER  (touched in last 30m: $TOUCHED)"
[ "$BEFORE" -eq "$AFTER" ] || echo "[canary] *** SHARED CACHE GREW — INVESTIGATE ***"

echo "=== EXIT $RC ==="
