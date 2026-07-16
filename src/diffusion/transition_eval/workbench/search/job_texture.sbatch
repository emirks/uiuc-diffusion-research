#!/bin/bash
#SBATCH --job-name=wb_search_texture
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/slurm/%x-%j.out

# Metric-search GPU track: DINOv2-large [121,1024] CLS for all 223 corpus clips.
# Same short_side=256 decode + DINOv2 processor as base; only the backbone grows.
# Writes ONLY to $WB_CACHE/search. The certified shared cache is untouched.

source ~/.bashrc                 # LAB, HF_HOME, cache redirects
cd "$LAB/diffusion-research/.claude/worktrees/metric-workbench"
# Direct env interpreter: `conda activate <path>` silently fails in batch (stays on
# base anaconda, no torch); the absolute python is bulletproof and carries cu121.
PY="$LAB/envs/diffusion/bin/python"
echo "python: $PY"; "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# canary: the certified shared cache must be byte-identical before and after
SHARED="$LAB/diffusion-research/.claude/worktrees/eval-v3-spec/outputs/eval/cache"
BEFORE=$(ls "$SHARED" | wc -l)
echo "[canary] shared cache entries BEFORE: $BEFORE"

PYTHONPATH=src "$PY" -u -m diffusion.transition_eval.workbench.search.build_texture
RC=$?

AFTER=$(ls "$SHARED" | wc -l)
TOUCHED=$(find "$SHARED" -newermt '-60 minutes' | wc -l)
echo "[canary] shared cache entries AFTER: $AFTER (touched in last 60m: $TOUCHED)"
echo "=== EXIT $RC ==="
exit $RC
