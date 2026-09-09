#!/usr/bin/env bash
#SBATCH --job-name=collapse_score
#SBATCH --partition=ghx4
#SBATCH --constraint=taiga
#SBATCH --account=bgjg-dtai-gh
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=24 --mem=110g
#SBATCH --time=00:30:00
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-09-08_collapse_probe/logs/%x_%j.out
# Score R1/R2/R3 with the continuous instrument + DINO realized change. Compute node because login-node
# cv2 decoding intermittently returns garbage frames under load. Launch with --dependency=afterok:<r2r3 jobid>.
set -uo pipefail
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
CAMP=$DR/misc/2026-09-08_collapse_probe
cd "$DR"
source "$LAB/envs-aarch64/activate"
export HF_HOME=$LAB/cache/huggingface HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
echo "=== collapse score host=$(hostname) $(date) ==="
python "$CAMP/preflight.py" --post-r1 | tail -2
python "$CAMP/score_probe.py"
RC=$?
echo "=== EXIT $RC $(date) ==="
exit $RC
