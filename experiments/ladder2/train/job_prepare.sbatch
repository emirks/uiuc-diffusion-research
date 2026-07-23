#!/bin/bash
#SBATCH --job-name=ladder2_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# ladder2 — everything between "the token probe finished" and "training can start", in one
# dependency-chainable job so nothing waits on a human:
#   1. token verdict from the probe (pre-registered rule) -> arms.yaml
#   2. rebuild registry.jsonl + the 12 training configs with the winning token
#   3. render + encode the leak-free training captions (Gemma; the only GPU part)
#   4. assemble the 12 preprocessed roots and assert their source counts match
# Idempotent: re-running skips finished work. Submit with dependencies on the probe and the
# cond-clean precompute.
set -eo pipefail
source ~/.bashrc || true
module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
L2=$DR/experiments/ladder2
mkdir -p "$DR/outputs/logs/slurm"
cd "$DR"
echo "[prepare] host=$(hostname) $(date)"
nvidia-smi -L || true

conda activate "$LAB/envs/diffusion"
echo "--- 1. token verdict"
python "$L2/train/token_verdict.py" --apply
TOKEN=$(python -c "import yaml;print(yaml.safe_load(open('$L2/arms.yaml'))['token'])")
echo "[prepare] token = $TOKEN"

echo "--- 2. rebuild registry + configs with the winning token"
python "$L2/build_registry.py"
python "$L2/train/make_configs.py"
python -m pytest tests/test_ladder2_prompts.py -q

echo "--- 3. training text embeddings (leak-free rendered captions)"
PYTHONPATH="$LAB/LTX-2-cond-bleed-fix/packages/ltx-trainer/src" \
  "$LAB/LTX-2-official/.venv/bin/python" "$L2/train/precompute.py" --mode text --device cuda

echo "--- 4. assemble training roots"
python "$L2/train/assemble_roots.py"

touch "$L2/train/PREPARED"
echo "[prepare] done $(date)"
