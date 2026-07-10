#!/bin/bash
# exp_059 — LTX-2 inference benchmark. One array task = one arm = one process.
# Partition/account/gres are passed at submit time, e.g.:
#   sbatch --partition=secondary --account=campusclusterusers --gres=gpu:H100:1 \
#          --array=0-7 experiments/exp_059_ltx2_inference_benchmark/job_bench.sbatch
#   sbatch --partition=HCESC-H100-normal --account=hcesc-h100 --gres=gpu:H100:1 ...
# Single arm: --export=ALL,ARM=smoke_dist (no --array needed).
#SBATCH --job-name=exp059_bench
#SBATCH --cpus-per-task=16
#SBATCH --mem=200g
#SBATCH --time=03:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out

source ~/.bashrc   # cache env vars (HF_HOME etc.) — before set -u (/etc/bashrc is not -u clean)
set -euo pipefail

LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
REPO=$LAB/diffusion-research
EXP=$REPO/experiments/exp_059_ltx2_inference_benchmark
RUN_DIR=${RUN_DIR:-$REPO/outputs/videos/exp_059_ltx2_inference_benchmark/run_0001}

ARMS=(dev_720p_eager dev_1080p_eager dist_720p_eager dist_1080p_eager \
      dev_720p_compile dev_1080p_compile dist_720p_compile dist_1080p_compile \
      dist_720p_registry dist_1080p_registry)
if [[ -z "${ARM:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
fi
: "${ARM:?set ARM (via --export) or submit as --array=0-7}"

mkdir -p "$RUN_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True   # torch>=2.9 name
export TORCHINDUCTOR_CACHE_DIR=$RUN_DIR/$ARM/inductor_cache

echo "[job] node=$(hostname) arm=$ARM run_dir=$RUN_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cd "$REPO"
# torch.compile needs Python <=3.13 -> compile arms pass PYBIN=.venv-py312
PYBIN=${PYBIN:-$LAB/LTX-2-official/.venv/bin/python}
exec "$PYBIN" "$EXP/run.py" --arm "$ARM" --run-dir "$RUN_DIR"
