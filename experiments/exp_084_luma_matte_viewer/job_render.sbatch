#!/bin/bash
#SBATCH --job-name=exp084_luma
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-ONLY: moderngl on Mesa llvmpipe via EGL, plus numpy/scipy/cv2 matte
# synthesis. No GPU requested. NOT submitted to `secondary` — another session
# sweeps that partition.

# NOTE: source ~/.bashrc BEFORE `set -u` — /etc/bashrc references an unbound
# BASHRCSOURCED and would abort the job.
source ~/.bashrc
set -eo pipefail

module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"

# Force the Mesa software EGL vendor: on a GPU node without --gres, libEGL still
# enumerates the NVIDIA card, fails to open /dev/dri/card* and eglInitialize
# returns 0x3001.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export GALLIUM_DRIVER=llvmpipe

# Bound every thread pool: llvmpipe + swscale + x264 each try to grab all cores.
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$LAB/diffusion-research/experiments/exp_084_luma_matte_viewer"
python run.py
