#!/bin/bash
#SBATCH --job-name=exp083_d3pilot
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-ONLY: the renderer is moderngl on Mesa llvmpipe via EGL, and depth is read
# straight out of exp_082's cached stacks, so no GPU is requested and this never
# competes with training. NOT submitted to `secondary` — another session sweeps it.

# NOTE: source ~/.bashrc BEFORE `set -u` — /etc/bashrc references an unbound
# BASHRCSOURCED and would abort the job.
source ~/.bashrc
set -eo pipefail

module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LAB/envs/diffusion"

# SOFTWARE EGL, forced. On a GPU node with no --gres, libEGL still enumerates the
# NVIDIA card, fails to open /dev/dri/card* (Permission denied) and eglInitialize
# returns 0x3001. Pinning the vendor ICD to Mesa and the platform to surfaceless
# keeps it on llvmpipe, which is what we want anyway — this job asks for no GPU.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
# (EGL_PLATFORM/LIBGL_ALWAYS_SOFTWARE removed: with an explicitly selected EGL
# device Mesa refuses to force software. MeshRenderer walks the device list instead.)
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export GALLIUM_DRIVER=llvmpipe

# Keep every thread pool bounded — llvmpipe + swscale + x264 otherwise each try
# to grab all cores of the node and blow past the per-user thread limit.
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HOME="$LAB/cache/huggingface"

cd "$LAB/diffusion-research/experiments/exp_083_d3_pilot"
python run.py
