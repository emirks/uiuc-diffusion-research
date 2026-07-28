#!/bin/bash
#SBATCH --job-name=ep2_detect
#SBATCH --partition=scavenger
#SBATCH --account=campusclusterusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output=outputs/synth_endpoints_logs/detect_v2-%j.out
#SBATCH --error=outputs/synth_endpoints_logs/detect_v2-%j.err

# CPU variant. scavenger had 24 idle nodes while `secondary` sat at (Priority) with ~100 jobs
# ahead of it, and the endpoint-expansion window is 12 h — queue time is the scarce resource
# here, not FLOPs. detect.py auto-selects CPU when no CUDA device is present; the models are
# light (EAST via cv2.dnn, Faster R-CNN MobileNetV3, CLIP ViT-B/32) over 3 frames per
# candidate. Idempotent + --requeue, so preemption just resumes.
set -x
source ~/.bashrc
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
cd $LAB/diffusion-research
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
$LAB/envs/diffusion/bin/python data/processed/ctt_v2_strata/endpoints_v2/_work/detect_v2.py
echo "=== EXIT $? ==="
