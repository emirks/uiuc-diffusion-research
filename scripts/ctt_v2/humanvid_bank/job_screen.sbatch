#!/bin/bash
#SBATCH --job-name=hvbank_screen
#SBATCH --partition=HCESC-L40S-normal
#SBATCH --account=hcesc-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48g
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/.claude/worktrees/bottleneck-branch/outputs/logs/slurm/%x-%j.out
# Screen all downloaded HumanVid clips through the endpoint-bank QC cascade and build
# the humanvid_bank (std121 contract). Idempotent: detect.py skips refs already done,
# so a requeue resumes. collect -> detect (GPU) -> build --dry (report) -> build.
set -eo pipefail
source ~/.bashrc
set -u
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
SCRIPTS=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch/scripts/ctt_v2/humanvid_bank
WORK=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch/data/processed/humanvid_bank/_work
mkdir -p "$WORK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# EAST weights: copy once from the blessed bank's work dir (read-only source)
cp -n "$LAB/diffusion-research/data/processed/synth_endpoints/_work/frozen_east_text_detection.pb" "$WORK/" 2>/dev/null || true
cd "$SCRIPTS"
$LAB/envs/diffusion/bin/python collect.py
$LAB/envs/diffusion/bin/python detect.py
$LAB/envs/diffusion/bin/python build.py --dry
$LAB/envs/diffusion/bin/python build.py --cap 1500
echo "=== HVBANK COMPLETE exit=$? ==="
