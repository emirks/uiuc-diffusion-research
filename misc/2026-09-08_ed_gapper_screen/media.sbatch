#!/bin/bash
# EffectData gapper screen — standardise the screen clips (CPU work; 1 GPU requested only to fit ghx4's sharing rules).
#SBATCH --job-name=eds_media
#SBATCH --partition=ghx4
#SBATCH --constraint=taiga
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110g
#SBATCH --time=02:30:00
#SBATCH --output=/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-09-08_ed_gapper_screen/logs/%x-%A.out
set -uo pipefail
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa; DR=$LAB/diffusion-research
source $LAB/envs-aarch64/activate; export PATH=$LAB/bin:$PATH OPENBLAS_NUM_THREADS=1
cd $DR; python scripts/ed_screen/build_screen.py media --workers ${WORKERS:-24}; rc=$?; echo "[eds_media] done rc=$rc $(date -Is)"; exit $rc
