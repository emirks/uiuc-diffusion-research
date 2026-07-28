#!/bin/bash
#SBATCH --job-name=ladder2_passA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%j.out
# Amendment-1 Pass A: clip-level dominance diagnostic (ep_align / ref_align / ref_dominated).
source ~/.bashrc || true
set -eo pipefail
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
PY_BIN="$LAB/envs/diffusion/bin/python"
cd "$LAB/diffusion-research"
nvidia-smi --query-gpu=name --format=csv,noheader || true
# PASSA_ARGS lets a side lane (exp_077 d2_gen) reuse this script: e.g.
#   --arm d2_gen --extra-registry .../registry_d2gen.jsonl --gens outputs/videos/d2gen_eval
#   --out eval_ladder/dominance_passA_d2.jsonl
"$PY_BIN" eval_ladder/dominance.py --mode passA ${PASSA_ARGS}
