#!/bin/bash
#SBATCH --job-name=exp073_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/experiments/exp_073_cond_bleed_fix/logs/slurm/%x-%j.out
# Regenerate eval arms with exp_073 adapters. Submit examples:
#  specialist: sbatch --job-name=exp073_gen_ss_fix_s42 --partition=HCESC-H100-secondary \
#    --account=hcesc-h100 --gres=gpu:H100:1 --time=01:59:00 \
#    --export=ALL,MODE=specialist,CLS=shadow_smoke,SEED=42,ARM=fix job_gen.sbatch
#  ic3:        sbatch --job-name=exp073_gen_ic3_fix_s42_c0 --partition=HCESC-H200-secondary \
#    --account=hcesc-h200 --gres=gpu:H200:1 --time=03:59:00 \
#    --export=ALL,MODE=ic3,MANIFEST=dataset/manifest_ic3.json,SEED=42,ARM=fix,\
#ADAPTER=outputs/training/exp_073_cond_bleed_fix/ic3_fix/checkpoints/lora_weights_step_05000.safetensors,\
#OUT_ROOT=outputs/videos/exp_073_cond_bleed_fix/ic3/fix,CHUNK=0,NUM_CHUNKS=1,TWO_SIDED_ONLY=1 job_gen.sbatch
set -eo pipefail
source ~/.bashrc || true
WT=/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-cond-bleed-fix
VENV=/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv
TR="$WT/packages/ltx-trainer"
DR=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
EX="$DR/experiments/exp_073_cond_bleed_fix"
cd "$EX"
echo "[gen] host=$(hostname) MODE=$MODE $(date)"; nvidia-smi -L || true
if [ "$MODE" = "specialist" ]; then
    PYTHONPATH="$TR/src" "$VENV/bin/python" run_c2v_specialists.py \
        --class "$CLS" --seed "$SEED" --arm "$ARM"
elif [ "$MODE" = "ic3" ]; then
    EXTRA=""; [ "${TWO_SIDED_ONLY:-0}" = "1" ] && EXTRA="--two-sided-only"
    PYTHONPATH="$TR/src" "$VENV/bin/python" run_grid_ic3.py \
        --manifest "$MANIFEST" --seed "$SEED" --adapter "$ADAPTER" --out-root "$OUT_ROOT" \
        --chunk "${CHUNK:-0}" --num-chunks "${NUM_CHUNKS:-1}" $EXTRA
else
    echo "set MODE=specialist|ic3"; exit 2
fi
echo "[gen] done $(date)"
