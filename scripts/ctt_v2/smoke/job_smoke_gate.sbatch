#!/bin/bash
#SBATCH --job-name=ctt2_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --exclude=ccc0424
#SBATCH --output=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/logs/slurm/%x-%A_%a.out
#
# CTT v2 — the MANDATORY mixed-format smoke gate (A9 §3 item 3).
# Two independent array tasks so neither can take the other down:
#   task 0  PROBE   per-sample per-format loss, sigma-matched comparison, realized shifts
#   task 1  TRAIN   the REAL certified `scripts/train.py` over the same mixed root
#
#   sbatch --array=0-1 --partition=<p> --account=<a> scripts/ctt_v2/smoke/job_smoke_gate.sbatch
#
# Short walltime on purpose: A9 budgets 1-2 GPU-h and short jobs backfill.
# `--exclude=ccc0424` — that node has no working EGL. No GL work here, kept as a standing rule.
set -eo pipefail
source ~/.bashrc || true

LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
DR=$LAB/diffusion-research
WT=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch
TR=$LAB/LTX-2-cond-bleed-fix/packages/ltx-trainer
# ABSOLUTE interpreter paths only. The worktree depth makes a relative interpreter path
# resolve wrong and a job died that way; both of these are absolute and stay absolute.
VENV_PY=$LAB/LTX-2-official/.venv/bin/python      # the trainer's venv (rich, ltx_core, peft)
ROOT=$DR/outputs/ctt_v2/smoke/root_mixed
ART=$LAB/misc/ctt_v2_final/artefacts/smoke_gate
TASK=${SLURM_ARRAY_TASK_ID:-0}

mkdir -p "$DR/outputs/logs/slurm" "$ART"
echo "[smoke] host=$(hostname) task=$TASK $(date)"
nvidia-smi -L || true
test -f "$ROOT/SMOKE_ROOT_MANIFEST.json" || { echo "[smoke] no smoke root at $ROOT"; exit 3; }

if [ "$TASK" = "0" ]; then
    echo "[smoke] ===== TASK 0: per-format probe ====="
    cd "$TR"
    PYTHONPATH="$TR/src" "$VENV_PY" \
        "$WT/scripts/ctt_v2/smoke/mixed_format_probe.py" \
        --root "$ROOT" --out "$ART"
    rc=$?
    echo "[smoke] probe exit=$rc"
    exit $rc
fi

if [ "$TASK" = "1" ]; then
    echo "[smoke] ===== TASK 1: the REAL certified trainer over the mixed root ====="
    CFG=$DR/outputs/ctt_v2/smoke/mixed_smoke.yaml
    OUT=$DR/outputs/training/ctt_v2/smoke_mixed
    STEPS=30
    "$VENV_PY" "$WT/scripts/ctt_v2/smoke/make_smoke_config.py" \
        --root "$ROOT" --out "$CFG" --output-dir "$OUT" --steps "$STEPS"
    # _find_checkpoint RAISES on a path that is neither file nor dir; an EMPTY dir => None
    mkdir -p "$OUT/checkpoints"
    cd "$TR"
    set +e
    PYTHONPATH="$TR/src" "$VENV_PY" scripts/train.py "$CFG" --disable-progress-bars \
        2>&1 | tee "$ART/train_mixed.log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "[smoke] train exit=$rc"
    # the seatbelt-7 gate: the trainer's own index line must show N of N
    "$VENV_PY" "$WT/scripts/ctt_v2/smoke/check_train_log.py" \
        --log "$ART/train_mixed.log" --root "$ROOT" --steps "$STEPS" \
        --out "$ART/SMOKE_TRAIN_GATE.json"
    grc=$?
    echo "[smoke] train-log gate exit=$grc"
    [ "$rc" = "0" ] || exit $rc
    exit $grc
fi

echo "[smoke] unknown task $TASK"; exit 4
