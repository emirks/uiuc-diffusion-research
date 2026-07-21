#!/bin/bash
# Submit exp_073 generation jobs. Idempotent (gen scripts skip existing outputs).
# Usage: submit_gen.sh specialists | ic3
set -eo pipefail
DR=/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
EX=$DR/experiments/exp_073_cond_bleed_fix
H100="--partition=HCESC-H100-secondary --account=hcesc-h100 --gres=gpu:H100:1"
H200="--partition=HCESC-H200-secondary --account=hcesc-h200 --gres=gpu:H200:1"

spec() {  # class seed arm
  sbatch --parsable --job-name=exp073_gen_${1}_${3}_s${2} $H100 --time=01:59:00 \
    --export=ALL,MODE=specialist,CLS=$1,SEED=$2,ARM=$3 $EX/job_gen.sbatch
}
ic3() {  # arm seed tag manifest extra_export
  local arm=$1 seed=$2 tag=$3 manifest=$4 extra=$5
  local AD=outputs/training/exp_073_cond_bleed_fix/ic3_${arm}/checkpoints/lora_weights_step_05000.safetensors
  local OR=outputs/videos/exp_073_cond_bleed_fix/ic3/${arm}
  sbatch --parsable --job-name=exp073_gen_ic3_${arm}_${tag}_s${seed} $H200 --time=03:59:00 \
    --export=ALL,MODE=ic3,MANIFEST=$manifest,SEED=$seed,ADAPTER=$AD,OUT_ROOT=$OR,$extra $EX/job_gen.sbatch
}

case "$1" in
  specialists)
    for arm in fix nullA nullB; do for s in 42 43 44; do echo "$(spec shadow_smoke $s $arm) ss_${arm}_s${s}"; done; done
    for arm in fix nullA;       do for s in 42 43 44; do echo "$(spec hero_flight  $s $arm) hf_${arm}_s${s}"; done; done
    ;;
  ic3)
    IC3IDS=$(python3 -c "import json;print(','.join(json.load(open('$EX/control_draw.json'))['ic3_onesided_ids']))")
    ICXIDS=$(python3 -c "import json;print(','.join(json.load(open('$EX/control_draw.json'))['ic3x_ids']))")
    for arm in fix nullA nullB; do for s in 42 43 44; do
      echo "$(ic3 $arm $s ts   dataset/manifest_ic3.json   TWO_SIDED_ONLY=1) ic3_${arm}_ts_s${s}"
      echo "$(ic3 $arm $s ctl  dataset/manifest_ic3.json   ONLY_IDS=$IC3IDS)  ic3_${arm}_ctl_s${s}"
      echo "$(ic3 $arm $s ctlx dataset/manifest_ic3_x.json ONLY_IDS=$ICXIDS)  ic3_${arm}_ctlx_s${s}"
    done; done
    ;;
  *) echo "usage: submit_gen.sh specialists|ic3"; exit 2;;
esac
