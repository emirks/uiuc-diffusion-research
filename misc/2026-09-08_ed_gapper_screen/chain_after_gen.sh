#!/bin/bash
# EffectData gapper screen — after gen + ceilings: register -> aggregate ceilings -> plan -> submit scoring -> wait -> summary.
set -uo pipefail
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa; DR=$LAB/diffusion-research; CAMP=$DR/misc/2026-09-08_ed_gapper_screen
PY=$LAB/envs-aarch64/ltx2/bin/python; cd $DR
GEN=$($PY -c "import json;print(json.load(open('$CAMP/ledger.json'))['gen'])"); CEIL=$($PY -c "import json;print(json.load(open('$CAMP/ledger.json'))['ceilings'])")
until [ "$(squeue -j $GEN,$CEIL -h -o %T 2>/dev/null | wc -l)" -eq 0 ]; do sleep 120; done
echo "[chain2] gen $GEN: $(sacct -j $GEN -X --format=State --noheader | sort | uniq -c | tr '\n' ' ') | ceilings $CEIL: $(sacct -j $CEIL -X --format=State --noheader | sort | uniq -c | tr '\n' ' ')"
echo "[chain2] clips: $(ls store/gens/005_base_cond/08_effect_edscreen__dai/videos | wc -l)/900"
source $LAB/envs-aarch64/activate >/dev/null 2>&1
module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1; PYTHONPATH=$DR/.claude/worktrees/eval-v4-cert/src $LAB/envs-aarch64/refvfx/bin/python scripts/ed_screen/warm_ceilings.py --aggregate 2>&1 | tail -2
$PY scripts/ed_screen/launch.py register 2>&1 | tail -3
$PY scripts/ed_screen/launch.py plan 2>&1 | tail -2
$PY scripts/ed_screen/launch.py submit 2>&1 | tail -1
SC=$($PY -c "import json;print(json.load(open('$CAMP/ledger.json')).get('score',''))")
[ -z "$SC" ] && { echo "[chain2] no score job — stopping"; exit 1; }
until [ "$(squeue -j $SC -h -o %T 2>/dev/null | wc -l)" -eq 0 ]; do sleep 120; done
echo "[chain2] score $SC: $(sacct -j $SC -X --format=State --noheader | sort | uniq -c | tr '\n' ' ')"
$PY scripts/ed_screen/launch.py status 2>&1 | tail -1
$PY scripts/ed_screen/launch.py summary 2>&1 | tail -20
echo "[chain2] done $(date -Is)"
