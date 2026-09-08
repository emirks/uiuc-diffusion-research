#!/bin/bash
# EffectData gapper screen — after the media job: clauses -> rows -> manifest -> ceilings job + gen job.
set -uo pipefail
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa; DR=$LAB/diffusion-research; CAMP=$DR/misc/2026-09-08_ed_gapper_screen
PY=$LAB/envs-aarch64/ltx2/bin/python
cd $DR
MEDIA=$($PY -c "import json;print(json.load(open('$CAMP/ledger.json'))['media'])")
until [ "$(squeue -j $MEDIA -h -o %T 2>/dev/null | wc -l)" -eq 0 ]; do sleep 60; done
echo "[chain] media job $MEDIA: $(sacct -j $MEDIA -X --format=State,Elapsed --noheader | head -1)"
tail -3 $CAMP/logs/eds_media-$MEDIA.out
source $LAB/envs-aarch64/activate >/dev/null 2>&1; export PATH=$LAB/bin:$PATH OPENBLAS_NUM_THREADS=1
source $LAB/secrets/gemini_transition.env
echo "[chain] clauses"; $PY scripts/ed_screen/build_screen.py clauses --workers 6 2>&1 | grep -v 'FAIL' | tail -3
echo "[chain] rows";    $PY scripts/ed_screen/build_screen.py rows 2>&1 | tail -3
echo "[chain] manifest"; $PY scripts/ed_screen/build_screen.py manifest 2>&1 | tail -2
NROWS=$(wc -l < eval_ladder/registry_base_cond_effect_edscreen.jsonl); echo "[chain] registry rows: $NROWS"
if [ "$NROWS" -lt 800 ]; then echo "[chain] too few rows — stopping before submitting jobs"; exit 1; fi
C=$(sbatch --parsable --account=bhwp-dtai-gh --array=0-5 --export=ALL,NSHARDS=6 $CAMP/warm_ceilings.sbatch); echo "[chain] ceilings job $C"
$PY - <<PYX
import json; p='$CAMP/ledger.json'; d=json.load(open(p)); d['ceilings']='$C'; json.dump(d, open(p,'w'), indent=1)
PYX
$PY scripts/ed_screen/launch.py gen 2>&1 | tail -2
echo "[chain] done $(date -Is)"
