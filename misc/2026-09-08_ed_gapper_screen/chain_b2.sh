#!/bin/bash
# stage B2 chain: wait for gen_b2 -> register -> plan -> submit scoring -> wait -> readout -> rebuild hardness viewer.
set -uo pipefail
LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa; DR=$LAB/diffusion-research; CAMP=$DR/misc/2026-09-08_ed_gapper_screen
PY=$LAB/envs-aarch64/ltx2/bin/python; cd $DR
GEN=$($PY -c "import json;print(json.load(open('$CAMP/ledger.json'))['gen_b2'])")
until [ "$(squeue -j $GEN -h -o %T 2>/dev/null | wc -l)" -eq 0 ]; do sleep 60; done
echo "[b2] gen $GEN: $(sacct -j $GEN -X --format=State --noheader | sort | uniq -c | tr '\n' ' ') | clips $(ls store/gens/032_dualforce_dcg_w6/08_effect_edscreen_r12__dai/videos | wc -l)/80 $(date -Is)"
N=$(ls store/gens/032_dualforce_dcg_w6/08_effect_edscreen_r12__dai/videos | wc -l)
if [ "$N" -lt 80 ]; then
  echo "[b2] $N/80 clips — resubmitting the missing tasks once (skip-if-exists)"
  $PY - <<'PYX'
import json,subprocess,sys
from pathlib import Path
CAMP=Path('misc/2026-09-08_ed_gapper_screen'); reg=Path('eval_ladder/registry_dualforce_dcg_w6_effect_edscreen_r12.jsonl')
rows=[json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
vids=Path('store/gens/032_dualforce_dcg_w6/08_effect_edscreen_r12__dai/videos')
todo=[i for i,r in enumerate(rows) if not (vids/f"{r['item_id']}__s42.mp4").exists()]
r=subprocess.run(["sbatch","--parsable","--account=bgjg-dtai-gh",f"--array={','.join(map(str,todo))}","--time=00:40:00","--job-name=eds_dcg2r",f"--output={CAMP.resolve()}/gen/logs/%x-%A_%a.out",
  "--export=ALL,ARM=dualforce_dcg_w6_effect_edscreen,REG="+str(reg)+",OUT=store/gens/032_dualforce_dcg_w6/08_effect_edscreen_r12__dai,NCHUNKS=80,W=6.0,GEN_FRAMES=81,GEN_PREFIX_FRAMES=1,LADDER_SPLIT_FILE=split_screen.json",
  "misc/2026-09-07_eval_grid_v2/gen/job_dcg_v3.sbatch"],capture_output=True,text=True)
led=json.loads((CAMP/'ledger.json').read_text()); led['gen_b2_retry']=r.stdout.strip(); (CAMP/'ledger.json').write_text(json.dumps(led,indent=1)); print('retry',r.stdout.strip(),len(todo),'tasks',r.stderr[-200:])
PYX
  RJ=$($PY -c "import json;print(json.load(open('$CAMP/ledger.json')).get('gen_b2_retry',''))")
  [ -n "$RJ" ] && until [ "$(squeue -j $RJ -h -o %T 2>/dev/null | wc -l)" -eq 0 ]; do sleep 60; done
  echo "[b2] after retry: clips $(ls store/gens/032_dualforce_dcg_w6/08_effect_edscreen_r12__dai/videos | wc -l)/80"
fi
$PY scripts/ed_screen/stage_b2.py register 2>&1 | tail -3
$PY scripts/ed_screen/stage_b2.py plan 2>&1 | tail -3
$PY scripts/ed_screen/stage_b2.py submit 2>&1 | tail -1
SC=$($PY -c "import json;print(json.load(open('$CAMP/ledger.json')).get('score_b2',''))")
[ -z "$SC" ] && { echo "[b2] no score job — stopping"; exit 1; }
until [ "$(squeue -j $SC -h -o %T 2>/dev/null | wc -l)" -eq 0 ]; do sleep 60; done
echo "[b2] score $SC: $(sacct -j $SC -X --format=State --noheader | sort | uniq -c | tr '\n' ' ') $(date -Is)"
$PY scripts/ed_screen/stage_b2.py readout 2>&1 | tail -4
OPENBLAS_NUM_THREADS=1 $PY scripts/grid_v3/hardness_viewer.py 2>&1 | tail -2
OPENBLAS_NUM_THREADS=1 $PY scripts/grid_v3/gapper_select.py 2>&1 | tail -3
echo "[b2] done $(date -Is)"
