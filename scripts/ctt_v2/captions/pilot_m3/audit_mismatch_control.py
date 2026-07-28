"""CONTROL for the Layer-2 auditor (operator-added, mirrors the M2 control discipline).

Matched condition  = the accepted descriptions with their OWN source video (already
                     measured during the pilot).
Mismatched control = each accepted description paired with a DIFFERENT clip's video
                     of the same role.  A working auditor must say inaccurate=YES
                     on essentially all of these.
"""
import json, sys, os, random
from pathlib import Path
sys.path.insert(0, '.')
from generate_descriptions import audit_one
from caption_common import STRIPS_INDEX
from concurrent.futures import ThreadPoolExecutor

recs = json.loads(Path(sys.argv[1]).read_text())
idx = json.loads(STRIPS_INDEX.read_text())
acc = [v for v in recs.values() if v.get('description')]
rng = random.Random(1234)
# derangement within role
byrole = {}
for v in acc: byrole.setdefault(v['role'], []).append(v)
jobs = []
for role, vs in byrole.items():
    shifted = vs[7:] + vs[:7]          # guarantees clip_id != clip_id
    for v, other in zip(vs, shifted):
        assert v['clip_id'] != other['clip_id']
        jobs.append((v['clip_id'], role, v['description'], idx[other['clip_id']][f'{role}_video']))

def one(j):
    cid, role, desc, vid = j
    r = audit_one(cid, role, desc, vid)
    return (r.get('verdict') or {})

with ThreadPoolExecutor(max_workers=120) as ex:
    out = list(ex.map(one, jobs))
n = len(out)
inacc = sum(1 for v in out if v.get('inaccurate') == 'YES')
leak  = sum(1 for v in out if v.get('leak') == 'YES')
err   = sum(1 for v in out if not v)
print(f'MISMATCH CONTROL n={n}')
print(f'  inaccurate=YES : {inacc}/{n} = {100*inacc/n:.1f}%   (must be near 100% for the instrument to have power)')
print(f'  leak=YES       : {leak}/{n} = {100*leak/n:.1f}%')
print(f'  parse/API errors: {err}')
json.dump({'n':n,'inaccurate_yes':inacc,'leak_yes':leak,'errors':err,
           'inaccurate_pct':round(100*inacc/n,2),'leak_pct':round(100*leak/n,2)},
          open(sys.argv[2],'w'), indent=1)
