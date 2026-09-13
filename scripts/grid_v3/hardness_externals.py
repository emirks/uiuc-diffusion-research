import json, sys, statistics as st, collections
from pathlib import Path
sys.path.insert(0, 'eval_ladder')
import run_eval
REPO = Path('.')
E028 = next(REPO.glob('store/evals/028_grid_v3_paper_arms__dai__*'))
E030 = REPO / 'store/evals/030_external_zs_authornative__dai__2026-09-12'
ceil = dict(run_eval.ceilings())
# registry_v3 (ic_gen rows) -> (cell, endpoint, reference) -> row
reg = {}
for l in open('eval_ladder/registry_v3.jsonl'):
    if not l.strip(): continue
    r = json.loads(l)
    if r['arm'] != 'ic_gen': continue
    reg[(r['cell'], r['endpoint'], r['reference'])] = r
def key_of(iid, ha):
    cell, rest = iid.split(f'__{ha}__', 1)
    ep, ref = rest.split('__ref_', 1)
    return (cell, ep, ref)
def arm_levels(scores_dir, ha):
    """(item_id, seed) -> (level, row)"""
    out = {}
    for (iid, seed), vals in run_eval.pool_means(scores_dir).items():
        r = reg.get(key_of(iid, ha))
        if r is None or not vals or r['gt_pool_class'] not in ceil: continue
        out[(iid, seed)] = (st.mean(vals) / ceil[r['gt_pool_class']] * 100, r)
    return out
fam_of = lambda r: 'ED' if r['gt_pool_class'].startswith('ed.') else 'HF'
paper = {}
for arm in ('base_cond', 'ic_gen', 'dualforce_control', 'dualforce_dcg_w6'):
    for tier in ('neutral', 'effect'):
        for suf in ('v3', 'v3ed81'):
            ha = f'{arm}_{tier}_{suf}'
            paper[(arm, tier, suf)] = arm_levels(E028 / ha, ha)
ext = {}
for ha in ('refvfx_author_native', 'vap_author_native', 'vfxmaster_author_native'):
    ext[ha] = arm_levels(E030 / ha, ha)
    print(ha, 'rows scored', len(ext[ha]), 'fam', collections.Counter(fam_of(r) for _, r in ext[ha].values()),
          'novelty', collections.Counter(r['ref_novelty'] for _, r in ext[ha].values()),
          'content', collections.Counter(r['content'] for _, r in ext[ha].values()), file=sys.stderr)
# hardness A per class = base_cond effect level over all rows, both seeds
A = collections.defaultdict(list)
for suf in ('v3', 'v3ed81'):
    for (iid, seed), (lv, r) in paper[('base_cond', 'effect', suf)].items():
        if r['ref_novelty'] == 'zero_shot': A[r['gt_pool_class']].append(lv)
A = {c: st.mean(v) for c, v in A.items()}
groups = {}
for fam in ('HF', 'ED'):
    cls = [c for c in A if (c.startswith('ed.')) == (fam == 'ED')]
    hard = [c for c in cls if A[c] < 80]; rest = [c for c in cls if A[c] >= 80]
    groups[(fam, 'hard')] = set(hard); groups[(fam, 'rest')] = set(rest); groups[(fam, 'all')] = set(cls)
    print(fam, 'hard', sorted(hard), file=sys.stderr)
def mean_over(levels, classes, content):
    v = [lv for (iid, seed), (lv, r) in levels.items() if r['ref_novelty'] == 'zero_shot' and r['gt_pool_class'] in classes and (content == 'all' or r['content'] == 'same')]
    return (st.mean(v), len(v)) if v else (None, 0)
def fmt(m): return '—' if m is None else f'{m:.1f}'
short = {'base_cond': 'base', 'ic_gen': 'ic_gen', 'dualforce_control': 'control', 'dualforce_dcg_w6': 'DCG w6'}
exts = [('refvfx_author_native', 'refVFX'), ('vap_author_native', 'VAP'), ('vfxmaster_author_native', 'VFXMaster')]
lines = []
for content in ('all', 'same'):
    for tier in ('neutral', 'effect'):
        lines.append(f'\n### {tier} prompts — zero-shot classes, {"all content rows" if content=="all" else "same rows only"} (externals = author-native captions, their only tier; levels, 2 seeds)\n')
        hdr = ['slice', 'classes', 'rows/seed (ours)', 'rows/seed (ext)'] + [f'{short[a]} {tier[:3]}' for a in short] + [f'{n} (author-native)' for _, n in exts]
        lines.append('| ' + ' | '.join(hdr) + ' |'); lines.append('|' + '---|' * len(hdr))
        for fam in ('HF', 'ED'):
            suf = 'v3ed81' if fam == 'ED' else 'v3'
            for g in ('hard', 'rest', 'all'):
                cls = groups[(fam, g)]
                cells = []; n_ours = None; n_ext = None
                for a in short:
                    m, n = mean_over(paper[(a, tier, suf)], cls, content); cells.append(fmt(m)); n_ours = n // 2 if n_ours is None else n_ours
                for ha, _ in exts:
                    m, n = mean_over(ext[ha], cls, content); cells.append(fmt(m)); n_ext = n // 2 if n_ext is None else n_ext
                label = f'{fam} {g}' + (' (base eff < 80)' if g == 'hard' else '')
                lines.append('| ' + ' | '.join([label, str(len(cls)), str(n_ours), str(n_ext)] + cells) + ' |')
out = '\n'.join(lines)
print(out)
Path('misc/2026-09-07_eval_grid_v2/eval/HARDNESS.md').open('a').write('\n\n## Hard vs rest zero-shot classes with the external baselines (evals/030, owner 2026-09-13)\n\nhard = prompt-only effect level (A) below 80 over all zero-shot rows; externals scored at author-native captions only, so the same numbers appear in the neutral and effect tables.\n' + out + '\n')
