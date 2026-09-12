#!/usr/bin/env python
"""Hardness scene viewer (owner 2026-09-12): per class, hardest first, GT reference + endpoint clip beside the
base_cond and DCG generations (neutral + effect tiers, both seeds) for grid v3 Higgsfield, grid v3 EffectData and
the 300-effect EffectData probe (prompt-only; DCG effect on the 40 lowest). Levels = pool-% of m1a per generation.

Writes outputs/viewers/hardness_scene/index.html + media/ symlinks. Serve: viewerctl serve (repo root, port 8017).
"""
from __future__ import annotations
import collections, csv, json, statistics as st, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_ladder")); sys.path.insert(0, str(REPO / "scripts/grid_v3"))
import run_eval, closeout  # noqa: E402

OUT = REPO / "outputs/viewers/hardness_scene"; MEDIA = OUT / "media"
STD = REPO / "data/processed/transitions_std121"
SCREEN = REPO / "misc/2026-09-08_ed_gapper_screen"
E029 = REPO / "store/evals/029_ed_gapper_screen__dai__2026-09-08"
ARMS = [("base_cond", "005_base_cond", (4, 5, 6, 7), "base"), ("ic_gen", "001_ic_gen", (3, 4, 5, 6), "ic_gen"),
        ("dualforce_control", "013_dualforce_control", (3, 4, 5, 6), "ctrl"), ("dualforce_dcg_w6", "032_dualforce_dcg_w6", (3, 4, 5, 6), "DCG")]
TIERS = [("neutral", "hf"), ("neutral", "ed"), ("effect", "hf"), ("effect", "ed")]

def nov_of(cell):
    return "seen" if cell in ("G-fit", "G-memo-probe") else "unseen" if cell.startswith("G-unseen") else "zero_shot"

def link(name, target):
    p = MEDIA / name
    if p.is_symlink() or p.exists():
        p.unlink()
    p.symlink_to(target)

def levels(scores_dir, rows, ceil):
    """(item_id, seed) -> level (pool-% x100) for rows in the given registry."""
    out = {}
    for (iid, seed), vals in run_eval.pool_means(scores_dir).items():
        r = rows.get(iid)
        if r and r["gt_pool_class"] in ceil and vals:
            out[(iid, seed)] = st.mean(vals) / ceil[r["gt_pool_class"]] * 100
    return out

HV = REPO / "data/processed/humanvid_bank/clips"; CONDS = REPO / "eval_ladder/conds"

def gt(clip, cls):
    """Served path of a clip: corpus classes under STD/<class>; foreign endpoints from the humanvid bank or the
    DAVIS conditioning windows (foreign content carries no GT effect — the viewer labels it as content only)."""
    if clip.startswith("humanvid_"):
        return f"media/humanvid/{clip}.mp4" if (HV / f"{clip}.mp4").exists() else None
    if clip.startswith("davis_"):
        c = sorted(CONDS.glob(f"{clip}*.mp4")) if CONDS.exists() else []
        return f"media/conds/{c[0].name}" if c else None
    return f"media/gt/{cls}/{clip}.mp4" if (STD / cls / f"{clip}.mp4").exists() else None

def main():
    MEDIA.mkdir(parents=True, exist_ok=True); link("gt", STD); link("humanvid", HV)
    if CONDS.exists(): link("conds", CONDS)
    ceil = dict(run_eval.ceilings())
    cs = json.load(open(SCREEN / "ceilings_screen.json"))["ceilings"]
    for k, v in cs.items():
        ceil.setdefault(k, v["ceiling"])
    E = closeout.eval_entry()
    sections = []; missing = collections.Counter()

    def new_row(r, c):
        return {"cell": r["cell"], "content": r["content"], "nov": nov_of(r["cell"]), "endpoint": r["endpoint"], "endpoint_class": r["endpoint_class"],
                "reference": r["reference"], "sided": r.get("sided"), "src": r.get("endpoint_source"),
                "gt_ref": gt(r["reference"], r.get("donor_class") or c), "gt_end": gt(r["endpoint"], r["endpoint_class"]), "prompts": {}, "gens": {}}

    def ingest(ha, sub, scores_dir, tier, seeds, classes):
        """gens[ha][seed] = level (None = scored row missing); the clip path is rebuilt client-side from
        media/<ha>/<cell>__<ha>__<endpoint>__ref_<reference>__s<seed>.mp4 — the store's item_id contract."""
        rows = {r["item_id"]: r for r in map(json.loads, filter(str.strip, open(REPO / f"eval_ladder/registry_{ha}.jsonl"))) if r["arm"] == ha}
        lv = levels(scores_dir, rows, ceil)
        for iid, r in rows.items():
            c = r["gt_pool_class"]; key = (r["cell"], r["endpoint"], r["reference"])
            assert iid == f"{r['cell']}__{ha}__{r['endpoint']}__ref_{r['reference']}", iid
            row = classes[c].setdefault(key, new_row(r, c)); row["prompts"].setdefault(tier, r["prompt"])
            for seed in seeds:
                if (sub / "videos" / f"{iid}__s{seed}.mp4").exists():
                    row["gens"].setdefault(ha, {})[str(seed)] = lv.get((iid, seed))
                else:
                    missing[ha] += 1

    def class_mean(rl, ha, only_same=False):
        v = [x for r in rl if (not only_same or r["content"] == "same") for x in r["gens"].get(ha, {}).values() if x is not None]
        return st.mean(v) if v else None

    # ---- grid v3 families
    for fam, label, frames in (("hf", "grid v3 · Higgsfield (121 f, seeds 42/43)", 121), ("ed", "grid v3 · EffectData (81 f, seeds 42/43)", 81)):
        suffix = "v3" + ("ed81" if fam == "ed" else ""); classes = collections.defaultdict(dict); cols = []
        for arm, gdir, kks, short in ARMS:
            for i, (tier, f) in enumerate(TIERS):
                if f != fam: continue
                ha = f"{arm}_{tier}_{suffix}"; sub = REPO / f"store/gens/{gdir}/{kks[i]:02d}_{tier}_{suffix}__dai"
                link(ha, sub / "videos"); cols.append({"ha": ha, "arm": short, "tier": tier})
                ingest(ha, sub, E / ha, tier, (42, 43), classes)
        cards = []
        for c, rws in classes.items():
            rl = list(rws.values()); m = lambda ha, s=False: class_mean(rl, ha, s)
            be, bn, dn, de = m(f"base_cond_effect_{suffix}"), m(f"base_cond_neutral_{suffix}"), m(f"dualforce_dcg_w6_neutral_{suffix}"), m(f"dualforce_dcg_w6_effect_{suffix}")
            cards.append({"cls": c, "nov": "/".join(sorted({r["nov"] for r in rl})), "ceiling": ceil.get(c), "n_all": len(rl), "n_same": sum(r["content"] == "same" for r in rl),
                          "A": be, "A_same": m(f"base_cond_effect_{suffix}", True), "B": (dn - be) if None not in (dn, be) else None, "C": (be - bn) if None not in (be, bn) else None,
                          "base_neu": bn, "dcg_neu": dn, "dcg_eff": de,
                          "rows": sorted(rl, key=lambda r: ({"same": 0, "cross": 1, "foreign": 2}[r["content"]], r["endpoint"]))})
        sections.append({"id": f"v3_{fam}", "label": label, "frames": frames, "cols": cols, "seeds": [42, 43], "kind": "v3", "cards": cards})
    # ---- EffectData probe (screen stage A + B)
    cols = []; classes = collections.defaultdict(dict)
    for ha, sub in (("base_cond_effect_edscreen", REPO / "store/gens/005_base_cond/08_effect_edscreen__dai"), ("dualforce_dcg_w6_effect_edscreen", REPO / "store/gens/032_dualforce_dcg_w6/07_effect_edscreen__dai")):
        link(ha, sub / "videos"); cols.append({"ha": ha, "arm": "base" if ha.startswith("base") else "DCG", "tier": "effect"})
        ingest(ha, sub, E029 / ha, "effect", (42,), classes)
    screen = {f"ed.{r['effect']}": r for r in csv.DictReader(open(SCREEN / "eval/screen_levels.csv"))}
    cards = []
    for c, rws in classes.items():
        rl = list(rws.values()); s = screen.get(c, {}); bl = class_mean(rl, "base_cond_effect_edscreen"); dl = class_mean(rl, "dualforce_dcg_w6_effect_edscreen")
        A = float(s["prompt_only_level"]) if s.get("prompt_only_level") else bl
        cards.append({"cls": c, "nov": "zero_shot", "ceiling": ceil.get(c), "n_all": len(rl), "n_same": len(rl), "A": A, "A_same": A, "B": None, "C": None, "base_neu": None, "dcg_neu": None,
                      "dcg_eff": dl, "role": s.get("role"), "category": s.get("category"),
                      "rows": sorted(rl, key=lambda r: (0 if "dualforce_dcg_w6_effect_edscreen" in r["gens"] else 1, r["endpoint"]))})
    sections.append({"id": "probe", "label": "EffectData probe · 300 effects (81 f, seed 42; DCG effect on the 40 lowest)", "frames": 81, "cols": cols, "seeds": [42], "kind": "probe", "cards": cards})
    import datetime
    index = {"built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "sections": [{k: v for k, v in s.items() if k != "cards"} | {"n": len(s["cards"])} for s in sections]}
    sizes = []
    for s in sections:
        f = OUT / f"data_{s['id']}.json"; f.write_text(json.dumps(s["cards"], separators=(",", ":"))); sizes.append(f"{s['id']}={len(s['cards'])} classes/{f.stat().st_size // 1024} KB")
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    (OUT / "index.html").write_text(TEMPLATE)
    for stale in ("data.json",):
        (OUT / stale).unlink(missing_ok=True)
    print(f"wrote {OUT.relative_to(REPO)}/index.html · " + ", ".join(sizes) + f" · missing clips: {dict(missing) or 0}")
    for s in sections:
        no_gt = sum(1 for c in s["cards"] for r in c["rows"] if not r["gt_end"] or not r["gt_ref"])
        if no_gt: print(f"  {s['id']}: {no_gt} rows without a GT clip on disk (foreign DAVIS windows or removed clips)")

TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Hardness scene · base vs DCG</title>
<style>
:root{--bg:#111;--fg:#ddd;--mut:#888;--acc:#f6c945;--hard:#e5533d;--ok:#4caf7d;--card:#1a1a1a;--line:#2a2a2a}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.35 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:#0d0d0d;border-bottom:1px solid var(--line);padding:8px 14px;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
header h1{font-size:15px;margin:0 8px 0 0;color:var(--acc)}
.tabs button{background:#222;color:var(--fg);border:1px solid #333;padding:4px 10px;cursor:pointer;border-radius:4px;margin-right:4px}
.tabs button.on{background:var(--acc);color:#000;border-color:var(--acc)}
label.ck{margin-right:8px;cursor:pointer;white-space:nowrap}label.ck input{vertical-align:middle}
select,input[type=text]{background:#222;color:var(--fg);border:1px solid #333;padding:3px 6px;border-radius:4px}
#stats{color:var(--mut);font-size:12px;margin-left:auto}
#err{display:none;background:#5a1d17;color:#fff;padding:6px 14px;font-family:monospace;white-space:pre-wrap}
#status{color:var(--acc);padding:10px 14px}
main{padding:10px 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;margin:0 0 14px;padding:8px 10px}
.card h2{font-size:14px;margin:0 0 4px;display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline}
.card h2 .cls{color:#fff}.card h2 .nov{color:var(--mut);font-weight:normal}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:600}
.tag.hard{background:var(--hard);color:#fff}.tag.mid{background:#a77b1a;color:#fff}.tag.easy{background:#2f5d3f;color:#dfe}
.kv{color:var(--mut);font-weight:normal;font-size:12px}.kv b{color:var(--fg);font-weight:600}
.prompt{color:#aaa;font-size:12px;margin:2px 0 6px;white-space:pre-wrap}
.row{display:flex;gap:6px;align-items:flex-start;margin:6px 0;padding-top:6px;border-top:1px dashed #262626;overflow-x:auto}
.row .meta{min-width:120px;max-width:120px;color:var(--mut);font-size:11px;word-break:break-all}
.row .meta b{color:var(--fg)}
.cell{min-width:172px;max-width:172px}
.cell .cap{font-size:11px;color:var(--mut);display:flex;justify-content:space-between;margin-top:2px}
.cell .cap b{color:var(--fg)}.cell .cap .lv{font-weight:700}
.cell .cap .lv.lo{color:var(--hard)}.cell .cap .lv.hi{color:var(--ok)}
video{width:172px;background:#000;border-radius:3px;display:block;aspect-ratio:3/4}
.cell.gt video{outline:1px solid #3a3a3a}
.none{width:172px;aspect-ratio:3/4;background:#161616;border:1px dashed #333;border-radius:3px;color:#555;display:flex;align-items:center;justify-content:center;font-size:11px}
#more{margin:10px 0 30px;text-align:center;color:var(--mut)}
.legend{color:var(--mut);font-size:12px;margin:0 0 10px}
</style></head><body>
<header><h1>Hardness scene</h1>
<div class="tabs" id="tabs"></div>
<span>sort <select id="sort"><option value="A">A · prompt-only effect (hardest first)</option><option value="B">B · DCG neutral − base effect (largest first)</option><option value="C">C · effect − neutral (smallest first)</option><option value="dcg_eff">DCG effect (lowest first)</option><option value="cls">class name</option></select></span>
<span>rows <select id="content"><option value="same">same only</option><option value="all">same + cross + foreign</option></select></span>
<span>seeds <select id="seeds"><option value="both">42 + 43</option><option value="42">42</option><option value="43">43</option></select></span>
<span id="cols"></span>
<input type="text" id="q" placeholder="filter class…" size="16">
<label class="ck"><input type="checkbox" id="hardonly"> A &lt; 80 only</label>
<span id="stats"></span></header>
<div id="err"></div><div id="status">loading index.json …</div>
<main><div class="legend" id="legend"></div><div id="cards"></div><div id="more"></div></main>
<script>
'use strict';
function showErr(m){const e=document.getElementById('err');e.style.display='block';e.textContent+=m+'\n'}
window.onerror=function(m,src,l,c){showErr('JS error: '+m+' @'+l+':'+c);return false};
let D=null;const CARDS={};
const S={tab:null,sort:'A',content:'same',seeds:'both',cols:null,q:'',hard:false,shown:0,gtref:true,gtend:true};
const LS='hardness_scene_v2';try{Object.assign(S,JSON.parse(localStorage.getItem(LS)||'{}'));S.shown=0}catch(e){}
function save(){try{localStorage.setItem(LS,JSON.stringify({tab:S.tab,sort:S.sort,content:S.content,seeds:S.seeds,cols:S.cols,hard:S.hard,gtref:S.gtref,gtend:S.gtend}))}catch(e){}}
function sec(){return D.sections.find(function(s){return s.id===S.tab})||D.sections[0]}
const DEF=['base|effect','DCG|effect','DCG|neutral'];
function colOn(c){const k=c.arm+'|'+c.tier;return S.cols?S.cols.indexOf(k)>=0:DEF.indexOf(k)>=0}
function fmt(v){return v==null?'—':v.toFixed(1)}
function sgn(v){return v==null?'—':(v>=0?'+':'')+v.toFixed(1)}
function esc(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;')}
function tagA(a){if(a==null)return '';const k=a<80?'hard':a<95?'mid':'easy';return '<span class="tag '+k+'">'+k+' · A '+a.toFixed(1)+'</span>'}
function drawTabs(){document.getElementById('tabs').innerHTML=D.sections.map(function(s){return '<button class="'+(s.id===S.tab?'on':'')+'" data-t="'+s.id+'">'+esc(s.label)+' · '+s.n+'</button>'}).join('');
 document.querySelectorAll('#tabs button').forEach(function(b){b.onclick=function(){S.tab=b.dataset.t;save();load()}})}
function drawCols(){const s=sec();let h='cols: <label class="ck"><input type="checkbox" data-c="gt_ref" '+(S.gtref?'checked':'')+'> reference GT</label><label class="ck"><input type="checkbox" data-c="gt_end" '+(S.gtend?'checked':'')+'> endpoint GT</label>';
 h+=s.cols.map(function(c){return '<label class="ck"><input type="checkbox" data-c="'+c.arm+'|'+c.tier+'" '+(colOn(c)?'checked':'')+'> '+c.arm+' '+c.tier+'</label>'}).join('');
 document.getElementById('cols').innerHTML=h;
 document.querySelectorAll('#cols input').forEach(function(i){i.onchange=function(){const k=i.dataset.c;if(k==='gt_ref'){S.gtref=i.checked}else if(k==='gt_end'){S.gtend=i.checked}else{S.cols=S.cols||DEF.slice();S.cols=i.checked?S.cols.concat(S.cols.indexOf(k)<0?[k]:[]):S.cols.filter(function(x){return x!==k})}save();draw(true)}})}
function sortCards(cards){const k=S.sort;function v(c){return c[k]}function nul(c){return v(c)==null?1:0}
 return cards.slice().sort(function(a,b){if(k==='cls')return a.cls.localeCompare(b.cls);const d=nul(a)-nul(b);if(d)return d;if(nul(a))return 0;return k==='B'?v(b)-v(a):v(a)-v(b)})}
function visible(){let c=CARDS[S.tab]||[];if(S.hard)c=c.filter(function(x){return x.A!=null&&x.A<80});if(S.q){const q=S.q.toLowerCase();c=c.filter(function(x){return x.cls.toLowerCase().indexOf(q)>=0})}return sortCards(c)}
function cellHTML(src,cap,lv,gtc){if(!src)return '<div class="cell"><div class="none">missing</div><div class="cap"><b>'+esc(cap)+'</b></div></div>';
 const lvc=lv==null?'':lv<60?'lo':lv>=90?'hi':'';
 return '<div class="cell '+(gtc||'')+'"><video muted loop playsinline preload="none" data-src="'+src+'"></video><div class="cap"><b>'+esc(cap)+'</b>'+(lv==null?'':'<span class="lv '+lvc+'">'+lv.toFixed(1)+'</span>')+'</div></div>'}
function genSrc(ha,r,sd){return 'media/'+ha+'/'+r.cell+'__'+ha+'__'+r.endpoint+'__ref_'+r.reference+'__s'+sd+'.mp4'}
function rowHTML(s,r){const seeds=S.seeds==='both'?s.seeds:[+S.seeds];
 let h='<div class="row"><div class="meta"><b>'+r.content+'</b> · '+r.nov+'<br>'+r.cell+'<br>end: '+esc(r.endpoint)+'<br>ref: '+esc(r.reference)+(r.sided?'<br>'+r.sided:'')+'</div>';
 if(S.gtref)h+=cellHTML(r.gt_ref,'reference (GT)',null,'gt');
 if(S.gtend)h+=cellHTML(r.gt_end,r.content==='foreign'?'endpoint content (foreign, no GT effect)':r.content==='cross'?'endpoint clip (other class)':'endpoint clip (GT, same effect)',null,'gt');
 for(const c of s.cols){if(!colOn(c))continue;const g=r.gens[c.ha]||{};for(const sd of seeds){if(!(String(sd) in g)){if(s.seeds.length===1)continue;h+=cellHTML(null,c.arm+' '+c.tier+' s'+sd,null);continue}h+=cellHTML(genSrc(c.ha,r,sd),c.arm+' '+c.tier+' s'+sd,g[String(sd)])}}
 return h+'</div>'}
function cardHTML(s,c){const rows=c.rows.filter(function(r){return S.content==='all'||r.content==='same'});const p=(c.rows[0]&&c.rows[0].prompts.effect)||'';
 const kv=s.kind==='v3'?'<span class="kv">A all rows <b>'+fmt(c.A)+'</b> · same <b>'+fmt(c.A_same)+'</b></span><span class="kv">B DCG neu − base eff <b>'+sgn(c.B)+'</b></span><span class="kv">C eff − neu <b>'+sgn(c.C)+'</b></span><span class="kv">DCG neu <b>'+fmt(c.dcg_neu)+'</b> · DCG eff <b>'+fmt(c.dcg_eff)+'</b></span>'
  :'<span class="kv">screen A (3 rows) <b>'+fmt(c.A)+'</b></span><span class="kv">DCG eff (stage B row) <b>'+fmt(c.dcg_eff)+'</b></span><span class="kv">'+esc(c.role||'')+(c.category?' · '+esc(c.category):'')+'</span>';
 return '<div class="card"><h2><span class="cls">'+esc(c.cls)+'</span>'+tagA(c.A)+'<span class="nov">'+c.nov+'</span>'+kv+'<span class="kv">ceiling <b>'+(c.ceiling==null?'—':c.ceiling.toFixed(2))+'</b> · rows '+c.n_same+' same / '+c.n_all+' all</span></h2><div class="prompt">'+esc(p)+'</div>'+(rows.map(function(r){return rowHTML(s,r)}).join('')||'<div class="kv">no rows under this filter</div>')+'</div>'}
const io=new IntersectionObserver(function(es){for(const e of es){const v=e.target;if(e.isIntersecting){if(!v.getAttribute('src'))v.src=v.dataset.src;const p=v.play();if(p&&p.catch)p.catch(function(){})}else{v.pause()}}},{rootMargin:'300px'});
const more=new IntersectionObserver(function(es){if(es.some(function(e){return e.isIntersecting}))append()});
function append(){const s=sec();const vis=visible();const box=document.getElementById('cards');const next=vis.slice(S.shown,S.shown+8);
 for(const c of next){const d=document.createElement('div');d.innerHTML=cardHTML(s,c);const el=d.firstElementChild;box.appendChild(el);el.querySelectorAll('video').forEach(function(v){io.observe(v)})}
 S.shown+=next.length;document.getElementById('more').textContent=S.shown<vis.length?S.shown+' / '+vis.length+' classes shown — scroll for more':vis.length+' classes';
 document.getElementById('stats').textContent=vis.length+' classes · '+s.frames+' f · A<80: '+vis.filter(function(c){return c.A!=null&&c.A<80}).length}
function draw(keep){const box=document.getElementById('cards');box.innerHTML='';S.shown=0;drawTabs();if(!keep)drawCols();
 document.getElementById('legend').innerHTML='<b>A</b> = base_cond effect level (prompt-only; lower = harder, red &lt; 80). <b>B</b> = DCG w6 neutral − base_cond effect (reference alone vs best text). <b>C</b> = base effect − base neutral. Levels under each clip = that generation\'s pool-% of m1a (mean app_ref vs ≤8 same-class GT clips ÷ class ceiling ×100); the yardstick saturates past 100 on low-ceiling classes. Videos autoplay muted when in view. Built '+D.built+'.';
 append();more.disconnect();more.observe(document.getElementById('more'))}
function load(){const st=document.getElementById('status');if(CARDS[S.tab]){st.style.display='none';draw();return}
 st.style.display='block';st.textContent='loading data_'+S.tab+'.json …';document.getElementById('cards').innerHTML='';drawTabs();
 fetch('data_'+S.tab+'.json',{cache:'no-store'}).then(function(r){if(!r.ok)throw new Error('HTTP '+r.status);return r.json()}).then(function(cards){CARDS[S.tab]=cards;st.style.display='none';draw()}).catch(function(e){st.textContent='';showErr('failed to load data_'+S.tab+'.json: '+e)})}
document.getElementById('sort').value=S.sort;document.getElementById('content').value=S.content;document.getElementById('seeds').value=S.seeds;document.getElementById('hardonly').checked=S.hard;
document.getElementById('sort').onchange=function(e){S.sort=e.target.value;save();draw(true)};document.getElementById('content').onchange=function(e){S.content=e.target.value;save();draw(true)};
document.getElementById('seeds').onchange=function(e){S.seeds=e.target.value;save();draw(true)};document.getElementById('hardonly').onchange=function(e){S.hard=e.target.checked;save();draw(true)};
document.getElementById('q').oninput=function(e){S.q=e.target.value;draw(true)};
fetch('index.json',{cache:'no-store'}).then(function(r){if(!r.ok)throw new Error('HTTP '+r.status);return r.json()}).then(function(d){D=d;if(!D.sections.some(function(s){return s.id===S.tab}))S.tab=D.sections[0].id;load()}).catch(function(e){document.getElementById('status').textContent='';showErr('failed to load index.json: '+e)});
</script></body></html>"""

if __name__ == "__main__":
    main()
