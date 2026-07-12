#!/usr/bin/env python3
"""Generate outputs/taxonomy/viewer.html — validation UI for class_axes.yaml.

One card per class: exemplar videos (relative paths into the std corpus),
the 7 annotated fields as editable controls prefilled from class_axes.yaml,
hard-call badges, manifest cross-reference (old tags + sidedness), and an
Export button that downloads corrections.json containing only what was edited.

Usage: python scripts/gen_taxonomy_viewer.py
Open the result next to the repo (relative video paths: ../../data/...).
"""
import json, html, sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml (research env)")

ROOT = Path(__file__).resolve().parents[1]
AXES = ROOT / "outputs/taxonomy/class_axes.yaml"
CORPUS = ROOT / "data/processed/transitions_std121/corpus_manifest.json"
OUT = ROOT / "outputs/taxonomy/viewer.html"

FIELDS = {
    "scene_swap": [True, False],
    "sidedness": ["A_only", "B_only", "two_sided"],
    "mechanism": ["occlusion", "morph", "traversal", "dressed_cut"],
    "camera_defining": [True, False],
    "inserted_content": [True, False],
    "stylization": [True, False],
    "subject_anchored": [True, False],
}

axes = yaml.safe_load(AXES.read_text())["classes"]
corpus = json.loads(CORPUS.read_text())["classes"]

cards = []
for cls in sorted(axes):
    a = axes[cls]
    m = corpus.get(cls, {})
    vids = "".join(
        f'<div class="v"><video src="../../data/processed/transitions_std121/{cls}/{c}.mp4" '
        f'controls muted loop preload="none"></video><span>{html.escape(c)}</span></div>'
        for c in a.get("clips_viewed", []))
    ctrls = []
    for f, opts in FIELDS.items():
        cur = a.get(f)
        os_ = "".join(
            f'<option value="{o}" {"selected" if o == cur else ""}>{o}</option>'
            for o in opts)
        hc = ' class="hard"' if f in (a.get("hard_call") or []) else ""
        ctrls.append(f'<label{hc}>{f}<select data-cls="{cls}" data-f="{f}" '
                     f'data-orig="{cur}">{os_}</select></label>')
    flags = []
    if a.get("sidedness_conflict"): flags.append("SIDEDNESS CONFLICT vs manifest")
    if a.get("heterogeneous"): flags.append("HETEROGENEOUS")
    flagtxt = f'<div class="flag">{" · ".join(flags)}</div>' if flags else ""
    cards.append(f"""
<div class="card" id="{cls}">
 <h2>{cls} <small>n={m.get("n_clips","?")} · manifest: {m.get("sidedness","?")} / {"+".join(m.get("tags",[]))}</small></h2>
 {flagtxt}
 <div class="vids">{vids}</div>
 <div class="ctrls">{"".join(ctrls)}</div>
 <p class="notes">{html.escape(str(a.get("notes","")))}</p>
 <label class="ok"><input type="checkbox" data-cls="{cls}" class="validated"> validated</label>
</div>""")

page = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Transition taxonomy — validate</title>
<style>
body{font-family:system-ui;margin:16px;background:#111;color:#ddd}
.card{border:1px solid #333;border-radius:8px;padding:12px;margin:14px 0;background:#1a1a1a}
h2{margin:0 0 6px}h2 small{color:#888;font-weight:400;font-size:.7em}
.vids{display:flex;gap:8px;flex-wrap:wrap}.v{display:flex;flex-direction:column;font-size:.75em;color:#999}
video{width:240px;border-radius:4px;background:#000}
.ctrls{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
label{display:flex;flex-direction:column;font-size:.72em;color:#aaa}
label.hard select{outline:2px solid #b58900}
select{background:#222;color:#eee;border:1px solid #444;border-radius:4px;padding:2px}
select.changed{outline:2px solid #d33}
.flag{color:#f66;font-weight:600;font-size:.85em;margin:4px 0}
.notes{color:#8a8;font-size:.85em;margin:6px 0}
.ok{flex-direction:row;gap:6px;align-items:center;font-size:.8em}
#bar{position:sticky;top:0;background:#111;padding:8px 0;z-index:9;border-bottom:1px solid #333}
button{background:#2a6;border:0;border-radius:5px;padding:8px 14px;color:#fff;font-weight:600;cursor:pointer}
#count{margin-left:12px;color:#888}
</style></head><body>
<div id="bar"><button onclick="exp()">Export corrections.json</button><span id="count"></span>
<span style="color:#666;margin-left:12px">yellow outline = annotator hard-call · red = you changed it</span></div>
""" + "\n".join(cards) + """
<script>
const sel=document.querySelectorAll('select');
sel.forEach(s=>s.addEventListener('change',()=>{s.classList.toggle('changed',s.value!==s.dataset.orig);n()}));
function n(){const c=[...sel].filter(s=>s.value!==s.dataset.orig).length;
document.getElementById('count').textContent=c+' correction(s), '+
document.querySelectorAll('.validated:checked').length+' validated';}
document.querySelectorAll('.validated').forEach(b=>b.addEventListener('change',n));
function exp(){const out={corrections:{},validated:[]};
sel.forEach(s=>{if(s.value!==s.dataset.orig){(out.corrections[s.dataset.cls]??={})[s.dataset.f]=s.value;}});
document.querySelectorAll('.validated:checked').forEach(b=>out.validated.push(b.dataset.cls));
const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));
a.download='corrections.json';a.click();}
n();
</script></body></html>"""
OUT.write_text(page)
print(f"wrote {OUT} ({len(axes)} classes)")
