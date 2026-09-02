#!/usr/bin/env python
"""Regenerate the dino_shufsig_eye page: the DERANGED-signal eye test for every DINO-signal arm
(A1 channels · A2 tokens · A5 xattn), one TAB per arm. Supersedes gen_dino_a2_shufsig_eye.py.

Per row (zero-shot + unseen same-class cells only) a card shows four clips sharing endpoints, neutral prompt
(family 001), seed 42 and the SAME pixel reference in context:
  own reference | donor clip (its 44-ch DINO signal is fed INSTEAD in the deranged gen) | matched gen | deranged gen
plus a FLIP slot swapping matched<->deranged in place (same playhead). Rows come from each arm's shufsig registry
(`signal_source` = rotation-by-2 derangement, PROTOCOL_LOCKED §28 — identical map for all arms). A clip that has
not landed yet is marked pending; re-run after the gen arrays finish. Relative paths only (viewer rule): media
reach the page through per-arm symlinks in the viewer directory. Run with the ltx2 venv python."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VD = REPO / "outputs/viewers/dino_shufsig_eye"
STD = REPO / "data/processed/transitions_std121"
GEN = REPO / "misc/2026-08-27_dino_signal_training/gen"
CELLS = ["G-zs-same", "G-unseen-same"]
CELL_LABEL = {"G-zs-same": "Zero-shot classes — class, reference and signal never in training",
              "G-unseen-same": "Unseen clips of seen classes (same operator class)"}
SEED = 42
# (tab id, arm slug, store entry, label, one-line mechanism)
ARMS = [
    ("A1", "dino_a1_channels", "027_dino_a1_channels", "A1 · channels",
     "signal → Linear(44→4096), ADDED to the target token embeddings after patchify_proj"),
    ("A2", "dino_a2_tokens", "024_dino_a2_tokens", "A2 · tokens",
     "pooled signal → pseudo-latent bank tokens appended at the target block"),
    ("A5", "dino_a5_xattn", "026_dino_a5_xattn", "A5 · xattn",
     "pooled signal is the QUERY, the own reference block is K/V; fused tokens appended (deranged = donor query, own K/V)"),
]
VD.mkdir(parents=True, exist_ok=True)


def relink(name, target):
    link = VD / name
    rel = os.path.relpath(target, VD)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == rel:
            return
        link.unlink()
    os.symlink(rel, link)


relink("refs", STD)
REF_REL = {p.stem: "%s/%s" % (p.parent.name, p.name) for p in STD.rglob("*.mp4")}


def card(arm, r):
    tab, slug, entry, _, _ = arm
    ep, ref, src, cell = r["endpoint"], r["reference"], r["signal_source"], r["cell"]
    m_file = "%s__%s_neutral__%s__ref_%s__s%d.mp4" % (cell, slug, ep, ref, SEED)
    d_file = "%s__%s_shufsig_neutral__%s__ref_%s__s%d.mp4" % (cell, slug, ep, ref, SEED)
    m_dir, d_dir = "matched_%s" % tab.lower(), "deranged_%s" % tab.lower()
    m_ok = (VD / m_dir / m_file).exists()
    d_ok = (VD / d_dir / d_file).exists()
    if ref not in REF_REL or src not in REF_REL:
        raise SystemExit("reference clip missing in std121: %s / %s" % (ref, src))
    vid = lambda cls, d, f: '<video class="%s" src="%s/%s" loop muted preload="metadata" playsinline></video>' % (cls, d, f)
    m_video = vid("mv", m_dir, m_file) if m_ok else '<div class="pending">matched clip missing<br><small>%s</small></div>' % m_file
    d_video = vid("dv", d_dir, d_file) if d_ok else '<div class="pending">deranged clip pending<br><small>%s</small></div>' % d_file
    if m_ok and d_ok:
        flip = ('<div class="v flip" data-state="m" title="click / press F to flip">'
                '<span class="lab lab-flip">FLIP · showing <b class="fs">matched</b></span>'
                '<video class="fm" src="%s/%s" loop muted preload="metadata" playsinline></video>'
                '<video class="fd" src="%s/%s" loop muted preload="metadata" playsinline hidden></video></div>'
                % (m_dir, m_file, d_dir, d_file))
    else:
        flip = '<div class="v flip"><div class="pending">flip needs both clips</div></div>'
    return ('<figure class="card">\n  <div class="quad">\n'
            '    <div class="v"><span class="lab lab-ref">own reference · in context for BOTH</span>'
            '<video src="refs/%s" loop muted preload="metadata" playsinline></video></div>\n'
            '    <div class="v"><span class="lab lab-donor">donor · its SIGNAL fed to deranged</span>'
            '<video src="refs/%s" loop muted preload="metadata" playsinline></video></div>\n'
            '    <div class="v"><span class="lab lab-m">matched gen · signal = own ref</span>%s</div>\n'
            '    <div class="v"><span class="lab lab-d">deranged gen · signal = donor</span>%s</div>\n'
            '  </div>\n  %s\n'
            '  <figcaption><button class="play">▶ play all</button><b>%s</b>'
            '<span class="k">ref <i>%s</i></span><span class="k donor">signal ▸ <i>%s</i></span>'
            '<span class="cell">%s</span><span class="seed">s%d</span></figcaption>\n</figure>'
            % (REF_REL[ref], REF_REL[src], m_video, d_video, flip, ep, ref, src, cell, SEED)), d_ok


panels, tabs, status_lines = [], [], []
for arm in ARMS:
    tab, slug, entry, label, mech = arm
    relink("matched_%s" % tab.lower(), REPO / "store/gens" / entry / "01_neutral__dai/videos")
    relink("deranged_%s" % tab.lower(), REPO / "store/gens" / entry / "03_neutral_shufsig__dai/videos")
    reg = GEN / ("registry_%s_shufsig_neutral.jsonl" % slug)
    rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["cell"] in CELLS]
    rows.sort(key=lambda r: (CELLS.index(r["cell"]), r["endpoint"]))
    by_cell, n_ready = {}, 0
    for r in rows:
        html, ok = card(arm, r)
        n_ready += int(ok)
        by_cell.setdefault(r["cell"], []).append(html)
    st = "%d / %d deranged clips" % (n_ready, len(rows)) + ("" if n_ready == len(rows) else " — pending, re-run the generator after the gen arrays")
    status_lines.append("%s: %s" % (label, st))
    secs = "".join('<section><h2>%s <span class="tag">%s · %d rows</span></h2><div class="grid">%s</div></section>'
                   % (CELL_LABEL[c], c, len(by_cell.get(c, [])), "\n".join(by_cell.get(c, []))) for c in CELLS)
    panels.append('<div class="panel" data-arm="%s" hidden><div class="mech"><b>%s</b> — %s · <span class="st">%s</span></div>%s</div>'
                  % (tab, label, mech, st, secs))
    tabs.append('<button class="tab" data-arm="%s">%s <span class="cnt">%d/%d</span></button>' % (tab, label, n_ready, len(rows)))

html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DINO arms — deranged-signal eye test</title>
<style>
:root{color-scheme:dark light}
body{margin:0;background:#0e1116;color:#e6edf3;font:14px/1.5 system-ui,sans-serif}
header{padding:14px 22px 10px;border-bottom:1px solid #222b36;position:sticky;top:0;background:#0e1116ee;backdrop-filter:blur(6px);z-index:5}
h1{margin:0 0 4px;font-size:18px} .sub{color:#9fb0c3;font-size:12.5px;max-width:1150px}
.sub b{color:#e6edf3}
.tabs{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}
.tab{background:#161b22;color:#cdd9e5;border:1px solid #2a3542;border-radius:8px;padding:6px 12px;font-size:13px;cursor:pointer}
.tab.on{background:#243447;border-color:#4a6a8f;color:#fff} .cnt{font:11px ui-monospace,monospace;color:#8aa0b6;margin-left:6px}
.mech{padding:10px 22px 0;color:#9fb0c3;font-size:12.5px} .mech b{color:#e6edf3} .st{font:11px ui-monospace,monospace;color:#8aa0b6}
section{padding:14px 22px 8px} h2{font-size:15px;margin:10px 0;color:#cdd9e5}
.tag{font:11px ui-monospace,monospace;color:#8aa0b6;background:#1b232e;padding:2px 7px;border-radius:10px;margin-left:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}
.card{margin:0;background:#161b22;border:1px solid #232d38;border-radius:8px;overflow:hidden}
.quad{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:#232d38}
.v{position:relative;background:#000;min-height:120px}
.lab{position:absolute;top:0;left:0;z-index:1;font:10px ui-monospace,monospace;padding:2px 6px;border-bottom-right-radius:6px;letter-spacing:.02em}
.lab-ref{background:#3a2d1b;color:#e0b76c} .lab-donor{background:#3a1b2d;color:#e08ac0}
.lab-m{background:#183028;color:#6cc4a1} .lab-d{background:#1b2a3a;color:#7fb8e6} .lab-flip{background:#2a2a3a;color:#d6d6ff}
video{width:100%;display:block;background:#000;aspect-ratio:4/3}
.flip{border-top:2px solid #232d38;cursor:pointer} .flip[data-state="d"] .lab-flip{background:#1b2a3a;color:#7fb8e6}
.pending{display:grid;place-items:center;height:100%;min-height:160px;color:#7d8ea0;font:12px ui-monospace,monospace;text-align:center;padding:10px}
figcaption{padding:7px 9px;font-size:12px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
figcaption b{color:#e6edf3} .k{color:#9fb0c3;font:11px ui-monospace,monospace} .k i{color:#e0b76c;font-style:normal}
.k.donor i{color:#e08ac0} .cell{color:#8aa0b6;font:11px ui-monospace,monospace} .seed{color:#7d8ea0;margin-left:auto;font:11px ui-monospace,monospace}
.play{background:#1b232e;color:#cdd9e5;border:1px solid #2a3542;border-radius:6px;padding:2px 8px;font-size:11px;cursor:pointer}
</style></head><body>
<header><h1>DINO-signal arms — matched vs DERANGED signal (eye test)</h1>
<div class="sub">Every row: same endpoints, same neutral prompt, same seed 42, and the <b>same pixel reference in context</b> for both generations.
The only difference: the matched gen receives the 44-ch DINO signal extracted from its <b>own reference</b>; the deranged gen receives the signal
extracted from the <b>donor clip</b> (another operator class; rotation-by-2 over the 36 eval references, frozen in PROTOCOL_LOCKED §28 — the same map for every arm).
If an arm reads the signal, its deranged clip should drift toward the donor operator or at least differ visibly; if the training probe's Δderanged ≈ 0 is real,
matched and deranged will be near-identical frame for frame. <b>FLIP</b>: click the bottom slot (or press F with the mouse over a card) to swap matched↔deranged
at the same playhead. ▶ play all starts a card's four clips together. Cells: G-zs-same (8) + G-unseen-same (13); staged 10k adapters, rank 128, dataset 005.</div>
<div class="tabs">@@TABS@@</div></header>
@@PANELS@@
<script>
function playAll(card){var vs=card.querySelectorAll('video');for(var i=0;i<vs.length;i++){try{vs[i].currentTime=0;vs[i].play();}catch(e){}}}
function flipCard(card){var f=card.querySelector('.flip');if(!f)return;var m=f.querySelector('.fm'),d=f.querySelector('.fd');if(!m||!d)return;
 var toD=(f.getAttribute('data-state')==='m');var from=toD?m:d,to=toD?d:m;try{to.currentTime=from.currentTime;}catch(e){}
 from.hidden=true;to.hidden=false;if(!from.paused){to.play();}from.pause();f.setAttribute('data-state',toD?'d':'m');
 var s=f.querySelector('.fs');if(s)s.textContent=toD?'DERANGED':'matched';}
function showTab(id){var ps=document.querySelectorAll('.panel');for(var i=0;i<ps.length;i++){ps[i].hidden=(ps[i].getAttribute('data-arm')!==id);}
 var ts=document.querySelectorAll('.tab');for(var j=0;j<ts.length;j++){ts[j].className='tab'+(ts[j].getAttribute('data-arm')===id?' on':'');}
 try{localStorage.setItem('dino_shufsig_tab',id);}catch(e){} if(history.replaceState){history.replaceState(null,'','#'+id);}}
document.addEventListener('click',function(e){var t=e.target.closest('.tab');if(t){showTab(t.getAttribute('data-arm'));return;}
 var b=e.target.closest('.play');if(b){playAll(b.closest('.card'));return;}
 var f=e.target.closest('.flip');if(f){flipCard(f.closest('.card'));}});
var hover=null;document.addEventListener('mouseover',function(e){var c=e.target.closest('.card');if(c)hover=c;});
document.addEventListener('keydown',function(e){if((e.key==='f'||e.key==='F')&&hover){flipCard(hover);}});
var init=(location.hash||'').replace('#','');if(!init){try{init=localStorage.getItem('dino_shufsig_tab')||'';}catch(e){}}
if(!document.querySelector('.panel[data-arm="'+init+'"]')){init=document.querySelector('.tab').getAttribute('data-arm');}
showTab(init);
</script>
</body></html>'''.replace("@@TABS@@", "".join(tabs)).replace("@@PANELS@@", "".join(panels))

(VD / "index.html").write_text(html)
(VD / "viewer.json").write_text(json.dumps({
    "title": "DINO arms — deranged-signal eye test",
    "blurb": "Matched vs deranged 44-ch DINO signal for A1 channels / A2 tokens / A5 xattn (tabs); same pixel reference, endpoints, prompt, seed; zs + unseen same-class rows; flip slot. " + " · ".join(status_lines),
    "group": "Experiment runs", "featured": True}, indent=1))
print("wrote %s\n  " % (VD / "index.html") + "\n  ".join(status_lines))
