#!/usr/bin/env python3
"""Generate outputs/taxonomy/viewer.html — validation UI for the v2 taxonomy.

v2 (protocol: docs/taxonomy/PROTOCOL_v2_PROPOSAL.md, gate-passed 2026-07-16).
One card per class: exemplar videos, the v2 fields as editable controls prefilled
from class_axes_v2.yaml, RULING/CONFLICT banners, and an Export button that
downloads corrections.json containing only what was edited + validated ticks.
Cards are ordered: owner-ruling classes first, then sidedness conflicts, then rest.

Usage: python scripts/build_class_axes_v2.py && python scripts/gen_taxonomy_viewer.py
Serve from repo root so relative video paths resolve.
"""
import html, sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml (research env)")

ROOT = Path(__file__).resolve().parents[1]
AXES = ROOT / "outputs/taxonomy/class_axes_v2.yaml"
OUT = ROOT / "outputs/taxonomy/viewer.html"

FIELDS = {
    "mechanism": ["cover", "transform", "overlay", "traverse", "cut"],
    "overlay_direction": ["none", "add", "remove", "state"],
    "scene_swap": [True, False],
    "sidedness": ["A_only", "B_only", "two_sided"],
    "camera_defining": [True, False],
    "stylization": [True, False],
    "middle_only": [True, False],
    "subject_anchored": [True, False],
}

axes = yaml.safe_load(AXES.read_text())["classes"]

def card_order(item):
    cls, a = item
    return (0 if a.get("owner_ruling") else (1 if a.get("sidedness_conflict") else 2), cls)

cards = []
for cls, a in sorted(axes.items(), key=card_order):
    vids = "".join(
        f'<div class="v"><video src="../../data/processed/transitions_std121/{cls}/{c}.mp4" '
        f'controls muted loop preload="none"></video><span>{html.escape(c)}</span></div>'
        for c in a.get("clips_viewed", []))
    ctrls = []
    for f, opts in FIELDS.items():
        cur = a.get(f)
        if f == "overlay_direction" and cur is None:
            cur = "none"
        os_ = "".join(
            f'<option value="{o}" {"selected" if str(o) == str(cur) else ""}>{o}</option>'
            for o in opts)
        crit = ' class="crit"' if f == "sidedness" else ""
        ctrls.append(f'<label{crit}>{f}<select data-cls="{cls}" data-f="{f}" '
                     f'data-orig="{cur}">{os_}</select></label>')
    flags = []
    if a.get("owner_ruling"):
        flags.append('<div class="flag rule">&#9873; MECHANISM RULING NEEDED — see notes below</div>')
    if a.get("sidedness_conflict"):
        flags.append('<div class="flag conf">&#9670; SIDEDNESS CONFLICT vs manifest — instrument-critical, rule this</div>')
    if a.get("cam_recheck"):
        flags.append('<div class="flag cam">camera_defining is arguable — re-judge with the locked-off-tripod test</div>')
    cards.append(f"""
<div class="card" id="{cls}">
 <h2>{cls} <small>v1 was: {a.get("v1_mechanism","?")} &rarr; v2: {a.get("mechanism")}{("/"+a["overlay_direction"]) if a.get("overlay_direction") else ""}</small></h2>
 {''.join(flags)}
 <div class="vids">{vids}</div>
 <div class="ctrls">{"".join(ctrls)}</div>
 <p class="notes">{html.escape(str(a.get("notes","")))}</p>
 <label class="ok"><input type="checkbox" data-cls="{cls}" class="validated"> validated</label>
</div>""")

LEGEND = """
<details id="legend" open>
<summary>Taxonomy v2 — mechanism decision procedure &amp; field definitions (gate-passed; judge from the FRAMES) · click to collapse</summary>
<div class="legend-body">
 <div class="def handoff"><b>Apply IN ORDER — the order IS the tie-breaker.</b> Endpoints are given as conditioning; classify what the MIDDLE must do. Apparatus (smoke, rings, petals, hands, blur) never decides.
 <b>TB0:</b> strip frame-wide treatment/motion-blur only when a residual effect remains underneath (a pure frame-wide restyle is T1); judge compounds at the maximal-effect (handoff) frame.</div>
 <div class="def" style="grid-column:1/-1"><h4>mechanism<span class="vals">cover / transform / overlay / traverse / cut</span></h4>
  <p class="sub"><b>T1 transform</b> — pre-existing content undergoes visible CONVERSION with correspondence: deforms; dissolves into matter derived from itself (with or WITHOUT reforming into B — dispersal-to-absence counts); substance substitution; re-render/restyle in place. External matter that covers, extracts, or deletes without visible conversion is never transform. <i>Conversion beats coverage.</i></p>
  <p class="sub"><b>T2 overlay</b> — B is the SAME scene as A changed only by content added / removed / accrued as state, while survivors keep tracker-confirmable identity. Sub-tag add / remove / state. <i>Same-scene beats coverage.</i></p>
  <p class="sub"><b>T3 cover</b> — frame substantially blocked by matter/flash at handoff; a DIFFERENT shot handed off at clearance, or inside the occluder's interior (screens). Residue into B's first frames allowed; translucent media passed through during ego-motion are not blocking. <i>Covered beats camera.</i></p>
  <p class="sub"><b>T4 traverse</b> — the CAMERA/view itself travels (through space or an open aperture: hole, doorway, haze) into B's place. Subject travels while camera stays &rarr; never traverse. Camera exits through the far side of an aperture &rarr; traverse; B lives inside the occluder (picture/screen) &rarr; cover.</p>
  <p class="sub"><b>T5 cut</b> — none of the above; a discontinuity underneath, staged/dressed.</p></div>
 <div class="def"><h4>scene_swap<span class="vals">yes / no</span></h4><p>First vs last frames: different shot/world &rarr; yes; same scene changed &rarr; no. Consistency: cover/traverse/cut &rArr; yes; overlay &rArr; no; transform may be either.</p></div>
 <div class="def"><h4>sidedness<span class="vals">A_only / B_only / two_sided</span></h4><p>Which endpoint's frames the effect visibly alters. <b style="color:#f77">INSTRUMENT-CRITICAL — feeds mask S; semantics FROZEN (v1). Rule the &#9670; conflicts.</b></p></div>
 <div class="def"><h4>camera_defining<span class="vals">yes / no</span></h4><p>"Replace the camera path with a locked-off tripod shot — does the effect still function?" No &rarr; yes. In doubt &rarr; no, unless the class's identity IS an ego-motion/scale-reveal move.</p></div>
 <div class="def"><h4>stylization<span class="vals">yes / no</span></h4><p>Frame-wide appearance treatment beyond the effect region at any point (class-majority across exemplars).</p></div>
 <div class="def"><h4>middle_only<span class="vals">yes / no</span></h4><p>Looking at the FIRST and LAST ~1s only: any effect matter/treatment visible in either? None &rarr; yes (conditioning contains zero evidence of the effect).</p></div>
 <div class="def"><h4>subject_anchored<span class="vals">yes / no</span></h4><p>Metadata only in v2 (no metric consumes it): effect originates from / targets / tracks one endpoint entity.</p></div>
</div>
</details>
"""

page = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Transition taxonomy v2 — validate</title>
<style>
body{font-family:system-ui;margin:16px;background:#111;color:#ddd}
.card{border:1px solid #333;border-radius:8px;padding:12px;margin:14px 0;background:#1a1a1a}
h2{margin:0 0 6px}h2 small{color:#888;font-weight:400;font-size:.7em}
.vids{display:flex;gap:8px;flex-wrap:wrap}.v{display:flex;flex-direction:column;font-size:.75em;color:#999}
video{width:240px;border-radius:4px;background:#000}
.ctrls{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
label{display:flex;flex-direction:column;font-size:.72em;color:#aaa}
label.crit select{outline:2px solid #b58900}
select{background:#222;color:#eee;border:1px solid #444;border-radius:4px;padding:2px}
select.changed{outline:2px solid #d33}
.flag{font-weight:600;font-size:.85em;margin:4px 0;padding:4px 8px;border-radius:5px}
.flag.rule{color:#fff;background:#7a1f1f}
.flag.conf{color:#111;background:#e0a34a}
.flag.cam{color:#9fd0c9;background:#15302c;font-weight:400}
.notes{color:#8a8;font-size:.85em;margin:6px 0}
.ok{flex-direction:row;gap:6px;align-items:center;font-size:.8em}
#bar{position:sticky;top:0;background:#111;padding:8px 0;z-index:9;border-bottom:1px solid #333}
button{background:#2a6;border:0;border-radius:5px;padding:8px 14px;color:#fff;font-weight:600;cursor:pointer}
#count{margin-left:12px;color:#888}
#legend{border:1px solid #2c3a37;border-radius:8px;background:#141d1a;margin:14px 0}
#legend>summary{cursor:pointer;padding:12px 14px;font-weight:600;font-size:.92em;color:#7fcfc6;list-style:none}
#legend>summary::-webkit-details-marker{display:none}
#legend>summary::before{content:"\\25b8  ";color:#4a8}
#legend[open]>summary::before{content:"\\25be  "}
.legend-body{padding:0 16px 14px;display:grid;grid-template-columns:1fr 1fr;gap:12px 26px}
.def{font-size:.82em;line-height:1.5}
.def h4{margin:8px 0 2px;color:#eee;font-size:1em;font-weight:600}
.def h4 .vals{color:#e0a34a;font-family:monospace;font-weight:400;font-size:.88em;margin-left:6px}
.def p{margin:2px 0;color:#b8c2bf}
.def .sub{color:#9fd0c9;margin:2px 0}
.handoff{grid-column:1/-1;color:#d7cfa0;border-bottom:1px solid #2c3a37;padding-bottom:9px;margin-bottom:2px}
@media(max-width:820px){.legend-body{grid-template-columns:1fr}}
</style></head><body>
<div id="bar"><button onclick="exp()">Export corrections.json</button><span id="count"></span>
<span style="color:#666;margin-left:12px">cards ordered: &#9873; rulings first, then &#9670; sidedness conflicts &middot; yellow outline = instrument-critical field &middot; red = you changed it</span></div>
""" + LEGEND + "\n".join(cards) + """
<script>
const sel=document.querySelectorAll('select');
sel.forEach(s=>s.addEventListener('change',()=>{s.classList.toggle('changed',s.value!==s.dataset.orig);n()}));
function n(){const c=[...sel].filter(s=>s.value!==s.dataset.orig).length;
document.getElementById('count').textContent=c+' correction(s), '+
document.querySelectorAll('.validated:checked').length+' validated';}
document.querySelectorAll('.validated').forEach(b=>b.addEventListener('change',n));
function exp(){const out={protocol:"v2",corrections:{},validated:[]};
sel.forEach(s=>{if(s.value!==s.dataset.orig){(out.corrections[s.dataset.cls]??={})[s.dataset.f]=s.value;}});
document.querySelectorAll('.validated:checked').forEach(b=>out.validated.push(b.dataset.cls));
const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));
a.download='corrections.json';a.click();}
n();
</script></body></html>"""
OUT.write_text(page)
print(f"wrote {OUT} ({len(axes)} classes, v2)")
