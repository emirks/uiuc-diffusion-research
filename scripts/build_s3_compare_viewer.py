"""Build the S3 (3D depth-parallax) comparison viewer.

Every S3 clip that exists on disk, grouped by the mechanism that produced it:

  A  exp_080_depth3d_realstream_121/run_0001   31 clips  APPROVED MECHANISM
  B  ctt_v2_s3/pilot                           63 clips  FAILED GATE PILOT
  C  exp_083_d3_pilot/run_0001                109 clips  SUPERSEDED MECHANISM

None of the three carries the three structural fixes of the currently-approved
design — this viewer is a record of what the pre-fix mechanisms produced.

House style follows outputs/videos/exp_083_d3_pilot/run_0001/viewer.html:
~230 px auto-fill grid, IntersectionObserver autoplay, preload="none",
filmstrip toggle, light/dark toggle, per-card metadata, colour-coded metrics.

Usage:  python scripts/build_s3_compare_viewer.py [repo_root]
Output: outputs/viewers/s3_dataset/index.html
Serve from the REPO ROOT (media is referenced as /outputs/...).
"""

from __future__ import annotations

import html
import json
import pathlib
import statistics as st
import sys

# ---------------------------------------------------------------- paths -----

A_RUN = "outputs/videos/exp_080_depth3d_realstream_121/run_0001"
B_RUN = "outputs/videos/ctt_v2_s3/pilot"
C_RUN = "outputs/videos/exp_083_d3_pilot/run_0001"
B_EXP = "experiments/exp_082_s3_stratum"
C_EXP = "experiments/exp_083_d3_pilot"
OUT = "outputs/viewers/s3_dataset/index.html"

# hole-radius gate from the exp_083 blind audit: every shippable clip sat below
# 85 px, 13 of the 14 BAD clips above it.
HOLE_OK, HOLE_WARN = 85.0, 120.0
JOIN_OK, JOIN_BAR = 1.3, 2.0      # join/seam ratio: bar 2.0, median bar 1.3
PI_BAR = 2.0                      # parallax index: 1.0 == flat, bar 2.0

# ------------------------------------------------------------------ css -----

CSS = """
:root{--bg:#0f1115;--fg:#e6e8ec;--mut:#9aa3b2;--card:#171a21;--line:#252a34;
      --acc:#7cc0ff;--acc2:#c4a3ff;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;
      --panel:#131720}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--fg:#12141a;--mut:#5d6675;
      --card:#fff;--line:#e2e5ea;--acc:#1668c9;--acc2:#6d3fc4;--ok:#15803d;
      --warn:#a16207;--bad:#b91c1c;--panel:#eef1f6}}
:root[data-theme=light]{--bg:#f7f8fa;--fg:#12141a;--mut:#5d6675;--card:#fff;
      --line:#e2e5ea;--acc:#1668c9;--acc2:#6d3fc4;--ok:#15803d;--warn:#a16207;
      --bad:#b91c1c;--panel:#eef1f6}
:root[data-theme=dark]{--bg:#0f1115;--fg:#e6e8ec;--mut:#9aa3b2;--card:#171a21;
      --line:#252a34;--acc:#7cc0ff;--acc2:#c4a3ff;--ok:#4ade80;--warn:#fbbf24;
      --bad:#f87171;--panel:#131720}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1560px;margin:0 auto;padding:30px 22px 90px}
h1{font-size:27px;margin:0 0 6px}
h2{font-size:21px;margin:0 0 2px}
h3{font-size:13px;margin:26px 0 6px;color:var(--acc2);font-weight:600;
   letter-spacing:.02em;grid-column:1/-1}
h3 small{color:var(--mut);font-weight:400;letter-spacing:0}
.sub{color:var(--mut);margin:0 0 14px;max-width:96ch}
.sub b{color:var(--fg)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
     background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:0 4px}
.banner{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--acc2);
        border-radius:10px;padding:13px 16px;margin:16px 0 22px}
.banner p{margin:0 0 8px}.banner p:last-child{margin:0}
.stats{display:flex;flex-wrap:wrap;gap:9px;margin:14px 0 6px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:9px 13px}
.stat b{display:block;font-size:19px;line-height:1.25}
.stat span{color:var(--mut);font-size:11.5px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;
      display:flex;flex-direction:column}
.card.v-bad{border-color:#7f1d1d}
.card.v-good{border-color:#14532d}
@media(prefers-color-scheme:light){.card.v-bad{border-color:#fca5a5}.card.v-good{border-color:#86efac}}
:root[data-theme=light] .card.v-bad{border-color:#fca5a5}
:root[data-theme=light] .card.v-good{border-color:#86efac}
.card video{width:100%;display:block;background:#000;aspect-ratio:480/640;object-fit:cover}
.strip{width:100%;display:block;border-top:1px solid var(--line)}
.meta{padding:8px 10px 10px;font-size:11.5px}
.meta .sh{font-weight:600;color:var(--acc);font-size:12.5px;word-break:break-word}
.meta .kv{color:var(--mut);margin-top:3px;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;
          word-break:break-word}
.meta .why{margin-top:4px;color:var(--bad);font-size:10.5px;font-style:italic}
.tag{display:inline-block;background:var(--line);border-radius:4px;padding:1px 5px;
     margin:0 4px 3px 0;font-size:10px;white-space:nowrap}
.tag.c-subject{background:#6d3fc4;color:#fff}.tag.c-world{background:#1668c9;color:#fff}
.tag.c-depth{background:#2f7d63;color:#fff}.tag.c-none{background:#6b7280;color:#fff}
.pill{display:inline-block;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700;
      margin:0 4px 3px 0}
.pill.bad{background:#b91c1c;color:#fff}.pill.good{background:#15803d;color:#fff}
.pill.none{background:var(--line);color:var(--mut)}
.pill.contested{background:#a16207;color:#fff}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.mut{color:var(--mut)}
.ghdr{border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:38px 0 16px;
      background:var(--panel)}
.ghdr.gA{border-left:4px solid var(--ok)}
.ghdr.gB{border-left:4px solid var(--bad)}
.ghdr.gC{border-left:4px solid var(--warn)}
.badge{display:inline-block;border-radius:5px;padding:2px 8px;font-size:11px;font-weight:700;
       vertical-align:middle;margin-left:9px}
.badge.gA{background:#15803d;color:#fff}.badge.gB{background:#b91c1c;color:#fff}
.badge.gC{background:#a16207;color:#fff}
.ctl{position:sticky;top:0;z-index:9;background:var(--bg);border-bottom:1px solid var(--line);
     padding:10px 0 9px;margin:6px 0 4px;display:flex;flex-wrap:wrap;gap:14px;align-items:center;
     font-size:12.5px;color:var(--mut)}
.ctl label{cursor:pointer}
.ctl select,.ctl button{background:var(--card);color:var(--fg);border:1px solid var(--line);
     border-radius:6px;padding:3px 7px;font:inherit;font-size:12px;cursor:pointer}
.ctl input[type=range]{vertical-align:middle;width:120px}
.count{margin-left:auto;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;font-size:12px;margin:6px 0 10px}
th,td{border:1px solid var(--line);padding:3px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600}
.key{font-size:12px;color:var(--mut);margin:8px 0 0}
.empty{grid-column:1/-1;color:var(--mut);font-style:italic;padding:8px 0}
"""

JS = """
(function(){
 var root=document.documentElement;
 var saved=null; try{saved=localStorage.getItem('s3theme');}catch(e){}
 if(saved) root.setAttribute('data-theme',saved);
 document.addEventListener('DOMContentLoaded',function(){
  var tb=document.getElementById('theme');
  function lbl(){var t=root.getAttribute('data-theme');tb.textContent='theme: '+(t||'auto');}
  lbl();
  tb.addEventListener('click',function(){
    var t=root.getAttribute('data-theme');
    var nx=t==='dark'?'light':(t==='light'?null:'dark');
    if(nx){root.setAttribute('data-theme',nx);try{localStorage.setItem('s3theme',nx);}catch(e){}}
    else{root.removeAttribute('data-theme');try{localStorage.removeItem('s3theme');}catch(e){}}
    lbl();});

  var strips=document.getElementById('showstrips');
  function syncStrips(){
    var d=strips.checked?'block':'none';
    document.querySelectorAll('.strip').forEach(function(s){s.style.display=d;});}
  strips.addEventListener('change',syncStrips); syncStrips();

  var play=document.getElementById('autoplay');
  var io=new IntersectionObserver(function(es){es.forEach(function(e){
    if(e.isIntersecting&&play.checked){e.target.play().catch(function(){});}
    else{e.target.pause();}});},{threshold:.25});
  document.querySelectorAll('video').forEach(function(v){io.observe(v);});
  play.addEventListener('change',function(){
    document.querySelectorAll('video').forEach(function(v){
      if(play.checked&&v.getBoundingClientRect().top<innerHeight&&v.getBoundingClientRect().bottom>0)
        v.play().catch(function(){});
      else v.pause();});});

  var vs=document.getElementById('verdict'),
      so=document.getElementById('sort'),
      hr=document.getElementById('holemin'),
      hl=document.getElementById('holelbl'),
      gs=document.getElementById('grp'),
      ct=document.getElementById('count');

  function apply(){
    var v=vs.value, s=so.value, hmin=+hr.value, g=gs.value, shown=0, total=0;
    hl.textContent=hmin?('\\u2265 '+hmin+' px'):'off';
    document.querySelectorAll('section.group').forEach(function(sec){
      var vis=(g==='all'||g===sec.dataset.g);
      sec.style.display=vis?'':'none';
      var grid=sec.querySelector('.grid');
      var items=Array.prototype.slice.call(grid.children);
      var cards=items.filter(function(e){return e.classList.contains('card');});
      var here=0;
      cards.forEach(function(c){
        total++;
        var ok=true;
        if(v!=='all'&&c.dataset.verdict!==v) ok=false;
        if(hmin>0){var h=parseFloat(c.dataset.hole); if(!(h>=hmin)) ok=false;}
        c.style.display=ok?'':'none';
        if(ok){here++; if(vis) shown++;}});
      // order
      var keyed=(s!=='default');
      if(keyed){
        var arr=cards.slice().sort(function(a,b){
          var av=parseFloat(a.dataset[s]), bv=parseFloat(b.dataset[s]);
          if(isNaN(av))av=-1; if(isNaN(bv))bv=-1;
          if(s==='pi') return av-bv;          // flattest first
          return bv-av;});                    // biggest first
        arr.forEach(function(c){grid.appendChild(c);});
        items.filter(function(e){return e.tagName==='H3';})
             .forEach(function(h){h.style.display='none';grid.appendChild(h);});
      }else{
        items.slice().sort(function(a,b){return (+a.dataset.idx)-(+b.dataset.idx);})
             .forEach(function(e){grid.appendChild(e);});
        // a subheading is shown only if a visible card follows it before the next one
        var kids=Array.prototype.slice.call(grid.children), cur=null, any=false;
        kids.forEach(function(e){
          if(e.tagName==='H3'){ if(cur) cur.style.display=any?'':'none'; cur=e; any=false; }
          else if(e.classList.contains('card')&&e.style.display!=='none'){any=true;}});
        if(cur) cur.style.display=any?'':'none';
      }
      var em=sec.querySelector('.empty');
      if(em){em.style.display=(here===0)?'':'none'; grid.appendChild(em);}
    });
    ct.textContent=shown+' / '+total+' clips shown';
  }
  [vs,so,gs].forEach(function(e){e.addEventListener('change',apply);});
  hr.addEventListener('input',apply);
  document.getElementById('reset').addEventListener('click',function(){
    vs.value='all';so.value='default';hr.value=0;gs.value='all';apply();});
  apply();
 });
})();
"""

# ----------------------------------------------------------- formatting -----


def esc(x) -> str:
    return html.escape(str(x))


def cls_join(v: float | None) -> str:
    if v is None:
        return ""
    return "ok" if v <= JOIN_OK else ("warn" if v <= JOIN_BAR else "bad")


def cls_hole(v: float | None) -> str:
    if v is None:
        return ""
    return "ok" if v < HOLE_OK else ("warn" if v < HOLE_WARN else "bad")


def cls_pi(v: float | None) -> str:
    if v is None:
        return ""
    return "bad" if v <= 1.0 else ("warn" if v < PI_BAR else "ok")


def table(head, rows) -> str:
    h = "".join(f"<th>{esc(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def stat(v, label) -> str:
    return f'<div class="stat"><b>{v}</b><span>{label}</span></div>'


def card_html(c: dict, idx: int) -> str:
    """One clip card. `c` is the normalised record built below."""
    vd = c["verdict"] or "none"
    tags = "".join(c["tags"])
    if c["verdict"] == "bad":
        pill = '<span class="pill bad">BAD</span>'
    elif c["verdict"] == "good":
        pill = '<span class="pill good">GOOD</span>'
    else:
        pill = '<span class="pill none">unlabelled</span>'
    if c.get("contested"):
        pill += '<span class="pill contested">contested</span>'

    bits = []
    if c["join"] is not None:
        bits.append(f'{c["join_name"]} <span class="{cls_join(c["join"])}">{c["join"]:.2f}</span>')
    if c["hole"] is not None:
        bits.append(f'hole <span class="{cls_hole(c["hole"])}">{c["hole"]:.0f}px</span>')
    else:
        bits.append('hole <span class="mut">n/a</span>')
    if c["pi"] is not None:
        bits.append(f'PI <span class="{cls_pi(c["pi"])}">{c["pi"]:.2f}</span>')
    metrics = " &middot; ".join(bits)

    why = f'<div class="why">{esc(c["reason"])}</div>' if c.get("reason") else ""
    extra = f'<div class="kv">{esc(c["extra"])}</div>' if c.get("extra") else ""
    hole_attr = "" if c["hole"] is None else f"{c['hole']:.1f}"
    return f"""<div class="card v-{vd}" data-idx="{idx}" data-verdict="{vd}"
 data-hole="{hole_attr}" data-join="{'' if c['join'] is None else f'{c["join"]:.3f}'}"
 data-pi="{'' if c['pi'] is None else f'{c["pi"]:.3f}'}">
<video src="{c['video']}" muted loop playsinline preload="none"></video>
<a href="{c['strip']}" target="_blank" rel="noopener"><img class="strip" src="{c['strip']}" loading="lazy" alt=""></a>
<div class="meta">
 <div class="sh">{esc(c['title'])}</div>
 <div>{pill}{tags}</div>
 <div class="kv">{esc(c['describe'])}</div>
 <div class="kv">{esc(c['from'])} &rarr; {esc(c['to'])}</div>
 <div class="kv">{metrics}</div>
 {extra}{why}
</div></div>"""


def grid_html(sections: list[tuple[str, str, list[dict]]]) -> str:
    """sections = [(subheading, subheading_note, [records])]"""
    out, idx = ['<div class="grid">'], 0
    for head, note, rows in sections:
        if head:
            n = f' <small>{note}</small>' if note else ""
            out.append(f'<h3 data-idx="{idx}">{head}{n}</h3>')
            idx += 1
        for r in rows:
            out.append(card_html(r, idx))
            idx += 1
    out.append('<div class="empty" data-idx="99999">nothing matches the current filter.</div>')
    out.append("</div>")
    return "".join(out)


def tag(txt, kind="") -> str:
    k = f" c-{kind}" if kind else ""
    return f'<span class="tag{k}">{esc(txt)}</span>'


def op_tags(p: dict) -> list[str]:
    """The optional-operator tags shared by all three groups' param dicts."""
    t = []
    if p.get("dissolve", "none") != "none":
        t.append(tag(f'dissolve {p["dissolve"]}'))
    if p.get("blend") and p["blend"] != "crossfade":
        t.append(tag(p["blend"]))
    if p.get("fog", 0):
        t.append(tag("fog"))
    if p.get("focus", 0):
        t.append(tag("rack focus"))
    if p.get("handheld", 0):
        t.append(tag("handheld"))
    if p.get("dolly_zoom", 0):
        t.append(tag("dolly-zoom"))
    if p.get("motion_blur", 1) > 1:
        t.append(tag(f'mblur {p["motion_blur"]}x'))
    return t


# ------------------------------------------------------------------ main ----


def main() -> None:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    L = lambda p: json.load(open(root / p))  # noqa: E731

    # ---------------------------------------------------------- group A ----
    a_man = L(f"{A_RUN}/manifest.json")
    a_recs = []
    for m in a_man:
        p = m["params"]
        j = max(m["join_ratio_in"], m["join_ratio_out"])
        a_recs.append({
            "group": "A", "sub": m["tag"], "stem": m["stem"],
            "title": f'{m["family"]} · {m["blend"]}',
            "from": m["from"], "to": m["to"], "describe": m["describe"],
            "join": j, "join_name": "join", "hole": None,
            "pi": m["parallax"]["pi"], "verdict": None, "reason": None,
            "video": f"/{A_RUN}/videos/{m['stem']}.mp4",
            "strip": f"/{A_RUN}/filmstrips/{m['stem']}.jpg",
            "extra": (f'121f · onset {m["onset"]} release {m["release"]} · '
                      f'rho {m["parallax"]["rho"]:.2f} · flow {m["parallax"]["flow_px"]:.1f}px · '
                      f'{m["render_s"]:.1f}s'),
            "tags": [tag(m["family"]), tag("121f"), tag(m["tag"])] + op_tags(p),
        })
    a_join = sorted(r["join"] for r in a_recs)
    a_pi = sorted(r["pi"] for r in a_recs)
    a_sec = st.mean(m["render_s"] for m in a_man)
    # the run log quotes the median pooled over both joins (0.94); the cards and
    # the tile above quote the worse of the two joins per clip, which is higher.
    a_pooled = st.median([m["join_ratio_in"] for m in a_man]
                         + [m["join_ratio_out"] for m in a_man])

    A_SUBS = [
        ("family showcase", "one clip per camera family, shared timing", "family"),
        ("effect showcase", "one clip per optional operator", None),   # effect_*
        ("shared operator 0", "same operator, different content pairs", "sharedop0"),
        ("shared operator 1", "same operator, different content pairs", "sharedop1"),
        ("counterfactual", "same pair, operator varied", "counterfactual"),
        ("diverse", "free sample of the operator space", "diverse"),
    ]
    a_sections = []
    for head, note, key in A_SUBS:
        rows = ([r for r in a_recs if r["sub"].startswith("effect_")] if key is None
                else [r for r in a_recs if r["sub"] == key])
        if rows:
            a_sections.append((f"{head} ({len(rows)})", note, rows))

    # ---------------------------------------------------------- group B ----
    b_pr = L(f"{B_EXP}/PILOT_RESULT.json")
    b_adj = L(f"{B_EXP}/GATE_ADJUDICATED.json")["per_clip_label"]
    b_blind = {e["stem"]: e["bad"] for e in L(f"{B_EXP}/PILOT_VISUAL_AUDIT.json")}
    b_cal = {e["stem"]: e for e in L(f"{B_EXP}/GATE_CALIB.json")["per_clip"]}
    b_rel = L(f"{B_EXP}/relabel_registry/RELABEL_AGREEMENT.json")
    b_contested = set(b_rel["contested"])
    b_raters = b_rel["raters"]
    b_recs = []
    for c in b_pr["clips"]:
        p, s = c["params"], c["stem"]
        cal = b_cal.get(s, {})
        lab = b_adj.get(s)
        flags = sum(bool(b_raters[k].get(s)) for k in ("operator_original", "passA", "passB"))
        b_recs.append({
            "group": "B", "sub": c["family"], "stem": s,
            "title": s,
            "from": c["A"], "to": c["B"], "describe": c["describe"],
            "join": c["join_max"], "join_name": "join",
            "hole": cal.get("hole_r_max"), "pi": c["parallax"]["pi"],
            "verdict": (None if lab is None else ("bad" if lab else "good")),
            "contested": s in b_contested,
            "reason": None,
            "video": f"/{B_RUN}/videos/{s}.mp4",
            "strip": f"/{B_RUN}/filmstrips/{s}.jpg",
            "extra": (f'121f · onset {c["onset"]} release {c["release"]} · '
                      f'uncovered {100*cal.get("unc_max", 0):.0f}% max · '
                      f'raters flagged {flags}/3 · blind r1 '
                      f'{"BAD" if b_blind.get(s) else "good"} · {c["render_s"]:.1f}s'),
            "tags": [tag(c["family"]), tag("121f"), tag(c["tag"])] + op_tags(p),
        })
    b_bad = sum(1 for r in b_recs if r["verdict"] == "bad")
    b_good = sum(1 for r in b_recs if r["verdict"] == "good")
    b_fam = sorted({r["sub"] for r in b_recs})
    b_sections = [(f"{f} ({sum(1 for r in b_recs if r['sub'] == f)})",
                   f"{sum(1 for r in b_recs if r['sub'] == f and r['verdict'] == 'bad')} adjudicated BAD",
                   [r for r in b_recs if r["sub"] == f]) for f in b_fam]
    b_holes_bad = [r["hole"] for r in b_recs if r["verdict"] == "bad" and r["hole"] is not None]
    b_holes_good = [r["hole"] for r in b_recs if r["verdict"] == "good" and r["hole"] is not None]

    # ---------------------------------------------------------- group C ----
    c_man = L(f"{C_RUN}/manifest.json")
    c_pr = L(f"{C_EXP}/PILOT_RESULT.json")
    c_aud = L(f"{C_EXP}/PILOT_VISUAL_AUDIT.json")
    c_sampled, c_bad, c_reason = set(c_aud["sampled"]), c_aud["bad"], c_aud["reason_class"]
    COUPLING = {"none": "camera only", "depth": "depth-coupled",
                "world": "world-coupled", "subject": "subject-anchored"}
    c_recs = []
    for m in c_man:
        s, p = m["stem"], m["params"]
        seam = max(m["seam_ratio_in"], m["seam_ratio_out"])
        v = ("bad" if s in c_bad else "good") if s in c_sampled else None
        c_recs.append({
            "group": "C", "sub": m["block"], "stem": s,
            "title": f'{m["recipe"]} · {m["family"]}',
            "from": m["from"], "to": m["to"], "describe": m["describe"],
            "join": seam, "join_name": "seam", "hole": m["hole_radius_max"],
            "pi": m["parallax"]["pi"], "verdict": v,
            "reason": (f'{c_reason.get(s, "")}: {c_bad[s]}' if s in c_bad else None),
            "video": f"/{C_RUN}/videos/{s}.mp4",
            "strip": f"/{C_RUN}/filmstrips/{s}.jpg",
            "extra": (f'{m["n_frames"]}f = 9 + {m["n_middle"]} + 9 · '
                      f'uncovered {100*m["uncovered_max"]:.0f}% max · '
                      f'dissolve {m["dissolve"]} · {m["label_from"]}→{m["label_to"]} · '
                      f'{m["render_s"]:.1f}s'),
            "tags": [tag(COUPLING[m["coupling"]], m["coupling"]), tag(m["family"]),
                     tag(f'{m["n_frames"]}f')] + op_tags(p),
        })
    C_SUBS = [
        ("length", "LENGTH SWEEP — one content pair, 4 operators × n_frames 25/33/41/49; "
                   "seam falls monotonically as the middle gets longer"),
        ("axisop0", "OPERATOR AXIS — one content pair held fixed, 11 different operators"),
        ("axisop1", "OPERATOR AXIS — a second content pair held fixed, the same 11 operators"),
        ("axiscontent0", "CONTENT AXIS — one operator (subject_smoke, roll, 25f) over 6 pairs"),
        ("axiscontent1", "CONTENT AXIS — one operator (world_fbm, spiral, 41f) over 6 pairs"),
        ("axiscontent2", "CONTENT AXIS — one operator (world_worley, crane, 25f) over 6 pairs"),
        ("axiscontent3", "CONTENT AXIS — one operator (bare_move, orbit, 33f) over 6 pairs"),
        ("amp", "amplitude sweep — one pair, one operator, 10 camera amplitudes"),
        ("family", "one clip per camera family"),
        ("diverse", "free sample of the operator × content space"),
    ]
    c_sections = []
    for key, note in C_SUBS:
        rows = [r for r in c_recs if r["sub"] == key]
        if rows:
            c_sections.append((f"{key} ({len(rows)})", note, rows))
    seen = {k for k, _ in C_SUBS}
    rest = [r for r in c_recs if r["sub"] not in seen]
    if rest:
        c_sections.append((f"other ({len(rest)})", "", rest))
    c_seam_len = c_pr["seam"]["by_n_frames"]
    c_hole_fam = c_pr["disocclusion"]["by_dissolve_family"]
    c_bad_n, c_samp_n = len(c_bad), len(c_sampled)
    c_holes_bad = [r["hole"] for r in c_recs if r["verdict"] == "bad"]
    c_holes_good = [r["hole"] for r in c_recs if r["verdict"] == "good"]

    # ------------------------------------------------------------- page ----
    total = len(a_recs) + len(b_recs) + len(c_recs)

    head_A = f"""<div class="ghdr gA"><h2>A &middot; exp_080_depth3d_realstream_121 / run_0001
<span class="badge gA">APPROVED MECHANISM</span></h2>
<p class="sub"><b>Full 121 frames on the D2 contract.</b> Both streams keep playing the whole way
through — the world never freezes — and the depth used to displace the mesh is a
<b>per-frame, temporally-stabilised</b> Depth&nbsp;Anything&nbsp;V2 stack, not one frozen map. The pure
phases before onset and after release are <b>byte-identical</b> to their source streams in every
clip. This is the mechanism the owner looked at and approved; it is the one to build on.</p>
<div class="stats">
{stat(len(a_recs), 'clips')}
{stat(f'{st.median(a_join):.2f}', 'join ratio, median of the worse join (bar 2.0)')}
{stat(f'{a_join[int(.9*len(a_join))]:.2f} / {max(a_join):.2f}', 'join p90 / max')}
{stat(f'{st.median(a_pi):.2f}', 'parallax index, median (1.0 = flat, bar 2.0)')}
{stat(f'{sum(r["join"] > JOIN_BAR for r in a_recs)}', 'clips over the 2.0 join bar')}
{stat(f'{len({r["from"] for r in a_recs} | {r["to"] for r in a_recs})}',
      'distinct endpoint clips (4 pairs)')}
{stat(f'{a_sec:.0f}s', 'mean render per clip (L40S)')}
</div>
<p class="key">Tiles and cards quote the <b>worse of a clip's two joins</b>; pooled over both joins the
median is {a_pooled:.2f}, which is the 0.94 quoted in the run log. Both are on the same
bar&nbsp;2.0.</p>
<p class="key"><b>Read this before you compare A to B.</b> These 31 clips cover only
<b>4 content pairs / 8 distinct source clips</b>, all from the stock-VFX
<code>transitions_std121</code> bank (money_rain, portal, shadow_smoke, gas_transformation,
polygon, action_run, super_fast_run). Group B is the <i>same engine at the same
<code>amplitude_scale&nbsp;1.6</code></i> run over 63 real-video pairs. No per-clip hole-radius was
computed for this run and no clip here carries an adjudicated verdict, so every card in group A is
<span class="pill none">unlabelled</span> and shows <code>hole n/a</code>. The clean numbers are
real, but they are measured on a much narrower slice of content than B's.</p></div>"""

    head_B = f"""<div class="ghdr gB"><h2>B &middot; ctt_v2_s3 / pilot (exp_082 S3 stratum)
<span class="badge gB">FAILED GATE PILOT</span></h2>
<p class="sub"><b>The same 121-frame mechanism as A</b> — <code>engine3d/</code> copied byte-identical
from exp_080, same <code>amplitude_scale 1.6</code> — run as the mandatory pilot before a planned
1,800-clip S3 render, this time over <b>63 distinct real-video pairs</b> (DAVIS / VCBench / OpenVid)
instead of A's four. It came back <b>{b_bad} BAD / {b_good} GOOD</b> on adjudicated labels
({100*b_bad/(b_bad+b_good):.0f}% defective) and the stratum was dropped.</p>
<div class="stats">
{stat(len(b_recs), 'clips')}
{stat(f'{b_bad} / {b_good}', 'adjudicated BAD / GOOD')}
{stat(f'{b_pr["join_distribution"]["median"]:.2f}', 'join median (bar 1.3)')}
{stat(f'{b_pr["join_distribution"]["p90"]:.2f} / {b_pr["join_distribution"]["max"]:.2f}',
      'join p90 / max — 6 clips over 2.0')}
{stat(f'{b_pr["parallax"]["median_pi"]:.2f}', 'parallax median — FAILS the 2.0 bar')}
{stat(f'{len({r["from"] for r in b_recs} | {r["to"] for r in b_recs})}',
      'distinct endpoint clips (63 pairs)')}
{stat(f'{b_pr["timing"]["sec_mean"]:.1f}s', 'mean render per clip')}
</div>
<p class="key"><b>Labels are noisy and that is measured, not assumed.</b> Three raters — the operator
plus two fully blind passes with independent shuffles — flagged
{100*b_rel["flag_rates"]["operator"]:.0f}% / {100*b_rel["flag_rates"]["A"]:.0f}% /
{100*b_rel["flag_rates"]["B"]:.0f}% of the same 63 clips. Pairwise Cohen &kappa; ran
{min(v["kappa"] for v in b_rel["pairwise"].values()):.2f}–{max(v["kappa"] for v in b_rel["pairwise"].values()):.2f};
{len(b_rel["unanimous"])} clips were unanimous and {len(b_contested)} contested. Cards carry the
<b>adjudicated</b> verdict (A/B agreement + operator adjudication of the contested ones); an amber
<span class="pill contested">contested</span> pill marks the {len(b_contested)} where the raters
disagreed, and each card's metadata line reports how many of the three flagged it.</p>
<p class="key"><b>And the hole-radius gate does not save this group.</b> Median
<code>hole_r_max</code> is {st.median(b_holes_bad):.0f}&nbsp;px on the BAD clips against
{st.median(b_holes_good):.0f}&nbsp;px on the GOOD ones — overlapping, not separating. No statistic
tried (coverage quantity, hole radius, salience-weighted location, patch resemblance) reached the
pre-committed &ge;87% recall at &le;7.5% FP operating point, so <code>S3_DROPPED.json</code>
concluded the defect is inpaint <i>plausibility</i>, a semantic property, and not a measurable
geometric quantity. Sorting this group by hole radius is therefore instructive precisely because
it <b>fails</b> to sort the red cards to the top.</p></div>"""

    head_C = f"""<div class="ghdr gC"><h2>C &middot; exp_083_d3_pilot / run_0001
<span class="badge gC">SUPERSEDED MECHANISM</span></h2>
<p class="sub"><b>The older exp_076-style construction.</b> Each clip is
<b>9 verbatim anchor frames + a rendered middle + 9 verbatim anchor frames</b>, total
25/33/41/49 frames (every length legal for the causal VAE, F&nbsp;=&nbsp;8k+1). The depth is a
<b>single static map per endpoint</b> and the endpoints are <b>frozen</b> — the world stops while the
camera flies. Blind audit: <b>{c_bad_n} of {c_samp_n} sampled clips BAD ({100*c_bad_n/c_samp_n:.0f}%)</b>,
sample drawn from a fixed seed before any clip was viewed. The other
{len(c_recs)-c_samp_n} clips were never labelled and show as
<span class="pill none">unlabelled</span>.</p>
<div class="stats">
{stat(len(c_recs), 'clips')}
{stat(f'{c_bad_n} / {c_samp_n}', 'blind BAD / sampled (47%)')}
{stat(f'{c_pr["seam"]["per_clip_worst_of_two_joins"]["median"]:.2f}', 'seam ratio, median')}
{stat(f'{c_pr["disocclusion"]["hole_radius_px_all"]["median"]:.0f}px',
      'hole radius, median (gate at 85 px)')}
{stat(f'{c_pr["parallax"]["pi"]["median"]:.2f}', 'parallax index, median')}
{stat(f'{c_pr["n_distinct_endpoint_clips"]}', 'distinct endpoint clips (40 pairs)')}
{stat(f'{c_pr["cost"]["render_s_per_clip"]["median"]:.1f}s', 'median CPU render per clip')}
</div>
<p class="key"><b>The two controlled axes are kept as separate sub-grids below</b> (they collapse when
you sort): <code>axisop0/1</code> hold the content pair fixed and vary the operator,
<code>axiscontent0–3</code> hold the operator fixed and vary the content pair, and
<code>length</code> is the 25/33/41/49 sweep. Longer middles are cleanly monotone in seam because
the same total motion is spread over more frames:</p>
{table(["n_frames", "clips", "seam median", "p90", "max"],
       [[k, v["n"], f'{v["median"]:.2f}', f'{v["p90"]:.2f}', f'{v["max"]:.2f}']
        for k, v in sorted(c_seam_len.items(), key=lambda kv: int(kv[0]))])}
<p class="key">Hole radius by dissolve family — the defect is not spread evenly, it tracks how far
the dissolve pushes the camera past what the single depth layer ever saw:</p>
{table(["dissolve", "clips", "hole radius median", "p90", "max"],
       [[k, v["n"], f'{v["median"]:.0f}px', f'{v["p90"]:.0f}px', f'{v["max"]:.0f}px']
        for k, v in sorted(c_hole_fam.items(), key=lambda kv: kv[1]["median"])])}
<p class="key">On this group the gate <i>does</i> work: median <code>hole_radius_max</code> is
{st.median(c_holes_bad):.0f}&nbsp;px on the BAD clips against {st.median(c_holes_good):.0f}&nbsp;px on
the shippable ones. Sort by <b>hole radius</b> and the red cards come to the top.</p></div>"""

    parts = [f"""<title>S3 — every 3D depth-transition clip, three mechanisms side by side</title>
<style>{CSS}</style><script>{JS}</script>
<div class="wrap">
<h1>S3 &mdash; 3D depth-parallax transitions: all {total} clips, three mechanisms</h1>
<p class="sub">Every S3 clip that exists on disk. S3 means the transition is rendered in
2.5D: each facing frame is unprojected by its Depth&nbsp;Anything&nbsp;V2 depth map into a displaced
mesh, and one continuous virtual camera flies out of scene&nbsp;A and comes to rest in scene&nbsp;B,
so near geometry sweeps faster than far geometry. No 2D shader can do that. Three different
mechanisms produced these clips; they are grouped by mechanism and each group's header states its
format, its depth handling, its measured numbers and its status.</p>
<div class="banner">
<p><b>All {total} clips predate the three structural fixes.</b> The currently-approved S3 design has
not been built — zero clips exist with its fixes. What is on this page is the record of what each
earlier mechanism actually produced, which is what makes the comparison worth looking at.</p>
<p><b>The shared defect, in all three groups.</b> <code>composite()</code> computes the total alpha
<code>den</code> per pixel and hands it to <code>_fill_holes</code>, a push-pull inpainter that
blurs at 9/31/81&nbsp;px and so cannot pull real colour across more than about
<b>40&nbsp;px</b>. Wherever the camera looks past the edge of the mesh, <code>den</code> collapses:
small holes inpaint invisibly, medium ones come out as a <b>flat smear or a melted blob</b>, large
ones stay <b>hard black</b>. The world-space dissolve compounds it — it thresholds layer&nbsp;A and
layer&nbsp;B against <b>independent</b> fields, so both layers can be absent at the same pixel and at
mid-transition roughly <b>25% of the frame</b> has no geometry from either scene.</p>
<p><b>To see it for yourself:</b> set <i>verdict</i> to <code>BAD</code>, or drag the
<i>min hole radius</i> slider up past 120&nbsp;px, or sort by <b>hole radius</b>. Then flip the
verdict filter to <code>GOOD</code> for the contrast. Turn on filmstrips — the defect is far easier
to catch in a strip of stills than in a 230&nbsp;px looping video; click any strip for the
full-resolution version.</p>
</div>
<div class="stats">
{stat(total, 'clips total')}
{stat(f'{len(a_recs)} / {len(b_recs)} / {len(c_recs)}', 'group A / B / C')}
{stat(f'{b_bad + c_bad_n}', 'clips with an adjudicated or blind BAD verdict')}
{stat(f'{b_good + len(c_holes_good)}', 'clips explicitly judged GOOD')}
{stat(f'{len(a_recs) + (len(c_recs) - c_samp_n)}', 'clips never labelled')}
</div>
<p class="key">Colour key &mdash; join / seam ratio: <span class="ok">&le; 1.3</span>
<span class="warn">&le; 2.0 (the bar)</span> <span class="bad">&gt; 2.0</span> &middot;
hole radius: <span class="ok">&lt; 85 px</span> <span class="warn">85&ndash;120</span>
<span class="bad">&gt; 120</span> &middot; parallax index:
<span class="bad">&le; 1.0 (flat)</span> <span class="warn">&lt; 2.0</span>
<span class="ok">&ge; 2.0</span>. A ratio near 1.0 means the join is as smooth as the content's own
natural motion.</p>

<div class="ctl">
<label><input type="checkbox" id="autoplay" checked> autoplay on scroll</label>
<label><input type="checkbox" id="showstrips"> filmstrips</label>
<label>group <select id="grp">
 <option value="all">all three</option><option value="A">A — approved</option>
 <option value="B">B — failed pilot</option><option value="C">C — superseded</option></select></label>
<label>verdict <select id="verdict">
 <option value="all">all</option><option value="bad">BAD only</option>
 <option value="good">GOOD only</option><option value="none">unlabelled only</option></select></label>
<label>sort <select id="sort">
 <option value="default">grouped by axis</option><option value="hole">hole radius &darr;</option>
 <option value="join">join / seam &darr;</option><option value="pi">parallax &uarr; (flattest first)</option>
 </select></label>
<label>min hole <input type="range" id="holemin" min="0" max="320" step="5" value="0">
 <span id="holelbl">off</span></label>
<button id="reset">reset</button>
<button id="theme">theme</button>
<span class="count" id="count"></span>
</div>
"""]

    parts.append(f'<section class="group" data-g="A">{head_A}{grid_html(a_sections)}</section>')
    parts.append(f'<section class="group" data-g="B">{head_B}{grid_html(b_sections)}</section>')
    parts.append(f'<section class="group" data-g="C">{head_C}{grid_html(c_sections)}</section>')
    parts.append("</div>")

    out = root / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"  A {len(a_recs)}  B {len(b_recs)} ({b_bad} bad/{b_good} good)  "
          f"C {len(c_recs)} ({c_bad_n} bad/{c_samp_n} sampled)  total {total}")


if __name__ == "__main__":
    main()
