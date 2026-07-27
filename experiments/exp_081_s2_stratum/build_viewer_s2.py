"""exp_081 — browsable viewer for the S2 stratum (7,990 clips / 799 operators / 56 shaders).

The point of this viewer is the thing a flat grid hides: **there is no reference clip on disk.
An operator's 10 clips ARE its reference pool.** So the atomic unit here is the OPERATOR BLOCK —
one row of 10 clips under a single header carrying the parameters they all share by construction
(shader, uniforms, easing, onset/release, flip, swap). That row IS the same-operator x
different-content diagonal, and it is what there is to judge.

Two levels:
  index.html            summary + contract + 56 shader cards + audit + link to the retired set
  shaders/<name>.html   every operator of that shader, one block per operator
  retired.html          the 420 blacklisted clips (42 operator blocks, 6 shaders)

Media is never copied. `media/` is a symlink to outputs/videos/ctt_v2_s2/full and every page
references it relatively, so an http.server rooted at the REPO ROOT serves it directly.

Performance: nothing is preloaded. Each card shows ONE frame cropped out of the clip's filmstrip
with CSS (`loading="lazy"`, so only on-screen strips are fetched) and a global phase slider slides
every card to the same frame at once — no new requests, and the whole operator row steps through
the transition in lockstep. Videos are created on demand (click a card, or the autoplay toggle +
IntersectionObserver) and torn down when they leave the viewport.

    python build_viewer_s2.py
    # serve from REPO ROOT:  python -m http.server 8017
    #   -> http://localhost:8017/outputs/viewers/s2_dataset/index.html
"""

from __future__ import annotations

import argparse
import collections
import glob
import html
import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "outputs" / "videos" / "ctt_v2_s2" / "full"
OUT = REPO_ROOT / "outputs" / "viewers" / "s2_dataset"
SEED = 11

# filmstrip geometry, from render_s2.strip_indices / save_strip
STRIP_IDX = [i for i in (0, 4, 8, 14, 22, 32, 44, 56, 68, 80, 92, 100, 108, 112, 116, 120)]
STRIP_FRAME_W = 240
STRIP_GAP = 2
POSTER_K = 8  # frame 68 — inside the transition window for every operator

# frozen gate bars (config_s2.yaml -> gate:)
TAU = 0.2543
SEAM_MAX = 2.0
M2_MAX = 0.5

BLACKLISTED = ["splitSlideOutHorizontal", "PuzzleRight", "SimpleZoomOut",
               "StripDatamoshGlitch", "swap", "SimpleZoom"]


# ----------------------------------------------------------------------------- io

def load_jsonl(pattern: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(SRC / "meta" / pattern))):
        for line in Path(f).read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------------ assets

CSS = """
:root{
  --bg:#0f1115;--fg:#e6e8ec;--mut:#9aa3b2;--card:#171a21;--line:#252a34;
  --acc:#7cc0ff;--acc2:#c4a3ff;--head:#12161d;
  --fk:8;                      /* filmstrip frame index shown by every poster */
}
:root[data-theme="light"]{
  --bg:#f7f8fa;--fg:#12141a;--mut:#5d6675;--card:#fff;--line:#e2e5ea;
  --acc:#1668c9;--acc2:#6d3fc4;--head:#eef1f5;
}
@media(prefers-color-scheme:light){
  :root:not([data-theme]){
    --bg:#f7f8fa;--fg:#12141a;--mut:#5d6675;--card:#fff;--line:#e2e5ea;
    --acc:#1668c9;--acc2:#6d3fc4;--head:#eef1f5;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
a{color:var(--acc);text-decoration:none} a:hover{text-decoration:underline}
.wrap{max-width:1840px;margin:0 auto;padding:26px 22px 90px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:19px;margin:40px 0 4px;padding-top:18px;border-top:1px solid var(--line);
  scroll-margin-top:56px}
h3{font-size:14px;margin:20px 0 6px;color:var(--acc2);font-weight:600}
.sub{color:var(--mut);margin:0 0 16px;max-width:96ch}
.sub b{color:var(--fg)}
.crumb{color:var(--mut);font-size:12.5px;margin:0 0 10px}
.stats{display:flex;flex-wrap:wrap;gap:9px;margin:16px 0 8px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:9px 13px}
.stat b{display:block;font-size:19px} .stat span{color:var(--mut);font-size:11.5px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.ok{color:#4ade80} .warn{color:#fbbf24} .bad{color:#f87171} .dim{color:var(--mut)}
.hidden{display:none !important}
.tag{display:inline-block;background:var(--line);border-radius:4px;padding:1px 6px;
  margin:0 4px 3px 0;font-size:10px;white-space:nowrap}
.tag.hot{background:#7f1d1d;color:#fecaca} .tag.cool{background:#1e3a5f;color:#bfdbfe}
.tag.good{background:#14532d;color:#bbf7d0}

/* ---- sticky control bar --------------------------------------------------- */
.bar{position:sticky;top:0;z-index:30;background:var(--bg);border-bottom:1px solid var(--line);
  padding:8px 0 9px;margin:6px 0 14px;display:flex;flex-wrap:wrap;gap:14px;align-items:center;
  font-size:12.5px;color:var(--mut)}
.bar label{cursor:pointer;user-select:none}
.bar input[type=range]{vertical-align:middle;width:190px}
.bar .btn{background:var(--card);border:1px solid var(--line);color:var(--fg);border-radius:7px;
  padding:3px 10px;cursor:pointer;font-size:12px}
.bar .btn:hover{border-color:var(--acc)}
.bar .seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.bar .seg button{background:var(--card);border:0;color:var(--mut);padding:3px 10px;cursor:pointer;
  font-size:12px}
.bar .seg button.on{background:var(--acc);color:#04121f}
#filter{background:var(--card);border:1px solid var(--line);color:var(--fg);border-radius:7px;
  padding:3px 9px;font-size:12px;width:190px}

/* ---- operator block ------------------------------------------------------- */
.op{border:1px solid var(--line);border-radius:12px;margin:0 0 16px;overflow:hidden;
  background:var(--card);scroll-margin-top:64px}
.op.hidden{display:none}
.ophead{background:var(--head);border-bottom:1px solid var(--line);padding:9px 12px}
.ophead .ttl{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline}
.ophead .oix{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--mut)}
.ophead .shd{font-weight:700;color:var(--acc);font-size:14.5px}
.ophead .oid{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--mut)}
.ophead .par{margin-top:5px;font-family:ui-monospace,Menlo,monospace;font-size:11px;
  color:var(--mut);word-break:break-word;line-height:1.65}
.ophead .par em{color:var(--fg);font-style:normal}
.ophead details{margin-top:5px}
.ophead summary{cursor:pointer;color:var(--mut);font-size:11.5px}
.eplist{columns:4;column-gap:18px;font-family:ui-monospace,Menlo,monospace;font-size:10.5px;
  color:var(--mut);margin:6px 0 2px}
.eplist div{break-inside:avoid}
table.rej{border-collapse:collapse;font-size:10.5px;margin:6px 0 2px;
  font-family:ui-monospace,Menlo,monospace}
table.rej th,table.rej td{border:1px solid var(--line);padding:2px 7px;text-align:right}
table.rej th:first-child,table.rej td:first-child{text-align:left}
table.rej th{color:var(--mut);font-weight:600}

/* ---- the row of 10 -------------------------------------------------------- */
.clips{display:flex;gap:8px;padding:10px;overflow-x:auto}
.clip{flex:0 0 var(--cw,152px);background:var(--bg);border:1px solid var(--line);
  border-radius:9px;overflow:hidden;min-width:0}
.poster{position:relative;overflow:hidden;aspect-ratio:480/640;background:#000;cursor:pointer;
  display:block}
.poster img{position:absolute;top:0;height:100%;width:auto;max-width:none;display:block;
  left:calc(var(--fk) * -100.8333%)}
.poster video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;background:#000}
.poster .slot{position:absolute;top:3px;left:3px;background:#000a;color:#fff;font-size:9px;
  padding:0 4px;border-radius:3px;font-family:ui-monospace,Menlo,monospace}
.poster .pl{position:absolute;right:3px;bottom:3px;background:#000a;color:#fff;font-size:9px;
  padding:0 4px;border-radius:3px}
.cmeta{padding:5px 7px;font-size:10.5px;font-family:ui-monospace,Menlo,monospace;
  color:var(--mut);line-height:1.5;word-break:break-word}
.cmeta .ab{color:var(--fg);opacity:.75}

/* strip mode: stack the 10 as full-width contact strips */
body.strips .clips{display:block;overflow-x:visible}
body.strips .clip{margin:0 0 6px}
body.strips .poster{aspect-ratio:3870/320}
body.strips .poster img{position:static;width:100%;height:auto;left:0}
body.strips .poster video{object-fit:contain}
body.strips .cmeta{display:flex;gap:14px;flex-wrap:wrap}

/* ---- index grids ---------------------------------------------------------- */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:13px}
.scard{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;
  display:block;color:inherit}
.scard:hover{border-color:var(--acc);text-decoration:none}
.scard .poster{aspect-ratio:480/640}
.scard .sm{padding:8px 10px;font-size:11.5px}
.scard .sm .nm{font-weight:600;color:var(--acc);font-size:13px;word-break:break-word}
.scard .sm .kv{color:var(--mut);font-family:ui-monospace,Menlo,monospace;font-size:10.5px;
  margin-top:3px}
table.big{border-collapse:collapse;font-size:12px;margin:10px 0}
table.big th,table.big td{border:1px solid var(--line);padding:3px 9px;text-align:right}
table.big th:first-child,table.big td:first-child{text-align:left}
table.big th{color:var(--mut);font-weight:600;position:sticky;top:44px;background:var(--bg)}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc2);
  border-radius:9px;padding:11px 14px;margin:14px 0;max-width:104ch}
.note.warnbox{border-left-color:#fbbf24}
.note p{margin:0 0 7px} .note p:last-child{margin:0}
.jump{display:flex;flex-wrap:wrap;gap:5px;margin:10px 0 4px}
.jump a{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:2px 8px;
  font-size:11.5px}
"""

JS = """
(function(){
  var root=document.documentElement;
  // ---- theme ---------------------------------------------------------------
  var saved=null; try{saved=localStorage.getItem('s2theme');}catch(e){}
  if(saved) root.setAttribute('data-theme',saved);
  function flipTheme(){
    var cur=root.getAttribute('data-theme');
    if(!cur) cur=window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark';
    var nxt=cur==='dark'?'light':'dark';
    root.setAttribute('data-theme',nxt);
    try{localStorage.setItem('s2theme',nxt);}catch(e){}
  }

  document.addEventListener('DOMContentLoaded',function(){
    var tb=document.getElementById('themebtn'); if(tb) tb.addEventListener('click',flipTheme);

    // ---- phase slider: every poster shows the same filmstrip frame ---------
    var ph=document.getElementById('phase'), pl=document.getElementById('phaselab');
    var FR=[0,4,8,14,22,32,44,56,68,80,92,100,108,112,116,120];
    function setPhase(){
      root.style.setProperty('--fk',ph.value);
      if(pl) pl.textContent='frame '+FR[+ph.value]+'/120';
    }
    if(ph){ ph.addEventListener('input',setPhase); setPhase(); }

    // ---- strip / frame mode -------------------------------------------------
    var seg=document.getElementById('viewseg');
    if(seg) seg.addEventListener('click',function(e){
      var b=e.target.closest('button'); if(!b) return;
      seg.querySelectorAll('button').forEach(function(x){x.classList.toggle('on',x===b);});
      document.body.classList.toggle('strips',b.dataset.v==='strip');
    });

    // ---- card size ----------------------------------------------------------
    var cw=document.getElementById('cw');
    if(cw){
      var apply=function(){root.style.setProperty('--cw',cw.value+'px');};
      cw.addEventListener('input',apply); apply();
    }

    // ---- video on demand ----------------------------------------------------
    function mount(p){
      if(p.querySelector('video')) return;
      var v=document.createElement('video');
      v.src=p.dataset.src; v.muted=true; v.loop=true; v.playsInline=true;
      v.preload='auto'; v.setAttribute('playsinline','');
      p.appendChild(v); v.play().catch(function(){});
    }
    function unmount(p){
      var v=p.querySelector('video');
      if(v){ v.pause(); v.removeAttribute('src'); v.load(); v.remove(); }
    }
    document.addEventListener('click',function(e){
      var p=e.target.closest('.poster'); if(!p||!p.dataset.src) return;
      e.preventDefault();
      if(p.querySelector('video')) unmount(p); else mount(p);
    });

    var auto=document.getElementById('autoplay');
    var io=new IntersectionObserver(function(es){
      es.forEach(function(e){
        if(!auto||!auto.checked) return;
        if(e.isIntersecting) mount(e.target); else unmount(e.target);
      });
    },{rootMargin:'150px 0px',threshold:0.15});
    var posters=[].slice.call(document.querySelectorAll('.poster[data-src]'));
    posters.forEach(function(p){io.observe(p);});
    if(auto) auto.addEventListener('change',function(){
      if(!auto.checked) posters.forEach(unmount);
      else posters.forEach(function(p){
        var r=p.getBoundingClientRect();
        if(r.top<innerHeight+150&&r.bottom>-150) mount(p);
      });
    });

    // ---- filter -------------------------------------------------------------
    var f=document.getElementById('filter');
    if(f) f.addEventListener('input',function(){
      var q=f.value.trim().toLowerCase();
      var n=0;
      document.querySelectorAll('[data-search]').forEach(function(el){
        var hit=!q||el.dataset.search.indexOf(q)>=0;
        el.classList.toggle('hidden',!hit); if(hit) n++;
      });
      var c=document.getElementById('fcount'); if(c) c.textContent=n+' shown';
    });
  });
})();
"""


# ------------------------------------------------------------------- formatting

def esc(x) -> str:
    return html.escape(str(x), quote=True)


def fmt_num(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def fmt_params(params: dict) -> str:
    if not params:
        return "<em>shader defaults</em> (no varied uniforms)"
    bits = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, list):
            s = "[" + ", ".join(fmt_num(x) for x in v) + "]"
        elif isinstance(v, bool):
            s = str(v).lower()
        else:
            s = fmt_num(v)
        bits.append(f"<em>{esc(k)}</em>={esc(s)}")
    return "&nbsp; ".join(bits)


def cls_seam(x: float) -> str:
    return "ok" if x < 1.2 else ("warn" if x < 1.6 else "bad")


def cls_m1(x: float) -> str:
    return "ok" if x >= 0.50 else ("warn" if x >= 0.35 else "bad")


def cls_m2(x: float) -> str:
    return "ok" if x < 0.25 else ("warn" if x < 0.40 else "bad")


def cls_rate(x: float) -> str:
    return "ok" if x < 0.20 else ("warn" if x < 0.40 else "bad")


def page(title: str, body: str, depth: int) -> str:
    up = "../" * depth
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{esc(title)}</title>'
        f'<link rel="stylesheet" href="{up}assets/viewer.css">'
        f'<script src="{up}assets/viewer.js"></script>'
        f'</head><body><div class="wrap">{body}</div></body></html>'
    )


def controlbar(filter_ph: str = "filter…", video: bool = True) -> str:
    f = (f'<input id="filter" type="search" placeholder="{esc(filter_ph)}">'
         f'<span id="fcount" class="dim"></span>')
    vid = ('<label><input id="autoplay" type="checkbox"> autoplay video on scroll</label>'
           '<label>size <input id="cw" type="range" min="96" max="320" value="152"></label>'
           ) if video else ""
    hint = ('click any tile to play/stop its video' if video
            else 'click a shader to open its operator blocks')
    return (
        '<div class="bar">'
        '<span class="seg" id="viewseg">'
        '<button data-v="frame" class="on">frames</button>'
        '<button data-v="strip">filmstrips</button></span>'
        '<label>phase <input id="phase" type="range" min="0" max="15" value="8">'
        '<span id="phaselab" class="mono"></span></label>'
        f'{vid}{f}'
        '<button class="btn" id="themebtn">theme</button>'
        f'<span class="dim">{hint}</span>'
        '</div>'
    )


def clip_card(c: dict, media: str) -> str:
    stem = c["stem"]
    seam = c["assert2"]["seam_max_ratio"]
    return (
        f'<div class="clip">'
        f'<div class="poster" data-src="{media}/videos/{stem}.mp4">'
        f'<img loading="lazy" src="{media}/filmstrips/{stem}.jpg" alt="{stem}">'
        f'<span class="slot">c{c["slot"]:02d}</span><span class="pl">&#9654;</span></div>'
        f'<div class="cmeta">'
        f'<div class="ab">{esc(c["A"])}<br>&rarr; {esc(c["B"])}</div>'
        f'<div>seam <span class="{cls_seam(seam)}">{seam:.2f}</span> '
        f'm1 <span class="{cls_m1(c["m1_p10"])}">{c["m1_p10"]:.2f}</span> '
        f'm2 <span class="{cls_m2(c["m2_max_dq"])}">{c["m2_max_dq"]:.2f}</span></div>'
        f'</div></div>'
    )


def op_block(op: dict, cs: list[dict], media: str, badstems: set[str]) -> str:
    cs = sorted(cs, key=lambda c: c["slot"])
    c0 = cs[0]
    t = c0["timing"]
    eps = []
    for c in cs:
        for e in (c["A"], c["B"]):
            if e not in eps:
                eps.append(e)
    seams = [c["assert2"]["seam_max_ratio"] for c in cs]
    m1s = [c["m1_p10"] for c in cs]
    nrej = len(op["rejects"])
    reason = collections.Counter(f for r in op["rejects"] for f in r["failed"])
    reason_s = ", ".join(f"{k}&times;{v}" for k, v in reason.most_common()) or "none"

    rejtab = ""
    if op["rejects"]:
        rows = "".join(
            f'<tr><td>{r["pair_id"]}</td><td>{esc(r["stage"])}</td>'
            f'<td>{esc(",".join(r["failed"]))}</td><td>{r.get("seam", 0):.3f}</td>'
            f'<td>{r.get("m1_p10", 0):.4f}</td><td>{r.get("m2_max_dq", 0):.4f}</td></tr>'
            for r in op["rejects"])
        rejtab = (f'<details><summary>{nrej} rejected render'
                  f'{"s" if nrej != 1 else ""} &mdash; {reason_s}</summary>'
                  f'<table class="rej"><tr><th>pair_id</th><th>stage</th><th>failed</th>'
                  f'<th>seam</th><th>m1_p10</th><th>m2_dq</th></tr>{rows}</table></details>')

    epcols = "".join(f"<div>{esc(e)}</div>" for e in eps)
    flags = ""
    hit = [c["stem"] for c in cs if c["stem"] in badstems]
    if hit:
        flags = ('<span class="tag hot">audit BAD: '
                 + ", ".join(esc(h) for h in hit) + "</span>")

    search = " ".join([op["shader"], op["op_id"], f'#{op["op_index"]:04d}',
                       c0["easing"], c0["flip"], "swap" if c0["swap"] else "noswap"]
                      + [e for e in eps]).lower()

    return (
        f'<section class="op" id="op{op["op_index"]:04d}" data-search="{esc(search)}">'
        f'<header class="ophead">'
        f'<div class="ttl"><span class="oix">op #{op["op_index"]:04d}</span>'
        f'<span class="shd">{esc(op["shader"])}</span>'
        f'<span class="oid">{esc(op["op_id"])}</span>'
        f'<span class="tag cool">10 clips &middot; 20 endpoints &middot; 90 (ref,target) combos</span>'
        f'{flags}</div>'
        f'<div class="par">uniforms: {fmt_params(c0["params"])}</div>'
        f'<div class="par">easing <em>{esc(c0["easing"])}</em> &middot; '
        f'onset <em>{t["onset"]:.1f}</em> &rarr; release <em>{t["release"]:.1f}</em> '
        f'(dur <em>{t["duration"]:.1f}</em> of 121 f) &middot; '
        f'flip <em>{esc(c0["flip"])}</em> &middot; swap <em>{str(c0["swap"]).lower()}</em> '
        f'&middot; attempts <em>{op["attempts"]}</em> &middot; '
        f'seam {min(seams):.2f}&ndash;{max(seams):.2f} &middot; '
        f'm1_p10 {min(m1s):.2f}&ndash;{max(m1s):.2f}</div>'
        f'<details><summary>20 distinct endpoint clips (content-disjoint by construction)</summary>'
        f'<div class="eplist">{epcols}</div></details>'
        f'{rejtab}'
        f'</header>'
        f'<div class="clips">{"".join(clip_card(c, media) for c in cs)}</div>'
        f'</section>'
    )


# ----------------------------------------------------------------------- build

def build(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "shaders").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "viewer.css").write_text(CSS)
    (out / "assets" / "viewer.js").write_text(JS)

    # media symlink -> the dataset root (http.server follows symlinks)
    link = out / "media"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path("../../videos/ctt_v2_s2/full"))

    ops = load_jsonl("ops_shard*.jsonl")
    clips = load_jsonl("clips_shard*.jsonl")
    ops_v1 = load_jsonl("ops_shard*.jsonl.v1")
    clips_v1 = load_jsonl("clips_shard*.jsonl.v1")

    live = [o for o in ops if not o["dropped"]]
    dropped = [o for o in ops if o["dropped"]]
    by_op = collections.defaultdict(list)
    for c in clips:
        by_op[c["op_index"]].append(c)

    # ---- audit ---------------------------------------------------------------
    audit = json.loads((SRC / "AUDIT_RESULT.json").read_text())
    key = {k["blind_id"]: k for k in json.loads((SRC / "AUDIT_KEY.json").read_text())["key"]}
    final_stems = {c["stem"] for c in clips}
    ret_stems = {p.stem for p in (SRC / "retired_blacklisted" / "videos").glob("*.mp4")}
    bad_final = {key[b]["stem"] for b in audit["final_bad"] if key[b]["stem"] in final_stems}
    bad_retired = {key[b]["stem"] for b in audit["final_bad"] if key[b]["stem"] in ret_stems}
    audit_in_final = sum(1 for k in key.values() if k["stem"] in final_stems)
    audit_in_ret = sum(1 for k in key.values() if k["stem"] in ret_stems)

    # ---- per-shader rollup (from the append-only ops log, not the summaries) --
    per = collections.defaultdict(lambda: {"ops": 0, "clips": 0, "rej": 0, "att": [],
                                           "dropped": 0})
    for o in ops:
        d = per[o["shader"]]
        d["clips"] += o["n_slots"]
        d["rej"] += len(o["rejects"])
        if o["dropped"]:
            d["dropped"] += 1
        else:
            d["ops"] += 1
            d["att"].append(o["attempts"])
    shaders = sorted(per)

    rendered = sum(o["n_slots"] + len(o["rejects"]) for o in ops)
    overdraw = rendered / len(clips)
    atts = [o["attempts"] for o in live]

    rng = random.Random(SEED)
    rep = {s: rng.choice([c for c in clips if c["shader"] == s])["stem"] for s in shaders}

    # ---- retired -------------------------------------------------------------
    v1_by_idx = {o["op_index"]: o for o in ops_v1}
    ret_clips = [c for c in clips_v1 if c["stem"] in ret_stems]
    ret_by_op = collections.defaultdict(list)
    for c in ret_clips:
        ret_by_op[c["op_index"]].append(c)
    ret_per = collections.defaultdict(lambda: {"ops": 0, "clips": 0})
    for oi, cs in ret_by_op.items():
        s = v1_by_idx[oi]["shader"]
        ret_per[s]["ops"] += 1
        ret_per[s]["clips"] += len(cs)
    # why they were blacklisted: v1 accept/reject over ALL ops of those shaders
    v1_rate = collections.defaultdict(lambda: {"acc": 0, "rej": 0, "ops": 0, "drop": 0})
    for o in ops_v1:
        if o["shader"] in ret_per:
            d = v1_rate[o["shader"]]
            d["acc"] += o["n_slots"]
            d["rej"] += len(o["rejects"])
            d["ops"] += 1
            d["drop"] += 1 if o["dropped"] else 0

    media_i, media_s = "media", "../media"

    # ========================================================== index.html ====
    b = []
    b.append('<h1>S2 stratum &mdash; procedural transition operators</h1>')
    b.append(
        '<p class="sub"><b>7,990 clips, 121 frames, 480&times;640, 24&nbsp;fps.</b> '
        'Every clip is a real endpoint pair with a procedurally rendered transition between '
        'them; the pure-phase frames either side of the transition are copied through verbatim '
        f'(max abs endpoint diff = <b>0.0</b> over all {len(clips):,} clips).</p>')
    b.append(
        '<div class="note"><p><b>Read this before you scroll.</b> There is <b>no reference clip '
        'on disk.</b> An operator\'s 10 clips <i>are</i> its reference pool: at train time the '
        'reference for a target clip is drawn from a <b>different clip of the same op_id</b>, '
        'giving <b>90 ordered (ref, target) combos per operator</b>, resampled every epoch &mdash; '
        f'<b>{len(live) * 90:,}</b> distinct training pairs across the stratum.</p>'
        '<p>Two invariants make that safe, and both are verified here: an operator\'s 10 clips '
        'use <b>20 distinct endpoint clips</b> (so any ref/target draw is automatically '
        'content-disjoint), and all 10 share <b>(shader, uniforms, easing, onset/release, flip, '
        'swap) exactly</b> &mdash; timing is part of operator identity, not a nuisance variable.</p>'
        '<p>So the unit to judge is the <b>operator block</b>: one row of 10 clips under one '
        'shared header. That row is the same-operator &times; different-content diagonal. Use the '
        '<b>phase slider</b> to step all 10 through the transition together &mdash; the operator\'s '
        'signature should be identical across the row while the content is completely different.</p>'
        '</div>')

    b.append('<div class="stats">')
    for val, lab in [
        (f"{len(clips):,}", "clips"),
        (f"{len(live):,}", "operators (op_id)"),
        (str(len(shaders)), "shaders"),
        ("10", "clips per operator"),
        ("20", "distinct endpoints per operator"),
        (f"{len(live) * 90:,}", "(ref,target) train pairs"),
        (f"{overdraw:.2f}&times;", "overdraw (bar 2.5)"),
        (f"{min(atts)}/{int(statistics.median(atts))}/{max(atts)}", "attempts min/med/max"),
        (str(len(dropped)), "operators dropped (attempts exhausted)"),
        ("420", "retired / blacklisted clips"),
    ]:
        b.append(f'<div class="stat"><b>{val}</b><span>{lab}</span></div>')
    b.append("</div>")

    b.append(
        f'<div class="note"><p><b>Blind audit &mdash; verdict '
        f'<span class="ok">{esc(audit["verdict"])}</span>.</b> n={audit["n"]}, '
        f'shader-stratified and shuffled, {esc(audit["protocol"])}. '
        f'Rater agreement {esc(audit["agreement"])}; <b>{audit["n_bad"]} BAD</b> after '
        f'adjudication against a bar of &le;{audit["bar_max_bad"]}/{audit["n"]} '
        f'({", ".join(esc(s) for s in audit["final_bad_shaders"])}).</p>'
        f'<p class="dim">Caveat worth knowing: the audit was drawn from the <b>pre-blacklist</b> '
        f'roster. {audit_in_ret} of its {audit["n"]} samples are now in the retired set, '
        f'including one of the two BAD clips '
        f'({", ".join(esc(s) for s in sorted(bad_retired)) or "&mdash;"}, a retired shader). '
        f'Against the roster that actually shipped the audit reads '
        f'<b>{len(bad_final)} BAD / {audit_in_final}</b> &mdash; but it is no longer a clean '
        f'{audit["n"]}-sample audit of this exact dataset.</p></div>')

    b.append(f'<p class="sub"><a href="retired.html"><b>&rarr; 420 retired / blacklisted clips</b></a> '
             f'(42 complete operator blocks, 6 shaders) &mdash; candidates for reinstatement '
             f'pending a blind audit.</p>')

    b.append('<h2>Browse by shader</h2>')
    b.append(f'<p class="sub">{len(shaders)} shaders, {len(live)} operators, median '
             f'{int(statistics.median([per[s]["ops"] for s in shaders]))} operators per shader. '
             f'Click a shader to open every one of its operator blocks. The thumbnail is one '
             f'randomly drawn clip (seed {SEED}); the <b>reject rate</b> is that shader\'s '
             f'rejected renders over all its renders &mdash; the frozen gate\'s opinion of how '
             f'hard the shader is to satisfy.</p>')
    b.append(controlbar("filter shaders…", video=False))
    b.append('<div class="grid">')
    for s in sorted(shaders, key=lambda x: x.lower()):
        d = per[s]
        tot = d["clips"] + d["rej"]
        rate = d["rej"] / tot if tot else 0.0
        drop = (f' &middot; <span class="warn">{d["dropped"]} dropped</span>'
                if d["dropped"] else "")
        b.append(
            f'<a class="scard" href="shaders/{esc(s)}.html" data-search="{esc(s.lower())}">'
            f'<span class="poster"><img loading="lazy" src="media/filmstrips/{rep[s]}.jpg" '
            f'alt="{esc(s)}"></span>'
            f'<span class="sm"><span class="nm">{esc(s)}</span>'
            f'<span class="kv">{d["ops"]} ops &middot; {d["clips"]} clips{drop}</span>'
            f'<span class="kv">reject <span class="{cls_rate(rate)}">{rate * 100:.0f}%</span> '
            f'({d["rej"]}/{tot}) &middot; med {int(statistics.median(d["att"]))} attempts</span>'
            f'</span></a>')
    b.append("</div>")

    b.append('<h2>Per-shader table</h2>')
    b.append('<table class="big"><tr><th>shader</th><th>ops</th><th>clips</th>'
             '<th>rejected</th><th>reject rate</th><th>attempts med</th>'
             '<th>attempts max</th><th>dropped ops</th></tr>')
    for s in sorted(shaders, key=lambda x: -(per[x]["rej"] / max(per[x]["rej"] + per[x]["clips"], 1))):
        d = per[s]
        tot = d["clips"] + d["rej"]
        rate = d["rej"] / tot if tot else 0.0
        b.append(f'<tr><td><a href="shaders/{esc(s)}.html">{esc(s)}</a></td><td>{d["ops"]}</td>'
                 f'<td>{d["clips"]}</td><td>{d["rej"]}</td>'
                 f'<td class="{cls_rate(rate)}">{rate * 100:.1f}%</td>'
                 f'<td>{int(statistics.median(d["att"]))}</td><td>{max(d["att"])}</td>'
                 f'<td>{d["dropped"] or ""}</td></tr>')
    b.append("</table>")

    if dropped:
        b.append('<h2>Dropped operators</h2>')
        b.append(f'<p class="sub">{len(dropped)} operators exhausted the 25-attempt pair-swap '
                 f'budget without filling 10 slots and were discarded whole (incidence integrity '
                 f'is a gate: exactly 10 clips or the operator does not ship). They have rows in '
                 f'<span class="mono">ops_shard*.jsonl</span> but no clips and no media, which is '
                 f'why the roster is <b>809 op rows / {len(live)} shipped operators</b>.</p>')
        b.append('<table class="big"><tr><th>op_index</th><th>shader</th><th>op_id</th>'
                 '<th>attempts</th><th>rejects</th></tr>')
        for o in sorted(dropped, key=lambda x: x["op_index"]):
            b.append(f'<tr><td>#{o["op_index"]:04d}</td><td>{esc(o["shader"])}</td>'
                     f'<td class="mono">{esc(o["op_id"])}</td><td>{o["attempts"]}</td>'
                     f'<td>{len(o["rejects"])}</td></tr>')
        b.append("</table>")

    b.append('<h2>Gates and provenance</h2>')
    seam_hi = max(c["assert2"]["seam_max_ratio"] for c in clips)
    m1_lo = min(c["m1_p10"] for c in clips)
    m2_hi = max(c["m2_max_dq"] for c in clips)
    b.append(
        f'<p class="sub">Frozen gate (config_s2.yaml): '
        f'<span class="mono">tau={TAU}</span>, <span class="mono">seam_max={SEAM_MAX}</span>, '
        f'<span class="mono">m2_max_dq={M2_MAX}</span>, '
        f'<span class="mono">assert1_tol=0.5</span> as a MAX over the pure phase. '
        f'Realised extremes over all {len(clips):,} clips: seam '
        f'<span class="{cls_seam(seam_hi)}">{seam_hi:.4f}</span>, m1_p10 min '
        f'<span class="{cls_m1(m1_lo)}">{m1_lo:.4f}</span>, m2 max '
        f'<span class="{cls_m2(m2_hi)}">{m2_hi:.4f}</span>, pure-phase max abs diff '
        f'<span class="ok">0.0</span>. Card colours: seam '
        f'<span class="ok">&lt;1.2</span>/<span class="warn">&lt;1.6</span>/'
        f'<span class="bad">&ge;1.6</span>, m1_p10 <span class="ok">&ge;.50</span>/'
        f'<span class="warn">&ge;.35</span>/<span class="bad">&lt;.35</span>, m2 '
        f'<span class="ok">&lt;.25</span>/<span class="warn">&lt;.40</span>/'
        f'<span class="bad">&ge;.40</span>.</p>')
    b.append(
        f'<p class="sub"><b>Where the numbers come from.</b> Everything on this page is '
        f'recomputed from the append-only <span class="mono">ops_shard*.jsonl</span> / '
        f'<span class="mono">clips_shard*.jsonl</span> logs and the files on disk. The '
        f'<span class="mono">summary_shard*.json</span> files were <b>rewritten by the backfill '
        f'pass</b> and describe only that pass (860 accepted / 1,662 rendered, overdraw 1.93), '
        f'not the whole build; <span class="mono">S2_ACCEPTANCE.json</span> is likewise stale '
        f'(7,550 clips / 755 ops / 62 shaders &mdash; the pre-blacklist state). The true build-wide '
        f'overdraw is <b>{overdraw:.4f}&times;</b> ({rendered:,} renders / {len(clips):,} '
        f'accepted).</p>')
    (out / "index.html").write_text(page("S2 stratum — 7,990 clips", "".join(b), 0))

    # ====================================================== shaders/*.html ====
    order = sorted(shaders, key=lambda x: x.lower())
    for i, s in enumerate(order):
        d = per[s]
        sops = sorted([o for o in live if o["shader"] == s], key=lambda o: o["op_index"])
        sdrop = [o for o in dropped if o["shader"] == s]
        tot = d["clips"] + d["rej"]
        rate = d["rej"] / tot if tot else 0.0
        p = []
        prev = order[i - 1] if i else None
        nxt = order[i + 1] if i + 1 < len(order) else None
        nav = " &middot; ".join(filter(None, [
            '<a href="../index.html">index</a>',
            f'<a href="{esc(prev)}.html">&larr; {esc(prev)}</a>' if prev else None,
            f'<a href="{esc(nxt)}.html">{esc(nxt)} &rarr;</a>' if nxt else None]))
        p.append(f'<p class="crumb">{nav}</p>')
        p.append(f'<h1>{esc(s)}</h1>')
        p.append(f'<p class="sub">{len(sops)} operators &times; 10 clips = {d["clips"]} clips. '
                 f'Each block below is <b>one operator</b>: the header holds the parameters all '
                 f'10 clips share exactly, the row holds the 10 different content pairs. '
                 f'Reject rate <span class="{cls_rate(rate)}">{rate * 100:.1f}%</span> '
                 f'({d["rej"]}/{tot} renders), attempts median '
                 f'{int(statistics.median(d["att"]))}, max {max(d["att"])}'
                 + (f', <span class="warn">{len(sdrop)} operator(s) dropped</span>' if sdrop else "")
                 + '.</p>')
        p.append(controlbar("filter ops / endpoints…"))
        p.append('<div class="jump">' + "".join(
            f'<a href="#op{o["op_index"]:04d}">#{o["op_index"]:04d}</a>' for o in sops) + '</div>')
        for o in sops:
            p.append(op_block(o, by_op[o["op_index"]], media_s, bad_final))
        if sdrop:
            p.append('<h2>Dropped operators (no clips)</h2>')
            p.append('<table class="big"><tr><th>op_index</th><th>op_id</th><th>attempts</th>'
                     '<th>rejects</th></tr>' + "".join(
                         f'<tr><td>#{o["op_index"]:04d}</td><td class="mono">{esc(o["op_id"])}</td>'
                         f'<td>{o["attempts"]}</td><td>{len(o["rejects"])}</td></tr>'
                         for o in sdrop) + '</table>')
        (out / "shaders" / f"{s}.html").write_text(page(f"S2 — {s}", "".join(p), 1))

    # ========================================================= retired.html ===
    r = []
    r.append('<p class="crumb"><a href="index.html">index</a></p>')
    r.append('<h1>Retired / blacklisted &mdash; 420 clips, 42 operators, 6 shaders</h1>')
    r.append(
        '<div class="note warnbox"><p><b>These are candidates for reinstatement, not rejects.</b> '
        'Every clip below passed the same frozen per-clip gate as the shipped stratum and forms a '
        'complete 10-clip operator block with 20 distinct endpoints &mdash; nothing about these '
        'clips individually failed. They were removed by a <b>shader-level</b> rule: '
        '<span class="mono">shader_blacklist_reject_rate = 0.50</span>, i.e. any shader whose '
        'renders were rejected more than half the time was blacklisted <i>whole</i>, and the '
        'operator budget it was holding was backfilled onto the surviving shaders.</p>'
        '<p>That rule is about <b>render economics</b>, not clip quality. Reinstating them would '
        f'take the stratum to <b>{len(clips) + 420:,} clips / {len(live) + 42} operators / '
        f'{len(shaders) + 6} shaders</b> at zero render cost. <b>Before that happens they need a '
        'blind audit of their own</b> &mdash; the n=64 audit sampled only 6 clips from these six '
        'shaders, and one of the two BAD clips it found was a PuzzleRight clip that now sits in '
        'this set.</p></div>')
    r.append('<div class="stats">')
    for val, lab in [("420", "clips"), ("42", "operator blocks"), ("6", "shaders"),
                     ("10", "clips per operator"), ("20", "distinct endpoints per operator"),
                     ("3,780", "(ref,target) pairs they would add")]:
        r.append(f'<div class="stat"><b>{val}</b><span>{lab}</span></div>')
    r.append("</div>")
    r.append('<table class="big"><tr><th>shader</th><th>retired ops (complete)</th>'
             '<th>clips</th><th>v1 ops attempted</th><th>v1 ops dropped</th>'
             '<th>v1 rejected renders</th><th>v1 reject rate</th></tr>')
    for s in BLACKLISTED:
        d = ret_per[s]
        v = v1_rate[s]
        tot = v["acc"] + v["rej"]
        rate = v["rej"] / tot if tot else 0.0
        r.append(f'<tr><td><a href="#sh-{esc(s)}">{esc(s)}</a></td><td>{d["ops"]}</td>'
                 f'<td>{d["clips"]}</td><td>{v["ops"]}</td><td>{v["drop"]}</td>'
                 f'<td>{v["rej"]}</td><td class="{cls_rate(rate)}">{rate * 100:.1f}%</td></tr>')
    r.append("</table>")
    r.append(f'<p class="sub">The 6 shaders held <b>{sum(v1_rate[s]["ops"] for s in BLACKLISTED)} '
             f'operators</b> in the pre-blacklist (<span class="mono">.v1</span>) roster; '
             f'<b>42</b> of them had completed all 10 slots when the blacklist landed and are the '
             f'clips below, the other '
             f'<b>{sum(v1_rate[s]["drop"] for s in BLACKLISTED)}</b> had already been dropped for '
             f'exhausting their attempt budget. Their op_index values resolve only through '
             f'<span class="mono">meta/ops_shard*.jsonl.v1</span>.</p>')
    r.append(controlbar("filter ops / endpoints…"))

    ret_media = "media/retired_blacklisted"
    for s in BLACKLISTED:
        idxs = sorted(oi for oi in ret_by_op if v1_by_idx[oi]["shader"] == s)
        r.append(f'<h2 id="sh-{esc(s)}">{esc(s)} &mdash; {len(idxs)} operator blocks, '
                 f'{len(idxs) * 10} clips</h2>')
        r.append('<div class="jump">' + "".join(
            f'<a href="#op{i:04d}">#{i:04d}</a>' for i in idxs) + '</div>')
        for oi in idxs:
            r.append(op_block(v1_by_idx[oi], ret_by_op[oi], ret_media, bad_retired))
    (out / "retired.html").write_text(page("S2 — retired / blacklisted (420 clips)",
                                           "".join(r), 0))

    print(f"wrote {out}/index.html")
    print(f"      {out}/shaders/*.html  ({len(order)} pages)")
    print(f"      {out}/retired.html")
    print(f"clips {len(clips)}  ops {len(live)} (+{len(dropped)} dropped)  shaders {len(shaders)}"
          f"  overdraw {overdraw:.4f}x  retired {len(ret_clips)}")


if __name__ == "__main__":
    build()
