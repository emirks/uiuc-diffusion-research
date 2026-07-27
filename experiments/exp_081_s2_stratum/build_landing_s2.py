#!/usr/bin/env python
"""Build a landing page for the S2 viewer: curated examples that give the feel of the dataset."""
import json, glob, random, pathlib, html, collections

VIEW = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/outputs/viewers/s2_dataset")
META = VIEW / "media" / "meta"
rng = random.Random(20260727)

ops = []
for f in sorted(glob.glob(str(META / "ops_shard*.jsonl"))):
    for l in open(f):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if r.get("dropped") or not r.get("complete"):
            continue
        ops.append(r)

by_shader = collections.defaultdict(list)
for r in ops:
    by_shader[r["shader"]].append(r)
shaders = sorted(by_shader)

# --- curation -------------------------------------------------------------
# hero: 4 visually distinct, well-known-good shaders
HERO_PREF = ["CrossZoom", "FilmBurn", "Dreamy", "ColourDistance",
             "ButterflyWaveScrawler", "GridFlip", "BookFlip", "PolkaDotsCurtain"]
hero = []
for s in HERO_PREF:
    if s in by_shader and len(hero) < 4:
        hero.append(rng.choice(by_shader[s]))

# showcase: ONE operator, all 10 clips -> the same-op x diff-content diagonal
show_pool = [r for r in ops if r["shader"] in
             ("CrossZoom", "Dreamy", "FilmBurn", "ColourDistance", "GridFlip")]
showcase = rng.choice(show_pool if show_pool else ops)

# variety: one clip from each of 28 distinct shaders
variety = []
for s in rng.sample(shaders, min(28, len(shaders))):
    op = rng.choice(by_shader[s])
    variety.append((s, op, rng.choice(op["stems"])))

TOTAL_CLIPS = sum(r["n_slots"] for r in ops)


def poster(stem, label="", sub="", big=False):
    cls = "poster big" if big else "poster"
    return (f'<div class="{cls}" data-src="media/videos/{stem}.mp4">'
            f'<img loading="lazy" src="media/filmstrips/{stem}.jpg" class="strip">'
            f'<div class="play">&#9654;</div>'
            + (f'<div class="cap"><b>{html.escape(label)}</b>'
               f'<span>{html.escape(sub)}</span></div>' if label else "")
            + '</div>')


hero_html = "".join(poster(rng.choice(r["stems"]), r["shader"],
                           f'op {r["op_index"]} · 10 clips', big=True) for r in hero)
show_html = "".join(poster(s, "", "") for s in showcase["stems"])
var_html = "".join(poster(stem, s, f'op {op["op_index"]}') for s, op, stem in variety)
shader_links = "".join(
    f'<a class="chip" href="shaders/{html.escape(s)}.html">{html.escape(s)}'
    f'<span>{len(by_shader[s])}</span></a>' for s in shaders)

DOC = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S2 stratum — {TOTAL_CLIPS:,} procedural transitions</title>
<link rel="stylesheet" href="assets/viewer.css">
<style>
:root{{--bg:#0d0e10;--fg:#f2f3f5;--mut:#9aa0a8;--line:#25272b;--card:#16181c;--acc:#5eead4}}
:root[data-theme=light]{{--bg:#fbfbfc;--fg:#16181c;--mut:#666d76;--line:#e4e6ea;--card:#fff;--acc:#0d9488}}
@media(prefers-color-scheme:light){{:root:not([data-theme=dark]){{--bg:#fbfbfc;--fg:#16181c;--mut:#666d76;--line:#e4e6ea;--card:#fff;--acc:#0d9488}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif;margin:0}}
.wrap{{max-width:1360px;margin:0 auto;padding:28px 22px 80px}}
h1{{font-size:30px;line-height:1.15;margin:0 0 8px;letter-spacing:-.02em}}
h2{{font-size:17px;margin:44px 0 4px;letter-spacing:-.01em}}
h2 .n{{color:var(--mut);font-weight:400;font-size:14px;margin-left:8px}}
.lede{{color:var(--mut);max-width:76ch;margin:0 0 6px;font-size:15px}}
.hint{{color:var(--mut);font-size:12.5px;margin:0 0 16px}}
.bar{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:18px 0 26px;
  position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:20;border-bottom:1px solid var(--line)}}
button,.lnk{{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:7px;
  padding:6px 12px;cursor:pointer;font:inherit;text-decoration:none;display:inline-block}}
button:hover,.lnk:hover{{border-color:var(--acc)}}
.lnk.primary{{border-color:var(--acc);color:var(--acc)}}
label.tog{{display:inline-flex;gap:7px;align-items:center;color:var(--mut);cursor:pointer;
  background:var(--card);border:1px solid var(--line);border-radius:7px;padding:6px 12px}}
.stats{{display:flex;gap:26px;flex-wrap:wrap;margin:20px 0 4px;padding:16px 18px;
  background:var(--card);border:1px solid var(--line);border-radius:10px}}
.stat b{{display:block;font-size:22px;letter-spacing:-.02em}}
.stat span{{color:var(--mut);font-size:12px}}
.hero{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin:8px 0 4px}}
.row{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(196px,1fr));gap:12px}}
.poster{{position:relative;background:#000;border:1px solid var(--line);border-radius:9px;
  overflow:hidden;cursor:pointer;aspect-ratio:3/4}}
.poster.big{{aspect-ratio:3/4}}
.poster img.strip{{position:absolute;inset:0;height:100%;width:1612.5%;max-width:none;
  object-fit:cover;left:calc(var(--fk,4) * -100.8333%)}}
.poster video{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;background:#000}}
.poster .play{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  color:#fff;font-size:26px;text-shadow:0 2px 12px #000;opacity:.55;pointer-events:none;transition:opacity .15s}}
.poster:hover .play{{opacity:.95}} .poster.on .play{{display:none}}
.cap{{position:absolute;left:0;right:0;bottom:0;padding:7px 9px;display:flex;flex-direction:column;
  background:linear-gradient(transparent,rgba(0,0,0,.82));pointer-events:none}}
.cap b{{font-size:12px;color:#fff}} .cap span{{font-size:10.5px;color:#c9ced4}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}}
.chip{{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:4px 11px;
  font-size:12px;color:var(--fg);text-decoration:none}}
.chip span{{color:var(--mut);margin-left:6px}} .chip:hover{{border-color:var(--acc)}}
.note{{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);
  border-radius:8px;padding:14px 16px;margin:14px 0;max-width:88ch}}
.note p{{margin:0 0 8px}} .note p:last-child{{margin:0}}
.slider{{display:flex;align-items:center;gap:9px;color:var(--mut);font-size:12.5px}}
input[type=range]{{width:190px;accent-color:var(--acc)}}
</style></head><body><div class="wrap">

<h1>S2 &mdash; procedural transition operators</h1>
<p class="lede">{TOTAL_CLIPS:,} clips. Each one is a pair of <b>real</b> video endpoints with a
procedurally rendered transition between them. The frames either side of the transition are copied
through verbatim, so endpoint fidelity is exact by construction &mdash; max abs diff <b>0.0</b>
across all {TOTAL_CLIPS:,}.</p>

<div class="bar">
  <label class="tog"><input type="checkbox" id="autoplay" checked> autoplay on scroll</label>
  <div class="slider">phase <input type="range" id="phase" min="0" max="15" value="4"></div>
  <button id="themebtn">theme</button>
  <a class="lnk primary" href="browse.html">browse all {len(shaders)} shaders &rarr;</a>
  <a class="lnk" href="retired.html">retired set (420)</a>
</div>

<div class="stats">
  <div class="stat"><b>{TOTAL_CLIPS:,}</b><span>clips</span></div>
  <div class="stat"><b>{len(ops)}</b><span>exact operators</span></div>
  <div class="stat"><b>{len(shaders)}</b><span>shaders</span></div>
  <div class="stat"><b>10</b><span>clips per operator</span></div>
  <div class="stat"><b>71,910</b><span>(ref, target) pairs</span></div>
  <div class="stat"><b>121f</b><span>480&times;640 &middot; 24fps</span></div>
</div>

<h2>A first look</h2>
<p class="hint">Hover or scroll to play. Click any tile to toggle it.</p>
<div class="hero">{hero_html}</div>

<h2>One operator, ten different scenes<span class="n">{html.escape(showcase['shader'])} &middot;
op {showcase['op_index']}</span></h2>
<div class="note"><p><b>This row is the point of the dataset.</b> All ten clips share
<i>(shader, uniforms, easing, onset/release, flip, swap)</i> exactly &mdash; one operator &mdash;
over twenty completely different endpoint clips. There is <b>no reference clip on disk</b>: at
train time the reference for any target is drawn from a <b>different clip of this same row</b>,
so the model can only succeed by reading the <i>manner</i>, never the content.</p>
<p>Drag the <b>phase</b> slider &mdash; every tile steps through the transition together. The
operator's signature should look identical across the row while the scenes stay unrelated.</p></div>
<div class="row">{show_html}</div>

<h2>The range of manners<span class="n">one clip from each of {len(variety)} shaders</span></h2>
<div class="grid">{var_html}</div>

<h2>All shaders<span class="n">{len(shaders)} &middot; number = operators</span></h2>
<div class="chips">{shader_links}</div>

</div>
<script>
(function(){{
  var root=document.documentElement;
  var saved=localStorage.getItem('s2theme'); if(saved) root.dataset.theme=saved;
  document.getElementById('themebtn').addEventListener('click',function(){{
    var d=root.dataset.theme==='dark'?'light':'dark'; root.dataset.theme=d;
    localStorage.setItem('s2theme',d);
  }});
  var ph=document.getElementById('phase');
  ph.addEventListener('input',function(){{root.style.setProperty('--fk',ph.value);}});
  root.style.setProperty('--fk',ph.value);

  function mount(p){{
    if(p.querySelector('video')) return;
    var v=document.createElement('video');
    v.src=p.dataset.src; v.muted=true; v.loop=true; v.playsInline=true; v.preload='auto';
    p.appendChild(v); p.classList.add('on'); v.play().catch(function(){{}});
  }}
  function unmount(p){{
    var v=p.querySelector('video');
    if(v){{v.pause(); v.removeAttribute('src'); v.load(); v.remove();}}
    p.classList.remove('on');
  }}
  document.addEventListener('click',function(e){{
    var p=e.target.closest('.poster'); if(!p) return;
    if(p.querySelector('video')) unmount(p); else mount(p);
  }});
  var auto=document.getElementById('autoplay');
  var posters=[].slice.call(document.querySelectorAll('.poster[data-src]'));
  var io=new IntersectionObserver(function(es){{
    es.forEach(function(e){{
      if(!auto.checked) return;
      if(e.isIntersecting) mount(e.target); else unmount(e.target);
    }});
  }},{{rootMargin:'200px 0px',threshold:.12}});
  posters.forEach(function(p){{io.observe(p);}});
  auto.addEventListener('change',function(){{
    if(!auto.checked) posters.forEach(unmount);
    else posters.forEach(function(p){{
      var r=p.getBoundingClientRect();
      if(r.top<innerHeight+200&&r.bottom>-200) mount(p);
    }});
  }});
}})();
</script></body></html>"""

(VIEW / "index.html").write_text(DOC)
print(f"landing page written: {len(DOC):,} bytes")
print(f"  hero {len(hero)} · showcase {len(showcase['stems'])} ({showcase['shader']} op {showcase['op_index']}) · variety {len(variety)}")
print(f"  totals: {TOTAL_CLIPS} clips / {len(ops)} ops / {len(shaders)} shaders")
