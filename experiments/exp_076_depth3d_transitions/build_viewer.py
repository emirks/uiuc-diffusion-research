"""Build an HTML viewer for an exp_076 run.

Usage:  python build_viewer.py <run_dir>
Serve from the REPO ROOT so relative media paths resolve:
    cd $LAB/diffusion-research && python -m http.server 8077
"""

from __future__ import annotations

import html
import json
import pathlib
import sys
from collections import OrderedDict

SECTIONS = [
    ("family", "Camera families — one operator per family, same endpoint pair",
     "Seven trajectory types, matched amplitude and easing, so the families can be "
     "compared directly. All parallax comes from real per-pixel depth, so foreground "
     "and background separate correctly under camera motion."),
    ("counterfactual", "Counterfactual block — one endpoint pair, many operators",
     "Content fixed, operator varies. This is the signal the 49 real clips cannot "
     "provide: the same endpoints admit many valid transitions."),
    ("sharedop", "Shared-operator block — one operator, several endpoint pairs",
     "The mirror image: operator fixed, content varies."),
    ("diverse", "Diversity sample — random operators × random endpoint pairs",
     "Every pair takes its two endpoints from <b>different</b> source clips."),
]

CSS = """
:root{--bg:#0f1115;--fg:#e6e8ec;--mut:#9aa3b2;--card:#171a21;--line:#252a34;--acc:#7cc0ff}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--fg:#12141a;--mut:#5d6675;--card:#fff;--line:#e2e5ea;--acc:#1668c9}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:44px 0 4px;padding-top:20px;border-top:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 18px;max-width:78ch}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 8px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.stat b{display:block;font-size:20px} .stat span{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card video{width:100%;display:block;background:#000;aspect-ratio:480/640;object-fit:cover}
.meta{padding:9px 11px;font-size:11.5px}
.meta .sh{font-weight:600;color:var(--acc);font-size:13px}
.meta .kv{color:var(--mut);margin-top:3px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;word-break:break-word}
.tag{display:inline-block;background:var(--line);border-radius:4px;padding:1px 5px;margin-right:4px;font-size:10px}
.strip{width:100%;display:block;border-top:1px solid var(--line)}
.ok{color:#4ade80} .warn{color:#fbbf24} .bad{color:#f87171}
.toggle{margin:10px 0 16px;color:var(--mut);font-size:12.5px}
.toggle label{margin-right:16px;cursor:pointer}
"""

JS = """
document.addEventListener('DOMContentLoaded',()=>{
  const strips=document.getElementById('showstrips');
  const sync=()=>document.querySelectorAll('.strip').forEach(s=>s.style.display=strips.checked?'block':'none');
  strips.addEventListener('change',sync); sync();
  const play=document.getElementById('autoplay');
  const io=new IntersectionObserver(es=>es.forEach(e=>{
    if(e.isIntersecting&&play.checked)e.target.play().catch(()=>{});else e.target.pause();}),{threshold:.25});
  document.querySelectorAll('video').forEach(v=>io.observe(v));
  play.addEventListener('change',()=>document.querySelectorAll('video').forEach(v=>{
    if(play.checked)v.play().catch(()=>{});else v.pause();}));
});
"""


def card(m: dict, rel: str = ".") -> str:
    r = max(m["seam_ratio_in"], m["seam_ratio_out"])
    cls = "ok" if r <= 2.0 else ("warn" if r <= 4.0 else "bad")
    p = m["params"]
    tags = [f'<span class="tag">{html.escape(m["family"])}</span>',
            f'<span class="tag">{html.escape(m["blend"])}</span>']
    if p.get("fog", 0):
        tags.append('<span class="tag">fog</span>')
    if p.get("focus", 0):
        tags.append('<span class="tag">rack focus</span>')
    if p.get("handheld", 0):
        tags.append('<span class="tag">handheld</span>')
    return f"""<div class="card">
<video src="{rel}/videos/{m['stem']}.mp4" muted loop playsinline preload="metadata"></video>
<img class="strip" src="{rel}/filmstrips/{m['stem']}.jpg" loading="lazy">
<div class="meta">
  <div class="sh">{html.escape(m['family'])}</div>
  <div>{''.join(tags)}</div>
  <div class="kv">{html.escape(m['describe'])}</div>
  <div class="kv">{html.escape(m['pair_id'])}</div>
  <div class="kv">seam ratio <span class="{cls}">{r:.2f}</span></div>
</div></div>"""


def main() -> None:
    run_dir = pathlib.Path(sys.argv[1]).resolve()
    man = json.load(open(run_dir / "manifest.json"))
    ratios = [max(m["seam_ratio_in"], m["seam_ratio_out"]) for m in man]
    ratios.sort()
    med = ratios[len(ratios) // 2]
    fams = len({m["family"] for m in man})
    mean_s = sum(m["render_s"] for m in man) / len(man)

    parts = [f"""<title>exp_076 — 3D-plausible procedural transitions</title>
<style>{CSS}</style><script>{JS}</script>
<div class="wrap">
<h1>3D-plausible procedural transitions</h1>
<p class="sub">Each clip is <b>33 frames</b>: the given <code>start9</code> bucket, a
rendered middle, and the given <code>end9</code> bucket — the two buckets are copied
through <b>verbatim</b>. The middle is a 2.5D render: each endpoint frame is unprojected
by its Depth&nbsp;Anything&nbsp;V2 depth map into a displaced mesh, and a single continuous
virtual camera flies out of scene A and comes to rest in scene B. Parallax is real, so
foreground and background separate correctly; fog is Beer–Lambert extinction along the
view ray; defocus is circle-of-confusion about a racking focus plane.</p>
<div class="stats">
<div class="stat"><b>{len(man)}</b><span>clips</span></div>
<div class="stat"><b>{fams}</b><span>camera families</span></div>
<div class="stat"><b>{med:.2f}</b><span>median seam ratio (1.0 = as smooth as the content's own motion)</span></div>
<div class="stat"><b>{mean_s:.1f}s</b><span>mean render per clip (CPU)</span></div>
</div>
<div class="toggle">
<label><input type="checkbox" id="autoplay" checked> autoplay on scroll</label>
<label><input type="checkbox" id="showstrips"> show filmstrips</label>
</div>"""]

    for prefix, title, blurb in SECTIONS:
        rows = [m for m in man if m["tag"].startswith(prefix)]
        if not rows:
            continue
        parts.append(f'<h2>{html.escape(title)}</h2><p class="sub">{blurb}</p>')
        if prefix == "sharedop":
            groups = OrderedDict()
            for m in sorted(rows, key=lambda m: (m["tag"], m["pair_id"])):
                groups.setdefault(m["tag"], []).append(m)
            for ms in groups.values():
                parts.append(f'<p class="sub"><b>{html.escape(ms[0]["describe"])}</b></p>')
                parts.append('<div class="grid">' + "".join(card(m) for m in ms) + "</div>")
        else:
            parts.append('<div class="grid">' + "".join(card(m) for m in rows) + "</div>")

    parts.append("</div>")
    out = run_dir / "viewer.html"
    out.write_text("\n".join(parts))
    print(f"[viewer] {out}")


if __name__ == "__main__":
    main()
