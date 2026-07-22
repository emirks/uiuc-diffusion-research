"""Build a self-contained HTML viewer for an exp_075 run.

Usage:  python build_viewer.py <run_dir>
Serve from the repo root so the relative media paths resolve:
    python -m http.server 8000        # then open /outputs/videos/.../viewer.html
"""

from __future__ import annotations

import html
import json
import pathlib
import sys
from collections import OrderedDict

SECTIONS = [
    ("counterfactual", "Counterfactual block — one endpoint pair, many operators",
     "Content is held fixed; only the operator changes. This is the training signal "
     "the 49 real clips cannot provide: the model sees that the same endpoints admit "
     "many valid transitions, so it must read the operator, not memorise the content."),
    ("sharedop", "Shared-operator block — one operator, many endpoint pairs",
     "The mirror image: the operator is held fixed and the content changes. Together "
     "with the block above this is the operator ⊥ content factorisation."),
    ("diverse", "Diversity sample — random operators × random endpoint pairs",
     "What the bulk of a pretraining corpus would look like. Pairs marked "
     "<b>cross</b> take their two endpoints from different source clips."),
    ("ext_", "Layer-extension ablation — same pair, same operator, 3 policies",
     "How the 9 given frames are extended underneath the effect. <b>hold</b> freezes "
     "them (teaches 'motion stops during a transition'); <b>boomerang</b> ping-pongs "
     "them; <b>flow</b> extrapolates the terminal optical flow."),
    ("real", "Reference — real human-made transitions",
     "The ground truth from the 49-clip corpus, for calibration."),
]

CSS = """
:root{--bg:#0f1115;--fg:#e6e8ec;--mut:#9aa3b2;--card:#171a21;--line:#252a34;--acc:#7cc0ff}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--fg:#12141a;--mut:#5d6675;--card:#fff;--line:#e2e5ea;--acc:#1668c9}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:44px 0 4px;padding-top:20px;border-top:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 18px;max-width:76ch}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 8px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.stat b{display:block;font-size:20px} .stat span{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card video{width:100%;display:block;background:#000;aspect-ratio:480/640;object-fit:cover}
.meta{padding:9px 11px;font-size:11.5px}
.meta .sh{font-weight:600;color:var(--acc);font-size:13px;word-break:break-word}
.meta .kv{color:var(--mut);margin-top:3px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;word-break:break-word}
.tag{display:inline-block;background:var(--line);border-radius:4px;padding:1px 5px;margin-right:4px;font-size:10px;color:var(--fg)}
.strip{width:100%;display:block;border-top:1px solid var(--line)}
.ok{color:#4ade80} .bad{color:#f87171}
.toggle{margin:10px 0 16px;color:var(--mut);font-size:12.5px}
.toggle label{margin-right:16px;cursor:pointer}
"""

JS = """
document.addEventListener('DOMContentLoaded',()=>{
  const strips=document.getElementById('showstrips');
  const sync=()=>document.querySelectorAll('.strip').forEach(s=>s.style.display=strips.checked?'block':'none');
  strips.addEventListener('change',sync); sync();
  const play=document.getElementById('autoplay');
  play.addEventListener('change',()=>document.querySelectorAll('video').forEach(v=>{
    if(play.checked){v.setAttribute('autoplay','');v.play().catch(()=>{});}else{v.pause();v.removeAttribute('autoplay');}
  }));
  // only start videos once they scroll into view
  const io=new IntersectionObserver(es=>es.forEach(e=>{
    if(e.isIntersecting&&play.checked)e.target.play().catch(()=>{});else e.target.pause();}),{threshold:.25});
  document.querySelectorAll('video').forEach(v=>io.observe(v));
});
"""


def card(m: dict, rel: str) -> str:
    ep = max(m["endpoint_mae_start"], m["endpoint_mae_end"])
    cls = "ok" if ep < 0.5 else ("ok" if ep < 2.0 else "bad")
    tags = [f'<span class="tag">{html.escape(m["pair_kind"])}</span>']
    if m.get("swap"):
        tags.append('<span class="tag">reversed</span>')
    if m.get("flip") and m["flip"] != "none":
        tags.append(f'<span class="tag">flip {m["flip"]}</span>')
    if m.get("aux_kind"):
        tags.append(f'<span class="tag">map {m["aux_kind"]}</span>')
    params = " ".join(f"{k}={v}" for k, v in sorted((m.get("params") or {}).items()))
    return f"""<div class="card">
<video src="{rel}/videos/{m['stem']}.mp4" muted loop playsinline preload="metadata"></video>
<img class="strip" src="{rel}/filmstrips/{m['stem']}.jpg" loading="lazy">
<div class="meta">
  <div class="sh">{html.escape(m['shader'])}</div>
  <div>{''.join(tags)}</div>
  <div class="kv">pair {html.escape(m['pair_id'])}</div>
  <div class="kv">ease {html.escape(str(m['easing']))} · ext {html.escape(str(m['extension']))}</div>
  {f'<div class="kv">{html.escape(params)}</div>' if params else ''}
  <div class="kv">endpoint MAE <span class="{cls}">{ep:.3f}</span></div>
</div></div>"""


def main() -> None:
    run_dir = pathlib.Path(sys.argv[1]).resolve()
    manifest = json.load(open(run_dir / "manifest.json"))
    validation = json.load(open(run_dir / "bank_validation.json"))
    rel = "."

    n_ok = sum(1 for v in validation if v["status"] == "ok")
    gen = [m for m in manifest if m["tag"] != "real"]
    worst = max(max(m["endpoint_mae_start"], m["endpoint_mae_end"]) for m in gen)
    shaders_used = len({m["shader"] for m in gen})
    mean_s = sum(m["render_s"] for m in gen) / len(gen)

    parts = [f"""<title>exp_075 — procedural transition operator engine</title>
<style>{CSS}</style><script>{JS}</script>
<div class="wrap">
<h1>Procedural transition operator engine</h1>
<p class="sub">GL-Transitions shader bank applied to the 9-frame endpoint clips
(<code>*_start9.mp4</code> / <code>*_end9.mp4</code>). Every clip is 121 frames at
480×640/24fps; frames 0–8 and 112–120 reproduce the given endpoints exactly, so each
output is a drop-in training sample for the transition task.</p>
<div class="stats">
<div class="stat"><b>{n_ok}/{len(validation)}</b><span>shaders passing the endpoint gate</span></div>
<div class="stat"><b>{shaders_used}</b><span>distinct shaders sampled here</span></div>
<div class="stat"><b>{len(gen)}</b><span>procedural clips rendered</span></div>
<div class="stat"><b>{worst:.3f}</b><span>worst endpoint MAE (0–255)</span></div>
<div class="stat"><b>{mean_s:.1f}s</b><span>mean render time per 121-frame clip (CPU)</span></div>
</div>
<div class="toggle">
<label><input type="checkbox" id="autoplay" checked> autoplay on scroll</label>
<label><input type="checkbox" id="showstrips"> show filmstrips</label>
</div>"""]

    for prefix, title, blurb in SECTIONS:
        rows = [m for m in manifest if m["tag"].startswith(prefix)]
        if not rows:
            continue
        if prefix == "sharedop":
            rows.sort(key=lambda m: (m["tag"], m["pair_id"]))
        parts.append(f'<h2>{html.escape(title)}</h2><p class="sub">{blurb}</p>')
        if prefix == "sharedop":
            groups = OrderedDict()
            for m in rows:
                groups.setdefault(m["tag"], []).append(m)
            for tag, ms in groups.items():
                parts.append(f'<p class="sub"><b>{html.escape(ms[0]["describe"])}</b></p>')
                parts.append('<div class="grid">' + "".join(card(m, rel) for m in ms) + "</div>")
        else:
            parts.append('<div class="grid">' + "".join(card(m, rel) for m in rows) + "</div>")

    bad = [v for v in validation if v["status"] != "ok"]
    parts.append('<h2>Shader bank validation</h2><p class="sub">Every shader is checked '
                 'for the identities <code>transition(uv, p=0) == from</code> and '
                 '<code>transition(uv, p=1) == to</code>. A shader that fails would '
                 'silently corrupt the conditioning frames of every sample it '
                 'generates, so it is dropped from the bank.</p>')
    parts.append('<div class="kv" style="color:var(--mut);font-family:ui-monospace">'
                 + "<br>".join(
                     f'{html.escape(v["shader"])} — {v["status"]} '
                     f'(MAE p0 {v.get("mae_p0")}, p1 {v.get("mae_p1")})' for v in bad)
                 + "</div>")
    parts.append("</div>")

    out = run_dir / "viewer.html"
    out.write_text("\n".join(parts))
    print(f"[viewer] {out}")


if __name__ == "__main__":
    main()
