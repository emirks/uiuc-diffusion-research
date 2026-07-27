"""Build outputs/viewers/luma_matte/ — the paired A/B luma-matte viewer.

The whole point of the page is one judgement the owner has to make with their
eyes: the aux-map family was killed for looking fake, but the maps and the
compositor were never varied independently. So every card here is a *pair* —
same matte, same seed, same footage, same easing, two compositors side by side —
and the matte itself is shown next to its own result.

Usage:  python build_viewer.py [run_dir]
"""

from __future__ import annotations

import html
import json
import pathlib
import shutil
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
VIEW_DIR = REPO_ROOT / "outputs" / "viewers" / "luma_matte"

ARM_PAIRS = [
    ("shipped", "A1_current", "A2_soft_same_map",
     "Shipped maps — <b>hard <code>step()</code></b> vs <b>feathered</b>",
     "The seven maps already in <code>exp_075/engine/maps.py</code>, unchanged, at the "
     "same seed. Left column is literally what was judged and set to 0%: "
     "<code>luma.glsl</code>'s <code>step(progress, m)</code>. Right column is the same "
     "field through <code>luma_soft.glsl</code> — smoothstep feather, a rim colour "
     "painted into the advancing band, an additive glow lobe ahead of it. "
     "<b>If the compositor was the problem, the right column rescues the left.</b>"),
    ("new", "A4_new_map_hard", "A3_new_map_soft",
     "New arrival-time maps — <b>hard <code>step()</code></b> vs <b>feathered</b>",
     "Eikonal fronts, invasion percolation and CC0-brush stamping: non-stationary, "
     "anisotropic, source-anchored fields that fbm cannot produce. Same two "
     "compositors, same pairing. The left column isolates how much of the new "
     "maps' quality survives the old compositor."),
]

ARM_LABEL = {
    "A1_current": "A1 · hard step()",
    "A2_soft_same_map": "A2 · feathered",
    "A3_new_map_soft": "A3 · feathered",
    "A4_new_map_hard": "A4 · hard step()",
}

CSS = """
:root{--bg:#0f1115;--fg:#e6e8ec;--mut:#9aa3b2;--card:#171a21;--line:#252a34;
      --acc:#7cc0ff;--acc2:#c4a3ff;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--fg:#12141a;--mut:#5d6675;
      --card:#fff;--line:#e2e5ea;--acc:#1668c9;--acc2:#6d3fc4}}
:root[data-theme=dark]{--bg:#0f1115;--fg:#e6e8ec;--mut:#9aa3b2;--card:#171a21;
      --line:#252a34;--acc:#7cc0ff;--acc2:#c4a3ff}
:root[data-theme=light]{--bg:#f7f8fa;--fg:#12141a;--mut:#5d6675;--card:#fff;
      --line:#e2e5ea;--acc:#1668c9;--acc2:#6d3fc4}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1560px;margin:0 auto;padding:30px 22px 80px}
h1{font-size:25px;margin:0 0 6px}
h2{font-size:19px;margin:40px 0 4px;padding-top:18px;border-top:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 14px;max-width:96ch}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em;
     background:var(--line);border-radius:4px;padding:1px 4px}
.stats{display:flex;flex-wrap:wrap;gap:9px;margin:16px 0 6px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:9px 13px}
.stat b{display:block;font-size:19px} .stat span{color:var(--mut);font-size:11.5px}
.bar{position:sticky;top:0;z-index:9;background:var(--bg);border-bottom:1px solid var(--line);
     display:flex;gap:7px;flex-wrap:wrap;align-items:center;padding:9px 0;margin:14px 0 8px}
button,select{background:var(--card);color:var(--fg);border:1px solid var(--line);
     border-radius:6px;padding:5px 10px;cursor:pointer;font:inherit;font-size:12.5px}
button.on{border-color:var(--acc);color:var(--acc)}
.bar .sep{width:1px;height:20px;background:var(--line);margin:0 4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:13px}
body.big .grid{grid-template-columns:repeat(auto-fill,minmax(760px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;overflow:hidden}
.row{display:grid;grid-template-columns:.52fr 1fr 1fr;gap:2px;background:var(--line)}
.cell{position:relative;background:#000}
.cell video,.cell img.mapimg{width:100%;display:block;background:#000;
     aspect-ratio:480/640;object-fit:cover}
.cell .tag{position:absolute;left:0;top:0;font-size:9.5px;letter-spacing:.02em;
     padding:2px 5px;background:rgba(0,0,0,.66);color:#fff;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.cell.hardcol .tag{background:rgba(200,60,60,.82)}
.cell.softcol .tag{background:rgba(50,150,90,.85)}
.cell.mapcol .tag{background:rgba(80,80,110,.85)}
.strips{display:none;border-top:1px solid var(--line)}
body.strips .strips{display:block}
body.strips .row{display:none}
.strips .sline{display:flex;align-items:center;gap:6px;padding:3px 6px}
.strips .sline b{font-size:9.5px;width:74px;flex:none;color:var(--mut);
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.strips img{width:100%;display:block;min-width:0}
.meta{padding:8px 10px;font-size:11.5px}
.meta .nm{font-weight:600;color:var(--acc);font-size:12.5px}
.meta .kv{color:var(--mut);margin-top:2px;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;
     word-break:break-word}
.chip{display:inline-block;border-radius:4px;padding:1px 5px;margin-right:4px;font-size:10px}
.chip.ink{background:#2b2f45;color:#cfd6ff} .chip.paint{background:#5a1f18;color:#ffd8cf}
.chip.leak{background:#5a4310;color:#ffe6ad} .chip.frost{background:#12384a;color:#c8ecff}
.chip.burn{background:#5c2a06;color:#ffcd9b}
.ok{color:var(--ok)} .bad{color:var(--bad)}
table{border-collapse:collapse;font-size:12px;margin:10px 0 4px}
th,td{border:1px solid var(--line);padding:4px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc2);
      border-radius:8px;padding:11px 14px;margin:14px 0;font-size:13px;max-width:96ch}
"""

JS = """
document.addEventListener('DOMContentLoaded',()=>{
  const io=new IntersectionObserver(es=>es.forEach(e=>{
    const v=e.target; if(!window.__play){v.pause();return;}
    if(e.isIntersecting){v.preload='auto';v.play().catch(()=>{});}else v.pause();
  }),{rootMargin:'140px',threshold:.15});
  window.__play=true;
  document.querySelectorAll('video').forEach(v=>io.observe(v));

  window.tgl=(b,cls)=>{document.body.classList.toggle(cls);b.classList.toggle('on');};
  window.theme=()=>{const r=document.documentElement;
    r.dataset.theme=r.dataset.theme==='dark'?'light':'dark';};
  window.play=(b)=>{window.__play=!window.__play;b.classList.toggle('on');
    document.querySelectorAll('video').forEach(v=>window.__play?v.play().catch(()=>{}):v.pause());};
  window.filt=()=>{
    const p=document.getElementById('fpair').value, m=document.getElementById('fmap').value;
    document.querySelectorAll('.card').forEach(c=>{
      c.style.display=((p==='*'||c.dataset.pair===p)&&(m==='*'||c.dataset.map===m))?'':'none';});
  };
});
"""


def esc(s) -> str:
    return html.escape(str(s))


def build(run_dir: pathlib.Path) -> pathlib.Path:
    man = json.load(open(run_dir / "manifest.json"))
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("videos", "filmstrips", "maps"):
        dst = VIEW_DIR / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(run_dir / sub, dst)

    by_key = {(m["arm"], m["map"], m["pair_id"]): m for m in man}
    pairs = sorted({m["pair_id"] for m in man})
    maps_all = sorted({m["map"] for m in man})
    n_bad_hard = [m["endpoint_bad_px"] for m in man if m["compositor_key"] == "hard"]
    n_bad_soft = [m["endpoint_bad_px"] for m in man if m["compositor_key"] == "soft"]

    out: list[str] = []
    out.append("<title>luma-matte transitions — was it the map, or was it step()?</title>")
    out.append(f"<style>{CSS}</style><script>{JS}</script>")
    out.append('<div class="wrap">')
    out.append("<h1>Luma-matte transitions — was it the map, or was it <code>step()</code>?</h1>")
    out.append(
        '<p class="sub">A luma matte is a static greyscale field; the compositor sweeps a '
        'threshold across it, so the field <i>is</i> an arrival-time map '
        '<code>T(x)</code> and an animated matte needs no video. This family '
        '(<code>MAP_KINDS = fbm, radial, linear, stripes, checker, spiral, voronoi</code> '
        'through <code>luma.glsl</code>) was judged fake-looking and shipped at <b>0%</b>. '
        'But the shipped compositor is one line — '
        '<code>mix(getToColor(uv), getFromColor(uv), step(progress, texture2D(luma,uv).r))</code> '
        ' — a hard binary threshold with no feather, no rim colour and no glow, while ink, '
        'paint and light leaks read as real almost entirely <i>because of the boundary</i>. '
        'Map quality and compositor quality were therefore never separated. '
        'This page is the 2×2 that separates them.</p>')
    out.append(
        '<div class="note"><b>How to read a card.</b> Left tile is the raw greyscale matte. '
        'Middle tile is that matte through the <b>hard <code>step()</code></b>. Right tile is '
        'the <b>same matte, same seed, same footage, same easing</b> through '
        '<code>luma_soft.glsl</code>. Nothing else differs between the two videos in a card, '
        'so any difference you see is the compositor and only the compositor. '
        'Both layers are real frames from the curated 227-clip endpoint bank playing '
        'forward — the outgoing shot plays its last 61 frames, the incoming shot its first '
        '61 — and the 6-frame anchor blocks are verbatim.</div>')

    out.append('<div class="stats">')
    for b, s in [(len(man), "clips rendered"),
                 (len(pairs), "content pairs"),
                 ("7", "shipped maps (exp_075)"),
                 ("12", "new arrival-time maps"),
                 ("2", "compositors"),
                 (f"{max(m['endpoint_mae'] for m in man):.3f}", "worst endpoint MAE"),
                 (f"{max(n_bad_hard)}", "endpoint bad px · step()"),
                 (f"{max(n_bad_soft)}", "endpoint bad px · soft")]:
        out.append(f'<div class="stat"><b>{esc(b)}</b><span>{s}</span></div>')
    out.append("</div>")

    out.append('<div class="bar">')
    out.append('<button class="on" onclick="play(this)">autoplay</button>')
    out.append('<button onclick="tgl(this,\'strips\')">filmstrips</button>')
    out.append('<button onclick="tgl(this,\'big\')">big</button>')
    out.append('<button onclick="theme()">theme</button>')
    out.append('<span class="sep"></span>')
    out.append('<select id="fpair" onchange="filt()"><option value="*">all content pairs</option>')
    for p in pairs:
        out.append(f'<option value="{esc(p)}">{esc(p)}</option>')
    out.append("</select>")
    out.append('<select id="fmap" onchange="filt()"><option value="*">all maps</option>')
    for m in maps_all:
        out.append(f'<option value="{esc(m)}">{esc(m)}</option>')
    out.append("</select></div>")

    for fam, arm_hard, arm_soft, title, blurb in ARM_PAIRS:
        fam_maps = sorted({m["map"] for m in man if m["map_family"] == fam})
        out.append(f"<h2>{title}</h2>")
        out.append(f'<p class="sub">{blurb}</p>')
        out.append('<div class="grid">')
        for mp in fam_maps:
            for p in pairs:
                h = by_key.get((arm_hard, mp, p))
                s = by_key.get((arm_soft, mp, p))
                if not h or not s:
                    continue
                out.append(f'<div class="card" data-pair="{esc(p)}" data-map="{esc(mp)}">')
                out.append('<div class="row">')
                out.append(f'<div class="cell mapcol"><img class="mapimg" loading="lazy" '
                           f'src="maps/{esc(s["map_png"])}">'
                           f'<span class="tag">matte T(x)</span></div>')
                for cell, cls, rec in ((arm_hard, "hardcol", h), (arm_soft, "softcol", s)):
                    out.append(
                        f'<div class="cell {cls}"><video src="videos/{esc(rec["stem"])}.mp4" '
                        f'muted loop playsinline preload="none"></video>'
                        f'<span class="tag">{esc(ARM_LABEL[cell])}</span></div>')
                out.append("</div>")
                out.append('<div class="strips">')
                for cell, rec in ((arm_hard, h), (arm_soft, s)):
                    out.append(f'<div class="sline"><b>{esc(ARM_LABEL[cell].split(" · ")[1])}'
                               f'</b><img loading="lazy" src="filmstrips/{esc(rec["stem"])}.jpg">'
                               f'</div>')
                out.append("</div>")
                out.append('<div class="meta">')
                out.append(f'<div class="nm">{esc(mp)} '
                           f'<span class="chip {esc(s["style"])}">{esc(s["style"])}</span></div>')
                out.append(f'<div class="kv">feather={s["feather"]} rim={s["rim_amount"]} '
                           f'glow={s["glow_amount"]} · seed={s["map_seed"]}</div>')
                out.append(f'<div class="kv">{esc(h["from"])} ({esc(h["label_from"])}) '
                           f'&rarr; {esc(h["to"])} ({esc(h["label_to"])})</div>')
                out.append(f'<div class="kv">endpoint bad px: step()={h["endpoint_bad_px"]} '
                           f'soft={s["endpoint_bad_px"]}</div>')
                out.append("</div></div>")
        out.append("</div>")

    # ---- blind audit ------------------------------------------------------
    audit_p = run_dir / "AUDIT_RESULT.json"
    if audit_p.exists():
        aud = json.load(open(audit_p))
        pa = aud["per_arm"]
        GEO = {"stripes", "checker", "spiral", "voronoi"}
        APE = {"fbm", "radial", "linear"}
        per_clip = aud["per_clip"].values()

        def sub(pred):
            s = [c for c in per_clip if pred(c)]
            b = sum(c["grade"] == "BAD" for c in s)
            return b, len(s)

        out.append("<h2>Blind BAD-rate audit</h2>")
        out.append(
            '<p class="sub">16 clips per arm, drawn at random from a fixed seed before '
            'anything was looked at, cut into anonymous 3-frame strips at progress '
            '0.3 / 0.5 / 0.7, shuffled across arms and graded with no arm, map or '
            'compositor label visible; the id&rarr;arm key was joined only afterwards. '
            'Rubric fixed in advance: <b>BAD = reads as a digital artefact</b> (a hard '
            'alpha cut with no material at the boundary, a visible synthetic primitive, '
            'an undirected crossfade, or speckle dirt along the front); borderline counts '
            'as BAD. <b>Caveat: a single grader, and that grader rendered the clips.</b> '
            'Treat the ordering as the finding and the exact percentages as soft.</p>')
        out.append("<table><tr><th>&nbsp;</th><th>hard <code>step()</code></th>"
                   "<th>feathered <code>luma_soft</code></th></tr>")
        out.append(f'<tr><td>shipped maps (exp_075)</td>'
                   f'<td class="bad">{pa["A1_current"]["bad_rate"]:.0%} BAD</td>'
                   f'<td>{pa["A2_soft_same_map"]["bad_rate"]:.0%} BAD</td></tr>')
        out.append(f'<tr><td>new arrival-time maps</td>'
                   f'<td class="bad">{pa["A4_new_map_hard"]["bad_rate"]:.0%} BAD</td>'
                   f'<td class="ok">{pa["A3_new_map_soft"]["bad_rate"]:.0%} BAD</td></tr>')
        out.append("</table>")
        ah, as_ = (sub(lambda c: c["map"] in APE and c["compositor"] == "hard"),
                   sub(lambda c: c["map"] in APE and c["compositor"] == "soft"))
        gh, gs = (sub(lambda c: c["map"] in GEO and c["compositor"] == "hard"),
                  sub(lambda c: c["map"] in GEO and c["compositor"] == "soft"))
        out.append("<table><tr><th>shipped maps, split</th><th>hard</th><th>soft</th></tr>"
                   f"<tr><td>aperiodic — fbm / radial / linear</td>"
                   f"<td>{ah[0]}/{ah[1]}</td><td class='ok'>{as_[0]}/{as_[1]}</td></tr>"
                   f"<tr><td>geometric — stripes / checker / spiral / voronoi</td>"
                   f"<td>{gh[0]}/{gh[1]}</td><td class='bad'>{gs[0]}/{gs[1]}</td></tr></table>")
        out.append(
            '<div class="note"><b>Answer: it was <code>step()</code> first, and the maps '
            'second — but only three of the seven shipped maps were ever salvageable.</b>'
            '<ul style="margin:8px 0 0 18px;padding:0">'
            '<li>Better maps through the old compositor buy <b>nothing</b>: 14/16 BAD '
            'either way (Fisher p = 1.00). The <code>step()</code> gates everything.</li>'
            '<li>Fixing only the compositor takes the shipped maps from 88% to 56% BAD '
            '(p = 0.11, n = 16 — direction clear, size not resolved).</li>'
            '<li>The 56% residual is not spread evenly. The three <i>aperiodic</i> shipped '
            f'maps go {ah[0]}/{ah[1]} &rarr; {as_[0]}/{as_[1]} BAD — the feather rescues '
            f'them outright. The four <i>geometric</i> ones stay {gs[0]}/{gs[1]} BAD: a '
            'feathered checkerboard is still a checkerboard, and no boundary treatment '
            'fixes a periodic tiling.</li>'
            '<li>New maps + fixed compositor is the best cell at 31% BAD (p = 0.003 vs the '
            'shipped baseline), but it is <b>not</b> statistically separable from '
            'rescued-aperiodic-shipped (p = 0.62). Their value is variety and '
            'content-awareness, not a higher ceiling.</li></ul></div>')

    # ---- per-arm summary --------------------------------------------------
    out.append("<h2>Per-arm summary</h2>")
    out.append('<p class="sub">The 2×2. Endpoint identity is a side finding: '
               '<code>step(progress, m)</code> returns 1 when <code>m == progress</code>, so at '
               '<code>progress = 1</code> every pixel sitting at the matte\'s maximum keeps '
               'showing the outgoing shot. Any matte normalised to [0,1] has such pixels, so the '
               'shipped compositor leaks a speckle of frame A into the final conditioning block. '
               '<code>luma_soft.glsl</code> remaps the threshold to '
               '<code>p = progress·(1+2f) − f</code> and is exactly clean.</p>')
    out.append("<table><tr><th>arm</th><th>maps</th><th>compositor</th><th>clips</th>"
               "<th>endpoint MAE (max)</th><th>endpoint bad px (max)</th></tr>")
    for arm in ("A1_current", "A2_soft_same_map", "A3_new_map_soft", "A4_new_map_hard"):
        rows = [m for m in man if m["arm"] == arm]
        if not rows:
            continue
        out.append(
            f"<tr><td>{esc(arm)}</td><td>{esc(rows[0]['map_family'])}</td>"
            f"<td>{esc(rows[0]['compositor'])}</td><td>{len(rows)}</td>"
            f"<td>{max(r['endpoint_mae'] for r in rows):.4f}</td>"
            f"<td>{max(r['endpoint_bad_px'] for r in rows)}</td></tr>")
    out.append("</table>")

    out.append('<p class="sub" style="margin-top:26px">Brush alphas: David Revoy '
               '<i>Krita brushes 2025-01</i> bundle, <b>CC-0 / public domain</b> '
               '(davidrevoy.com/article1060). No commercial or ML-restricted matte pack was '
               'downloaded. Footage: the repo\'s own curated endpoint bank '
               '(<code>data/processed/synth_endpoints</code>).</p>')
    out.append("</div>")

    dst = VIEW_DIR / "index.html"
    dst.write_text("\n".join(out))
    return dst


if __name__ == "__main__":
    rd = (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
          else sorted((REPO_ROOT / "outputs" / "videos"
                       / "exp_084_luma_matte_viewer").glob("run_*"))[-1])
    print(build(rd))
