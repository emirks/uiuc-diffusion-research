"""Build the HTML viewer for an exp_083 D3 pilot run.

Same shape as the exp_076 run_0005 viewer the owner liked — compact ~230px
auto-fill grid, IntersectionObserver autoplay, filmstrip toggle, light/dark,
colour-coded metrics — regrouped so the two design axes are visually obvious.

Usage:  python build_viewer.py <run_dir>
Serve from the REPO ROOT so relative media paths resolve.
"""

from __future__ import annotations

import html
import json
import pathlib
import statistics
import sys
from collections import OrderedDict

COUPLING_NOTE = {
    "none": ("camera only", "Pure camera move — nothing is attached to the scene. "
             "This is the contrast group."),
    "depth": ("depth-coupled", "The effect is driven by the depth field: wipe order, "
              "Beer-Lambert extinction, or the focus plane."),
    "world": ("world-coupled", "The dissolve field is sampled at unprojected scene "
              "positions, so the pattern sticks to surfaces, parallaxes with the "
              "camera and foreshortens on oblique geometry."),
    "subject": ("subject-anchored", "The field is centred on the foreground "
                "subject's own world position, so the effect emanates from the "
                "object and travels with it. This is the manner a screen-space "
                "shader structurally cannot express."),
}

CSS = """
:root{--bg:#0f1115;--fg:#e6e8ec;--mut:#9aa3b2;--card:#171a21;--line:#252a34;--acc:#7cc0ff;--acc2:#c4a3ff}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--fg:#12141a;--mut:#5d6675;--card:#fff;--line:#e2e5ea;--acc:#1668c9;--acc2:#6d3fc4}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:44px 0 4px;padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:14px;margin:22px 0 6px;color:var(--acc2);font-weight:600}
.sub{color:var(--mut);margin:0 0 18px;max-width:82ch}
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
.tag.c-subject{background:#6d3fc4;color:#fff} .tag.c-world{background:#1668c9;color:#fff}
.tag.c-depth{background:#2f7d63;color:#fff} .tag.c-none{background:#6b7280;color:#fff}
.strip{width:100%;display:block;border-top:1px solid var(--line);image-rendering:auto}
.ok{color:#4ade80} .warn{color:#fbbf24} .bad{color:#f87171}
.toggle{margin:10px 0 16px;color:var(--mut);font-size:12.5px;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5;border-bottom:1px solid var(--line)}
.toggle label{margin-right:16px;cursor:pointer}
table{border-collapse:collapse;font-size:12px;margin:8px 0 4px}
th,td{border:1px solid var(--line);padding:4px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600}
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
    hr = m["hole_radius_max"]
    # 85 px is where the blind audit flips: every clip below it was shippable,
    # 13 of the 14 BAD clips were above it.
    hcls = "ok" if hr < 85 else ("warn" if hr < 120 else "bad")
    p = m["params"]
    tags = [f'<span class="tag c-{m["coupling"]}">{COUPLING_NOTE[m["coupling"]][0]}</span>',
            f'<span class="tag">{html.escape(m["family"])}</span>',
            f'<span class="tag">{m["n_frames"]}f</span>']
    if m["dissolve"] != "none":
        tags.append(f'<span class="tag">{html.escape(m["dissolve"])}</span>')
    if p.get("fog", 0):
        tags.append('<span class="tag">fog</span>')
    if p.get("focus", 0):
        tags.append('<span class="tag">rack focus</span>')
    if p.get("handheld", 0):
        tags.append('<span class="tag">handheld</span>')
    if p.get("dolly_zoom", 0):
        tags.append('<span class="tag">dolly-zoom</span>')
    if p.get("motion_blur", 1) > 1:
        tags.append(f'<span class="tag">mblur {p["motion_blur"]}x</span>')
    return f"""<div class="card">
<video src="{rel}/videos/{m['stem']}.mp4" muted loop playsinline preload="none"></video>
<img class="strip" src="{rel}/filmstrips/{m['stem']}.jpg" loading="lazy">
<div class="meta">
  <div class="sh">{html.escape(m['recipe'])}</div>
  <div>{''.join(tags)}</div>
  <div class="kv">{html.escape(m['describe'])}</div>
  <div class="kv">{html.escape(m['from'])} &rarr; {html.escape(m['to'])}</div>
  <div class="kv">seam <span class="{cls}">{r:.2f}</span> &middot;
      hole <span class="{hcls}">{hr:.0f}px</span> &middot;
      PI {m['parallax']['pi']:.2f}</div>
</div></div>"""


def grid(rows) -> str:
    return '<div class="grid">' + "".join(card(m) for m in rows) + "</div>"


def table(head: list[str], rows: list[list]) -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>"
                for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def main() -> None:
    run_dir = pathlib.Path(sys.argv[1]).resolve()
    man = json.load(open(run_dir / "manifest.json"))
    ratios = sorted(max(m["seam_ratio_in"], m["seam_ratio_out"]) for m in man)
    med = statistics.median(ratios)
    over = sum(r > 2.0 for r in ratios)
    fid = max(m["endpoint_maxabs"] for m in man)
    mean_s = sum(m["render_s"] for m in man) / len(man)
    by_len: dict[int, int] = {}
    for m in man:
        by_len[m["n_frames"]] = by_len.get(m["n_frames"], 0) + 1

    aud = run_dir / "audit.json"
    audit = json.load(open(aud)) if aud.exists() else None
    verdict = ""
    if audit and audit.get("bad"):
        nb, ns = len(audit["bad"]), len(audit["sampled"])
        hb = [max(m["hole_radius_max"] for m in man if m["stem"] == s) for s in audit["bad"]]
        ok = [m["hole_radius_max"] for m in man
              if m["stem"] in audit["sampled"] and m["stem"] not in audit["bad"]]
        gated = sum(m["hole_radius_max"] >= 85 for m in man)
        verdict = f"""<h2>Honest verdict — {nb} of {ns} sampled clips are unusable ({100*nb/ns:.0f}%)</h2>
<p class="sub">The sample was drawn from a fixed seed <b>before any clip was viewed</b>, and judged
from full-resolution contact sheets of the rendered middle. Almost every failure is the
<b>same defect</b>: the world-space dissolve punches alpha holes in <i>both</i> layers at once,
so at mid-transition ~25&nbsp;% of the frame has no geometry from either scene, and the push-pull
inpainter — which can only pull real colour across about 40&nbsp;px — leaves a hard black void or
a flat smear. Marginal-but-shippable clips (soft edge bands, small corner wedges, a green-tinted
fog cast) were <b>not</b> counted bad.</p>
<p class="sub"><b>The good news is that it is gateable, for free.</b> Max hole radius, which the
renderer already computes, separates the sample almost perfectly: every BAD clip except one sat
at {min(hb):.0f}&nbsp;px or above, every shippable clip at {max(ok):.0f}&nbsp;px or below. A
<code>hole_radius &lt; 85&nbsp;px</code> gate catches 13 of the 14 failures and rejects none of
the good clips — at the cost of dropping {gated} of {len(man)} clips
({100*gated/len(man):.0f}&nbsp;%) of this pilot's operator mix. The one miss is a different
mechanism: a source clip's letterbox bar warped into a black slab.</p>
<p class="sub">Cards are colour-coded on that gate: <span class="ok">green &lt; 85&nbsp;px</span>,
<span class="warn">amber 85–120</span>, <span class="bad">red &gt; 120</span>. Judge for
yourself — the red ones are the argument.</p>"""

    P = [f"""<title>exp_083 — D3 pilot: depth-parallax 3D procedural transitions</title>
<style>{CSS}</style><script>{JS}</script>
<div class="wrap">
<h1>D3 pilot — depth-parallax 3D procedural transitions</h1>
<p class="sub">Each clip is <b>start9 + a rendered middle + end9</b>. The two 9-frame anchors
are <b>real consecutive frames sliced out of a real bank clip</b> and copied through
<b>verbatim</b> — nothing is generated, extended, held or reversed — so conditioning fidelity
is exact by construction. The middle is a 2.5D render: each facing frame is unprojected by its
Depth&nbsp;Anything&nbsp;V2 depth map into a displaced mesh and re-rendered from one continuous
virtual camera flying out of scene A and coming to rest in scene B.</p>
<p class="sub"><b>Why this pilot exists.</b> The D2 shader stratum assumed operator&nbsp;&perp;&nbsp;content:
any screen-space wipe applies to any pair. Its generalist missed the win bar and the losses
concentrated on exactly the <i>content-coupled</i> donors — shadow, saint_glow,
display_transition — effects that attach to a foreground object and travel with it. A
screen-space shader structurally cannot do that. Here the dissolve field is sampled at
<b>unprojected scene positions</b>, so it sticks to surfaces; the <span class="tag c-subject">subject-anchored</span>
families additionally centre it on the foreground object's own world position. Judge the
coloured coupling tag, not just the camera move.</p>
<p class="sub"><b>Varying lengths.</b> n_middle &isin; {{7, 15, 23, 31}} with 9+9 anchors gives
totals 25 / 33 / 41 / 49 frames — every one legal for the causal VAE (F = 8k+1). Nothing is
padded.</p>
<div class="stats">
<div class="stat"><b>{len(man)}</b><span>clips</span></div>
<div class="stat"><b>{med:.2f}</b><span>median seam ratio (1.0 = as smooth as the content's own motion)</span></div>
<div class="stat"><b>{over}</b><span>clips over the 2.0 seam bar</span></div>
<div class="stat"><b>{fid}</b><span>max abs endpoint diff (0 = verbatim)</span></div>
<div class="stat"><b>{len({m['from'] for m in man} | {m['to'] for m in man})}</b><span>distinct bank endpoints</span></div>
<div class="stat"><b>{mean_s:.1f}s</b><span>mean CPU render per clip</span></div>
</div>
<p class="sub">{table(["n_frames", "clips"], [[k, v] for k, v in sorted(by_len.items())])}</p>
<div class="toggle">
<label><input type="checkbox" id="autoplay" checked> autoplay on scroll</label>
<label><input type="checkbox" id="showstrips"> show filmstrips</label>
</div>
{verdict}"""]

    def sec(title: str, blurb: str) -> None:
        P.append(f'<h2>{html.escape(title)}</h2><p class="sub">{blurb}</p>')

    # ---- AXIS A: same content x different operator -----------------------
    sec("Axis A — same content × different operator",
        "One endpoint pair, every operator recipe in the bank. This is the signal a real "
        "corpus cannot supply: the same two endpoints admit many valid transitions. Recipes "
        "are ordered by coupling, so the contrast between a bare camera move and a "
        "subject-anchored dissolve is side by side.")
    order = {"none": 0, "depth": 1, "world": 2, "subject": 3}
    groups: OrderedDict[str, list] = OrderedDict()
    for m in man:
        if m["block"].startswith("axisop"):
            groups.setdefault(m["pair_id"], []).append(m)
    for pid, rows in groups.items():
        rows.sort(key=lambda m: (order[m["coupling"]], m["recipe"]))
        P.append(f'<h3>pair: {html.escape(pid)}</h3>')
        P.append(grid(rows))

    # ---- AXIS B: same operator x different content -----------------------
    sec("Axis B — same operator × different content",
        "The mirror image: one operator instance with byte-identical parameters, applied "
        "across several endpoint pairs. A content-coupled operator should look like the "
        "<i>same manner</i> happening to different subjects, not like the same overlay "
        "pasted on different pictures.")
    groups = OrderedDict()
    for m in man:
        if m["block"].startswith("axiscontent"):
            groups.setdefault(m["tag"], []).append(m)
    for tag, rows in sorted(groups.items()):
        r0 = rows[0]
        P.append(f'<h3>{html.escape(r0["recipe"])} — {html.escape(r0["describe"])}</h3>')
        P.append(grid(rows))

    # ---- lengths ---------------------------------------------------------
    sec("Length sweep — 25 / 33 / 41 / 49 frames",
        "Same pair, same operator, the four legal totals. The middle stretches; the anchors "
        "never do. A short middle gives an abrupt, cut-like transition; a long one gives the "
        "camera room to travel, at the price of more disocclusion.")
    groups = OrderedDict()
    for m in man:
        if m["block"] == "length":
            groups.setdefault(m["recipe"], []).append(m)
    for rec, rows in groups.items():
        rows.sort(key=lambda m: m["n_frames"])
        P.append(f'<h3>{html.escape(rec)}</h3>')
        P.append(grid(rows))

    # ---- camera families -------------------------------------------------
    sec("Camera families — bare move, one pair",
        "Seven trajectory types with everything else matched, as the contrast group: this is "
        "what the operator looks like with <i>nothing</i> attached to the content.")
    P.append(grid([m for m in man if m["block"] == "family"]))

    # ---- amplitude sweep -------------------------------------------------
    amp = [m for m in man if m["block"] == "amp"]
    if amp:
        sec("Amplitude sweep — the disocclusion probe",
            "Bare camera move (no dissolve, so the holes are purely geometric). A camera move "
            "reveals geometry the single depth layer never saw; push-pull inpainting can only "
            "pull real colour across about 40&nbsp;px, so <b>hole radius</b> is the honest "
            "measure of when the 2.5D approximation breaks. Watch the number climb with "
            "amplitude.")
        rows = [[f'{m["family"]} @ {m["params"]["amplitude"]:.1f}',
                 f'{100 * m["uncovered_mean"]:.1f}%', f'{m["hole_radius_max"]:.0f}px',
                 f'{max(m["seam_ratio_in"], m["seam_ratio_out"]):.2f}']
                for m in sorted(amp, key=lambda m: (m["family"],
                                                    m["params"]["amplitude"]))]
        P.append(table(["operator", "uncovered (mean)", "max hole radius", "seam"], rows))
        P.append(grid(sorted(amp, key=lambda m: (m["family"],
                                                 m["params"]["amplitude"]))))

    # ---- diversity -------------------------------------------------------
    sec("Diversity sample — random recipe × random pair × random length",
        "Both endpoints of every pair come from <b>different</b> source clips, with the "
        "optional effects (handheld, motion blur, fog) re-enabled so the sample is not "
        "sterile.")
    P.append(grid([m for m in man if m["block"] == "diverse"]))

    P.append("</div>")
    out = run_dir / "viewer.html"
    out.write_text("\n".join(P))
    print(f"[viewer] {out}")


if __name__ == "__main__":
    main()
