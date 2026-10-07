#!/usr/bin/env python3
"""Generate the DCG flavor-(b) TARGET-NULL viewer (target-dissolve null vs current reference-crossfade null).

Per (operator, endpoint): the TRUE demo + baseline, then MATCHED and SWAPPED sections, each with three
w-aligned rows — current-null (CN), target-null CONST (TNc), target-null ANNEAL (TNa) — so you can compare
the cheaper target-dissolve null against the current null and see the const-vs-anneal (1/sigma) artifact
pattern as w grows. Global seed selector (42/43/44).

Regeneratable: `python3 scripts/viewers/gen_dcg_b.py`. Media via the `tn/`, `cn/`, `corpus/` symlinks in the
viewer dir (relative paths only).
"""
import glob
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # repo root
M = HERE / "misc/2026-08-12_method_novelty"
STD = HERE / "data/processed/transitions_std121"
VDIR = HERE / "outputs/viewers/dcg_b"
VDIR.mkdir(parents=True, exist_ok=True)


def rel_link(link: Path, target: Path) -> None:
    rel = os.path.relpath(target, link.parent)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == rel:
            return
        link.unlink()
    link.symlink_to(rel)


rel_link(VDIR / "tn", M / "dcg_grid_b/videos")   # target-null clips (this experiment)
rel_link(VDIR / "cn", M / "dcg_grid/videos")     # current-null + baseline (reused grid)
rel_link(VDIR / "corpus", STD)

# ---- item + ref map from the existing grid manifest ----
rows = []
for f in sorted(glob.glob(str(M / "dcg_grid/manifest/shard_*.jsonl"))):
    for line in open(f):
        line = line.strip()
        if line:
            d = json.loads(line)
            if "arm" in d:
                rows.append(d)


def corpus_rel(clip):
    hits = glob.glob(str(STD / "*" / f"{clip}.mp4"))
    return "corpus/" + os.path.relpath(hits[0], STD) if hits else None


items = {}
for d in rows:
    key = (d["op"], d["endpoint"])
    it = items.setdefault(key, {"op": d["op"], "endpoint": d["endpoint"], "matched_ref": None, "swapped_ref": None})
    if d.get("kind") in ("baseline", "matched_dcg") and d.get("ref_used"):
        it["matched_ref"] = d["ref_used"]
    if d.get("kind") == "swapped_dcg" and d.get("ref_used"):
        it["swapped_ref"] = d["ref_used"]
items = [items[k] for k in sorted(items)]
SEEDS = [42, 43, 44]
WCOLS = ["1.25", "1.5", "3.0", "6.0"]
# CN arm ids by (side, w): matched A1/A2/A3, swapped A4/A5/A6 (CN has no w1.25)
CN = {("m", "1.5"): "A1", ("m", "3.0"): "A2", ("m", "6.0"): "A3",
      ("s", "1.5"): "A4", ("s", "3.0"): "A5", ("s", "6.0"): "A6"}


def cell(op, ep, dir_, arm):
    """One video cell (src set by JS from data-*), or an empty placeholder if arm is None."""
    if arm is None:
        return '<div class="pad"></div>'
    return (f'<video data-op="{op}__{ep}" data-dir="{dir_}" data-arm="{arm}" muted loop playsinline '
            f'preload="none"></video>')


def row(op, ep, label, cells, tone):
    tiles = "".join(f'<div class="tile">{c}<span class="w">{w}</span></div>'
                    for w, c in zip(WCOLS, cells))
    return f'<div class="nrow"><div class="rlab {tone}">{label}</div><div class="strip">{tiles}</div></div>'


def section(op, ep, side, tone, title):
    cn_cells = [cell(op, ep, "cn", CN.get((side, w))) for w in WCOLS]
    tnc = [cell(op, ep, "tn", f"TNc_{side}_w{w}") for w in WCOLS]
    tna = [cell(op, ep, "tn", f"TNa_{side}_w{w}") for w in WCOLS]
    return (f'<div class="section {tone}"><div class="sechead {tone}">{title} &nbsp;— w →</div>'
            f'<div class="whead"><div class="rlab"></div><div class="strip">'
            + "".join(f'<div class="tile whcol">{w}</div>' for w in WCOLS) + '</div></div>'
            + row(op, ep, "current null (CN)", cn_cells, tone)
            + row(op, ep, "target-null CONST", tnc, tone)
            + row(op, ep, "target-null ANNEAL", tna, tone)
            + '</div>')


def ref_fig(rel, label, cls):
    if not rel:
        return f'<figure class="ref"><div class="missing">demo not found</div><figcaption>{label}</figcaption></figure>'
    return (f'<figure class="ref {cls}"><video data-src="{rel}" muted loop playsinline preload="none">'
            f'</video><figcaption>{label}</figcaption></figure>')


def base_fig(op, ep):
    return (f'<figure class="ref"><video data-op="{op}__{ep}" data-dir="cn" data-arm="A0" muted loop '
            f'playsinline preload="none"></video>'
            f'<figcaption>baseline (no DCG)</figcaption></figure>')


cards = []
for it in items:
    op, ep = it["op"], it["endpoint"]
    mref = corpus_rel(it["matched_ref"]) if it["matched_ref"] else None
    sref = corpus_rel(it["swapped_ref"]) if it["swapped_ref"] else None
    cards.append(f"""
  <section class="card">
    <h2>{op} <span class="ep">/ {ep}</span></h2>
    <div class="refs">
      {ref_fig(mref, "TRUE demo — target look ("+str(it['matched_ref'])+")", "true")}
      {base_fig(op, ep)}
      {ref_fig(sref, "WRONG demo — swapped arms ("+str(it['swapped_ref'])+")", "wrong")}
    </div>
    {section(op, ep, "m", "good", "MATCHED — amplify TRUE demo")}
    {section(op, ep, "s", "bad", "SWAPPED — amplify WRONG demo")}
  </section>""")

blocks = "".join(cards)
seed_btns = "".join(f'<button data-seed="{s}"{" class=on" if s==42 else ""}>seed {s}</button>' for s in SEEDS)

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>DCG target-null · b</title>
<style>
 :root{{--bg:#0e0f13;--fg:#e6e6e6;--mut:#9aa0a6;--card:#15171d;--line:#23262f;--good:#7ee787;--bad:#ff9e9e}}
 body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:22px}}
 h1{{margin:0 0 4px}} .sub{{color:var(--mut);max-width:1050px}}
 .legend{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:14px 0 6px;max-width:1050px;font-size:13px}}
 .legend b.g{{color:var(--good)}} .legend b.r{{color:var(--bad)}} .legend .k{{color:#ffd166}}
 .bar{{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0 12px;margin-bottom:6px;border-bottom:1px solid var(--line)}}
 .bar button{{background:#1a1c22;color:var(--fg);border:1px solid #2a2d36;border-radius:6px;padding:6px 12px;margin-right:8px;cursor:pointer;font:inherit}}
 .bar button.on{{background:#2b3550;border-color:#3f5088;color:#cfe0ff}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:16px 0}}
 h2{{margin:0 0 10px;font-size:16px}} h2 .ep{{color:var(--mut);font-weight:normal}}
 .refs{{display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap}}
 figure{{margin:0}} video{{width:100%;border-radius:5px;background:#000;aspect-ratio:16/10;object-fit:cover;display:block}}
 figcaption{{font-size:11px;color:#8b919b;text-align:center;margin-top:3px}}
 .ref{{width:200px}} .ref.true figcaption{{color:var(--good)}} .ref.wrong figcaption{{color:var(--bad)}}
 .ref.true video{{outline:2px solid #2f6b3a}} .ref.wrong video{{outline:2px solid #6b2f2f}}
 .missing{{width:100%;aspect-ratio:16/10;background:#191b21;border:1px dashed #33363f;border-radius:5px;display:flex;align-items:center;justify-content:center;color:#5b6270;font-size:11px}}
 .section{{margin:10px 0;padding:8px;border-radius:8px;border:1px solid var(--line)}}
 .section.good{{border-color:#2f6b3a55}} .section.bad{{border-color:#6b2f2f55}}
 .sechead{{font-size:12px;font-weight:700;margin-bottom:6px}} .sechead.good{{color:var(--good)}} .sechead.bad{{color:var(--bad)}}
 .nrow,.whead{{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin-bottom:6px}}
 .rlab{{font-size:12px;color:var(--mut);font-weight:600}} .rlab.good{{color:#bfe9c4}} .rlab.bad{{color:#f0c4c4}}
 .strip{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px}}
 .tile{{position:relative}} .tile .w{{position:absolute;top:2px;left:4px;font-size:10px;color:#cfe0ff;background:#0009;padding:0 3px;border-radius:3px}}
 .tile.whcol{{color:var(--mut);font-size:11px;text-align:center;padding:2px 0}} .whead .rlab{{}}
 .pad{{width:100%;aspect-ratio:16/10;border:1px dashed #2a2d36;border-radius:5px;opacity:.35}}
 @media (max-width:900px){{.nrow,.whead{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>DCG flavor-(b) · target-dissolve null vs current null</h1>
<div class="sub">Champion <b>ctt_v2 / store/runs/002</b>, NEUTRAL prompt, no text-CFG (guidance_scale 1.0) —
judge the <b>contrast</b>, not absolute polish. Hover to play; use the seed selector.</div>
<div class="legend">
 <b>What changed.</b> DCG amplifies how much the generation follows a demonstration by contrasting it against a
 <i>null</i>. <b>CN</b> = the current null (model forward on a crossfade of the <i>reference's</i> endpoints — needs a
 2nd forward pass). <b>TN</b> = the owner's target-dissolve null: the pre-encoded pixel dissolve of the <i>target's</i>
 own endpoints, used directly as the null <b>with no 2nd forward (~half the cost)</b>. <b class="k">const</b> uses a
 fixed weight (predicted to blow up at high w — a fixed null gives a ~1/σ velocity kick at low noise);
 <b class="k">anneal</b> ramps it down as noise falls (<code>w_eff=1+(w−1)·σ/σ_max</code>, bounded).
 <b>Judge:</b> does <b>TN-anneal</b> follow the operator as well as <b class="g">CN</b> at equal/better quality, and do
 <b class="g">matched</b> vs <b class="r">swapped</b> still diverge (reads the specific demo)? Watch const degrade as w↑.
</div>
<div class="bar">{seed_btns} <span class="sub" style="margin-left:10px">12 items · CN vs target-null const/anneal · w{{1.25,1.5,3,6}} · seeds 42/43/44</span></div>
{blocks}
<script>
 let seed = 42;
 function srcFor(v){{
   if(v.dataset.op) return `${{v.dataset.dir}}/${{v.dataset.op}}__s${{seed}}__${{v.dataset.arm}}.mp4`;
   return v.dataset.src || "";
 }}
 // Lazy: only load a video's src when it scrolls near the viewport; pause off-screen.
 // Keeps the ~360-video page from crashing the browser (matches iclora_neutral_effect).
 const io = new IntersectionObserver(es=>es.forEach(e=>{{
   const v = e.target;
   if(e.isIntersecting){{ const s=srcFor(v); if(v.getAttribute('src')!==s){{ v.setAttribute('src', s); v.load(); }} v.play().catch(()=>{{}}); }}
   else v.pause();
 }}), {{rootMargin:"300px"}});
 const observeAll = () => document.querySelectorAll('video').forEach(v=>io.observe(v));
 document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{{
   seed=+b.dataset.seed;
   document.querySelectorAll('.bar button').forEach(x=>x.classList.toggle('on', x===b));
   document.querySelectorAll('video[data-op]').forEach(v=>{{ v.pause(); v.removeAttribute('src'); v.load(); }});
   io.disconnect(); observeAll();
 }});
 observeAll();
</script>
</body></html>"""
(VDIR / "index.html").write_text(html)
print(f"[gen] wrote {VDIR/'index.html'} — {len(items)} items")
print(f"[gen] symlinks: tn -> {os.readlink(VDIR/'tn')} ; cn -> {os.readlink(VDIR/'cn')} ; corpus -> {os.readlink(VDIR/'corpus')}")
