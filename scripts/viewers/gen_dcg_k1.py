#!/usr/bin/env python3
"""Generate the DCG cycle-0 K1 viewer (matched-DCG vs swapped-DCG, champion store/runs/002).

The K1 reference-reading test: for each (operator, endpoint), does amplifying the TRUE demonstration
(matched DCG) make the generated transition follow that operator MORE than amplifying a WRONG
demonstration (swapped DCG)? Each card shows the true demo + the wrong demo, then the 8 arms grouped
by condition, with a global seed selector (42/43/44).

Regeneratable: `python3 scripts/viewers/gen_dcg_k1.py`. Media arrives via the `gens/` and `corpus/`
symlinks in the viewer dir (relative paths only — the rule that keeps viewers alive).
"""
import glob
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # repo root
M = HERE / "misc/2026-08-12_method_novelty"
STD = HERE / "data/processed/transitions_std121"
VDIR = HERE / "outputs/viewers/dcg_k1"
VDIR.mkdir(parents=True, exist_ok=True)


def rel_link(link: Path, target: Path) -> None:
    """Create/refresh a RELATIVE symlink at `link` pointing to `target` (viewerctl convention)."""
    rel = os.path.relpath(target, link.parent)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == rel:
            return
        link.unlink()
    link.symlink_to(rel)


rel_link(VDIR / "gens", M / "dcg_grid/videos")
rel_link(VDIR / "corpus", STD)

# ---- parse manifest into 12 items ----
rows = []
for f in sorted(glob.glob(str(M / "dcg_grid/manifest/shard_*.jsonl"))):
    for line in open(f):
        line = line.strip()
        if line:
            d = json.loads(line)
            if "arm" in d:
                rows.append(d)


def corpus_rel(clip: str):
    hits = glob.glob(str(STD / "*" / f"{clip}.mp4"))
    if not hits:
        return None
    return "corpus/" + os.path.relpath(hits[0], STD)


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

# arm groups: (arm, label)
MATCHED = [("A1", "w1.5"), ("A2", "w3.0"), ("A3", "w6.0")]
SWAPPED = [("A4", "w1.5"), ("A5", "w3.0"), ("A6", "w6.0")]


def vid(op, ep, arm):
    # src set by JS from data-* so the seed selector can swap it; poster-less, hover-to-play
    return (f'<video data-op="{op}__{ep}" data-arm="{arm}" muted loop playsinline preload="none" '
            f'onmouseover="this.play()" onmouseout="this.pause()"></video>')


def ref_fig(rel, label, cls):
    if not rel:
        return f'<figure class="ref"><div class="missing">demo not found</div><figcaption>{label}</figcaption></figure>'
    return (f'<figure class="ref {cls}"><video src="{rel}" muted loop playsinline preload="metadata" '
            f'onmouseover="this.play()" onmouseout="this.pause()"></video><figcaption>{label}</figcaption></figure>')


cards = []
for it in items:
    op, ep = it["op"], it["endpoint"]
    mref = corpus_rel(it["matched_ref"]) if it["matched_ref"] else None
    sref = corpus_rel(it["swapped_ref"]) if it["swapped_ref"] else None

    def arm_fig(arm, label, tone):
        return (f'<figure class="gen {tone}">{vid(op, ep, arm)}'
                f'<figcaption>{label}</figcaption></figure>')

    matched_figs = "".join(arm_fig(a, l, "good") for a, l in MATCHED)
    swapped_figs = "".join(arm_fig(a, l, "bad") for a, l in SWAPPED)
    cards.append(f"""
  <section class="card">
    <h2>{op} <span class="ep">/ {ep}</span></h2>
    <div class="refs">
      {ref_fig(mref, "TRUE demo — target look ("+str(it['matched_ref'])+")", "true")}
      {ref_fig(sref, "WRONG demo — used by swapped arms ("+str(it['swapped_ref'])+")", "wrong")}
    </div>
    <div class="arms">
      <div class="col ctx">
        <div class="colhead">context</div>
        {arm_fig("A0", "baseline (no DCG)", "ctx")}
        {arm_fig("A13", "endpoint-only (no demo)", "ctx")}
      </div>
      <div class="col matched">
        <div class="colhead good">MATCHED DCG — amplify TRUE demo →</div>
        <div class="strip">{matched_figs}</div>
      </div>
      <div class="col swapped">
        <div class="colhead bad">SWAPPED DCG — amplify WRONG demo →</div>
        <div class="strip">{swapped_figs}</div>
      </div>
    </div>
  </section>""")

blocks = "".join(cards)
seed_btns = "".join(f'<button data-seed="{s}"{" class=on" if s==42 else ""}>seed {s}</button>' for s in SEEDS)

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>DCG K1 · matched vs swapped</title>
<style>
 :root{{--bg:#0e0f13;--fg:#e6e6e6;--mut:#9aa0a6;--card:#15171d;--line:#23262f;--good:#7ee787;--bad:#ff9e9e}}
 body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:22px}}
 h1{{margin:0 0 4px}} .sub{{color:var(--mut);max-width:1000px}}
 .legend{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:14px 0 6px;max-width:1000px;font-size:13px}}
 .legend b.g{{color:var(--good)}} .legend b.r{{color:var(--bad)}} .legend .k{{color:#ffd166}}
 .bar{{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0 12px;margin-bottom:6px;border-bottom:1px solid var(--line)}}
 .bar button{{background:#1a1c22;color:var(--fg);border:1px solid #2a2d36;border-radius:6px;padding:6px 12px;margin-right:8px;cursor:pointer;font:inherit}}
 .bar button.on{{background:#2b3550;border-color:#3f5088;color:#cfe0ff}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:16px 0}}
 h2{{margin:0 0 10px;font-size:16px}} h2 .ep{{color:var(--mut);font-weight:normal}}
 .refs{{display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap}}
 figure{{margin:0}} video{{width:100%;border-radius:5px;background:#000;aspect-ratio:16/10;object-fit:cover;display:block}}
 figcaption{{font-size:11px;color:#8b919b;text-align:center;margin-top:3px}}
 .ref{{width:220px}} .ref.true figcaption{{color:var(--good)}} .ref.wrong figcaption{{color:var(--bad)}}
 .ref.true video{{outline:2px solid #2f6b3a}} .ref.wrong video{{outline:2px solid #6b2f2f}}
 .missing{{width:100%;aspect-ratio:16/10;background:#191b21;border:1px dashed #33363f;border-radius:5px;display:flex;align-items:center;justify-content:center;color:#5b6270;font-size:11px}}
 .arms{{display:grid;grid-template-columns:170px 1fr 1fr;gap:14px;align-items:start}}
 .colhead{{font-size:12px;color:var(--mut);margin-bottom:6px;font-weight:600}}
 .colhead.good{{color:var(--good)}} .colhead.bad{{color:var(--bad)}}
 .col.ctx .gen{{margin-bottom:8px}}
 .strip{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}}
 .gen.good figcaption{{color:#bfe9c4}} .gen.bad figcaption{{color:#f0c4c4}}
 @media (max-width:900px){{.arms{{grid-template-columns:1fr}} .strip{{grid-template-columns:1fr 1fr 1fr}}}}
</style></head><body>
<h1>DCG cycle-0 · K1 — matched vs swapped demonstration</h1>
<div class="sub">Champion adapter <b>ctt_v2 / store/runs/002</b> (step 10000), NEUTRAL prompt, no text-CFG / no STG
(guidance_scale 1.0) — these arms isolate the <i>reference channel</i>, so judge the <b>matched-vs-swapped
contrast</b>, not absolute polish. Hover any clip to play; use the seed selector.</div>
<div class="legend">
 <b>The test.</b> DCG = demonstration-contrastive guidance: it <i>amplifies</i> how much the generation follows a
 demonstration. <b class="g">MATCHED DCG</b> amplifies the <b class="g">TRUE</b> demo for that operator;
 <b class="r">SWAPPED DCG</b> amplifies a <b class="r">WRONG</b> demo (a different operator, per the derangement map).
 Strength grows left→right (w1.5 → 3 → 6). <b>K1 passes</b> if, as you turn DCG up, the <b class="g">matched</b>
 clips look <i>more like the TRUE demo's transition manner</i> while the <b class="r">swapped</b> clips drift toward a
 <i>different</i> manner. If matched and swapped look alike, the model isn't reading the specific demonstration.
 Pre-registered bar: matched preferred over swapped in <span class="k">&gt;55%</span> of items → reads; else dies.
 Context arms: <b>baseline</b> (matched demo, no DCG) and <b>endpoint-only</b> (no demo at all).
</div>
<div class="bar">{seed_btns} <span class="sub" style="margin-left:10px">12 operators×endpoints · 8 arms each · seeds 42/43/44</span></div>
{blocks}
<script>
 let seed = 42;
 function apply(){{
   document.querySelectorAll('video[data-op]').forEach(v=>{{
     const src = `gens/${{v.dataset.op}}__s${{seed}}__${{v.dataset.arm}}.mp4`;
     if(v.getAttribute('src')!==src){{ v.setAttribute('src', src); v.load(); }}
   }});
 }}
 document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{{
   seed=+b.dataset.seed;
   document.querySelectorAll('.bar button').forEach(x=>x.classList.toggle('on', x===b));
   apply();
 }});
 apply();
</script>
</body></html>"""
(VDIR / "index.html").write_text(html)
print(f"[gen] wrote {VDIR/'index.html'} — {len(items)} items × 8 arms × {len(SEEDS)} seeds")
print(f"[gen] symlinks: gens -> {os.readlink(VDIR/'gens')} ; corpus -> {os.readlink(VDIR/'corpus')}")
