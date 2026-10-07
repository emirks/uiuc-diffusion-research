#!/usr/bin/env python3
"""Explanatory (non-blinded) results viewer for the DCG-CN KEEP@w3 finding — best_shot campaign.
Per effect class: DEMO + BASELINE, then the guidance SWEEP — matched-DCG and swapped-DCG each at
w=1.5/3/6 (left->right = stronger guidance), all autoplaying & labelled, takeaway spelled out.
Regen: python scripts/viewers/gen_dcg_cn_results.py"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
GRID = HERE / "misc/2026-08-12_method_novelty/dcg_grid/videos"
STD = HERE / "data/processed/transitions_std121"
VDIR = HERE / "outputs/viewers/dcg_cn_results"
VDIR.mkdir(parents=True, exist_ok=True)


def rel_link(link, target):
    rel = os.path.relpath(target, link.parent)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == rel:
            return
        link.unlink()
    link.symlink_to(rel)


rel_link(VDIR / "gens", GRID)
rel_link(VDIR / "corpus", STD)

# (op, endpoint, demo_clip, demo_desc, wrong_class) — one clean/clear item per effect class
ITEMS = [
    ("shadow_smoke", "shadow_smoke_2", "shadow_smoke_1", "dissolve into black smoke", "portal"),
    ("color_rain",   "color_rain_3",   "color_rain_1",   "wash in falling color",      "wireframe"),
    ("hero_flight",  "hero_flight_5",  "hero_flight_0",  "launch into flight",         "animalization"),
    ("wireframe",    "wireframe_0",    "wireframe_1",    "break into a wireframe mesh","color_rain"),
    ("animalization","animalization_3","animalization_1","morph toward an animal",     "hero_flight"),
    ("portal",       "portal_10",      "portal_0",       "sweep through a glowing portal","shadow_smoke"),
]
# matched arms A1/A2/A3 and swapped arms A4/A5/A6 at these guidance strengths:
WSWEEP = [("1.5", "A1", "A4"), ("3", "A2", "A5"), ("6", "A3", "A6")]


def corpus_rel(clip):
    import glob
    h = glob.glob(str(STD / "*" / f"{clip}.mp4"))
    return "corpus/" + os.path.relpath(h[0], STD) if h else None


def vid(src, cls, cap):
    v = (f'<video data-src="{src}" muted loop playsinline preload="none"></video>' if src
         else '<div class="missing">missing</div>')
    return f'<figure class="{cls}">{v}<figcaption>{cap}</figcaption></figure>'


cards = []
for op, ep, demo, desc, wrong in ITEMS:
    base = f"gens/{op}__{ep}__s42__A0.mp4"
    refs = (vid(corpus_rel(demo), "demo", "<b>THE DEMO</b><br><span>the effect to copy</span>") +
            vid(base, "base", "<b>BASELINE</b><br><span>no DCG (w=1)</span>"))
    matched = "".join(vid(f"gens/{op}__{ep}__s42__{a}.mp4", "match", f"w={w}") for w, a, _ in WSWEEP)
    swapped = "".join(vid(f"gens/{op}__{ep}__s42__{a}.mp4", "swap", f"w={w}") for w, _, a in WSWEEP)
    cards.append(f"""
  <section class="card">
    <h2>{op.replace('_',' ')} <span class="desc">— {desc}</span></h2>
    <div class="refs">{refs}</div>
    <div class="sweep">
      <div class="lbl g">MATCHED-DCG — amplify the TRUE demo, guidance →</div>
      <div class="strip">{matched}</div>
    </div>
    <div class="sweep">
      <div class="lbl r">SWAPPED-DCG — amplify the WRONG ({wrong}) demo, guidance →</div>
      <div class="strip">{swapped}</div>
    </div>
    <p class="take">As guidance grows left→right: <b class="g">MATCHED</b> intensifies the demo's own
    effect; <b class="r">SWAPPED</b> intensifies a <i>different</i> effect (the {wrong} it was fed).
    <b>w=1.5</b> is faint, <b>w=3</b> is the clean sweet spot, <b>w=6</b> is strongest but starts to
    over-amplify (artifacts / copying the demo).</p>
  </section>""")

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>DCG-CN · guidance sweep</title>
<style>
 :root{{--bg:#0e0f13;--fg:#e6e6e6;--mut:#9aa0a6;--card:#15171d;--line:#23262f;--demo:#ffd166;--g:#7ee787;--r:#ff9e9e;--n:#9aa0a6}}
 body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0 auto;padding:24px;max-width:1200px}}
 h1{{margin:0 0 6px;font-size:22px}} .lede{{color:var(--mut);max-width:900px;margin-bottom:8px}}
 .verdict{{display:inline-block;background:#16351d;border:1px solid #2f6b3a;color:#bff0c8;border-radius:8px;padding:8px 14px;font-weight:600;margin:8px 0 4px}}
 .how{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0 18px;max-width:900px;font-size:13px}}
 .how b.g{{color:var(--g)}} .how b.r{{color:var(--r)}} .how b.n{{color:var(--n)}} .how b.d{{color:var(--demo)}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:18px 0}}
 h2{{margin:0 0 10px;font-size:17px}} h2 .desc{{color:var(--mut);font-weight:normal;font-size:14px}}
 .refs{{display:grid;grid-template-columns:180px 180px;gap:12px;margin-bottom:12px}}
 .sweep{{margin:8px 0}} .lbl{{font-size:12px;font-weight:600;margin-bottom:5px}} .lbl.g{{color:var(--g)}} .lbl.r{{color:var(--r)}}
 .strip{{display:grid;grid-template-columns:repeat(3,minmax(0,180px));gap:12px}}
 figure{{margin:0}} video{{width:100%;border-radius:6px;background:#000;aspect-ratio:3/4;object-fit:cover;display:block}}
 figcaption{{font-size:11.5px;text-align:center;margin-top:4px;color:var(--mut)}} figcaption b{{font-size:12px}} figcaption span{{font-size:10.5px}}
 .demo b{{color:var(--demo)}} .demo video{{outline:2px solid #6b5a2f}} .base b{{color:var(--n)}}
 .match video{{outline:2px solid #2f6b3a}} .match figcaption{{color:#bfe9c4}}
 .swap video{{outline:2px solid #6b2f2f}} .swap figcaption{{color:#f0c4c4}}
 .take{{font-size:12.5px;color:#b8bec8;margin:12px 0 2px}} .take .g{{color:var(--g)}} .take .r{{color:var(--r)}}
 .missing{{aspect-ratio:3/4;background:#191b21;border:1px dashed #33363f;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#5b6270}}
 @media (max-width:640px){{.strip{{grid-template-columns:1fr 1fr}} .refs{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<h1>DCG-CN — the guidance sweep (does it follow the demo, and how hard to push?)</h1>
<div class="lede">The model sees a <b style="color:var(--demo)">demo</b> of a transition effect and applies
it between two new clips. DCG is a test-time trick (no retraining) that amplifies how strongly it follows
that demo, by a strength knob <b>w</b>. Every tile is a looping video; within each strip w grows left→right.</div>
<div class="verdict">✓ VERDICT: KEEP @ w=3 — follows the demo at generation time. Matched beat baseline 11/12; followed the correct demo 12/12.</div>
<div class="how"><b>How to read a row.</b> <b class="d">THE DEMO</b> = the target effect. <b class="n">BASELINE</b>
= plain model, no DCG. <b class="g">MATCHED strip</b> = DCG amplifying the TRUE demo at w=1.5→3→6 (watch the
effect intensify). <b class="r">SWAPPED strip</b> = DCG amplifying a WRONG demo (watch it drift to a
different effect). Dose-response: demo-following grows +0.05 (w1.5) → +0.20 (w3) → +0.27 (w6), but quality
degrades past w≈3 — so <b>w=3</b> is the operating point, <b>w=6</b> is the over-amplified extreme.</div>
{''.join(cards)}
<p style="color:#5b6270;font-size:11px;margin-top:20px">Champion ctt_v2 / store/runs/002 step 10000, NEUTRAL prompt (effect never named in text),
text-CFG off, STG off, 30 steps, seed 42, 480×640×121f. Automated instrument (DINOv2): matched-vs-swapped attribution 12/12, robust 9/9.
Class-level result; instance-level = future work (CROSS).</p>
<script>
 const io=new IntersectionObserver(es=>es.forEach(e=>{{const v=e.target;
   if(e.isIntersecting){{ if(!v.getAttribute('src')){{v.setAttribute('src',v.dataset.src);v.load();}} v.play().catch(()=>{{}}); }}
   else v.pause(); }}),{{rootMargin:"250px"}});
 document.querySelectorAll('video').forEach(v=>io.observe(v));
</script>
</body></html>"""
(VDIR / "index.html").write_text(html)
print(f"[gen] wrote {VDIR/'index.html'} — {len(ITEMS)} effects × (baseline + matched/swapped w1.5/3/6)")
