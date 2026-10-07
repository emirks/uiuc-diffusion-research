#!/usr/bin/env python3
"""Viewer for the DCG-on-deployed-ctt_v2 sweep (dcg_conditioning campaign). Per item: the DEMO +
baseline(w1) + DCG w1.5/3/6, so you can SEE the honest gain at w1.5 and the reference-content
INTRUSION creeping in at w3/w6 (gen starts resembling the demo's content, not just its effect).
Regen: python scripts/viewers/gen_dcg_sweep_view.py"""
import glob, json, os
from pathlib import Path

DR = Path("/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
VIDS = DR / "misc/2026-08-14_dcg_conditioning/sweep/videos"
STD = DR / "data/processed/transitions_std121"
VDIR = DR / "outputs/viewers/dcg_sweep"
VDIR.mkdir(parents=True, exist_ok=True)
roster = [json.loads(l) for l in open(DR / "eval_ladder/registry.jsonl") if json.loads(l).get("arm") == "ic_gen"]

# pick a few clear, class-diverse items that exist
WANT = ["animalization", "color_rain", "portal", "shadow_smoke", "hero_flight", "wireframe"]
picked = []
for cls in WANT:
    for r in roster:
        if r["donor_class"] == cls and (VIDS / f"{r['item_id']}__neutral__w1.0__s42.mp4").exists():
            picked.append(r); break


def rel_link(link, target):
    rel = os.path.relpath(target, link.parent)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == rel: return
        link.unlink()
    link.symlink_to(rel)


rel_link(VDIR / "gens", VIDS); rel_link(VDIR / "corpus", STD)


def corpus_rel(clip):
    h = glob.glob(str(STD / "*" / f"{clip}.mp4")); return "corpus/" + os.path.relpath(h[0], STD) if h else None


def vid(src, cls, cap):
    v = f'<video data-src="{src}" muted loop playsinline preload="none"></video>' if src else '<div class="miss">—</div>'
    return f'<figure class="{cls}">{v}<figcaption>{cap}</figcaption></figure>'


cards = []
for r in picked:
    iid = r["item_id"]; op = r["donor_class"]
    demo = corpus_rel(r["reference"])
    cells = vid(demo, "demo", "<b>DEMO</b><br><span>the effect</span>")
    for w, lab, cls in [("1.0", "BASELINE (w1)", "base"), ("1.5", "DCG w1.5 ✓", "good"),
                        ("3.0", "DCG w3 ⚠ intrusion", "warn"), ("6.0", "DCG w6 ⚠ intrusion", "warn")]:
        cells += vid(f"gens/{iid}__neutral__w{w}__s42.mp4", cls, f"<b>{lab}</b>")
    cards.append(f'<section class="card"><h2>{op.replace("_"," ")}</h2><div class="row">{cells}</div></section>')

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>DCG on ctt_v2 · sweep</title><style>
 :root{{--bg:#0e0f13;--fg:#e6e6e6;--mut:#9aa0a6;--card:#15171d;--line:#23262f;--demo:#ffd166;--good:#7ee787;--warn:#ffb454;--n:#9aa0a6}}
 body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0 auto;padding:24px;max-width:1250px}}
 h1{{margin:0 0 6px;font-size:21px}} .lede{{color:var(--mut);max-width:960px}}
 .verdict{{background:#2a2410;border:1px solid #6b5a2f;color:#ffe0a3;border-radius:8px;padding:10px 14px;margin:12px 0;max-width:960px;font-size:13px}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:16px 0}}
 h2{{margin:0 0 8px;font-size:16px}}
 .row{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}
 figure{{margin:0}} video{{width:100%;border-radius:6px;background:#000;aspect-ratio:3/4;object-fit:cover;display:block}}
 figcaption{{font-size:11px;text-align:center;margin-top:4px;color:var(--mut)}}
 .demo b{{color:var(--demo)}} .demo video{{outline:2px solid #6b5a2f}} .base b{{color:var(--n)}}
 .good b{{color:var(--good)}} .good video{{outline:2px solid #2f6b3a}} .warn b{{color:var(--warn)}} .warn video{{outline:2px solid #6b5330}}
 .miss{{aspect-ratio:3/4;background:#191b21;border:1px dashed #33363f;border-radius:6px}}
 @media(max-width:720px){{.row{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<h1>DCG added to deployed ctt_v2 — guidance sweep (neutral)</h1>
<div class="lede">Each row: the <b style="color:var(--demo)">demo</b> effect, then the plain model and DCG at rising strength.
Watch the effect strengthen — but at <b style="color:var(--warn)">w3/w6</b> the output starts to resemble the demo's own CONTENT (intrusion), not just its effect.</div>
<div class="verdict"><b>Verdict (eval-v4, neutral):</b> DCG helps ctt_v2 <b>modestly & honestly at w=1.5</b> (+3.3pp appearance %same, demo-following up, copy guards clean).
The bigger +9pp@w3 / +13pp@w6 %same gains are substantially <b>reference-content intrusion</b> (the demo-copy guard fails: GT-exceed 16%→27%→34%), not real transition quality — so w=1.5 is the operating point.</div>
{''.join(cards)}
<p style="color:#5b6270;font-size:11px;margin-top:18px">ctt_v2 store/runs/002 step 10000, deployed config gs=4/stg=1, neutral prompt, seed 42. eval-v4 (refvfx/eval-v4-cert), 152-row roster.</p>
<script>
 const io=new IntersectionObserver(es=>es.forEach(e=>{{const v=e.target;
   if(e.isIntersecting){{ if(!v.getAttribute('src')){{v.setAttribute('src',v.dataset.src);v.load();}} v.play().catch(()=>{{}}); }} else v.pause(); }}),{{rootMargin:"250px"}});
 document.querySelectorAll('video').forEach(v=>io.observe(v));
</script></body></html>"""
(VDIR / "index.html").write_text(html)
print(f"[gen] wrote {VDIR/'index.html'} — {len(picked)} items")
