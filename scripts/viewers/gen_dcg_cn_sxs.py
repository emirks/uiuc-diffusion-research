#!/usr/bin/env python3
"""Owner blinded-SxS confirmation bundle for the DCG-CN KEEP verdict (best_shot campaign).

Per item (12): the TRUE demonstration (labelled) + 4 BLINDED candidate clips in a
deterministically-shuffled order — {A0 baseline, A2 matched-DCG w3, A5 swapped-DCG w3, A13
endpoint-only}. The owner marks which blinded clip performs the demonstrated transition class
on the endpoints, without quality collapse or verbatim demo copying. Answer key + flags written
to key.json (NOT shown in the viewer). Automated instrument already returned KEEP@w=3
(matched attribution 12/12, robust to guard exclusion 9/9); this is human confirmation.

Regeneratable: `python scripts/viewers/gen_dcg_cn_sxs.py`. Media via relative symlinks.
"""
import json, os, random
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
GRID = HERE / "misc/2026-08-12_method_novelty/dcg_grid/videos"
STD = HERE / "data/processed/transitions_std121"
SPEC = json.load(open(HERE / "misc/2026-08-12_method_novelty/GRID_SPEC.json"))
VDIR = HERE / "outputs/viewers/dcg_cn_sxs"
VDIR.mkdir(parents=True, exist_ok=True)

FLAGGED = {"color_rain_0", "wireframe_2", "portal_10"}  # w=3 guard flags (2 quality + 1 copy)
BLIND_ARMS = ["A0", "A2", "A5", "A13"]  # baseline / matched-w3 / swapped-w3 / endpoint-only
ARM_DESC = {"A0": "baseline (true demo in-context, no DCG)", "A2": "matched-DCG w=3 (TRUE demo amplified)",
            "A5": "swapped-DCG w=3 (WRONG-class demo amplified)", "A13": "endpoint-only (no demo)"}
SEEDS = [42, 43, 44]
MATCHED_REF = {o["op"]: o["matched_ref"] for o in SPEC["ops"]}
DERANGED_REF = {o["op"]: o["deranged_ref"] for o in SPEC["ops"]}
ITEMS = [(o["op"], r["endpoint"], r.get("cell", "")) for o in SPEC["ops"] for r in o["rows"]]


def rel_link(link: Path, target: Path):
    rel = os.path.relpath(target, link.parent)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == rel:
            return
        link.unlink()
    link.symlink_to(rel)


rel_link(VDIR / "gens", GRID)
rel_link(VDIR / "corpus", STD)


def corpus_rel(clip):
    hits = list(STD.glob(f"*/{clip}.mp4"))
    return "corpus/" + os.path.relpath(hits[0], STD) if hits else None


rng = random.Random(20260814)  # fixed → reproducible blinding
key = {}
cards = []
for idx, (op, ep, cell) in enumerate(ITEMS):
    perm = BLIND_ARMS[:]
    rng.shuffle(perm)  # per-item deterministic blind order
    key[f"{op}__{ep}"] = {"cell": cell, "flagged": ep in FLAGGED,
                          "positions": {f"C{i+1}": a for i, a in enumerate(perm)}}
    demo = corpus_rel(MATCHED_REF[op])
    demo_fig = (f'<figure class="demo"><video data-src="{demo}" muted loop playsinline preload="none"></video>'
                f'<figcaption>DEMONSTRATED EFFECT — “{op}” ({MATCHED_REF[op]})</figcaption></figure>'
                if demo else '<div class="missing">demo missing</div>')
    cand = "".join(
        f'<figure class="cand"><video data-op="{op}__{ep}" data-arm="{a}" muted loop playsinline '
        f'preload="none"></video><figcaption>candidate {i+1}</figcaption></figure>'
        for i, a in enumerate(perm))
    flag = ' <span class="flag">⚑ guard-flagged</span>' if ep in FLAGGED else ''
    cards.append(f"""
  <section class="card">
    <h2>item {idx+1} <span class="ep">/ {op} · {ep} · {cell}</span>{flag}</h2>
    <div class="row">
      <div class="democol">{demo_fig}</div>
      <div class="cands">{cand}</div>
    </div>
  </section>""")

json.dump(key, open(VDIR / "key.json", "w"), indent=1)
seed_btns = "".join(f'<button data-seed="{s}"{" class=on" if s == 42 else ""}>seed {s}</button>' for s in SEEDS)
blocks = "".join(cards)

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>DCG-CN · blinded SxS confirmation</title>
<style>
 :root{{--bg:#0e0f13;--fg:#e6e6e6;--mut:#9aa0a6;--card:#15171d;--line:#23262f;--demo:#ffd166;--flag:#ff9e9e}}
 body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:22px}}
 h1{{margin:0 0 4px}} .sub{{color:var(--mut);max-width:1000px}}
 .legend{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:14px 0;max-width:1000px;font-size:13px}}
 .bar{{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0 12px;border-bottom:1px solid var(--line)}}
 .bar button{{background:#1a1c22;color:var(--fg);border:1px solid #2a2d36;border-radius:6px;padding:6px 12px;margin-right:8px;cursor:pointer;font:inherit}}
 .bar button.on{{background:#2b3550;border-color:#3f5088;color:#cfe0ff}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:16px 0}}
 h2{{margin:0 0 10px;font-size:16px}} h2 .ep{{color:var(--mut);font-weight:normal}} .flag{{color:var(--flag);font-size:12px}}
 .row{{display:grid;grid-template-columns:240px 1fr;gap:16px;align-items:start}}
 figure{{margin:0}} video{{width:100%;border-radius:5px;background:#000;aspect-ratio:3/4;object-fit:cover;display:block}}
 figcaption{{font-size:11px;color:#8b919b;text-align:center;margin-top:3px}}
 .demo figcaption{{color:var(--demo)}} .demo video{{outline:2px solid #6b5a2f}}
 .cands{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
 .missing{{aspect-ratio:3/4;background:#191b21;border:1px dashed #33363f;border-radius:5px;display:flex;align-items:center;justify-content:center;color:#5b6270}}
 @media (max-width:820px){{.row{{grid-template-columns:1fr}} .cands{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<h1>DCG-CN — blinded side-by-side confirmation (owner)</h1>
<div class="sub">Champion <b>ctt_v2 / store/runs/002</b> step 10000, NEUTRAL prompt. The automated
DINO instrument returned <b>KEEP @ w=3</b> (matched-vs-swapped attribution 12/12; robust to excluding
every guard-flagged clip, 9/9). This bundle is the human confirmation of perceptual quality.</div>
<div class="legend"><b>Your task, per item.</b> The yellow clip is the <b>demonstrated effect</b>
(a reference video of the transition). Among the four <b>blinded candidates</b>, which one(s)
<i>perform that same transition class</i> on the item's own endpoints — <i>without</i> visible quality
collapse (noise/smearing) or verbatim copying of the demo's content? One candidate is matched-DCG
(true demo amplified), one is swapped-DCG (a wrong-class demo amplified), one is the no-DCG baseline,
one is endpoint-only (no demo). If the method works, matched-DCG should read as the demonstrated class
while swapped drifts to a different manner. Order is randomized per item; the key is withheld
(<code>key.json</code>). Hover to play; ⚑ marks the 3 guard-flagged items (judge, then check they don't
carry the verdict). Seed selector below.</div>
<div class="bar">{seed_btns} <span class="sub" style="margin-left:10px">12 items · demo + 4 blinded candidates · seeds 42/43/44</span></div>
{blocks}
<script>
 let seed = 42;
 const io = new IntersectionObserver(es=>es.forEach(e=>{{
   const v=e.target;
   if(e.isIntersecting){{
     let s = v.dataset.op ? `gens/${{v.dataset.op}}__s${{seed}}__${{v.dataset.arm}}.mp4` : (v.dataset.src||"");
     if(v.getAttribute('src')!==s){{ v.setAttribute('src', s); v.load(); }}
     v.play().catch(()=>{{}});
   }} else v.pause();
 }}), {{rootMargin:"300px"}});
 const observeAll=()=>document.querySelectorAll('video').forEach(v=>io.observe(v));
 document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{{
   seed=+b.dataset.seed;
   document.querySelectorAll('.bar button').forEach(x=>x.classList.toggle('on',x===b));
   document.querySelectorAll('video[data-op]').forEach(v=>{{ v.pause(); v.removeAttribute('src'); v.load(); }});
   io.disconnect(); observeAll();
 }});
 observeAll();
</script>
</body></html>"""
(VDIR / "index.html").write_text(html)
print(f"[gen] wrote {VDIR/'index.html'} — {len(ITEMS)} items, blinded quads, key.json withheld")
print(f"[gen] symlinks gens->{os.readlink(VDIR/'gens')} corpus->{os.readlink(VDIR/'corpus')}")
