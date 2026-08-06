#!/usr/bin/env python3
"""Generate the slide-pack viewer page (relative media/ paths only)."""
import os, html, re

ROOT = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research"
PACK = os.path.join(ROOT, "outputs/presentation/slide_pack_2026-08-05")
OUT  = os.path.join(ROOT, "outputs/viewers/slide_pack_2026_08_05/index.html")

SECTIONS = [
    ("02_coldopen", "02 · Cold open — shadow_smoke_0 ← ref shadow_smoke_1 (memo-probe, s42)",
     ["start", "end", "reference", "ctt_v2"]),
    ("03_lerp", "03 · Lerp collapse — base model, endpoints + prompt, no working reference",
     ["dissolve_1", "dissolve_2"]),
    ("04_refvfx", "04 · refVFX vs ctt_v2 — one row (tennis→snowboard ← firelava_0), A = effect not named, B = effect named",
     ["refvfx_promptA", "refvfx_promptB", "cttv2_promptA", "cttv2_promptB"]),
    ("06_dataset", "06 · Dataset strata — 4 samples each of S0 / S1 / S2 / S4",
     [f"s{k}_{i}" for k in (0, 1, 2, 4) for i in (1, 2, 3, 4)]),
    ("07_counterfactual", "07 · Counterfactual 3×3 — opA animalization · opB shadow_smoke · opC polygon × ep1 bench-man · ep2 cook · ep3 pottery",
     [f"op{o}_ep{e}" for o in "ABC" for e in (1, 2, 3)]),
]

ROWS_IID = [
    ("row1", "davis_tennis_snowboard ← shadow_smoke_0 · unseen-foreign · two-sided"),
    ("row2", "davis_lucia ← animalization_0 · unseen-foreign · one-sided"),
    ("row3", "davis_mallard_water ← super_fast_run_0 · unseen-foreign · one-sided"),
    ("row4", "gas_transformation_6 ← earth_element_4 · unseen-cross · one-sided"),
    ("row5", "davis_lucia ← earth_element_4 · unseen-foreign · one-sided (cttv2 s43)"),
    ("row6", "earth_element_6 ← earth_element_4 · unseen-SAME · one-sided"),
]
ROWS_ZS = [
    ("row1", "davis_tennis_snowboard ← firelava_0 · zs-foreign · two-sided"),
    ("row2", "shadow_smoke_7 ← firelava_0 · zs-cross · two-sided (cttv2 s43)"),
    ("row3", "davis_lucia ← saint_glow_0 · zs-foreign · one-sided"),
    ("row4", "davis_tennis_snowboard ← display_transition_1 · zs-foreign · two-sided"),
    ("row5", "davis_tennis_snowboard ← raven_transition_0 · zs-foreign · two-sided (cttv2 s43)"),
    ("row6", "hero_flight_5 ← display_transition_1 · zs-cross · two-sided"),
    ("row7", "money_rain_3 ← live_concert_1 · zs-cross · one-sided"),
]
COLS = ["start", "end", "reference", "base", "refvfx", "cttv2"]

def card(rel, label):
    return (f'<figure><video preload="metadata" muted loop playsinline '
            f'src="{html.escape(rel)}"></video>'
            f'<figcaption>{html.escape(label)}</figcaption></figure>')

parts = []
for folder, title, names in SECTIONS:
    cards = []
    for n in names:
        p = os.path.join(PACK, folder, f"{folder}__{n}.mp4")
        if os.path.exists(p):
            cards.append(card(f"media/{folder}/{folder}__{n}.mp4", n))
    parts.append(f'<section><h2>{html.escape(title)}</h2><div class="grid">{"".join(cards)}</div></section>')

for folder, title, rows in [("12_iid", "12 · IID generalization — unseen reference classes", ROWS_IID),
                            ("13_zeroshot", "13 · Zero-shot reference classes", ROWS_ZS)]:
    blocks = []
    for row, desc in rows:
        cards = []
        for c in COLS:
            p = os.path.join(PACK, folder, f"{folder}__{row}_{c}.mp4")
            if os.path.exists(p):
                cards.append(card(f"media/{folder}/{folder}__{row}_{c}.mp4", c))
        blocks.append(f'<h3>{row} — {html.escape(desc)}</h3><div class="grid">{"".join(cards)}</div>')
    parts.append(f'<section><h2>{html.escape(title)}</h2>{"".join(blocks)}</section>')

page = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slide pack 2026-08-05 — deck media</title>
<style>
body{margin:0;background:#101418;color:#dde3ea;font:14px/1.5 system-ui,sans-serif;padding:24px}
h1{font-size:20px;margin:0 0 4px}
p.sub{color:#8b96a5;margin:0 0 24px}
h2{font-size:16px;margin:36px 0 10px;border-bottom:1px solid #2a3038;padding-bottom:6px}
h3{font-size:13px;color:#aab4c0;margin:18px 0 8px;font-weight:600}
.grid{display:flex;flex-wrap:wrap;gap:12px}
figure{margin:0;width:180px}
figure video{width:180px;border-radius:6px;background:#000;display:block;cursor:pointer}
figcaption{font-size:12px;color:#8b96a5;margin-top:4px;text-align:center;word-break:break-all}
</style></head><body>
<h1>Slide pack 2026-08-05 — deck media</h1>
<p class="sub">Mirrors the upload zip (outputs/presentation/slide_pack_2026-08-05). Videos play while visible; click one to restart it. Provenance and prompt strings: MANIFEST.md next to the media.</p>
""" + "\n".join(parts) + """
<script>
const io=new IntersectionObserver(es=>es.forEach(e=>{const v=e.target;
  if(e.isIntersecting){v.play().catch(()=>{});}else{v.pause();}}),{rootMargin:"100px"});
document.querySelectorAll("video").forEach(v=>{io.observe(v);
  v.addEventListener("click",()=>{v.currentTime=0;v.play().catch(()=>{});});});
</script>
</body></html>"""

with open(OUT, "w") as f:
    f.write(page)
n = page.count("<video")
print(f"wrote {OUT} ({n} videos)")
