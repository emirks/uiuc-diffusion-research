#!/usr/bin/env python3
"""Generate the bneck_redesign HRC-residual results viewer (matched vs deranged-code, side by side).

Regeneratable: `python3 scripts/viewers/gen_bneck_hrc.py`. Media arrives via the `matched/` and
`deranged/` symlinks in the viewer dir (relative paths only — the rule that keeps viewers alive).
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # repo root
VDIR = HERE / "outputs/viewers/bneck_redesign_hrc"
MATCHED = VDIR / "matched"

# claim cells first, then the rest
CELL_ORDER = ["G-unseen-same", "G-unseen-cross", "G-unseen-foreign", "G-fit", "G-memo-probe",
              "G-ref-control", "G-zs-same", "G-zs-cross", "G-zs-foreign", "G-unseen-cross-foreign"]
FN = re.compile(r"^(?P<cell>[^_]+(?:-[^_]+)*)__ctt_v2__(?P<tgt>.+?)__ref_(?P<donor>.+?)__s(?P<seed>\d+)\.mp4$")

rows = []
for p in sorted(MATCHED.glob("*.mp4")):
    m = FN.match(p.name)
    if not m:
        continue
    if not (VDIR / "deranged" / p.name).exists():
        continue
    rows.append({**m.groupdict(), "file": p.name})

# one row per (cell,tgt,donor) using seed 42 (fall back to any seed)
by_key = {}
for r in rows:
    key = (r["cell"], r["tgt"], r["donor"])
    if key not in by_key or r["seed"] == "42":
        by_key[key] = r
uniq = list(by_key.values())
uniq.sort(key=lambda r: (CELL_ORDER.index(r["cell"]) if r["cell"] in CELL_ORDER else 99, r["donor"], r["tgt"]))

def cell_block(cell, items):
    cards = []
    for r in items:
        f = r["file"]
        cards.append(f"""
      <div class="card">
        <div class="cap"><b>{r['donor']}</b> → target <span class="tgt">{r['tgt']}</span> <span class="seed">s{r['seed']}</span></div>
        <div class="pair">
          <figure><video src="matched/{f}" muted loop playsinline preload="metadata" onmouseover="this.play()" onmouseout="this.pause()"></video><figcaption>matched code</figcaption></figure>
          <figure><video src="deranged/{f}" muted loop playsinline preload="metadata" onmouseover="this.play()" onmouseout="this.pause()"></video><figcaption>deranged code</figcaption></figure>
        </div>
      </div>""")
    claim = " claim-cell" if cell in ("G-unseen-same", "G-unseen-cross") else ""
    return f'<section class="cell{claim}"><h2>{cell}{" &nbsp;·&nbsp; CLAIM CELL" if claim else ""} <span class="n">({len(items)})</span></h2><div class="grid">{"".join(cards)}</div></section>'

cells = {}
for r in uniq:
    cells.setdefault(r["cell"], []).append(r)
blocks = "".join(cell_block(c, cells[c]) for c in sorted(cells, key=lambda c: CELL_ORDER.index(c) if c in CELL_ORDER else 99))

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>bneck_redesign · HRC-residual (Idea 2)</title>
<style>
 body{{background:#0e0f13;color:#e6e6e6;font:14px/1.4 system-ui,sans-serif;margin:0;padding:24px}}
 h1{{margin:0 0 4px}} .sub{{color:#9aa0a6;margin-bottom:16px}}
 .verdict{{background:#1a1c22;border:1px solid #2a2d36;border-radius:10px;padding:14px 18px;margin:0 0 22px;max-width:900px}}
 .verdict b{{color:#ff7b72}} .verdict .k{{color:#7ee787}}
 h2{{margin:26px 0 10px;font-size:16px;border-bottom:1px solid #22252d;padding-bottom:6px}}
 .claim-cell h2{{color:#ffd166}} .n{{color:#6b7280;font-weight:normal}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}}
 .card{{background:#15171d;border:1px solid #23262f;border-radius:8px;padding:8px}}
 .cap{{font-size:12px;color:#c9ced6;margin-bottom:6px}} .tgt{{color:#9aa0a6}} .seed{{color:#5b6270;float:right}}
 .pair{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
 figure{{margin:0}} video{{width:100%;border-radius:5px;background:#000;aspect-ratio:16/10;object-fit:cover}}
 figcaption{{font-size:11px;color:#8b919b;text-align:center;margin-top:3px}}
 figure:first-child figcaption{{color:#7ee787}} figure:last-child figcaption{{color:#ff9e9e}}
</style></head><body>
<h1>bneck_redesign — HRC-residual arm (Idea 2)</h1>
<div class="sub">Hover a clip to play. Left = <b style="color:#7ee787">matched</b> operator code; right = <b style="color:#ff9e9e">deranged</b> (different operator's code, same target). {len(uniq)} rows · seed 42.</div>
<div class="verdict">
 <b>RESULT: NULL — the generator does not read which transition the code specifies.</b><br>
 Paired read: G-unseen-same <span class="k">6/13</span> (bar ≥9), G-unseen-cross <span class="k">4/13</span> (bar ≥8), pooled Δapp_ref <span class="k">−0.003</span> (CI incl. 0). Liveness 0.578 (live but not class-directed). Temporal r 0.82 (no motion read).<br>
 <span style="color:#9aa0a6">What to look for: if the code were read, the <b style="color:#7ee787">matched</b> clip would show the donor's transition manner and the <b style="color:#ff9e9e">deranged</b> a different one. They mostly look alike — that's the null, qualitatively.</span>
</div>
{blocks}
</body></html>"""
(VDIR / "index.html").write_text(html)
print(f"[gen] wrote {VDIR/'index.html'} — {len(uniq)} paired rows across {len(cells)} cells")
