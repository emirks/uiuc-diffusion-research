#!/usr/bin/env python
"""Regenerate the dino_a2_sample gallery (raw A2-tokens sample clips, zero-shot + unseen).
Each card shows the DEMO REFERENCE fed to the model (the signal source) beside the
generation, so you can read whether the gen follows the reference's transition.
Scans the viewer's media/ symlink and writes index.html with relative paths only.
Run with the ltx2 venv python (system python3 on the login node is 3.6)."""
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VD = REPO / "outputs/viewers/dino_a2_sample"
MEDIA = VD / "media"
STD = REPO / "data/processed/transitions_std121"      # demo-reference tree (<class>/<name>.mp4)

# refs/ symlink -> std121 tree, relative so it survives restarts/repo moves (viewer rule).
REFS_LINK = VD / "refs"
REFS_TARGET = os.path.relpath(STD, VD)                 # ../../../data/processed/transitions_std121
if REFS_LINK.is_symlink() or REFS_LINK.exists():
    if not (REFS_LINK.is_symlink() and os.readlink(REFS_LINK) == REFS_TARGET):
        REFS_LINK.unlink()
        os.symlink(REFS_TARGET, REFS_LINK)
else:
    os.symlink(REFS_TARGET, REFS_LINK)

# name -> "<class>/<name>.mp4" (stems are unique across the tree; asserted below)
REF_REL = {p.stem: f"{p.parent.name}/{p.name}" for p in STD.rglob("*.mp4")}

# <cell>__<arm>__<endpoint>__ref_<ref>__s<seed>.mp4
PAT = re.compile(r"^(?P<cell>[^_].*?)__(?P<arm>.+?)__(?P<endpoint>.+?)__ref_(?P<ref>.+?)__s(?P<seed>\d+)\.mp4$")
CELL_LABEL = {"G-unseen-same": "Unseen endpoints (same operator class)",
              "G-zs-cross": "Zero-shot (cross operator class)"}

clips = []
missing = []
for f in sorted(MEDIA.glob("*.mp4")):
    m = PAT.match(f.name)
    if m:
        c = m.groupdict() | {"file": f.name}
        c["ref_rel"] = REF_REL.get(c["ref"])
        if c["ref_rel"] is None:
            missing.append(c["ref"])
        clips.append(c)
if missing:
    raise SystemExit(f"reference clips not found in {STD}: {sorted(set(missing))}")

by_cell = {}
for c in clips:
    by_cell.setdefault(c["cell"], []).append(c)

def card(c):
    return f'''<figure class="card">
  <div class="pair">
    <div class="v"><span class="lab lab-ref">reference · signal source</span>
      <video src="refs/{c['ref_rel']}" controls loop muted preload="metadata" playsinline></video></div>
    <div class="v"><span class="lab lab-gen">generation</span>
      <video src="media/{c['file']}" controls loop muted preload="metadata" playsinline></video></div>
  </div>
  <figcaption><b>{c['endpoint']}</b><span class="ref">signal ▸ {c['ref']}</span><span class="seed">s{c['seed']}</span></figcaption>
</figure>'''

sections = []
for cell in sorted(by_cell):
    cards = "\n".join(card(c) for c in by_cell[cell])
    sections.append(f'<section><h2>{CELL_LABEL.get(cell, cell)} <span class="tag">{cell} · {len(by_cell[cell])}</span></h2><div class="grid">{cards}</div></section>')

html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DINO-signal A2_tokens — sample gens</title>
<style>
:root{{color-scheme:dark light}}
body{{margin:0;background:#0e1116;color:#e6edf3;font:14px/1.5 system-ui,sans-serif}}
header{{padding:18px 22px;border-bottom:1px solid #222b36;position:sticky;top:0;background:#0e1116e8;backdrop-filter:blur(6px)}}
h1{{margin:0 0 4px;font-size:18px}} .sub{{color:#9fb0c3;font-size:12.5px}}
section{{padding:14px 22px 8px}} h2{{font-size:15px;margin:10px 0;color:#cdd9e5}}
.tag{{font:11px ui-monospace,monospace;color:#8aa0b6;background:#1b232e;padding:2px 7px;border-radius:10px;margin-left:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}}
.card{{margin:0;background:#161b22;border:1px solid #232d38;border-radius:8px;overflow:hidden}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:#232d38}}
.v{{position:relative;background:#000}}
.lab{{position:absolute;top:0;left:0;z-index:1;font:10px ui-monospace,monospace;padding:2px 6px;border-bottom-right-radius:6px;letter-spacing:.02em}}
.lab-ref{{background:#3a2d1b;color:#e0b76c}} .lab-gen{{background:#183028;color:#6cc4a1}}
video{{width:100%;display:block;background:#000;aspect-ratio:4/3}}
figcaption{{padding:7px 9px;font-size:12px;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}}
figcaption b{{color:#e6edf3}} .ref{{color:#6cc4a1;font:11px ui-monospace,monospace}} .seed{{color:#7d8ea0;margin-left:auto;font:11px ui-monospace,monospace}}
</style></head><body>
<header><h1>DINO-signal · A2_tokens (neutral, matched signal) — reference ▸ generation</h1>
<div class="sub">{len(clips)} clips · seed 42 · generalization cells only (zero-shot + unseen) · staged 10k adapter. Left = demo reference fed to the model (also the matched signal source); right = generation. Plumbing sample — NOT scored; matched signal only (no controls/baseline yet).</div></header>
{''.join(sections)}
</body></html>'''

(VD / "index.html").write_text(html)
print(f"wrote {VD/'index.html'} with {len(clips)} clips across {len(by_cell)} cells:",
      {k: len(v) for k, v in by_cell.items()})
print(f"refs symlink: {REFS_LINK} -> {os.readlink(REFS_LINK)}")
