#!/usr/bin/env python3
"""Build the flowsig Phase-A pilot page: matched vs deranged program, side by side.

Store-first: clips are read from the registered gen subentries under
``store/gens/<GENID>/<KK_variant>__dai/videos/`` and the demos/GT from
``data/processed/transitions_std121``. Nothing is read from a campaign scratch tree, so the
page cannot outlive the artifacts it shows.

Layout: one card per grid row (13 rows = the G-fit cell, one per operator class). Inside a
card, one column per condition and one video row per seed, so the matched/deranged pair is
horizontally adjacent -- the comparison the page exists to make.

  python3 scripts/viewers/gen_flowsig_pilot.py [--genid 022_flowsig_ball] [--slug flowsig_pilot]
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STD = REPO / "data/processed/transitions_std121"
REG = REPO / ("misc/2026-08-24_flow_signal_conditioning/step2eval/gen/"
              "registry_flowsig_ball_pilot.jsonl")

COLS = [
    ("gt", "GROUND TRUTH", "the row's own target transition (never shown to the model)"),
    ("demo_m", "DEMO — matched", "the clip the MATCHED program is computed from"),
    ("demo_d", "DEMO — deranged", "the clip the DERANGED program is computed from (wrong operator)"),
    ("02_pilot_code_matched", "code_only · MATCHED", "program only, no pixel demo"),
    ("03_pilot_code_deranged", "code_only · DERANGED", "wrong-operator program, no pixel demo"),
    ("04_pilot_both_matched", "both · MATCHED", "pixel demo + matched program"),
    ("05_pilot_null", "null", "no program, no pixel demo (endpoints + neutral text only)"),
]
VARIANTS = [c[0] for c in COLS if c[0].startswith("0")]


def find_clip(name: str) -> Path | None:
    hits = list(STD.rglob(f"{name}.mp4"))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genid", default="022_flowsig_ball")
    ap.add_argument("--slug", default="flowsig_pilot")
    ap.add_argument("--seeds", default="42,43")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    gens = REPO / "store/gens" / args.genid
    vdir = REPO / "outputs/viewers" / args.slug
    (vdir / "media").mkdir(parents=True, exist_ok=True)
    # relative symlinks only -- every href in the page is relative to the viewer directory,
    # which is the one rule that keeps viewers alive across restarts and repo moves.
    for link, target in (("media/std121", STD), ("media/gens", gens)):
        p = vdir / link
        if p.is_symlink() or p.exists():
            p.unlink()
        # depth is counted from the link's PARENT directory, not from the link path -- getting
        # this wrong yields a symlink that resolves nowhere and a page of black boxes.
        ups = len(p.parent.relative_to(REPO).parts)
        p.symlink_to(Path("../" * ups) / target.relative_to(REPO))

    rows = [json.loads(l) for l in REG.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["endpoint_class"])

    have = {v: len(list((gens / f"{v}__dai/videos").glob("*.mp4")))
            if (gens / f"{v}__dai/videos").is_dir() else 0 for v in VARIANTS}

    def src(col: str, row: dict, seed: int) -> str | None:
        if col == "gt":
            c = find_clip(row["endpoint"])
            return f"media/std121/{c.parent.name}/{c.name}" if c else None
        if col in ("demo_m", "demo_d"):
            name = row["program_source"] if col == "demo_m" else row["program_source_deranged"]
            c = find_clip(name)
            return f"media/std121/{c.parent.name}/{c.name}" if c else None
        f = gens / f"{col}__dai/videos" / f"{row['item_id']}__s{seed}.mp4"
        return f"media/gens/{col}__dai/videos/{f.name}" if f.exists() else None

    cards = []
    for row in rows:
        cells = []
        for col, label, blurb in COLS:
            vids = []
            for seed in (seeds if col.startswith("0") else seeds[:1]):
                s = src(col, row, seed)
                tag = (f'<video src="{s}" muted loop playsinline preload="none" '
                       f'controls></video>' if s
                       else '<div class="missing">not generated</div>')
                cap = f"seed {seed}" if col.startswith("0") else "reference clip"
                vids.append(f'<figure>{tag}<figcaption>{cap}</figcaption></figure>')
            kind = ("in" if col in ("demo_m", "demo_d") else
                    "gt" if col == "gt" else "out")
            cells.append(f'<div class="col {kind}"><h4>{html.escape(label)}</h4>'
                         f'<p class="blurb">{html.escape(blurb)}</p>{"".join(vids)}</div>')
        cards.append(
            f'<section class="card">'
            f'<header><h3>{html.escape(row["endpoint_class"])}</h3>'
            f'<span class="meta">target <code>{html.escape(row["endpoint"])}</code> · '
            f'matched demo <code>{html.escape(row["program_source"])}</code> · '
            f'deranged demo <code>{html.escape(row["program_source_deranged"])}</code> '
            f'({html.escape(row["deranged_class"])})</span></header>'
            f'<div class="grid">{"".join(cells)}</div></section>')

    counts = " · ".join(f"{v.split('_', 2)[2]} {have[v]}" for v in VARIANTS)
    page = f"""<!doctype html><meta charset="utf-8">
<title>flowsig b_all — matched vs deranged program (Phase-A pilot)</title>
<style>
:root{{--bg:#0f1115;--fg:#e6e8eb;--mut:#8b94a3;--line:#242833;--in:#2a3b52;--out:#33304a;--gt:#1f3a2f}}
*{{box-sizing:border-box}}
body{{margin:0;padding:24px;background:var(--bg);color:var(--fg);
 font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;font-weight:600;color:var(--mut);margin:0 0 18px}}
.note{{border:1px solid var(--line);border-left:3px solid #6b7a99;border-radius:6px;
 padding:12px 14px;margin:0 0 22px;color:#c3c9d4;max-width:1100px}}
.note b{{color:var(--fg)}}
.card{{border:1px solid var(--line);border-radius:8px;margin:0 0 20px;overflow:hidden}}
.card header{{display:flex;gap:14px;align-items:baseline;padding:10px 14px;background:#161a22;
 border-bottom:1px solid var(--line);flex-wrap:wrap}}
.card h3{{margin:0;font-size:15px}} .meta{{color:var(--mut);font-size:12px}}
code{{background:#1c212b;padding:1px 5px;border-radius:3px;font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat({len(COLS)},minmax(0,1fr));gap:10px;padding:12px}}
.col{{border:1px solid var(--line);border-radius:6px;padding:8px;min-width:0}}
.col.in{{background:var(--in)}} .col.out{{background:var(--out)}} .col.gt{{background:var(--gt)}}
.col h4{{margin:0 0 2px;font-size:12px;letter-spacing:.02em}}
.blurb{{margin:0 0 8px;font-size:11px;color:#aab2c0;min-height:28px}}
figure{{margin:0 0 8px}} video{{width:100%;border-radius:4px;background:#000;display:block}}
figcaption{{font-size:11px;color:var(--mut);padding-top:2px}}
.missing{{aspect-ratio:3/4;display:grid;place-items:center;background:#12151c;border:1px dashed #333;
 border-radius:4px;color:#5a6273;font-size:11px}}
.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5;border-bottom:1px solid var(--line);
 margin-bottom:16px}}
button{{background:#232838;color:var(--fg);border:1px solid var(--line);border-radius:5px;
 padding:6px 12px;cursor:pointer;font-size:13px}}
button:hover{{background:#2c3244}}
</style>
<h1>flowsig arm <code>b_all</code> — does the model follow the fed transition program?</h1>
<h2>Phase-A qualitative pilot · 13 G-fit rows (one per operator class) · checkpoint 10000 · neutral prompt</h2>
<div class="note">
<b>How to read a card.</b> Left three panels are INPUTS, not outputs: the ground-truth target
transition, the clip the <i>matched</i> program was computed from, and the clip the
<i>deranged</i> program was computed from. The right four panels are what the model generated
from the same endpoints, the same neutral caption and the same seed — differing only in what
conditioning it received.
<b>The comparison this page exists for is columns 4 vs 5:</b> the only difference between them is
which clip the 18-channel appearance-free program came from. If the program is being followed,
the deranged column should perform the <i>wrong</i> operator; if it is ignored, the two columns
should look alike. <b>Column 7 (null)</b> is the no-program floor.
<br><br><b>Caveat that belongs on every code_only column.</b> The trained model carries a known
recipe defect (<code>textdrop-coupled</code>): during training the text-dropout draw was
rank-coupled to the conditioning-cell draw, so text was dropped on exactly the <i>both</i> cell
and nowhere else. The model therefore <b>never saw code_only with the text absent</b> — the
program was never the sole operator description in context at any training step. Every clip here
uses the standard neutral caption, so the code_only columns are in the <i>trained</i>
(text-present) configuration, not an out-of-distribution one. The consequence for reading:
a positive matched-vs-deranged difference gains force (the program would be doing work despite
the caption also being available), while an all-alike outcome is non-diagnostic — it cannot
separate "the model can't read the program" from "the model was never given a reason to".
<br><br><b>Status:</b> {counts}. This is a qualitative pilot — no statistic, bar, or verdict is
attached to it; the reading tests were deferred by the owner pending this page.
</div>
<div class="bar"><button id="pa">▶ play all</button> <button id="ps">⏸ pause all</button></div>
{''.join(cards)}
<script>
const vs = () => Array.from(document.querySelectorAll('video'));
document.getElementById('pa').onclick = () => vs().forEach(v => {{ v.load(); v.play().catch(()=>{{}}); }});
document.getElementById('ps').onclick = () => vs().forEach(v => v.pause());
// autoplay a clip only while it is on screen -- 90+ videos otherwise stall the page
const io = new IntersectionObserver(es => es.forEach(e => {{
  if (e.isIntersecting) {{ e.target.play().catch(()=>{{}}); }} else {{ e.target.pause(); }}
}}), {{threshold: .25}});
vs().forEach(v => io.observe(v));
</script>
"""
    (vdir / "index.html").write_text(page)
    print(f"[viewer] {vdir/'index.html'}  rows={len(rows)} clips per variant: {have}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
