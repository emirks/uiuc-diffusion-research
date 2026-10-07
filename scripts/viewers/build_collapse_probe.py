#!/usr/bin/env python3
"""Build the `collapse_probe` viewer: R1 / R2 / R3 generations of the from-scratch collapse probe
(misc/2026-09-08_collapse_probe) side by side, per prompt and seed, with the three-part prompt and the
confinement residual of every clip.

Reads (derived only, nothing hand-kept):
  misc/2026-09-08_collapse_probe/prompts/prompts.jsonl
  misc/2026-09-08_collapse_probe/results/per_clip.csv   (path, DR_med, M, realized_dino per clip)
  misc/2026-09-08_collapse_probe/results/paired.csv
Writes outputs/viewers/collapse_probe/index.html. Media reach the page through the `media` symlink
(-> misc/.../out) and the `results` symlink (-> misc/.../results); every path in the page is relative.

    source $LAB/envs-aarch64/activate && python scripts/viewers/build_collapse_probe.py
"""
from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "misc/2026-09-08_collapse_probe"
VIEW = REPO / "outputs/viewers/collapse_probe"
RUN_LABEL = {"R1": "R1 · start anchor + full prompt", "R2": "R2 · both anchors + full prompt",
             "R3": "R3 · both anchors · captions only"}


def link(name: str, target: Path) -> None:
    dst = VIEW / name
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.readlink(dst) == os.path.relpath(target, VIEW):
            return
        dst.unlink()
    dst.symlink_to(os.path.relpath(target, VIEW))


def main() -> None:
    VIEW.mkdir(parents=True, exist_ok=True)
    link("media", CAMP / "out")
    link("results", CAMP / "results")

    prompts = [json.loads(l) for l in open(CAMP / "prompts/prompts.jsonl")]
    clips = list(csv.DictReader(open(CAMP / "results/per_clip.csv")))
    paired = list(csv.DictReader(open(CAMP / "results/paired.csv")))

    def media_path(p: str) -> str:
        # per_clip.path is repo-relative: misc/.../out/r2/base_cond_neutral/x.mp4 -> media/r2/base_cond_neutral/x.mp4
        i = p.find("/out/")
        return "media/" + p[i + 5:] if i >= 0 else p

    by_key = {}
    for c in clips:
        if c.get("decode_ok", "True") in ("False", "0"):
            continue
        by_key[(c["run"], c["prompt_id"], str(int(float(c["seed"]))))] = dict(
            src=media_path(c["path"]), DR=float(c["DR_med"]), M=float(c["M"]) if c["M"] else None,
            dino=float(c["realized_dino"]) if c.get("realized_dino") not in (None, "", "nan") else None)
    pmap = {(p["prompt_id"], str(int(float(p["seed"])))): p for p in paired}
    seeds = sorted({k[2] for k in by_key})

    data = []
    for p in prompts:
        pid = p["prompt_id"]
        rows = []
        for s in seeds:
            cells = {r: by_key.get((r, pid, s)) for r in ("R1", "R2", "R3")}
            if not any(cells.values()):
                continue
            pr = pmap.get((pid, s), {})
            rows.append(dict(seed=s, cells=cells,
                             d21=float(pr["dDR_R2_minus_R1"]) if pr.get("dDR_R2_minus_R1") else None,
                             d32=float(pr["dDR_R3_minus_R2"]) if pr.get("dDR_R3_minus_R2") else None,
                             d31=float(pr["dDR_R3_minus_R1"]) if pr.get("dDR_R3_minus_R1") else None,
                             dino=float(pr["realized_dino"]) if pr.get("realized_dino") not in (None, "", "nan") else None))
        data.append(dict(prompt_id=pid, tier=p["tier"], endpoint=p["endpoint"], endpoint_class=p["endpoint_class"],
                         start=p["start_caption"], change=p["change_clause"], end=p["end_caption"],
                         mechanism=p.get("mechanism_family", ""), rows=rows))

    # summary numbers for the header (medians per run x tier)
    import statistics as st
    def med(run, tier):
        v = [c["DR"] for (r, pid, s), c in by_key.items() if r == run and next(p for p in prompts if p["prompt_id"] == pid)["tier"] == tier]
        return st.median(v) if v else float("nan")
    summary = {t: {r: med(r, t) for r in ("R1", "R2", "R3")} for t in ("high", "low")}
    n_neg = sum(1 for p in paired if p["tier"] == "high" and float(p["dDR_R3_minus_R2"]) < 0)
    n_hi = sum(1 for p in paired if p["tier"] == "high")

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>collapse probe — R1 / R2 / R3</title>
<style>
:root{{--bg:#0f1115;--card:#171a21;--ink:#e6e6e6;--mut:#9aa3ad;--acc:#f0b429;--red:#e5533d;--blue:#4c8dff;--grn:#3bb273}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}}
header{{padding:16px 22px 8px;border-bottom:1px solid #262a33}} h1{{margin:0 0 6px;font-size:20px}}
.sub{{color:var(--mut);max-width:1100px}} .sub b{{color:var(--ink)}}
.bar{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;padding:10px 22px;border-bottom:1px solid #262a33;position:sticky;top:0;background:var(--bg);z-index:5}}
.bar label{{color:var(--mut)}} .bar select,.bar button{{background:#1d2129;color:var(--ink);border:1px solid #333a46;border-radius:6px;padding:4px 8px}}
.bar button.on{{border-color:var(--acc);color:var(--acc)}}
.summary{{display:flex;gap:18px;flex-wrap:wrap;padding:8px 22px;color:var(--mut)}} .summary span b{{color:var(--ink)}}
.figs{{display:flex;gap:12px;flex-wrap:wrap;padding:6px 22px 12px}} .figs img{{max-height:280px;border:1px solid #262a33;border-radius:6px}}
details.fig{{padding:0 22px 8px}} details.fig summary{{cursor:pointer;color:var(--mut)}}
main{{padding:10px 22px 40px;display:grid;gap:14px}}
.card{{background:var(--card);border:1px solid #262a33;border-radius:10px;padding:12px 14px}}
.card.hidden{{display:none}}
.head{{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}}
.pid{{font-weight:700;color:var(--acc)}} .tier{{font-size:11px;padding:1px 7px;border-radius:10px;border:1px solid}}
.tier.high{{color:var(--red);border-color:var(--red)}} .tier.low{{color:var(--blue);border-color:var(--blue)}}
.ep{{color:var(--mut);font-size:12px}}
.prompt{{margin:4px 0 10px;color:var(--mut)}} .prompt .ch{{color:var(--ink);background:#2a2412;border-left:3px solid var(--acc);padding:1px 6px;border-radius:3px}}
.prompt .tag{{font-size:11px;color:var(--acc);margin-left:6px}}
.seedrow{{display:grid;grid-template-columns:70px repeat(3,1fr);gap:10px;align-items:start;margin-top:8px}}
.seedlab{{color:var(--mut);font-size:12px;padding-top:6px}} .seedlab b{{display:block;color:var(--ink)}}
.cell video{{width:100%;aspect-ratio:4/3;background:#000;border-radius:6px;display:block}}
.cell .cap{{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);margin-top:4px}}
.cell .cap b{{color:var(--ink)}} .dr{{font-variant-numeric:tabular-nums}} .dr.low{{color:var(--red);font-weight:700}}
.deltas{{font-size:12px;color:var(--mut);margin-top:4px}} .deltas b{{color:var(--ink)}}
.neg{{color:var(--red)}} .pos{{color:var(--grn)}}
</style></head><body>
<header><h1>Collapse probe — same start clip, same end anchor, same seed; only the anchor set and the text change</h1>
<div class="sub">Base LTX-2, no adapter. <b>R1</b>: start anchor + full prompt (start caption + <span style="color:var(--acc)">change clause</span> + end caption).
<b>R2</b>: start anchor + end anchor cut from R1's own last 9 frames, same full prompt. <b>R3</b>: same anchors, captions only (the change clause is dropped).
DR = median residual of the interior frames to the line between the two anchor frames (0 = a frame-wise blend of the anchors; the interpolation family). M = share of frames mid-segment (dissolve 0.5, cut/freeze 0).
30 scene-change prompts + 10 in-place controls on 30 start clips from in-place classes, seeds {', '.join(seeds)}. Campaign: <code>misc/2026-09-08_collapse_probe</code> (REPORT.md, TABLES.md). Videos are 121 f @ 24 fps; the first 9 and (R2/R3) last 8 frames are the conditioned anchors.</div></header>
<div class="summary">
<span>scene-change medians R1→R2→R3: <b>{summary['high']['R1']:.2f} → {summary['high']['R2']:.2f} → {summary['high']['R3']:.2f}</b></span>
<span>in-place controls: <b>{summary['low']['R1']:.2f} → {summary['low']['R2']:.2f} → {summary['low']['R3']:.2f}</b></span>
<span>R3 below R2 on <b>{n_neg}/{n_hi}</b> scene-change prompt×seed pairs</span>
</div>
<details class="fig"><summary>figures (paired shifts · strips)</summary><div class="figs"><img src="results/fig_probe_paired.png"><img src="results/fig_strips_r1r2r3.png"></div></details>
<div class="bar">
<label>tier <select id="tier"><option value="all">all</option><option value="high">scene change</option><option value="low">in-place control</option></select></label>
<label>seed <select id="seed">{''.join(f'<option value="{s}">{s}</option>' for s in seeds)}<option value="all">all seeds</option></select></label>
<label>sort <select id="sort"><option value="id">prompt id</option><option value="d32">ΔDR R3−R2 (most negative first)</option><option value="r3">R3 DR (lowest first)</option></select></label>
<button id="play">▶ play all visible</button><button id="pause">⏸ pause all</button>
<span id="count" style="color:var(--mut)"></span>
</div>
<main id="main"></main>
<script>
const DATA = {json.dumps(data)};
const LABEL = {json.dumps(RUN_LABEL)};
const fmt = x => x==null||Number.isNaN(x) ? '–' : x.toFixed(2);
const sgn = x => x==null ? '' : (x<0 ? 'neg' : 'pos');
function render(){{
  const tier=document.getElementById('tier').value, seed=document.getElementById('seed').value, sort=document.getElementById('sort').value;
  let items=DATA.filter(d=>tier==='all'||d.tier===tier);
  const key=d=>{{const rows=seed==='all'?d.rows:d.rows.filter(r=>r.seed===seed); const v=rows.map(r=>sort==='d32'?r.d32:(r.cells.R3?r.cells.R3.DR:null)).filter(x=>x!=null); return v.length?Math.min(...v):1e9;}};
  if(sort!=='id') items=[...items].sort((a,b)=>key(a)-key(b));
  const main=document.getElementById('main'); main.innerHTML='';
  let n=0;
  for(const d of items){{
    const rows=seed==='all'?d.rows:d.rows.filter(r=>r.seed===seed); if(!rows.length) continue; n++;
    const card=document.createElement('div'); card.className='card';
    card.innerHTML=`<div class="head"><span class="pid">${{d.prompt_id}}</span><span class="tier ${{d.tier}}">${{d.tier==='high'?'scene change':'in-place control'}}</span><span class="ep">${{d.endpoint}} · ${{d.endpoint_class}}${{d.mechanism?' · '+d.mechanism:''}}</span></div>
      <div class="prompt">${{esc(d.start)}} <span class="ch">${{esc(d.change)}}</span><span class="tag">← dropped in R3</span> ${{esc(d.end)}}</div>`;
    for(const r of rows){{
      const row=document.createElement('div'); row.className='seedrow';
      row.innerHTML=`<div class="seedlab"><b>seed ${{r.seed}}</b>realized change ${{fmt(r.dino)}}<div class="deltas">ΔR2−R1 <b class="${{sgn(r.d21)}}">${{fmt(r.d21)}}</b><br>ΔR3−R2 <b class="${{sgn(r.d32)}}">${{fmt(r.d32)}}</b></div></div>`;
      for(const run of ['R1','R2','R3']){{
        const c=r.cells[run]; const cell=document.createElement('div'); cell.className='cell';
        cell.innerHTML= c ? `<video src="${{c.src}}" preload="none" muted loop playsinline controls></video><div class="cap"><span>${{LABEL[run]}}</span><span class="dr ${{c.DR<=0.12?'low':''}}">DR <b>${{fmt(c.DR)}}</b> · M ${{fmt(c.M)}}</span></div>` : `<div class="cap">${{LABEL[run]}} — missing</div>`;
        row.appendChild(cell);
      }}
      card.appendChild(row);
    }}
    main.appendChild(card);
  }}
  document.getElementById('count').textContent=`${{n}} prompts shown`;
}}
function esc(s){{return s.replace(/&/g,'&amp;').replace(/</g,'&lt;');}}
for(const id of ['tier','seed','sort']) document.getElementById(id).addEventListener('change',render);
document.getElementById('play').onclick=()=>document.querySelectorAll('video').forEach(v=>{{v.play().catch(()=>{{}});}});
document.getElementById('pause').onclick=()=>document.querySelectorAll('video').forEach(v=>v.pause());
render();
</script></body></html>"""
    (VIEW / "index.html").write_text(page)
    print(f"[collapse_probe] {len(data)} prompts, {len(by_key)} clips -> {VIEW/'index.html'}")


if __name__ == "__main__":
    main()
