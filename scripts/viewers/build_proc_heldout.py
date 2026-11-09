#!/usr/bin/env python
"""Viewer: procedural held-out tier x NRG (rebuttal 2026-11-09, misc/2026-11-02_rebuttal/procedural).

Per item (12 shader families x 3 items, seed 42): reference render (the in-context demo), GT target render (same exact
op on the target endpoints), SEGUE w=1 (`dualforce_control`), SEGUE NRG w=6 (`dualforce_dcg_w6`), and the NRG null the
sampler contrasts against = linear crossfade of the REFERENCE's first and last frames over all 121 frames
(ValidationRunner._build_crossfade_null; built here with the sampler's own code and written as mp4).

    $LAB/envs-aarch64/ltx2/bin/python scripts/viewers/build_proc_heldout.py
    -> outputs/reports/proc_heldout_nrg/index.html   (+ nulls in outputs/videos/proc_heldout_nrg/nulls/)
Mounted at outputs/viewers/proc_heldout_nrg/ by viewerctl (links store / misc / outputs beside the page).
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

DR = Path(__file__).resolve().parents[2]
PROC = DR / "misc/2026-11-02_rebuttal/procedural"
OUT = DR / "outputs/reports/proc_heldout_nrg"
NULLS = DR / "outputs/videos/proc_heldout_nrg/nulls"
sys.path.insert(0, str(PROC / "score"))
SUB = {"w1": "store/gens/013_dualforce_control/07_neutral_proc__dai",
       "w6": "store/gens/032_dualforce_dcg_w6/09_neutral_proc__dai"}


def build_nulls(rows: list[dict]) -> None:
    import dnull_lib as L
    from diffusion.transition_eval.video_io import write_video
    NULLS.mkdir(parents=True, exist_ok=True)
    for r in rows:
        dst = NULLS / f"{r['reference']}_null.mp4"
        if not dst.exists():
            _, null = L.exact_null(DR / r["reference_path"], 121)
            write_video(null, dst, fps=24.0)


def main() -> None:
    rows = [json.loads(l) for l in (PROC / "registry/registry_dualforce_control_neutral_proc.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r: (r["seen"], r["family"], r["item"]))
    build_nulls(rows)
    ops = {}
    for f in (PROC / "render/ops").glob("*.jsonl"):
        for l in f.read_text().splitlines():
            o = json.loads(l)
            if o["ok"]:
                ops[o["stem"]] = o
    gtd = {}
    for f in (PROC / "score").glob("part2_res_neutral*_s*.jsonl"):
        for l in f.read_text().splitlines():
            q = json.loads(l)
            gtd[(Path(q["gen"]).name)] = q
    suite = {}
    sp = PROC / "score/full/suite_rows.jsonl"
    if sp.exists():
        for l in sp.read_text().splitlines():
            q = json.loads(l)
            suite[Path(q["video"]).name] = q
    cards = []
    for r in rows:
        o = ops[r["endpoint"]]
        gid = r["item_id"]
        w6id = gid.replace("dualforce_control_neutral_proc", "dualforce_dcg_w6_neutral_proc")
        vids = {"ref": r["reference_path"], "gt": r["gt_path"],
                "w1": f"{SUB['w1']}/videos/{gid}__s42.mp4", "w6": f"{SUB['w6']}/videos/{w6id}__s42.mp4",
                "null": f"nulls/{r['reference']}_null.mp4"}
        num = {}
        for k, name in (("w1", f"{gid}__s42.mp4"), ("w6", f"{w6id}__s42.mp4")):
            g, s = gtd.get(name), suite.get(name)
            num[k] = {"gt_lpips": g and round(g["gt_lpips"], 3), "gt_dino": g and round(g["gt_dino"], 3),
                      "transport": s and round(s["transport"], 1)}
        g = gtd.get(f"{gid}__s42.mp4")
        cards.append({"family": r["family"], "seen": r["seen"], "item": r["item"], "vids": vids, "num": num,
                      "dnull": g and round(g["gt_dnull_lpips"], 3),
                      "op": f"{o['shader']} · easing {o['easing']} · flip {o['flip']} · swap {o['swap']} · "
                            f"onset {o['timing']['onset']:.1f} / release {o['timing']['release']:.1f} · params "
                            f"{json.dumps(o['params'])}",
                      "prompt": r["prompt"]})
    OUT.mkdir(parents=True, exist_ok=True)
    page = TEMPLATE.replace("__DATA__", json.dumps(cards))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"[viewer] {len(cards)} items -> {OUT / 'index.html'}")


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Procedural held-out x NRG</title>
<style>
:root{--bg:#fafafa;--fg:#222;--mut:#666;--card:#fff;--bd:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mut:#9a9a9a;--card:#202020;--bd:#333}}
body{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:16px}
h1{font-size:18px;margin:0 0 6px} .mut{color:var(--mut)} .bar{position:sticky;top:0;background:var(--bg);padding:6px 0;z-index:2}
.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:10px;margin:10px 0}
.row{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}
.cell video{width:100%;border-radius:4px;background:#000} .cell b{font-size:12px}
.meta{font-size:12px;margin-top:6px} select,button{font:inherit}
</style></head><body>
<div class="bar"><h1>Procedural held-out shader families × NRG — SEGUE w=1 vs NRG w=6 (neutral prompt, seed 42)</h1>
<span class="mut">Columns: reference render (in-context demo) · GT target (same exact op on the target endpoints) · w=1 · NRG w=6 ·
NRG null = linear crossfade of the reference's first and last frames over all 121 frames (what NRG contrasts against).
Numbers: d_gt = mean per-frame distance to GT over frames [9,113) (lower = closer); d_null = GT ↔ its own crossfade null.</span><br>
family <select id="fam"><option value="">all</option></select> <button id="play">play all</button> <button id="sync">restart all</button></div>
<div id="cards"></div>
<script>
const D=__DATA__;
const fam=document.getElementById('fam');
[...new Set(D.map(c=>c.family))].forEach(f=>{const o=document.createElement('option');o.value=o.textContent=f;fam.appendChild(o)});
function n(x){return x==null?'--':x}
function render(){const box=document.getElementById('cards');box.innerHTML='';
 D.filter(c=>!fam.value||c.family===fam.value).forEach(c=>{const d=document.createElement('div');d.className='card';
  const cols=[['ref','reference render'],['gt','GT target'],['w1','SEGUE w=1'],['w6','SEGUE NRG w=6'],['null','NRG null (ref crossfade)']];
  d.innerHTML=`<b>${c.family}</b> ${c.seen?'(seen anchor)':'(held-out)'} · item ${c.item} <span class="mut">· d_null(GT) LPIPS ${n(c.dnull)}</span>
  <div class="row">${cols.map(([k,l])=>`<div class="cell"><b>${l}</b>${k==='w1'||k==='w6'?` <span class="mut">d_gt LPIPS ${n(c.num[k].gt_lpips)} · DINO ${n(c.num[k].gt_dino)} · Transport ${n(c.num[k].transport)}</span>`:''}<video src="${c.vids[k]}" muted loop playsinline preload="metadata" controls></video></div>`).join('')}</div>
  <div class="meta mut">op: ${c.op}<br>prompt: ${c.prompt.replace(/</g,'&lt;')}</div>`;box.appendChild(d)})}
fam.onchange=render;render();
document.getElementById('play').onclick=()=>document.querySelectorAll('video').forEach(v=>v.play());
document.getElementById('sync').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play()});
</script></body></html>
"""

if __name__ == "__main__":
    main()
