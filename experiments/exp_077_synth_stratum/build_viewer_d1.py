"""exp_077 — build the D1 dataset viewer.

The dataset only "works" if two structural properties are visible, so the viewer is built
around exactly those two diagonals (everything else is a file dump):

  1. SAME OPERATOR / DIFFERENT CONTENT  (a tuple)
     reference and target share the operator (shader + continuous params + easing + timing)
     but sit on DISJOINT endpoint pairs. This is the pair the model learns "extract the
     operator from the demo, apply it here" from.

  2. SAME CONTENT / DIFFERENT OPERATORS  (path entropy)
     one target endpoint pair rendered under its 8 operators. p(target | endpoints) is
     explicitly multimodal, so the endpoint prior cannot solve the task and the reference
     becomes the only disambiguating signal.

Serve from the REPO ROOT so the root-relative /outputs/... video paths resolve:
    python -m http.server 8000        # then open /outputs/viewers/d1_dataset/
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
D1 = REPO_ROOT / "outputs/videos/exp_077_synth_stratum/d1"
META = D1 / "meta"
VID_URL = "/outputs/videos/exp_077_synth_stratum/d1/videos"
OUT = REPO_ROOT / "outputs/viewers/d1_dataset"


def load() -> list[dict]:
    rows = []
    for f in sorted(META.glob("tuple_*.json")):
        m = json.loads(f.read_text())
        op, tm = m["operator"], m["timing"]
        rows.append({
            "id": m["tuple_id"],
            "ti": m.get("target_index"),
            "shader": op["shader"],
            "op_id": op["op_id"],
            "params": op["params"],
            "easing": op["easing"],
            "flip": op["flip"],
            "swap": op["swap"],
            "aux": op.get("aux_kind"),
            "onset": round(tm["onset"], 1),
            "dur": round(tm["duration"], 1),
            "win": tm["window"],
            "nf": tm["num_frames"],
            "ref": m["reference"]["video"],
            "tgt": m["target"]["video"],
            "ref_A": m["reference"].get("endpoint_A_id"),
            "ref_B": m["reference"].get("endpoint_B_id"),
            "tgt_A": m["target"].get("endpoint_A_id"),
            "tgt_B": m["target"].get("endpoint_B_id"),
            "mae": max(m["target"]["rendered_endpoint_mae"]["start"],
                       m["target"]["rendered_endpoint_mae"]["end"]),
        })
    return rows


def main() -> None:
    rows = load()
    groups = defaultdict(list)
    for r in rows:
        groups[r["ti"]].append(r)
    # path-entropy groups: same target endpoint pair under N operators
    pe = [{"ti": k,
           "A": v[0]["tgt_A"], "B": v[0]["tgt_B"],
           "ops": sorted(v, key=lambda x: x["id"])}
          for k, v in sorted(groups.items())]

    shaders = sorted({r["shader"] for r in rows})
    auxes = sorted({r["aux"] or "none" for r in rows})
    stats = {
        "n_tuples": len(rows),
        "n_target_pairs": len(pe),
        "ops_per_pair": sorted({len(g["ops"]) for g in pe}),
        "n_shaders": len(shaders),
        "n_aux": sum(1 for r in rows if r["aux"]),
        "pct_aux": round(100 * sum(1 for r in rows if r["aux"]) / max(1, len(rows)), 1),
        "max_mae": round(max((r["mae"] for r in rows), default=0), 4),
        "shaders_per_pair_min": min((len({o["shader"] for o in g["ops"]}) for g in pe), default=0),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(
        {"stats": stats, "shaders": shaders, "auxes": auxes, "tuples": rows, "pe": pe}))
    (OUT / "index.html").write_text(HTML.replace("__VID__", VID_URL))
    print(f"[viewer] {stats}")
    print(f"[viewer] -> {OUT.relative_to(REPO_ROOT)}/index.html")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>D1 synthetic dataset — viewer</title>
<style>
:root{--bg:#0e1116;--pan:#161b22;--ln:#2a313c;--tx:#e6edf3;--mu:#8b949e;--ac:#58a6ff;--ok:#3fb950;--wr:#d29922}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--ln);background:var(--pan);position:sticky;top:0;z-index:10}
h1{margin:0 0 6px;font-size:17px}.sub{color:var(--mu);font-size:12.5px}
.stats{display:flex;flex-wrap:wrap;gap:14px;margin-top:9px}
.st{background:#0d1117;border:1px solid var(--ln);border-radius:6px;padding:5px 10px;font-size:12px}
.st b{color:var(--ac);font-size:14px}
.tabs{display:flex;gap:6px;margin-top:11px}
.tab{padding:6px 13px;border:1px solid var(--ln);border-radius:6px;background:#0d1117;color:var(--mu);cursor:pointer;font-size:13px}
.tab.on{background:var(--ac);color:#04121f;border-color:var(--ac);font-weight:600}
.ctl{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:10px}
select,input{background:#0d1117;color:var(--tx);border:1px solid var(--ln);border-radius:6px;padding:5px 8px;font-size:12.5px}
main{padding:16px 20px;max-width:1500px}
.why{background:#0d1117;border-left:3px solid var(--ac);border-radius:0 6px 6px 0;padding:9px 13px;margin-bottom:15px;color:var(--mu);font-size:12.5px}
.why b{color:var(--tx)}
.card{background:var(--pan);border:1px solid var(--ln);border-radius:9px;padding:12px;margin-bottom:14px}
.chd{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:9px}
.pill{background:#0d1117;border:1px solid var(--ln);border-radius:999px;padding:2px 9px;font-size:11.5px;color:var(--mu)}
.pill.k{color:var(--ac);border-color:#1f4d78}.pill.a{color:var(--wr);border-color:#5a4611}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.slot{background:#0d1117;border:1px solid var(--ln);border-radius:7px;padding:8px}
.slot .lb{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--mu);margin-bottom:5px}
.slot.r .lb{color:var(--ac)}.slot.t .lb{color:var(--ok)}
video{width:100%;border-radius:5px;background:#000;display:block;aspect-ratio:480/640;object-fit:cover}
.ep{font-size:10.5px;color:#6e7681;margin-top:5px;word-break:break-all;font-family:ui-monospace,monospace}
.grid8{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.g8 .lb{font-size:10.5px;color:var(--mu);margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tl{height:7px;background:#0d1117;border:1px solid var(--ln);border-radius:4px;position:relative;margin-top:6px}
.tl i{position:absolute;top:0;bottom:0;background:linear-gradient(90deg,#1f6feb,#58a6ff);border-radius:3px}
.tl u{position:absolute;top:0;bottom:0;width:7.4%;background:#21262d}.tl u.e{right:0}
.mono{font-family:ui-monospace,monospace;font-size:11px;color:var(--mu)}
.more{display:block;margin:18px auto;padding:9px 22px;background:var(--pan);border:1px solid var(--ln);color:var(--tx);border-radius:7px;cursor:pointer}
</style></head><body>
<header>
<h1>D1 synthetic transition dataset — viewer</h1>
<div class="sub">Procedural operators (gl-transitions, <b>flow</b> extension) over curated real endpoints. First 9 / last 9 frames are the pinned endpoint anchors; the middle is the transition.</div>
<div class="stats" id="stats"></div>
<div class="tabs">
  <div class="tab on" data-v="tuple">① Same operator → different content</div>
  <div class="tab" data-v="pe">② Same content → different operators (path entropy)</div>
</div>
<div class="ctl">
  <label class="mono">shader <select id="fsh"><option value="">all</option></select></label>
  <label class="mono">aux <select id="fax"><option value="">all</option></select></label>
  <label class="mono">search <input id="fq" placeholder="op_id / endpoint id" size="22"></label>
  <span class="mono" id="cnt"></span>
</div>
</header>
<main><div class="why" id="why"></div><div id="list"></div>
<button class="more" id="more">load more</button></main>
<script>
const V="__VID__"; let D=null,view="tuple",page=0,PER=12;
const WHY={
 tuple:"<b>Diagonal 1 — the positive pair.</b> Both clips share the SAME operator (identical shader, continuous parameters, easing and timing) but run on <b>disjoint endpoint pairs</b>. Reference content is therefore useless for predicting the target: storing it never lowers the loss. This is what teaches <i>extract the operator, apply it to new content</i>.",
 pe:"<b>Diagonal 2 — path entropy.</b> One endpoint pair under <b>8 different operators</b>. p(target | endpoints) is explicitly multimodal, so guessing from the endpoints (the lerp / endpoint-prior shortcut) is high-loss by construction and the demo becomes the only disambiguating signal. This is the property aimed at the measured failure: the adapter reads the demo but applies the operator weakly."
};
const vid=(f)=>`<video preload="none" muted loop playsinline src="${V}/${f}" onmouseenter="this.play()" onmouseleave="this.pause()" onclick="this.paused?this.play():this.pause()"></video>`;
const tl=(o)=>{const w=o.win[1]-o.win[0],a=100*(o.onset-o.win[0])/w,b=100*o.dur/w;
 return `<div class="tl"><u></u><i style="left:${Math.max(0,a)}%;width:${Math.min(100-a,b)}%"></i><u class="e"></u></div>
 <div class="mono">onset ${o.onset} · dur ${o.dur} · ease ${o.easing}${o.flip!=='none'?' · flip '+o.flip:''}${o.swap?' · swap':''}</div>`};
const pills=(o)=>`<span class="pill k">${o.shader}</span>${o.aux?`<span class="pill a">aux:${o.aux}</span>`:''}
 <span class="pill">${Object.entries(o.params).map(([k,v])=>k+'='+(+v).toFixed(3)).join(' · ')||'no params'}</span>`;
function filt(){const s=document.getElementById('fsh').value,a=document.getElementById('fax').value,q=document.getElementById('fq').value.toLowerCase();
 const ok=(o)=>(!s||o.shader===s)&&(!a||(o.aux||'none')===a)&&(!q||o.op_id.toLowerCase().includes(q)||(o.tgt_A||'').toLowerCase().includes(q)||(o.tgt_B||'').toLowerCase().includes(q));
 return view==='tuple'?D.tuples.filter(ok):D.pe.filter(g=>g.ops.some(ok));}
function render(reset){if(reset){page=0;document.getElementById('list').innerHTML=''}
 const items=filt(),slice=items.slice(page*PER,(page+1)*PER);
 document.getElementById('cnt').textContent=`${items.length} ${view==='tuple'?'tuples':'endpoint pairs'}`;
 document.getElementById('why').innerHTML=WHY[view];
 const h=slice.map(x=>view==='tuple'?card(x):peCard(x)).join('');
 document.getElementById('list').insertAdjacentHTML('beforeend',h);
 page++;document.getElementById('more').style.display=(page*PER>=items.length)?'none':'block';}
const card=(o)=>`<div class="card"><div class="chd">${pills(o)}<span class="pill">tuple ${o.id}</span>
 <span class="pill" style="color:#3fb950">endpoint MAE ${o.mae}</span></div>
 <div class="pair">
  <div class="slot r"><div class="lb">reference — the demo (operator shown here)</div>${vid(o.ref)}<div class="ep">A ${o.ref_A}<br>B ${o.ref_B}</div></div>
  <div class="slot t"><div class="lb">target — same operator, different content</div>${vid(o.tgt)}<div class="ep">A ${o.tgt_A}<br>B ${o.tgt_B}</div></div>
 </div>${tl(o)}</div>`;
const peCard=(g)=>`<div class="card"><div class="chd"><span class="pill k">endpoint pair #${g.ti}</span>
 <span class="pill">${g.ops.length} operators</span><span class="pill">${new Set(g.ops.map(o=>o.shader)).size} distinct shaders</span></div>
 <div class="ep" style="margin-bottom:9px">A ${g.A} &nbsp;→&nbsp; B ${g.B}</div>
 <div class="grid8">${g.ops.map(o=>`<div class="g8"><div class="lb">${o.shader}${o.aux?' +'+o.aux:''}</div>${vid(o.tgt)}<div class="mono">on ${o.onset} d ${o.dur}</div></div>`).join('')}</div></div>`;
fetch('data.json').then(r=>r.json()).then(d=>{D=d;
 const s=d.stats;document.getElementById('stats').innerHTML=
  `<div class="st"><b>${s.n_tuples}</b> tuples</div><div class="st"><b>${s.n_target_pairs}</b> endpoint pairs</div>
   <div class="st"><b>${s.ops_per_pair.join('/')}</b> ops per pair</div><div class="st"><b>${s.n_shaders}</b> shaders</div>
   <div class="st"><b>${s.pct_aux}%</b> aux / medium-bearing</div><div class="st">min <b>${s.shaders_per_pair_min}</b> distinct shaders/pair</div>
   <div class="st">endpoint MAE ≤ <b>${s.max_mae}</b></div>`;
 d.shaders.forEach(x=>fsh.add(new Option(x,x)));d.auxes.forEach(x=>fax.add(new Option(x,x)));
 render(true)});
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{document.querySelectorAll('.tab').forEach(z=>z.classList.remove('on'));
 t.classList.add('on');view=t.dataset.v;render(true)});
['fsh','fax'].forEach(i=>document.getElementById(i).onchange=()=>render(true));
document.getElementById('fq').oninput=()=>render(true);
document.getElementById('more').onclick=()=>render(false);
</script></body></html>"""


if __name__ == "__main__":
    main()
