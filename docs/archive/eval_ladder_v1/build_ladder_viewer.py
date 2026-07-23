"""Build the ladder-v3 side-by-side viewer.

Joins the exp_066 eval manifests (video paths, references, endpoints), the
certified v3.0.0 per-item scores (outputs/eval/ladder_v3/*/items.jsonl), and
the certification trust map into one self-contained HTML page.

Comparison unit = (transition class, endpoints clip, seed): every arm that
attempted that instance renders side by side — GT/demo clips first, then
base -> specialists -> ic3 — each cell carrying its certified metrics, with
the synthesized crossfade/hold controls as per-item floor chips.

Run:   python3 docs/eval_ladder/build_ladder_viewer.py
View:  cd <repo root> && python3 -m http.server 8123
       -> http://localhost:8123/outputs/eval/ladder_v3/_viewer/index.html
(must be served from the repo root so /outputs and /data resolve)
"""
import collections
import glob
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
DS = REPO / "experiments/exp_066_ladder_v3_scoring/dataset"
SCORES = REPO / "outputs/eval/ladder_v3"
TRUST = REPO / "outputs/eval/certification/3.0.0-draft.8/exam/trust_map.json"
OUT = SCORES / "_viewer/index.html"

SKIP_LABELS = {"_contrasts", "_viewer", "sigma_hero_recheck"}
SKIP_ARMS = {"sigma_hero_recheck"}

ARM_ORDER = [
    "r0", "r1", "r1k", "r1k_ext", "ic2_r4", "ic2_r5",
    "r2_ckpt250", "r2_ckpt2000", "r3_ckpt250", "r3_ckpt2000", "r3x",
    "ic3_a", "ic3_b", "ic3_c", "ic3_x",
]
ARM_LABEL = {
    "r0": "base · prompt only (R0)",
    "r1": "base · +endpoints (R1)",
    "r1k": "base·PE keyed (R1K)",
    "r1k_ext": "base·PE keyed (R1K ext)",
    "ic2_r4": "ic2 legacy · ref",
    "ic2_r5": "ic2 legacy · zero-shot",
    "r2_ckpt250": "specialist SEEN @250",
    "r2_ckpt2000": "specialist SEEN @2000",
    "r3_ckpt250": "specialist UNSEEN @250",
    "r3_ckpt2000": "specialist UNSEEN @2000",
    "r3x": "specialist FOREIGN (R3X)",
    "ic3_a": "ic3 · held-in (A)",
    "ic3_b": "ic3 · unseen (B)",
    "ic3_c": "ic3 · zero-shot (C)",
    "ic3_x": "ic3 · foreign+ref (X)",
}


def rel(p):
    """Make any manifest path root-relative for serving from the repo root."""
    s = str(p)
    if s.startswith(str(REPO)):
        s = s[len(str(REPO)):]
    if not s.startswith("/"):
        s = "/" + s
    return s


def parse_id(item_id):
    """<rung>__<class>__<ep_clip>__s<seed>[__ckptK] (classes use single '_')."""
    parts = item_id.split("__")
    if len(parts) < 4 or not parts[3].startswith("s"):
        return None
    rung, cls, clip = parts[0], parts[1], parts[2]
    seed = parts[3][1:]
    ckpt = parts[4] if len(parts) > 4 else ""
    return rung, cls, clip, seed, ckpt


def clip_class(path):
    """data/processed/transitions_std121/<cls>/<clip>.mp4 -> cls ('' if n/a)."""
    if not path:
        return ""
    q = pathlib.Path(str(path))
    return q.parent.name if q.parent.name != "transitions_std121" else ""


def main():
    # ---- 1. eval manifests: item_id -> video paths / reference / endpoints
    meta = {}
    for f in sorted(DS.glob("eval_*.json")):
        if "sigma_hero" in f.name:
            continue
        doc = json.loads(f.read_text())
        rows = doc["rows"] if isinstance(doc, dict) else doc
        for r in rows:
            m = dict(
                video=rel(r["generated_video"]),
                ref=rel(r.get("reference_video", "")) if r.get("reference_video") else "",
                arm=r["arm"], notes=r.get("notes", ""),
            )
            cp = r.get("condition_prefix") or {}
            m["ep_video"] = rel(cp["video"]) if cp.get("video") else ""
            meta[(r["item_id"], r["arm"])] = m

    # ---- 2. certified scores (+ controls as per-item floors)
    METRICS = ["margin", "app_ref", "max_seam_z", "copy_max", "near_copy",
               "cam_dtw", "obj_match", "core_degenerate", "cross_high"]
    scored = {}
    floors = collections.defaultdict(dict)   # key -> {"lerp": {...}, "hold": {...}}
    for f in sorted(SCORES.glob("*/items.jsonl")):
        if f.parent.name in SKIP_LABELS:
            continue
        for line in open(f):
            r = json.loads(line)
            arm = r["arm"]
            if arm in SKIP_ARMS:
                continue
            iid = r["item_id"]
            if arm.startswith("control_"):
                kind = arm.split("_", 1)[1]              # lerp | hold
                parent = iid.split("__", 1)[1]           # strip control_x__
                pp = parse_id(parent)
                if pp:
                    key = (pp[1], pp[2], pp[3])
                    if kind not in floors[key]:          # first wins; dupes ~identical
                        floors[key][kind] = {k: r.get(k) for k in ("margin", "app_ref", "max_seam_z")}
                continue
            scored[(iid, arm)] = {k: r.get(k) for k in METRICS} | dict(
                sidedness=r.get("sidedness"), tags=r.get("tags") or [])

    # ---- 3. cards keyed (class, ep_clip, seed)
    cards = {}
    for (iid, arm), sc in scored.items():
        pp = parse_id(iid)
        if pp is None:
            continue
        rung, cls, clip, seed, _ckpt = pp
        key = (cls, clip, seed)
        c = cards.setdefault(key, dict(
            cls=cls, clip=clip, seed=seed, sidedness=sc["sidedness"],
            tags=sc["tags"], cells={}, refs={}, ep_video="", floors={}))
        mm = meta.get((iid, arm), {})
        cell = dict(arm=arm, rung=rung, item_id=iid,
                    video=mm.get("video", ""), notes=mm.get("notes", ""))
        cell.update({k: sc.get(k) for k in METRICS})
        c["cells"][arm] = cell
        if mm.get("ep_video"):
            c["ep_video"] = mm["ep_video"]
        if mm.get("ref"):
            c["refs"].setdefault(mm["ref"], []).append(arm)
    for key, c in cards.items():
        c["floors"] = floors.get(key, {})
        # r0 has no condition_prefix; its scoring reference IS the endpoints clip
        if not c["ep_video"]:
            ep = f"/data/processed/transitions_std121/{clip_class_guess(c['clip'])}/{c['clip']}.mp4"
            c["ep_video"] = ep
        c["ep_cls"] = clip_class(c["ep_video"])
        # distinct reference clips, demo (ic input) first, minus the endpoints clip
        refs = [dict(path=p, arms=sorted(a)) for p, a in c["refs"].items() if p != c["ep_video"]]
        refs.sort(key=lambda r: (0 if any(x.startswith(("ic3", "ic2")) for x in r["arms"]) else 1, r["path"]))
        c["refs"] = refs
        missing = [a for a, cell in c["cells"].items()
                   if not cell["video"] or not (REPO / cell["video"].lstrip("/")).exists()]
        c["missing"] = missing

    # ---- 4. trust map -> per-class per-channel booleans
    trust = {}
    for cls, t in json.loads(TRUST.read_text()).items():
        trust[cls] = dict(app_ref=bool(t.get("m1a")), margin=bool(t.get("m2b")),
                          cam_dtw=bool(t.get("m1b")), obj_match=bool(t.get("m1c")))

    out_cards = [cards[k] for k in sorted(cards)]
    data = dict(cards=out_cards, trust=trust, arm_order=ARM_ORDER, arm_label=ARM_LABEL,
                mde=dict(margin=0.037, app_ref=0.024, copy_max=0.022, cam_dtw=0.076))
    n_cells = sum(len(c["cells"]) for c in out_cards)
    n_miss = sum(len(c["missing"]) for c in out_cards)
    print(f"[build] cards={len(out_cards)}  cells={n_cells}  missing_videos={n_miss}")

    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"[write] {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


def clip_class_guess(clip):
    """flame_transition_0 -> flame_transition? No — clips live under their class
    dir; strip the trailing _<n> index to recover it."""
    return clip.rsplit("_", 1)[0]


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Ladder v3 — side-by-side viewer</title>
<style>
:root{--bg:#101215;--card:#191c21;--edge:#2a2f37;--fg:#dde3ea;--dim:#8b95a1;
--green:#39d98a;--red:#ff6b6b;--amber:#ffc857;--blue:#6ab7ff;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:9;background:#0c0e11f2;border-bottom:1px solid var(--edge);
padding:10px 16px;backdrop-filter:blur(4px)}
h1{font-size:16px;margin:0 0 8px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.tab{padding:4px 12px;border:1px solid var(--edge);border-radius:14px;cursor:pointer;color:var(--dim)}
.tab.on{background:#26547c;color:#fff;border-color:#3a6ea5}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;font-size:13px}
select,input[type=text]{background:#14171b;color:var(--fg);border:1px solid var(--edge);
border-radius:6px;padding:3px 8px}
#agg{margin:6px 16px 0;overflow-x:auto}
#agg table{border-collapse:collapse;font-size:12px;white-space:nowrap}
#agg td,#agg th{padding:2px 9px;border-bottom:1px solid var(--edge);text-align:right}
#agg th{color:var(--dim);font-weight:500}#agg td:first-child,#agg th:first-child{text-align:left}
main{padding:12px 16px}
.card{background:var(--card);border:1px solid var(--edge);border-radius:10px;
padding:10px 12px;margin-bottom:16px}
.chead{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
.chead b{font-size:15px}.chead .dim{color:var(--dim);font-size:12px}
.floor{font-size:12px;color:var(--amber)}
.strip{display:flex;gap:10px;overflow-x:auto;padding-bottom:6px}
.cell{flex:0 0 252px;background:#14171b;border:1px solid var(--edge);border-radius:8px;padding:6px}
.cell.gt{border-color:#3a5f43}.cell.demo{border-color:#3a4d6e}
.cell h4{margin:0 0 4px;font-size:12px;font-weight:600;color:var(--blue);
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cell.gt h4{color:var(--green)}
video{width:240px;height:180px;background:#000;border-radius:4px;display:block;object-fit:contain}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px;font-size:11px}
.chip{padding:1px 6px;border-radius:9px;background:#20242a;color:var(--dim)}
.chip.g{background:#153826;color:var(--green)}.chip.r{background:#3a1a1a;color:var(--red)}
.chip.a{background:#3a2f14;color:var(--amber)}
.chip.copy{background:#5c1f1f;color:#ff9c9c;font-weight:700}
.delta{color:var(--blue)}
.pbtn{margin-left:auto;background:#26547c;border:0;color:#fff;border-radius:6px;
padding:3px 10px;cursor:pointer;font-size:12px}
.note{font-size:11px;color:var(--dim);margin-top:4px;max-width:240px;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#count{color:var(--dim);font-size:12px;margin:0 16px}
.missing{color:var(--red);font-size:11px}
</style></head><body>
<header>
<h1>Eval ladder v3 — paired side-by-side viewer <span class="dim" style="font-size:12px">(certified v3.0.0 rows · GT green · demo blue · floors amber)</span></h1>
<div class="tabs" id="tabs"></div>
<div class="bar">
class <select id="fcls"><option value="">all</option></select>
seed <select id="fseed"><option value="42">42</option><option value="43">43</option><option value="44">44</option><option value="">all</option></select>
sort <select id="fsort"><option value="cls">class</option><option value="dmargin">Δmargin ic3−spec</option><option value="margin">best margin</option></select>
find <input id="fq" type="text" placeholder="class or clip…" size="18">
<span id="count"></span>
</div>
</header>
<div id="agg"></div>
<main id="main"></main>
<script>
const D = __DATA__;
const PANELS = {
 all:       {label:"All", test:c=>true},
 cond:      {label:"Conditioning (R0 vs R1)", test:c=>c.cells.r0&&c.cells.r1},
 tierA:     {label:"Held-in A (SEEN)", test:c=>c.cells.ic3_a||c.cells.r2_ckpt250||c.cells.r2_ckpt2000},
 tierB:     {label:"Unseen B (PRIMARY)", test:c=>c.cells.ic3_b||c.cells.r3_ckpt250||c.cells.r3_ckpt2000},
 tierC:     {label:"Zero-shot C", test:c=>c.cells.ic3_c},
 tierX:     {label:"Foreign transfer X", test:c=>c.cells.r3x||c.cells.ic3_x},
};
let panel="tierB";
const $=s=>document.querySelector(s);
const fmt=(v,d=3)=>v==null?"–":(+v).toFixed(d);
function trustMark(cls,ch){const t=D.trust[cls];return t&&t[ch]===false?"†":"";}
function baseCell(c){return c.cells.r1k||c.cells.r1k_ext||c.cells.r1||null;}
function specCell(c){return c.cells.r3_ckpt2000||c.cells.r2_ckpt2000||c.cells.r3x||null;}
function icCell(c){return c.cells.ic3_b||c.cells.ic3_a||c.cells.ic3_c||c.cells.ic3_x||null;}
function chip(lbl,v,cls,extra){return `<span class="chip ${cls||''}" title="${extra||''}">${lbl} ${v}</span>`;}
function cellHtml(c,cell){
 const fl=c.floors.lerp||{};
 let mc="";
 if(cell.margin!=null&&fl.margin!=null){
   mc=cell.margin>fl.margin+0.02?"g":(cell.margin<fl.margin-0.02?"r":"a");}
 const b=baseCell(c);
 let d="";
 if(b&&b.arm!==cell.arm&&cell.margin!=null&&b.margin!=null)
   d=`<span class="chip delta" title="vs ${b.arm} · MDE ${D.mde.margin}">Δm ${(cell.margin-b.margin>=0?"+":"")+fmt(cell.margin-b.margin)}</span>`;
 const chips=[
  chip("m"+trustMark(c.cls,"margin"),fmt(cell.margin),mc,"margin (floor "+fmt(fl.margin)+")"),
  chip("app"+trustMark(c.cls,"app_ref"),fmt(cell.app_ref,2),"","app_ref (floor "+fmt(fl.app_ref,2)+")"),
  chip("seam",fmt(cell.max_seam_z,1),cell.max_seam_z>6?"a":"","max_seam_z"),
  cell.near_copy?`<span class="chip copy">NEAR-COPY ${fmt(cell.copy_max,2)}</span>`:chip("copy",fmt(cell.copy_max,2),"","copy_max τ=0.858"),
 ];
 if((c.tags||[]).includes("camera")) chips.push(chip("cam"+trustMark(c.cls,"cam_dtw"),fmt(cell.cam_dtw,2),"","cam_dtw lower=better"));
 if(cell.obj_match!=null) chips.push(chip("obj"+trustMark(c.cls,"obj_match"),fmt(cell.obj_match,2),"","obj_match"));
 if(d) chips.push(d);
 if(cell.core_degenerate) chips.push('<span class="chip r">core_degenerate</span>');
 const missing=!cell.video;
 return `<div class="cell"><h4 title="${cell.item_id}">${D.arm_label[cell.arm]||cell.arm}</h4>
 ${missing?'<div class="missing">video missing</div>':`<video preload="none" muted loop playsinline data-src="${cell.video}"></video>`}
 <div class="chips">${chips.join("")}</div>
 <div class="note" title="${(cell.notes||"").replace(/"/g,'&quot;')}">${cell.notes||""}</div></div>`;
}
function gtHtml(c){
 const own=c.ep_cls===c.cls;
 return `<div class="cell gt"><h4>${own?"GT — endpoints' own transition":"endpoints donor clip ("+c.ep_cls+")"}</h4>
 <video preload="none" muted loop playsinline data-src="${c.ep_video}"></video>
 <div class="note">${c.ep_video.split("/").slice(-2).join("/")}</div></div>`;
}
function refHtml(c,r){
 const demo=r.arms.some(a=>a.startsWith("ic"));
 return `<div class="cell demo"><h4>${demo?"demo reference (ic input)":"class demo (scoring ref)"}</h4>
 <video preload="none" muted loop playsinline data-src="${r.path}"></video>
 <div class="note">${r.path.split("/").slice(-2).join("/")} · ${r.arms.join(", ")}</div></div>`;
}
function cardHtml(c,i){
 const fl=c.floors.lerp,fh=c.floors.hold;
 const cells=D.arm_order.filter(a=>c.cells[a]).map(a=>cellHtml(c,c.cells[a]));
 return `<div class="card" data-i="${i}">
 <div class="chead"><b>${c.cls}</b><span class="dim">endpoints ${c.clip} · seed ${c.seed} · ${c.sidedness||""} · ${(c.tags||[]).join(",")}</span>
 ${fl?`<span class="floor">crossfade floor: m ${fmt(fl.margin)} app ${fmt(fl.app_ref,2)}</span>`:""}
 ${fh?`<span class="floor" style="opacity:.7">hold: m ${fmt(fh.margin)}</span>`:""}
 <button class="pbtn" onclick="playAll(this)">▶ play all</button></div>
 <div class="strip">${gtHtml(c)}${c.refs.map(r=>refHtml(c,r)).join("")}${cells.join("")}</div></div>`;
}
function playAll(btn){
 btn.closest(".card").querySelectorAll("video").forEach(v=>{
  if(!v.src)v.src=v.dataset.src; v.currentTime=0; v.play();});
}
function visible(){
 let cs=D.cards.filter(PANELS[panel].test);
 const cls=$("#fcls").value, seed=$("#fseed").value, q=$("#fq").value.toLowerCase();
 if(cls)cs=cs.filter(c=>c.cls===cls);
 if(seed)cs=cs.filter(c=>c.seed===seed);
 if(q)cs=cs.filter(c=>(c.cls+" "+c.clip).toLowerCase().includes(q));
 const s=$("#fsort").value;
 if(s==="dmargin")cs=cs.slice().sort((a,b)=>{
   const f=c=>{const i=icCell(c),sp=specCell(c);return i&&sp&&i.margin!=null&&sp.margin!=null?i.margin-sp.margin:-1e9;};
   return f(b)-f(a);});
 else if(s==="margin")cs=cs.slice().sort((a,b)=>{
   const f=c=>Math.max(...Object.values(c.cells).map(x=>x.margin??-1e9));return f(b)-f(a);});
 return cs;
}
function agg(cs){
 const by={};
 cs.forEach(c=>Object.values(c.cells).forEach(x=>{
  const e=by[x.arm]??={n:0,m:0,a:0,s:0,nc:0,mn:0,an:0,sn:0};
  e.n++;
  if(x.margin!=null){e.m+=x.margin;e.mn++;}
  if(x.app_ref!=null){e.a+=x.app_ref;e.an++;}
  if(x.max_seam_z!=null){e.s+=x.max_seam_z;e.sn++;}
  if(x.near_copy)e.nc++;}));
 let fm=0,fa=0,fn=0;
 cs.forEach(c=>{if(c.floors.lerp){fm+=c.floors.lerp.margin??0;fa+=c.floors.lerp.app_ref??0;fn++;}});
 const rows=D.arm_order.filter(a=>by[a]).map(a=>{const e=by[a];
  return `<tr><td>${D.arm_label[a]||a}</td><td>${e.n}</td><td>${fmt(e.m/e.mn)}</td><td>${fmt(e.a/e.an,2)}</td><td>${fmt(e.s/e.sn,1)}</td><td>${(100*e.nc/e.n).toFixed(0)}%</td></tr>`;});
 $("#agg").innerHTML=`<table><tr><th>arm (filtered set)</th><th>n</th><th>margin</th><th>app_ref</th><th>seam z</th><th>near-copy</th></tr>
 ${fn?`<tr><td style="color:var(--amber)">crossfade floor</td><td>${fn}</td><td style="color:var(--amber)">${fmt(fm/fn)}</td><td style="color:var(--amber)">${fmt(fa/fn,2)}</td><td>–</td><td>–</td></tr>`:""}
 ${rows.join("")}</table>`;
}
let io;
function render(){
 const cs=visible();
 $("#count").textContent=cs.length+" items";
 $("#main").innerHTML=cs.slice(0,400).map((c,i)=>cardHtml(c,i)).join("");
 agg(cs);
 io&&io.disconnect();
 io=new IntersectionObserver(es=>es.forEach(e=>{
  if(e.isIntersecting)e.target.querySelectorAll("video").forEach(v=>{
   if(!v.src){v.src=v.dataset.src;v.play().catch(()=>{});}});
 }),{rootMargin:"200px"});
 document.querySelectorAll(".card").forEach(c=>io.observe(c));
}
function initTabs(){
 $("#tabs").innerHTML=Object.entries(PANELS).map(([k,p])=>
  `<span class="tab${k===panel?" on":""}" data-p="${k}">${p.label}</span>`).join("");
 document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{panel=t.dataset.p;initTabs();render();});
}
const classes=[...new Set(D.cards.map(c=>c.cls))].sort();
classes.forEach(c=>{const o=document.createElement("option");o.value=o.textContent=c;$("#fcls").append(o);});
["fcls","fseed","fsort"].forEach(id=>$("#"+id).onchange=render);
$("#fq").oninput=()=>render();
initTabs();render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
