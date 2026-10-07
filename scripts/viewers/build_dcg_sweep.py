#!/usr/bin/env python3
"""Build the DCG-null sweep viewer — a SIMPLE arm-comparison page.

A deliberately stripped-down cousin of eval_ladder/viewer/build_neutral_effect_v2.py (v1 archived under eval_ladder/viewer/archive/neutral_effect_v1_2026-09-22/):
it keeps that page's selection model (category -> arm -> variant-pill entries, an
always-open arm panel, an input-keyed card grid that lines clips up side by side)
and DROPS everything else — no store-contract seatbelts, no scoring, no metrics
table, no ontology matrix. It only shows the generated transition clips with an
arm picker, per the owner's "keep it VERY SIMPLE" instruction.

Clips are lined up by `input_key` (the per-input hash that is identical across
arms/models for the same endpoint x reference x cell), so the same input shows one
card with one cell per selected arm; a missing clip is just a blank cell.

Reads ONLY the store (store/gens/<NNN>/<subentry>/grid.jsonl + videos/). Writes
outputs/reports/dcg_sweep/index.html; the registry mount symlinks it into
outputs/viewers/dcg_sweep/ next to a `store` symlink so `store/gens/...` paths
resolve. Rebuild:  <aarch64 env>/bin/python scripts/viewers/build_dcg_sweep.py
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# NB: slug is `dcg_null_sweep` — the older 2026-08-14 DCG-conditioning guidance
# sweep already owns the `dcg_sweep` viewer dir, so this one stays clear of it.
OUT = REPO / "outputs/reports/dcg_null_sweep/index.html"

# Reuse the gen's OWN path resolution for the source input clips so the tiles
# point at exactly what each generation was conditioned on: endpoint anchors
# eval_ladder/conds/<ep>_{start9,end9}.mp4 (ec.cond_paths) and the reference
# demo data/processed/transitions_std121/<class>/<ref>.mp4 (ec.STD + P.clip_class).
sys.path.insert(0, str(REPO / "eval_ladder"))
try:
    import encode_conditioning as ec  # noqa: E402
    import prompts as P               # noqa: E402
except Exception as _e:               # keep building even if the resolvers can't import
    ec = P = None
    print("[warn] eval_ladder resolvers unavailable ({}) — source tiles will be blank".format(_e))


def resolve_src(endpoint, reference, sided):
    """Repo-relative paths of the input media a row was conditioned on (None if unresolved)."""
    d = {"ep_start": None, "ep_end": None, "ref": None,
         "ep_name": endpoint, "ref_name": reference}
    if ec is not None:
        p = ec.CONDS / "{}_start9.mp4".format(endpoint)
        if p.exists():
            d["ep_start"] = str(p.relative_to(REPO))
        if sided == "two":
            q = ec.CONDS / "{}_end9.mp4".format(endpoint)
            if q.exists():
                d["ep_end"] = str(q.relative_to(REPO))
    if ec is not None and P is not None and reference:
        try:
            r = ec.STD / P.clip_class(reference) / "{}.mp4".format(reference)
            if r.exists():
                d["ref"] = str(r.relative_to(REPO))
        except Exception:
            pass
    return d

# ── arm catalog ───────────────────────────────────────────────────────────
# Grouped by MODEL (category), then null/w kind (arm), then grid family or w
# (variant-pill entry).  `sub` = store gen subentry (relative to store/gens/).
CATS = [
    {"id": "ctt_v2", "label": "ctt_v2 (runs/002) — 3-null core", "arms": [
        {"id": "ctt_base", "label": "base (w1)", "entries": [
            {"tier": "ctt_base_hf", "variant": "HF", "label": "HF-121",
             "sub": "041_ctt_v2_dcg_nulls/01_base_w1_v3zs60__dai"},
            {"tier": "ctt_base_ed", "variant": "ED", "label": "ED-81",
             "sub": "041_ctt_v2_dcg_nulls/05_base_w1_v3ed81zs60__dai"}]},
        {"id": "ctt_xfade", "label": "crossfade (w3)", "entries": [
            {"tier": "ctt_xfade_hf", "variant": "HF", "label": "HF-121",
             "sub": "041_ctt_v2_dcg_nulls/02_crossfade_w3_v3zs60__dai"},
            {"tier": "ctt_xfade_ed", "variant": "ED", "label": "ED-81",
             "sub": "041_ctt_v2_dcg_nulls/06_crossfade_w3_v3ed81zs60__dai"}]},
        {"id": "ctt_empty", "label": "empty (w3)", "entries": [
            {"tier": "ctt_empty_hf", "variant": "HF", "label": "HF-121",
             "sub": "041_ctt_v2_dcg_nulls/03_empty_w3_v3zs60__dai"},
            {"tier": "ctt_empty_ed", "variant": "ED", "label": "ED-81",
             "sub": "041_ctt_v2_dcg_nulls/07_empty_w3_v3ed81zs60__dai"}]},
        {"id": "ctt_hold8", "label": "hold-swap8 (w3)", "entries": [
            {"tier": "ctt_hold8_hf", "variant": "HF", "label": "HF-121",
             "sub": "041_ctt_v2_dcg_nulls/04_holdswap8_w3_v3zs60__dai"},
            {"tier": "ctt_hold8_ed", "variant": "ED", "label": "ED-81",
             "sub": "041_ctt_v2_dcg_nulls/08_holdswap8_w3_v3ed81zs60__dai"}]},
    ]},
    {"id": "dualforce", "label": "dualforce (runs/012)", "arms": [
        {"id": "df_base", "label": "base (control)", "entries": [
            {"tier": "df_base_hf", "variant": "HF", "label": "HF-121",
             "sub": "013_dualforce_control/03_neutral_v3__dai"},
            {"tier": "df_base_ed", "variant": "ED", "label": "ED-81",
             "sub": "013_dualforce_control/04_neutral_v3ed81__dai"}]},
        {"id": "df_xfade", "label": "crossfade (w6)", "entries": [
            {"tier": "df_xfade_hf", "variant": "HF", "label": "HF-121",
             "sub": "032_dualforce_dcg_w6/03_neutral_v3__dai"},
            {"tier": "df_xfade_ed", "variant": "ED", "label": "ED-81",
             "sub": "032_dualforce_dcg_w6/04_neutral_v3ed81__dai"}]},
        {"id": "df_vaelerp", "label": "vae-lerp (w6)", "entries": [
            {"tier": "df_vaelerp_hf", "variant": "HF", "label": "HF-121",
             "sub": "036_dualforce_dcg_vaelerp_w6/01_neutral_v3zs__dai"},
            {"tier": "df_vaelerp_ed", "variant": "ED", "label": "ED-81",
             "sub": "036_dualforce_dcg_vaelerp_w6/02_neutral_v3ed81zs__dai"}]},
        {"id": "df_emptynull", "label": "empty-null (w6) — full grid v3 (2026-09-24)", "entries": [
            {"tier": "df_emptynull_hf", "variant": "HF", "label": "HF-121",
             "sub": "040_dualforce_dcg_emptynull_w6/03_neutral_v3__dai"},
            {"tier": "df_emptynull_ed", "variant": "ED", "label": "ED-81",
             "sub": "040_dualforce_dcg_emptynull_w6/04_neutral_v3ed81__dai"}]},
        {"id": "df_emptynull60", "label": "empty-null (w6) — 60-row subset (2026-09-16)", "entries": [
            {"tier": "df_emptynull60_hf", "variant": "HF", "label": "HF-121",
             "sub": "040_dualforce_dcg_emptynull_w6/01_neutral_v3zs60__dai"},
            {"tier": "df_emptynull60_ed", "variant": "ED", "label": "ED-81",
             "sub": "040_dualforce_dcg_emptynull_w6/02_neutral_v3ed81zs60__dai"}]},
    ]},
    {"id": "vfxmaster", "label": "VFXMaster (external) — DCG w6", "arms": [
        {"id": "vfx_xfade", "label": "crossfade (w6)", "entries": [
            {"tier": "vfx_xfade_hf", "variant": "HF", "label": "HF-121",
             "sub": "037_vfxmaster_dcg_w6/01_neutral_v3zs__dai"},
            {"tier": "vfx_xfade_ed", "variant": "ED", "label": "ED-81",
             "sub": "037_vfxmaster_dcg_w6/02_neutral_v3ed81zs__dai"}]},
    ]},
    {"id": "wsweep", "label": "w-sweeps · crossfade (secondary, 152-row grid)", "arms": [
        {"id": "ctt_sweep", "label": "ctt_v2 crossfade", "entries": [
            {"tier": "ctt_sweep_w1",   "variant": "w1",   "label": "w1",   "sub": "015_dcg_w1/01_neutral__dai"},
            {"tier": "ctt_sweep_w1p5", "variant": "w1.5", "label": "w1.5", "sub": "016_dcg_w1p5/01_neutral__dai"},
            {"tier": "ctt_sweep_w3",   "variant": "w3",   "label": "w3",   "sub": "017_dcg_w3/01_neutral__dai"},
            {"tier": "ctt_sweep_w6",   "variant": "w6",   "label": "w6",   "sub": "018_dcg_w6/01_neutral__dai"}]},
        {"id": "df_sweep", "label": "dualforce crossfade", "entries": [
            {"tier": "df_sweep_w1",   "variant": "w1",   "label": "w1",   "sub": "029_dualforce_dcg_w1/01_neutral__dai"},
            {"tier": "df_sweep_w1p5", "variant": "w1.5", "label": "w1.5", "sub": "030_dualforce_dcg_w1p5/01_neutral__dai"},
            {"tier": "df_sweep_w3",   "variant": "w3",   "label": "w3",   "sub": "031_dualforce_dcg_w3/01_neutral__dai"},
            {"tier": "df_sweep_w6",   "variant": "w6",   "label": "w6",   "sub": "032_dualforce_dcg_w6/01_neutral__dai"}]},
    ]},
]

# entries whose HF variant seeds the default selection (the interesting ctt core)
DEFAULT_TIERS = ["df_base_hf", "df_xfade_hf", "df_emptynull_hf"]  # owner 2026-09-24: SEGUE w/o NRG vs crossfade null vs empty null, full grid v3


def video_name(row):
    """The clip filename for a grid row — out_name when baked in, else <item_id>__s<seed>.mp4."""
    if row.get("out_name"):
        return row["out_name"], str(row.get("seed", "42"))
    return "{}__s42.mp4".format(row["item_id"]), "42"


def main():
    cards = {}          # input_key -> card dict
    seeds = set()
    tot_clips = 0
    report = []
    for cat in CATS:
        for arm in cat["arms"]:
            for e in arm["entries"]:
                sub = e["sub"]
                base = REPO / "store/gens" / sub
                gp = base / "grid.jsonl"
                vdir = base / "videos"
                have = set(os.listdir(vdir)) if vdir.is_dir() else set()
                rows = [json.loads(l) for l in open(gp)] if gp.exists() else []
                matched = 0
                for r in rows:
                    k = r["input_key"]
                    c = cards.get(k)
                    if c is None:
                        c = cards[k] = {
                            "key": k,
                            "cell": r.get("cell", ""),
                            "content": r.get("content", ""),
                            "endpoint": r.get("endpoint", ""),
                            "endpoint_class": r.get("endpoint_class", ""),
                            "reference": r.get("reference", ""),
                            "sided": r.get("sided", ""),
                            "prompt": r.get("prompt", ""),
                            "src": resolve_src(r.get("endpoint", ""), r.get("reference", ""), r.get("sided", "")),
                            "slots": {},
                        }
                    fn, sd = video_name(r)
                    rel = "store/gens/{}/videos/{}".format(sub, fn)
                    exists = fn in have
                    if exists:
                        matched += 1
                        tot_clips += 1
                    seeds.add(sd)
                    slot = c["slots"].setdefault(e["tier"], {"videos": {}})
                    if exists:
                        slot["videos"][sd] = rel
                report.append("  {:22s} {:6d} rows  {:6d} clips  [{}]".format(
                    e["tier"], len(rows), matched, sub))

    # drop slots that ended up with no playable clip; drop empty cards
    for c in cards.values():
        c["slots"] = {t: s for t, s in c["slots"].items() if s["videos"]}
    card_list = [c for c in cards.values() if c["slots"]]
    # stable order: cell, endpoint, reference
    card_list.sort(key=lambda c: (c["cell"], c["endpoint"], c["reference"]))

    seed_list = sorted(seeds, key=lambda s: (s != "42", s))  # 42 first
    data = {
        "meta": {
            "rel": "",
            "seeds": seed_list,
            "cards": len(card_list),
            "clips": tot_clips,
            "default": DEFAULT_TIERS,
        },
        "cats": CATS,
        "cards": card_list,
    }

    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")

    n_ep = sum(1 for c in card_list if c["src"]["ep_start"])
    n_end = sum(1 for c in card_list if c["src"]["ep_end"])
    n_ref = sum(1 for c in card_list if c["src"]["ref"])
    print("\n".join(report))
    print("[src ] endpoint-start {}/{} · endpoint-end {} (two-sided) · reference {}/{}".format(
        n_ep, len(card_list), n_end, n_ref, len(card_list)))
    print("\n[done] {} cards · {} clips · seeds {} -> {}".format(
        len(card_list), tot_clips, seed_list, OUT.relative_to(REPO)))


TEMPLATE = r"""<!-- generated by scripts/viewers/build_dcg_sweep.py — do not hand-edit -->
<meta charset="utf-8">
<title>DCG-null sweep</title>
<style>
:root{--bg:#0e1014;--card:#171a1f;--card2:#1d2127;--edge:#2a2f37;--edge2:#3a414c;
 --fg:#dde3ea;--dim:#8b95a1;--dim2:#666f7a;--sel:#3b82f6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);overflow-x:hidden;
 font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h1,h2,h3,h4{margin:0;font-weight:600}
.wrap{max-width:1900px;margin:0 auto;padding:16px}
header{border-bottom:1px solid var(--edge);padding-bottom:10px;margin-bottom:14px}
header h1{font-size:19px;letter-spacing:.2px}
header .sub{color:var(--dim);font-size:12px;margin-top:4px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.panel{background:var(--card);border:1px solid var(--edge);border-radius:10px;padding:14px;margin-bottom:14px}
.panel>h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin-bottom:10px}
.btn{background:var(--card2);border:1px solid var(--edge);color:var(--fg);border-radius:6px;
 padding:5px 10px;cursor:pointer;font-size:11px;font-family:inherit}
.btn:hover{border-color:var(--edge2)}
.lbl{color:var(--dim);font-size:11px}
input.flt{background:var(--card2);border:1px solid var(--edge);color:var(--fg);border-radius:6px;
 padding:5px 9px;font-size:11px;font-family:inherit;min-width:200px}
.seg{display:inline-flex;border:1px solid var(--edge);border-radius:6px;overflow:hidden}
.seg button{background:var(--card2);border:0;border-right:1px solid var(--edge);color:var(--dim);
 padding:5px 10px;cursor:pointer;font-size:11px;font-family:inherit}
.seg button:last-child{border-right:0}
.seg button.on{background:#132b4d;color:var(--fg)}

/* arm selector */
.aptool{display:flex;gap:6px;align-items:center;flex-wrap:wrap;padding-bottom:8px;
 border-bottom:1px solid var(--edge);margin-bottom:8px}
#armlist{max-height:52vh;overflow-y:auto}
#armlist::-webkit-scrollbar{width:9px}
#armlist::-webkit-scrollbar-thumb{background:var(--edge2);border-radius:5px}
.apcat{border:1px solid var(--edge);border-radius:8px;margin-bottom:8px;overflow:hidden}
.apcat>.ch{display:flex;align-items:center;gap:8px;padding:6px 9px;background:var(--card2);
 font-size:11px;font-weight:700;letter-spacing:.05em;text-transform:uppercase}
.apcat>.ch .cdot{width:8px;height:8px;border-radius:2px;flex:none}
.apcat>.ch .sp{flex:1}
.apcat>.ch button{background:none;border:1px solid var(--edge);border-radius:4px;color:var(--dim);
 cursor:pointer;font-size:10px;padding:1px 6px}
.apcat>.ch button:hover{color:var(--fg);border-color:var(--edge2)}
.aparm{display:flex;align-items:baseline;gap:8px;padding:5px 9px;border-top:1px solid #20242b;flex-wrap:wrap}
.aparm .an{flex:0 0 175px;font-size:11.5px;color:var(--fg)}
.pillrow{display:flex;gap:4px;flex-wrap:wrap}
.pill{font-size:10.5px;padding:3px 8px;border-radius:6px;border:1px solid var(--edge2);
 background:#12151b;color:var(--dim);cursor:pointer;user-select:none;font-weight:600;transition:.1s}
.pill:hover{border-color:var(--sel);color:var(--fg)}
.pill.on{background:#132b4d;border-color:var(--sel);color:#dbeafe}
.pill.v-HF.on{background:#0e2f26;border-color:#1c6b57;color:#5eead4}
.pill.v-ED.on{background:#33240a;border-color:#7a5410;color:#fbbf24}

/* cards */
.empty{padding:36px;text-align:center;color:var(--dim2)}
.more{padding:12px;text-align:center}
.card{background:var(--card);border:1px solid var(--edge);border-radius:10px;
 margin-bottom:12px;overflow:hidden;min-width:0;max-width:100%}
.chead{display:flex;align-items:center;gap:10px;padding:9px 12px;background:#13161b;
 border-bottom:1px solid var(--edge);flex-wrap:wrap}
.chead b{font-size:13px}
.chead .dim{color:var(--dim);font-size:11px}
.chead .pill2{margin-left:auto}
.pbtn{background:#132b4d;border:1px solid var(--sel);color:#cfe2fb;border-radius:5px;
 padding:3px 9px;cursor:pointer;font-size:10.5px;font-family:inherit}
.grp{padding:10px 12px}
.row{display:flex;flex-wrap:nowrap;gap:10px;overflow-x:auto;overflow-y:hidden;padding-bottom:6px;align-items:stretch}
.row::-webkit-scrollbar{height:9px}
.row::-webkit-scrollbar-thumb{background:var(--edge2);border-radius:5px}
.row::-webkit-scrollbar-track{background:#0d0f13}
/* the row splits into an INPUTS band (source media the gen saw) and, past a dashed
   divider, the OUT band (per-arm generated clips): [inputs] | [arm1] [arm2] … */
.band{display:flex;gap:10px;flex:0 0 auto;align-items:stretch}
.out-band{border-left:2px dashed var(--edge2);margin-left:6px;padding-left:12px}
.cellbox.src{background:#111c22;border-color:#1e3a44}
.cellbox.src h4{color:#8fd3e6}
.cellbox.ref{background:#1a1330;border-color:#3b2f5e}
.cellbox.ref h4{color:#c4b5fd}
.pptxt{flex:0 0 230px;background:#0f1620;border:1px solid #24303c;border-radius:8px;padding:8px;
 display:flex;flex-direction:column;gap:5px}
.pptxt h4{font-size:11px;color:#8fd3e6}
.pptxt .t{font-size:11px;color:#c7d0da;line-height:1.45;overflow-y:auto;max-height:180px}
.pptxt .meta{font-size:10px;color:var(--dim2);word-break:break-all}
.cellbox{flex:0 0 210px;background:var(--card2);border:1px solid var(--edge);border-radius:8px;
 padding:8px;display:flex;flex-direction:column;gap:6px}
.cellbox h4{font-size:11px;line-height:1.35}
.cellbox h4 .arm{color:var(--dim);font-weight:400;display:block;font-size:10px}
.cellbox video{width:100%;aspect-ratio:3/4;object-fit:cover;background:#000;border-radius:5px;display:block}
.cellbox .miss{width:100%;aspect-ratio:3/4;display:grid;place-items:center;color:var(--dim2);
 background:#0b0d10;border-radius:5px;font-size:11px}
.cellbox .fn{color:var(--dim2);font-size:9.5px;word-break:break-all;line-height:1.3}
.cellbox.out{box-shadow:inset 3px 0 0 var(--sel)}
.cellbox.blank{background:transparent;border:1px dashed #2a3038;opacity:.5;justify-content:center;
 min-height:150px;text-align:center}
.cellbox.blank .bl{color:var(--dim2);font-size:10px}
.vtag{display:inline-block;font-size:9px;font-weight:700;padding:1px 6px;border-radius:5px;margin-left:4px;
 background:#132b4d;color:#dbeafe;border:1px solid var(--sel)}
.vtag.v-HF{background:#0e2f26;color:#5eead4;border:1px solid #1c6b57}
.vtag.v-ED{background:#33240a;color:#fbbf24;border:1px solid #7a5410}
</style>

<div class="wrap">
<header>
  <h1>DCG-null sweep — arm comparison</h1>
  <div class="sub" id="sub"></div>
</header>

<div class="panel">
  <h2>arms</h2>
  <div class="aptool">
    <button class="btn" onclick="armAll(1)">select all</button>
    <button class="btn" onclick="armAll(0)">clear</button>
    <span class="lbl" style="margin-left:6px" title="applies only to arms that already have a selected pill">for chosen arms:</span>
    <button class="btn" onclick="bulkVar('HF',1)">HF on</button>
    <button class="btn" onclick="bulkVar('HF',0)">HF off</button>
    <button class="btn" onclick="bulkVar('ED',1)">ED on</button>
    <button class="btn" onclick="bulkVar('ED',0)">ED off</button>
    <span class="lbl" style="margin-left:auto" id="armcount"></span>
  </div>
  <div id="armlist"></div>
</div>

<div class="panel">
  <h2 style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">examples
    <span class="lbl" id="cardinfo" style="text-transform:none;letter-spacing:0"></span>
    <span style="margin-left:auto;display:flex;gap:8px;align-items:center">
      <span class="lbl">seed</span><span class="seg" id="seeds"></span>
      <input class="flt" id="flt" placeholder="filter: endpoint / reference / cell">
    </span></h2>
  <div id="cards"></div>
  <div class="more" id="more"></div>
</div>
</div>

<script>
const D = /*__DATA__*/null;
const PAGE = 20;
let shown = PAGE;
const esc = s => String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/"/g,"&quot;");

// ── catalog + selection state ──────────────────────────────────────────────
const CATS = D.cats;
const ENTRY = {};                                  // tier -> {cat, catLabel, armLabel, variant, label}
for(const c of CATS) for(const a of c.arms) for(const e of a.entries)
  ENTRY[e.tier] = {cat:c.id, catLabel:c.label, armLabel:a.label, variant:e.variant, label:e.label};
const ALL_TIERS = Object.keys(ENTRY);
const CPAL = ["#60a5fa","#a78bfa","#4ade80","#f472b6","#e8d49a","#2dd4bf"];
const CAT_COLOR = Object.fromEntries(CATS.map((c,i)=>[c.id, CPAL[i % CPAL.length]]));
const LS = "dcg_sweep_sel";

function loadSel(){
  let arr = null;
  try{ arr = JSON.parse(localStorage.getItem(LS)); }catch(e){}
  if(Array.isArray(arr)){ const s = new Set(arr.filter(t=>ENTRY[t])); if(s.size) return s; }
  return new Set(D.meta.default.filter(t=>ENTRY[t]));
}
let tiers = loadSel();
function saveSel(){ try{ localStorage.setItem(LS, JSON.stringify([...tiers])); }catch(e){} }

// display order = catalog order, selected only
function selTiers(){
  const out = [];
  for(const c of CATS) for(const a of c.arms) for(const e of a.entries)
    if(tiers.has(e.tier)) out.push(e.tier);
  return out;
}

let seed = D.meta.seeds[0] || "42";
let flt = "";

// ── arm panel ──────────────────────────────────────────────────────────────
function drawArms(){
  document.getElementById("armcount").textContent =
    tiers.size + " of " + ALL_TIERS.length + " entries selected";
  let h = "";
  for(const c of CATS){
    h += `<div class="apcat"><div class="ch">
        <span class="cdot" style="background:${CAT_COLOR[c.id]}"></span>${esc(c.label)}
        <span class="sp"></span>
        <button onclick="catAll('${c.id}',1)">all</button>
        <button onclick="catAll('${c.id}',0)">none</button></div>`;
    for(const a of c.arms){
      h += `<div class="aparm"><span class="an">${esc(a.label)}</span><span class="pillrow">` +
        a.entries.map(e =>
          `<span class="pill v-${esc(e.variant)} ${tiers.has(e.tier)?"on":""}"
             onclick="togTier('${e.tier}')">${esc(e.label)}</span>`).join("") +
        `</span></div>`;
    }
    h += `</div>`;
  }
  document.getElementById("armlist").innerHTML = h;
}
function togTier(t){ tiers.has(t)?tiers.delete(t):tiers.add(t); saveSel(); shown=PAGE; render(); }
function armAll(on){ tiers = on ? new Set(ALL_TIERS) : new Set(); saveSel(); shown=PAGE; render(); }
function catAll(cid, on){
  const c = CATS.find(x=>x.id===cid); if(!c) return;
  for(const a of c.arms) for(const e of a.entries){ if(on) tiers.add(e.tier); else tiers.delete(e.tier); }
  saveSel(); shown=PAGE; render();
}
// bulk variant — only over arms that already have >=1 selected entry
function bulkVar(v, on){
  const eng = new Set();
  for(const t of tiers){ const m = ENTRY[t]; if(m) eng.add(m.cat+"::"+m.armLabel); }
  for(const [t, m] of Object.entries(ENTRY)){
    if(m.variant !== v) continue;
    if(!eng.has(m.cat+"::"+m.armLabel)) continue;
    if(on) tiers.add(t); else tiers.delete(t);
  }
  saveSel(); shown=PAGE; render();
}

// ── cards ───────────────────────────────────────────────────────────────────
function matchFlt(c){
  if(!flt) return true;
  const hay = (c.cell+" "+c.endpoint+" "+c.endpoint_class+" "+c.reference+" "+c.content).toLowerCase();
  return flt.split(/\s+/).every(w => hay.includes(w));
}
function selectedCards(){
  const st = selTiers();
  const out = [];
  for(const c of D.cards){
    if(!matchFlt(c)) continue;
    if(st.some(t => c.slots[t])) out.push(c);
  }
  return out;
}
function vid(src){
  return src ? `<video preload="metadata" muted loop playsinline data-src="${D.meta.rel}${src}"></video>`
             : `<div class="miss">no clip</div>`;
}
function cardHtml(c){
  const st = selTiers();
  const cells = st.map(t=>{
    const m = ENTRY[t];
    const slot = c.slots[t];
    if(!slot) return `<div class="cellbox blank"><div class="bl"><b>${esc(m.armLabel)}</b><br>${esc(m.label)}<br>no clip</div></div>`;
    const src = slot.videos[seed] || Object.values(slot.videos)[0];
    const fn = src ? src.split("/").pop() : "";
    return `<div class="cellbox out" style="box-shadow:inset 3px 0 0 ${CAT_COLOR[m.cat]}">
      <h4>${esc(m.armLabel)}<span class="vtag v-${esc(m.variant)}">${esc(m.label)}</span>
        <span class="arm">${esc(m.catLabel.split(" ")[0])}</span></h4>
      ${vid(src)}<div class="fn">${esc(fn)}</div></div>`;
  }).join("");
  const info = `<div class="pptxt"><h4>INPUT</h4>
      <div class="meta"><b>endpoint</b> ${esc(c.endpoint)} · ${esc(c.endpoint_class)}</div>
      <div class="meta"><b>reference</b> ${esc(c.reference)}</div>
      <div class="meta"><b>cell</b> ${esc(c.cell)} · ${esc(c.content)} · ${esc(c.sided)}-sided</div>
      <div class="meta"><b>key</b> ${esc(c.key)}</div>
      <h4 style="margin-top:4px">PROMPT</h4><div class="t">${esc(c.prompt)}</div></div>`;
  // source input media the generation was conditioned on (endpoint anchors + reference demo)
  const s = c.src || {};
  const srcTile = (src, cls, title, name) =>
    `<div class="cellbox ${cls}"><h4>${title}<span class="arm">${esc(name||"")}</span></h4>${vid(src)}</div>`;
  let srcTiles = srcTile(s.ep_start, "src", "endpoint · start", s.ep_name||c.endpoint);
  if(c.sided === "two") srcTiles += srcTile(s.ep_end, "src", "endpoint · end", s.ep_name||c.endpoint);
  srcTiles += srcTile(s.ref, "ref", "reference · demo", s.ref_name||c.reference);
  return `<div class="card">
    <div class="chead"><b>${esc(c.endpoint)}</b>
      <span class="dim">ref <b>${esc(c.reference)}</b> · ${esc(c.cell)}</span>
      <span class="pill2"><button class="pbtn" onclick="playAll(this)">↻ restart row synced</button></span></div>
    <div class="grp"><div class="row">
      <div class="band">${info}${srcTiles}</div>
      <div class="band out-band">${cells}</div>
    </div></div>
  </div>`;
}
function drawCards(){
  const cs = selectedCards();
  document.getElementById("cardinfo").textContent =
    `${cs.length} input cards · showing ${Math.min(shown,cs.length)}`;
  const el = document.getElementById("cards");
  if(!cs.length){ el.innerHTML = "<div class='empty'>no cards — pick some arms above</div>";
    document.getElementById("more").innerHTML=""; return; }
  el.innerHTML = cs.slice(0,shown).map(cardHtml).join("");
  document.getElementById("more").innerHTML = cs.length>shown
    ? `<button class="btn" onclick="shown+=${PAGE};drawCards()">show ${Math.min(PAGE,cs.length-shown)} more of ${cs.length-shown}</button>` : "";
  lazy();
}

// ── video machinery ─────────────────────────────────────────────────────────
let io;
function lazy(){
  if(io) io.disconnect();
  io = new IntersectionObserver(es=>es.forEach(e=>{
    const v = e.target;
    if(e.isIntersecting){ if(!v.src) v.src = v.dataset.src; v.play().catch(()=>{}); }
    else v.pause();
  }), {rootMargin:"300px"});
  document.querySelectorAll("video").forEach(v=>io.observe(v));
}
function playAll(btn){
  const vids = [...btn.closest(".card").querySelectorAll("video")];
  vids.forEach(v=>{ if(!v.src) v.src = v.dataset.src; });
  const go = () => vids.forEach(v=>{ v.currentTime = 0; v.play().catch(()=>{}); });
  const notReady = vids.filter(v=>v.readyState < 1);
  if(!notReady.length){ go(); return; }
  let left = notReady.length;
  notReady.forEach(v=>v.addEventListener("loadedmetadata", ()=>{ if(--left===0) go(); }, {once:true}));
}

// ── chrome ──────────────────────────────────────────────────────────────────
function drawSeeds(){
  const el = document.getElementById("seeds");
  if(D.meta.seeds.length < 2){ el.innerHTML = `<button class="on">${seed}</button>`; return; }
  el.innerHTML = D.meta.seeds.map(s=>`<button class="${s===seed?"on":""}" onclick="setSeed('${s}')">${s}</button>`).join("");
}
function setSeed(s){ seed = s; drawSeeds(); drawCards(); }
document.getElementById("flt").addEventListener("input", e=>{ flt = e.target.value.trim().toLowerCase(); shown=PAGE; drawCards(); });

function render(){ shown = Math.max(shown, PAGE); drawArms(); drawSeeds(); drawCards(); }

document.getElementById("sub").innerHTML =
  `<b class="mono">${D.meta.cards}</b> input cards · <b class="mono">${D.meta.clips}</b> clips · ` +
  `seeds ${D.meta.seeds.join(", ")} · lined up by <span class="mono">input_key</span> — each card shows the ` +
  `source inputs it was conditioned on (endpoint anchor(s) + reference demo), then the per-arm generated clips; ` +
  `a blank cell = that arm has no clip for that input`;
render();
</script>
"""


if __name__ == "__main__":
    main()
