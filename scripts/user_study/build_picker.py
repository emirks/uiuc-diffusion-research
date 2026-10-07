#!/usr/bin/env python3
"""Build the organizer row-picker page for the SEGUE human study.

The owner chooses the study rows themselves, fairly, from the candidates that
`select_pairs.enumerate_candidates` computes. The page shows, per candidate row,
its INPUTS (given start / given end / reference) and — behind an off-by-default
toggle — the GENERATED videos too (SEGUE and every prior-work opponent of the
task), so the owner can eyeball outputs before committing. Whether that toggle
was ever switched on is recorded and exported into picks.json as
`outputs_revealed`, so the provenance of the selection is not lost.

Run with the aarch64 media python (imports select_pairs, which imports
imageio_ffmpeg at module load):
    $LAB/envs-aarch64/ltx2/bin/python scripts/user_study/build_picker.py

Writes under outputs/viewers/user_study_picker/:
    index.html      the picker page (one file, vanilla HTML/CSS/JS)
    rows_all.json   candidate rows per task + output URLs + the rows.json selection
    std121 -> ../../../data/processed/transitions_std121     (reference videos)
    conds  -> ../../../eval_ladder/conds                     (given windows)
    gens/<slug> -> ../../../../store/gens/<...>              (one per system/dir)
so every media path in the page is relative to the viewer dir (viewer rule).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_pairs as sp  # noqa: E402  (candidate enumeration + dir map reused)

REPO = sp.REPO
STUDY = sp.OUT  # misc/2026-09-22_user_study
VIEWER = os.path.join(REPO, "outputs", "viewers", "user_study_picker")

STD121_MARKER = "/data/processed/transitions_std121/"
SEEDS = ("42", "43")
NOTES = {}  # filled in main() from the owner's curated collections

# store gen dir (imported from select_pairs, never duplicated here) -> viewer
# symlink slug. Keys come straight from select_pairs so the dir list has one
# home; only the short slug names are picker-local.
# Extra SEGUE arms shown for browsing only (owner 2026-09-22): label -> (HF dir, ED dir).
# They are NOT study systems; select_pairs.py never sees them.
EXTRA_SEGUE = {
    "SEGUE w=3": ("store/gens/031_dualforce_dcg_w3/03_neutral_v3__dai",
                  "store/gens/031_dualforce_dcg_w3/04_neutral_v3ed81__dai"),
    "SEGUE w=6 effect": ("store/gens/032_dualforce_dcg_w6/05_effect_v3__dai",
                         "store/gens/032_dualforce_dcg_w6/06_effect_v3ed81__dai"),
}

DIR_SLUG = {
    sp.SEGUE_HF: "segue_hf",
    sp.SEGUE_ED: "segue_ed",
    # the study's per-task SEGUE arms (select_pairs.SEGUE_ARMS); transfer = w=3 effect since 2026-09-25
    sp.SEGUE_ARMS["one"][0]: "segue_w3eff_hf",
    sp.SEGUE_ARMS["one"][1]: "segue_w3eff_ed",
    EXTRA_SEGUE["SEGUE w=3"][0]: "segue_w3_hf",
    EXTRA_SEGUE["SEGUE w=3"][1]: "segue_w3_ed",
    EXTRA_SEGUE["SEGUE w=6 effect"][0]: "segue_w6eff_hf",
    EXTRA_SEGUE["SEGUE w=6 effect"][1]: "segue_w6eff_ed",
    sp.TASK_OPP["two"]["Base LTX-2"]:   "base_ltx2",
    sp.TASK_OPP["two"]["VACE"]:         "vace",
    sp.TASK_OPP["two"]["refVFX"]:       "refvfx_teg",
    sp.TASK_OPP["two"]["Wan2.1 FLF2V"]: "wan_flf2v",
    sp.TASK_OPP["one"]["VAP"]:          "vap",
    sp.TASK_OPP["one"]["VFXMaster"]:    "vfxmaster",
    sp.TASK_OPP["one"]["refVFX"]:       "refvfx_vet",
}


def rel_reference(abs_path):
    """Absolute transitions_std121 path -> path relative to the viewer dir."""
    if STD121_MARKER not in abs_path:
        sys.exit(f"reference not under transitions_std121: {abs_path}")
    return "std121/" + abs_path.split(STD121_MARKER, 1)[1]


def out_entry(store_dir, item_id, missing):
    """One system's output URL template ({seed} placeholder) + missing-seed check.

    Returns the `gens/<slug>/videos/<item_id>__s{seed}.mp4` URL (page fills
    {seed}); appends missing seeds (per store_dir) to the caller's collector.
    """
    slug = DIR_SLUG[store_dir]
    url = f"gens/{slug}/videos/{item_id}__s{{seed}}.mp4"
    miss = []
    for s in SEEDS:
        abs_p = sp.apath(os.path.join(store_dir, "videos", f"{item_id}__s{s}.mp4"))
        if not os.path.exists(abs_p):
            miss.append(s)
        else:
            missing["distinct_out"].add(abs_p)
    return url, miss


def build_rows():
    """Per-task candidate rows with input + output media paths (relative)."""
    out = {"two": [], "one": []}
    order = {}
    distinct_ref = set()
    distinct_conds = set()
    missing = {"list": [], "distinct_out": set()}
    for task in ("two", "one"):
        opp = sp.TASK_OPP[task]
        order[task] = ["SEGUE"] + list(EXTRA_SEGUE.keys()) + list(opp.keys())
        ce = sp.enumerate_candidates(task)
        extra_maps = {lbl: (sp.key_map(sp.load_grid(hf), task), sp.key_map(sp.load_grid(ed), task))
                      for lbl, (hf, ed) in EXTRA_SEGUE.items()}
        seg_hf_m = ce["seg_hf_m"]
        seg_ed_m = ce["seg_ed_m"]
        opp_maps = ce["opp_maps"]
        for k in ce["candidates"]:
            ep, ref = k
            is_ed = k not in seg_hf_m
            sr = seg_hf_m.get(k) or seg_ed_m[k]
            ref_url = rel_reference(sp.reference_path(sr, opp_maps, k))
            start_url = f"conds/{ep}_start9.mp4"
            end_url = f"conds/{ep}_end9.mp4" if task == "two" else None
            # assert given windows exist (fast; no probing)
            if not os.path.exists(sp.apath(os.path.join(
                    "eval_ladder", "conds", ep + "_start9.mp4"))):
                sys.exit(f"missing start window for {ep}")
            if end_url and not os.path.exists(sp.apath(os.path.join(
                    "eval_ladder", "conds", ep + "_end9.mp4"))):
                sys.exit(f"missing end window for {ep}")
            distinct_ref.add(ref_url)
            distinct_conds.add(start_url)
            if end_url:
                distinct_conds.add(end_url)

            # generated videos: SEGUE (its own grid) + every opponent of the task
            outputs = {}
            out_miss = {}
            seg_dir = sp.SEGUE_ARMS[task][1] if is_ed else sp.SEGUE_ARMS[task][0]   # the study arm of this task
            url, miss = out_entry(seg_dir, sr["item_id"], missing)
            outputs["SEGUE"] = url
            if miss:
                out_miss["SEGUE"] = miss
            for lbl, (hf_dir, ed_dir) in EXTRA_SEGUE.items():
                hf_m, ed_m = extra_maps[lbl]
                xrow = hf_m.get(k) or ed_m.get(k)
                if xrow is None:
                    out_miss[lbl] = list(SEEDS)
                    continue
                url, miss = out_entry(ed_dir if k not in hf_m else hf_dir, xrow["item_id"], missing)
                outputs[lbl] = url
                if miss:
                    out_miss[lbl] = miss
            for lbl in opp:
                orow = opp_maps[lbl][k]
                url, miss = out_entry(opp[lbl], orow["item_id"], missing)
                outputs[lbl] = url
                if miss:
                    out_miss[lbl] = miss
            for lbl, seeds in out_miss.items():
                for s in seeds:
                    missing["list"].append(
                        {"task": task, "endpoint": ep, "system": lbl, "seed": s})

            out[task].append({
                "endpoint": ep,
                "reference": ref,
                "cls": sr["gt_pool_class"],
                "cell": sr["cell"],
                "content": sr["content"],
                "note": NOTES.get((task, ep, ref), ""),
                "ed": is_ed,  # SEGUE clip served by grid 04 (ED81)
                "ref_url": ref_url,
                "start": start_url,
                "end": end_url,
                "outputs": outputs,
                "outputs_missing": out_miss,
            })
    return out, order, distinct_ref, distinct_conds, missing


def owner_notes():
    """(task, endpoint, reference) -> the owner's note from the curated collections (if mapped)."""
    notes = {}
    path = os.path.join(STUDY, "owner_collection_map.json")
    if os.path.exists(path):
        with open(path) as f:
            for rows in json.load(f).values():
                for r in rows:
                    if r.get("notes", "").strip():
                        notes[(r["task"], r["endpoint"], r["reference"])] = r["notes"].strip()
    return notes


def proposed_selection():
    """misc/.../picks_proposed.json (owner picks + rule draw) -> {task: [[ep, ref], ...]} or empty."""
    path = os.path.join(STUDY, "picks_proposed.json")
    if not os.path.exists(path):
        return {"two": [], "one": []}
    with open(path) as f:
        d = json.load(f)
    return {"two": d.get("two", []), "one": d.get("one", [])}


def screened_out(rows):
    """Rows the owner screened OUT on 2026-09-22 = candidates not in the owner's collection
    (misc/.../owner_collection_map.json: teg_user_study / vfx_transfer_user_study). {task: [[ep, ref], ...]}"""
    path = os.path.join(STUDY, "owner_collection_map.json")
    out = {"two": [], "one": []}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        cm = json.load(f)
    for task, coll in (("two", "teg_user_study"), ("one", "vfx_transfer_user_study")):
        kept = {(e["endpoint"], e["reference"]) for e in cm.get(coll, [])}
        out[task] = [[r["endpoint"], r["reference"]] for r in rows[task]
                     if (r["endpoint"], r["reference"]) not in kept]
    return out


def initial_selection():
    """The current frozen selection (misc/.../rows.json) -> {task: [[ep, ref], ...]}."""
    init = {"two": [], "one": []}
    path = os.path.join(STUDY, "rows.json")
    if os.path.exists(path):
        with open(path) as f:
            for r in json.load(f):
                init[r["task"]].append([r["endpoint"], r["reference"]])
    return init


def symlink(link, target):
    if os.path.islink(link) or os.path.exists(link):
        if os.path.islink(link) and os.readlink(link) == target:
            return
        os.remove(link)
    os.symlink(target, link)


def main():
    os.makedirs(VIEWER, exist_ok=True)
    symlink(os.path.join(VIEWER, "std121"),
            "../../../data/processed/transitions_std121")
    symlink(os.path.join(VIEWER, "conds"), "../../../eval_ladder/conds")
    gens_dir = os.path.join(VIEWER, "gens")
    os.makedirs(gens_dir, exist_ok=True)
    # one symlink per system/dir; target is relative to gens/ (4 levels to repo)
    for store_dir, slug in DIR_SLUG.items():
        symlink(os.path.join(gens_dir, slug), "../../../../" + store_dir)

    global NOTES
    NOTES = owner_notes()
    rows, order, distinct_ref, distinct_conds, missing = build_rows()
    data = {"two": rows["two"], "one": rows["one"],
            "output_order": order, "initial": initial_selection(),
            "proposed": proposed_selection(), "screened_out": screened_out(rows)}
    with open(os.path.join(VIEWER, "rows_all.json"), "w") as f:
        json.dump(data, f, indent=1)

    with open(os.path.join(VIEWER, "index.html"), "w") as f:
        f.write(PAGE)

    n_strips = len(rows["two"]) + len(rows["one"])
    n_out_refs = sum(len(r["outputs"]) for r in rows["two"] + rows["one"]) * len(SEEDS)
    print(f"[picker] candidates/strips: two={len(rows['two'])} one={len(rows['one'])} "
          f"total={n_strips}")
    print(f"[picker] distinct reference files={len(distinct_ref)} "
          f"distinct conds files={len(distinct_conds)}")
    print(f"[picker] output refs (rows x systems x {len(SEEDS)} seeds)={n_out_refs} "
          f"distinct output files={len(missing['distinct_out'])}")
    if missing["list"]:
        print(f"[picker] MISSING output clips: {len(missing['list'])}")
        for m in missing["list"]:
            print(f"[picker]   miss  task={m['task']} {m['system']} "
                  f"{m['endpoint']} s{m['seed']}")
    else:
        print("[picker] all referenced output clips exist for both seeds")
    print(f"[picker] initial selection: two={len(data['initial']['two'])} "
          f"one={len(data['initial']['one'])}")
    print(f"[picker] wrote {VIEWER}/index.html, rows_all.json + std121/, conds/, "
          f"gens/* symlinks")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SEGUE study — row picker (organizer)</title>
<style>
  :root{
    --bg:#f6f7f8; --panel:#ffffff; --ink:#1c2024; --muted:#6b7280;
    --line:#e2e5e9; --accent:#2f6f4f; --accent-bg:#e7f2ec; --warn:#8a5300;
    --pick:#1f6feb;
    --cell-h:200px; --cell-w:calc(var(--cell-h) * 3 / 4); --topbar-h:48px;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  a{color:var(--pick)}
  .topbar{position:sticky;top:0;z-index:20;background:var(--panel);
    border-bottom:1px solid var(--line);padding:8px 14px;display:flex;
    flex-wrap:wrap;gap:8px 16px;align-items:center}
  .title{font-weight:600}
  .title .sub{font-weight:400;color:var(--muted);font-size:12px;margin-left:8px}
  .counter{font-variant-numeric:tabular-nums;color:var(--muted)}
  .counter b{color:var(--ink)}
  .counter .hit{color:var(--accent)}
  .counter .over{color:var(--warn)}
  .tabs{display:flex;gap:6px}
  .tab{background:var(--bg);border:1px solid var(--line);border-radius:6px;
    padding:5px 10px;cursor:pointer;font:inherit;color:var(--ink)}
  .tab.active{background:var(--accent-bg);border-color:var(--accent);
    color:var(--accent);font-weight:600}
  .ctl{display:flex;gap:6px;align-items:center;color:var(--muted);font-size:12px}
  .ctl input[type=range]{width:120px}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
  .seg button{background:var(--bg);border:0;border-left:1px solid var(--line);
    padding:4px 10px;cursor:pointer;font:inherit;color:var(--ink)}
  .seg button:first-child{border-left:0}
  .seg button.on{background:var(--accent-bg);color:var(--accent);font-weight:600}
  .actions{margin-left:auto;display:flex;gap:6px}
  button.btn,.actions button,.tb-btn{background:var(--bg);border:1px solid var(--line);
    border-radius:6px;padding:5px 10px;cursor:pointer;font:inherit;color:var(--ink)}
  button.btn:hover,.actions button:hover,.tb-btn:hover{border-color:var(--muted)}
  .tb-btn.on{background:var(--accent-bg);border-color:var(--accent);color:var(--accent)}
  .io{background:var(--panel);border-bottom:1px solid var(--line);padding:10px 14px}
  .io textarea{width:100%;height:120px;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;
    border:1px solid var(--line);border-radius:6px;padding:8px;resize:vertical}
  .io-actions{margin-top:8px;display:flex;gap:8px;align-items:center}
  .io-msg{color:var(--muted);font-size:12px}
  .panel{padding:12px 14px 60px}
  .cls{margin:0 0 18px;border:1px solid var(--line);border-radius:8px;
    background:var(--panel);overflow:hidden}
  .cls-head{position:sticky;top:var(--topbar-h);z-index:10;background:var(--panel);
    border-bottom:1px solid var(--line);padding:8px 12px;display:flex;
    gap:12px;align-items:center;flex-wrap:wrap}
  .cls-name{font-weight:600}
  .cls-count{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
  .cls-count b{color:var(--accent)}
  .cls-head .spacer{flex:1}
  .cls-head button{font-size:12px;padding:3px 8px}
  .strips{display:flex;flex-direction:column}
  .strip{display:flex;gap:12px;align-items:flex-start;padding:10px 12px;
    border-top:1px solid var(--line)}
  .strip:first-child{border-top:0}
  .strip.sel{background:rgba(31,111,235,.06);box-shadow:inset 3px 0 0 var(--pick)}
  .left{flex:0 0 210px;display:flex;flex-direction:column;gap:4px;min-width:0}
  .pickrow{display:flex;gap:8px;align-items:flex-start;cursor:pointer;user-select:none}
  .pickrow input{width:16px;height:16px;margin-top:1px;flex:0 0 auto}
  .ep{font-weight:600;word-break:break-all;font-size:13px}
  .refname{color:var(--muted);font-size:12px;word-break:break-all}
  .badges{display:flex;gap:5px;flex-wrap:wrap;margin-top:2px}
  .note{font-size:11.5px;color:var(--warn);margin-top:3px;white-space:pre-wrap;word-break:break-word}
  .task-head{font-weight:650;font-size:15px;margin:18px 0 8px;color:var(--ink)}
  .badge{font-size:10px;padding:1px 6px;border-radius:10px;border:1px solid var(--line);
    color:var(--muted);white-space:nowrap}
  .badge.ed{color:var(--warn);border-color:#e6d3ad;background:#fbf3e2}
  .badge.hf{color:var(--accent);border-color:#c9e3d5;background:var(--accent-bg)}
  .sync{align-self:flex-start;margin-top:2px;font-size:11px;padding:2px 8px;
    background:var(--bg);border:1px solid var(--line);border-radius:6px;
    cursor:pointer;color:var(--ink)}
  .sync:hover{border-color:var(--muted)}
  .cells{flex:1;min-width:0;display:flex;gap:8px;align-items:flex-start;
    overflow-x:auto;padding-bottom:4px}
  .cell{margin:0;flex:0 0 auto;width:var(--cell-w)}
  .cell figcaption{font-size:11px;color:var(--muted);text-align:center;
    margin-bottom:2px;height:15px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
  .cell.out figcaption{color:var(--ink);font-weight:600}
  .vid{height:var(--cell-h);width:var(--cell-w);background:#000;border-radius:4px;
    display:block;object-fit:contain;cursor:pointer}
  .vid.miss{display:flex;align-items:center;justify-content:center;color:var(--muted);
    border:1px dashed var(--line);background:var(--bg);font-size:11px;cursor:default}
  .divider{flex:0 0 1px;align-self:stretch;background:var(--line);margin:0 6px}
  .empty{color:var(--muted);padding:20px}
</style>
</head>
<body>
<header class="topbar">
  <div class="title">SEGUE study — row picker
    <span class="sub">organizer only</span></div>
  <div class="counter" id="counter">loading…</div>
  <div class="tabs">
    <button class="tab active" data-tab="two">Task two</button>
    <button class="tab" data-tab="one">Task one</button>
    <button class="tab" data-tab="picked">Picked</button>
    <button class="tab" data-tab="out" title="rows the owner screened out on 2026-09-22 (not in the study)">Screened out</button>
  </div>
  <label class="ctl">size <input type="range" id="size" min="120" max="320" step="10">
    <span id="size-val"></span></label>
  <span class="ctl">seed <span class="seg" id="seed-seg">
    <button data-seed="42">42</button><button data-seed="43">43</button></span></span>
  <label class="ctl"><input type="checkbox" id="showgen"> Show generated videos</label>
  <button class="tb-btn" id="pauseall">Pause all</button>
  <div class="actions">
    <button id="btn-export">Export picks.json</button>
    <button id="btn-import">Import</button>
    <button id="btn-proposed" title="Load the proposed selection (owner picks + one-per-class draw)">Load proposed</button>
    <button id="btn-reset" title="Reload the current rows.json selection">Load rows.json</button>
  </div>
</header>

<section class="io" id="io" hidden>
  <textarea id="io-text" spellcheck="false" placeholder='{"two": [["endpoint","reference"], ...], "one": [...]}'></textarea>
  <div class="io-actions">
    <button id="io-apply" class="btn">Apply pasted JSON</button>
    <button id="io-copy" class="btn">Copy</button>
    <button id="io-close" class="btn">Close</button>
    <span class="io-msg" id="io-msg"></span>
  </div>
</section>

<main class="panel" id="panel"></main>

<script>
const SEP = "\u0001";   // unit separator; never appears in endpoints/references
const TARGET = 30;
const LS_KEY = "segue_picker_sel_v1";
const LS_SIZE = "segue_picker_size_v1";
const LS_SEED = "segue_picker_seed_v1";
const LS_SHOWGEN = "segue_picker_showgen_v1";
const LS_REVEAL = "segue_picker_revealed_v1";
const kOf = (ep, ref) => ep + SEP + ref;
const splitK = k => { const i = k.indexOf(SEP); return [k.slice(0, i), k.slice(i + 1)]; };
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let DATA = null;                       // {two, one, output_order, initial}
let valid = { two: new Set(), one: new Set() };   // candidate keys per task
let sel = { two: new Set(), one: new Set() };      // selected keys per task
let cur = "two";
let H = 200, SEED = "42", SHOWGEN = false, PAUSED = false;
let REVEALED = false, REVEALED_AT = null;
let io = null;                         // one IntersectionObserver for lazy media

// ---- prefs (localStorage) ---------------------------------------------------
function loadPrefs() {
  try { const v = parseInt(localStorage.getItem(LS_SIZE), 10); if (v >= 120 && v <= 320) H = v; } catch (e) {}
  try { const s = localStorage.getItem(LS_SEED); if (s === "42" || s === "43") SEED = s; } catch (e) {}
  try { SHOWGEN = localStorage.getItem(LS_SHOWGEN) === "1"; } catch (e) {}
  try {
    const r = JSON.parse(localStorage.getItem(LS_REVEAL) || "null");
    if (r && r.revealed) { REVEALED = true; REVEALED_AT = r.at || null; }
  } catch (e) {}
}
function saveReveal() {
  try { localStorage.setItem(LS_REVEAL, JSON.stringify({ revealed: REVEALED, at: REVEALED_AT })); } catch (e) {}
}

// ---- selection --------------------------------------------------------------
function loadSel() {
  let stored = null;
  try { stored = JSON.parse(localStorage.getItem(LS_KEY) || "null"); } catch (e) {}
  const src = stored || DATA.initial;
  for (const t of ["two", "one"]) {
    sel[t] = new Set();
    for (const [ep, ref] of (src[t] || [])) {
      const k = kOf(ep, ref);
      if (valid[t].has(k)) sel[t].add(k);   // drop anything not a candidate
    }
  }
}
function saveSel() {
  const obj = { two: [...sel.two].map(splitK), one: [...sel.one].map(splitK) };
  try { localStorage.setItem(LS_KEY, JSON.stringify(obj)); } catch (e) {}
}

function updateCounter() {
  const n2 = sel.two.size, n1 = sel.one.size;
  document.getElementById("counter").innerHTML =
    `task two: <b>${n2}</b> &nbsp;·&nbsp; task one: <b>${n1}</b>`;
  const pt = document.querySelector('[data-tab="picked"]');
  if (pt) pt.textContent = `Picked (${n2} + ${n1})`;
}

function updateClassCounts() {
  const panel = document.getElementById("panel");
  for (const sec of panel.querySelectorAll(".cls")) {
    const task = sec.dataset.task;
    const strips = sec.querySelectorAll(".strip");
    let k = 0;
    strips.forEach(s => { if (sel[task].has(s.dataset.key)) k++; });
    sec.querySelector(".cls-count").innerHTML =
      `${strips.length} rows · <b>${k}</b> selected`;
  }
}

function setStripSel(strip, task) {
  const on = sel[task].has(strip.dataset.key);
  strip.classList.toggle("sel", on);
  strip.querySelector(".pick").checked = on;
}

// ---- media cells (lazy) -----------------------------------------------------
function mkCell(caption, dataSrc, opts) {
  opts = opts || {};
  const fig = document.createElement("figure");
  fig.className = "cell" + (opts.out ? " out" : "");
  const cap = document.createElement("figcaption");
  cap.textContent = caption; cap.title = caption;
  fig.appendChild(cap);
  if (opts.missing) {
    const box = document.createElement("div");
    box.className = "vid miss"; box.textContent = "missing";
    fig.appendChild(box);
  } else {
    const v = document.createElement("video");
    v.className = "vid";
    v.muted = true; v.loop = true; v.playsInline = true; v.preload = "none";
    v.dataset.src = dataSrc;   // src set only when the strip is near the viewport
    fig.appendChild(v);
  }
  return fig;
}

function buildStrip(row, task) {
  const key = kOf(row.endpoint, row.reference);
  const strip = document.createElement("div");
  strip.className = "strip"; strip.dataset.key = key; strip.dataset.task = task;

  const left = document.createElement("div");
  left.className = "left";
  const lab = document.createElement("label");
  lab.className = "pickrow";
  lab.innerHTML = `<input type="checkbox" class="pick"><span class="ep">${esc(row.endpoint)}</span>`;
  const refn = document.createElement("div");
  refn.className = "refname"; refn.textContent = "ref: " + row.reference;
  const badges = document.createElement("div");
  badges.className = "badges";
  badges.innerHTML =
    `<span class="badge">cell ${esc(row.cell)}</span>` +
    `<span class="badge ${row.ed ? "ed" : "hf"}">${row.ed ? "ED" : "HF"}</span>`;
  const sync = document.createElement("button");
  sync.className = "sync"; sync.textContent = "↺ sync";
  sync.title = "restart every video in this row together";
  left.appendChild(lab); left.appendChild(refn);
  left.appendChild(badges);
  if (row.note) {
    const note = document.createElement("div");
    note.className = "note"; note.textContent = "note: " + row.note;
    left.appendChild(note);
  }
  left.appendChild(sync);

  const cells = document.createElement("div");
  cells.className = "cells";
  cells.appendChild(mkCell("Given start", row.start));
  if (row.end) cells.appendChild(mkCell("Given end", row.end));
  cells.appendChild(mkCell("Reference", row.ref_url));
  if (SHOWGEN) {
    const div = document.createElement("div");
    div.className = "divider"; cells.appendChild(div);
    for (const label of DATA.output_order[task]) {
      const tmpl = row.outputs[label];
      if (tmpl === undefined) continue;
      const missing = (row.outputs_missing[label] || []).includes(SEED);
      cells.appendChild(mkCell(label, missing ? null : tmpl.replace("{seed}", SEED),
        { out: true, missing }));
    }
  }

  strip.appendChild(left); strip.appendChild(cells);
  setStripSel(strip, task);
  return strip;
}

const TASK_TITLE = { two: "Task two — both endpoints", one: "Task one — start only" };

function renderSections(panel, rows, task) {
  const byCls = {};
  for (const r of rows) (byCls[r.cls] = byCls[r.cls] || []).push(r);
  for (const c in byCls) byCls[c].sort((a, b) => a.endpoint < b.endpoint ? -1 : 1);
  for (const clsName of Object.keys(byCls).sort()) {
    const sec = document.createElement("section");
    sec.className = "cls"; sec.dataset.cls = clsName; sec.dataset.task = task;
    const head = document.createElement("div");
    head.className = "cls-head";
    head.innerHTML =
      `<span class="cls-name">${esc(clsName)}</span>` +
      `<span class="cls-count"></span><span class="spacer"></span>` +
      `<button class="btn pick-one">pick one here</button>` +
      `<button class="btn clear-cls">clear class</button>`;
    sec.appendChild(head);
    const wrap = document.createElement("div");
    wrap.className = "strips";
    for (const r of byCls[clsName]) {
      const strip = buildStrip(r, task);
      wrap.appendChild(strip);
      io.observe(strip);
    }
    sec.appendChild(wrap);
    panel.appendChild(sec);
  }
}

function renderCurrent() {
  const panel = document.getElementById("panel");
  if (io) io.disconnect();
  panel.innerHTML = "";
  if (cur === "picked") {
    let any = false;
    for (const task of ["two", "one"]) {
      const rows = DATA[task].filter(r => sel[task].has(kOf(r.endpoint, r.reference)));
      if (!rows.length) continue;
      any = true;
      const h = document.createElement("div");
      h.className = "task-head"; h.textContent = `${TASK_TITLE[task]} — ${rows.length} picked`;
      panel.appendChild(h);
      renderSections(panel, rows, task);
    }
    if (!any) panel.innerHTML = '<div class="empty">nothing picked yet — tick rows in the task tabs, or press “Load proposed”</div>';
  } else if (cur === "out") {
    const so = DATA.screened_out || {two: [], one: []};
    let any = false;
    for (const task of ["two", "one"]) {
      const keys = new Set((so[task] || []).map(([ep, ref]) => kOf(ep, ref)));
      const rows = DATA[task].filter(r => keys.has(kOf(r.endpoint, r.reference)));
      if (!rows.length) continue;
      any = true;
      const h = document.createElement("div");
      h.className = "task-head"; h.textContent = `${TASK_TITLE[task]} — ${rows.length} screened out by the owner (not in the study)`;
      panel.appendChild(h);
      renderSections(panel, rows, task);
    }
    if (!any) panel.innerHTML = '<div class="empty">no screened-out rows recorded</div>';
  } else {
    const rows = DATA[cur];
    if (!rows.length) { panel.innerHTML = '<div class="empty">no candidates</div>'; return; }
    renderSections(panel, rows, cur);
  }
  updateClassCounts();
}

// ---- lazy load / unload + global playback -----------------------------------
function loadStrip(strip) {
  strip.querySelectorAll("video").forEach(v => {
    if (!v.getAttribute("src") && v.dataset.src) {
      v.src = v.dataset.src; v.load();
      if (!PAUSED) v.play().catch(() => {});
    }
  });
}
function unloadStrip(strip) {
  strip.querySelectorAll("video").forEach(v => {
    if (v.getAttribute("src")) { v.pause(); v.removeAttribute("src"); v.load(); }
  });
}
function setPaused(p) {
  PAUSED = p;
  document.getElementById("pauseall").textContent = p ? "Play all" : "Pause all";
  document.getElementById("pauseall").classList.toggle("on", p);
  document.querySelectorAll("#panel video").forEach(v => {
    if (v.getAttribute("src")) { if (p) v.pause(); else v.play().catch(() => {}); }
  });
}
function syncStrip(strip) {
  strip.querySelectorAll("video").forEach(v => {
    if (v.getAttribute("src")) { v.currentTime = 0; v.play().catch(() => {}); }
  });
}

// ---- controls ---------------------------------------------------------------
function applySize() {
  document.documentElement.style.setProperty("--cell-h", H + "px");
  document.getElementById("size-val").textContent = H + "px";
}
function applySeedButtons() {
  for (const b of document.querySelectorAll("#seed-seg button"))
    b.classList.toggle("on", b.dataset.seed === SEED);
}
function measureTopbar() {
  const tb = document.querySelector(".topbar");
  document.documentElement.style.setProperty("--topbar-h", tb.offsetHeight + "px");
}

// ---- event wiring (delegation on the single panel) --------------------------
function wirePanel() {
  const panel = document.getElementById("panel");
  panel.addEventListener("change", e => {
    if (!e.target.classList.contains("pick")) return;
    const strip = e.target.closest(".strip");
    const key = strip.dataset.key, t = strip.dataset.task;
    if (e.target.checked) sel[t].add(key); else sel[t].delete(key);
    strip.classList.toggle("sel", e.target.checked);
    saveSel(); updateCounter(); updateClassCounts();
  });
  panel.addEventListener("click", e => {
    if (e.target.tagName === "VIDEO") {
      const v = e.target;
      if (v.getAttribute("src")) { v.currentTime = 0; v.play().catch(() => {}); }
      return;
    }
    const sync = e.target.closest(".sync");
    if (sync) { syncStrip(sync.closest(".strip")); return; }
    const btn = e.target.closest("button");
    if (!btn) return;
    const sec = btn.closest(".cls");
    if (!sec) return;
    const t = sec.dataset.task;
    if (btn.classList.contains("pick-one")) {
      for (const strip of sec.querySelectorAll(".strip")) {
        if (!sel[t].has(strip.dataset.key)) {
          sel[t].add(strip.dataset.key); setStripSel(strip, t); break;
        }
      }
      saveSel(); updateCounter(); updateClassCounts();
    } else if (btn.classList.contains("clear-cls")) {
      for (const strip of sec.querySelectorAll(".strip")) {
        sel[t].delete(strip.dataset.key); setStripSel(strip, t);
      }
      saveSel(); updateCounter(); updateClassCounts();
    }
  });
}

function switchTab(task) {
  if (task === cur) return;
  cur = task;
  document.querySelectorAll(".tab").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === task));
  renderCurrent();
}

function download(name, text) {
  const blob = new Blob([text], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function exportObj() {
  const obj = {
    two: [...sel.two].map(splitK), one: [...sel.one].map(splitK),
    outputs_revealed: REVEALED,
  };
  if (REVEALED && REVEALED_AT) obj.outputs_revealed_at = REVEALED_AT;
  return obj;
}

function applyImport(text) {
  const msg = document.getElementById("io-msg");
  let obj;
  try { obj = JSON.parse(text); } catch (e) { msg.textContent = "invalid JSON: " + e.message; return; }
  let unknown = 0, applied = 0;
  for (const t of ["two", "one"]) {
    sel[t] = new Set();
    for (const pair of (obj[t] || [])) {
      const k = kOf(pair[0], pair[1]);
      if (valid[t].has(k)) { sel[t].add(k); applied++; } else unknown++;
    }
  }
  saveSel(); renderCurrent(); updateCounter();
  msg.textContent = `applied ${applied} rows` + (unknown ? `, skipped ${unknown} not-a-candidate` : "");
}

function setShowgen(on) {
  SHOWGEN = on;
  document.getElementById("showgen").checked = on;
  try { localStorage.setItem(LS_SHOWGEN, on ? "1" : "0"); } catch (e) {}
  if (on && !REVEALED) { REVEALED = true; REVEALED_AT = new Date().toISOString(); saveReveal(); }
  renderCurrent();
}

function init() {
  for (const t of ["two", "one"]) {
    valid[t] = new Set(DATA[t].map(r => kOf(r.endpoint, r.reference)));
  }
  loadPrefs();
  // outputs visible at load => provenance is already revealed (keep them consistent)
  if (SHOWGEN && !REVEALED) { REVEALED = true; REVEALED_AT = REVEALED_AT || new Date().toISOString(); saveReveal(); }
  loadSel();

  // dynamic tab labels (candidate counts)
  document.querySelector('[data-tab="two"]').textContent =
    `Task two — both endpoints (${DATA.two.length})`;
  document.querySelector('[data-tab="one"]').textContent =
    `Task one — start only (${DATA.one.length})`;
  const nso = ((DATA.screened_out || {}).two || []).length + ((DATA.screened_out || {}).one || []).length;
  document.querySelector('[data-tab="out"]').textContent = `Screened out (${nso})`;

  // controls reflect prefs
  const size = document.getElementById("size");
  size.value = H; applySize();
  applySeedButtons();
  document.getElementById("showgen").checked = SHOWGEN;

  io = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (e.isIntersecting) loadStrip(e.target); else unloadStrip(e.target);
    }
  }, { root: null, rootMargin: "100% 0px 100% 0px", threshold: 0 });

  wirePanel();
  renderCurrent();
  updateCounter();

  // topbar height -> sticky offset for class headers
  measureTopbar();
  if (window.ResizeObserver) new ResizeObserver(measureTopbar).observe(document.querySelector(".topbar"));
  window.addEventListener("resize", measureTopbar);

  document.querySelectorAll(".tab").forEach(b =>
    b.addEventListener("click", () => switchTab(b.dataset.tab)));

  size.addEventListener("input", () => {
    H = parseInt(size.value, 10); applySize();
    try { localStorage.setItem(LS_SIZE, String(H)); } catch (e) {}
  });
  document.getElementById("seed-seg").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.seed === SEED) return;
    SEED = b.dataset.seed; applySeedButtons();
    try { localStorage.setItem(LS_SEED, SEED); } catch (e) {}
    renderCurrent();   // output URLs carry the seed
  });
  document.getElementById("showgen").addEventListener("change", e => setShowgen(e.target.checked));
  document.getElementById("pauseall").addEventListener("click", () => setPaused(!PAUSED));

  document.getElementById("btn-export").addEventListener("click", () => {
    const text = JSON.stringify(exportObj(), null, 1);
    document.getElementById("io").hidden = false;
    document.getElementById("io-text").value = text;
    document.getElementById("io-msg").textContent =
      "picks.json downloaded + shown for copy-paste" +
      (REVEALED ? " (outputs_revealed: yes)" : " (outputs_revealed: no)");
    download("picks.json", text);
  });
  document.getElementById("btn-import").addEventListener("click", () => {
    document.getElementById("io").hidden = false;
    document.getElementById("io-msg").textContent = "paste picks JSON, then Apply";
    document.getElementById("io-text").focus();
  });
  document.getElementById("btn-proposed").addEventListener("click", () => {
    const np = (DATA.proposed.two || []).length + (DATA.proposed.one || []).length;
    if (!np) { alert("no proposed selection in rows_all.json"); return; }
    if (!confirm(`Replace the current selection with the proposed ${DATA.proposed.two.length} + ${DATA.proposed.one.length} rows?`)) return;
    for (const t of ["two", "one"]) {
      sel[t] = new Set();
      for (const [ep, ref] of (DATA.proposed[t] || [])) {
        const k = kOf(ep, ref); if (valid[t].has(k)) sel[t].add(k);
      }
    }
    saveSel(); renderCurrent(); updateCounter();
  });
  document.getElementById("btn-reset").addEventListener("click", () => {
    if (!confirm("Discard edits and reload the current rows.json selection?")) return;
    for (const t of ["two", "one"]) {
      sel[t] = new Set();
      for (const [ep, ref] of (DATA.initial[t] || [])) {
        const k = kOf(ep, ref); if (valid[t].has(k)) sel[t].add(k);
      }
    }
    saveSel(); renderCurrent(); updateCounter();
  });
  document.getElementById("io-apply").addEventListener("click", () =>
    applyImport(document.getElementById("io-text").value));
  document.getElementById("io-copy").addEventListener("click", () => {
    const ta = document.getElementById("io-text");
    ta.value = JSON.stringify(exportObj(), null, 1);
    ta.select(); document.execCommand && document.execCommand("copy");
    document.getElementById("io-msg").textContent = "copied";
  });
  document.getElementById("io-close").addEventListener("click", () =>
    document.getElementById("io").hidden = true);
}

fetch("rows_all.json").then(r => r.json()).then(d => { DATA = d; init(); })
  .catch(e => {
    document.getElementById("counter").textContent = "failed to load rows_all.json";
    document.getElementById("panel").innerHTML =
      '<div class="empty">Could not load rows_all.json — serve this page over HTTP ' +
      '(the :8017 hub) from the repo root.</div>';
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
