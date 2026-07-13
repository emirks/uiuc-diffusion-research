"""Build the deep-dive results explorer for a finished certification run.

Consumes the certification artifacts plus the exam deep-dive produced by
scripts/exam_confusion_analysis.py and emits one self-contained HTML page
(no external requests): confusion matrices with class distances, per-clip
margins, tag-level error intensity, R1/R2 divergence, probe/archive/stability
charts, and the eight bar verdicts.

    python scripts/build_results_explorer.py \
        --cert-dir outputs/eval/certification/3.0.0-draft.8
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict

R4 = lambda x: None if x is None else round(float(x), 4)


def load_jsonl(p: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in p.open()] if p.exists() else []


def prep(cert: pathlib.Path) -> dict:
    ana = json.load(open(cert / "analysis" / "analysis.json"))
    exam = json.load(open(cert / "exam" / "exam.json"))
    trust = json.load(open(cert / "exam" / "trust_map.json"))
    record = json.load(open(cert / "record.json"))
    ci = json.load(open(cert / "content_invariance.json"))
    sib = load_jsonl(cert / "cert_siblings" / "items.jsonl")
    prb = load_jsonl(cert / "cert_probes" / "items.jsonl")
    blc = load_jsonl(cert / "cert_blockc" / "items.jsonl")

    clips = ana["clips"]
    # class -> stratum for row ordering: twosided | onesided | camera-tagged onesided
    cls_side, cls_cam = {}, defaultdict(bool)
    for c in clips:
        cls_side[c["class"]] = c["sidedness"]
        if "camera" in c["tags"]:
            cls_cam[c["class"]] = True
    classes = sorted(cls_side)
    n_by_class = Counter(c["class"] for c in clips)

    def stratum(c):
        if cls_side[c] == "twosided":
            return "two-sided"
        return "one-sided+camera" if cls_cam[c] else "one-sided"

    order = (sorted([c for c in classes if stratum(c) == "two-sided"]) +
             sorted([c for c in classes if stratum(c) == "one-sided"]) +
             sorted([c for c in classes if stratum(c) == "one-sided+camera"]))
    cidx = {c: i for i, c in enumerate(order)}

    metrics = {}
    for name, m in ana["metrics"].items():
        r = m["retrieval"]
        conf = [[cidx[a], cidx[b], n]
                for a, row in r["confusion"].items()
                for b, n in row.items() if n > 0]
        cd = m["class_dist"]
        dm = [[R4(cd["mean"][a][b]) for b in order] for a in order]
        dmin = [[R4(cd["min"][a][b]) for b in order] for a in order]
        rows = [[rw["key"], cidx[rw["label"]],
                 (cidx[rw["pred"]] if rw.get("pred") else None),
                 rw.get("nn_key"), R4(rw.get("nn_dist")),
                 R4(rw.get("d_within_min")), R4(rw.get("d_cross_min"))]
                for rw in m["rows"]]
        metrics[name] = {
            "acc": R4(r["accuracy_1nn"]), "wilson": [R4(v) for v in r["accuracy_wilson95"]],
            "d": R4(r["separation_cohens_d"]), "coverage": R4(r["coverage"]),
            "chance": R4(r["chance"]),
            "within_mean": R4(r["within_mean"]), "cross_mean": R4(r["cross_mean"]),
            "recall": {c: R4(v) for c, v in r["per_class_recall"].items()},
            "conf": conf, "dmean": dm, "dmin": dmin, "rows": rows,
        }

    # tag-group error intensity per metric (computed here, rendered as a table)
    clip_groups = {}
    for c in clips:
        pat = c["sidedness"] + ("_" + "_".join(c["tags"]) if c["tags"] else "")
        clip_groups[c["key"]] = {"coarse": [c["sidedness"], *c["tags"]], "pattern": pat}
    patterns = sorted({g["pattern"] for g in clip_groups.values()})
    coarse = ["twosided", "onesided", "object", "style", "camera"]
    tag_table = {"patterns": [], "coarse": []}
    for kind, names in (("patterns", patterns), ("coarse", coarse)):
        for gname in names:
            keys = [k for k, g in clip_groups.items()
                    if (gname == g["pattern"] if kind == "patterns"
                        else gname in g["coarse"])]
            row = {"group": gname, "n": len(keys)}
            for mname, m in metrics.items():
                by_key = {r[0]: r for r in m["rows"]}
                graded = [by_key[k] for k in keys if by_key[k][2] is not None]
                errs = [1 for r in graded if r[2] != r[1]]
                row[mname] = R4(len(errs) / len(graded)) if graded else None
            tag_table[kind].append(row)

    r2rows = [[rw["key"], rw["label"], R4(rw.get("margin")),
               rw.get("intruder"), rw.get("correct")] for rw in ana["r2"]["rows"]]

    g = record["grades"]
    data = {
        "version": record["version"],
        "overall": record["overall_pass"],
        "stamp": {"commit": record["stamp"]["git"]["commit_short"],
                  "corpus": (record["stamp"].get("corpus_sha256") or "")[:16],
                  "bars": record["bars_sha256"][:16]},
        "order": order,
        "strata": {c: stratum(c) for c in classes},
        "n_by_class": dict(n_by_class),
        "clip_tags": {c["key"]: c["tags"] for c in clips},
        "metrics": metrics,
        "tag_table": tag_table,
        "r2": {"accuracy": R4(ana["r2"]["accuracy"]),
               "winner": ana["r2"]["winner_mask"],
               "n_graded": ana["r2"]["n_graded"],
               "recall": {c: R4(v) for c, v in ana["r2"]["per_class_recall"].items()},
               "rows": r2rows},
        "trust": trust,
        "exam": {"bar1": exam["bar1"], "mask": exam["mask_adoption"],
                 "motion": exam["motion_adoption"], "o7": exam["o7_conditional"]},
        "ci": {"pooled": R4(ci["pooled_partial_corr"]), "n_pairs": ci["n_pairs"],
               "alarm": 0.4,
               "per_class": {c: {"n": v["n_pairs"], "corr": R4(v["corr"])}
                             for c, v in ci["per_class"].items()}},
        "siblings": [{k: (R4(r[k]) if isinstance(r[k], float) else r[k])
                      for k in ("item_id", "arm", "style", "sidedness", "app_ref",
                                "copy_max", "margin", "near_copy", "core_degenerate",
                                "obj_match", "max_seam_z", "intruder")}
                     for r in sib],
        "probes": [{k: (R4(r[k]) if isinstance(r[k], float) else r[k])
                    for k in ("item_id", "arm", "style", "app_ref", "copy_max",
                              "margin", "max_seam_z", "near_copy")}
                   for r in prb],
        "blockc": [{k: (R4(r[k]) if isinstance(r.get(k), float) else r.get(k))
                    for k in ("item_id", "arm", "style", "app_ref", "copy_max",
                              "margin", "obj_match", "max_seam_z", "near_copy",
                              "core_degenerate", "cross_high", "twin_of")}
                   for r in blc],
        "grades": {
            "controls": g["controls"], "siblings": g["siblings"],
            "splices": g["splices"], "reversal": {k: v for k, v in g["reversal"].items()},
            "m3": g["m3_panel"], "twins": g["copy_twins"], "bar8": g["bar8"],
        },
        "verdicts": record["verdicts"],
        "blockc_meta": {"n_scored": record["blockc"]["n_scored"],
                        "n_excluded": len(record["blockc"]["excluded"]),
                        "bridge": record["blockc"]["bridge"]},
        "calibration": record["calibration"]["tau_copy"],
    }
    return data


HTML = r"""<title>transition-eval __VER__ — results explorer</title>
<style>
:root{
  --bg:#F6F6F3; --surface:#FFFFFF; --ink:#21262B; --muted:#64707A;
  --line:#E1E3DD; --acc:#3D5A9E; --acc-soft:#E8ECF6;
  --ok:#2F7A3B; --bad:#B5472E; --bad-soft:#F6E9E5; --warn:#9A731B;
  --heat0:#F1F2EE; --heat1:#2C4770; --miss1:#B5472E;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#14171B; --surface:#1B2026; --ink:#E5E7E2; --muted:#97A1AA;
  --line:#2A313A; --acc:#86A0DC; --acc-soft:#232B3B;
  --ok:#6EBE7A; --bad:#E28063; --bad-soft:#3A2620; --warn:#D3AC55;
  --heat0:#20252C; --heat1:#8FB0E8; --miss1:#E28063;
}}
:root[data-theme="light"]{
  --bg:#F6F6F3; --surface:#FFFFFF; --ink:#21262B; --muted:#64707A;
  --line:#E1E3DD; --acc:#3D5A9E; --acc-soft:#E8ECF6;
  --ok:#2F7A3B; --bad:#B5472E; --bad-soft:#F6E9E5; --warn:#9A731B;
  --heat0:#F1F2EE; --heat1:#2C4770; --miss1:#B5472E;
}
:root[data-theme="dark"]{
  --bg:#14171B; --surface:#1B2026; --ink:#E5E7E2; --muted:#97A1AA;
  --line:#2A313A; --acc:#86A0DC; --acc-soft:#232B3B;
  --ok:#6EBE7A; --bad:#E28063; --bad-soft:#3A2620; --warn:#D3AC55;
  --heat0:#20252C; --heat1:#8FB0E8; --miss1:#E28063;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font:15px/1.55 system-ui,'Segoe UI',sans-serif}
.mono{font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
nav{position:sticky;top:0;z-index:40;background:var(--surface);
  border-bottom:1px solid var(--line);display:flex;gap:2px;align-items:center;
  padding:6px 18px;overflow-x:auto}
nav a{color:var(--muted);text-decoration:none;padding:5px 10px;border-radius:4px;
  font-family:ui-monospace,monospace;font-size:12px;letter-spacing:.06em;
  text-transform:uppercase;white-space:nowrap}
nav a:hover,nav a:focus-visible{color:var(--acc);background:var(--acc-soft);outline:none}
nav .chip{margin-left:auto}
main{max-width:1180px;margin:0 auto;padding:26px 22px 90px}
header.mast{padding:34px 0 8px}
h1{font-family:'Iowan Old Style',Georgia,'Times New Roman',serif;
  font-weight:500;font-size:34px;line-height:1.15;margin:0 0 6px;text-wrap:balance}
.sub{color:var(--muted);max-width:68ch}
h2{font-family:'Iowan Old Style',Georgia,serif;font-weight:500;font-size:24px;
  margin:0 0 4px;text-wrap:balance}
h3{font-size:15.5px;font-weight:600;margin:0 0 2px}
section{margin-top:54px}
.eyebrow{font-family:ui-monospace,monospace;font-size:11.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--acc);margin-bottom:6px}
.note{color:var(--muted);font-size:13.5px;max-width:80ch;margin:2px 0 14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;
  padding:16px 18px;margin-top:14px}
.grid{display:grid;gap:14px}
.grid.m6{grid-template-columns:repeat(auto-fit,minmax(168px,1fr))}
.grid.two{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.chip{display:inline-block;border-radius:999px;padding:2px 11px;font-size:12px;
  font-family:ui-monospace,monospace;letter-spacing:.05em}
.chip.pass{background:color-mix(in srgb,var(--ok) 14%,transparent);color:var(--ok)}
.chip.fail{background:color-mix(in srgb,var(--bad) 14%,transparent);color:var(--bad)}
.chip.info{background:var(--acc-soft);color:var(--acc)}
.big{font-family:ui-monospace,monospace;font-size:26px;font-variant-numeric:tabular-nums}
.kv{color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);text-align:left;font-weight:500;
  padding:6px 10px;border-bottom:1px solid var(--line)}
td{padding:5px 10px;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums}
td.num{font-family:ui-monospace,monospace;text-align:right}
tr:hover td{background:var(--acc-soft)}
.scroll{overflow-x:auto}
.tabs{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px}
.tabs button{border:1px solid var(--line);background:var(--surface);color:var(--muted);
  border-radius:4px;padding:5px 12px;font:12px ui-monospace,monospace;cursor:pointer;
  letter-spacing:.04em}
.tabs button[aria-pressed="true"]{background:var(--acc);border-color:var(--acc);
  color:#fff}
:root[data-theme="dark"] .tabs button[aria-pressed="true"]{color:#10141A}
.tabs button:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);
  border:1px solid var(--line);border-radius:5px;padding:7px 10px;font-size:12.5px;
  box-shadow:0 4px 18px rgba(0,0,0,.18);z-index:60;display:none;max-width:340px}
#tip .mono{font-size:12px}
svg text{font-family:ui-monospace,monospace;fill:var(--muted)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);
  margin-top:8px;align-items:center}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;
  vertical-align:-1px}
.hm-wrap{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
@media (max-width:900px){.hm-wrap{grid-template-columns:1fr}}
#hmDetail{font-size:13px}
#hmDetail .clipline{padding:3px 0;border-bottom:1px dashed var(--line)}
canvas{display:block;max-width:100%}
.bar-cards{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.bar-cards .card{margin-top:0}
footer{margin-top:70px;color:var(--muted);font-size:12.5px;
  border-top:1px solid var(--line);padding-top:14px}
@media (prefers-reduced-motion: no-preference){
  .card{transition:border-color .15s}
  .card:hover{border-color:color-mix(in srgb,var(--acc) 40%,var(--line))}
}
</style>

<nav aria-label="sections">
  <a href="#exam">A · Exam</a><a href="#confusion">Confusion</a>
  <a href="#tags">Tags</a><a href="#r2">R2</a><a href="#trust">Trust</a>
  <a href="#probes">B · Probes</a><a href="#archives">C · Archives</a>
  <a href="#stability">D · Stability</a><a href="#bars">Bars</a>
  <span class="chip fail" id="navchip"></span>
</nav>
<div id="tip" role="tooltip"></div>
<main>
<header class="mast">
  <div class="eyebrow">certification record · <span id="mastver" class="mono"></span></div>
  <h1>Six metrics sit the same exam.</h1>
  <p class="sub">223 corpus clips, 39 transition styles, leave-one-out retrieval.
  Every chart below is drawn from the frozen run's artifacts — distances recomputed
  with the deployed metric code, nothing re-judged. <span id="maststamp" class="mono kv"></span></p>
  <div class="card" style="margin-top:18px"><div id="heroAxis"></div></div>
</header>

<section id="exam">
  <div class="eyebrow">Block A · R1 clip-level retrieval</div>
  <h2>How each metric performs at the exam</h2>
  <p class="note">Accuracy is leave-one-out 1-nearest-neighbour over the full corpus
  (chance 0.067). d is the separation between within-style and cross-style distances.
  The winning mask (v3_sided) is the deployed one; the others sat the same exam.</p>
  <div class="grid m6" id="scorecards"></div>
</section>

<section id="confusion">
  <div class="eyebrow">Block A · confusion structure</div>
  <h2>Who gets mistaken for whom, and how far apart they are</h2>
  <p class="note">Rows are the true style, columns the retrieved one, grouped by
  stratum (two-sided · one-sided · one-sided camera). Diagonal in blue, misses in
  rust. Hover any cell for the count and the mean inter-style distance; click a row
  label to pin the style's clip-level detail on the right.</p>
  <div class="card">
    <div class="tabs" id="hmTabs" role="tablist"></div>
    <div class="hm-wrap">
      <div class="scroll"><canvas id="hm" aria-label="confusion matrix heatmap"></canvas></div>
      <div id="hmDetail"><span class="kv">Click a row label to inspect a style.</span></div>
    </div>
    <div class="legend" id="hmLegend"></div>
  </div>
  <div class="card">
    <h3>Most-confused pairs</h3>
    <p class="note">Off-diagonal cells ranked by count for the selected metric.
    D(mean) is the average distance between the two styles' clips under that metric;
    "vs within" divides it by the true style's own within-style mean — values near
    or below 1.0 mean the neighbouring style genuinely sits as close as the style
    itself.</p>
    <div class="scroll"><div id="pairsTable"></div></div>
  </div>
  <div class="card">
    <h3>Per-clip retrieval margin</h3>
    <p class="note">One dot per clip: nearest cross-style distance minus nearest
    same-style distance. Left of zero, the nearest neighbour is from another style —
    a retrieval miss waiting to happen. Rust dots are actual misses.</p>
    <div id="marginStrip"></div>
  </div>
</section>

<section id="tags">
  <div class="eyebrow">Block A · error intensity by transition tag</div>
  <h2>Where each metric loses its footing</h2>
  <p class="note">Error rate (1 − recall) over the clips carrying each tag, per
  metric. The upper rows are the coarse tags; below them, the exact source-folder
  patterns. Darker rust = more of that group misretrieved.</p>
  <div class="card scroll"><div id="tagTable"></div></div>
</section>

<section id="r2">
  <div class="eyebrow">Block A · R2 pool-level readout</div>
  <h2>Clip-vs-pool: the second opinion</h2>
  <p class="note">R2 asks a different question: does the clip sit inside its own
  style's pooled features with positive margin? Left: per-style R1 recall against
  R2 recall — off-diagonal styles are where the two readouts disagree. Right: R2
  margins per clip; the intruder is the style whose pool comes closest.</p>
  <div class="grid two">
    <div class="card"><h3>R1 vs R2 recall per style</h3><div id="r1r2"></div></div>
    <div class="card"><h3>R2 margins &amp; intruders</h3><div id="r2strip"></div>
      <div class="scroll" style="margin-top:10px"><div id="intruderTable"></div></div></div>
  </div>
</section>

<section id="trust">
  <div class="eyebrow">Block A · trust map</div>
  <h2>What the exam certifies, per style</h2>
  <p class="note">A metric is trusted for a style when recall ≥ 0.5 with n ≥ 4
  clips. Greyed styles are auto-untrusted (too few clips). This map is the
  instrument's own statement of scope.</p>
  <div class="card scroll"><div id="trustGrid"></div></div>
</section>

<section id="probes">
  <div class="eyebrow">Block B · constructed probes</div>
  <h2>Known-truth probes: siblings, splices, reversal, seams</h2>
  <div class="grid two">
    <div class="card"><h3>Bar 2 &amp; 3 · sibling vs control floor</h3>
      <p class="note">Per style: app_ref of the degenerate control (open dot)
      against the true sibling (filled dot). The bar passes per style when the
      control scores lower. One inversion: nature_bloom.</p>
      <div id="dumbbell"></div></div>
    <div class="card"><h3>Bar 4 · splice detection</h3>
      <p class="note">copy_max of honest siblings vs 74 constructed splices.
      Dashed lines: initial τ 0.88 and recalibrated τ 0.858 (midpoint rule).</p>
      <div id="spliceStrip"></div></div>
    <div class="card"><h3>Bar 5 · reversal sensitivity</h3>
      <p class="note">Correlation drop when the reference plays backwards,
      for the 15 analytically reversal-sensitive pairs. Sign test p = 0.0176.</p>
      <div id="revBars"></div></div>
    <div class="card"><h3>Bar 6 · endpoint swap &amp; hard cut</h3>
      <p class="note">Left per style: transition-shape similarity to the true
      endpoints (blue) vs swapped endpoints (rust). Right: hard-cut seam z on a
      log scale against the z &gt; 3 detection line.</p>
      <div id="swapPlot"></div><div id="hcStrip" style="margin-top:10px"></div></div>
    <div class="card"><h3>Bar 7 · copy twins</h3>
      <p class="note">The 11 archived generations known to copy their reference.
      All flagged near-copy; dots are copy_max against both τ lines.</p>
      <div id="twinStrip"></div></div>
    <div class="card"><h3>Content-invariance audit (non-gating)</h3>
      <p class="note">Within-style partial correlation between appearance score
      and endpoint-content similarity, per style; pooled 0.818 over 751 pairs.
      Recorded against the 0.4 alarm level; both sides are DINO cosines, so a
      shared-backbone correlation is expected — the per-style spread is the
      information.</p>
      <div id="ciStrip"></div></div>
  </div>
</section>

<section id="archives">
  <div class="eyebrow">Block C · real generations</div>
  <h2>The instrument on 97 archived items</h2>
  <p class="note">Every convertible archived generation from exp_056–058, scored
  by the frozen instrument. Arms follow the original experiment structure
  (base vs IC-LoRA × in-class / cross / unseen / one-sided). 53 exp_058 items
  were excluded loudly (reference not recoverable).</p>
  <div class="card"><div class="tabs" id="bcTabs"></div><div id="bcStrip"></div></div>
  <div class="card"><h3>v2 ↔ v3 bridge (Spearman, shared items)</h3>
    <div class="scroll"><div id="bridgeTable"></div></div></div>
</section>

<section id="stability">
  <div class="eyebrow">Block D · stability</div>
  <h2>Same inputs, same numbers</h2>
  <div class="grid two">
    <div class="card"><h3>Warm rerun · 74 shared rows</h3>
      <p class="note">Full sibling re-scoring on the same node: every metric
      reproduced bitwise.</p><div class="big">max |Δ| = 0.0</div></div>
    <div class="card"><h3>Cold anchors · first execution</h3>
      <p class="note">Six anchors re-scored from scratch in a fresh cache
      (features recomputed). Worst per-metric deltas against the 0.04 tolerance,
      log scale.</p><div id="coldBars"></div></div>
  </div>
</section>

<section id="bars">
  <div class="eyebrow">The eight bars</div>
  <h2>Verdicts, with their arithmetic</h2>
  <div class="bar-cards" id="barCards"></div>
</section>

<footer id="foot"></footer>
</main>

<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (x, d=3) => x==null ? '·' : (+x).toFixed(d);
const METRICS = Object.keys(DATA.metrics);
const MLABEL = {m1a__v3_sided:'M1a · v3_sided ★', m1a__v2_envelope:'M1a · v2_envelope',
  m1a__all_frames:'M1a · all_frames', m1b_camera:'M1b · camera',
  m1c_object:'M1c · object', m_incumbent:'M · incumbent (v2 MFS)'};
const ORDER = DATA.order, N = ORDER.length;
const STR_BREAKS = [];
{let prev=null; ORDER.forEach((c,i)=>{const s=DATA.strata[c];
  if(s!==prev){STR_BREAKS.push([i,s]); prev=s;}});}

/* ---------- tooltip ---------- */
const tip = $('#tip');
function showTip(ev, html){tip.innerHTML=html; tip.style.display='block';
  const w=tip.offsetWidth,h=tip.offsetHeight;
  tip.style.left=Math.min(ev.clientX+14, innerWidth-w-8)+'px';
  tip.style.top=Math.min(ev.clientY+14, innerHeight-h-8)+'px';}
function hideTip(){tip.style.display='none';}

/* ---------- tiny SVG helpers ---------- */
function svgEl(w,h){return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" style="max-width:${w}px">`}
function mix(t, c0, c1){ // hex mix
  const p=x=>[1,3,5].map(i=>parseInt(x.slice(i,i+2),16));
  const a=p(c0),b=p(c1);
  return '#'+a.map((v,i)=>Math.round(v+(b[i]-v)*t).toString(16).padStart(2,'0')).join('');}

/* ---------- hero axis ---------- */
function hero(){
  const w=1080,h=110,x0=70,x1=w-30;
  const X=v=>x0+(x1-x0)*v;
  let s=svgEl(w,h);
  s+=`<line x1="${x0}" y1="70" x2="${x1}" y2="70" stroke="${css('--line')}"/>`;
  [0,.2,.4,.6,.8,1].forEach(v=>{s+=`<line x1="${X(v)}" y1="66" x2="${X(v)}" y2="74" stroke="${css('--line')}"/>
    <text x="${X(v)}" y="90" font-size="11" text-anchor="middle">${v}</text>`});
  const ch=DATA.metrics[METRICS[0]].chance;
  s+=`<line x1="${X(ch)}" y1="20" x2="${X(ch)}" y2="70" stroke="${css('--muted')}" stroke-dasharray="3 3"/>
      <text x="${X(ch)}" y="14" font-size="11" text-anchor="middle">chance ${fmt(ch,3)}</text>`;
  const floor=DATA.exam.bar1.acc_min;
  s+=`<line x1="${X(floor)}" y1="20" x2="${X(floor)}" y2="70" stroke="${css('--bad')}" stroke-dasharray="4 3"/>
      <text x="${X(floor)}" y="14" font-size="11" text-anchor="middle" fill="${css('--bad')}">bar-1 floor ${floor}</text>`;
  const lanes={}; // avoid label collisions: alternate above axis
  METRICS.forEach((m,i)=>{
    const a=DATA.metrics[m].acc, star=m==='m1a__v3_sided';
    const y=70, ly=[36,52][i%2];
    s+=`<line x1="${X(a)}" y1="${ly+4}" x2="${X(a)}" y2="${y-5}" stroke="${css('--line')}"/>
      <circle cx="${X(a)}" cy="${y}" r="${star?6:4.5}" fill="${star?css('--acc'):css('--surface')}"
        stroke="${css('--acc')}" stroke-width="1.6"/>
      <text x="${X(a)}" y="${ly}" font-size="11" text-anchor="middle"
        ${star?`fill="${css('--acc')}" font-weight="600"`:''}>${esc(MLABEL[m].split(' · ')[1])} ${fmt(a,3)}</text>`;
  });
  $('#heroAxis').innerHTML=s+'</svg>';
}

/* ---------- scorecards ---------- */
function scorecards(){
  $('#scorecards').innerHTML = METRICS.map(m=>{
    const d=DATA.metrics[m], star=m==='m1a__v3_sided';
    const rec=ORDER.map(c=>d.recall[c]);
    const bars=rec.map((r,i)=>{
      const col = r==null?css('--line'): r>=0.5?css('--acc'):css('--bad');
      return `<rect x="${i*5}" y="${34-30*(r||0)}" width="4" height="${30*(r||0)+1}" fill="${col}"><title>${esc(ORDER[i])} ${fmt(r,2)}</title></rect>`;}).join('');
    return `<div class="card" style="${star?`border-color:${css('--acc')}`:''}">
      <div class="kv mono">${esc(MLABEL[m])}</div>
      <div class="big">${fmt(d.acc,3)}</div>
      <div class="kv mono">wilson ${fmt(d.wilson[0],2)}–${fmt(d.wilson[1],2)} · d ${fmt(d.d,2)} · cov ${fmt(d.coverage,2)}</div>
      <svg viewBox="0 0 ${N*5} 36" width="100%" style="margin-top:8px" role="img"
        aria-label="per-style recall">${bars}</svg>
      <div class="kv">per-style recall, exam order · blue ≥ 0.5</div></div>`;
  }).join('');
}

/* ---------- confusion heatmap ---------- */
let CUR = 'm1a__v3_sided', PIN = null;
const CELL=17, PAD_L=150, PAD_T=118;
function drawHM(){
  const cv=$('#hm'), d=DATA.metrics[CUR];
  const W=PAD_L+N*CELL+8, H=PAD_T+N*CELL+8, dpr=devicePixelRatio||1;
  cv.width=W*dpr; cv.height=H*dpr; cv.style.width=W+'px';
  const g=cv.getContext('2d'); g.scale(dpr,dpr);
  g.clearRect(0,0,W,H);
  const conf={}; let maxOff=1;
  d.conf.forEach(([i,j,n])=>{conf[i+','+j]=n; if(i!==j&&n>maxOff)maxOff=n;});
  const heat0=css('--heat0'),heat1=css('--heat1'),miss=css('--miss1'),bg=css('--bg');
  for(let i=0;i<N;i++)for(let j=0;j<N;j++){
    const n=conf[i+','+j]||0, x=PAD_L+j*CELL, y=PAD_T+i*CELL;
    let col=heat0;
    if(n>0) col = i===j ? mix(Math.min(1,n/DATA.n_by_class[ORDER[i]]),heat0,heat1)
                        : mix(Math.min(1,.25+.75*n/maxOff),heat0,miss);
    g.fillStyle=col; g.fillRect(x,y,CELL-1,CELL-1);
  }
  // stratum separators
  g.strokeStyle=css('--muted'); g.lineWidth=1;
  STR_BREAKS.slice(1).forEach(([i])=>{
    g.beginPath();g.moveTo(PAD_L-4,PAD_T+i*CELL-0.5);g.lineTo(PAD_L+N*CELL,PAD_T+i*CELL-0.5);g.stroke();
    g.beginPath();g.moveTo(PAD_L+i*CELL-0.5,PAD_T-4);g.lineTo(PAD_L+i*CELL-0.5,PAD_T+N*CELL);g.stroke();});
  // labels
  g.font='10.5px ui-monospace,monospace';
  ORDER.forEach((c,i)=>{
    g.fillStyle = PIN===i?css('--acc'):css('--muted');
    g.textAlign='right'; g.fillText(c.slice(0,18),PAD_L-7,PAD_T+i*CELL+12);
    g.save(); g.translate(PAD_L+i*CELL+12,PAD_T-7); g.rotate(-Math.PI/3);
    g.textAlign='left'; g.fillText(c.slice(0,14),0,0); g.restore();});
  STR_BREAKS.forEach(([i,s],k)=>{
    const end = k+1<STR_BREAKS.length?STR_BREAKS[k+1][0]:N;
    g.fillStyle=css('--acc'); g.textAlign='left';
    g.save(); g.translate(14, PAD_T+(i+end)/2*CELL); g.rotate(-Math.PI/2);
    g.textAlign='center'; g.fillText(s,0,0); g.restore();});
}
function hmTabs(){
  $('#hmTabs').innerHTML=METRICS.map(m=>
    `<button role="tab" aria-pressed="${m===CUR}" data-m="${m}">${esc(MLABEL[m])}</button>`).join('');
  $('#hmTabs').querySelectorAll('button').forEach(b=>b.onclick=()=>{
    CUR=b.dataset.m; hmTabs(); drawHM(); pairsTable(); marginStrip(); detail();});
}
function hmHover(ev){
  const r=$('#hm').getBoundingClientRect();
  const x=ev.clientX-r.left, y=ev.clientY-r.top;
  const j=Math.floor((x-PAD_L)/CELL), i=Math.floor((y-PAD_T)/CELL);
  if(i<0||j<0||i>=N||j>=N){hideTip();return;}
  const d=DATA.metrics[CUR];
  const n=(d.conf.find(([a,b])=>a===i&&b===j)||[0,0,0])[2];
  const dm=d.dmean[i][j], dmin=d.dmin[i][j], wi=d.dmean[i][i];
  showTip(ev,`<b>${esc(ORDER[i])}</b> → <b>${esc(ORDER[j])}</b><br>
    <span class="mono">${n} of ${DATA.n_by_class[ORDER[i]]} clips</span><br>
    <span class="mono">D mean ${fmt(dm)} · min ${fmt(dmin)}</span><br>
    <span class="mono kv">within-${esc(ORDER[i])} mean ${fmt(wi)}</span>`);
}
function hmClick(ev){
  const r=$('#hm').getBoundingClientRect();
  const y=ev.clientY-r.top, x=ev.clientX-r.left;
  const i=Math.floor((y-PAD_T)/CELL);
  if(x<PAD_L&&i>=0&&i<N){PIN=PIN===i?null:i; drawHM(); detail();}
}
function detail(){
  const el=$('#hmDetail');
  if(PIN==null){el.innerHTML='<span class="kv">Click a row label to inspect a style.</span>';return;}
  const c=ORDER[PIN], d=DATA.metrics[CUR];
  const rows=d.rows.filter(r=>r[1]===PIN);
  const misses=rows.filter(r=>r[2]!==null&&r[2]!==PIN);
  el.innerHTML=`<h3 class="mono">${esc(c)}</h3>
    <div class="kv">${DATA.n_by_class[c]} clips · ${esc(DATA.strata[c])} ·
      recall ${fmt(d.recall[c],2)} under ${esc(MLABEL[CUR])}</div>
    <div style="margin-top:8px">${rows.map(r=>{
      const ok=r[2]===PIN;
      return `<div class="clipline mono" style="font-size:11.5px">
        <span class="sw" style="background:${ok?css('--acc'):css('--bad')}"></span>
        ${esc(r[0].split('/').pop())} → ${r[2]==null?'—':esc(ORDER[r[2]])}
        <span class="kv">d ${fmt(r[4])}${r[5]!=null?` · own ${fmt(r[5])}`:''}</span></div>`;
    }).join('')}</div>
    ${misses.length?`<div class="kv" style="margin-top:6px">${misses.length} miss(es);
      nearest same-style clip sat ${misses.every(m=>m[5]!=null&&m[4]<m[5])?'farther than the confuser in every case':'closer in some cases'}.</div>`:''}`;
}
function pairsTable(){
  const d=DATA.metrics[CUR];
  const off=d.conf.filter(([i,j])=>i!==j).sort((a,b)=>b[2]-a[2]).slice(0,15);
  $('#pairsTable').innerHTML=`<table><thead><tr><th>true</th><th>retrieved</th>
    <th>clips</th><th>share of style</th><th>D(mean)</th><th>D(min)</th><th>vs within</th></tr></thead><tbody>`+
    off.map(([i,j,n])=>{
      const dm=d.dmean[i][j], wi=d.dmean[i][i];
      const ratio=wi?dm/wi:null;
      return `<tr><td class="mono">${esc(ORDER[i])}</td><td class="mono">${esc(ORDER[j])}</td>
        <td class="num">${n}</td><td class="num">${fmt(n/DATA.n_by_class[ORDER[i]],2)}</td>
        <td class="num">${fmt(dm)}</td><td class="num">${fmt(d.dmin[i][j])}</td>
        <td class="num" style="${ratio!=null&&ratio<1.05?`color:${css('--bad')}`:''}">${fmt(ratio,2)}</td></tr>`;
    }).join('')+'</tbody></table>';
}
function marginStrip(){
  const d=DATA.metrics[CUR];
  const vals=d.rows.filter(r=>r[2]!==null&&r[5]!=null&&r[6]!=null);
  const w=1080,h=120,x0=60,x1=w-20;
  const lo=Math.min(...vals.map(r=>r[6]-r[5])), hi=Math.max(...vals.map(r=>r[6]-r[5]));
  const X=v=>x0+(x1-x0)*(v-lo)/(hi-lo);
  let s=svgEl(w,h);
  s+=`<line x1="${X(0)}" y1="12" x2="${X(0)}" y2="${h-26}" stroke="${css('--muted')}" stroke-dasharray="3 3"/>
    <text x="${X(0)}" y="${h-10}" font-size="11" text-anchor="middle">margin 0</text>
    <text x="${x0}" y="${h-10}" font-size="11">${fmt(lo,2)}</text>
    <text x="${x1}" y="${h-10}" font-size="11" text-anchor="end">+${fmt(hi,2)}</text>`;
  // jitter lanes deterministic by index
  vals.forEach((r,k)=>{
    const m=r[6]-r[5], miss=r[2]!==r[1];
    const y=24+((k*37)%17)*4.6;
    s+=`<circle cx="${X(m)}" cy="${y}" r="3.4" fill="${miss?css('--bad'):css('--acc')}"
      fill-opacity="${miss?0.9:0.45}" data-k="${k}"><title>${esc(r[0])} · margin ${fmt(m)} → ${esc(ORDER[r[2]])}</title></circle>`;});
  s+='</svg>';
  const nm=vals.filter(r=>r[2]!==r[1]).length;
  $('#marginStrip').innerHTML=s+`<div class="legend">
    <span><span class="sw" style="background:${css('--acc')}"></span>retrieved own style</span>
    <span><span class="sw" style="background:${css('--bad')}"></span>miss (${nm} of ${vals.length})</span></div>`;
}

/* ---------- tag table ---------- */
function tagTable(){
  const cols=METRICS;
  const row=r=>`<tr><td class="mono">${esc(r.group)}</td><td class="num">${r.n}</td>`+
    cols.map(m=>{
      const v=r[m];
      const col=v==null?'transparent':mix(Math.min(1,v*1.6),css('--heat0'),css('--miss1'));
      const tc=v!=null&&v>0.45?'#fff':'inherit';
      return `<td class="num" style="background:${col};color:${tc}">${v==null?'·':fmt(v,2)}</td>`;}).join('')+'</tr>';
  $('#tagTable').innerHTML=`<table><thead><tr><th>group</th><th>n</th>`+
    cols.map(m=>`<th>${esc(MLABEL[m])}</th>`).join('')+`</tr></thead><tbody>`+
    DATA.tag_table.coarse.map(row).join('')+
    `<tr><td colspan="${cols.length+2}" style="border-bottom:none"><span class="kv">exact source patterns</span></td></tr>`+
    DATA.tag_table.patterns.map(row).join('')+'</tbody></table>';
}

/* ---------- R2 ---------- */
function r1r2(){
  const w=520,h=440,p=52;
  const X=v=>p+(w-p-16)*v, Y=v=>h-p+ (p+16-h)*v;
  let s=svgEl(w,h);
  [0,.25,.5,.75,1].forEach(v=>{
    s+=`<line x1="${X(v)}" y1="${Y(0)}" x2="${X(v)}" y2="${Y(1)}" stroke="${css('--line')}"/>
      <line x1="${X(0)}" y1="${Y(v)}" x2="${X(1)}" y2="${Y(v)}" stroke="${css('--line')}"/>
      <text x="${X(v)}" y="${h-30}" font-size="11" text-anchor="middle">${v}</text>
      <text x="${p-8}" y="${Y(v)+4}" font-size="11" text-anchor="end">${v}</text>`;});
  s+=`<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="${css('--muted')}" stroke-dasharray="4 4"/>
    <text x="${w/2}" y="${h-8}" font-size="12" text-anchor="middle">R1 recall (m1a, v3_sided)</text>
    <text x="14" y="${h/2}" font-size="12" text-anchor="middle" transform="rotate(-90 14 ${h/2})">R2 recall (pool margin)</text>`;
  const win=DATA.metrics['m1a__v3_sided'].recall;
  ORDER.forEach(c=>{
    const a=win[c], b=DATA.r2.recall[c];
    if(a==null||b==null)return;
    const dev=Math.abs(a-b)>=0.5;
    s+=`<circle cx="${X(a)}" cy="${Y(b)}" r="${4+DATA.n_by_class[c]/4}"
      fill="${dev?css('--bad'):css('--acc')}" fill-opacity="0.55" stroke="${dev?css('--bad'):css('--acc')}">
      <title>${esc(c)} · R1 ${fmt(a,2)} / R2 ${fmt(b,2)} · n=${DATA.n_by_class[c]}</title></circle>`;
    if(dev) s+=`<text x="${X(a)+10}" y="${Y(b)+4}" font-size="11" fill="${css('--bad')}">${esc(c)}</text>`;});
  $('#r1r2').innerHTML=s+'</svg>';
}
function r2strip(){
  const rows=DATA.r2.rows.filter(r=>r[2]!=null);
  const lo=Math.min(...rows.map(r=>r[2])), hi=Math.max(...rows.map(r=>r[2]));
  const w=520,h=110,x0=20,x1=w-20;
  const X=v=>x0+(x1-x0)*(v-lo)/(hi-lo);
  let s=svgEl(w,h);
  s+=`<line x1="${X(0)}" y1="10" x2="${X(0)}" y2="${h-24}" stroke="${css('--muted')}" stroke-dasharray="3 3"/>
    <text x="${X(0)}" y="${h-8}" font-size="11" text-anchor="middle">margin 0</text>`;
  rows.forEach((r,k)=>{
    const y=18+((k*29)%15)*4.4, ok=r[4];
    s+=`<circle cx="${X(r[2])}" cy="${y}" r="3.2" fill="${ok?css('--acc'):css('--bad')}"
      fill-opacity="${ok?0.4:0.9}"><title>${esc(r[0])} · ${fmt(r[2])} · intruder ${esc(r[3]||'—')}</title></circle>`;});
  const acc=DATA.r2.accuracy;
  $('#r2strip').innerHTML=s+'</svg>'+
    `<div class="kv mono">pool accuracy ${fmt(acc,3)} over ${DATA.r2.n_graded} graded clips · winner mask ${esc(DATA.r2.winner)}</div>`;
}
function intruderTable(){
  const cnt={};
  DATA.r2.rows.forEach(r=>{if(r[4]===false&&r[3]) {
    const k=r[1]+'→'+r[3]; cnt[k]=(cnt[k]||0)+1;}});
  const top=Object.entries(cnt).sort((a,b)=>b[1]-a[1]).slice(0,12);
  $('#intruderTable').innerHTML=`<table><thead><tr><th>style</th><th>losing to pool of</th>
    <th>clips</th></tr></thead><tbody>`+top.map(([k,n])=>{
      const [a,b]=k.split('→');
      return `<tr><td class="mono">${esc(a)}</td><td class="mono">${esc(b)}</td><td class="num">${n}</td></tr>`;
    }).join('')+'</tbody></table>';
}

/* ---------- trust ---------- */
function trustGrid(){
  const ms=['m1a','m1b','m1c','m2b'];
  $('#trustGrid').innerHTML=`<table><thead><tr><th>style</th><th>n</th><th>stratum</th>`+
    ms.map(m=>`<th>${m}</th>`).join('')+`<th>R2 recall</th></tr></thead><tbody>`+
    ORDER.map(c=>{
      const t=DATA.trust[c]||{}, el=t.eligible;
      return `<tr style="${el?'':'opacity:.45'}"><td class="mono">${esc(c)}</td>
        <td class="num">${t.n_clips??DATA.n_by_class[c]}</td><td class="kv">${esc(DATA.strata[c])}</td>`+
        ms.map(m=>{
          const v=t[m];
          return `<td>${!el?'<span class="kv">n&lt;4</span>':
            v?`<span class="chip pass">trusted</span>`:`<span class="chip fail">—</span>`}</td>`;
        }).join('')+
        `<td class="num">${fmt(DATA.r2.recall[c],2)}</td></tr>`;
    }).join('')+'</tbody></table>';
}

/* ---------- Block B ---------- */
function dumbbell(){
  const g=DATA.grades.controls.per_class;
  const rows=Object.entries(g).map(([c,v])=>({c,ctrl:v.control_m1a,sib:v.sibling_m1a,pass:v.pass}))
    .sort((a,b)=>(b.sib-b.ctrl)-(a.sib-a.ctrl));
  const w=520,rh=15,h=rows.length*rh+40,x0=150,x1=w-14;
  const hi=Math.max(...rows.map(r=>Math.max(r.ctrl,r.sib)));
  const X=v=>x0+(x1-x0)*v/hi;
  let s=svgEl(w,h);
  rows.forEach((r,i)=>{
    const y=18+i*rh, bad=!r.pass;
    s+=`<text x="${x0-6}" y="${y+4}" font-size="10.5" text-anchor="end"
        ${bad?`fill="${css('--bad')}" font-weight="600"`:''}>${esc(r.c)}</text>
      <line x1="${X(r.ctrl)}" y1="${y}" x2="${X(r.sib)}" y2="${y}"
        stroke="${bad?css('--bad'):css('--line')}" stroke-width="2"/>
      <circle cx="${X(r.ctrl)}" cy="${y}" r="3.6" fill="${css('--surface')}"
        stroke="${bad?css('--bad'):css('--muted')}" stroke-width="1.6">
        <title>${esc(r.c)} control ${fmt(r.ctrl)}</title></circle>
      <circle cx="${X(r.sib)}" cy="${y}" r="3.6" fill="${bad?css('--bad'):css('--acc')}">
        <title>${esc(r.c)} sibling ${fmt(r.sib)}</title></circle>`;});
  $('#dumbbell').innerHTML=s+'</svg>'+`<div class="legend">
    <span><span class="sw" style="background:${css('--surface')};border:1.5px solid ${css('--muted')}"></span>control (hold/lerp)</span>
    <span><span class="sw" style="background:${css('--acc')}"></span>true sibling</span>
    <span><span class="sw" style="background:${css('--bad')}"></span>inversion (fails bar 3)</span></div>`;
}
function spliceStrip(){
  const honest=DATA.siblings.filter(r=>r.arm==='probe_sibling').map(r=>({v:r.copy_max,l:r.item_id,g:'honest'}));
  const spl=DATA.probes.filter(r=>r.arm.startsWith('probe_splice')).map(r=>({v:r.copy_max,l:r.item_id,g:r.arm.includes('verbatim')?'verbatim':'perturbed'}));
  const all=[...honest,...spl];
  const w=520,h=170,x0=90,x1=w-16;
  const lo=Math.min(...all.map(a=>a.v)),hi=1.0;
  const X=v=>x0+(x1-x0)*(v-lo)/(hi-lo);
  const lanes={honest:34,verbatim:84,perturbed:124};
  let s=svgEl(w,h);
  const cal=DATA.calibration;
  [[cal.initial,'τ 0.88'],[cal.recalibrated,'τ→ '+fmt(cal.recalibrated,3)]].forEach(([t,lab],i)=>{
    s+=`<line x1="${X(t)}" y1="14" x2="${X(t)}" y2="${h-24}" stroke="${css('--warn')}" stroke-dasharray="${i?'5 3':'2 3'}"/>
      <text x="${X(t)}" y="${10}" font-size="10.5" text-anchor="middle" fill="${css('--warn')}">${lab}</text>`;});
  Object.entries(lanes).forEach(([gname,y])=>{
    s+=`<text x="${x0-8}" y="${y+4}" font-size="11" text-anchor="end">${gname}</text>`;
    all.filter(a=>a.g===gname).forEach((a,k)=>{
      const col=gname==='honest'?css('--acc'):css('--bad');
      s+=`<circle cx="${X(a.v)}" cy="${y+((k*13)%3-1)*8}" r="3.4" fill="${col}" fill-opacity="0.6">
        <title>${esc(a.l)} · ${fmt(a.v)}</title></circle>`;});});
  s+=`<text x="${x0}" y="${h-8}" font-size="11">${fmt(lo,2)}</text>
      <text x="${x1}" y="${h-8}" font-size="11" text-anchor="end">1.0</text>`;
  const gr=DATA.grades.splices;
  $('#spliceStrip').innerHTML=s+'</svg>'+
    `<div class="kv mono">honest max ${fmt(gr.honest_max)} · splice min ${fmt(gr.splice_min)} · gap ${fmt(gr.gap)} (≥ 0.05)</div>`;
}
function revBars(){
  const rows=[...DATA.grades.reversal.rows].sort((a,b)=>b.drop-a.drop);
  const w=520,rh=17,h=rows.length*rh+34,x0=140;
  const lo=Math.min(0,...rows.map(r=>r.drop)),hi=Math.max(...rows.map(r=>r.drop));
  const X=v=>x0+(w-x0-14)*(v-lo)/(hi-lo);
  let s=svgEl(w,h);
  s+=`<line x1="${X(0)}" y1="8" x2="${X(0)}" y2="${h-22}" stroke="${css('--muted')}" stroke-dasharray="3 3"/>`;
  rows.forEach((r,i)=>{
    const y=14+i*rh, pos=r.drop>0;
    s+=`<text x="${x0-6}" y="${y+4}" font-size="10.5" text-anchor="end">${esc(r.class)}</text>
      <rect x="${Math.min(X(0),X(r.drop))}" y="${y-5}" width="${Math.abs(X(r.drop)-X(0))}" height="10"
        fill="${pos?css('--acc'):css('--bad')}" fill-opacity="0.8">
        <title>${esc(r.gen)} vs ${esc(r.ref)} · drop ${fmt(r.drop)}</title></rect>`;});
  const g=DATA.grades.reversal;
  $('#revBars').innerHTML=s+'</svg>'+
    `<div class="kv mono">${g.wins}W / ${g.losses}L · ${esc(g.rule)} · median drop ${fmt(g.median_drop)}</div>`;
}
function swapPlot(){
  const sw=DATA.grades.m3.swap.per_class;
  const rows=Object.entries(sw).map(([c,v])=>({c,t:v['true'],s:v.swapped})).sort((a,b)=>a.s-b.s);
  const w=520,h=160,x0=20,x1=w-16;
  const X=(i)=>x0+(x1-x0)*i/(rows.length-1);
  let s=svgEl(w,h);
  s+=`<text x="${x0}" y="14" font-size="11">similarity to true endpoints (blue, all ≈ 1.0) vs swapped (rust)</text>`;
  rows.forEach((r,i)=>{
    const Y=v=>h-26-(h-52)*v;
    s+=`<line x1="${X(i)}" y1="${Y(r.t)}" x2="${X(i)}" y2="${Y(r.s)}" stroke="${css('--line')}"/>
      <circle cx="${X(i)}" cy="${Y(r.t)}" r="2.8" fill="${css('--acc')}"><title>${esc(r.c)} true ${fmt(r.t,4)}</title></circle>
      <circle cx="${X(i)}" cy="${Y(r.s)}" r="2.8" fill="${css('--bad')}"><title>${esc(r.c)} swapped ${fmt(r.s,3)}</title></circle>`;});
  s+=`<text x="${x0}" y="${h-8}" font-size="10.5">37 styles, sorted by swapped score (max ${fmt(Math.max(...rows.map(r=>r.s)),2)})</text>`;
  $('#swapPlot').innerHTML=s+'</svg>';
}
function hcStrip(){
  const hc=DATA.grades.m3.hard_cut.per_class;
  const vals=Object.entries(hc).map(([c,v])=>({c,z:v.max_seam_z}));
  const w=520,h=86,x0=20,x1=w-16;
  const lo=Math.log10(2),hi=Math.log10(300);
  const X=v=>x0+(x1-x0)*(Math.log10(v)-lo)/(hi-lo);
  let s=svgEl(w,h);
  s+=`<line x1="${X(3)}" y1="12" x2="${X(3)}" y2="${h-24}" stroke="${css('--warn')}" stroke-dasharray="4 3"/>
    <text x="${X(3)}" y="10" font-size="10.5" fill="${css('--warn')}" text-anchor="middle">z = 3</text>`;
  [3,10,30,100,300].forEach(t=>s+=`<text x="${X(t)}" y="${h-8}" font-size="10.5" text-anchor="middle">${t}</text>`);
  vals.forEach((v,k)=>{
    s+=`<circle cx="${X(v.z)}" cy="${26+((k*23)%5)*8}" r="3.2" fill="${css('--acc')}" fill-opacity="0.6">
      <title>${esc(v.c)} · z ${fmt(v.z,1)}</title></circle>`;});
  $('#hcStrip').innerHTML=s+'</svg>'+`<div class="kv mono">hard-cut seam z, log scale · 37/37 above 3</div>`;
}
function twinStrip(){
  const tw=DATA.grades.twins.per_twin;
  const rows=Object.entries(tw).map(([id,v])=>({id,v:v.copy_max}));
  const w=520,h=90,x0=20,x1=w-16;
  const lo=0.8,hi=1.0;
  const X=v=>x0+(x1-x0)*(v-lo)/(hi-lo);
  const cal=DATA.calibration;
  let s=svgEl(w,h);
  [[cal.initial,'0.88'],[cal.recalibrated,fmt(cal.recalibrated,3)]].forEach(([t,lab],i)=>{
    s+=`<line x1="${X(t)}" y1="14" x2="${X(t)}" y2="${h-24}" stroke="${css('--warn')}" stroke-dasharray="${i?'5 3':'2 3'}"/>
      <text x="${X(t)}" y="10" font-size="10.5" fill="${css('--warn')}" text-anchor="middle">τ ${lab}</text>`;});
  rows.forEach((r,k)=>{
    s+=`<circle cx="${X(r.v)}" cy="${30+((k*17)%4)*9}" r="3.6" fill="${css('--acc')}">
      <title>${esc(r.id)} · copy_max ${fmt(r.v)}</title></circle>`;});
  s+=`<text x="${x0}" y="${h-8}" font-size="11">0.80</text><text x="${x1}" y="${h-8}" font-size="11" text-anchor="end">1.0</text>`;
  $('#twinStrip').innerHTML=s+'</svg>'+`<div class="kv mono">11/11 flagged near-copy · copy_max 0.974–0.988</div>`;
}
function ciStrip(){
  const pc=Object.entries(DATA.ci.per_class).map(([c,v])=>({c,...v})).sort((a,b)=>a.corr-b.corr);
  const w=520,h=110,x0=20,x1=w-16;
  const X=v=>x0+(x1-x0)*v;
  let s=svgEl(w,h);
  s+=`<line x1="${X(DATA.ci.alarm)}" y1="12" x2="${X(DATA.ci.alarm)}" y2="${h-24}" stroke="${css('--warn')}" stroke-dasharray="4 3"/>
    <text x="${X(DATA.ci.alarm)}" y="10" font-size="10.5" fill="${css('--warn')}" text-anchor="middle">alarm 0.4</text>
    <line x1="${X(DATA.ci.pooled)}" y1="12" x2="${X(DATA.ci.pooled)}" y2="${h-24}" stroke="${css('--acc')}"/>
    <text x="${X(DATA.ci.pooled)}" y="10" font-size="10.5" fill="${css('--acc')}" text-anchor="middle">pooled ${fmt(DATA.ci.pooled,2)}</text>`;
  [0,.25,.5,.75,1].forEach(t=>s+=`<text x="${X(t)}" y="${h-8}" font-size="10.5" text-anchor="middle">${t}</text>`);
  pc.forEach((r,k)=>{
    s+=`<circle cx="${X(Math.max(0,r.corr))}" cy="${26+((k*19)%6)*8}" r="${2.4+Math.sqrt(r.n)/2}"
      fill="${css('--muted')}" fill-opacity="0.55">
      <title>${esc(r.c)} · corr ${fmt(r.corr,2)} · ${r.n} pairs</title></circle>`;});
  $('#ciStrip').innerHTML=s+'</svg>'+`<div class="kv mono">per-style spread ${fmt(pc[0].corr,2)} (${esc(pc[0].c)}) → ${fmt(pc[pc.length-1].corr,2)} (${esc(pc[pc.length-1].c)})</div>`;
}

/* ---------- Block C ---------- */
let BCM='app_ref';
function bcTabs(){
  const ms=['app_ref','copy_max','margin','obj_match','max_seam_z'];
  $('#bcTabs').innerHTML=ms.map(m=>
    `<button aria-pressed="${m===BCM}" data-m="${m}">${m}</button>`).join('');
  $('#bcTabs').querySelectorAll('button').forEach(b=>b.onclick=()=>{BCM=b.dataset.m;bcTabs();bcStrip();});
}
function bcStrip(){
  const fam=r=>r.arm.replace(/^(base|ic)_/,''), model=r=>r.arm.startsWith('ic')?'ic':'base';
  const fams=[...new Set(DATA.blockc.map(fam))].sort();
  const log=BCM==='max_seam_z';
  const vals=DATA.blockc.map(r=>r[BCM]).filter(v=>v!=null&&isFinite(v)&&(!log||v>0));
  if(!vals.length){$('#bcStrip').innerHTML='<span class="kv">no finite values</span>';return;}
  let lo=Math.min(...vals),hi=Math.max(...vals);
  if(log){lo=Math.log10(Math.max(lo,0.05));hi=Math.log10(hi);}
  const w=1080,rh=26,h=fams.length*rh+42,x0=150,x1=w-20;
  const X=v=>{const t=log?Math.log10(Math.max(v,0.05)):v;return x0+(x1-x0)*(t-lo)/(hi-lo+1e-9);};
  let s=svgEl(w,h);
  fams.forEach((f,i)=>{
    const y=22+i*rh;
    s+=`<text x="${x0-8}" y="${y+4}" font-size="11" text-anchor="end">${esc(f)}</text>
      <line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${css('--line')}"/>`;
    DATA.blockc.filter(r=>fam(r)===f).forEach((r,k)=>{
      const v=r[BCM]; if(v==null||!isFinite(v))return;
      const ic=model(r)==='ic';
      s+=`<circle cx="${X(v)}" cy="${y+(ic?-5:5)}" r="3.4"
        fill="${ic?css('--acc'):css('--warn')}" fill-opacity="0.65">
        <title>${esc(r.item_id)} · ${BCM} ${fmt(v)}${r.near_copy?' · near_copy':''}${r.twin_of?' · twin':''}</title></circle>`;});});
  s+=`<text x="${x0}" y="${h-8}" font-size="11">${log?'log scale':fmt(log?Math.pow(10,lo):lo,2)}</text>
      <text x="${x1}" y="${h-8}" font-size="11" text-anchor="end">${fmt(log?Math.pow(10,hi):hi,2)}</text>`;
  $('#bcStrip').innerHTML=s+'</svg>'+`<div class="legend">
    <span><span class="sw" style="background:${css('--acc')}"></span>IC-LoRA (upper lane)</span>
    <span><span class="sw" style="background:${css('--warn')}"></span>base (lower lane)</span>
    <span class="kv">${DATA.blockc_meta.n_scored} scored · ${DATA.blockc_meta.n_excluded} excluded (exp_058, reference unrecoverable)</span></div>`;
}
function bridgeTable(){
  const br=DATA.blockc_meta.bridge;
  let rows='';
  Object.entries(br).forEach(([run,v])=>{
    Object.entries(v.pairs||{}).forEach(([pair,p])=>{
      rows+=`<tr><td class="mono">${esc(run)}</td><td class="mono">${esc(pair)}</td>
        <td class="num">${p.n}</td><td class="num">${fmt(p.spearman,3)}</td></tr>`;});});
  $('#bridgeTable').innerHTML=`<table><thead><tr><th>run</th><th>v2 → v3 pair</th>
    <th>n</th><th>spearman</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/* ---------- Block D ---------- */
function coldBars(){
  const cd=DATA.grades.bar8.cold_anchors.per_metric_max_abs_delta;
  const w=520,rh=22,names=Object.keys(cd),h=names.length*rh+40,x0=110,x1=w-60;
  const lo=-7,hi=Math.log10(0.08);
  const X=v=>x0+(x1-x0)*((v<=0?lo:Math.max(Math.log10(v),lo))-lo)/(hi-lo);
  let s=svgEl(w,h);
  s+=`<line x1="${X(0.04)}" y1="8" x2="${X(0.04)}" y2="${h-24}" stroke="${css('--bad')}" stroke-dasharray="4 3"/>
    <text x="${X(0.04)}" y="${h-8}" font-size="10.5" fill="${css('--bad')}" text-anchor="middle">tol 0.04</text>`;
  names.forEach((n,i)=>{
    const y=16+i*rh,v=cd[n];
    s+=`<text x="${x0-6}" y="${y+4}" font-size="10.5" text-anchor="end">${esc(n)}</text>
      <rect x="${x0}" y="${y-5}" width="${Math.max(2,X(v)-x0)}" height="10" fill="${css('--acc')}" fill-opacity="0.8">
        <title>${esc(n)} ${v.toExponential(2)}</title></rect>
      <text x="${X(v)+6}" y="${y+4}" font-size="10">${v===0?'0':v.toExponential(1)}</text>`;});
  $('#coldBars').innerHTML=s+'</svg>'+
    `<div class="kv mono">6/6 anchors shared · worst ${Math.max(...Object.values(cd)).toExponential(2)} · pass</div>`;
}

/* ---------- bars ---------- */
function barCards(){
  const V=DATA.verdicts, ex=DATA.exam, g=DATA.grades;
  const items=[
    ['bar1_m1a_floor','M1a exam floor',`acc ${fmt(ex.bar1.acc,3)} vs ≥ ${ex.bar1.acc_min} · d ${fmt(ex.bar1.d,2)} vs ≥ ${ex.bar1.d_min}`,'#exam'],
    ['bar2_siblings','Sibling separation',`${g.siblings.n_pass}/${g.siblings.n_classes} styles ≥ ${g.siblings.min_classes} · miss: nature_bloom`,'#probes'],
    ['bar3_controls','Control floor',`${g.controls.n_pass}/${g.controls.n_classes} vs required ${g.controls.min_classes} · nature_bloom inverted`,'#probes'],
    ['bar4_splices','Splice detection',`74/74 at τ 0.88 · gap ${fmt(g.splices.gap,3)} ≥ 0.05`,'#probes'],
    ['bar5_reversal','Reversal sensitivity',`${g.reversal.wins}W/${g.reversal.losses}L · p 0.0176`,'#probes'],
    ['bar6_m3_panel','Swap + hard cut',`swap 37/37 · hard-cut 37/37 (z 5–234)`,'#probes'],
    ['bar7_copy_twins','Copy twins',`${g.twins.n_pass}/${g.twins.n_twins} flagged`,'#probes'],
    ['bar8_integration_determinism','Integration & determinism',`0 crashes · 0 error rows · warm Δ 0.0 · cold Δ ≤ 3.9e-4`,'#stability'],
  ];
  $('#barCards').innerHTML=items.map(([k,t,d,href])=>{
    const p=V[k];
    return `<div class="card"><div style="display:flex;justify-content:space-between;align-items:baseline">
      <h3>${esc(t)}</h3><span class="chip ${p?'pass':'fail'}">${p?'PASS':'FAIL'}</span></div>
      <div class="kv mono" style="margin-top:6px">${d}</div>
      <div style="margin-top:8px"><a href="${href}" style="color:var(--acc);font-size:12.5px">evidence ↓</a></div></div>`;
  }).join('');
}

/* ---------- boot & theme ---------- */
function renderAll(){
  hero();scorecards();hmTabs();drawHM();pairsTable();marginStrip();detail();
  tagTable();r1r2();r2strip();intruderTable();trustGrid();
  dumbbell();spliceStrip();revBars();swapPlot();hcStrip();twinStrip();ciStrip();
  bcTabs();bcStrip();bridgeTable();coldBars();barCards();
  $('#navchip').textContent=`overall ${DATA.overall?'PASS':'FAIL'} · 6/8`;
  $('#mastver').textContent=DATA.version;
  $('#maststamp').textContent=`commit ${DATA.stamp.commit} · corpus ${DATA.stamp.corpus}… · bars ${DATA.stamp.bars}…`;
  $('#foot').innerHTML=`transition-eval ${esc(DATA.version)} · generated by
    scripts/build_results_explorer.py from record.json + analysis/analysis.json ·
    distances recomputed with deployed metric code (job 9470438) · presentation only,
    no re-judging.`;
}
$('#hm').addEventListener('mousemove',hmHover);
$('#hm').addEventListener('mouseleave',hideTip);
$('#hm').addEventListener('click',hmClick);
new MutationObserver(renderAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',renderAll);
renderAll();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert-dir", required=True)
    args = ap.parse_args()
    cert = pathlib.Path(args.cert_dir)
    data = prep(cert)
    html = HTML.replace("__VER__", data["version"]) \
               .replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = cert / "results_explorer.html"
    out.write_text(html)
    print(f"[explorer] {out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
