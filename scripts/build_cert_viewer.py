"""Build a self-contained HTML viewer for a transition-eval certification run.

Reads a certification output directory (record.json + exam/ + audit) plus the
corpus manifest, inlines everything as JSON, and writes viewer.html next to
the artifacts. No server, no external assets — open the file anywhere.

    python scripts/build_cert_viewer.py \
        --cert-dir outputs/eval/certification/3.0.0-draft.7 \
        --corpus data/processed/transitions_std121/corpus_manifest.json
"""

from __future__ import annotations

import argparse
import json
import pathlib


def load(p: pathlib.Path):
    return json.loads(p.read_text()) if p.exists() else None


def build_data(cert_dir: pathlib.Path, corpus_path: pathlib.Path) -> dict:
    record = load(cert_dir / "record.json")
    exam = load(cert_dir / "exam/exam.json")
    trust = load(cert_dir / "exam/trust_map.json")
    audit = load(cert_dir / "content_invariance.json")
    stamp = load(cert_dir / "stamp.json")
    corpus = load(corpus_path)
    if record is None or exam is None:
        raise SystemExit(f"no record.json/exam under {cert_dir}")

    # enrich bar-3 rows with control arm + sidedness from the scored items
    ctrl_meta = {}
    items = cert_dir / "cert_siblings/items.jsonl"
    if items.exists():
        for line in items.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("arm", "").startswith("control"):
                cls = r["item_id"].split("__sib__")[-1]
                ctrl_meta[cls] = {"arm": r["arm"], "sidedness": r.get("sidedness"),
                                  "core_frames": r.get("core_frames"),
                                  "core_frac_strict": r.get("core_frac_strict")}

    classes = {}
    if corpus:
        for c, v in corpus["classes"].items():
            classes[c] = {"n": v.get("n_clips"), "sidedness": v.get("sidedness"),
                          "camera": "camera" in v.get("tags", [])}

    return {"record": record, "exam": exam, "trust": trust, "audit": audit,
            "stamp": stamp, "classes": classes, "ctrl_meta": ctrl_meta}


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#f7f8f7; --surface:#ffffff; --ink:#20282b; --muted:#5c6b70;
  --line:#dce3e2; --accent:#177e89; --accent-ink:#0e5a62;
  --pass:#2e7d4f; --fail:#b3403a; --warn:#a8741f;
  --pass-bg:#e5f1ea; --fail-bg:#f7e8e6; --warn-bg:#f5edda;
  --heat-good:23,126,137; --heat-bad:179,64,58;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#131a1c; --surface:#1b2427; --ink:#e2eae9; --muted:#8fa1a3;
  --line:#2c393c; --accent:#4fb3bf; --accent-ink:#7ccdd6;
  --pass:#57b380; --fail:#e07b72; --warn:#d9a54c;
  --pass-bg:#1d3227; --fail-bg:#3a2422; --warn-bg:#372c17;
  --heat-good:79,179,191; --heat-bad:224,123,114;
}}
:root[data-theme="light"]{
  --bg:#f7f8f7; --surface:#ffffff; --ink:#20282b; --muted:#5c6b70;
  --line:#dce3e2; --accent:#177e89; --accent-ink:#0e5a62;
  --pass:#2e7d4f; --fail:#b3403a; --warn:#a8741f;
  --pass-bg:#e5f1ea; --fail-bg:#f7e8e6; --warn-bg:#f5edda;
  --heat-good:23,126,137; --heat-bad:179,64,58;
}
:root[data-theme="dark"]{
  --bg:#131a1c; --surface:#1b2427; --ink:#e2eae9; --muted:#8fa1a3;
  --line:#2c393c; --accent:#4fb3bf; --accent-ink:#7ccdd6;
  --pass:#57b380; --fail:#e07b72; --warn:#d9a54c;
  --pass-bg:#1d3227; --fail-bg:#3a2422; --warn-bg:#372c17;
  --heat-good:79,179,191; --heat-bad:224,123,114;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
main{max-width:1160px;margin:0 auto;padding:28px 20px 80px;display:flex;
  flex-direction:column;gap:28px;}
h1{font-size:21px;margin:0;font-weight:650;letter-spacing:-.01em;text-wrap:balance}
h2{font-size:16px;margin:0 0 4px;font-weight:650}
.eyebrow{font-size:11px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted)}
.mono,td.num,.chip .stat{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;
  padding:18px 20px;}
.card.fail-detail{border-left:3px solid var(--fail)}
.card.pass-detail{border-left:3px solid var(--pass)}
header.card{display:flex;flex-direction:column;gap:10px}
.headline{display:flex;flex-wrap:wrap;align-items:center;gap:12px}
.verdict-pill{font-weight:700;font-size:14px;padding:4px 14px;border-radius:999px;
  letter-spacing:.04em}
.verdict-pill.fail{background:var(--fail);color:#fff}
.verdict-pill.pass{background:var(--pass);color:#fff}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--muted)}
.meta b{color:var(--ink);font-weight:600}
.chips{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.chip{background:var(--surface);border:1px solid var(--line);border-radius:8px;
  padding:12px 14px;display:flex;flex-direction:column;gap:3px;cursor:pointer;
  text-decoration:none;color:inherit}
.chip:hover{border-color:var(--accent)}
.chip .top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.chip .name{font-weight:650;font-size:13.5px}
.chip .stat{font-size:12.5px;color:var(--muted)}
.badge{font-size:10.5px;font-weight:700;letter-spacing:.06em;padding:2px 8px;
  border-radius:4px}
.badge.pass{background:var(--pass-bg);color:var(--pass)}
.badge.fail{background:var(--fail-bg);color:var(--fail)}
.badge.warn{background:var(--warn-bg);color:var(--warn)}
.badge.info{background:var(--line);color:var(--muted)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
  font-weight:600;text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);
  white-space:nowrap}
td{padding:5px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td.num{text-align:right}
tr.miss td{background:var(--fail-bg)}
.heat{text-align:right}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-left:6px;
  vertical-align:1px}
.dot.t{background:var(--pass)} .dot.u{border:1.5px solid var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 14px;font-size:13px}
.kv div:nth-child(odd){color:var(--muted)}
.note{font-size:12.5px;color:var(--muted);max-width:72ch}
.section-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;
  flex-wrap:wrap}
.barlist{display:flex;flex-direction:column;gap:4px;margin-top:8px}
.barrow{display:grid;grid-template-columns:180px 1fr 90px;gap:10px;align-items:center;
  font-size:12.5px}
.barrow .track{height:8px;border-radius:4px;background:var(--line);position:relative}
.barrow .fill{position:absolute;left:0;top:0;bottom:0;border-radius:4px;
  background:var(--accent);opacity:.85}
details{font-size:13px} summary{cursor:pointer;color:var(--accent-ink);font-weight:600}
nav{position:sticky;top:0;z-index:5;background:var(--bg);padding:8px 0;display:flex;
  gap:14px;flex-wrap:wrap;font-size:12.5px;border-bottom:1px solid var(--line)}
nav a{color:var(--muted);text-decoration:none;font-weight:600}
nav a:hover{color:var(--accent-ink)}
.claims{font-size:13.5px;font-style:italic;max-width:78ch}
@media (prefers-reduced-motion: no-preference){.chip{transition:border-color .12s}}
</style></head><body><main id="app"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const R = D.record, EX = D.exam, TR = D.trust || {}, AU = D.audit || {},
      CL = D.classes || {}, CM = D.ctrl_meta || {};
const fmt = (v, n=3) => (v===null||v===undefined||Number.isNaN(v)) ? '—'
      : (typeof v === 'number' ? v.toFixed(n) : String(v));
const pct = v => (v===null||v===undefined) ? '—' : (100*v).toFixed(0)+'%';
const badge = (ok, txt) => `<span class="badge ${ok?'pass':'fail'}">${txt ?? (ok?'PASS':'FAIL')}</span>`;
function heat(v){ if(v===null||v===undefined||Number.isNaN(v)) return '';
  const good = v>=0.5, base = good?'--heat-good':'--heat-bad';
  const a = good ? 0.12+0.38*(v-0.5)/0.5 : 0.12+0.38*(0.5-v)/0.5;
  return `background:rgba(var(${base}),${a.toFixed(2)})`; }
const el = html => { const d=document.createElement('div'); d.innerHTML=html; return d; };
const app = document.getElementById('app');

/* ---------- header ---------- */
const st = D.stamp || R.stamp || {};
const git = (st.git||{});
app.append(el(`<header class="card">
 <div class="headline">
   <h1>Certification record — transition-eval/${R.version}</h1>
   <span class="verdict-pill ${R.overall_pass?'pass':'fail'}">OVERALL ${R.overall_pass?'PASS':'FAIL'}</span>
 </div>
 <div class="meta">
   <span>commit <b class="mono">${git.commit_short||'—'}</b></span>
   <span>bars <b class="mono">${(R.bars_sha256||'').slice(0,16)}…</b></span>
   <span>corpus <b class="mono">${(st.corpus_sha256||'').slice(0,16)}…</b></span>
   <span>instrument dirty: <b>${git.dirty===undefined?'—':git.dirty}</b></span>
   ${R.post_hoc_assembly?'<span class="badge warn">RECORD ASSEMBLED POST-HOC (driver crash — see §8)</span>':''}
 </div></header>`).firstElementChild);

/* ---------- nav + verdict chips ---------- */
const BARS = [
 ['bar1_m1a_floor','1 · M1a floor', ()=>`acc ${fmt(EX.bar1.acc)} vs ${EX.bar1.acc_min} · d ${fmt(EX.bar1.d,2)} vs ${EX.bar1.d_min}`,'#exam'],
 ['bar2_siblings','2 · Siblings', ()=>{const g=R.grades.siblings;return `${g.n_pass}/${g.n_classes} (min ${g.min_classes})`},'#b23'],
 ['bar3_controls','3 · Controls', ()=>{const g=R.grades.controls;return `${g.n_pass}/${g.n_classes} (min ${g.min_classes})`},'#b23'],
 ['bar4_splices','4 · Splices', ()=>R.grades.splices.reason||`gap ${fmt(R.grades.splices.gap)}`,'#crashed'],
 ['bar5_reversal','5 · Reversal', ()=>{const g=R.grades.reversal;return `${g.wins}W/${g.losses}L · ${g.rule}`},'#b5'],
 ['bar6_m3_panel','6 · M3 panel', ()=>{const g=R.grades.m3_panel;return g.swap?`swap ${g.swap.n_pass} · cut ${g.hard_cut.n_pass}`:'no rows'},'#crashed'],
 ['bar7_copy_twins','7 · Copy twins', ()=>{const g=R.grades.copy_twins;return `${g.n_pass}/${g.n_twins} flagged`},'#crashed'],
 ['bar8_integration_determinism','8 · Stability', ()=>{const g=R.grades.bar8;return `${g.crashes.length} crashes · warm ${g.warm?fmt(g.warm.worst,7):'—'}`},'#b8'],
];
app.append(el(`<nav>
 <a href="#exam">Exam & adoption</a><a href="#matrix">Per-class matrix</a>
 <a href="#b23">Bars 2–3</a><a href="#b5">Bar 5</a><a href="#crashed">Bars 4/6/7</a>
 <a href="#b8">Bar 8</a><a href="#audit">Audit</a><a href="#blockc">Block C</a>
 <a href="#calib">Calibration & claims</a></nav>`).firstElementChild);
app.append(el(`<div class="chips">`+BARS.map(([k,name,statf,href])=>{
  const ok = R.verdicts[k];
  let stat=''; try{stat=statf()}catch(e){stat='—'}
  return `<a class="chip" href="${href}"><div class="top"><span class="name">${name}</span>${badge(ok)}</div><span class="stat">${stat}</span></a>`;
}).join('')+`</div>`).firstElementChild);

/* ---------- exam & adoption ---------- */
const ma = EX.mask_adoption, mo = EX.motion_adoption, o7 = EX.o7_conditional;
const stt = ma.checks.stratum_sign_test, mst = mo.checks.stratum_sign_test;
const ndg = ma.checks.nondegenerate_rates;
const r1rows = Object.entries(EX.r1).map(([k,v])=>{
  const ci = v.accuracy_wilson95||[];
  return `<tr><td>${k}</td><td class="num" style="${heat(v.accuracy_1nn)}">${fmt(v.accuracy_1nn)}</td>
  <td class="num">[${fmt(ci[0],2)}, ${fmt(ci[1],2)}]</td><td class="num">${fmt(v.chance,3)}</td>
  <td class="num">${fmt(v.separation_cohens_d,2)}</td><td class="num">${pct(v.coverage)}</td></tr>`;}).join('');
app.append(el(`<section class="card" id="exam">
 <div class="section-head"><div><span class="eyebrow">Block A</span>
 <h2>Exam — readouts & pre-registered adoption</h2></div>
 ${badge(R.verdicts.bar1_m1a_floor,'BAR 1 '+(R.verdicts.bar1_m1a_floor?'PASS':'FAIL'))}</div>
 <div class="grid2" style="margin-top:10px">
  <div><div class="eyebrow">Mask adoption → <b>${ma.winner}</b></div>
   <div class="kv">
    <div>stratum sign test</div><div class="mono">${stt.wins}W / ${stt.losses}L / ${stt.ties}T · p=${fmt(stt.p_one_sided,4)}</div>
    <div>overall tolerance ok</div><div>${ma.checks.overall_ok}</div>
    <div>regression guard</div><div>${ma.checks.regression_guard_ok} ${ma.checks.regressed_classes.length?('· '+ma.checks.regressed_classes.join(', ')):''}</div>
    <div>non-degenerate (1-sided strict cores)</div>
    <div class="mono">sided ${pct(ndg.v3_sided)} · envelope ${pct(ndg.v2_envelope)} · all ${pct(ndg.all_frames)}</div>
   </div></div>
  <div><div class="eyebrow">Motion adoption → <b>${mo.winner}</b></div>
   <div class="kv">
    <div>stratum sign test</div><div class="mono">${mst.wins}W / ${mst.losses}L / ${mst.ties}T · p=${fmt(mst.p_one_sided,4)}</div>
    <div>M1c guard (median, defined)</div>
    <div class="mono">dec ${fmt(mo.checks.m1c_median_recall_defined,2)} vs inc ${fmt(mo.checks.incumbent_median_recall_defined,2)} → ${mo.checks.m1c_guard_ok}</div>
    <div>O7 conditional</div>
    <div>camera-stratum mean recall <b class="mono">${fmt(o7.camera_stratum_mean_recall,3)}</b> → Huber triggered: <b>${o7.huber_triggered}</b></div>
   </div></div>
 </div>
 <div class="tablewrap" style="margin-top:14px"><table>
  <thead><tr><th>R1 / R2 readout</th><th>acc (1-NN / margin)</th><th>Wilson 95</th><th>chance</th><th>d</th><th>coverage</th></tr></thead>
  <tbody>${r1rows}
  <tr><td>m2b pool margin (R2, ${EX.r2.n_graded} clips)</td>
   <td class="num" style="${heat(EX.r2.accuracy)}">${fmt(EX.r2.accuracy)}</td>
   <td class="num">—</td><td class="num">—</td><td class="num">—</td>
   <td class="num">margin μ ${fmt(EX.r2.margins_mean,3)}</td></tr></tbody></table></div>
 <p class="note">Bar 1 gates on the winning mask variant (${ma.winner}): acc ${fmt(EX.bar1.acc)} ≥ ${EX.bar1.acc_min} is the failing clause; d ${fmt(EX.bar1.d,2)} ≥ ${EX.bar1.d_min} passes. Frozen floor was anchored to exp_054's 0.851/d 2.22 — measured on 47 clips / 11 styles at chance 0.213.</p>
</section>`).firstElementChild);

/* ---------- per-class matrix ---------- */
const m1aW = EX.r1['m1a__'+ma.winner].per_class_recall;
const rows = Object.keys(TR).sort().map(c=>{
  const t=TR[c]||{}, meta=CL[c]||{}, rec=t.recall||{};
  const aud=(AU.per_class||{})[c]||{};
  const trust = m => t[m]===true?'<span class="dot t" title="trusted"></span>':'<span class="dot u" title="untrusted"></span>';
  const r2v=(EX.r2.per_class_recall||{})[c];
  const cells = [
    ['m1a', m1aW[c], trust('m1a')], ['m1b', rec.m1b, trust('m1b')],
    ['m1c', rec.m1c, trust('m1c')], ['m2b', r2v, trust('m2b')]];
  return `<tr><td>${c}${t.eligible===false?' <span class="badge info">n&lt;4</span>':''}</td>
   <td class="num">${meta.n??t.n_clips??'—'}</td>
   <td>${meta.sidedness||'—'}${meta.camera?' 🎥':''}</td>`+
   cells.map(([m,v,d])=>`<td class="num heat" style="${heat(v)}">${fmt(v,2)}${d}</td>`).join('')+
   `<td class="num">${fmt((EX.m1c_definedness||{})[c],2)}</td>
    <td class="num" style="${aud.corr!==undefined&&aud.corr!==null?heat(1-Math.abs(aud.corr)):''}">${fmt(aud.corr,2)}</td></tr>`;}).join('');
app.append(el(`<section class="card" id="matrix">
 <span class="eyebrow">Trust map · exam recalls · audit</span>
 <h2>Per-class matrix (winning variants: ${ma.winner} / ${mo.winner})</h2>
 <p class="note">Dot filled = trusted (recall ≥ 0.5, n ≥ 4; M1c also needs definedness ≥ 0.5). Audit corr = within-class style-sim vs content-sim (want ≈ 0; heat penalizes |corr|).</p>
 <div class="tablewrap"><table>
  <thead><tr><th>class</th><th>n</th><th>side</th><th>M1a</th><th>M1b</th><th>M1c</th><th>M2b (R2)</th><th>M1c def.</th><th>audit corr</th></tr></thead>
  <tbody>${rows}</tbody></table></div></section>`).firstElementChild);

/* ---------- bars 2 & 3 ---------- */
const g2=R.grades.siblings, g3=R.grades.controls;
const b23rows = Object.keys(g2.per_class).sort().map(c=>{
  const s=g2.per_class[c], k=g3.per_class[c]||{}, m=CM[c]||{};
  const missS=!s.pass, missC=!k.pass;
  return `<tr class="${(missS||missC)?'miss':''}"><td>${c}</td>
   <td>${(m.arm||'').replace('control_','')||'—'}</td>
   <td class="num">${fmt(s.m1a)}</td><td class="num">${fmt(s.m1a_control??k.control_m1a)}</td>
   <td>${s.near_copy===true?'<span class="badge fail">fired</span>':'silent'}</td>
   <td class="num">${fmt(s.copy_max)}</td>
   <td>${k.core_degenerate===true?'yes':'no'} <span class="mono" style="color:var(--muted)">${m.core_frames!==undefined?('('+m.core_frames+'f)'):''}</span></td>
   <td>${badge(!!s.pass,'B2 '+(s.pass?'✓':'✗'))} ${badge(!!k.pass,'B3 '+(k.pass?'✓':'✗'))}</td></tr>`;}).join('');
app.append(el(`<section class="card ${(g2.pass&&g3.pass)?'':'fail-detail'}" id="b23">
 <div class="section-head"><div><span class="eyebrow">Block B — constructed truth</span>
 <h2>Bars 2 & 3 — bar-pair siblings vs degenerate controls</h2></div>
 <div>${badge(g2.pass,'BAR 2 '+g2.n_pass+'/'+g2.n_classes)} ${badge(g3.pass,'BAR 3 '+g3.n_pass+'/'+g3.n_classes)}</div></div>
 <p class="note">Bar 2: sibling M1a beats its class control ∧ M2a silent (need ${g2.min_classes}). Bar 3: control below sibling ∧ core_degenerate fires (need ${g3.min_classes}). Highlighted rows miss at least one bar.</p>
 <div class="tablewrap"><table>
  <thead><tr><th>class</th><th>control arm</th><th>sibling M1a</th><th>control M1a</th><th>M2a</th><th>sib copy_max</th><th>ctl degenerate</th><th>verdicts</th></tr></thead>
  <tbody>${b23rows}</tbody></table></div></section>`).firstElementChild);

/* ---------- bar 5 ---------- */
const g5=R.grades.reversal;
const b5rows=(g5.rows||[]).map(r=>`<tr class="${r.drop>1e-9?'':'miss'}">
 <td>${r['class']}</td><td class="mono" style="font-size:12px">${r.gen} vs ${r.ref}</td>
 <td class="num">${fmt(r.corr_unreversed)}</td><td class="num">${fmt(r.corr_reversed)}</td>
 <td class="num">${fmt(r.drop)}</td></tr>`).join('');
app.append(el(`<section class="card ${g5.pass?'pass-detail':'fail-detail'}" id="b5">
 <div class="section-head"><div><span class="eyebrow">Block B — direction sensitivity</span>
 <h2>Bar 5 — reversal on enumerated-sensitive camera pairs</h2></div>
 ${badge(g5.pass,'BAR 5 '+(g5.pass?'PASS':'FAIL'))}</div>
 <p class="note">${g5.rule} · ${g5.wins} wins / ${g5.losses} losses · median drop ${fmt(g5.median_drop)} (ties excluded). Reversed references are re-tracked real videos, not analytic reversals.</p>
 <div class="tablewrap"><table>
 <thead><tr><th>class</th><th>pair (gen vs ref)</th><th>corr unreversed</th><th>corr reversed</th><th>drop</th></tr></thead>
 <tbody>${b5rows}</tbody></table></div></section>`).firstElementChild);

/* ---------- crashed bars 4/6/7 ---------- */
const g4=R.grades.splices, g6=R.grades.m3_panel, g7=R.grades.copy_twins;
const twinRows=Object.entries(g7.per_twin||{}).map(([id,v])=>`<tr class="${v.pass?'':'miss'}">
 <td class="mono" style="font-size:12px">${id}</td><td>${v.reason||''}</td>
 <td class="num">${fmt(v.copy_max)}</td><td>${v.near_copy===undefined?'—':v.near_copy}</td>
 <td class="num">${fmt(v.max_seam_z,1)}</td><td>${badge(!!v.pass,v.pass?'flagged':'unflagged')}</td></tr>`).join('');
app.append(el(`<section class="card fail-detail" id="crashed">
 <div class="section-head"><div><span class="eyebrow">Blocks B & C</span>
 <h2>Bars 4, 6, 7 — ungraded: scoring stages crashed</h2></div>
 <div>${badge(false,'BAR 4')} ${badge(false,'BAR 6')} ${badge(false,'BAR 7')}</div></div>
 <div class="kv" style="margin-top:8px">
  <div>Bar 4 splices</div><div>${g4.reason||('gap '+fmt(g4.gap))} — 74 splice videos built, never scored; τ_copy NOT recalibrated</div>
  <div>Bar 6 M3 panel</div><div>${g6.swap?('swap '+g6.swap.n_pass+' · cut '+g6.hard_cut.n_pass):'no swap/hard-cut rows scored'}</div>
  <div>Bar 7 copy twins</div><div>${g7.n_pass}/${g7.n_twins} — Block C scoring crashed before any of the 11 verified copy-regime twins scored</div>
 </div>
 <details style="margin-top:10px"><summary>per-twin table (all unscored)</summary>
 <div class="tablewrap"><table><thead><tr><th>twin id</th><th>reason</th><th>copy_max</th><th>near_copy</th><th>max_seam_z</th><th>flag</th></tr></thead>
 <tbody>${twinRows}</tbody></table></div></details></section>`).firstElementChild);

/* ---------- bar 8 ---------- */
const g8=R.grades.bar8, warm=g8.warm||{};
const wrow=Object.entries(warm.per_metric_max_abs_delta||{}).map(([m,v])=>
 `<tr><td>${m}</td><td class="num">${v===0?'0.0':fmt(v,9)}</td></tr>`).join('');
app.append(el(`<section class="card fail-detail" id="b8">
 <div class="section-head"><div><span class="eyebrow">Block D</span>
 <h2>Bar 8 — integration & determinism</h2></div>${badge(g8.pass,'BAR 8 '+(g8.pass?'PASS':'FAIL'))}</div>
 <div class="grid2" style="margin-top:8px">
  <div><div class="eyebrow">Crashes (no_crash = ${g8.no_crash})</div>
   <ol style="font-size:13px;padding-left:18px;margin:6px 0">${(g8.crashes||[]).map(c=>`<li style="margin-bottom:6px">${c}</li>`).join('')}</ol></div>
  <div><div class="eyebrow">Warm rerun — ${warm.n_shared??'—'} rows, worst |Δ| = ${warm.worst===0?'0.0':fmt(warm.worst,9)} (tol 1e-6) ${badge(!!warm.pass)}</div>
   <div class="tablewrap"><table><thead><tr><th>metric</th><th>max |Δ|</th></tr></thead><tbody>${wrow}</tbody></table></div>
   <p class="note">Cold anchors: ${g8.cold_anchors?JSON.stringify(g8.cold_anchors):'never ran (driver crashed at the anchor rule)'}</p></div>
 </div></section>`).firstElementChild);

/* ---------- audit ---------- */
const audRows = Object.entries(AU.per_class||{}).filter(([,v])=>v.corr!==null&&v.corr!==undefined)
 .sort((a,b)=>Math.abs(b[1].corr)-Math.abs(a[1].corr)).map(([c,v])=>{
  const w=Math.min(100,Math.abs(v.corr)*100);
  return `<div class="barrow"><span>${c} <span style="color:var(--muted)">(${v.n_pairs}p)</span></span>
   <div class="track"><div class="fill" style="width:${w}%;${Math.abs(v.corr)>=0.4?'background:var(--fail)':''}"></div></div>
   <span class="mono" style="text-align:right">${fmt(v.corr,2)}</span></div>`;}).join('');
app.append(el(`<section class="card ${AU.pooled_partial_corr>0.4?'fail-detail':''}" id="audit">
 <div class="section-head"><div><span class="eyebrow">Required record artifact — NON-GATING</span>
 <h2>Content-invariance audit</h2></div>
 <span class="badge ${AU.pooled_partial_corr>0.4?'warn':'pass'}">pooled ${fmt(AU.pooled_partial_corr,3)} vs alarm 0.4</span></div>
 <p class="note">Within-class Pearson of style-similarity (deployed M1a on cores) vs content-similarity (endpoint features), pooled over ${AU.n_pairs} sibling pairs after per-class centering. Expectation ≈ 0; measured ${fmt(AU.pooled_partial_corr,2)} — M1a within-class variation substantially tracks content overlap. Red bars exceed the alarm level.</p>
 <div class="barlist">${audRows}</div></section>`).firstElementChild);

/* ---------- block C ---------- */
const bc=R.blockc||{};
const excl=bc.excluded||[];
const byReason={};
excl.forEach(e=>{const k=(e.run||'?')+' — '+(e.reason||'?');(byReason[k]=byReason[k]||[]).push(e.item_id||'?')});
app.append(el(`<section class="card" id="blockc">
 <span class="eyebrow">Block C — realism (descriptive)</span>
 <h2>Archives: ${bc.n_convertible??'—'} convertible · ${excl.length} excluded loudly · ${bc.n_scored??0} scored</h2>
 <p class="note">Bridge & arm distributions: ${typeof bc.bridge==='string'?bc.bridge:'computed (see record.json)'}.</p>
 ${Object.entries(byReason).map(([k,ids])=>`<details><summary>${k} (${ids.length})</summary>
  <p class="mono note" style="font-size:11.5px">${ids.join(' · ')}</p></details>`).join('')}
</section>`).firstElementChild);

/* ---------- calibration & claims ---------- */
const cal=R.calibration||{}, fl=cal.sidedness_floors||{};
app.append(el(`<section class="card" id="calib">
 <span class="eyebrow">Calibration outputs & claims</span>
 <h2>Constants, floors, pending items</h2>
 <div class="kv" style="margin-top:6px">
  <div>τ_copy</div><div class="mono">initial ${cal.tau_copy?cal.tau_copy.initial:'—'} · recalibrated ${cal.tau_copy&&cal.tau_copy.recalibrated!==null&&cal.tau_copy.recalibrated!==undefined?fmt(cal.tau_copy.recalibrated):'NOT SET (splice crash)'}</div>
  <div>core fallback</div><div class="mono">k=${cal.core_fallback?cal.core_fallback.min_frames:'—'}, δ=${cal.core_fallback?cal.core_fallback.delta:'—'} (flagged-only)</div>
  <div>control floors</div><div class="mono">two-sided ${fl.twosided?fmt(fl.twosided.mean)+' (n='+fl.twosided.n+')':'—'} · one-sided ${fl.onesided?fmt(fl.onesided.mean)+' (n='+fl.onesided.n+')':'—'}</div>
  <div>reversal-sensitive set</div><div>${(cal.reversal_sensitive_set||[]).length} pairs (enumerated pre-freeze)</div>
  <div>σ_seed</div><div>${cal.sigma_seed?cal.sigma_seed.status:'—'} — gates the first model report, not the tag</div>
 </div>
 <p class="claims" style="margin-top:14px">“${R.claims||''}”</p>
 ${R.post_hoc_assembly?`<details><summary>post-hoc assembly note</summary>
  <div class="kv" style="margin-top:6px"><div>why</div><div>${R.post_hoc_assembly.why}</div>
  <div>how</div><div>${R.post_hoc_assembly.how}</div>
  <div>not computed</div><div>${(R.post_hoc_assembly.not_computed||[]).join(' · ')}</div></div></details>`:''}
</section>`).firstElementChild);
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default=None, help="default: <cert-dir>/viewer.html")
    args = ap.parse_args()
    cert_dir = pathlib.Path(args.cert_dir)
    data = build_data(cert_dir, pathlib.Path(args.corpus))
    ver = data["record"].get("version", "?")
    html = (TEMPLATE
            .replace("__TITLE__", f"cert {ver} — {'PASS' if data['record'].get('overall_pass') else 'FAIL'}")
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    out = pathlib.Path(args.out) if args.out else cert_dir / "viewer.html"
    out.write_text(html)
    print(f"[viewer] {out}  ({out.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
