"""Picker page for the AutoTransition real-transition candidates (ICLR 2027 rebuttal, 2026-11-13).

Reads misc/2026-11-02_rebuttal/autotransition/cands.json (written by cut.py there) and writes
outputs/viewers/autotransition_candidates/index.html. Media = ./media -> misc/.../autotransition/cands (viewerctl new).
Picks (target / reference per candidate) persist in the browser and export as JSON.

    /usr/bin/python3.12 scripts/viewers/build_autotransition_candidates.py
"""
import json, pathlib, datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "misc/2026-11-02_rebuttal/autotransition/cands.json"
OUT = REPO / "outputs/viewers/autotransition_candidates/index.html"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoTransition candidates</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#dedede;--acc:#2f6fdb;--tgt:#1f9d55;--ref:#c77c02;--warn:#b42318}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mute);font-size:13px}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:8px}
.bar label{display:flex;gap:4px;align-items:center}button,select{font:inherit;padding:3px 8px;border:1px solid var(--line);background:#fff;border-radius:4px;cursor:pointer}
.toc{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}.toc a{font-size:12px;padding:1px 6px;border:1px solid var(--line);border-radius:3px;text-decoration:none;color:var(--ink);background:#fff}
.toc a.has{border-color:var(--tgt);color:var(--tgt)}
main{padding:8px 16px 60px}section{margin:18px 0 6px}
section h2{font-size:16px;margin:0;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.tag{font-size:12px;font-weight:400;color:var(--mute)}.tag.none{color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-top:8px}
.card{background:var(--card);border:2px solid var(--line);border-radius:6px;padding:6px}
.card.target{border-color:var(--tgt)}.card.ref{border-color:var(--ref)}.card.flagged.hideflag{display:none}
video{width:100%;aspect-ratio:3/4;background:#000;display:block;border-radius:3px}
.tl{position:relative;height:8px;background:#e9e9e9;border-radius:2px;margin:5px 0 3px}
.tl .tr{position:absolute;top:0;bottom:0;background:#9db9ec}.tl .ph{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--ink)}
.row{display:flex;justify-content:space-between;align-items:center;gap:4px;flex-wrap:wrap}
.cid{font-weight:600;font-size:13px}.meta{color:var(--mute);font-size:11px}
.chip{font-size:11px;padding:0 5px;border-radius:3px;background:#fde8e6;color:var(--warn)}
.pick button{padding:1px 7px;font-size:12px}.pick button.on.t{background:var(--tgt);color:#fff;border-color:var(--tgt)}
.pick button.on.r{background:var(--ref);color:#fff;border-color:var(--ref)}
textarea{width:100%;height:110px;font:12px monospace;margin-top:6px;display:none}
@media (max-width:480px){.grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header>
 <h1>AutoTransition — real-transition candidates</h1>
 <div class="sub" id="sub"></div>
 <div class="bar">
  <label>speed <select id="speed"><option>1</option><option>0.5</option><option>0.25</option></select></label>
  <label><input type="checkbox" id="hideflag"> hide flagged</label>
  <span id="count" class="sub"></span>
  <button id="export">copy picks (JSON)</button><button id="clear">clear picks</button>
 </div>
 <div class="toc" id="toc"></div>
 <textarea id="out" readonly></textarea>
</header>
<main id="main"></main>
<script>
const D = __DATA__;
const KEY = "autotr_picks_v1";
let picks = {};
try { picks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(picks)); } catch (e) {} };
const WIN = 121 / 24 * 1000;
document.getElementById("sub").textContent =
  `${D.candidates.length} candidate windows over ${Object.keys(D.types).length} transition types, test split of AutoTransition (Shen et al., ECCV 2022). ` +
  `Each clip = what a method would see: 121 frames, 24 fps, centre-cropped to 3:4, 480x640. Blue band = annotated transition. ` +
  `Pick T (target: its first/last 9 frames are the endpoints, the clip is the ground truth) or R (reference demo of that type). ` +
  `“S2 look-alike” = a same-named family in SEGUE's training shader bank (by name only). Built ${D.built}.`;
function flagsOf(c) {
  const f = c.flags, out = [];
  if (f.dark_endpoints > 0.3) out.push(`dark ends ${Math.round(f.dark_endpoints * 100)}%`);
  if (f.flat_rows > 0.15) out.push(`bars ${Math.round(f.flat_rows * 100)}%`);
  if (f.landscape_src) out.push("landscape src");
  if (f.endpoint_diff < 8) out.push("ends similar");
  return out;
}
const byType = {};
D.candidates.forEach(c => { if (!c.broken_src) (byType[c.type] = byType[c.type] || []).push(c); });
const main = document.getElementById("main"), toc = document.getElementById("toc");
Object.keys(D.types).forEach(ty => {
  const T = D.types[ty], cs = byType[ty] || [];
  const a = document.createElement("a"); a.href = "#t-" + ty; a.dataset.ty = ty; toc.appendChild(a);
  const s = document.createElement("section"); s.id = "t-" + ty;
  s.innerHTML = `<h2>${ty} <span class="tag">${T.clean_windows_test} clean windows in test</span>` +
    `<span class="tag ${T.lookalike_S2 ? "" : "none"}">${T.lookalike_S2 ? "S2 look-alike: " + T.lookalike_S2 : "no S2 look-alike (by name)"}</span>` +
    `<span class="tag" data-picked="${ty}"></span></h2>`;
  const g = document.createElement("div"); g.className = "grid";
  if (!cs.length) g.innerHTML = `<div class="meta">no clip fetched yet</div>`;
  cs.forEach(c => {
    const fl = flagsOf(c), d = document.createElement("div");
    d.className = "card" + (fl.length ? " flagged" : ""); d.dataset.cid = c.cid;
    const l = c.tr_in_win_ms[0] / WIN * 100, w = (c.tr_in_win_ms[1] - c.tr_in_win_ms[0]) / WIN * 100;
    d.innerHTML = `<video muted loop playsinline preload="none" poster="${c.poster}" data-src="${c.src}"></video>
      <div class="tl"><div class="tr" style="left:${l}%;width:${w}%"></div><div class="ph"></div></div>
      <div class="row"><span class="cid">${c.cid}</span><span class="pick"><button class="t">T</button> <button class="r">R</button></span></div>
      <div class="meta">${c.end_ms - c.start_ms} ms · src ${c.src_wh[0]}x${c.src_wh[1]} · tmpl ${c.template}</div>
      <div>${fl.map(x => `<span class="chip">${x}</span>`).join(" ")}</div>`;
    const v = d.querySelector("video"), ph = d.querySelector(".ph");
    v.addEventListener("timeupdate", () => { ph.style.left = (v.currentTime * 1000 / WIN * 100) + "%"; });
    d.querySelector("button.t").onclick = () => setPick(c.cid, "target");
    d.querySelector("button.r").onclick = () => setPick(c.cid, "ref");
    g.appendChild(d);
  });
  s.appendChild(g); main.appendChild(s);
});
function setPick(cid, role) { if (picks[cid] === role) delete picks[cid]; else picks[cid] = role; save(); paint(); }
function paint() {
  document.querySelectorAll(".card").forEach(d => {
    const r = picks[d.dataset.cid];
    d.classList.toggle("target", r === "target"); d.classList.toggle("ref", r === "ref");
    d.querySelector("button.t").classList.toggle("on", r === "target");
    d.querySelector("button.r").classList.toggle("on", r === "ref");
  });
  const per = {};
  Object.entries(picks).forEach(([cid, r]) => { const ty = cid.replace(/-\d+$/, ""); per[ty] = per[ty] || {t: 0, r: 0}; per[ty][r === "target" ? "t" : "r"]++; });
  document.querySelectorAll("[data-picked]").forEach(e => { const p = per[e.dataset.picked]; e.textContent = p ? `picked: ${p.t} T · ${p.r} R` : ""; });
  toc.querySelectorAll("a").forEach(a => { const p = per[a.dataset.ty]; a.textContent = a.dataset.ty + (p ? ` (${p.t}T ${p.r}R)` : ""); a.classList.toggle("has", !!p); });
  const n = Object.values(picks), nt = n.filter(x => x === "target").length;
  document.getElementById("count").textContent = `picked ${nt} T · ${n.length - nt} R`;
}
paint();
// Only on-screen videos hold a source: load + play when visible, pause + drop the source (frees the decoder) when not.
const io = new IntersectionObserver(es => es.forEach(e => {
  const v = e.target;
  if (e.isIntersecting) {
    if (!v.getAttribute("src")) v.src = v.dataset.src;
    v.playbackRate = +document.getElementById("speed").value; v.play().catch(() => {});
  } else if (v.getAttribute("src")) {
    v.pause(); v.removeAttribute("src"); v.load();
  }
}), {rootMargin: "150px 0px"});
document.querySelectorAll("video").forEach(v => io.observe(v));
document.getElementById("speed").onchange = e => document.querySelectorAll("video[src]").forEach(v => v.playbackRate = +e.target.value);
document.getElementById("hideflag").onchange = e => document.querySelectorAll(".card.flagged").forEach(d => d.classList.toggle("hideflag", e.target.checked));
document.getElementById("export").onclick = () => {
  const o = document.getElementById("out"); o.style.display = "block";
  o.value = JSON.stringify(picks, null, 1); o.select();
  try { navigator.clipboard.writeText(o.value); } catch (e) {}
};
document.getElementById("clear").onclick = () => { if (confirm("Clear all picks?")) { picks = {}; save(); paint(); } };
</script></body></html>
"""


def main():
    d = json.load(open(SRC))
    d["built"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for c in d["candidates"]:
        if not c.get("broken_src"):
            c["src"], c["poster"] = f"media/{c['cid']}.mp4", f"media/{c['cid']}.jpg"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(d, separators=(",", ":"))))
    ok = sum(1 for c in d["candidates"] if not c.get("broken_src"))
    print(f"wrote {OUT.relative_to(REPO)}: {ok} clips, {len(d['types'])} types")


if __name__ == "__main__":
    main()
