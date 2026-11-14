"""Endpoint pairs for the real-transition TEG tier (ICLR 2027 rebuttal, 2026-11-13): MovieCuts (shot before / after a
hard cut), ImageNet-VID (first / last second of the same video), Edit3K (its two material slots). Each card = A | B side by
side (1 s each), motion of A and of B, DINOv2-base cosine between A and B; sortable, filterable, starrable.

Reads misc/2026-11-02_rebuttal/real_transitions/endpoint_pairs.json (written by endpoint_pairs.py there) and writes
outputs/viewers/endpoint_pairs/index.html; ./media -> that pairs dir.

    /usr/bin/python3.12 scripts/viewers/build_endpoint_pairs.py
"""
import json, pathlib, datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
RT = REPO / "misc/2026-11-02_rebuttal/real_transitions"
OUT = REPO / "outputs/viewers/endpoint_pairs/index.html"
SOURCES = [("moviecuts", "MovieCuts", "A = 1 s ending 2 frames before the hard cut, B = 1 s starting 2 frames after it (same clip); letterbox cropped. Label = editorial cut type."),
           ("humanvid", "HumanVid (procedural tier)", "The 63 reserved never-trained HumanVid clips the procedural tier used (480x640, 24 fps, clean stock footage of people): A = first second of clip X, B = last second of clip Y (the procedural conds are X[0:9] and Y[112:121]). Label = the procedural tier's 16 target pairs, or random pairs from the same clips."),
           ("imagenet_vid", "ImageNet-VID", "Same video (VidVRD clip, first <=12 s): A = first second, B = last second. Label = main object."),
           ("edit3k", "Edit3K pairs", "Edit3K's two ImageNet-VID material slots (as rendered with a plain dissolve): A = slot A, B = slot B.")]

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Endpoint pairs</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1f;--mute:#66666b;--line:#dedede;--acc:#2f6fdb;--star:#d99a00;--ok:#1f9d55}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mute);font-size:13px;max-width:1250px}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:8px}
.tabs button{font:inherit;padding:3px 10px;border:1px solid var(--line);background:#fff;border-radius:4px;cursor:pointer}
.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
select,input[type=search],button{font:inherit;padding:2px 8px;border:1px solid var(--line);border-radius:4px;background:#fff}
textarea{width:100%;height:50px;font:12px monospace;display:none;margin-top:6px}
main{padding:8px 16px 60px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px}.card.starred{border:2px solid var(--star)}
.ab{position:relative}.ab span{position:absolute;top:4px;font-size:11px;background:rgba(0,0,0,.6);color:#fff;padding:0 5px;border-radius:2px}
.ab .a{left:4px}.ab .b{left:calc(50% + 8px)}
video{width:100%;aspect-ratio:656/180;display:block;background:#111;border-radius:3px}
.row{display:flex;justify-content:space-between;align-items:center;gap:6px;margin-top:4px;flex-wrap:wrap}
.lab{font-weight:600}.m{color:var(--mute);font-size:12px}.num b{font-weight:600}.mv{color:var(--ok)}
.hide{display:none}
</style></head><body>
<header><h1>Endpoint pairs — motion of each endpoint, DINO similarity between them</h1>
<div class="sub" id="sub"></div>
<div class="bar"><span class="tabs" id="tabs"></span>
<label>sort <select id="sort">
 <option value="dino_up">A–B DINO ↑ (most different first)</option><option value="dino_down">A–B DINO ↓ (most similar first)</option>
 <option value="mmin_down">motion, weaker side ↓</option><option value="mmin_up">motion, weaker side ↑</option><option value="mmax_down">motion, stronger side ↓</option>
</select></label>
<label><input type="checkbox" id="both"> both moving (&gt;1.0)</label>
<input type="search" id="q" placeholder="filter label / id"><span id="cnt" class="m"></span><button id="export">copy starred</button></div>
<div class="sub" id="about"></div><textarea id="out" readonly></textarea></header>
<main><div class="grid" id="grid"></div></main>
<script>
const D = __DATA__, KEY = "endpoint_pairs_star_v1";
let star = {}; try { star = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(star)); } catch (e) {} };
let cur = D.sources[0][0];
document.getElementById("sub").textContent = `Each card = endpoint A | endpoint B, 1 s (24 frames) each, looping. Motion = mean pixel change per frame ` +
  `(64x36 gray, 0-255; stills ~0.01, clear motion > 1). A–B DINO = DINOv2-base CLS cosine between A and B (3 frames each): low = different scenes. Built ${D.built}.`;
const tabs = document.getElementById("tabs");
D.sources.forEach(([k, name]) => {
  const n = D.pairs.filter(p => p.source === k).length, b = document.createElement("button");
  b.textContent = `${name} (${n})`; b.dataset.k = k; b.onclick = () => { cur = k; render(); }; tabs.appendChild(b);
});
const g = document.getElementById("grid");
const io = new IntersectionObserver(es => es.forEach(e => {
  const v = e.target;
  if (e.isIntersecting) { if (!v.getAttribute("src")) v.src = v.dataset.src; v.play().catch(() => {}); }
  else if (v.getAttribute("src")) { v.pause(); v.removeAttribute("src"); v.load(); }
}), {rootMargin: "150px 0px"});
function render() {
  g.querySelectorAll("video[src]").forEach(v => { v.pause(); v.removeAttribute("src"); v.load(); });
  io.disconnect(); g.innerHTML = "";
  tabs.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.k === cur));
  document.getElementById("about").textContent = D.sources.find(s => s[0] === cur)[2];
  const k = document.getElementById("sort").value, both = document.getElementById("both").checked, q = document.getElementById("q").value.trim().toLowerCase();
  const lo = p => Math.min(...p.motion), hi = p => Math.max(...p.motion);
  const key = {dino_up: p => p.dino, dino_down: p => -p.dino, mmin_down: p => -lo(p), mmin_up: p => lo(p), mmax_down: p => -hi(p)}[k];
  const ps = D.pairs.filter(p => p.source === cur && (!both || lo(p) > 1) && (!q || (p.label + " " + p.id).toLowerCase().includes(q))).sort((a, b) => key(a) - key(b));
  document.getElementById("cnt").textContent = `${ps.length} pairs · ${Object.keys(star).length} starred`;
  ps.forEach(p => {
    const sk = p.source + ":" + p.id, c = document.createElement("div"); c.className = "card" + (star[sk] ? " starred" : "");
    const mv = x => x > 1 ? `<b class="mv">${x}</b>` : `<b>${x}</b>`;
    c.innerHTML = `<div class="ab"><video muted loop playsinline preload="none" poster="${p.poster}" data-src="${p.src}"></video><span class="a">A</span><span class="b">B</span></div>
      <div class="row"><span class="lab">${p.label}</span><span class="num m">motion A ${mv(p.motion[0])} · B ${mv(p.motion[1])} · A–B DINO <b>${p.dino}</b></span></div>
      <div class="row"><span class="m">${p.id}</span><button>★</button></div>`;
    c.querySelector("button").onclick = () => { if (star[sk]) delete star[sk]; else star[sk] = 1; save(); c.classList.toggle("starred", !!star[sk]); document.getElementById("cnt").textContent = `${ps.length} pairs · ${Object.keys(star).length} starred`; };
    g.appendChild(c); io.observe(c.querySelector("video"));
  });
}
["sort", "both"].forEach(id => document.getElementById(id).onchange = render);
document.getElementById("q").oninput = render;
document.getElementById("export").onclick = () => { const o = document.getElementById("out"); o.style.display = "block"; o.value = Object.keys(star).join(" "); o.select(); try { navigator.clipboard.writeText(o.value); } catch (e) {} };
render();
</script></body></html>
"""


def main():
    P = json.load(open(RT / "endpoint_pairs.json"))
    pairs = [{**p, "src": f"media/{p['file']}.mp4", "poster": f"media/{p['file']}.jpg"} for p in P]
    d = {"sources": SOURCES, "pairs": pairs, "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(d, separators=(",", ":"))))
    print(f"wrote {OUT.relative_to(REPO)}:", {s: sum(p["source"] == s for p in P) for s, _, _ in SOURCES})


if __name__ == "__main__":
    main()
