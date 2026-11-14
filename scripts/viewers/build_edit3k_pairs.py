"""Edit3K endpoint pairs: the 100 ImageNet-VID video pairs (materials 500-699), each shown as Edit3K rendered it with a
plain dissolve (class 1003716): 1 s clip A, 2 s dissolve, 1 s clip B (ICLR 2027 rebuttal, 2026-11-13).

Reads misc/2026-11-02_rebuttal/real_transitions/edit3k/pairs.json (+ pairs/<pair>.{mp4,jpg}) and writes
outputs/viewers/edit3k_pairs/index.html; ./media -> that pairs dir. Motion = mean |frame difference| (64x36 gray, 0-255)
inside slot A and slot B; "both moving" = both > 1.0. Stars persist in the browser and export as a list.

    /usr/bin/python3.12 scripts/viewers/build_edit3k_pairs.py
"""
import json, pathlib, datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
E3K = REPO / "misc/2026-11-02_rebuttal/real_transitions/edit3k"
OUT = REPO / "outputs/viewers/edit3k_pairs/index.html"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edit3K endpoint pairs</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1f;--mute:#66666b;--line:#dedede;--ok:#1f9d55;--mid:#c77c02;--off:#9a9a9f;--star:#d99a00}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mute);font-size:13px;max-width:1200px}
.bar{display:flex;gap:12px;align-items:center;margin-top:6px;flex-wrap:wrap}button,select{font:inherit;padding:2px 8px;border:1px solid var(--line);border-radius:4px;background:#fff;cursor:pointer}
textarea{width:100%;height:50px;font:12px monospace;display:none;margin-top:6px}
main{padding:8px 16px 60px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px}.card.starred{border:2px solid var(--star)}
video{width:100%;aspect-ratio:16/9;display:block;background:#111;border-radius:3px}
.row{display:flex;justify-content:space-between;align-items:center;margin-top:4px}.p{font-weight:600}
.tag{font-size:12px;padding:0 6px;border-radius:3px;color:#fff}.both{background:var(--ok)}.one{background:var(--mid)}.none{background:var(--off)}
.m{color:var(--mute);font-size:12px}.hide{display:none}
</style></head><body>
<header><h1>Edit3K endpoint pairs (ImageNet-VID)</h1>
<div class="sub" id="sub"></div>
<div class="bar"><label>show <select id="f"><option value="all">all</option><option value="both">both moving</option><option value="star">starred</option></select></label>
<span id="cnt" class="sub"></span><button id="export">copy starred</button></div><textarea id="out" readonly></textarea></header>
<main><div class="grid" id="grid"></div></main>
<script>
const D = __DATA__, KEY = "edit3k_pairs_star_v1";
let star = {}; try { star = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(star)); } catch (e) {} };
document.getElementById("sub").textContent = `The ${D.pairs.length} video endpoint pairs every Edit3K class is rendered on (materials 500-699 = ImageNet-VID clips, picked at random by Edit3K). ` +
  `Each card = that pair as Edit3K rendered it with a plain dissolve: 1 s clip A, 2 s dissolve, 1 s clip B. Motion = mean pixel change per frame inside A / inside B (0-255); ` +
  `"both moving" = both > 1.0, stills score ~0.01. Sorted by the weaker side. Star the pairs you like as endpoints. Built ${D.built}.`;
const g = document.getElementById("grid");
D.pairs.forEach(p => {
  const lo = Math.min(...p.motion), cls = lo > 1 ? "both" : (Math.max(...p.motion) > 0.3 ? "one" : "none");
  const c = document.createElement("div"); c.className = "card"; c.dataset.pair = p.pair; c.dataset.cls = cls;
  c.innerHTML = `<video muted loop playsinline preload="none" poster="${p.poster}" data-src="${p.src}"></video>
    <div class="row"><span class="p">${p.pair}</span><span class="tag ${cls}">${cls === "both" ? "both moving" : cls === "one" ? "one side moving" : "static"}</span></div>
    <div class="row"><span class="m">motion A ${p.motion[0]} · B ${p.motion[1]}</span><button>★</button></div>`;
  c.querySelector("button").onclick = () => { if (star[p.pair]) delete star[p.pair]; else star[p.pair] = 1; save(); paint(); };
  g.appendChild(c);
});
function paint() {
  const f = document.getElementById("f").value; let n = 0;
  document.querySelectorAll(".card").forEach(c => {
    c.classList.toggle("starred", !!star[c.dataset.pair]);
    const show = f === "all" || (f === "both" && c.dataset.cls === "both") || (f === "star" && star[c.dataset.pair]);
    c.classList.toggle("hide", !show); if (show) n++;
  });
  document.getElementById("cnt").textContent = `${n} shown · ${Object.keys(star).length} starred`;
}
document.getElementById("f").onchange = paint;
document.getElementById("export").onclick = () => { const o = document.getElementById("out"); o.style.display = "block"; o.value = Object.keys(star).join(" "); o.select(); try { navigator.clipboard.writeText(o.value); } catch (e) {} };
// Only on-screen videos hold a source: load + play when visible, pause + drop the source (frees the decoder) when not.
const io = new IntersectionObserver(es => es.forEach(e => {
  const v = e.target;
  if (e.isIntersecting) { if (!v.getAttribute("src")) v.src = v.dataset.src; v.play().catch(() => {}); }
  else if (v.getAttribute("src")) { v.pause(); v.removeAttribute("src"); v.load(); }
}), {rootMargin: "150px 0px"});
document.querySelectorAll("video").forEach(v => io.observe(v));
paint();
</script></body></html>
"""


def main():
    P = json.load(open(E3K / "pairs.json"))
    P.sort(key=lambda p: -min(p["motion"]))
    pairs = [{**p, "src": f"media/{p['pair']}.mp4", "poster": f"media/{p['pair']}.jpg"} for p in P]
    d = {"pairs": pairs, "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(d, separators=(",", ":"))))
    print(f"wrote {OUT.relative_to(REPO)}: {len(pairs)} pairs")


if __name__ == "__main__":
    main()
