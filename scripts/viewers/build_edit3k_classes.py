"""Edit3K transition classes, 5 videos each (ICLR 2027 rebuttal, real-transition tier, 2026-11-13).

Every class = one fixed CapCut transition preset; each row shows it on the SAME 5 ImageNet-VID material pairs, so rows
compare styles on identical content. Reads misc/2026-11-02_rebuttal/real_transitions/edit3k/classes/ (written by
build_class_gallery.py there) and writes outputs/viewers/edit3k_classes/index.html; ./media -> that classes/ dir.
Stars (per class) persist in the browser and export as a list.

    /usr/bin/python3.12 scripts/viewers/build_edit3k_classes.py
"""
import json, pathlib, datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
E3K = REPO / "misc/2026-11-02_rebuttal/real_transitions/edit3k"
OUT = REPO / "outputs/viewers/edit3k_classes/index.html"
PAIRS = ["698_699", "670_671", "572_573", "654_655", "688_689"]

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edit3K transition classes</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1f;--mute:#66666b;--line:#dedede;--star:#d99a00}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mute);font-size:13px;max-width:1200px}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:8px}
input[type=search],select,button{font:inherit;padding:3px 8px;border:1px solid var(--line);border-radius:4px;background:#fff}
button{cursor:pointer}textarea{width:100%;height:60px;font:12px monospace;margin-top:6px;display:none}
main{padding:6px 16px 60px}
.row{display:grid;grid-template-columns:110px repeat(5,minmax(0,1fr));gap:8px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px;margin-bottom:8px}
.row.starred{border-color:var(--star);box-shadow:inset 3px 0 0 var(--star)}
.lab .cid{font-weight:600;font-size:15px}.lab .nm{color:var(--mute);font-size:12px}
.lab button{margin-top:4px;padding:1px 8px}.row.starred .lab button{background:var(--star);color:#fff;border-color:var(--star)}
video{width:100%;aspect-ratio:16/9;display:block;background:#111;border-radius:3px}
.hdr{display:grid;grid-template-columns:110px repeat(5,minmax(0,1fr));gap:8px;color:var(--mute);font-size:12px;padding:0 7px 4px}
.hide{display:none}
@media (max-width:700px){.row,.hdr{grid-template-columns:repeat(2,minmax(0,1fr))}.lab{grid-column:1/-1}}
</style></head><body>
<header>
 <h1>Edit3K — transition classes</h1>
 <div class="sub" id="sub"></div>
 <div class="bar">
  <input type="search" id="q" placeholder="filter class id / name">
  <label><input type="checkbox" id="onlystar"> starred only</label>
  <label>speed <select id="speed"><option>1</option><option>0.5</option><option>0.25</option></select></label>
  <span id="cnt" class="sub"></span>
  <button id="export">copy starred</button>
 </div>
 <textarea id="out" readonly></textarea>
</header>
<main id="main"></main>
<script>
const D = __DATA__;
const KEY = "edit3k_classes_star_v1";
let star = {};
try { star = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(star)); } catch (e) {} };
document.getElementById("sub").textContent =
  `${D.classes.length} transition classes. Each class is ONE fixed CapCut preset (no parameters in the dataset); its 200 renders differ only in the material pair. ` +
  `Every row shows the class on the same 5 ImageNet-VID video pairs (columns), so rows compare styles on identical content. ` +
  `Clip = 1 s slot A, 2 s transition, 1 s slot B (4 s, 640x360, 24 fps). Names are by eye for the 17 earlier picks; others are ids only. Built ${D.built}.`;
const main = document.getElementById("main");
main.insertAdjacentHTML("beforeend", `<div class="hdr"><div>class</div>${D.pairs.map(p => `<div>pair ${p}</div>`).join("")}</div>`);
D.classes.forEach(c => {
  const r = document.createElement("div"); r.className = "row"; r.dataset.cid = c.id;
  r.dataset.q = (c.id + " " + (c.name || "")).toLowerCase();
  r.innerHTML = `<div class="lab"><div class="cid">${c.id}</div><div class="nm">${c.name || ""}</div><button>★ star</button></div>` +
    c.clips.map(k => `<video muted loop playsinline preload="none" data-poster="${k.poster}" data-src="${k.src}"></video>`).join("");
  r.querySelector("button").onclick = () => { if (star[c.id]) delete star[c.id]; else star[c.id] = 1; save(); paint(); };
  main.appendChild(r);
});
function paint() {
  document.querySelectorAll(".row").forEach(r => r.classList.toggle("starred", !!star[r.dataset.cid]));
  filter();
}
function filter() {
  const q = document.getElementById("q").value.trim().toLowerCase(), only = document.getElementById("onlystar").checked;
  let n = 0;
  document.querySelectorAll(".row").forEach(r => {
    const show = (!q || r.dataset.q.includes(q)) && (!only || star[r.dataset.cid]);
    r.classList.toggle("hide", !show); if (show) n++;
  });
  document.getElementById("cnt").textContent = `${n} shown · ${Object.keys(star).length} starred`;
}
document.getElementById("q").oninput = filter;
document.getElementById("onlystar").onchange = filter;
document.getElementById("export").onclick = () => {
  const o = document.getElementById("out"); o.style.display = "block";
  o.value = Object.keys(star).join(" "); o.select();
  try { navigator.clipboard.writeText(o.value); } catch (e) {}
};
// Posters: set once when a row comes near. Videos: hold a source only while on screen (load+play / pause+drop).
const speed = () => +document.getElementById("speed").value;
const pio = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) { e.target.poster = e.target.dataset.poster; pio.unobserve(e.target); }
}), {rootMargin: "1500px 0px"});
const vio = new IntersectionObserver(es => es.forEach(e => {
  const v = e.target;
  if (e.isIntersecting) {
    if (!v.getAttribute("src")) v.src = v.dataset.src;
    v.playbackRate = speed(); v.play().catch(() => {});
  } else if (v.getAttribute("src")) {
    v.pause(); v.removeAttribute("src"); v.load();
  }
}), {rootMargin: "100px 0px"});
document.querySelectorAll("video").forEach(v => { pio.observe(v); vio.observe(v); });
document.getElementById("speed").onchange = () => document.querySelectorAll("video[src]").forEach(v => v.playbackRate = speed());
paint();
</script></body></html>
"""


def main():
    meta = json.load(open(E3K / "classes.json")) if (E3K / "classes.json").exists() else None
    ids = meta["classes"] if meta else sorted((p.name for p in (E3K / "classes").iterdir() if p.is_dir()), key=int)
    names = {p["id"]: p["name"] for p in json.load(open(E3K / "picked.json"))}
    classes = []
    for cid in ids:
        clips = [{"src": f"media/{cid}/{p}.mp4", "poster": f"media/{cid}/{p}.jpg"} for p in PAIRS
                 if (E3K / "classes" / cid / f"{p}.mp4").exists()]
        if clips:
            classes.append({"id": cid, "name": names.get(cid, ""), "clips": clips})
    d = {"pairs": PAIRS, "classes": classes, "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(d, separators=(",", ":"))))
    print(f"wrote {OUT.relative_to(REPO)}: {len(classes)} classes, {sum(len(c['clips']) for c in classes)} clips")


if __name__ == "__main__":
    main()
