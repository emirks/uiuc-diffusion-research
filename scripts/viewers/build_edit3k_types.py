"""Edit3K overview: 6 random components of each of its 6 component types (ICLR 2027 rebuttal, 2026-11-13).

Reads misc/2026-11-02_rebuttal/real_transitions/edit3k/types.json (written by fetch_type_samples.py there) and writes
outputs/viewers/edit3k_types/index.html; ./media -> that edit3k/types dir. Videos load only while on screen.

    /usr/bin/python3.12 scripts/viewers/build_edit3k_types.py
"""
import json, pathlib, datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
E3K = REPO / "misc/2026-11-02_rebuttal/real_transitions/edit3k"
OUT = REPO / "outputs/viewers/edit3k_types/index.html"
TYPES = [("transition", "Transitions", "237 classes listed (209 with videos)"),
         ("video_effect", "Video effects", "903 classes"), ("video_animation", "Animations", "176 classes"),
         ("filter", "Filters", "228 classes"), ("sticker", "Stickers", "1,000 classes"), ("texts", "Text", "599 classes")]

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edit3K component types</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1f;--mute:#66666b;--line:#dedede}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mute);font-size:13px;max-width:1200px}nav a{margin-right:10px;color:var(--ink)}
main{padding:4px 16px 60px}section{margin-top:20px}h2{font-size:17px;margin:0 0 6px}h2 span{font-weight:400;color:var(--mute);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px}
video{width:100%;aspect-ratio:16/9;display:block;background:#111;border-radius:3px}.m{color:var(--mute);font-size:12px;margin-top:3px}
</style></head><body>
<header><h1>Edit3K — the 6 component types</h1>
<div class="sub" id="sub"></div><nav id="nav"></nav></header>
<main id="main"></main>
<script>
const D = __DATA__;
document.getElementById("sub").textContent = `Edit3K = 3,094 CapCut editing components of 6 types, 618,800 renders (1.74 TB). ` +
  `6 random components per type, each on a different moving ImageNet-VID video pair; every clip 4 s (transitions: 1 s A, 2 s transition, 1 s B). Built ${D.built}.`;
const main = document.getElementById("main"), nav = document.getElementById("nav");
D.types.forEach(([key, name, note]) => {
  const its = D.items.filter(i => i.type === key);
  nav.insertAdjacentHTML("beforeend", `<a href="#${key}">${name}</a>`);
  const s = document.createElement("section"); s.id = key;
  s.innerHTML = `<h2>${name} <span>${note}</span></h2><div class="grid">` + its.map(i =>
    `<div class="card"><video muted loop playsinline preload="none" poster="${i.poster}" data-src="${i.src}"></video>` +
    `<div class="m">component ${i.id} · pair ${i.pair}</div></div>`).join("") + `</div>`;
  main.appendChild(s);
});
// Only on-screen videos hold a source: load + play when visible, pause + drop the source (frees the decoder) when not.
const io = new IntersectionObserver(es => es.forEach(e => {
  const v = e.target;
  if (e.isIntersecting) { if (!v.getAttribute("src")) v.src = v.dataset.src; v.play().catch(() => {}); }
  else if (v.getAttribute("src")) { v.pause(); v.removeAttribute("src"); v.load(); }
}), {rootMargin: "150px 0px"});
document.querySelectorAll("video").forEach(v => io.observe(v));
</script></body></html>
"""


def main():
    items = [{"type": x["type"], "id": x["id"], "pair": x["pair"], "src": f"media/{x['file']}",
              "poster": f"media/{x['file'][:-4]}.jpg"} for x in json.load(open(E3K / "types.json"))]
    d = {"types": TYPES, "items": items, "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(d, separators=(",", ":"))))
    print(f"wrote {OUT.relative_to(REPO)}: {len(items)} clips")


if __name__ == "__main__":
    main()
