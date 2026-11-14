"""ImageNet-VID samples (the footage behind Edit3K's video endpoint pairs), via ImageNet-VidVRD (Shang et al. 2017, 1,000
ILSVRC-VID videos as mp4). 2 videos per object category, shown whole (first 12 s) at <=640 px (ICLR 2027 rebuttal, 2026-11-13).

Reads misc/2026-11-02_rebuttal/real_transitions/imagenet_vid/samples.json (written by fetch_samples.py there) and writes
outputs/viewers/imagenet_vid_samples/index.html; ./media -> that clips dir. Videos load only while on screen.

    /usr/bin/python3.12 scripts/viewers/build_imagenet_vid_samples.py
"""
import json, pathlib, datetime, collections

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "misc/2026-11-02_rebuttal/real_transitions/imagenet_vid"
OUT = REPO / "outputs/viewers/imagenet_vid_samples/index.html"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ImageNet-VID samples</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1f;--mute:#66666b;--line:#dedede}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mute);font-size:13px;max-width:1200px}
.toc{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}.toc a{font-size:12px;padding:0 6px;border:1px solid var(--line);border-radius:3px;background:#fff;color:var(--ink);text-decoration:none}
main{padding:4px 16px 60px}section{margin-top:16px}h2{font-size:16px;margin:0 0 6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px}
video{width:100%;aspect-ratio:16/9;object-fit:contain;display:block;background:#111;border-radius:3px}.m{color:var(--mute);font-size:12px;margin-top:3px}
</style></head><body>
<header><h1>ImageNet-VID — sample videos</h1><div class="sub" id="sub"></div><div class="toc" id="toc"></div></header>
<main id="main"></main>
<script>
const D = __DATA__;
document.getElementById("sub").textContent = `ImageNet-VID (ILSVRC 2015 video detection: YouTube snippets of 30 object categories, ~300 frames each) is the footage ` +
  `Edit3K used for its video endpoint pairs. Shown: ${D.n} videos, 2 per category, from ImageNet-VidVRD (1,000 VID videos as mp4), first 12 s at <=640 px; ` +
  `category = the main annotated object. Native size and length under each clip. Built ${D.built}.`;
const main = document.getElementById("main"), toc = document.getElementById("toc");
Object.entries(D.cats).forEach(([cat, its]) => {
  toc.insertAdjacentHTML("beforeend", `<a href="#c-${cat}">${cat}</a>`);
  const s = document.createElement("section"); s.id = "c-" + cat;
  s.innerHTML = `<h2>${cat}</h2><div class="grid">` + its.map(i =>
    `<div class="card"><video muted loop playsinline preload="none" poster="${i.poster}" data-src="${i.src}"></video>` +
    `<div class="m">${i.vid} · ${i.src_wh[0]}x${i.src_wh[1]} · ${i.frames} frames · objects: ${i.objects.join(", ")}</div></div>`).join("") + `</div>`;
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
    S = json.load(open(SRC / "samples.json"))
    cats = collections.OrderedDict()
    for x in S:
        cats.setdefault(x["category"], []).append({**x, "src": f"media/{x['vid']}.mp4", "poster": f"media/{x['vid']}.jpg"})
    d = {"cats": cats, "n": len(S), "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(d, separators=(",", ":"))))
    print(f"wrote {OUT.relative_to(REPO)}: {len(S)} clips, {len(cats)} categories")


if __name__ == "__main__":
    main()
