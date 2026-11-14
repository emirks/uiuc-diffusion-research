"""Side-by-side examples of real-transition datasets considered for a real-data TEG tier (ICLR 2027 rebuttal, 2026-11-13).

Reads the hand picks in misc/2026-11-02_rebuttal/real_transitions/picks.json (+ edit3k/picked.json,
clipshots/windows.json, ../autotransition/cands.json) and writes outputs/viewers/real_transition_sources/index.html.
Media = ./media -> misc/2026-11-02_rebuttal (viewerctl new --media). Videos load only while on screen.

    /usr/bin/python3.12 scripts/viewers/build_real_transition_sources.py
"""
import json, pathlib, datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
R = REPO / "misc/2026-11-02_rebuttal"
OUT = REPO / "outputs/viewers/real_transition_sources/index.html"

SOURCES = [
    ("edit3k", "Edit3K", "CapCut transition presets rendered by the Edit3K authors on random ImageNet-VID video pairs "
     "(Gu et al. 2024; HF Gstar666/Edit3K, gated, our token has access; 1.74 TB, any file fetchable alone). "
     "204 transition types x 200 renders; every clip 4 s = 1 s A, 2 s transition, 1 s B; 1920x1080 30 fps. "
     "Shown: 17 hand-picked types, one video pair each."),
    ("clipshots", "ClipShots", "Real edited YouTube / Weibo videos (Tang et al. 2018; MIT; 48 GB gzip on Drive, stream-only). "
     "Gradual transitions annotated by frame range but NOT by type (mostly dissolves, also fades, blinds, mosaics). "
     "Shown: 16 hand-picked 5 s windows from the first 1.5 GB (160 videos, 75 clean windows); type labels are by eye."),
    ("moviecuts", "MovieCuts (endpoint source)", "Real movie scenes from 1,958 films (Pardo et al. ECCV 2022; access by request, granted "
     "2026-11-13; 61 GB zip, any clip fetchable alone). Every transition is a HARD CUT; labels are editorial reasons for the cut. "
     "Candidate use: the two shots around a cut as real, unseen-source TEG endpoints (no ground truth). Native 455x256 letterboxed "
     "with a MOVIECLIPS watermark (shown upscaled to 640x360). Shown: 3 single-label val clips per cut type, 5 s around the cut "
     "(the thin band marks the cut)."),
    ("autotransition", "AutoTransition", "CapCut user templates (Shen et al. 2022; CC-BY-4.0; 53 GB plain tar). "
     "107 timestamped transition types; noisy (text, stickers, slideshow content). Shown cropped to 3:4 480x640, "
     "8 best of 131 cut candidates."),
]


def items():
    picks = json.load(open(R / "real_transitions/picks.json"))
    out = {k: [] for k, _, _ in SOURCES}
    for p in json.load(open(R / "real_transitions/edit3k/picked.json")):
        b = p["file"][:-4]
        out["edit3k"].append({"label": p["name"], "sub": f"type {p['id']} · pair {b[:7]}",
                              "src": f"media/real_transitions/edit3k/pwin/{b}.mp4",
                              "poster": f"media/real_transitions/edit3k/pwin/{b}.jpg", "dur": 4.07, "tr": [1.0, 3.0]})
    W = json.load(open(R / "real_transitions/clipshots/windows.json"))
    for i, name in picks["clipshots"].items():
        w = W[int(i)]
        out["clipshots"].append({"label": name, "sub": f"{w['video']} · {w['src_wh'][0]}x{w['src_wh'][1]}",
                                 "src": f"media/real_transitions/clipshots/win/{w['id']}.mp4",
                                 "poster": f"media/real_transitions/clipshots/win/{w['id']}.jpg",
                                 "dur": 121 / 24, "tr": w["tr_in_win_s"]})
    for x in json.load(open(R / "real_transitions/moviecuts/examples.json")):
        c = x["cut_in_win"]
        out["moviecuts"].append({"label": x["type"], "sub": x["clip_id"],
                                 "src": f"media/real_transitions/moviecuts/win/{x['clip_id']}.mp4",
                                 "poster": f"media/real_transitions/moviecuts/win/{x['clip_id']}.jpg",
                                 "dur": 121 / 24, "tr": [c - 0.04, c + 0.04]})
    C = {c["cid"]: c for c in json.load(open(R / "autotransition/cands.json"))["candidates"]}
    for cid in picks["autotransition"]:
        c = C[cid]
        out["autotransition"].append({"label": c["type"], "sub": f"{cid} · tmpl {c['template']}",
                                      "src": f"media/autotransition/cands/{cid}.mp4",
                                      "poster": f"media/autotransition/cands/{cid}.jpg",
                                      "dur": 121 / 24, "tr": [x / 1000 for x in c["tr_in_win_ms"]], "tall": True})
    return out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Real transition sources</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1f;--mute:#66666b;--line:#dedede;--band:#9db9ec}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0}nav a{margin-right:10px;color:var(--ink)}select{font:inherit}
main{padding:4px 16px 60px;max-width:1500px}section{margin-top:22px}h2{font-size:17px;margin:0 0 4px}
.about{color:var(--mute);margin:0 0 10px;max-width:1100px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.grid.tall{grid-template-columns:repeat(auto-fill,minmax(190px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px}
video{width:100%;display:block;background:#000;border-radius:3px;aspect-ratio:16/9;object-fit:contain}
.tall video{aspect-ratio:3/4}
.tl{position:relative;height:6px;background:#e9e9e9;border-radius:2px;margin:5px 0 3px}
.tl .tr{position:absolute;top:0;bottom:0;min-width:3px;background:var(--band)}.tl .ph{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--ink)}
.lab{font-weight:600}.sub{color:var(--mute);font-size:12px;word-break:break-all}
.also{color:var(--mute);margin-top:28px}
</style></head><body>
<header><h1>Real transition sources</h1><nav id="nav"></nav>
<label>speed <select id="speed"><option>1</option><option>0.5</option><option>0.25</option></select></label>
<span class="sub">blue band = the transition · videos load only while on screen</span></header>
<main id="main"></main>
<script>
const D = __DATA__;
const main = document.getElementById("main"), nav = document.getElementById("nav");
const speed = () => +document.getElementById("speed").value;
D.sources.forEach(([key, name, about]) => {
  const its = D.items[key] || [];
  nav.insertAdjacentHTML("beforeend", `<a href="#${key}">${name} (${its.length})</a>`);
  const s = document.createElement("section"); s.id = key;
  s.innerHTML = `<h2>${name}</h2><p class="about">${about}</p>`;
  const g = document.createElement("div"); g.className = "grid" + (its.some(i => i.tall) ? " tall" : "");
  its.forEach(it => {
    const c = document.createElement("div"); c.className = "card";
    const l = it.tr[0] / it.dur * 100, w = (it.tr[1] - it.tr[0]) / it.dur * 100;
    c.innerHTML = `<video muted loop playsinline preload="none" poster="${it.poster}" data-src="${it.src}"></video>
      <div class="tl"><div class="tr" style="left:${l}%;width:${w}%"></div><div class="ph"></div></div>
      <div class="lab">${it.label}</div><div class="sub">${it.sub}</div>`;
    const v = c.querySelector("video"), ph = c.querySelector(".ph");
    v.addEventListener("timeupdate", () => { ph.style.left = Math.min(100, v.currentTime / it.dur * 100) + "%"; });
    g.appendChild(c);
  });
  s.appendChild(g); main.appendChild(s);
});
main.insertAdjacentHTML("beforeend", `<p class="also"><b>Also checked, no usable samples:</b>
  OmniShotCutBench (real, typed dissolve/wipe/push/zoom/fade; not released yet).</p>`);

// Only on-screen videos hold a source: load + play when visible, pause + drop the source (frees the decoder) when not.
const io = new IntersectionObserver(es => es.forEach(e => {
  const v = e.target;
  if (e.isIntersecting) {
    if (!v.getAttribute("src")) v.src = v.dataset.src;
    v.playbackRate = speed(); v.play().catch(() => {});
  } else if (v.getAttribute("src")) {
    v.pause(); v.removeAttribute("src"); v.load();
  }
}), {rootMargin: "150px 0px"});
document.querySelectorAll("video").forEach(v => io.observe(v));
document.getElementById("speed").onchange = () => document.querySelectorAll("video[src]").forEach(v => v.playbackRate = speed());
</script></body></html>
"""


def main():
    d = {"sources": SOURCES, "items": items(), "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(d, separators=(",", ":"))))
    print(f"wrote {OUT.relative_to(REPO)}:", {k: len(v) for k, v in d["items"].items()})


if __name__ == "__main__":
    main()
