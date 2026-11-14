"""Real-transition TEG tier — the eval set (ICLR 2027 rebuttal, 2026-11-14): 220 rows = 110 Edit3K transition classes x 2
references, each paired with a MovieCuts target endpoint pair (shot before / after a real hard cut).

Per row: the REFERENCE (Edit3K render, model format 480x640, 97 f) | the TARGET endpoints (A | B, 1 s each, portrait crop as
the model sees them; conditioning = the 9 frames nearest the cut on each side) | stats: transition class (distance from
dissolve, extra motion of the transition itself, new content — measured endpoint-free), endpoint motion A/B, A-B DINO
(semantic gap), ref->target style gap, grid cells. Sortable / filterable. Videos load only while on screen.

Reads misc/2026-11-02_rebuttal/real_transitions/tier/items.json (select_tier.py) and writes
outputs/viewers/real_tier_eval_set/index.html; ./media -> misc/2026-11-02_rebuttal/real_transitions.

    /usr/bin/python3.12 scripts/viewers/build_real_tier_eval_set.py
"""
import json, pathlib, datetime, subprocess, os

REPO = pathlib.Path(__file__).resolve().parents[2]
RT = REPO / "misc/2026-11-02_rebuttal/real_transitions"
OUT = REPO / "outputs/viewers/real_tier_eval_set/index.html"
FF = os.environ.get("LAB", "/taiga/illinois/eng/cs/jrehg/users/emirkisa") + "/bin/ffmpeg"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Real-transition eval set</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1f;--mute:#66666b;--line:#dedede;--acc:#2f6fdb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mute);font-size:13px;max-width:1300px}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:8px}
select,input,button{font:inherit;padding:2px 8px;border:1px solid var(--line);border-radius:4px;background:#fff}
main{padding:8px 16px 60px}
.row{display:grid;grid-template-columns:150px 330px 1fr;gap:12px;align-items:start;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;margin-bottom:8px}
.vh{font-size:11px;color:var(--mute);margin-bottom:2px}
video{display:block;width:100%;background:#111;border-radius:3px}.ref video{aspect-ratio:3/4}.tgt video{aspect-ratio:488/320}
.st h3{margin:0 0 4px;font-size:15px}.st table{border-collapse:collapse;font-size:12.5px}.st td{padding:1px 10px 1px 0;vertical-align:top}
.st td:first-child{color:var(--mute)}.chip{display:inline-block;font-size:11px;padding:0 6px;border-radius:3px;background:#eef2fb;color:#2f4f8f;margin-right:4px}
.hide{display:none}
@media (max-width:760px){.row{grid-template-columns:1fr 2fr}.st{grid-column:1/-1}}
</style></head><body>
<header><h1>Real-transition TEG tier — eval set</h1>
<div class="sub" id="sub"></div>
<div class="bar">
<label>sort <select id="sort">
 <option value="item">item</option><option value="d_dissolve">transition: distance from dissolve ↓</option>
 <option value="excess_motion">transition: its own motion ↓</option><option value="new_content">transition: new content ↓</option>
 <option value="mlo">endpoints: motion (weaker side) ↓</option><option value="ab">endpoints: A–B DINO ↑ (most different first)</option>
 <option value="style">ref→target style gap ↓</option></select></label>
<label>transition <select id="ftb"><option value="">all</option><option>dissolve-like</option><option>mid</option><option>far from dissolve</option></select></label>
<label>motion <select id="fmb"><option value="">all</option><option>low</option><option>mid</option><option>high</option></select></label>
<label>A–B gap <select id="fgb"><option value="">all</option><option>similar</option><option>mid</option><option>different</option></select></label>
<input type="search" id="q" placeholder="class / cut type / item"><span id="cnt" class="sub"></span></div></header>
<main id="main"></main>
<script>
const D = __DATA__;
document.getElementById("sub").textContent = `${D.items.length} rows = ${D.n_classes} Edit3K transition classes x 2 references, each paired with a different MovieCuts target endpoint pair ` +
  `(the shot before and after a real hard cut; ${D.n_films} films). Targets are both-moving and spread evenly over a 3x3 grid of weaker-side motion (cuts ${D.mcut}) x A–B DINO ` +
  `(cuts ${D.dcut}); each transition tercile gets every grid cell equally. Generation: 97 frames, 24 fps, 480x640, conditioning = the 9 frames nearest the cut on each side. Built ${D.built}.`;
const main = document.getElementById("main"), f2 = x => (+x).toFixed(2);
const io = new IntersectionObserver(es => es.forEach(e => {
  const v = e.target;
  if (e.isIntersecting) { if (!v.getAttribute("src")) v.src = v.dataset.src; v.play().catch(() => {}); }
  else if (v.getAttribute("src")) { v.pause(); v.removeAttribute("src"); v.load(); }
}), {rootMargin: "200px 0px"});
function render() {
  main.querySelectorAll("video[src]").forEach(v => { v.pause(); v.removeAttribute("src"); v.load(); });
  io.disconnect(); main.innerHTML = "";
  const k = document.getElementById("sort").value, q = document.getElementById("q").value.trim().toLowerCase();
  const ft = document.getElementById("ftb").value, fm = document.getElementById("fmb").value, fg = document.getElementById("fgb").value;
  const key = {item: r => r.item, d_dissolve: r => -r.d_dissolve, excess_motion: r => -r.excess_motion, new_content: r => -r.new_content,
               mlo: r => -Math.min(...r.motion), ab: r => r.ab_dino, style: r => -r.style_gap}[k];
  const rs = D.items.filter(r => (!ft || r.transition_bin === ft) && (!fm || r.motion_bin === fm) && (!fg || r.gap_bin === fg) &&
    (!q || `${r.item} ${r.class} ${r.class_name} ${r.cut_type}`.toLowerCase().includes(q)))
    .sort((a, b) => { const x = key(a), y = key(b); return x < y ? -1 : x > y ? 1 : 0; });
  document.getElementById("cnt").textContent = `${rs.length} rows`;
  rs.forEach(r => {
    const d = document.createElement("div"); d.className = "row";
    d.innerHTML = `<div class="ref"><div class="vh">reference (Edit3K)</div><video muted loop playsinline preload="none" poster="${r.ref_poster}" data-src="${r.ref_src}"></video></div>
      <div class="tgt"><div class="vh">target endpoints: A (before cut) | B (after cut)</div><video muted loop playsinline preload="none" poster="${r.strip_poster}" data-src="${r.strip_src}"></video></div>
      <div class="st"><h3>${r.item} · class ${r.class}${r.class_name ? " (" + r.class_name + ")" : ""}</h3>
       <div><span class="chip">${r.transition_bin}</span><span class="chip">motion ${r.motion_bin}</span><span class="chip">gap ${r.gap_bin}</span></div>
       <table>
        <tr><td>transition</td><td>dist. from dissolve <b>${f2(r.d_dissolve)}</b> · own motion <b>${f2(r.excess_motion)}</b> · new content <b>${f2(r.new_content)}</b></td></tr>
        <tr><td>endpoints</td><td>motion A <b>${r.motion[0]}</b> · B <b>${r.motion[1]}</b> · A–B DINO <b>${f2(r.ab_dino)}</b></td></tr>
        <tr><td>ref→target</td><td>style gap <b>${f2(r.style_gap)}</b></td></tr>
        <tr><td>target</td><td>MovieCuts ${r.target} · ${r.cut_type} · film ${r.movie_id}</td></tr>
        <tr><td>reference</td><td>Edit3K pair ${r.ref_pair}</td></tr></table></div>`;
    main.appendChild(d); d.querySelectorAll("video").forEach(v => io.observe(v));
  });
}
["sort", "ftb", "fmb", "fgb"].forEach(id => document.getElementById(id).onchange = render);
document.getElementById("q").oninput = render;
render();
</script></body></html>
"""


def main():
    T = json.load(open(RT / "tier/items.json")); items = T["items"]
    for it in items:   # reference posters: the mid-transition frame (48 of 97)
        jpg = RT / (it["ref"][:-4] + ".jpg")
        if not jpg.exists():
            subprocess.run([FF, "-v", "error", "-threads", "1", "-y", "-i", str(RT / it["ref"]), "-vf", "select='eq(n,48)',scale=240:-2",
                            "-frames:v", "1", "-q:v", "5", str(jpg)], check=True)
        it.update({"ref_src": f"media/{it['ref']}", "ref_poster": f"media/{it['ref'][:-4]}.jpg",
                   "strip_src": f"media/{it['strip']}.mp4", "strip_poster": f"media/{it['strip']}.jpg"})
    d = {"items": items, "n_classes": len({i["class"] for i in items}), "n_films": len({i["movie_id"] for i in items}),
         "mcut": " / ".join(f"{x:.2f}" for x in T["motion_cuts"]), "dcut": " / ".join(f"{x:.2f}" for x in T["dino_cuts"]),
         "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(d, separators=(",", ":"))))
    print(f"wrote {OUT.relative_to(REPO)}: {len(items)} rows")


if __name__ == "__main__":
    main()
