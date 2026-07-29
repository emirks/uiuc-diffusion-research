#!/usr/bin/env python3
"""Build the S1 labelling viewer — grouped by class, autoplaying, click to reject.

S1 is the one stratum with no per-clip quality gate: a 33-clip mechanical pilot passed
(1/33 = 3.0 % rejects) and the blind 11-way batch gate was never run, so 1,384 of its 1,417
clips have never been looked at. This page exists so the owner can look at all of them
grouped by class and mark the bad ones, which is the input to a clip-removal pass.

Design decisions that matter:

* **Grouped by class (arm), not by layer.** The two two-sided arms carry a `__1sided`
  sibling group holding the 10 clips whose sidedness was DOWNGRADED by splice measurement
  (DOSSIER §A22.3) — those are rendered as their own labelled section, because they are the
  clips most likely to look wrong and the reader should know why they are separated.
* **`preload="none"` + IntersectionObserver.** 1,417 simultaneous `<video>` elements will
  hang a browser. Only tiles near the viewport get a `src` and play; tiles that scroll away
  are paused and released. This is the difference between usable and unusable at this scale.
* **Labels persist in `localStorage`** keyed by clip stem, so a reload or an accidental
  navigation does not lose an hour of labelling. Export writes a JSON the removal pass reads.
* **Every media path is relative to this viewer's own directory** via two symlinks
  (`media_foreign`, `media_s0cf`). That is the rule that keeps viewers alive across restarts;
  most historical viewer breakage is a violation of it.

    python3 scripts/build_s1_label_viewer.py
    python3 scripts/viewers/viewerctl.py serve
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INV = REPO / "outputs/ctt_v2/inventories/S1.json"
OUT = REPO / "outputs/viewers/s1_label"
LAYERS = {"foreign": "outputs/videos/ctt_v2_s1", "s0cf": "outputs/videos/ctt_v2_s1_s0cf"}


def main() -> None:
    inv = json.loads(INV.read_text())
    groups = inv["groups"]
    downgraded = {d.split(": ")[0].split("/")[1]
                  for d in inv["provenance"].get("per_clip_sidedness_downgrades", [])}

    # locate every clip: (stem -> (layer, arm_dir))
    where: dict[str, tuple[str, str]] = {}
    for layer, rel in LAYERS.items():
        for p in (REPO / rel).glob("spec_*/*.mp4"):
            where[p.stem] = (layer, p.parent.name)
    missing = [s for s in inv["clips"] if s not in where]
    if missing:
        raise SystemExit(f"{len(missing)} inventory clips have no mp4 on disk: {missing[:5]}")

    OUT.mkdir(parents=True, exist_ok=True)
    for layer, rel in LAYERS.items():
        link = OUT / f"media_{layer}"
        if link.is_symlink() or link.exists():
            link.unlink()
        # RELATIVE, from this viewer's own directory — never absolute, never ../../repo
        link.symlink_to(Path("../..") / Path(rel).relative_to("outputs"))

    # build the payload the page renders
    payload = []
    for gid, g in sorted(groups.items()):
        clips = []
        for stem in sorted(g["clips"]):
            layer, arm = where[stem]
            clips.append({"stem": stem, "src": f"media_{layer}/{arm}/{stem}.mp4",
                          "layer": layer, "dg": stem in downgraded})
        payload.append({"group": gid, "sided": g["sided"], "n": len(clips), "clips": clips})

    n_total = sum(b["n"] for b in payload)
    (OUT / "clips.json").write_text(json.dumps(payload))
    (OUT / "index.html").write_text(PAGE.replace("__N__", str(n_total))
                                        .replace("__NG__", str(len(payload))))
    print(f"[ok] {OUT.relative_to(REPO)}/index.html")
    print(f"     {n_total} clips in {len(payload)} groups; "
          f"{sum(1 for b in payload for c in b['clips'] if c['dg'])} sidedness-downgraded")
    print(f"     layers: {dict((l, sum(1 for b in payload for c in b['clips'] if c['layer']==l)) for l in LAYERS)}")


PAGE = r"""<!doctype html>
<meta charset="utf-8"><title>S1 — label bad generations</title>
<style>
:root{--bg:#0e0f12;--fg:#e8e8ea;--dim:#8b8d94;--bad:#e5484d;--ok:#30a46c;--line:#24262b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}
header{position:sticky;top:0;z-index:9;background:rgba(14,15,18,.96);backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.2px}
.stat{color:var(--dim);font-variant-numeric:tabular-nums}
.stat b{color:var(--fg)}
button{background:#1b1d22;color:var(--fg);border:1px solid var(--line);border-radius:6px;
  padding:6px 11px;font:inherit;cursor:pointer}
button:hover{border-color:#3a3d44}
button.primary{background:#1f3a2c;border-color:#2a5a41}
.hint{color:var(--dim);font-size:12.5px;padding:10px 16px;border-bottom:1px solid var(--line)}
kbd{background:#1b1d22;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11.5px}
section{padding:14px 16px 6px}
h2{font-size:13.5px;margin:0 0 2px;font-weight:600}
h2 span{color:var(--dim);font-weight:400}
.two{color:#f5a623}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px;margin-top:9px}
figure{margin:0;position:relative;border:2px solid transparent;border-radius:8px;overflow:hidden;
  background:#000;cursor:pointer;aspect-ratio:4/3}
figure video{width:100%;height:100%;object-fit:cover;display:block}
figure figcaption{position:absolute;left:0;right:0;bottom:0;padding:3px 6px;font-size:10.5px;
  color:#cfd1d6;background:linear-gradient(transparent,rgba(0,0,0,.85));
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
figure.bad{border-color:var(--bad)}
figure.bad::after{content:"REJECTED";position:absolute;top:6px;left:6px;background:var(--bad);
  color:#fff;font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;letter-spacing:.4px}
figure.bad video{opacity:.32}
figure .dg{position:absolute;top:6px;right:6px;background:#8b5cf6;color:#fff;font-size:9.5px;
  font-weight:700;padding:2px 5px;border-radius:4px}
</style>
<header>
  <h1>S1 — label bad generations</h1>
  <span class="stat"><b id="nbad">0</b> rejected / <b>__N__</b> clips · __NG__ groups</span>
  <button id="copy">copy reject JSON</button>
  <button id="dl" class="primary">download rejects.json</button>
  <button id="clear">clear all labels</button>
</header>
<div class="hint">
  <b>Click a clip to reject it</b> (click again to un-reject). Labels save automatically in this
  browser — reloading will not lose them. Only clips near the viewport load and play, so scrolling
  is smooth at 1,417 videos. <kbd>shift</kbd>+click rejects every clip from the last one you
  clicked through this one. Purple <b>1-sided</b> badge = sidedness downgraded by splice
  measurement, separated into its own group on purpose.
</div>
<main id="main"></main>
<script>
const KEY = 's1_label_rejects_v1';
let bad = new Set(JSON.parse(localStorage.getItem(KEY) || '[]'));
let lastIdx = null, order = [];

function save(){ localStorage.setItem(KEY, JSON.stringify([...bad]));
  document.getElementById('nbad').textContent = bad.size; }

fetch('clips.json').then(r => r.json()).then(blocks => {
  const main = document.getElementById('main');
  for (const b of blocks) {
    const sec = document.createElement('section');
    const two = b.sided === 'two';
    sec.innerHTML = `<h2>${b.group} <span>· ${b.n} clips · `
      + `<i class="${two?'two':''}">${b.sided}-sided</i></span></h2>`;
    const grid = document.createElement('div'); grid.className = 'grid';
    for (const c of b.clips) {
      const fig = document.createElement('figure');
      fig.dataset.stem = c.stem; fig.dataset.src = c.src;
      if (bad.has(c.stem)) fig.classList.add('bad');
      fig.innerHTML = `<video muted loop playsinline preload="none"></video>`
        + (c.dg ? `<span class="dg">1-sided</span>` : ``)
        + `<figcaption>${c.stem}</figcaption>`;
      grid.appendChild(fig); order.push(fig);
    }
    sec.appendChild(grid); main.appendChild(sec);
  }
  save();

  // only load + play what is visible; release what is not. 1,417 live <video> would hang.
  const io = new IntersectionObserver(es => {
    for (const e of es) {
      const v = e.target.querySelector('video');
      if (e.isIntersecting) {
        if (!v.src) v.src = e.target.dataset.src;
        v.play().catch(() => {});
      } else { v.pause(); }
    }
  }, {rootMargin: '250px 0px'});
  order.forEach(f => io.observe(f));

  main.addEventListener('click', ev => {
    const fig = ev.target.closest('figure'); if (!fig) return;
    const idx = order.indexOf(fig);
    if (ev.shiftKey && lastIdx !== null) {
      const [a, b2] = [Math.min(lastIdx, idx), Math.max(lastIdx, idx)];
      for (let i = a; i <= b2; i++) { bad.add(order[i].dataset.stem); order[i].classList.add('bad'); }
    } else {
      const s = fig.dataset.stem;
      if (bad.has(s)) { bad.delete(s); fig.classList.remove('bad'); }
      else { bad.add(s); fig.classList.add('bad'); }
    }
    lastIdx = idx; save();
  });
});

function rejectJson(){
  return JSON.stringify({schema:'ctt_v2_s1_rejects/v1', stratum:'S1',
    at:new Date().toISOString(), n_rejected:bad.size, rejects:[...bad].sort()}, null, 1);
}
document.getElementById('copy').onclick = () =>
  navigator.clipboard.writeText(rejectJson()).then(() => alert(`copied ${bad.size} rejects`));
document.getElementById('dl').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([rejectJson()], {type:'application/json'}));
  a.download = 'rejects.json'; a.click();
};
document.getElementById('clear').onclick = () => {
  if (!confirm(`Clear all ${bad.size} labels?`)) return;
  bad.clear(); document.querySelectorAll('figure.bad').forEach(f => f.classList.remove('bad')); save();
};
</script>
"""

if __name__ == "__main__":
    main()
