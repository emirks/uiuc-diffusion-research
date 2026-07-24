"""exp_077 — build the D2 (real-stream) dataset viewer, in the exp_076 viewer style.

Static HTML, compact auto-fill grid of small cards, autoplay-on-scroll (IntersectionObserver),
optional filmstrips, light/dark. Sections are chosen so the two structural properties the dataset
exists for are directly inspectable, plus a rejects section so the gate can be audited by eye.

D2 vs D1, the one change that matters: there is NO extension policy. Both layers are the REAL
121-frame source clips playing in lockstep, so the middle is genuine footage rather than fabricated
frames (D1's `flow` extrapolation melted; exp_075's `boomerang` ran motion backward; `hold` froze).
`assert1` proves it per clip: pure-phase frames are byte-identical to the source.

    python experiments/exp_077_synth_stratum/build_viewer_d2.py
    # serve from REPO ROOT:  python -m http.server 8017
    # -> /outputs/viewers/d2_dataset/
"""

from __future__ import annotations

import html
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
AUD = REPO_ROOT / "outputs/videos/exp_077_synth_stratum_d2/audit"
BASE = "/outputs/videos/exp_077_synth_stratum_d2/audit"
OUT = REPO_ROOT / "outputs/viewers/d2_dataset"
SEED = 11

CSS = """
:root{--bg:#0f1115;--fg:#e6e8ec;--mut:#9aa3b2;--card:#171a21;--line:#252a34;--acc:#7cc0ff}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--fg:#12141a;--mut:#5d6675;--card:#fff;--line:#e2e5ea;--acc:#1668c9}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:44px 0 4px;padding-top:20px;border-top:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 18px;max-width:80ch}
.sub b{color:var(--fg)}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 8px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.stat b{display:block;font-size:20px} .stat span{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.grid.pair{grid-template-columns:repeat(auto-fill,minmax(330px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card video{width:100%;display:block;background:#000;aspect-ratio:480/640;object-fit:cover}
.duo{display:grid;grid-template-columns:1fr 1fr;gap:2px}
.duo figure{margin:0;position:relative}
.duo figcaption{position:absolute;top:4px;left:4px;background:#000a;color:#fff;font-size:9.5px;
 padding:1px 5px;border-radius:3px;letter-spacing:.04em;text-transform:uppercase}
.meta{padding:9px 11px;font-size:11.5px}
.meta .sh{font-weight:600;color:var(--acc);font-size:13px}
.meta .kv{color:var(--mut);margin-top:3px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10.5px;word-break:break-word}
.tag{display:inline-block;background:var(--line);border-radius:4px;padding:1px 5px;margin-right:4px;font-size:10px}
.strip{width:100%;display:block;border-top:1px solid var(--line)}
.ok{color:#4ade80} .warn{color:#fbbf24} .bad{color:#f87171}
.toggle{margin:10px 0 16px;color:var(--mut);font-size:12.5px}
.toggle label{margin-right:16px;cursor:pointer}
"""

JS = """
document.addEventListener('DOMContentLoaded',()=>{
  const strips=document.getElementById('showstrips');
  const sync=()=>document.querySelectorAll('.strip').forEach(s=>s.style.display=strips.checked?'block':'none');
  strips.addEventListener('change',sync); sync();
  const play=document.getElementById('autoplay');
  const io=new IntersectionObserver(es=>es.forEach(e=>{
    if(e.isIntersecting&&play.checked)e.target.play().catch(()=>{});else e.target.pause();}),{threshold:.25});
  document.querySelectorAll('video').forEach(v=>io.observe(v));
  play.addEventListener('change',()=>document.querySelectorAll('video').forEach(v=>{
    if(play.checked)v.play().catch(()=>{});else v.pause();}));
});
"""


def cls(v: float, ok: float, warn: float) -> str:
    return "ok" if v <= ok else ("warn" if v <= warn else "bad")


def load() -> tuple[list[dict], dict]:
    verdicts = json.loads((AUD / "verdicts.json").read_text())
    rows = [json.loads(l) for f in sorted((AUD / "meta").glob("rows_shard*.jsonl"))
            for l in f.read_text().splitlines() if l.strip()]
    by = defaultdict(dict)
    for r in rows:
        by[r["tuple_id"]][r["role"]] = r
    tuples = []
    for tid, d in sorted(by.items()):
        ref, tgt = d.get("reference"), d.get("target")
        if not ref or not tgt:
            continue
        v_t, v_r = verdicts.get(tgt["stem"], {}), verdicts.get(ref["stem"], {})
        seam = [float(x) for x in (tgt.get("assert2", {}).get("seam_ratio") or [0, 0])[:2]]
        tm = tgt["timing"]
        tuples.append({
            "id": tid, "shader": tgt["shader"], "easing": tgt["easing"],
            "flip": tgt.get("flip"), "swap": tgt.get("swap"),
            "params": tgt.get("params", {}),
            "onset": tm["onset"], "rel": tm["release"], "dur": tm["duration"],
            "ref_stem": ref["stem"], "tgt_stem": tgt["stem"],
            "ref_A": ref.get("A"), "ref_B": ref.get("B"),
            "tgt_A": tgt.get("A"), "tgt_B": tgt.get("B"),
            "pure": float((tgt.get("assert1") or {}).get("max_pure", 0)),
            "seam": seam, "m1": float(tgt.get("m1_p10", 0)), "m2": float(tgt.get("m2_max_dq", 0)),
            "legs": {k: bool(v_t.get(k)) for k in ("assert1", "assert2", "m1", "m2")},
            "pass": bool(v_t.get("pass")) and bool(v_r.get("pass")),
        })
    return tuples, verdicts


def meta_block(t: dict, *, show_pass: bool = True) -> str:
    p = " ".join(f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
                 for k, v in t["params"].items()) or "no params"
    legs = " ".join(
        f'<span class="{"ok" if t["legs"][k] else "bad"}">{n}{"✓" if t["legs"][k] else "✗"}</span>'
        for k, n in (("assert1", "identity"), ("assert2", "seam"), ("m1", "mush"), ("m2", "cut")))
    tags = f'<span class="tag">{html.escape(t["easing"])}</span>'
    if t.get("flip") and t["flip"] != "none":
        tags += f'<span class="tag">flip {t["flip"]}</span>'
    if t.get("swap"):
        tags += '<span class="tag">swap</span>'
    s0, s1 = t["seam"]
    return (f'<div class="meta"><div class="sh">{html.escape(t["shader"])}</div>'
            f'<div>{tags}</div>'
            f'<div class="kv">{html.escape(p)}</div>'
            f'<div class="kv">onset {t["onset"]:.0f} → release {t["rel"]:.0f} (dur {t["dur"]:.0f} of 104)</div>'
            f'<div class="kv">seam <span class="{cls(max(s0,s1),2.0,4.0)}">{s0:.2f}/{s1:.2f}</span>'
            f' · pure <span class="{cls(t["pure"],0.5,5.0)}">{t["pure"]:.2f}</span>'
            f' · mush p10 {t["m1"]:.2f} · Δq {t["m2"]:.2f}</div>'
            + (f'<div class="kv">{legs}</div>' if show_pass else '')
            + f'<div class="kv">A {html.escape(str(t["tgt_A"]))[:34]}<br>B {html.escape(str(t["tgt_B"]))[:34]}</div>'
            '</div>')


def card_single(t: dict) -> str:
    return (f'<div class="card"><video src="{BASE}/videos/{t["tgt_stem"]}.mp4" muted loop playsinline '
            f'preload="none"></video>'
            f'<img class="strip" src="{BASE}/filmstrips/{t["tgt_stem"]}.jpg" loading="lazy">'
            + meta_block(t) + '</div>')


def card_pair(t: dict) -> str:
    duo = (f'<div class="duo">'
           f'<figure><video src="{BASE}/videos/{t["ref_stem"]}.mp4" muted loop playsinline preload="none"></video>'
           f'<figcaption>reference (demo)</figcaption></figure>'
           f'<figure><video src="{BASE}/videos/{t["tgt_stem"]}.mp4" muted loop playsinline preload="none"></video>'
           f'<figcaption>target</figcaption></figure></div>')
    strip = (f'<img class="strip" src="{BASE}/filmstrips/{t["tgt_stem"]}.jpg" loading="lazy">')
    return f'<div class="card">{duo}{strip}{meta_block(t)}</div>'


def main() -> None:
    tuples, _ = load()
    rng = random.Random(SEED)
    ok = [t for t in tuples if t["pass"]]
    bad = [t for t in tuples if not t["pass"]]

    by_sh = defaultdict(list)
    for t in ok:
        by_sh[t["shader"]].append(t)
    # shaders with the most passing instances = the reliable operators
    top_sh = sorted(by_sh.items(), key=lambda kv: -len(kv[1]))[:10]

    S = []
    S.append('<h2>Same operator → different content <span class="tag">the positive pair</span></h2>'
             '<p class="sub">Reference and target share the <b>same operator</b> — shader, continuous '
             'parameters, easing and timing — on <b>disjoint endpoint pairs</b>. Reference content is '
             'therefore useless for predicting the target, so the only way to solve the task is to '
             'extract the operator itself. Both middles are composed of real moving footage.</p>'
             '<div class="grid pair">'
             + "".join(card_pair(t) for t in rng.sample(ok, min(36, len(ok)))) + '</div>')

    for sh, items in top_sh:
        S.append(f'<h2>{html.escape(sh)} <span class="tag">{len(items)} passing</span></h2>'
                 f'<p class="sub">The same shader family across unrelated content — if it reads as the '
                 f'same effect on every scene, it is a learnable operator rather than content-specific noise.</p>'
                 '<div class="grid">'
                 + "".join(card_single(t) for t in items[:12]) + '</div>')

    S.append('<h2>Rejected by the gate <span class="tag">audit the filter</span></h2>'
             '<p class="sub">Clips the frozen gate threw out. Worth a look: if these are obviously fine, '
             'the gate is too strict; if they are mush, near-cuts or broken identity, it is doing its job. '
             'Failing legs are marked in red.</p>'
             '<div class="grid">'
             + "".join(card_single(t) for t in rng.sample(bad, min(24, len(bad)))) + '</div>')

    n_sh_pass = len(by_sh)
    stats = [
        (len(tuples), "audit tuples"), (len(ok), "gate-passing"),
        (f"{100*len(ok)/max(1,len(tuples)):.0f}%", "pass rate"),
        (n_sh_pass, "shaders with passing clips"),
        ("0", "aux-map wipes (removed)"), ("none", "extension policy"),
    ]
    stat_html = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>exp_077 D2 — real-stream procedural transitions</title>
<style>{CSS}</style><script>{JS}</script></head><body>
<div class="wrap">
<h1>D2 — real-stream procedural transitions</h1>
<p class="sub">Each clip is <b>121 frames</b>: the given <code>start9</code> bucket (frames 0–8), a
procedural transition, and the given <code>end9</code> bucket (frames 112–120). The change that
matters versus the rejected first build: <b>there is no extension policy</b>. Both layers are the
<b>real 121-frame source clips playing in lockstep</b>, so the middle is composed of genuine footage
instead of fabricated frames — the earlier build extrapolated optical flow, which compounded error
and melted the scene into mush; ping-pong ran the motion backward; hold froze it. Per-clip
<code>assert1</code> proves the pure phases are byte-identical to the source. Aux-map luma-matte
wipes are removed entirely.</p>
<div class="stats">{stat_html}</div>
<div class="toggle">
<label><input type="checkbox" id="autoplay" checked> autoplay on scroll</label>
<label><input type="checkbox" id="showstrips"> show filmstrips</label>
</div>
{''.join(S)}
</div></body></html>"""

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page)
    for f in ("data.json",):
        if (OUT / f).exists():
            (OUT / f).unlink()
    print(f"[viewer-d2] tuples={len(tuples)} pass={len(ok)} ({100*len(ok)/max(1,len(tuples)):.1f}%) "
          f"shaders_with_pass={n_sh_pass} sections={len(S)}")
    print(f"[viewer-d2] -> {(OUT / 'index.html').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
