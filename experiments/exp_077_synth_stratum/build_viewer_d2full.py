"""exp_077 — build the FINAL D2 dataset viewer (exp_076 style), over the mass build.

Static HTML, compact 230px auto-fill grid, autoplay-on-scroll (IntersectionObserver), filmstrip
toggle, colour-coded metrics, light/dark. Sections: same-operator pairs (the positive pair the
dataset exists for), one section per shader, and a rejects section so the frozen gate can be
audited by eye. Adapted from build_viewer_d2.py for the d2full metadata schema (accepted-tuple
JSONL + attempts JSONL for the rejects).

    python build_viewer_d2full.py [--sub d2full]
    # serve from REPO ROOT:  python -m http.server 8017   ->  /outputs/viewers/d2_dataset/
"""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
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


def meta_block(t: dict, base: str, *, legs: bool = False) -> str:
    p = " ".join(f"{k}={v}" for k, v in (t.get("params") or {}).items()) or "no params"
    tags = f'<span class="tag">{html.escape(t["easing"])}</span>'
    if t.get("flip") and t["flip"] != "none":
        tags += f'<span class="tag">flip {t["flip"]}</span>'
    if t.get("swap"):
        tags += '<span class="tag">swap</span>'
    nf = (t["clips"]["target"].get("dfg") or {}).get("n_flag", 0)
    if nf:
        tags += f'<span class="tag">dfg {nf} flagged frame{"s" if nf > 1 else ""}</span>'
    if t.get("m1_min_flag"):
        tags += '<span class="tag">m1_min flag</span>'
    c = t["clips"]["target"]
    s0, s1 = c["assert2"]["seam_ratio"][:2]
    tm = t["timing"]
    legs_html = ""
    if legs:
        v = c["verdict"]
        legs_html = '<div class="kv">' + " ".join(
            f'<span class="{"ok" if v[k] else "bad"}">{n}{"OK" if v[k] else "X"}</span>'
            for k, n in (("assert1", "identity "), ("assert2", "seam "),
                         ("m1", "mush "), ("m2", "cut "))) + "</div>"
    return (f'<div class="meta"><div class="sh">{html.escape(t["shader"])}</div>'
            f'<div>{tags}</div>'
            f'<div class="kv">{html.escape(p)[:150]}</div>'
            f'<div class="kv">onset {tm["onset"]:.0f} &rarr; release {tm["release"]:.0f} '
            f'(dur {tm["duration"]:.0f} of 104)</div>'
            f'<div class="kv">seam <span class="{cls(max(s0, s1), 2.0, 4.0)}">{s0:.2f}/{s1:.2f}</span>'
            f' &middot; pure <span class="{cls(c["assert1"]["max_pure"], 0.5, 5.0)}">'
            f'{c["assert1"]["max_pure"]:.2f}</span>'
            f' &middot; mush p10 {c["m1_p10"]:.2f} &middot; dq {c["m2_max_dq"]:.2f}</div>'
            + legs_html
            + f'<div class="kv">A {html.escape(str(t["target_pair"]["A"]))[:34]}<br>'
            f'B {html.escape(str(t["target_pair"]["B"]))[:34]}</div></div>')


def card_single(t: dict, base: str) -> str:
    s = t["target_stem"]
    return (f'<div class="card"><video src="{base}/videos/{s}.mp4" muted loop playsinline '
            f'preload="none"></video>'
            f'<img class="strip" src="{base}/filmstrips/{s}.jpg" loading="lazy">'
            + meta_block(t, base) + '</div>')


def card_pair(t: dict, base: str) -> str:
    duo = (f'<div class="duo">'
           f'<figure><video src="{base}/videos/{t["reference_stem"]}.mp4" muted loop playsinline '
           f'preload="none"></video><figcaption>reference (demo)</figcaption></figure>'
           f'<figure><video src="{base}/videos/{t["target_stem"]}.mp4" muted loop playsinline '
           f'preload="none"></video><figcaption>target</figcaption></figure></div>')
    return (f'<div class="card">{duo}'
            f'<img class="strip" src="{base}/filmstrips/{t["target_stem"]}.jpg" loading="lazy">'
            + meta_block(t, base) + '</div>')


def card_reject(r: dict, base: str) -> str:
    st = r["reject_stem"]
    # the saved clip is the reference only when the DFG rejected the reference leg
    role = "reference" if st.endswith("_ref") and r.get("reference") else "target"
    c = r[role]
    v = c["verdict"]
    legs = " ".join(f'<span class="{"ok" if v[k] else "bad"}">{n}{"OK" if v[k] else "X"}</span>'
                    for k, n in (("assert1", "identity "), ("assert2", "seam "),
                                 ("m1", "mush "), ("m2", "cut ")))
    why = r.get("reject_reason", "frozen_gate")
    d = c.get("dfg") or {}
    if why == "dfg" and d:
        bt = ", ".join(f"{k} {n}" for k, n in d.get("by_test", {}).items() if n)
        w = d.get("worst", {})
        legs += (f' <span class="bad">DFG X {d.get("n_flag")}/{d.get("n_window")} frames'
                 f'{" (" + bt + ")" if bt else ""}</span>'
                 f'<br><span class="bad">L {w.get("L_min")}&ndash;{w.get("L_max")} '
                 f'&middot; S_min {w.get("S_min")} &middot; sat_max {w.get("sat_max")}</span>')
    p = " ".join(f"{k}={v2}" for k, v2 in (r.get("params") or {}).items()) or "no params"
    return (f'<div class="card"><video src="{base}/rejects/{st}.mp4" muted loop playsinline '
            f'preload="none"></video>'
            f'<img class="strip" src="{base}/rejects_filmstrips/{st}.jpg" loading="lazy">'
            f'<div class="meta"><div class="sh">{html.escape(r["shader"])}</div>'
            f'<div><span class="tag">{html.escape(r.get("easing", "?"))}</span>'
            f'<span class="tag">REJECTED: {html.escape(why)}</span>'
            f'<span class="tag">{role}</span></div>'
            f'<div class="kv">{html.escape(p)[:150]}</div>'
            f'<div class="kv">seam {c["assert2"]["seam_max_ratio"]:.2f} &middot; '
            f'mush p10 {c["m1_p10"]:.2f} &middot; dq {c["m2_max_dq"]:.2f} &middot; '
            f'pure {c["assert1"]["max_pure"]:.2f}</div>'
            f'<div class="kv">{legs}</div></div></div>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config_d2full.yaml").read_text())
    sub = args.sub or cfg["outputs"]["subdir"]
    run = REPO_ROOT / cfg["outputs"]["dir"] / sub
    base = f"/{(run.relative_to(REPO_ROOT)).as_posix()}"
    out = REPO_ROOT / cfg["outputs"]["viewer_dir"]
    gt = cfg["gate"]

    tuples = [json.loads(l) for f in sorted((run / "meta").glob("tuples_shard*.jsonl"))
              for l in f.read_text().splitlines() if l.strip()]
    tuples.sort(key=lambda t: t["tuple_id"])
    rejects = [json.loads(p.read_text()) for p in sorted((run / "rejects").glob("*.json"))]
    if not tuples:
        raise SystemExit(f"[viewer] no accepted tuples under {run}")
    rng = random.Random(SEED)

    by_sh = defaultdict(list)
    for t in tuples:
        by_sh[t["shader"]].append(t)
    top_sh = sorted(by_sh.items(), key=lambda kv: -len(kv[1]))[:12]
    n_flag = sum(1 for t in tuples if t.get("m1_min_flag"))
    n_dfg_flag = sum(1 for t in tuples
                     if any((t["clips"][r].get("dfg") or {}).get("n_flag", 0)
                            for r in ("target", "reference")))
    dcfg = tuples[0].get("dfg_config") or {}

    S = []
    S.append('<h2>Same operator &rarr; different content <span class="tag">the positive pair</span></h2>'
             '<p class="sub">Reference and target share the <b>same operator</b> — shader, continuous '
             'parameters, easing, flip/swap <i>and</i> timing — applied to <b>content-disjoint '
             'endpoint pairs</b>. The reference content is therefore useless for predicting the '
             'target, so the only transferable signal is the operator itself. Both middles are '
             'composed of real moving footage.</p><div class="grid pair">'
             + "".join(card_pair(t, base) for t in rng.sample(tuples, min(36, len(tuples))))
             + '</div>')

    for sh, items in top_sh:
        S.append(f'<h2>{html.escape(sh)} <span class="tag">{len(items)} in the dataset</span></h2>'
                 '<p class="sub">The same shader family across unrelated content — if it reads as the '
                 'same effect on every scene it is a learnable operator rather than content-specific '
                 'noise.</p><div class="grid">'
                 + "".join(card_single(t, base) for t in items[:12]) + '</div>')

    if rejects:
        S.append('<h2>Rejected by the gates <span class="tag">audit the filter</span></h2>'
                 '<p class="sub">A bounded sample of clips thrown out during rejection sampling — '
                 'by the frozen gate (<code>reject_reason: frozen_gate</code>) or by the '
                 'degenerate-frame gate (<code>reject_reason: dfg</code>: sustained near-black, '
                 'white-blowout or flat/near-zero-variance frames inside the transition window). '
                 'If these look fine the gates are too strict; if they are mush, near-cuts, broken '
                 'identity or dead mattes they are doing their job.'
                 '</p><div class="grid">'
                 + "".join(card_reject(r, base) for r in rejects[:36]) + '</div>')

    stats = [
        (len(tuples), "tuples"), (len({t["target_index"] for t in tuples}), "target pairs"),
        (len(by_sh), "shaders"), (2 * len(tuples), "clips"),
        (f"{gt['tau']}", "frozen &tau;"), ("0", "aux maps"), ("none", "extension policy"),
        (n_dfg_flag, "with any DFG-flagged frame"), (n_flag, "m1_min flagged (non-gating)"),
    ]
    stat_html = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>exp_077 D2 — final synthetic transition dataset</title>
<style>{CSS}</style><script>{JS}</script></head><body>
<div class="wrap">
<h1>D2 — final synthetic transition dataset</h1>
<p class="sub">Each clip is <b>121 frames</b>: a pinned 9-frame start bucket (0&ndash;8), a procedural
transition, and a pinned 9-frame end bucket (112&ndash;120). Both layers are the <b>real 121-frame
source clips playing in lockstep</b> — there is <b>no extension policy</b>, so the middle is genuine
footage rather than fabricated frames. Per-clip <code>assert1</code> proves the pure phases are
byte-identical to the source. Every tuple shown passed the frozen gate
(&tau;={gt['tau']} mush &middot; max_pure&le;{gt['assert1_tol']} identity &middot;
seam&le;{gt['seam_max']} &middot; &Delta;q&le;{gt['m2_max_dq']}) on <b>both</b> its clips, selected by
per-slot rejection sampling, <b>and</b> the calibrated <b>degenerate-frame gate</b>
(<code>dfg.py</code>: fewer than K={dcfg.get('K')} frames in the transition window with mean
luma&lt;{dcfg.get('theta_black')} / &gt;{dcfg.get('theta_white')} or luma
std&lt;{dcfg.get('theta_flat')}). Parameter clamping was abandoned and never ran. Aux-map wipes are
absent entirely.</p>
<div class="stats">{stat_html}</div>
<div class="toggle">
<label><input type="checkbox" id="autoplay" checked> autoplay on scroll</label>
<label><input type="checkbox" id="showstrips"> show filmstrips</label>
</div>
{''.join(S)}
</div></body></html>"""

    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)
    print(f"[viewer] tuples={len(tuples)} shaders={len(by_sh)} rejects={len(rejects)} "
          f"sections={len(S)} dfg_flagged_tuples={n_dfg_flag} m1_min_flag={n_flag}")
    print(f"[viewer] -> {(out / 'index.html').relative_to(REPO_ROOT)}  (base={base})")


if __name__ == "__main__":
    main()
