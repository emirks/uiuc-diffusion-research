#!/usr/bin/env python3
"""Viewer for the semflow test-time guidance sweep (misc/2026-08-21_semflow_dit_guidance).

One card per zero-shot CTT row: the demonstration (reference) + the endpoint window, then the
generation at every guidance scale s ∈ {0, 0.03, 0.1, 0.3, 1} side by side (s=0 = plain ctt_v2@10k),
seed selector 42/43, per-clip guided-loss caption. Reads the campaign's build/out tree directly (pre-store,
quick-look); the store/flagship registration follows at campaign close.

Regen: $LAB/envs-aarch64/ltx2/bin/python scripts/viewers/gen_semflow_sweep.py
Media via relative symlinks gens/ corpus/ conds/ in the viewer dir (the rule that keeps viewers alive).
"""
import glob, html, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMP = ROOT / "misc/2026-08-21_semflow_dit_guidance/build"
OUT = CAMP / "out"
STD = ROOT / "data/processed/transitions_std121"
VDIR = ROOT / "outputs/viewers/semflow_sweep"
VDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "eval_ladder"))
import encode_conditioning as ec  # noqa: E402


def rel_link(link: Path, target: Path) -> None:
    rel = os.path.relpath(target, link.parent)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == rel:
            return
        link.unlink()
    link.symlink_to(rel)


rel_link(VDIR / "gens", OUT)
rel_link(VDIR / "corpus", STD)
rel_link(VDIR / "conds", ec.CONDS)

rows = [json.loads(l) for l in open(ROOT / "eval_ladder/registry.jsonl") if l.strip()]
rows = sorted([r for r in rows if r["arm"] == "ic_gen" and r["ref_novelty"] == "zero_shot"], key=lambda r: (r["cell"], r["item_id"]))
def arm_key(a):
    pre, tag = a.rsplit("_s", 1)
    return ({"semflow": 0, "semflow_early": 1, "semflow_mid": 2}.get(pre, 9), float(tag.replace("p", ".")))
ARMS = sorted((p.name for p in OUT.glob("semflow*_s*") if (p / "videos").is_dir()), key=arm_key)
WINDOW = {"semflow": "late σ≤0.85 (steps 17–29)", "semflow_mid": "mid σ 0.4–0.9 (steps 10–26)", "semflow_early": "early σ≥0.6 (steps 0–24)"}
FAM_DESC = {
    "semflow": "x_t-nudge, LATE window (steps 17–29), raw gradient on the noisy latent — the first 48-row sweep (partial): confetti from s≥0.3.",
    "semflow_pilot": "x_t-nudge, MID window (steps 13–26), gradient box-blurred (r=1) + per-token clipped (q=0.9) — 6-row pilot: s≤0.03 invisible, s=0.1 pasted-patch ghosts.",
    "semflow_pilot2": "x_t-nudge, wider window (steps 8–27), blur r=2, clip q=0.7 — same two regimes (0.02 invisible, 0.05 smears).",
    "semflow_ref": "s=0 baseline with the DiT-feature loss LOGGED only (no nudge) — video identical to s=0; gives the unguided loss curve.",
    "semflow_x0ref": "s=0 baseline with the clean-pass (x̂0 at σ=0) loss logged only — video identical to s=0; reference for the chain rebound ratio.",
    "semflow_x0": "detached x̂0 nudge (advisor: symptom-only) — cancelled before clips landed.",
    "semflow_chain": "gradient chained THROUGH the denoiser: x_t → x̂0 → clean-pass block-24 flow/velocity vs the demo; push x_{k+1} after the Euler step, steps 17–25 (σ 0.84–0.55).",
    "semflow_chain_der": "CONTROL — same as chain, but the flow target is a WRONG-effect demo (the model still sees the matched demo). If this looks like chain, the loss carries no operator.",
    "semflow_chain_frz": "CONTROL — same as chain with target flow/velocity = 0 (freeze). Can guidance even slow the clip down cleanly?",
    "semflow_dino": "DINO version: decode x̂0 → DINOv2-base flow/velocity on 16 latent-aligned frames (20×15 grid) vs the demo's → gradient on PIXELS → re-encode difference into x̂0; steps 17–25; s in [0,1] pixel units.",
}


def family(a):
    return a.rsplit("_s", 1)[0]


def arm_label(a):
    pre, tag = a.rsplit("_s", 1)
    w = {"semflow": "late", "semflow_mid": "mid", "semflow_early": "early"}.get(pre, pre)
    return f"s = {tag.replace('p', '.')}" + ("" if tag == "0" else f" · {w}")
SEEDS = [42, 43]


def corpus_rel(clip):
    hits = glob.glob(str(STD / "*" / f"{clip}.mp4"))
    return "corpus/" + os.path.relpath(hits[0], STD) if hits else None


def trace_caption(arm, iid, seed):
    p = OUT / arm / "logs" / f"{iid}__s{seed}.json"
    if not p.exists():
        return ""
    d = json.load(open(p)); t = d.get("trace") or []
    if not t:
        return f"{d.get('sec', 0):.0f}s"
    return f"L {t[0]['loss']:.2f}→{t[-1]['loss']:.2f} · {len(t)} steps · {d.get('sec', 0):.0f}s"


n_present = {a: len(list((OUT / a / "videos").glob("*.mp4"))) for a in ARMS}
cards = []
for r in rows:
    iid, ep, ref, sided = r["item_id"], r["endpoint"], r["reference"], r["sided"]
    demo = corpus_rel(ref)
    ep_src = corpus_rel(ep) or ("conds/" + ec.cond_paths(ep, sided)["prefix"].name)
    caps = {a: {s: trace_caption(a, iid, s) for s in SEEDS} for a in ARMS}
    cells = "".join(
        f'<figure class="gen" data-fam="{family(a)}"><video data-arm="{a}" data-iid="{html.escape(iid)}" muted loop playsinline preload="none" '
        f'onmouseover="this.play()"></video>'
        f'<figcaption><b>{arm_label(a)}</b>'
        f'<span class="tr" data-caps=\'{html.escape(json.dumps(caps[a]))}\'></span></figcaption></figure>'
        for a in ARMS)
    cards.append(f'''<section class="card" data-cell="{r['cell']}">
<header><span class="cell">{r['cell']}</span> <span class="meta">{html.escape(ep)} ← demo {html.escape(ref)} · {sided}-sided · {r['gt_pool_class']}</span>
<div class="prompt">{html.escape(r['prompt'])}</div></header>
<div class="row">
<figure class="ref"><video src="{demo}" muted loop playsinline preload="metadata" onmouseover="this.play()"></video><figcaption>demonstration</figcaption></figure>
<figure class="ref ep"><video src="{ep_src}" muted loop playsinline preload="metadata" onmouseover="this.play()"></video><figcaption>endpoint (target content)</figcaption></figure>
{cells}
</div></section>''')

cells_list = sorted({r["cell"] for r in rows})
page = f'''<!doctype html><html><head><meta charset="utf-8"><title>semflow — DiT feature-flow guidance sweep</title>
<style>
body{{font:14px/1.4 system-ui,sans-serif;margin:0;background:#111;color:#ddd}}
header.top{{position:sticky;top:0;background:#181818;border-bottom:1px solid #333;padding:10px 16px;z-index:2}}
h1{{font-size:18px;margin:0 0 4px}} .sub{{color:#999;font-size:12.5px}}
.ctl{{margin-top:6px}} .ctl button{{background:#2a2a2a;color:#ddd;border:1px solid #444;padding:3px 10px;margin-right:4px;cursor:pointer}}
.ctl button.on{{background:#3a6;color:#fff;border-color:#3a6}} .ctl button.fam.on{{background:#36a;border-color:#36a}}
.card{{padding:12px 16px;border-bottom:1px solid #2a2a2a}} .card header{{margin-bottom:6px}}
.cell{{background:#334;padding:1px 6px;border-radius:3px;font-size:12px}} .meta{{color:#aaa;font-size:12.5px;margin-left:6px}}
.prompt{{color:#888;font-size:12px;margin-top:2px}}
.row{{display:flex;gap:8px;overflow-x:auto}} figure{{margin:0;flex:0 0 auto;width:165px}} figure.ref{{width:150px}}
video{{width:100%;aspect-ratio:3/4;background:#000;border-radius:4px;display:block}}
figure.ref video{{border:1px solid #a63}} figure.ep video{{border:1px solid #36a}}
figcaption{{font-size:11.5px;color:#aaa;margin-top:3px}} figcaption b{{color:#ddd}} .tr{{display:block;color:#777;font-size:10.5px}}
.missing video{{opacity:.25}}
.headline{{margin:6px 0 2px;padding:6px 8px;background:#1f2a1f;border-left:3px solid #3a6;color:#cde;font-size:13px}}
.legend{{font-size:12px;color:#aaa;margin:4px 0}} .legend ul{{margin:4px 0 0 16px;padding:0}} .legend li{{margin:2px 0}}
</style></head><body>
<header class="top"><h1>semflow — test-time DiT feature-flow guidance on ctt_v2@10k · zero-shot rows · neutral prompt</h1>
<div class="sub">Guidance: match the target's block-24 feature optical-flow + feature-velocity to the demonstration's (centered, τ=0.02, r=3), nudge x_t ← x_t − s·σ·ĝ for σ∈[0.05,0.85] (14 steps). s=0 is the bitwise-identical baseline.
Windows: late = σ∈[0.05,0.85] (steps 17–29, the owner-spec default) · mid = σ∈[0.40,0.90] (steps 10–26, last 3 clean) · early = σ∈[0.60,1.0] (steps 0–24, last 5 clean; seed 42 only for mid/early). Clips present: {" · ".join(f"{a.replace('semflow','')}: {n}" for a, n in n_present.items())}. Hover to play. Regen: scripts/viewers/gen_semflow_sweep.py</div>
<div class="ctl">family: {"".join(f'<button class="fam" data-fam="{f}" title="{html.escape(FAM_DESC.get(f, ""))}">{f.replace("semflow_", "") or "sweep"}</button>' for f in sorted({family(a) for a in ARMS}, key=lambda f: -max((OUT / a / "videos").stat().st_mtime for a in ARMS if family(a) == f)))}<button class="fam" data-fam="all">all</button></div>
<div class="headline" id="headline"></div>
<details class="legend"><summary>all arms — one line each</summary><ul>{"".join(f"<li><b>{f.replace('semflow_', '') or 'sweep'}</b> — {html.escape(FAM_DESC.get(f, ''))}</li>" for f in sorted({family(a) for a in ARMS}))}</ul></details>
<div class="ctl">seed: <button class="seed on" data-seed="42">42</button><button class="seed" data-seed="43">43</button>
&nbsp; cell: <button class="cf on" data-cell="all">all</button>{"".join(f'<button class="cf" data-cell="{c}">{c}</button>' for c in cells_list)}</div></header>
{"".join(cards)}
<script>
let seed = 42;
function apply(){{
  document.querySelectorAll('video[data-arm]').forEach(v => {{
    const src = `gens/${{v.dataset.arm}}/videos/${{v.dataset.iid}}__s${{seed}}.mp4`;
    if (v.getAttribute('src') !== src) {{ v.setAttribute('src', src); v.parentElement.classList.remove('missing');
      v.onerror = () => v.parentElement.classList.add('missing'); }}
  }});
  document.querySelectorAll('.tr').forEach(t => {{ const c = JSON.parse(t.dataset.caps); t.textContent = c[seed] || 'pending'; }});
}}
const FAM_DESC = {json.dumps(FAM_DESC)};
function showFam(f){{
  document.getElementById('headline').textContent = f === 'all' ? 'all families shown' : ((f.replace('semflow_', '') || 'sweep') + ' — ' + (FAM_DESC[f] || ''));
  document.querySelectorAll('.fam').forEach(x => x.classList.toggle('on', x.dataset.fam === f));
  document.querySelectorAll('figure.gen').forEach(g => g.style.display = (f === 'all' || g.dataset.fam === f || g.dataset.fam === 'semflow_pilot' && g.querySelector('video').dataset.arm === 'semflow_pilot_s0') ? '' : 'none');
  try {{ localStorage.setItem('semflow_fam', f); }} catch(e) {{}}
}}
document.querySelectorAll('.fam').forEach(b => b.onclick = () => showFam(b.dataset.fam));
(() => {{ const fams = [...document.querySelectorAll('.fam')].map(b => b.dataset.fam); let f = null; try {{ f = localStorage.getItem('semflow_fam'); }} catch(e) {{}}
  showFam(fams.includes(f) ? f : fams[0]); }})();
document.querySelectorAll('.seed').forEach(b => b.onclick = () => {{ document.querySelectorAll('.seed').forEach(x => x.classList.remove('on')); b.classList.add('on'); seed = +b.dataset.seed; apply(); }});
document.querySelectorAll('.cf').forEach(b => b.onclick = () => {{ document.querySelectorAll('.cf').forEach(x => x.classList.remove('on')); b.classList.add('on');
  document.querySelectorAll('.card').forEach(c => c.style.display = (b.dataset.cell === 'all' || c.dataset.cell === b.dataset.cell) ? '' : 'none'); }});
apply();
// autoplay whatever is on screen (muted), pause when scrolled out
const io = new IntersectionObserver(es => es.forEach(e => {{ const v = e.target; if (e.isIntersecting && v.getAttribute('src')) {{ v.play().catch(()=>{{}}); }} else {{ v.pause(); }} }}), {{threshold: 0.2}});
document.querySelectorAll('video').forEach(v => io.observe(v));
</script></body></html>'''
(VDIR / "index.html").write_text(page)
print(f"[viewer] wrote {VDIR/'index.html'}: {len(rows)} cards × {len(ARMS)} arms; clips present {n_present}")
