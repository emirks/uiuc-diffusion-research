#!/usr/bin/env python
"""Build the HumanVid REAL (Pexels) inspection viewer.

IMPORTANT -- why this viewer streams instead of serving local files:
the real HumanVid subset is ~19k Pexels URLs, and the Pexels Terms of Service
prohibit "data mining, extraction, scraping and the use of programs or robots for
automatic data collection ... including without limitation for machine learning
purposes", plus "bulk, large-scale or systematic copying". So we do NOT copy the
media to disk. The viewer points <video> straight at the Pexels CDN: the owner's
browser plays them exactly as it would on pexels.com, and nothing is collected.

Filmstrips are likewise built client-side in a <canvas> from the already-streaming
video, so no frames are written anywhere either.

Usage:
    python scripts/build_humanvid_real_viewer.py \
        --clips data/manifests/humanvid_real/clips.jsonl.gz \
        --report data/manifests/humanvid_real/fitness_report.json \
        --out outputs/viewers/humanvid_real \
        --n-per-split 30
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

TARGET_AR = 480 / 640


def crop_retention(w: int, h: int) -> float:
    if (w / h) > TARGET_AR:
        cw, ch = h * TARGET_AR, float(h)
    else:
        cw, ch = float(w), w / TARGET_AR
    return (cw * ch) / (w * h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-per-split", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--verified",
        type=Path,
        default=None,
        help="optional file of URLs already HEAD-checked 200; sample only from these",
    )
    ap.add_argument(
        "--probe",
        type=Path,
        default=None,
        help="url_probe.json from the HEAD-only size/liveness probe",
    )
    args = ap.parse_args()

    opener = gzip.open if args.clips.suffix == ".gz" else open
    with opener(args.clips, "rt") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    report = json.loads(args.report.read_text())
    probe = json.loads(args.probe.read_text()) if args.probe and args.probe.exists() else {}

    if args.verified and args.verified.exists():
        alive = {u.strip() for u in args.verified.read_text().splitlines() if u.strip()}
        recs = [r for r in recs if r["url"] in alive]

    rng = random.Random(args.seed)
    picked = []
    for split in ("vertical", "horizontal"):
        pool = [r for r in recs if r["split"] == split and r.get("frames")]
        rng.shuffle(pool)
        picked.extend(pool[: args.n_per_split])
    picked.sort(key=lambda r: (r["split"], int(r["id"])))

    nv = sum(1 for r in picked if r["split"] == "vertical")
    nh = len(picked) - nv
    ov = report["overall"]
    ef = ov["endpoint_fitness"]

    cards = []
    for r in picked:
        dur = r["frames"] / r["fps"]
        ret = crop_retention(r["width"], r["height"])
        page = f"https://www.pexels.com/video/{r['id']}/"
        cards.append(
            f'<div class="c" data-fam="{r["split"]}" data-dur="{dur:.2f}">'
            f'<div class="vw"><video src="{r["url"]}" muted loop playsinline preload="none" '
            f'crossorigin="anonymous"></video><canvas class="fs"></canvas></div>'
            f'<div class="m"><b><a href="{page}" target="_blank" rel="noopener">pexels/{r["id"]}</a></b>'
            f'<span>{r["width"]}x{r["height"]} · {r["frames"]}f · {r["fps"]}fps · {dur:.1f}s</span>'
            f'<span>crop keeps {ret*100:.0f}% of frame</span></div></div>'
        )

    # NOTE: the repo HTTP server sends "Content-type: text/html" with no charset, so
    # the meta tag is required -- without it the browser falls back to latin-1 and every
    # non-ASCII glyph mojibakes (the older humanvid_sample viewer has this bug).
    html = f"""<meta charset="utf-8">
<title>HumanVid REAL (Pexels) — {len(picked)}-clip sample</title>
<style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--card:#fafafa;--warn:#8a4b00;--warnbg:#fff6e5;--warnln:#f0c98a}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#999;--line:#2a2a2a;--card:#1a1a1a;--warn:#ffb454;--warnbg:#2a1e0a;--warnln:#5a3d10}}}}
:root[data-theme=dark]{{--bg:#111;--fg:#eee;--mut:#999;--line:#2a2a2a;--card:#1a1a1a;--warn:#ffb454;--warnbg:#2a1e0a;--warnln:#5a3d10}}
:root[data-theme=light]{{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--card:#fafafa;--warn:#8a4b00;--warnbg:#fff6e5;--warnln:#f0c98a}}
body{{background:var(--bg);color:var(--fg);font:13px/1.4 ui-sans-serif,system-ui,sans-serif;margin:0;padding:16px}}
h1{{font-size:16px;margin:0 0 4px}} .sub{{color:var(--mut);margin-bottom:12px}}
a{{color:inherit}}
.warn{{background:var(--warnbg);border:1px solid var(--warnln);color:var(--warn);border-radius:8px;padding:10px 12px;margin-bottom:14px;max-width:1100px}}
.warn>b:first-child{{display:block;margin-bottom:4px;font-size:13px}}
.warn code{{font-size:11px}}
.bar{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;align-items:center}}
button{{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer;font:inherit}}
button.on{{border-color:#4a9;color:#4a9}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}}
.c{{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.vw{{position:relative;background:#000}}
.c video,.c canvas{{width:100%;display:block;background:#000}}
.c canvas.fs{{display:none}} body.strips .c video{{display:none}} body.strips .c canvas.fs{{display:block}}
.m{{padding:6px 8px;display:flex;flex-direction:column;gap:2px}}
.m b{{font-size:12px}} .m span{{color:var(--mut);font-size:11px}}
.note{{border:1px solid var(--line);background:var(--card);border-radius:8px;padding:10px 12px;margin-bottom:14px;max-width:1100px}}
.note>b{{display:block;margin-bottom:6px}}
.note ul{{margin:0;padding-left:18px}} .note li{{margin:3px 0}}
.note code{{font-size:11px}}
.tbl{{border-collapse:collapse;margin:0 0 14px;font-size:12px}}
.tbl td{{border:1px solid var(--line);padding:3px 8px}} .tbl td:first-child{{color:var(--mut)}}
</style>
<h1>HumanVid — REAL (Pexels) subset, {len(picked)}-clip sample</h1>
<div class="sub">{nv} vertical + {nh} horizontal, sampled from the full {ov['n_clips']:,}-URL manifest ·
streamed live from the Pexels CDN, <b>nothing downloaded</b></div>

<div class="warn"><b>⚠ Licence: these are NOT training-safe for us.</b>
The real HumanVid subset is not redistributed as media — it is {ov['n_clips']:,} <b>Pexels.com</b> URLs.
The Pexels Terms of Service prohibit “data mining, extraction, scraping and the use of programs or robots for
automatic data collection … <b>including without limitation for machine learning purposes</b>”, and “bulk,
large-scale or systematic copying”. The API terms further bar collecting content “to train, fine-tune,
<b>evaluate</b>, or develop ML/AI models <b>or datasets</b>” without explicit permission.
The Apache-2.0/CC-BY-4.0 on the HumanVid repos covers <i>their</i> code, camera annotations and UE renders — it
does not and cannot relicense third-party Pexels footage.
<b>Conclusion: no bulk fetch was performed and none should be.</b> This page streams a sample for visual
judgement only — the same thing your browser does on pexels.com.</div>

<table class="tbl">
<tr><td>real clips in manifest</td><td><b>{ov['n_clips']:,}</b> ({ov['orientation'].get('landscape',0):,} landscape / {ov['orientation'].get('portrait',0):,} portrait)</td></tr>
<tr><td>with camera annotation</td><td>{ov['n_with_camera_annotation']:,}</td></tr>
<tr><td>total footage</td><td>{ov['duration_s']['total_hours']} h · median clip {ov['duration_s']['median']}s (p10 {ov['duration_s']['p10']}s / p90 {ov['duration_s']['p90']}s)</td></tr>
<tr><td>fps mix</td><td>{' · '.join(f'{k}fps: {v:,}' for k,v in ov['fps'].items())}</td></tr>
<tr><td>≥5.04s (one 121f@24 endpoint)</td><td>{ef['duration_ge_5.04s']['n']:,} ({ef['duration_ge_5.04s']['pct']}%)</td></tr>
<tr><td>portrait crop w/o upscale</td><td>{ef['portrait_crop_no_upscale']['n']:,} ({ef['portrait_crop_no_upscale']['pct']}%) · median {ef['crop_area_retained']['median']*100:.0f}% of frame kept</td></tr>
<tr><td>max 121f endpoints if segmented</td><td>~{ef['max_121f_endpoints_if_segmented']:,}</td></tr>
<tr><td>URL liveness (HEAD probe)</td><td>{probe.get('liveness_pct','?')}% of {probe.get('n_probed','?')} sampled URLs return 200 · mean clip {probe.get('mean_clip_bytes',0)/1e6:.1f} MB @ {probe.get('mean_bitrate_mbps','?')} Mbps</td></tr>
<tr><td>projected size if fetched</td><td><b>~{probe.get('projected_full_fetch_gb','?')} GB</b> at source resolution — <b>not fetched, and should not be</b></td></tr>
</table>

<div class="note"><b>Fitness vs our endpoint contract</b> (480x640 portrait · 121f · 24fps · single subject bbox ≥0.15 · no letterbox · single-shot)
<ul>
<li><b>Resolution / length: pass.</b> All 1080p+; 97.5% hold ≥5.04s, so ~54k distinct 121f endpoints exist in principle. Every clip crops to 480x640 without upscaling.</li>
<li><b>fps: needs resample.</b> Only 2,538 clips (13%) are natively 24fps — 67% are 25fps and 19% 30fps.</li>
<li><b>Single-shot: pass.</b> HumanVid excluded videos with shot changes and with “exits, entrances or occlusions”, so shot-boundary risk is low.</li>
<li><b>Single subject: partial fail.</b> Their rule was “few people (n≤4)”, not one — multi-person clips are in scope and need our own filter. Their prominence floor was bbox r&gt;0.07, less than half our 0.15.</li>
<li><b>Portrait crop is lossy on landscape.</b> The 11,411 landscape clips keep only ~42% of frame area; the 7,851 vertical ones keep 75%. Cropping landscape does raise a subject's relative area (~0.07 → ~0.17), so it can help clear our 0.15 bar — but it also cuts framing context and risks clipping limbs.</li>
<li><b>Letterboxing: low risk.</b> Native stock resolutions throughout (1920x1080, 1080x1920, 2048x1080, 1080x2048); no letterbox-shaped anomalies beyond a handful of true cinematic crops.</li>
<li><b>Diversity: none — volume only.</b> The pool is already 85% label <code>person</code>, and this set is 100% human-centric by construction (~100 human keywords). It would deepen the dominant mode, not broaden it.</li>
</ul></div>

<div class="bar">
<button onclick="f(this,'all')" class="on">all ({len(picked)})</button>
<button onclick="f(this,'vertical')">vertical ({nv})</button>
<button onclick="f(this,'horizontal')">horizontal ({nh})</button>
<button onclick="strips(this)">filmstrips</button>
<button onclick="var r=document.documentElement;r.dataset.theme=r.dataset.theme==='dark'?'light':'dark'">theme</button>
<span class="sub" style="margin:0">clips served by <a href="https://www.pexels.com" target="_blank" rel="noopener">Pexels</a></span>
</div>

<div class="g">
{chr(10).join(cards)}
</div>

<script>
function f(b,k){{
  document.querySelectorAll('.bar button').forEach(x=>{{
    var t=x.textContent; if(t.indexOf('filmstrips')<0&&t.indexOf('theme')<0) x.classList.remove('on');
  }});
  b.classList.add('on');
  document.querySelectorAll('.c').forEach(c=>c.style.display=(k==='all'||c.dataset.fam===k)?'':'none');
}}

// Autoplay only what is on screen -- keeps concurrent CDN streams low.
var io=new IntersectionObserver(es=>es.forEach(e=>{{
  var v=e.target.querySelector('video'); if(!v) return;
  if(e.isIntersecting){{ v.preload='auto'; v.play().catch(()=>{{}}); }} else {{ v.pause(); }}
}}),{{rootMargin:'100px'}});
document.querySelectorAll('.c').forEach(c=>io.observe(c));

// Filmstrip: N frames sampled client-side from the stream into a canvas.
// Nothing is written to disk -- purely a render of what is already streaming.
// Tiles are downscaled to TILE_W so a 60-card page stays a few MB of canvas, and
// builds run through a small queue so we never open 60 concurrent CDN streams.
var N=5, TILE_W=200, MAX_PARALLEL=3, queue=[], running=0;

function pump(){{
  while(running<MAX_PARALLEL && queue.length){{ running++; queue.shift()(); }}
}}
function done(){{ running--; pump(); }}

function buildStrip(card){{
  if(card.dataset.strip) return; card.dataset.strip='1';
  queue.push(function(){{
    var src=card.querySelector('video').src, cv=card.querySelector('canvas.fs');
    var probe=document.createElement('video'), finished=false;
    function finish(ok){{
      if(finished) return; finished=true;
      if(!ok) card.dataset.strip='';
      probe.removeAttribute('src'); probe.load(); done();
    }}
    probe.muted=true; probe.playsInline=true; probe.crossOrigin='anonymous';
    probe.preload='metadata'; probe.src=src;
    setTimeout(function(){{ finish(false); }}, 30000);  // never wedge the queue
    probe.addEventListener('error',function(){{ finish(false); }});
    probe.addEventListener('loadedmetadata',function(){{
      var vw=probe.videoWidth, vh=probe.videoHeight;
      if(!vw||!vh){{ return finish(false); }}
      var tw=TILE_W, th=Math.round(vh*(tw/vw));
      cv.width=tw*N; cv.height=th;
      var ctx=cv.getContext('2d');
      var d=probe.duration||parseFloat(card.dataset.dur)||0, i=0, times=[];
      if(!d||!isFinite(d)){{ return finish(false); }}
      for(var k=0;k<N;k++) times.push(d*(k+0.5)/N);
      function step(){{ if(i>=N) return finish(true); probe.currentTime=times[i]; }}
      probe.addEventListener('seeked',function(){{
        try{{ ctx.drawImage(probe,i*tw,0,tw,th); }}catch(e){{}}
        i++; step();
      }});
      step();
    }});
  }});
  pump();
}}
function strips(b){{
  var on=document.body.classList.toggle('strips');
  b.classList.toggle('on',on);
  if(on) document.querySelectorAll('.c').forEach(function(c){{
    if(c.style.display!=='none') buildStrip(c);
  }});
}}
</script>
"""

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(html)
    (args.out / "sample.json").write_text(json.dumps(picked, indent=2))
    print(f"wrote {args.out/'index.html'} ({len(picked)} clips: {nv}V + {nh}H)")


if __name__ == "__main__":
    main()
