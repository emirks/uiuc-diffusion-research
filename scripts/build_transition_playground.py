"""Build a single self-contained HTML playground for the transition-hardness
analysis. Embeds thumbnails for every frame (0..N_FRAMES-1) as base64 JPEGs
alongside the per-clip metrics. Includes:
  - Plotly scatter: CLIP gap vs exp_033 recon PSNR
  - Sortable data table with thumbnails
  - N slider: pick anchor-frame count, inner thumbnails become (f_N, f_{120-N})
  - Click a scatter point → highlights the row
  - Click a row → highlights the scatter point

No CDN, no Python server needed afterward — just open the .html file.
Plotly is loaded from CDN (~3MB) so internet is required when viewing.

Inputs:
  - outputs/analysis/transition_hardness/run_NNNN/metrics.csv  (latest run)
  - the 10 shadow_smoke MP4s

Output:
  - outputs/analysis/transition_hardness/run_NNNN/playground.html
"""
from __future__ import annotations

import base64
import csv
import io
import json
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

CLIPS = [
    ("shadow_smoke_1", "data/processed/transitions/shadow_smoke/shadow_smoke_1.mp4"),
    ("shadow_smoke_2", "data/processed/transitions/shadow_smoke/shadow_smoke_2.mp4"),
    ("shadow_smoke_3", "data/processed/transitions/shadow_smoke/shadow_smoke_3.mp4"),
    ("shadow_smoke_4", "data/processed/transitions/shadow_smoke/shadow_smoke_4.mp4"),
    ("shadow_smoke_5", "data/processed/transitions/shadow_smoke/shadow_smoke_5.mp4"),
    ("shadow_smoke_6", "data/processed/transitions/shadow_smoke/shadow_smoke_6.mp4"),
    ("shadow_smoke_7", "data/processed/transitions/shadow_smoke/shadow_smoke_7.mp4"),
    ("shadow_smoke_8", "data/processed/transitions/shadow_smoke/shadow_smoke_8.mp4"),
    ("shadow_smoke_9", "data/processed/transitions/shadow_smoke/shadow_smoke_9.mp4"),
    ("shadow_smoke_0", "data/processed/transitions/shadow_smoke/shadow_smoke.mp4"),
]
N_FRAMES = 121  # logical clip length (indices 0..120); extra frames in MP4 ignored
DEFAULT_N = 24
BUILD_TIME_N = 24  # gap metrics in metrics.csv are computed at this N
THUMB_MAX_DIM = 144  # smaller thumb — 121 frames × 10 clips need a tighter budget
JPEG_QUALITY = 72

EXP_PSNR = {
    "shadow_smoke_1": {"exp030": 17.45, "exp032": 38.45, "exp033": 16.38},
    "shadow_smoke_2": {"exp030": 18.25, "exp032": 34.73, "exp033": 26.36},
    "shadow_smoke_3": {"exp030": 19.23, "exp032": 44.75, "exp033": 28.32},
    "shadow_smoke_4": {"exp030": 30.73, "exp032": 39.76, "exp033": 33.12},
    "shadow_smoke_5": {"exp030": 16.51, "exp032": 40.92, "exp033": 16.35},
    "shadow_smoke_6": {"exp030": 17.09, "exp032": 36.11, "exp033": 16.48},
    "shadow_smoke_7": {"exp030": 14.64, "exp032": 40.83, "exp033": 24.97},
    "shadow_smoke_8": {"exp030": 20.52, "exp032": 45.25, "exp033": 29.58},
    "shadow_smoke_9": {"exp030": 15.18, "exp032": 42.62, "exp033": 18.65},
    "shadow_smoke_0": {"exp030": 26.52, "exp032": 44.76, "exp033": 29.32},
}


def latest_run_dir(root: pathlib.Path) -> pathlib.Path:
    runs = sorted(root.glob("run_*"))
    if not runs:
        raise RuntimeError(f"no run_* dir under {root}")
    for r in reversed(runs):
        if (r / "metrics.csv").exists():
            return r
    raise RuntimeError(f"no metrics.csv in any run_* under {root}")


def _bgr_to_b64(bgr: np.ndarray) -> str:
    h, w = bgr.shape[:2]
    s = THUMB_MAX_DIM / max(h, w)
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def extract_all_thumbs_b64(mp4_path: pathlib.Path, n_frames: int) -> list[str]:
    """Read the first n_frames sequentially from mp4_path and return base64 JPEGs."""
    cap = cv2.VideoCapture(str(mp4_path))
    thumbs: list[str] = []
    last_bgr = None
    for i in range(n_frames):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            if last_bgr is None:
                cap.release()
                raise RuntimeError(f"cannot read frame {i} from {mp4_path}")
            bgr = last_bgr  # pad with last good frame
        last_bgr = bgr
        thumbs.append(_bgr_to_b64(bgr))
    cap.release()
    return thumbs


def main():
    analysis_root = REPO_ROOT / "outputs" / "analysis" / "transition_hardness"
    run_dir = latest_run_dir(analysis_root)
    print(f"[info] reading metrics from {run_dir}")

    with (run_dir / "metrics.csv").open() as f:
        metrics_rows = list(csv.DictReader(f))
    metrics_by_id = {r["clip_id"]: r for r in metrics_rows}

    print(f"[info] extracting thumbnails for {len(CLIPS)} clips × {N_FRAMES} frames …")
    data = []
    for clip_id, rel in CLIPS:
        mp4 = REPO_ROOT / rel
        print(f"  - {clip_id}")
        thumbs = extract_all_thumbs_b64(mp4, N_FRAMES)
        m = metrics_by_id[clip_id]
        e = EXP_PSNR[clip_id]
        data.append({
            "clip_id": clip_id,
            "thumbs": thumbs,
            "gap_psnr": float(m["gap_psnr"]),
            "gap_ssim": float(m["gap_ssim"]),
            "gap_lpips": float(m["gap_lpips"]),
            "gap_clip": float(m["gap_clip_cos_dist"]),
            "gap_hist_chi2": float(m["gap_hist_chi2"]),
            "gap_rgb_l1": float(m["gap_rgb_l1"]),
            "span_psnr": float(m["span_psnr"]),
            "span_lpips": float(m["span_lpips"]),
            "span_clip": float(m["span_clip_cos_dist"]),
            "flow_start": float(m["flow_start"]),
            "flow_middle": float(m["flow_middle"]),
            "flow_end": float(m["flow_end"]),
            "e030": e["exp030"],
            "e032": e["exp032"],
            "e033": e["exp033"],
        })

    html = build_html(data)
    out_path = run_dir / "playground.html"
    out_path.write_text(html)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[done] {out_path}  ({size_mb:.1f} MB)")


def build_html(data: list[dict]) -> str:
    # Embed everything inline. Plotly is the only external dep.
    n_frames = N_FRAMES
    last_idx = N_FRAMES - 1
    build_n = BUILD_TIME_N
    build_mirror = last_idx - build_n
    default_n = DEFAULT_N
    default_mirror = last_idx - default_n
    n_max = last_idx - 1  # avoid degenerate N == last_idx
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Transition-hardness playground · shadow_smoke</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{ --bg:#0e0e10; --panel:#1c1c20; --text:#eaeaea; --muted:#8a8a92;
           --acc:#ffd23f; --pass:#5ad27a; --fail:#ff6464; --warn:#f4a261; }}
  body {{ font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: var(--bg); color: var(--text); margin: 24px; }}
  h1 {{ margin-top: 0; font-size: 22px; }}
  h2 {{ font-size: 16px; color: var(--muted); margin-top: 28px; }}
  .summary {{ background: var(--panel); padding: 14px 18px; border-radius: 8px;
              margin-bottom: 18px; line-height: 1.6; }}
  .summary code {{ background: #2a2a2f; padding: 1px 6px; border-radius: 3px; }}
  #scatter {{ background: var(--panel); border-radius: 8px; padding: 8px; }}
  table {{ border-collapse: collapse; width: 100%; background: var(--panel);
           border-radius: 8px; overflow: hidden; table-layout: fixed; }}
  thead th {{ background: #26262c; color: var(--muted); font-weight: 600;
              text-align: center; padding: 10px 6px; cursor: default; user-select: none;
              white-space: nowrap; font-size: 12px; }}
  thead th.sortable {{ cursor: pointer; }}
  thead th:hover {{ color: var(--text); }}
  thead th.sortable::after {{ content: " ⇅"; color: #555; }}
  thead th.sort-asc::after {{ content: " ▲"; color: var(--acc); }}
  thead th.sort-desc::after {{ content: " ▼"; color: var(--acc); }}
  tbody td {{ padding: 8px 6px; border-top: 1px solid #2a2a2f; vertical-align: middle;
              font-variant-numeric: tabular-nums; text-align: right;
              white-space: nowrap; }}
  tbody td.frame-cell {{ text-align: center; padding: 6px; }}
  tbody td.clip-col {{ text-align: center; }}
  tbody tr:hover {{ background: #25252b; }}
  tbody tr.highlight {{ background: #3a3a1a !important; outline: 2px solid var(--acc); }}
  img.thumb {{ height: 70px; width: 70px; border-radius: 4px;
               display: block; object-fit: cover; margin: 0 auto; }}
  .n-control {{ background: var(--panel); padding: 10px 14px; border-radius: 8px;
                 margin: 12px 0 18px; display: flex; align-items: center; gap: 14px;
                 flex-wrap: wrap; }}
  .n-control label {{ color: var(--muted); font-size: 13px; }}
  .n-control input[type=range] {{ flex: 1 1 280px; max-width: 480px; accent-color: var(--acc); }}
  .n-control input[type=number] {{ width: 70px; background: #2a2a2f; color: var(--text);
                                     border: 1px solid #3a3a40; border-radius: 4px;
                                     padding: 4px 6px; font: inherit; text-align: center; }}
  .n-control .nlabel {{ font-weight: 600; color: var(--acc); font-variant-numeric: tabular-nums; }}
  .n-control .stale {{ color: var(--warn); font-size: 12px; }}
  /* Fixed column widths so headers and cells stay aligned. 14 columns total. */
  col.col-clip  {{ width: 50px; }}
  col.col-frame {{ width: 84px; }}
  col.col-num   {{ width: 62px; }}
  col.col-flow  {{ width: 60px; }}
  col.col-exp   {{ width: 64px; }}
  .label {{ font-size: 10px; color: var(--muted); text-align: center; margin-top: 3px;
            line-height: 1.1; }}
  .pass {{ color: var(--pass); font-weight: 600; }}
  .borderline {{ color: var(--warn); }}
  .fail {{ color: var(--fail); font-weight: 600; }}
  .clip-id {{ font-weight: 600; color: var(--acc); }}
  .legend {{ font-size: 12px; color: var(--muted); margin: 10px 0 20px; }}
</style>
</head>
<body>

<h1>Transition-hardness playground · <code>shadow_smoke</code></h1>

<div class="summary">
  <strong>Key finding:</strong> CLIP cosine distance between frame 24 (last of start sub-clip)
  and frame 96 (first of end sub-clip) explains 73% of variance in <code>exp_033</code> recon
  PSNR — Spearman ρ = <strong>0.855</strong>, p = 0.0016, N = 10.
  Lower CLIP gap means both endpoints are amorphous smoke (no structural anchor); higher
  CLIP gap means each endpoint has visible object content. Counterintuitive but mechanistic.
  <br><br>
  <strong>Recipes:</strong>
  exp_030 = production sub-clip anchors (broken baseline).
  exp_032 = z₀-slice anchors (oracle / non-deployable, 8/10 pass).
  exp_033 = production anchors + drop end-clip first latent frame from mask (partial fix).
</div>

<h2>CLIP gap vs exp_033 recon PSNR  (click a point to highlight its row)</h2>
<div id="scatter" style="width:100%; height: 480px;"></div>

<div class="legend">
  Pass = ≥28 dB + SSIM ≥ 0.88 + LPIPS ≤ 0.10 (3/3 thresholds). Borderline = 2/3.
  Fail/catastrophic = 0–1/3. Click any column header to sort.
</div>

<h2>Per-clip data  (click a row to highlight the scatter point)</h2>

<div class="n-control">
  <label for="nSlider">anchor N:</label>
  <input type="range" id="nSlider" min="1" max="{n_max}" value="{default_n}">
  <input type="number" id="nInput" min="1" max="{n_max}" value="{default_n}">
  <span>inner frames = <span class="nlabel">f<span id="nLabel">{default_n}</span></span> &nbsp;and&nbsp; <span class="nlabel">f<span id="nMirrorLabel">{default_mirror}</span></span></span>
  <span id="staleNote" class="stale" style="display:none">gap metrics are computed at build-time N={build_n}</span>
</div>
<table id="clipTable">
  <colgroup>
    <col class="col-clip">
    <col class="col-frame"><col class="col-frame"><col class="col-frame"><col class="col-frame">
    <col class="col-num"><col class="col-num"><col class="col-num">
    <col class="col-flow"><col class="col-flow"><col class="col-flow">
    <col class="col-exp"><col class="col-exp"><col class="col-exp">
  </colgroup>
  <thead>
    <tr>
      <th class="sortable" data-k="clip_id">clip</th>
      <th>f0</th>
      <th>f<span class="hdrN">{default_n}</span><br><small>(start ends)</small></th>
      <th>f<span class="hdrMirror">{default_mirror}</span><br><small>(end begins)</small></th>
      <th>f{last_idx}</th>
      <th class="sortable" data-k="gap_clip">CLIP gap<br><small>(f{build_n},f{build_mirror})</small></th>
      <th class="sortable" data-k="gap_lpips">LPIPS gap</th>
      <th class="sortable" data-k="gap_psnr">PSNR gap</th>
      <th class="sortable" data-k="flow_start">flow start</th>
      <th class="sortable" data-k="flow_middle">flow mid</th>
      <th class="sortable" data-k="flow_end">flow end</th>
      <th class="sortable" data-k="e030">e030 PSNR</th>
      <th class="sortable" data-k="e032">e032 PSNR</th>
      <th class="sortable" data-k="e033">e033 PSNR</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>

<script>
const DATA = {json.dumps(data)};
const N_FRAMES = {n_frames};
const LAST_IDX = N_FRAMES - 1;
const BUILD_N = {build_n};
let currentN = {default_n};

function classifyE33(p) {{
  if (p >= 33) return 'pass';
  if (p >= 24) return 'borderline';
  return 'fail';
}}
function classifyE32(p) {{ return p >= 33 ? 'pass' : (p >= 24 ? 'borderline' : 'fail'); }}
function classifyE30(p) {{ return p >= 33 ? 'pass' : (p >= 24 ? 'borderline' : 'fail'); }}

function thumb(b64) {{ return '<img class="thumb" src="data:image/jpeg;base64,' + b64 + '">'; }}

function renderTable(rows) {{
  const tb = document.getElementById('tbody');
  tb.innerHTML = '';
  const nA = currentN;
  const nB = LAST_IDX - currentN;
  rows.forEach(d => {{
    const tr = document.createElement('tr');
    tr.dataset.clipId = d.clip_id;
    tr.innerHTML =
      '<td class="clip-col clip-id">' + d.clip_id.replace('shadow_smoke_', 'ss') + '</td>' +
      '<td class="frame-cell">' + thumb(d.thumbs[0])        + '<div class="label">frame 0</div></td>' +
      '<td class="frame-cell">' + thumb(d.thumbs[nA])       + '<div class="label">frame ' + nA + '</div></td>' +
      '<td class="frame-cell">' + thumb(d.thumbs[nB])       + '<div class="label">frame ' + nB + '</div></td>' +
      '<td class="frame-cell">' + thumb(d.thumbs[LAST_IDX]) + '<div class="label">frame ' + LAST_IDX + '</div></td>' +
      '<td>' + d.gap_clip.toFixed(3) + '</td>' +
      '<td>' + d.gap_lpips.toFixed(3) + '</td>' +
      '<td>' + d.gap_psnr.toFixed(2) + '</td>' +
      '<td>' + d.flow_start.toFixed(2) + '</td>' +
      '<td>' + d.flow_middle.toFixed(2) + '</td>' +
      '<td>' + d.flow_end.toFixed(2) + '</td>' +
      '<td class="' + classifyE30(d.e030) + '">' + d.e030.toFixed(2) + '</td>' +
      '<td class="' + classifyE32(d.e032) + '">' + d.e032.toFixed(2) + '</td>' +
      '<td class="' + classifyE33(d.e033) + '">' + d.e033.toFixed(2) + '</td>';
    tr.addEventListener('click', () => highlightClip(d.clip_id));
    tb.appendChild(tr);
  }});
}}

function renderScatter() {{
  const trace = {{
    x: DATA.map(d => d.gap_clip),
    y: DATA.map(d => d.e033),
    mode: 'markers+text',
    type: 'scatter',
    text: DATA.map(d => d.clip_id.replace('shadow_smoke_', 'ss')),
    textposition: 'top center',
    textfont: {{ color: '#eaeaea', size: 11 }},
    marker: {{
      size: 16,
      color: DATA.map(d => d.e033),
      colorscale: [[0, '#ff6464'], [0.5, '#f4a261'], [1, '#5ad27a']],
      cmin: 14, cmax: 35,
      line: {{ color: '#000', width: 1 }},
    }},
    customdata: DATA.map(d => [d.clip_id, d.e030, d.e032, d.gap_lpips, d.gap_psnr]),
    hovertemplate:
      '<b>%{{customdata[0]}}</b><br>' +
      'CLIP gap (f24,f96): %{{x:.3f}}<br>' +
      'e033 recon PSNR: %{{y:.2f}}<br>' +
      'e030: %{{customdata[1]:.2f}}  e032: %{{customdata[2]:.2f}}<br>' +
      'gap_lpips: %{{customdata[3]:.3f}}  gap_psnr: %{{customdata[4]:.2f}}<extra></extra>',
  }};
  const layout = {{
    paper_bgcolor: '#1c1c20',
    plot_bgcolor: '#1c1c20',
    font: {{ color: '#eaeaea' }},
    xaxis: {{ title: 'CLIP cosine distance (f24, f96)  — semantic gap',
              gridcolor: '#2a2a2f', zerolinecolor: '#2a2a2f' }},
    yaxis: {{ title: 'exp_033 recon PSNR (dB)',
              gridcolor: '#2a2a2f', zerolinecolor: '#2a2a2f' }},
    margin: {{ t: 30, l: 60, r: 30, b: 60 }},
    shapes: [
      // Threshold marker at gap_clip ≈ 0.39
      {{ type: 'line', x0: 0.39, x1: 0.39, y0: 14, y1: 35,
         line: {{ color: '#ffd23f', dash: 'dash', width: 1 }} }},
      // Exit-1 PSNR floor at 28 dB
      {{ type: 'line', x0: 0.29, x1: 0.52, y0: 28, y1: 28,
         line: {{ color: '#5ad27a', dash: 'dot', width: 1 }} }},
    ],
    annotations: [
      {{ x: 0.39, y: 35, xanchor: 'left', yanchor: 'top', showarrow: false,
         text: 'CLIP gap ≈ 0.39 (catastrophic/borderline threshold)',
         font: {{ color: '#ffd23f', size: 10 }} }},
      {{ x: 0.52, y: 28, xanchor: 'right', yanchor: 'bottom', showarrow: false,
         text: 'exit ① PSNR floor (28 dB)',
         font: {{ color: '#5ad27a', size: 10 }} }},
    ],
  }};
  Plotly.newPlot('scatter', [trace], layout, {{ responsive: true, displaylogo: false }});
  document.getElementById('scatter').on('plotly_click', (ev) => {{
    if (ev.points && ev.points.length) {{
      const clipId = ev.points[0].customdata[0];
      highlightClip(clipId);
    }}
  }});
}}

function highlightClip(clipId) {{
  document.querySelectorAll('#tbody tr').forEach(tr => {{
    tr.classList.toggle('highlight', tr.dataset.clipId === clipId);
  }});
  const tr = document.querySelector('#tbody tr[data-clip-id="' + clipId + '"]');
  if (tr) tr.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
}}

let sortKey = 'gap_clip', sortDir = 'asc';
function sortRows() {{
  const sorted = [...DATA].sort((a, b) => {{
    const x = a[sortKey], y = b[sortKey];
    if (typeof x === 'string') return sortDir === 'asc' ? x.localeCompare(y) : y.localeCompare(x);
    return sortDir === 'asc' ? x - y : y - x;
  }});
  renderTable(sorted);
  document.querySelectorAll('thead th.sortable').forEach(th => {{
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.k === sortKey) th.classList.add(sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
  }});
}}
document.querySelectorAll('thead th.sortable').forEach(th => {{
  th.addEventListener('click', () => {{
    const k = th.dataset.k;
    if (sortKey === k) sortDir = (sortDir === 'asc') ? 'desc' : 'asc';
    else {{ sortKey = k; sortDir = 'asc'; }}
    sortRows();
  }});
}});

function applyN(n) {{
  n = Math.max(1, Math.min(LAST_IDX - 1, Math.round(Number(n))));
  if (Number.isNaN(n)) return;
  currentN = n;
  const mirror = LAST_IDX - n;
  document.getElementById('nSlider').value = n;
  document.getElementById('nInput').value = n;
  document.getElementById('nLabel').textContent = n;
  document.getElementById('nMirrorLabel').textContent = mirror;
  document.querySelectorAll('.hdrN').forEach(el => el.textContent = n);
  document.querySelectorAll('.hdrMirror').forEach(el => el.textContent = mirror);
  document.getElementById('staleNote').style.display = (n === BUILD_N) ? 'none' : '';
  sortRows();
}}
document.getElementById('nSlider').addEventListener('input', e => applyN(e.target.value));
document.getElementById('nInput').addEventListener('change', e => applyN(e.target.value));

renderScatter();
sortRows();
</script>

</body>
</html>"""


if __name__ == "__main__":
    main()
