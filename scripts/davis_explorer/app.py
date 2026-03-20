"""
DAVIS Dataset Explorer
======================
Interactive Gradio app for browsing, viewing and analysing the DAVIS 2017
video object segmentation dataset (480p split).

Usage (from repo root):
    python scripts/davis_explorer/app.py

    # Custom DAVIS root:
    DAVIS_ROOT=/path/to/DAVIS python scripts/davis_explorer/app.py

    # Public link:
    python scripts/davis_explorer/app.py --share

Dataset layout expected:
    <DAVIS_ROOT>/
        JPEGImages/480p/<sequence>/%05d.jpg
        Annotations/480p/<sequence>/%05d.png
        ImageSets/2016/{train,val}.txt
        ImageSets/2017/{train,val}.txt
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────

# Official ETH Zurich download — DAVIS 2017 trainval 480p (~800 MB zipped).
# The zip extracts to a top-level DAVIS/ directory.
DAVIS_ZIP_URL = (
    "https://data.vision.ee.ethz.ch/csergi/share/davis/"
    "DAVIS-2017-trainval-480p.zip"
)

IS_HF_SPACE = bool(os.environ.get("SPACE_ID"))

# Path resolution:
#   • HF Spaces with persistent storage → /data/DAVIS
#   • HF Spaces without persistent storage → /tmp/DAVIS
#   • Local                              → workspace path (or DAVIS_ROOT env var)
if IS_HF_SPACE:
    _hf_base    = Path("/data") if Path("/data").exists() else Path("/tmp")
    _local_root = _hf_base / "DAVIS"
else:
    _local_root = Path("/workspace/diffusion-research/data/raw/DAVIS")

DAVIS_ROOT = Path(os.environ.get("DAVIS_ROOT", str(_local_root)))

IMG_DIR  = DAVIS_ROOT / "JPEGImages" / "480p"
ANN_DIR  = DAVIS_ROOT / "Annotations" / "480p"
SETS_DIR = DAVIS_ROOT / "ImageSets"

# Cache lives as a sibling of DAVIS_ROOT so the path is always valid.
CACHE_DIR = Path(os.environ.get(
    "DAVIS_CACHE_DIR",
    str(DAVIS_ROOT.parent / "DAVIS_explorer_cache"),
))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DAVIS_PALETTE = np.array([
    [  0,   0,   0], [128,   0,   0], [  0, 128,   0], [128, 128,   0],
    [  0,   0, 128], [128,   0, 128], [  0, 128, 128], [128, 128, 128],
    [ 64,   0,   0], [192,   0,   0], [ 64, 128,   0], [192, 128,   0],
    [ 64,   0, 128], [192,   0, 128], [ 64, 128, 128], [192, 128, 128],
    [  0,  64,   0], [128,  64,   0], [  0, 192,   0], [128, 192,   0],
], dtype=np.uint8)

DEFAULT_FPS   = 24
DEFAULT_ALPHA = 0.55
DEFAULT_CRF   = 18
MAX_COMPARE   = 6      # slots in the Compare tab
PAGE_SIZE     = 9      # videos per page in Multi-Video tab
THUMB_W, THUMB_H = 320, 200   # thumbnail dimensions for Gallery

# ── Dataset download ───────────────────────────────────────────────────────────

def ensure_dataset() -> None:
    """Download and extract DAVIS 2017 trainval (480p) if not already present.

    Safe to call every startup — exits immediately when data is found.
    The zip extracts into a top-level ``DAVIS/`` directory, so we extract
    into ``DAVIS_ROOT.parent`` which gives the expected ``DAVIS_ROOT`` layout.
    """
    if IMG_DIR.exists() and any(IMG_DIR.iterdir()):
        return   # data already present

    import urllib.request
    import zipfile

    DAVIS_ROOT.mkdir(parents=True, exist_ok=True)
    zip_dst = DAVIS_ROOT.parent / "_davis_download.zip"

    print(f"DAVIS dataset not found at {DAVIS_ROOT}")
    print(f"Downloading {DAVIS_ZIP_URL}  (~800 MB) …")

    _last_pct: list[int] = [-1]

    def _progress(count: int, block: int, total: int) -> None:
        pct = min(100, int(count * block / total * 100))
        if pct != _last_pct[0] and pct % 5 == 0:
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"  [{bar}] {pct:3d}%", end="\r", flush=True)
            _last_pct[0] = pct

    try:
        urllib.request.urlretrieve(DAVIS_ZIP_URL, zip_dst, _progress)
    except Exception as exc:
        zip_dst.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {exc}") from exc

    print(f"\n  Download complete ({zip_dst.stat().st_size // 1_048_576} MB). Extracting…")

    with zipfile.ZipFile(zip_dst, "r") as zf:
        zf.extractall(DAVIS_ROOT.parent)

    zip_dst.unlink(missing_ok=True)

    if not IMG_DIR.exists():
        raise RuntimeError(
            f"Extraction failed — expected {IMG_DIR} not found. "
            "Check that the zip contains a top-level DAVIS/ directory."
        )
    print(f"  DAVIS dataset ready at {DAVIS_ROOT}")


# ── Dataset loading ────────────────────────────────────────────────────────────

def _read_split(year: str, split: str) -> list[str]:
    p = SETS_DIR / year / f"{split}.txt"
    return p.read_text().strip().splitlines() if p.exists() else []


def _count_objects(seq: str) -> int:
    ann_seq = ANN_DIR / seq
    if not ann_seq.exists():
        return 0
    files = sorted(ann_seq.iterdir())
    return int(np.max(np.array(Image.open(files[0])))) if files else 0


def build_dataframe() -> pd.DataFrame:
    seqs      = sorted(d.name for d in IMG_DIR.iterdir() if d.is_dir())
    s16_train = set(_read_split("2016", "train"))
    s16_val   = set(_read_split("2016", "val"))
    s17_train = set(_read_split("2017", "train"))
    s17_val   = set(_read_split("2017", "val"))
    rows = []
    for seq in seqs:
        imgs    = sorted((IMG_DIR / seq).glob("*.jpg"))
        n       = len(imgs)
        n_obj   = _count_objects(seq)
        w, h    = Image.open(imgs[0]).size if imgs else (0, 0)
        in16t, in16v = seq in s16_train, seq in s16_val
        in17t, in17v = seq in s17_train, seq in s17_val
        splits = (["2016-train"] * in16t + ["2016-val"] * in16v +
                  ["2017-train"] * in17t + ["2017-val"] * in17v)
        rows.append({
            "sequence": seq, "frames": n, "n_objects": n_obj,
            "width": w, "height": h, "resolution": f"{w}×{h}",
            "split": ", ".join(splits) or "unlisted",
            "in_2016": in16t or in16v, "in_2017": in17t or in17v,
            "in_train": in16t or in17t, "in_val": in16v or in17v,
        })
    return pd.DataFrame(rows)


ensure_dataset()
print("Loading DAVIS metadata…")
DF = build_dataframe()
ALL_SEQUENCES = sorted(DF["sequence"].tolist())
print(f"  {len(DF)} sequences  ·  frames {DF['frames'].min()}–{DF['frames'].max()}  "
      f"·  objects {DF['n_objects'].min()}–{DF['n_objects'].max()}")

DISPLAY_COLS = ["sequence", "frames", "n_objects", "resolution", "split"]

# ── Frame helpers ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=16)
def _get_frame_paths(seq: str) -> list[Path]:
    return sorted((IMG_DIR / seq).glob("*.jpg"))


@lru_cache(maxsize=16)
def _get_ann_paths(seq: str) -> list[Path]:
    d = ANN_DIR / seq
    return sorted(d.glob("*.png")) if d.exists() else []


def _blend(img_f32: np.ndarray, ann: np.ndarray, alpha: float) -> np.ndarray:
    ov  = DAVIS_PALETTE[np.clip(ann, 0, len(DAVIS_PALETTE) - 1)].astype(np.float32)
    a   = np.where(ann == 0, 0.0, alpha).astype(np.float32)[:, :, None]
    return (img_f32 * (1 - a) + ov * a).clip(0, 255).astype(np.uint8)


def render_frame(seq: str, idx: int, overlay: bool, alpha: float) -> Image.Image:
    fps = _get_frame_paths(seq)
    if not fps:
        return Image.new("RGB", (854, 480), 20)
    idx = min(max(0, idx), len(fps) - 1)
    arr = np.array(Image.open(fps[idx]).convert("RGB"), dtype=np.float32)
    if overlay:
        anns = _get_ann_paths(seq)
        if idx < len(anns):
            arr = _blend(arr, np.array(Image.open(anns[idx])), alpha).astype(np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def render_mask(seq: str, idx: int) -> Image.Image:
    anns = _get_ann_paths(seq)
    if not anns:
        return Image.new("RGB", (854, 480), 20)
    idx = min(max(0, idx), len(anns) - 1)
    ann = np.array(Image.open(anns[idx]))
    rgb = np.zeros((*ann.shape, 3), dtype=np.uint8)
    for oid in range(1, len(DAVIS_PALETTE)):
        m = ann == oid
        if m.any():
            rgb[m] = DAVIS_PALETTE[oid]
    return Image.fromarray(rgb)


# ── MP4 helpers ────────────────────────────────────────────────────────────────

def _mp4_path(seq: str, overlay: bool, alpha: float, fps: int) -> Path:
    tag = f"ov{int(alpha * 100):03d}" if overlay else "raw"
    return CACHE_DIR / f"{seq}_{tag}_{fps}fps.mp4"


def _ffmpeg(pattern: str, out: Path, fps: int) -> None:
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
           "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
           "-crf", str(DEFAULT_CRF), "-movflags", "+faststart",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-600:])


def encode_sequence(seq: str, overlay: bool, alpha: float, fps: int) -> Path:
    out = _mp4_path(seq, overlay, round(alpha, 2), fps)
    if out.exists():
        return out
    fps_paths = _get_frame_paths(seq)
    if not fps_paths:
        raise FileNotFoundError(f"No frames for {seq}")
    if not overlay:
        _ffmpeg(str(IMG_DIR / seq / "%05d.jpg"), out, fps)
        return out
    anns    = _get_ann_paths(seq)
    tmp     = CACHE_DIR / f"_tmp_{seq}_{int(alpha*100):03d}"
    tmp.mkdir(exist_ok=True)
    try:
        for i, fp in enumerate(fps_paths):
            arr = np.array(Image.open(fp).convert("RGB"), dtype=np.float32)
            if i < len(anns):
                arr = _blend(arr, np.array(Image.open(anns[i])), alpha).astype(np.float32)
            Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(
                tmp / f"{i:05d}.png", optimize=False)
        _ffmpeg(str(tmp / "%05d.png"), out, fps)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def get_video(seq: str, overlay: bool, alpha: float, fps: int) -> tuple[str | None, str]:
    if not seq or seq not in ALL_SEQUENCES:
        return None, "No sequence selected."
    try:
        p    = encode_sequence(seq, overlay, round(alpha, 2), fps)
        n    = int(DF[DF["sequence"] == seq].iloc[0]["frames"])
        size = p.stat().st_size // 1024
        mode = "overlay" if overlay else "raw"
        return str(p), f"✅ **{seq}** · {n} frames · {fps} fps · {mode} · {size} KB"
    except Exception as e:
        return None, f"❌ {e}"


# ── Background pre-cache ───────────────────────────────────────────────────────

_cache_progress: dict[str, str] = {}
_cache_lock = threading.Lock()


def _precache_worker(seq: str, fps: int) -> None:
    with _cache_lock:
        _cache_progress[seq] = "encoding…"
    try:
        encode_sequence(seq, False, DEFAULT_ALPHA, fps)
        encode_sequence(seq, True,  DEFAULT_ALPHA, fps)
        with _cache_lock:
            _cache_progress[seq] = "done"
    except Exception as e:
        with _cache_lock:
            _cache_progress[seq] = f"error: {e}"


def start_precache(fps: int = DEFAULT_FPS, workers: int = 4) -> None:
    missing = [s for s in ALL_SEQUENCES
               if not _mp4_path(s, False, DEFAULT_ALPHA, fps).exists()
               or not _mp4_path(s, True,  DEFAULT_ALPHA, fps).exists()]
    if not missing:
        print(f"  MP4 cache complete ({len(ALL_SEQUENCES)}×2 already exist)")
        for s in ALL_SEQUENCES:
            _cache_progress[s] = "done"
        return
    print(f"  Pre-caching {len(missing)} sequences (workers={workers})…")
    def _run():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_precache_worker, s, fps): s for s in missing}
            done = 0
            for f in as_completed(futs):
                done += 1
                s = futs[f]
                if done % 10 == 0 or done == len(missing):
                    print(f"  Cache {done}/{len(missing)} ({s}: {_cache_progress.get(s)})")
    threading.Thread(target=_run, daemon=True).start()


# ── Gallery helpers ────────────────────────────────────────────────────────────

def _make_thumb(seq: str, overlay: bool = False, alpha: float = 0.0) -> Image.Image:
    fps = _get_frame_paths(seq)
    if not fps:
        return Image.new("RGB", (THUMB_W, THUMB_H), 30)
    img = render_frame(seq, 0, overlay, alpha) if overlay else Image.open(fps[0]).convert("RGB")
    img = img.copy()
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    return img


def build_gallery_items(seqs: list[str], overlay: bool = False) -> list[tuple]:
    items = []
    for seq in seqs:
        row = DF[DF["sequence"] == seq].iloc[0]
        caption = f"{seq}  [{row['frames']}f · {row['n_objects']}obj]"
        items.append((_make_thumb(seq, overlay), caption))
    return items


print("Building gallery thumbnails…")
_ALL_THUMBS: list[tuple] = build_gallery_items(ALL_SEQUENCES)
print("  Done.")


# ── Filter helpers ─────────────────────────────────────────────────────────────

def filter_df(year_f, split_f, obj_f, fmin, fmax, search) -> pd.DataFrame:
    d = DF.copy()
    if year_f  == "2016 only":    d = d[d["in_2016"]]
    elif year_f == "2017 only":   d = d[d["in_2017"]]
    if split_f == "Train only":   d = d[d["in_train"]]
    elif split_f == "Val only":   d = d[d["in_val"]]
    if obj_f   == "1 object":     d = d[d["n_objects"] == 1]
    elif obj_f == "2 objects":    d = d[d["n_objects"] == 2]
    elif obj_f == "3+ objects":   d = d[d["n_objects"] >= 3]
    d = d[(d["frames"] >= fmin) & (d["frames"] <= fmax)]
    if search.strip():
        d = d[d["sequence"].str.lower().str.contains(search.strip().lower(), na=False)]
    return d[DISPLAY_COLS].reset_index(drop=True)


def _seq_info(seq: str) -> str:
    if seq not in ALL_SEQUENCES:
        return ""
    r = DF[DF["sequence"] == seq].iloc[0]
    return (f"**{seq}** — {r['frames']} frames · {r['n_objects']} obj · "
            f"{r['resolution']} · _{r['split']}_")


def get_legend(seq: str) -> str:
    if seq not in ALL_SEQUENCES:
        return ""
    n = int(DF[DF["sequence"] == seq].iloc[0]["n_objects"])
    if n == 0:
        return "*No annotated objects.*"
    lines = ["**Objects:**"]
    for i in range(1, min(n + 1, len(DAVIS_PALETTE))):
        hx = "#{:02X}{:02X}{:02X}".format(*DAVIS_PALETTE[i])
        lines.append(f"- <span style='color:{hx};font-weight:bold'>■</span> Object {i}")
    return "\n".join(lines)


def cache_status_md() -> str:
    with _cache_lock:
        done  = sum(1 for v in _cache_progress.values() if v == "done")
    total = len(ALL_SEQUENCES)
    pct   = done / total * 100 if total else 0
    bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    return f"`[{bar}]` **{done}/{total}** cached ({pct:.0f}%)"


# ── Stats plots ────────────────────────────────────────────────────────────────

def make_stats_plots():
    d = DF.copy()
    fig_frames = px.histogram(d, x="frames", nbins=30, title="Frame Count Distribution",
        color_discrete_sequence=["#3B82F6"], labels={"frames": "Frames"})
    fig_frames.update_layout(margin=dict(t=45, b=40))

    oc = d["n_objects"].value_counts().sort_index().reset_index()
    oc.columns = ["n_objects", "count"]
    fig_objs = px.bar(oc, x="n_objects", y="count", title="Sequences by Object Count",
        color="count", color_continuous_scale="Teal",
        labels={"n_objects": "Objects", "count": "# Sequences"})
    fig_objs.update_layout(coloraxis_showscale=False, margin=dict(t=45, b=40))
    fig_objs.update_xaxes(tickmode="linear", dtick=1)

    sp = {"2016-train": int(d["split"].str.contains("2016-train").sum()),
          "2016-val":   int(d["split"].str.contains("2016-val").sum()),
          "2017-train": int(d["split"].str.contains("2017-train").sum()),
          "2017-val":   int(d["split"].str.contains("2017-val").sum())}
    fig_splits = px.bar(x=list(sp.keys()), y=list(sp.values()), title="Sequences per Split",
        color=list(sp.keys()),
        color_discrete_sequence=["#3B82F6","#6366F1","#F59E0B","#EF4444"],
        labels={"x": "Split", "y": "# Sequences"})
    fig_splits.update_layout(showlegend=False, margin=dict(t=45, b=40))

    rc = d["resolution"].value_counts().reset_index()
    rc.columns = ["resolution", "count"]
    fig_res = px.pie(rc, names="resolution", values="count", title="Resolution Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_res.update_layout(margin=dict(t=45, b=20))

    fig_scatter = px.scatter(d, x="frames", y="n_objects", text="sequence",
        title="Frames vs. Object Count",
        color="n_objects", color_continuous_scale="Viridis",
        size="frames", size_max=18,
        labels={"frames": "Frames", "n_objects": "Objects"},
        hover_data=["sequence", "frames", "n_objects", "resolution", "split"])
    fig_scatter.update_traces(textposition="top center", textfont_size=8)
    fig_scatter.update_layout(coloraxis_showscale=False, margin=dict(t=45, b=40))

    return fig_frames, fig_objs, fig_splits, fig_res, fig_scatter


# ── Build UI ───────────────────────────────────────────────────────────────────

def build_ui():
    figs = make_stats_plots()
    n_multi = int((DF["n_objects"] > 1).sum())
    n_2016  = int(DF["in_2016"].sum())
    n_2017  = int(DF["in_2017"].sum())
    _first  = ALL_SEQUENCES[0]
    _first_n = len(_get_frame_paths(_first))
    total_pages = (len(ALL_SEQUENCES) + PAGE_SIZE - 1) // PAGE_SIZE

    with gr.Blocks(title="DAVIS Dataset Explorer") as demo:

        gr.Markdown(
            "# 🎬 DAVIS Dataset Explorer\n"
            f"**DAVIS 2017 · 480p** — {len(DF)} sequences · "
            f"frames {DF['frames'].min()}–{DF['frames'].max()} · "
            f"{n_2016} in DAVIS-2016 · {n_2017} in DAVIS-2017 · "
            f"{n_multi} multi-object"
        )

        with gr.Tabs():

            # ──────────────────────────────────────────────────────────────
            # Tab 1 · Browse
            # ──────────────────────────────────────────────────────────────
            with gr.TabItem("📋 Browse"):
                with gr.Row():
                    dd_year   = gr.Dropdown(["All years","2016 only","2017 only"],
                                            value="All years", label="Year", scale=1)
                    dd_split  = gr.Dropdown(["All splits","Train only","Val only"],
                                            value="All splits", label="Split", scale=1)
                    dd_obj    = gr.Dropdown(["Any # objects","1 object","2 objects","3+ objects"],
                                            value="Any # objects", label="Objects", scale=1)
                    txt_srch  = gr.Textbox(placeholder="Search…", label="Search", scale=2)
                with gr.Row():
                    fmin_sl = gr.Slider(int(DF["frames"].min()), int(DF["frames"].max()),
                                        int(DF["frames"].min()), step=1, label="Min frames", scale=3)
                    fmax_sl = gr.Slider(int(DF["frames"].min()), int(DF["frames"].max()),
                                        int(DF["frames"].max()), step=1, label="Max frames", scale=3)
                count_md = gr.Markdown(f"**{len(DF)} sequences** match.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        tbl = gr.DataFrame(value=DF[DISPLAY_COLS], interactive=False, wrap=False)
                    with gr.Column(scale=2):
                        detail_md = gr.Markdown("*Select a row to see details.*")

                filtered_state = gr.State(DF[DISPLAY_COLS].copy())
                selected_seq   = gr.State("")
                f_inputs = [dd_year, dd_split, dd_obj, fmin_sl, fmax_sl, txt_srch]

                def _on_filter(*a):
                    df = filter_df(*a)
                    return df, f"**{len(df)} sequences** match."
                for inp in f_inputs:
                    inp.change(_on_filter, f_inputs, [tbl, count_md])
                    inp.change(lambda *a: filter_df(*a), f_inputs, filtered_state)

                def _on_row(evt: gr.SelectData, fdf):
                    if evt is None or fdf is None or len(fdf) == 0:
                        return gr.update(), "Select a row."
                    seq = fdf.iloc[evt.index[0]]["sequence"]
                    r   = DF[DF["sequence"] == seq].iloc[0]
                    sc  = r["split"].replace(", ", "\n• ")
                    md  = (f"### `{seq}`\n| Field | Value |\n|---|---|\n"
                           f"| Frames | **{r['frames']}** |\n"
                           f"| Objects | **{r['n_objects']}** |\n"
                           f"| Resolution | {r['resolution']} |\n"
                           f"| Splits | • {sc} |\n\n"
                           f"> Open the **Viewer** or **Gallery** tab to watch.")
                    return seq, md
                tbl.select(_on_row, filtered_state, [selected_seq, detail_md])

            # ──────────────────────────────────────────────────────────────
            # Tab 2 · Viewer  (frame scrubber + single video)
            # ──────────────────────────────────────────────────────────────
            with gr.TabItem("🔍 Viewer"):
                with gr.Row():
                    seq_dd = gr.Dropdown(ALL_SEQUENCES, value=_first,
                                         label="Sequence", scale=5)
                seq_info_md = gr.Markdown(_seq_info(_first))

                gr.Markdown("#### Frame Scrubber")
                with gr.Row():
                    ov_cb    = gr.Checkbox(value=True, label="Mask overlay")
                    alpha_sl = gr.Slider(0.1, 1.0, DEFAULT_ALPHA, step=0.05,
                                          label="Overlay opacity")
                frame_sl = gr.Slider(0, _first_n - 1, 0, step=1,
                                      label=f"Frame  (0 – {_first_n - 1})")
                with gr.Row():
                    img_out = gr.Image(label="Frame (+overlay)", type="pil", height=360,
                                       value=render_frame(_first, 0, True, DEFAULT_ALPHA))
                    ann_out = gr.Image(label="Annotation mask",  type="pil", height=360,
                                       value=render_mask(_first, 0))
                legend_md = gr.Markdown(get_legend(_first))

                gr.Markdown("---\n#### Video Playback")
                gr.Markdown(
                    "Raw encodes directly from JPEGs (instant). "
                    "Overlay uses vectorised numpy. Both variants are cached permanently."
                )
                with gr.Row():
                    v_fps = gr.Slider(1, 30, DEFAULT_FPS, step=1, label="FPS", scale=2)
                    v_ov  = gr.Checkbox(value=True, label="Burn overlay", scale=1)
                    v_a   = gr.Slider(0.1, 1.0, DEFAULT_ALPHA, step=0.05,
                                       label="Overlay opacity", scale=2)
                with gr.Row():
                    btn_play = gr.Button("▶  Generate & Play", variant="primary", scale=1)
                    with gr.Column(scale=4):
                        v_status = gr.Markdown("*Click Generate & Play.*")
                video_out = gr.Video(label="Playback", height=390, autoplay=True)
                cache_md  = gr.Markdown(cache_status_md())
                gr.Button("↻  Refresh cache status", size="sm").click(
                    cache_status_md, outputs=cache_md)

                # wiring
                selected_seq.change(
                    lambda s: gr.Dropdown(value=s) if s and s in ALL_SEQUENCES else gr.Dropdown(),
                    selected_seq, seq_dd)

                def _on_seq(seq):
                    if seq not in ALL_SEQUENCES:
                        return gr.Slider(), None, None, "", ""
                    n  = len(_get_frame_paths(seq))
                    fi = render_frame(seq, 0, True, DEFAULT_ALPHA)
                    ai = render_mask(seq, 0)
                    sl = gr.Slider(minimum=0, maximum=n-1, value=0, step=1,
                                   label=f"Frame  (0 – {n-1})")
                    return sl, fi, ai, _seq_info(seq), get_legend(seq)

                seq_dd.change(_on_seq, seq_dd,
                              [frame_sl, img_out, ann_out, seq_info_md, legend_md])
                seq_dd.change(lambda *_: (None, "*Click Generate & Play.*"),
                              seq_dd, [video_out, v_status])

                def _fr(seq, idx, ov, a):
                    return render_frame(seq, int(idx), ov, a), render_mask(seq, int(idx))
                frame_sl.change(_fr, [seq_dd, frame_sl, ov_cb, alpha_sl], [img_out, ann_out])
                ov_cb.change(_fr,    [seq_dd, frame_sl, ov_cb, alpha_sl], [img_out, ann_out])
                alpha_sl.change(_fr, [seq_dd, frame_sl, ov_cb, alpha_sl], [img_out, ann_out])
                btn_play.click(get_video, [seq_dd, v_ov, v_a, v_fps], [video_out, v_status])

            # ──────────────────────────────────────────────────────────────
            # Tab 3 · Gallery  (thumbnail grid of all sequences)
            # ──────────────────────────────────────────────────────────────
            with gr.TabItem("🖼 Gallery"):
                gr.Markdown(
                    "Thumbnails of all sequences (first frame). "
                    "Use the filters to narrow down, then **click any thumbnail** "
                    "to instantly play that sequence below."
                )
                with gr.Row():
                    g_year  = gr.Dropdown(["All years","2016 only","2017 only"],
                                           value="All years", label="Year", scale=1)
                    g_split = gr.Dropdown(["All splits","Train only","Val only"],
                                           value="All splits", label="Split", scale=1)
                    g_obj   = gr.Dropdown(["Any # objects","1 object","2 objects","3+ objects"],
                                           value="Any # objects", label="Objects", scale=1)
                    g_srch  = gr.Textbox(placeholder="Search…", label="Search", scale=2)
                with gr.Row():
                    g_fmin = gr.Slider(int(DF["frames"].min()), int(DF["frames"].max()),
                                       int(DF["frames"].min()), step=1, label="Min frames", scale=3)
                    g_fmax = gr.Slider(int(DF["frames"].min()), int(DF["frames"].max()),
                                       int(DF["frames"].max()), step=1, label="Max frames", scale=3)
                with gr.Row():
                    g_ov = gr.Checkbox(value=False, label="Show mask overlay on thumbnails")

                g_count_md = gr.Markdown(f"**{len(ALL_SEQUENCES)} sequences**")

                # Gallery component
                gallery = gr.Gallery(
                    value=_ALL_THUMBS,
                    label="Sequences",
                    columns=5,
                    rows=None,
                    height="auto",
                    allow_preview=False,
                    show_label=False,
                )

                # State holding the sequence names matching current filter (in gallery order)
                g_seq_state = gr.State(ALL_SEQUENCES.copy())

                gr.Markdown("---")
                with gr.Row():
                    g_info_md  = gr.Markdown("*Click a thumbnail to play.*")
                with gr.Row():
                    g_fps      = gr.Slider(1, 30, DEFAULT_FPS, step=1,
                                            label="FPS", scale=2)
                    g_vid_ov   = gr.Checkbox(value=True, label="Burn overlay", scale=1)
                    g_vid_a    = gr.Slider(0.1, 1.0, DEFAULT_ALPHA, step=0.05,
                                            label="Opacity", scale=2)
                    g_btn_play = gr.Button("▶  Play selected", variant="primary", scale=1)

                with gr.Column(scale=5):
                    g_vid_status = gr.Markdown("")
                g_video = gr.Video(label="Playback", height=400, autoplay=True)
                g_selected = gr.State("")

                # Filter → rebuild gallery
                g_f_inputs = [g_year, g_split, g_obj, g_fmin, g_fmax, g_srch]

                def _on_g_filter(*args):
                    ov = args[-1]       # last arg is the overlay checkbox
                    fargs = args[:-1]   # filter args
                    fdf   = filter_df(*fargs)
                    seqs  = fdf["sequence"].tolist()
                    items = build_gallery_items(seqs, overlay=ov)
                    return items, seqs, f"**{len(seqs)} sequences**"

                for inp in g_f_inputs + [g_ov]:
                    inp.change(_on_g_filter, g_f_inputs + [g_ov],
                               [gallery, g_seq_state, g_count_md])

                # Click thumbnail → load info + auto-generate video
                def _on_gallery_click(evt: gr.SelectData, seqs, ov, a, fps):
                    if evt is None or not seqs:
                        return "", gr.update(), None, ""
                    seq  = seqs[evt.index]
                    info = _seq_info(seq)
                    path, status = get_video(seq, ov, a, fps)
                    return info, seq, path, status

                gallery.select(
                    _on_gallery_click,
                    inputs=[g_seq_state, g_vid_ov, g_vid_a, g_fps],
                    outputs=[g_info_md, g_selected, g_video, g_vid_status],
                )
                g_btn_play.click(
                    lambda seq, ov, a, fps: get_video(seq, ov, a, fps),
                    inputs=[g_selected, g_vid_ov, g_vid_a, g_fps],
                    outputs=[g_video, g_vid_status],
                )

            # ──────────────────────────────────────────────────────────────
            # Tab 4 · Multi-Video  (paged 3×3 grid, all as MP4)
            # ──────────────────────────────────────────────────────────────
            with gr.TabItem("📺 Multi-Video"):
                gr.Markdown(
                    f"Watch **{PAGE_SIZE} sequences at once** in a 3×3 grid. "
                    "Use Prev/Next to page through all {len(ALL_SEQUENCES)} sequences, "
                    "or filter first. Videos are encoded once and cached permanently."
                )
                with gr.Row():
                    mv_year  = gr.Dropdown(["All years","2016 only","2017 only"],
                                            value="All years", label="Year", scale=1)
                    mv_split = gr.Dropdown(["All splits","Train only","Val only"],
                                            value="All splits", label="Split", scale=1)
                    mv_obj   = gr.Dropdown(["Any # objects","1 object","2 objects","3+ objects"],
                                            value="Any # objects", label="Objects", scale=1)
                    mv_srch  = gr.Textbox(placeholder="Search…", label="Search", scale=2)
                with gr.Row():
                    mv_fps  = gr.Slider(1, 30, DEFAULT_FPS, step=1, label="FPS", scale=2)
                    mv_ov   = gr.Checkbox(value=True, label="Burn overlay", scale=1)
                    mv_a    = gr.Slider(0.1, 1.0, DEFAULT_ALPHA, step=0.05,
                                         label="Opacity", scale=2)
                    mv_load = gr.Button("▶  Load Page", variant="primary", scale=1)

                with gr.Row():
                    mv_prev = gr.Button("◀  Prev", scale=1)
                    with gr.Column(scale=3):
                        mv_page_lbl = gr.Markdown(f"**Page 1 / {total_pages}**")
                    mv_next = gr.Button("Next  ▶", scale=1)

                mv_status = gr.Markdown("")

                # 9 fixed video slots, 3 rows × 3 cols
                mv_vids  = []
                mv_lbls  = []
                for row_i in range(3):
                    with gr.Row():
                        for col_i in range(3):
                            with gr.Column():
                                lbl = gr.Markdown("—")
                                vid = gr.Video(height=260, autoplay=True, label="")
                                mv_lbls.append(lbl)
                                mv_vids.append(vid)

                # State: list of sequences currently matching filter, page index
                mv_seq_state  = gr.State(ALL_SEQUENCES.copy())
                mv_page_state = gr.State(0)

                def _mv_filter(*args):
                    fdf  = filter_df(*args)
                    seqs = fdf["sequence"].tolist()
                    tp   = max(1, (len(seqs) + PAGE_SIZE - 1) // PAGE_SIZE)
                    return seqs, 0, f"**Page 1 / {tp}**"

                mv_f_inputs = [mv_year, mv_split, mv_obj,
                               gr.Slider(int(DF["frames"].min()), int(DF["frames"].max()),
                                         int(DF["frames"].min()), step=1, label=""),
                               gr.Slider(int(DF["frames"].min()), int(DF["frames"].max()),
                                         int(DF["frames"].max()), step=1, label=""),
                               mv_srch]

                # Simpler: just use the three dropdowns + search for multi-video filter
                def _mv_filter_simple(yr, sp, ob, sr):
                    fdf  = filter_df(yr, sp, ob,
                                     int(DF["frames"].min()), int(DF["frames"].max()), sr)
                    seqs = fdf["sequence"].tolist()
                    tp   = max(1, (len(seqs) + PAGE_SIZE - 1) // PAGE_SIZE)
                    return seqs, 0, f"**Page 1 / {tp}**"

                mv_simple_f = [mv_year, mv_split, mv_obj, mv_srch]
                for inp in mv_simple_f:
                    inp.change(_mv_filter_simple, mv_simple_f,
                               [mv_seq_state, mv_page_state, mv_page_lbl])

                def _load_page(seqs, page, ov, a, fps):
                    # Block if pre-cache not yet finished
                    if not _is_cache_complete():
                        with _cache_lock:
                            done = sum(1 for v in _cache_progress.values() if v == "done")
                        total = len(ALL_SEQUENCES)
                        pct   = int(done / total * 100) if total else 0
                        bar   = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        status = (f"⏳ `[{bar}]` {done}/{total} sequences cached ({pct}%) — "
                                  "please wait for caching to finish then click Load Page again.")
                        out = []
                        for _ in range(PAGE_SIZE):
                            out.append("—")
                            out.append(None)
                        out.append(status)
                        out.append(f"**Page {page + 1} / ?**")
                        return out

                    start  = page * PAGE_SIZE
                    chunk  = seqs[start: start + PAGE_SIZE]
                    tp     = max(1, (len(seqs) + PAGE_SIZE - 1) // PAGE_SIZE)
                    pg_lbl = f"**Page {page + 1} / {tp}**"

                    # Sequential — encode_sequence() returns in <1 ms when already cached
                    res, lbs = [], []
                    for s in chunk:
                        try:
                            p, _ = get_video(s, ov, a, fps)
                            res.append(str(p) if p else None)
                        except Exception:
                            res.append(None)
                        lbs.append(s)

                    while len(res) < PAGE_SIZE:   # pad
                        res.append(None); lbs.append("—")

                    n_loaded = sum(1 for r in res if r)
                    status   = f"✅ {n_loaded}/{len(chunk)} videos loaded (page {page+1}/{tp})"

                    out = []
                    for lb, r in zip(lbs, res):
                        out.append(f"**{lb}**" if lb and lb != "—" else "—")
                        out.append(r)
                    out.append(status)
                    out.append(pg_lbl)
                    return out

                mv_page_outputs = []
                for l, v in zip(mv_lbls, mv_vids):
                    mv_page_outputs.append(l)
                    mv_page_outputs.append(v)
                mv_page_outputs.append(mv_status)
                mv_page_outputs.append(mv_page_lbl)

                mv_load.click(
                    _load_page,
                    inputs=[mv_seq_state, mv_page_state, mv_ov, mv_a, mv_fps],
                    outputs=mv_page_outputs,
                )

                def _prev(seqs, page):
                    new_p = max(0, page - 1)
                    tp    = max(1, (len(seqs) + PAGE_SIZE - 1) // PAGE_SIZE)
                    return new_p, f"**Page {new_p+1} / {tp}**"

                def _next(seqs, page):
                    tp    = max(1, (len(seqs) + PAGE_SIZE - 1) // PAGE_SIZE)
                    new_p = min(tp - 1, page + 1)
                    return new_p, f"**Page {new_p+1} / {tp}**"

                mv_prev.click(_prev, [mv_seq_state, mv_page_state],
                              [mv_page_state, mv_page_lbl])
                mv_next.click(_next, [mv_seq_state, mv_page_state],
                              [mv_page_state, mv_page_lbl])

            # ──────────────────────────────────────────────────────────────
            # Tab 5 · Compare  (up to 6 side-by-side)
            # ──────────────────────────────────────────────────────────────
            with gr.TabItem("⚖️ Compare"):
                gr.Markdown(
                    "Pick up to **6 sequences**, set FPS/overlay, "
                    "then **Load All** — encoded in parallel and cached."
                )
                with gr.Row():
                    cmp_fps = gr.Slider(1, 30, DEFAULT_FPS, step=1, label="FPS", scale=2)
                    cmp_ov  = gr.Checkbox(value=True, label="Burn overlay", scale=1)
                    cmp_a   = gr.Slider(0.1, 1.0, DEFAULT_ALPHA, step=0.05,
                                         label="Opacity", scale=2)
                    cmp_btn = gr.Button("▶  Load All", variant="primary", scale=1)

                cmp_dds  = []
                cmp_vids = []
                cmp_lbls = []
                default_seqs = (ALL_SEQUENCES + [None] * MAX_COMPARE)[:MAX_COMPARE]

                for row_i in range(2):
                    with gr.Row():
                        for col_i in range(3):
                            si = row_i * 3 + col_i
                            with gr.Column():
                                dd  = gr.Dropdown([""] + ALL_SEQUENCES,
                                                   value=default_seqs[si] or "",
                                                   label=f"Slot {si+1}")
                                vid = gr.Video(height=270, autoplay=True, label="")
                                lbl = gr.Markdown(
                                    f"*{default_seqs[si]}*" if default_seqs[si] else "*empty*")
                                cmp_dds.append(dd)
                                cmp_vids.append(vid)
                                cmp_lbls.append(lbl)

                cmp_status = gr.Markdown("")

                cmp_outputs = []
                for v, l in zip(cmp_vids, cmp_lbls):
                    cmp_outputs.append(v)
                    cmp_outputs.append(l)
                cmp_outputs.append(cmp_status)

                def _load_all(*args):
                    ov, a, fps = args[0], args[1], args[2]
                    slots = list(args[3:])
                    res   = [None] * MAX_COMPARE
                    lbs   = [""] * MAX_COMPARE

                    # Sequential — fast when pre-cached, safe in all environments
                    for i, seq in enumerate(slots):
                        if seq:
                            try:
                                p, _ = get_video(seq, ov, a, fps)
                                res[i] = str(p) if p else None
                            except Exception:
                                res[i] = None
                            lbs[i] = seq

                    n_ok = sum(1 for r in res if r)
                    out  = []
                    for r, l in zip(res, lbs):
                        out.append(r)
                        out.append(f"**{l}**" if l else "*empty*")
                    out.append(f"✅ {n_ok}/{len([s for s in slots if s])} slots loaded")
                    return out

                cmp_btn.click(_load_all,
                              inputs=[cmp_ov, cmp_a, cmp_fps] + cmp_dds,
                              outputs=cmp_outputs)

                for i, (dd, vid, lbl) in enumerate(zip(cmp_dds, cmp_vids, cmp_lbls)):
                    def _mk(idx):
                        def _single(seq, ov, a, fps):
                            p, _ = get_video(seq, ov, a, fps)
                            return p, f"**{seq}**" if seq else "*empty*"
                        return _single
                    dd.change(_mk(i), [dd, cmp_ov, cmp_a, cmp_fps], [vid, lbl])

            # ──────────────────────────────────────────────────────────────
            # Tab 6 · Statistics
            # ──────────────────────────────────────────────────────────────
            with gr.TabItem("📊 Statistics"):
                gr.Markdown("### Dataset Overview")
                with gr.Row():
                    gr.Plot(value=figs[0], label="Frame count")
                    gr.Plot(value=figs[1], label="Object count")
                with gr.Row():
                    gr.Plot(value=figs[2], label="Splits")
                    gr.Plot(value=figs[3], label="Resolution")
                with gr.Row():
                    gr.Plot(value=figs[4], label="Frames vs. Objects")
                gr.Markdown(f"""
**Quick facts**
- Total sequences: **{len(DF):,}** | Frame range: **{DF['frames'].min()}–{DF['frames'].max()}** (avg {DF['frames'].mean():.1f})
- Objects/seq: **{DF['n_objects'].min()}–{DF['n_objects'].max()}** (avg {DF['n_objects'].mean():.2f}) | Single-obj: **{int((DF['n_objects']==1).sum())}** · Multi-obj: **{int((DF['n_objects']>1).sum())}**
- DAVIS-2016: **{n_2016}** (30 train + 20 val) | DAVIS-2017: **{n_2017}** (60 train + 30 val)
- MP4 cache: `{CACHE_DIR}`
""")

            # ──────────────────────────────────────────────────────────────
            # Tab 7 · About
            # ──────────────────────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(f"""
## DAVIS — Densely Annotated VIdeo Segmentation

| Version | Train | Val | Total |
|---------|-------|-----|-------|
| DAVIS-2016 | 30 | 20 | 50 |
| DAVIS-2017 | 60 | 30 | 90 |

### Dataset structure
```
DAVIS/
├── JPEGImages/480p/<seq>/%05d.jpg    RGB frames
├── Annotations/480p/<seq>/%05d.png   palette-indexed masks (value = object ID)
└── ImageSets/2016|2017/train|val.txt
```

### MP4 cache  (`{CACHE_DIR}`)
- `<seq>_raw_<fps>fps.mp4`  — raw frames  
- `<seq>_ov055_<fps>fps.mp4` — DAVIS palette overlay @ 55 % opacity

### Annotation format
Pixel value = object ID. Rendered with the official DAVIS 20-colour palette.

### Citation
```bibtex
@article{{Pont-Tuset_arXiv_2017,
  author  = {{Jordi Pont-Tuset et al.}},
  title   = {{The 2017 DAVIS Challenge on Video Object Segmentation}},
  journal = {{arXiv:1704.00675}}, year = {{2017}}
}}
```
**Data root:** `{DAVIS_ROOT}`
""")

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

demo = build_ui()
start_precache(fps=DEFAULT_FPS, workers=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAVIS Dataset Explorer")
    parser.add_argument("--share",  action="store_true")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--host",   default="0.0.0.0")
    args = parser.parse_args()
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, theme=gr.themes.Soft())
