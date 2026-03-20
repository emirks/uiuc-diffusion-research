"""
DAVIS Dataset Explorer
======================
Interactive Gradio app for browsing, viewing and analysing the DAVIS 2017
video object segmentation dataset (480p split).

Usage (from repo root):
    python scripts/davis_explorer/app.py

    # Point at a custom DAVIS root:
    DAVIS_ROOT=/path/to/DAVIS python scripts/davis_explorer/app.py

    # Create a 72-h public Gradio link:
    python scripts/davis_explorer/app.py --share

Dataset layout expected:
    <DAVIS_ROOT>/
        JPEGImages/480p/<sequence>/%05d.jpg
        Annotations/480p/<sequence>/%05d.png   (palette-indexed, value = object ID)
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

_DEFAULT_ROOT = Path("/workspace/diffusion-research/data/raw/DAVIS")
DAVIS_ROOT    = Path(os.environ.get("DAVIS_ROOT", str(_DEFAULT_ROOT)))

IMG_DIR  = DAVIS_ROOT / "JPEGImages" / "480p"
ANN_DIR  = DAVIS_ROOT / "Annotations" / "480p"
SETS_DIR = DAVIS_ROOT / "ImageSets"

# Permanent MP4 cache — survives restarts; raw and overlay variants stored here.
CACHE_DIR = Path(os.environ.get(
    "DAVIS_CACHE_DIR",
    str(DAVIS_ROOT.parents[2] / "data/processed/DAVIS_explorer_cache"),
))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# DAVIS standard 20-colour palette (0 = background, 1-N = objects).
DAVIS_PALETTE = np.array([
    [  0,   0,   0],   # 0  background
    [128,   0,   0],   # 1
    [  0, 128,   0],   # 2
    [128, 128,   0],   # 3
    [  0,   0, 128],   # 4
    [128,   0, 128],   # 5
    [  0, 128, 128],   # 6
    [128, 128, 128],   # 7
    [ 64,   0,   0],   # 8
    [192,   0,   0],   # 9
    [ 64, 128,   0],   # 10
    [192, 128,   0],   # 11
    [ 64,   0, 128],   # 12
    [192,   0, 128],   # 13
    [ 64, 128, 128],   # 14
    [192, 128, 128],   # 15
    [  0,  64,   0],   # 16
    [128,  64,   0],   # 17
    [  0, 192,   0],   # 18
    [128, 192,   0],   # 19
], dtype=np.uint8)

# Default video encode settings
DEFAULT_FPS   = 24
DEFAULT_ALPHA = 0.55
DEFAULT_CRF   = 18

# Maximum video slots in the Compare tab
MAX_SLOTS = 6

# ── Dataset loading ────────────────────────────────────────────────────────────

def _read_split(year: str, split: str) -> list[str]:
    p = SETS_DIR / year / f"{split}.txt"
    return p.read_text().strip().splitlines() if p.exists() else []


def _count_objects(seq: str) -> int:
    ann_seq = ANN_DIR / seq
    if not ann_seq.exists():
        return 0
    files = sorted(ann_seq.iterdir())
    if not files:
        return 0
    return int(np.max(np.array(Image.open(files[0]))))


def build_dataframe() -> pd.DataFrame:
    seqs      = sorted(d.name for d in IMG_DIR.iterdir() if d.is_dir())
    s16_train = set(_read_split("2016", "train"))
    s16_val   = set(_read_split("2016", "val"))
    s17_train = set(_read_split("2017", "train"))
    s17_val   = set(_read_split("2017", "val"))

    rows = []
    for seq in seqs:
        img_files   = sorted((IMG_DIR / seq).glob("*.jpg"))
        n_frames    = len(img_files)
        n_objects   = _count_objects(seq)
        first_img   = Image.open(img_files[0]) if img_files else None
        w, h        = first_img.size if first_img else (0, 0)

        in_16_train = seq in s16_train
        in_16_val   = seq in s16_val
        in_17_train = seq in s17_train
        in_17_val   = seq in s17_val

        splits = (
            (["2016-train"] if in_16_train else []) +
            (["2016-val"]   if in_16_val   else []) +
            (["2017-train"] if in_17_train else []) +
            (["2017-val"]   if in_17_val   else [])
        )
        rows.append({
            "sequence":   seq,
            "frames":     n_frames,
            "n_objects":  n_objects,
            "width":      w,
            "height":     h,
            "resolution": f"{w}×{h}",
            "split":      ", ".join(splits) if splits else "unlisted",
            "in_2016":    in_16_train or in_16_val,
            "in_2017":    in_17_train or in_17_val,
            "in_train":   in_16_train or in_17_train,
            "in_val":     in_16_val   or in_17_val,
        })

    return pd.DataFrame(rows)


print("Loading DAVIS metadata…")
DF = build_dataframe()
ALL_SEQUENCES = sorted(DF["sequence"].tolist())
print(f"  {len(DF)} sequences  ·  frames {DF['frames'].min()}–{DF['frames'].max()}  "
      f"·  objects {DF['n_objects'].min()}–{DF['n_objects'].max()}")

DISPLAY_COLS = ["sequence", "frames", "n_objects", "resolution", "split"]

# ── Frame-level helpers ────────────────────────────────────────────────────────

@lru_cache(maxsize=16)
def _get_frame_paths(seq: str) -> list[Path]:
    return sorted((IMG_DIR / seq).glob("*.jpg"))


@lru_cache(maxsize=16)
def _get_ann_paths(seq: str) -> list[Path]:
    ann_seq = ANN_DIR / seq
    if not ann_seq.exists():
        return []
    return sorted(ann_seq.glob("*.png"))


def _blend_frame(img_arr: np.ndarray, ann_arr: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorised numpy blend: overlay DAVIS palette colours on top of RGB frame."""
    overlay_rgb = DAVIS_PALETTE[np.clip(ann_arr, 0, len(DAVIS_PALETTE) - 1)]
    a = np.where(ann_arr == 0, 0.0, alpha).astype(np.float32)[:, :, None]
    return (img_arr * (1.0 - a) + overlay_rgb.astype(np.float32) * a).clip(0, 255).astype(np.uint8)


def render_frame(seq: str, frame_idx: int, show_overlay: bool, alpha: float) -> Image.Image:
    frame_paths = _get_frame_paths(seq)
    if not frame_paths:
        return Image.new("RGB", (854, 480), (30, 30, 30))
    idx = min(max(0, frame_idx), len(frame_paths) - 1)
    img_arr = np.array(Image.open(frame_paths[idx]).convert("RGB"))
    if show_overlay:
        ann_paths = _get_ann_paths(seq)
        if idx < len(ann_paths):
            ann_arr = np.array(Image.open(ann_paths[idx]))
            img_arr = _blend_frame(img_arr.astype(np.float32), ann_arr, alpha)
    return Image.fromarray(img_arr)


def render_annotation_only(seq: str, frame_idx: int) -> Image.Image:
    ann_paths = _get_ann_paths(seq)
    if not ann_paths:
        return Image.new("RGB", (854, 480), (30, 30, 30))
    idx = min(max(0, frame_idx), len(ann_paths) - 1)
    ann = np.array(Image.open(ann_paths[idx]))
    rgb = np.zeros((*ann.shape, 3), dtype=np.uint8)
    for obj_id in range(1, len(DAVIS_PALETTE)):
        mask = ann == obj_id
        if mask.any():
            rgb[mask] = DAVIS_PALETTE[obj_id]
    return Image.fromarray(rgb)


# ── MP4 generation & pre-caching ───────────────────────────────────────────────

def _mp4_path(seq: str, overlay: bool, alpha: float, fps: int) -> Path:
    """Canonical path for a cached MP4."""
    tag = f"ov{int(alpha * 100):03d}" if overlay else "raw"
    return CACHE_DIR / f"{seq}_{tag}_{fps}fps.mp4"


def _ffmpeg_encode(input_pattern: str, out_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-crf", str(DEFAULT_CRF),
        "-movflags", "+faststart",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-600:])


def encode_sequence(seq: str, overlay: bool, alpha: float, fps: int) -> Path:
    """
    Encode one sequence to MP4.  Returns the output path.
    Checks the permanent cache first — skips ffmpeg if the file already exists.

    • No-overlay: ffmpeg reads JPEGs directly (no Python per-frame work).
    • Overlay:    vectorised numpy blend → temp PNGs → ffmpeg → cleanup.
    """
    out = _mp4_path(seq, overlay, alpha, fps)
    if out.exists():
        return out

    frame_paths = _get_frame_paths(seq)
    if not frame_paths:
        raise FileNotFoundError(f"No frames for {seq}")

    if not overlay:
        _ffmpeg_encode(str(IMG_DIR / seq / "%05d.jpg"), out, fps)
        return out

    # Overlay: render every frame, save as PNG strip, encode
    ann_paths  = _get_ann_paths(seq)
    tmp_dir    = CACHE_DIR / f"_tmp_{seq}_{int(alpha*100):03d}"
    tmp_dir.mkdir(exist_ok=True)
    try:
        for i, fp in enumerate(frame_paths):
            img_arr = np.array(Image.open(fp).convert("RGB"), dtype=np.float32)
            if i < len(ann_paths):
                ann_arr = np.array(Image.open(ann_paths[i]), dtype=np.uint8)
                img_arr = _blend_frame(img_arr, ann_arr, alpha).astype(np.float32)
            Image.fromarray(img_arr.clip(0, 255).astype(np.uint8)).save(
                tmp_dir / f"{i:05d}.png", optimize=False
            )
        _ffmpeg_encode(str(tmp_dir / "%05d.png"), out, fps)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out


# ── Background pre-caching ─────────────────────────────────────────────────────

_cache_progress: dict[str, str] = {}   # seq → "pending" | "done" | "error: …"
_cache_lock = threading.Lock()


def _precache_worker(seq: str, fps: int) -> None:
    """Encode raw + overlay MP4s for one sequence; update _cache_progress."""
    with _cache_lock:
        _cache_progress[seq] = "encoding…"
    try:
        encode_sequence(seq, overlay=False, alpha=DEFAULT_ALPHA, fps=fps)
        encode_sequence(seq, overlay=True,  alpha=DEFAULT_ALPHA, fps=fps)
        with _cache_lock:
            _cache_progress[seq] = "done"
    except Exception as e:
        with _cache_lock:
            _cache_progress[seq] = f"error: {e}"


def start_precache(fps: int = DEFAULT_FPS, max_workers: int = 4) -> None:
    """Launch background thread pool to pre-encode all sequences."""
    missing = [s for s in ALL_SEQUENCES
               if not _mp4_path(s, False, DEFAULT_ALPHA, fps).exists()
               or not _mp4_path(s, True,  DEFAULT_ALPHA, fps).exists()]

    if not missing:
        print(f"  MP4 cache complete ({len(ALL_SEQUENCES)} × 2 variants already exist)")
        for s in ALL_SEQUENCES:
            _cache_progress[s] = "done"
        return

    print(f"  Pre-caching {len(missing)} sequences in the background "
          f"(workers={max_workers})…")

    def _run():
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_precache_worker, s, fps): s for s in missing}
            done = 0
            for fut in as_completed(futures):
                done += 1
                seq = futures[fut]
                status = _cache_progress.get(seq, "?")
                if done % 10 == 0 or done == len(missing):
                    print(f"  Cache progress: {done}/{len(missing)} ({seq}: {status})")

    threading.Thread(target=_run, daemon=True).start()


# ── filter / plot helpers ─────────────────────────────────────────────────────

def filter_df(year_filter, split_filter, obj_filter, frame_min, frame_max, search):
    d = DF.copy()
    if year_filter == "2016 only":
        d = d[d["in_2016"]]
    elif year_filter == "2017 only":
        d = d[d["in_2017"]]
    if split_filter == "Train only":
        d = d[d["in_train"]]
    elif split_filter == "Val only":
        d = d[d["in_val"]]
    if obj_filter == "1 object":
        d = d[d["n_objects"] == 1]
    elif obj_filter == "2 objects":
        d = d[d["n_objects"] == 2]
    elif obj_filter == "3+ objects":
        d = d[d["n_objects"] >= 3]
    d = d[(d["frames"] >= frame_min) & (d["frames"] <= frame_max)]
    if search.strip():
        q = search.strip().lower()
        d = d[d["sequence"].str.lower().str.contains(q, na=False)]
    return d[DISPLAY_COLS].reset_index(drop=True)


def make_stats_plots():
    d = DF.copy()

    fig_frames = px.histogram(
        d, x="frames", nbins=30,
        title="Frame Count Distribution",
        color_discrete_sequence=["#3B82F6"],
        labels={"frames": "Number of Frames"},
    )
    fig_frames.update_layout(margin=dict(t=45, b=40))

    obj_counts = d["n_objects"].value_counts().sort_index().reset_index()
    obj_counts.columns = ["n_objects", "count"]
    fig_objs = px.bar(
        obj_counts, x="n_objects", y="count",
        title="Sequences by Object Count",
        color="count", color_continuous_scale="Teal",
        labels={"n_objects": "Objects per Sequence", "count": "# Sequences"},
    )
    fig_objs.update_layout(coloraxis_showscale=False, margin=dict(t=45, b=40))
    fig_objs.update_xaxes(tickmode="linear", dtick=1)

    split_data = {
        "2016-train": int(d["split"].str.contains("2016-train").sum()),
        "2016-val":   int(d["split"].str.contains("2016-val").sum()),
        "2017-train": int(d["split"].str.contains("2017-train").sum()),
        "2017-val":   int(d["split"].str.contains("2017-val").sum()),
    }
    fig_splits = px.bar(
        x=list(split_data.keys()), y=list(split_data.values()),
        title="Sequences per Split",
        color=list(split_data.keys()),
        color_discrete_sequence=["#3B82F6", "#6366F1", "#F59E0B", "#EF4444"],
        labels={"x": "Split", "y": "# Sequences"},
    )
    fig_splits.update_layout(showlegend=False, margin=dict(t=45, b=40))

    res_counts = d["resolution"].value_counts().reset_index()
    res_counts.columns = ["resolution", "count"]
    fig_res = px.pie(
        res_counts, names="resolution", values="count",
        title="Resolution Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_res.update_layout(margin=dict(t=45, b=20))

    fig_scatter = px.scatter(
        d, x="frames", y="n_objects",
        text="sequence",
        title="Frames vs. Object Count",
        color="n_objects", color_continuous_scale="Viridis",
        size="frames", size_max=18,
        labels={"frames": "Frames", "n_objects": "Objects"},
        hover_data=["sequence", "frames", "n_objects", "resolution", "split"],
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=8)
    fig_scatter.update_layout(coloraxis_showscale=False, margin=dict(t=45, b=40))

    return fig_frames, fig_objs, fig_splits, fig_res, fig_scatter


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _seq_info(seq: str) -> str:
    if seq not in ALL_SEQUENCES:
        return ""
    row = DF[DF["sequence"] == seq].iloc[0]
    return (f"**{seq}** — {row['frames']} frames · {row['n_objects']} object(s) · "
            f"{row['resolution']} · _{row['split']}_")


def get_object_legend(seq: str) -> str:
    if not seq or seq not in ALL_SEQUENCES:
        return ""
    n = int(DF[DF["sequence"] == seq].iloc[0]["n_objects"])
    if n == 0:
        return "*No annotated objects.*"
    lines = ["**Object colours:**"]
    for i in range(1, min(n + 1, len(DAVIS_PALETTE))):
        c    = DAVIS_PALETTE[i]
        hx   = "#{:02X}{:02X}{:02X}".format(*c)
        lines.append(f"- <span style='color:{hx};font-weight:bold'>■</span> Object {i}")
    return "\n".join(lines)


def cache_status_md() -> str:
    with _cache_lock:
        done  = sum(1 for v in _cache_progress.values() if v == "done")
        total = len(ALL_SEQUENCES)
    pct = done / total * 100 if total else 0
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    return f"`[{bar}]` **{done}/{total}** sequences cached ({pct:.0f} %)"


# ── Gradio callbacks ───────────────────────────────────────────────────────────

def on_filter(year_filter, split_filter, obj_filter, frame_min, frame_max, search):
    filtered = filter_df(year_filter, split_filter, obj_filter, frame_min, frame_max, search)
    return filtered, f"**{len(filtered)} sequences** match the current filters."


def on_row_select(evt: gr.SelectData, filtered_df: pd.DataFrame):
    if evt is None or filtered_df is None or len(filtered_df) == 0:
        return gr.update(), "Select a row to see details."
    seq = filtered_df.iloc[evt.index[0]]["sequence"]
    row = DF[DF["sequence"] == seq].iloc[0]
    splits_clean = row["split"].replace(", ", "  \n• ")
    md = (f"### `{seq}`\n\n"
          f"| Field | Value |\n|---|---|\n"
          f"| Frames | **{row['frames']}** |\n"
          f"| Objects | **{row['n_objects']}** |\n"
          f"| Resolution | {row['resolution']} |\n"
          f"| Splits | • {splits_clean} |\n\n"
          f"> Click the **Viewer** tab to explore frames and annotations.")
    return seq, md


def on_seq_change(seq: str):
    if not seq or seq not in ALL_SEQUENCES:
        return gr.Slider(minimum=0, maximum=0, value=0), None, None, "", ""
    n        = len(_get_frame_paths(seq))
    frame_img = render_frame(seq, 0, show_overlay=True, alpha=DEFAULT_ALPHA)
    ann_img   = render_annotation_only(seq, 0)
    slider    = gr.Slider(minimum=0, maximum=n - 1, value=0, step=1,
                          label=f"Frame  (0 – {n-1})")
    return slider, frame_img, ann_img, _seq_info(seq), get_object_legend(seq)


def on_frame_change(seq, frame_idx, show_overlay, alpha):
    if not seq or seq not in ALL_SEQUENCES:
        return None, None
    return (render_frame(seq, int(frame_idx), show_overlay, alpha),
            render_annotation_only(seq, int(frame_idx)))


def get_video(seq: str, overlay: bool, alpha: float, fps: int) -> tuple[str | None, str]:
    """Encode (or retrieve cached) an MP4 for one sequence."""
    if not seq or seq not in ALL_SEQUENCES:
        return None, "No sequence selected."
    try:
        path = encode_sequence(seq, overlay, round(alpha, 2), fps)
        n    = int(DF[DF["sequence"] == seq].iloc[0]["frames"])
        size = path.stat().st_size // 1024
        mode = "overlay" if overlay else "raw"
        return str(path), f"✅ **{seq}** · {n} frames · {fps} fps · {mode} · {size} KB"
    except Exception as e:
        return None, f"❌ {e}"


def load_compare_slot(seq: str, overlay: bool, alpha: float, fps: int) -> tuple[str | None, str]:
    """Load one compare slot — same as get_video but returns placeholder label."""
    if not seq:
        return None, ""
    path, msg = get_video(seq, overlay, alpha, fps)
    return path, seq   # label shown under the video


def load_all_compare(
    ov: bool, alpha: float, fps: int,
    s0, s1, s2, s3, s4, s5,
) -> list:
    """Encode/fetch all 6 slots in parallel and return (video, label) pairs."""
    slots = [s0, s1, s2, s3, s4, s5]
    results = [None] * MAX_SLOTS
    labels  = [""] * MAX_SLOTS

    def _load(i, seq):
        if seq:
            p, _ = get_video(seq, ov, round(alpha, 2), fps)
            results[i] = str(p) if p else None
            labels[i]  = seq

    with ThreadPoolExecutor(max_workers=MAX_SLOTS) as pool:
        futs = {pool.submit(_load, i, s): i for i, s in enumerate(slots) if s}
        for f in as_completed(futs):
            f.result()   # re-raise exceptions if any

    # Flatten to [vid0, label0, vid1, label1, …]
    out = []
    for i in range(MAX_SLOTS):
        out.append(results[i])
        out.append(labels[i])
    return out


# ── Build UI ───────────────────────────────────────────────────────────────────

def build_ui():
    fig_frames, fig_objs, fig_splits, fig_res, fig_scatter = make_stats_plots()
    n_multi = int((DF["n_objects"] > 1).sum())
    n_2016  = int(DF["in_2016"].sum())
    n_2017  = int(DF["in_2017"].sum())

    _first = ALL_SEQUENCES[0]
    _first_n = len(_get_frame_paths(_first))

    with gr.Blocks(title="DAVIS Dataset Explorer") as demo:

        gr.Markdown(
            "# 🎬 DAVIS Dataset Explorer\n"
            f"**DAVIS 2017 · 480p** — {len(DF)} sequences · "
            f"frames {DF['frames'].min()}–{DF['frames'].max()} · "
            f"{n_2016} in DAVIS-2016 · {n_2017} in DAVIS-2017 · "
            f"{n_multi} multi-object sequences"
        )

        with gr.Tabs():

            # ────────────────────────────────────────────────────────────────
            # Tab 1 · Browse
            # ────────────────────────────────────────────────────────────────
            with gr.TabItem("📋 Browse"):
                with gr.Row():
                    dd_year  = gr.Dropdown(
                        choices=["All years", "2016 only", "2017 only"],
                        value="All years", label="Dataset year", scale=1)
                    dd_split = gr.Dropdown(
                        choices=["All splits", "Train only", "Val only"],
                        value="All splits", label="Split", scale=1)
                    dd_obj   = gr.Dropdown(
                        choices=["Any # objects", "1 object", "2 objects", "3+ objects"],
                        value="Any # objects", label="Object count", scale=1)
                    txt_search = gr.Textbox(
                        placeholder="Search sequence name…", label="Search", scale=2)

                with gr.Row():
                    frame_min_sl = gr.Slider(
                        int(DF["frames"].min()), int(DF["frames"].max()),
                        value=int(DF["frames"].min()), step=1,
                        label="Min frames", scale=3)
                    frame_max_sl = gr.Slider(
                        int(DF["frames"].min()), int(DF["frames"].max()),
                        value=int(DF["frames"].max()), step=1,
                        label="Max frames", scale=3)

                count_md = gr.Markdown(f"**{len(DF)} sequences** match the current filters.")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        tbl = gr.DataFrame(value=DF[DISPLAY_COLS],
                                           interactive=False, wrap=False)
                    with gr.Column(scale=2):
                        detail_md = gr.Markdown("*Select a row to see details.*")

                filtered_state = gr.State(DF[DISPLAY_COLS].copy())
                selected_seq   = gr.State("")

                filter_inputs = [dd_year, dd_split, dd_obj, frame_min_sl, frame_max_sl, txt_search]
                for inp in filter_inputs:
                    inp.change(on_filter, filter_inputs, [tbl, count_md])
                    inp.change(lambda *a: filter_df(*a), filter_inputs, filtered_state)

                tbl.select(on_row_select, filtered_state, [selected_seq, detail_md])

            # ────────────────────────────────────────────────────────────────
            # Tab 2 · Viewer  (frame scrubber + single-sequence video)
            # ────────────────────────────────────────────────────────────────
            with gr.TabItem("🔍 Viewer"):

                with gr.Row():
                    seq_dd = gr.Dropdown(
                        choices=ALL_SEQUENCES, value=_first,
                        label="Sequence", scale=5)

                seq_info_md = gr.Markdown(_seq_info(_first))

                # ── Frame scrubber ────────────────────────────────────────
                gr.Markdown("#### Frame Scrubber")
                with gr.Row():
                    show_overlay_cb = gr.Checkbox(value=True, label="Show mask overlay")
                    alpha_sl = gr.Slider(0.1, 1.0, value=DEFAULT_ALPHA, step=0.05,
                                         label="Overlay opacity")

                frame_sl = gr.Slider(
                    minimum=0, maximum=_first_n - 1, value=0, step=1,
                    label=f"Frame  (0 – {_first_n - 1})")

                with gr.Row():
                    img_out = gr.Image(label="Frame (+overlay)", type="pil", height=370,
                                       value=render_frame(_first, 0, True, DEFAULT_ALPHA))
                    ann_out = gr.Image(label="Annotation mask",  type="pil", height=370,
                                       value=render_annotation_only(_first, 0))

                legend_md = gr.Markdown(get_object_legend(_first))

                # ── Video playback ────────────────────────────────────────
                gr.Markdown("---\n#### Video Playback")
                gr.Markdown(
                    "Raw (no overlay) encodes directly from JPEG files via ffmpeg — very fast. "
                    "Overlay blends masks with a vectorised numpy pass. "
                    "Both are cached permanently in `DAVIS_explorer_cache/`."
                )

                with gr.Row():
                    fps_sl = gr.Slider(1, 30, value=DEFAULT_FPS, step=1,
                                        label="FPS", scale=2)
                    vid_overlay_cb = gr.Checkbox(value=True,
                                                  label="Burn mask overlay", scale=1)
                    vid_alpha_sl = gr.Slider(0.1, 1.0, value=DEFAULT_ALPHA, step=0.05,
                                              label="Overlay opacity", scale=2)

                with gr.Row():
                    btn_play = gr.Button("▶  Generate & Play", variant="primary", scale=1)
                    with gr.Column(scale=4):
                        vid_status = gr.Markdown("*Click Generate & Play.*")

                video_out = gr.Video(label="Sequence playback", height=400, autoplay=True)

                # cache progress indicator
                cache_md = gr.Markdown(cache_status_md())
                btn_refresh_cache = gr.Button("↻  Refresh cache status", size="sm")
                btn_refresh_cache.click(cache_status_md, outputs=cache_md)

                # ── Event wiring ──────────────────────────────────────────

                selected_seq.change(
                    lambda s: gr.Dropdown(value=s) if s and s in ALL_SEQUENCES else gr.Dropdown(),
                    selected_seq, seq_dd)

                def _on_seq_change(seq):
                    slider, fi, ai, info, legend = on_seq_change(seq)
                    return slider, fi, ai, info, legend, None, "*Click Generate & Play.*"

                seq_dd.change(_on_seq_change, seq_dd,
                              [frame_sl, img_out, ann_out, seq_info_md,
                               legend_md, video_out, vid_status])

                frame_sl.change(on_frame_change,
                                [seq_dd, frame_sl, show_overlay_cb, alpha_sl],
                                [img_out, ann_out])
                show_overlay_cb.change(on_frame_change,
                                       [seq_dd, frame_sl, show_overlay_cb, alpha_sl],
                                       [img_out, ann_out])
                alpha_sl.change(on_frame_change,
                                [seq_dd, frame_sl, show_overlay_cb, alpha_sl],
                                [img_out, ann_out])

                btn_play.click(get_video,
                               [seq_dd, vid_overlay_cb, vid_alpha_sl, fps_sl],
                               [video_out, vid_status])

            # ────────────────────────────────────────────────────────────────
            # Tab 3 · Compare  (up to 6 videos side-by-side)
            # ────────────────────────────────────────────────────────────────
            with gr.TabItem("🎞 Compare"):
                gr.Markdown(
                    "Pick up to **6 sequences**, choose overlay/FPS settings, "
                    "then click **Load All** — videos are encoded in parallel "
                    "and cached for instant replays."
                )

                with gr.Row():
                    cmp_fps_sl = gr.Slider(1, 30, value=DEFAULT_FPS, step=1,
                                            label="FPS (all slots)", scale=2)
                    cmp_ov_cb  = gr.Checkbox(value=True,
                                              label="Burn mask overlay (all slots)", scale=1)
                    cmp_alpha  = gr.Slider(0.1, 1.0, value=DEFAULT_ALPHA, step=0.05,
                                            label="Overlay opacity", scale=2)
                    btn_load_all = gr.Button("▶  Load All", variant="primary", scale=1)

                # 6 slots in 2 rows × 3 cols
                slot_dds  = []   # dropdowns
                vid_outs  = []   # gr.Video
                lbl_outs  = []   # gr.Markdown labels under each video

                default_seqs = (ALL_SEQUENCES + [None] * MAX_SLOTS)[:MAX_SLOTS]

                for row_idx in range(2):
                    with gr.Row():
                        for col_idx in range(3):
                            slot_i = row_idx * 3 + col_idx
                            with gr.Column():
                                dd = gr.Dropdown(
                                    choices=[""] + ALL_SEQUENCES,
                                    value=default_seqs[slot_i] or "",
                                    label=f"Slot {slot_i + 1}",
                                )
                                vid = gr.Video(height=280, autoplay=True,
                                               label=default_seqs[slot_i] or "—")
                                lbl = gr.Markdown(
                                    f"*{default_seqs[slot_i]}*"
                                    if default_seqs[slot_i] else "*empty*"
                                )
                                slot_dds.append(dd)
                                vid_outs.append(vid)
                                lbl_outs.append(lbl)

                cmp_status = gr.Markdown("")

                # Interleave vid_outs and lbl_outs for the callback output list
                compare_outputs = []
                for v, l in zip(vid_outs, lbl_outs):
                    compare_outputs.append(v)
                    compare_outputs.append(l)

                def _load_all(ov, alpha, fps, s0, s1, s2, s3, s4, s5):
                    flat = load_all_compare(ov, alpha, fps, s0, s1, s2, s3, s4, s5)
                    return flat

                btn_load_all.click(
                    _load_all,
                    inputs=[cmp_ov_cb, cmp_alpha, cmp_fps_sl] + slot_dds,
                    outputs=compare_outputs,
                )

                # Individual slot: load immediately on dropdown change
                for i, (dd, vid, lbl) in enumerate(zip(slot_dds, vid_outs, lbl_outs)):
                    def _make_single_loader(slot_idx):
                        def _load(seq, ov, alpha, fps):
                            p, label = load_compare_slot(seq, ov, alpha, fps)
                            return p, label
                        return _load
                    dd.change(
                        _make_single_loader(i),
                        inputs=[dd, cmp_ov_cb, cmp_alpha, cmp_fps_sl],
                        outputs=[vid, lbl],
                    )

            # ────────────────────────────────────────────────────────────────
            # Tab 4 · Statistics
            # ────────────────────────────────────────────────────────────────
            with gr.TabItem("📊 Statistics"):
                gr.Markdown("### Dataset Overview")
                with gr.Row():
                    gr.Plot(value=fig_frames, label="Frame count distribution")
                    gr.Plot(value=fig_objs,   label="Object count")
                with gr.Row():
                    gr.Plot(value=fig_splits, label="Sequences per split")
                    gr.Plot(value=fig_res,    label="Resolution breakdown")
                with gr.Row():
                    gr.Plot(value=fig_scatter, label="Frames vs. Objects")

                gr.Markdown(f"""
**Quick facts**
- Total sequences: **{len(DF):,}**
- Frame range: **{DF['frames'].min()} – {DF['frames'].max()}** (avg {DF['frames'].mean():.1f})
- Objects per sequence: **{DF['n_objects'].min()} – {DF['n_objects'].max()}** (avg {DF['n_objects'].mean():.2f})
- Single-object: **{int((DF['n_objects']==1).sum())}** · Multi-object: **{int((DF['n_objects']>1).sum())}**
- In DAVIS-2016: **{n_2016}** (30 train + 20 val)
- In DAVIS-2017: **{n_2017}** (60 train + 30 val)
- Unique resolutions: {sorted(DF['resolution'].unique().tolist())}
- MP4 cache: `{CACHE_DIR}`
""")

            # ────────────────────────────────────────────────────────────────
            # Tab 5 · About
            # ────────────────────────────────────────────────────────────────
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
├── JPEGImages/480p/<sequence>/%05d.jpg    ← RGB frames
├── Annotations/480p/<sequence>/%05d.png   ← palette-indexed masks (value = object ID)
└── ImageSets/2016|2017/train|val.txt
```

### MP4 cache
Generated videos are stored permanently in:
`{CACHE_DIR}`

Two variants per sequence per FPS setting:
- `<seq>_raw_<fps>fps.mp4` — raw frames, no overlay
- `<seq>_ov055_<fps>fps.mp4` — DAVIS palette overlay at 55 % opacity

### Annotation format
Pixel value = object ID (0 = background, 1 = first object, …).
Rendered with the standard DAVIS 20-colour palette.

### Evaluation metrics
- **J** (Jaccard / region IoU)
- **F** (boundary F-measure)
- Score = mean(J, F) averaged over objects and sequences.

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

# Kick off background MP4 pre-caching (non-blocking)
start_precache(fps=DEFAULT_FPS, max_workers=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAVIS Dataset Explorer")
    parser.add_argument("--share",  action="store_true")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--no-precache", action="store_true",
                        help="Skip background MP4 pre-caching at startup")
    args = parser.parse_args()

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
