"""
VC-Bench Dataset Explorer
=========================
Interactive Gradio app for browsing and analysing the VC-Bench video dataset.

Run locally (point at your local vc-bench-hf snapshot):
    python scripts/vc_bench_explorer/app.py

    # or point at a custom data dir:
    VC_BENCH_HF_DIR=/path/to/vc-bench-hf python scripts/vc_bench_explorer/app.py

Share a 72-hour public link (Gradio tunnel):
    python scripts/vc_bench_explorer/app.py --share

HF Spaces deployment:
    gradio deploy          # run from this directory
    # The app auto-downloads VC-Bench.csv from the HF dataset hub on startup
    # and streams all videos via the HF CDN — no 45 GB upload required.
"""

import argparse
import json
import os
import urllib.parse
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px

# ── Configuration ──────────────────────────────────────────────────────────────
HF_DATASET = "Kevinson-lzp/VC-Bench"

# Local data directory: override with VC_BENCH_HF_DIR env var.
_DEFAULT_HF_DIR = Path("/workspace/diffusion-research/data/raw/vc-bench-hf")
HF_DIR = Path(os.environ.get("VC_BENCH_HF_DIR", str(_DEFAULT_HF_DIR)))

CSV_PATH = HF_DIR / "VC-Bench.csv"

# Write the path-index cache next to the data when possible, otherwise /tmp.
_CACHE_DIR = HF_DIR if HF_DIR.exists() else Path("/tmp")
PATH_INDEX_CACHE = _CACHE_DIR / ".vc_bench_path_index.json"

# True when running inside a Hugging Face Space.
IS_HF_SPACE = bool(os.environ.get("SPACE_ID"))

# Main category → subcategories (from disk structure).
# Key = folder name on disk, value = list of sub-folder names.
MAIN_CAT_MAP: dict[str, list[str]] = {
    "Actions & Activities": ["action", "conversation", "dancing", "meeting", "talk"],
    "Animals":              ["animal", "butterfly", "chicken", "horse", "turtle"],
    "Art":                  ["music", "painting", "photography"],
    "City & Architecture":  ["architecture", "building", "office", "sculpture"],
    "Education":            ["book", "education", "library", "reading", "school", "teacher"],
    "Food & Drinks":        ["lunch", "meal", "salad", "vegetable"],
    "Nature":               ["coast", "drone"],
    "Others":               ["light", "macro", "mirror"],
    "People":               ["teenager"],
    "Plants":               ["flower", "forest", "grass", "rose", "tulip"],
    "Sports":               ["football", "golf", "soccer"],
    "Technology":           ["ai", "phone", "robot"],
    "Travel":               ["destination", "holiday"],
    "Vehicles":             ["airplane", "bicycle", "boat", "bus", "car"],
    "Weather":              ["cloud", "stars", "storm", "sun", "sunset"],
}
SUB_TO_MAIN = {sub: main for main, subs in MAIN_CAT_MAP.items() for sub in subs}

DISPLAY_COLS = ["filename", "main_category", "category", "resolution",
                "fps", "length", "num_scenes", "aesthetic_score"]

# ── Data loading ───────────────────────────────────────────────────────────────

def _get_csv_path() -> Path:
    """Return a local path to VC-Bench.csv, downloading it from HF Hub if needed."""
    if CSV_PATH.exists():
        return CSV_PATH
    print("VC-Bench.csv not found locally — downloading from HF Hub…")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=HF_DATASET,
        filename="VC-Bench.csv",
        repo_type="dataset",
    ))


def _hf_cdn_url(hf_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/"
        + urllib.parse.quote(hf_path, safe="")
    )


def _infer_hf_path(filename: str, main_cat: str, sub_cat: str) -> str:
    """
    Reconstruct the HF repo-relative path from CSV columns.
    The dataset stores files as  MainCategory\\subcategory\\filename.mp4
    (literal backslashes, matching the original Windows upload layout).
    """
    return f"{main_cat}\\{sub_cat}\\{filename}"


def _build_path_index(hf_dir: Path) -> dict:
    """
    Scan vc-bench-hf for MP4 files and return:
      basename -> {"local": abs_path, "hf_path": repo-relative path}

    HF snapshot_download stores files with literal backslashes in the name on
    Linux, e.g. /vc-bench-hf/Weather\\sunset\\sunset_xyz.mp4.
    p.name therefore IS the repo-relative path (backslashes included).
    """
    index: dict[str, dict] = {}
    for p in hf_dir.rglob("*.mp4"):
        repo_path = p.name                          # e.g. "Weather\sunset\basename.mp4"
        basename = repo_path.rsplit(chr(92), 1)[-1] # just "basename.mp4"
        index[basename] = {
            "local": str(p),
            "hf_path": repo_path,
        }
    return index


def _load_path_index(hf_dir: Path, cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    if not hf_dir.exists():
        return {}
    print("Building path index (first run, may take a moment)…")
    index = _build_path_index(hf_dir)
    try:
        with open(cache_path, "w") as f:
            json.dump(index, f)
        print(f"  Indexed {len(index)} videos → cached at {cache_path}")
    except OSError as e:
        print(f"  Warning: could not write path index cache ({e})")
    return index


def load_dataframe() -> tuple[pd.DataFrame, dict]:
    csv_path = _get_csv_path()
    df = pd.read_csv(csv_path)

    index = _load_path_index(HF_DIR, PATH_INDEX_CACHE)

    df["main_category"] = df["category"].map(SUB_TO_MAIN).fillna("Unknown")

    # Always compute CDN URLs from CSV columns so every row is streamable on
    # Spaces (and as fallback when running locally without a full video mirror).
    df["hf_url"] = df.apply(
        lambda row: _hf_cdn_url(
            index[row["filename"]]["hf_path"]
            if row["filename"] in index
            else _infer_hf_path(row["filename"], row["main_category"], row["category"])
        ),
        axis=1,
    )
    df["local_path"] = df["filename"].map(lambda f: index.get(f, {}).get("local"))
    df["has_video"]  = df["local_path"].notna()

    def fmt_res(r):
        try:
            dims = json.loads(str(r).replace("'", '"').replace("(", "[").replace(")", "]"))
            return f"{dims[0]}×{dims[1]}"
        except Exception:
            return str(r)

    df["resolution"] = df["resolution"].apply(fmt_res)
    df["aesthetic_score"] = df["aesthetic_score"].round(3)

    return df, index


print("Loading VC-Bench data…")
DF, PATH_INDEX = load_dataframe()
_local_count = int(DF["has_video"].sum())
print(f"  {len(DF)} metadata rows · {_local_count} local · {len(DF)} streamable via CDN")

MAIN_CATS   = ["All"] + sorted(MAIN_CAT_MAP.keys())
ALL_SUBCATS = sorted(DF["category"].dropna().unique().tolist())

# ── Helper: filter dataframe ───────────────────────────────────────────────────

def filter_df(main_cat: str, sub_cat: str, search: str) -> pd.DataFrame:
    d = DF.copy()
    if main_cat != "All":
        d = d[d["main_category"] == main_cat]
    if sub_cat != "All":
        d = d[d["category"] == sub_cat]
    if search.strip():
        q = search.strip().lower()
        mask = (
            d["filename"].str.lower().str.contains(q, na=False)
            | d["caption"].str.lower().str.contains(q, na=False)
            | d["category"].str.lower().str.contains(q, na=False)
        )
        d = d[mask]
    return d[DISPLAY_COLS + ["local_path", "hf_url", "caption", "scene_start_frames"]].reset_index(drop=True)


def get_sub_cats(main_cat: str) -> gr.Dropdown:
    if main_cat == "All":
        choices = ["All"] + ALL_SUBCATS
    else:
        choices = ["All"] + sorted(MAIN_CAT_MAP.get(main_cat, []))
    return gr.Dropdown(choices=choices, value="All")

# ── Statistics plots ───────────────────────────────────────────────────────────

def make_stats_plots():
    d = DF.copy()

    fig_cat = px.bar(
        d.groupby("main_category").size().reset_index(name="count")
          .sort_values("count", ascending=True),
        x="count", y="main_category", orientation="h",
        title="Videos per Main Category",
        color="count", color_continuous_scale="Blues",
        labels={"main_category": "", "count": "# videos"},
    )
    fig_cat.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=160, r=20, t=40, b=40))

    fig_aes = px.histogram(
        d.dropna(subset=["aesthetic_score"]),
        x="aesthetic_score", nbins=40,
        title="Aesthetic Score Distribution",
        color_discrete_sequence=["#636EFA"],
        labels={"aesthetic_score": "Aesthetic Score"},
    )
    fig_aes.update_layout(margin=dict(t=40, b=40))

    dur_frames = pd.to_numeric(d["length"], errors="coerce").dropna()
    avg_fps    = pd.to_numeric(d["fps"], errors="coerce").mean()
    dur_secs   = dur_frames / avg_fps
    fig_dur = px.histogram(
        x=dur_secs, nbins=40,
        title=f"Duration Distribution (frames ÷ avg fps={avg_fps:.1f})",
        color_discrete_sequence=["#EF553B"],
        labels={"x": "Duration (s)"},
    )
    fig_dur.update_layout(margin=dict(t=40, b=40))

    fig_scenes = px.histogram(
        d.dropna(subset=["num_scenes"]),
        x="num_scenes", nbins=30,
        title="Number of Scenes per Video",
        color_discrete_sequence=["#00CC96"],
        labels={"num_scenes": "# Scenes"},
    )
    fig_scenes.update_layout(margin=dict(t=40, b=40))

    return fig_cat, fig_aes, fig_dur, fig_scenes

# ── Gradio UI ──────────────────────────────────────────────────────────────────

def on_filter(main_cat, sub_cat, search):
    filtered = filter_df(main_cat, sub_cat, search)
    display  = filtered[DISPLAY_COLS].copy()
    count    = f"**{len(filtered)} videos** match the current filters."
    return display, filtered, count


def on_row_select(evt: gr.SelectData, filtered_state):
    if filtered_state is None or evt is None:
        return None, "Select a row to see details."
    row_idx = evt.index[0]
    row = filtered_state.iloc[row_idx]

    # Prefer a local file; fall back to HF CDN stream.
    video_src = row.get("local_path") or row.get("hf_url")

    md = f"""
**File:** `{row['filename']}`

| Field | Value |
|---|---|
| Main category | {row.get('main_category', '—')} |
| Subcategory | {row.get('category', '—')} |
| Resolution | {row.get('resolution', '—')} |
| FPS | {row.get('fps', '—')} |
| Length (frames) | {row.get('length', '—')} |
| # Scenes | {row.get('num_scenes', '—')} |
| Scene starts | `{row.get('scene_start_frames', '—')}` |
| Aesthetic score | {row.get('aesthetic_score', '—')} |

**Caption:**
> {row.get('caption', '—')}
"""
    return video_src, md


def build_ui():
    fig_cat, fig_aes, fig_dur, fig_scenes = make_stats_plots()

    _local = int(DF["has_video"].sum())
    _video_note = (
        f"**{_local} local · {len(DF)} streamable via HF CDN**"
        if _local > 0
        else f"**{len(DF)} streamable via HF CDN**"
    )

    with gr.Blocks(title="VC-Bench Explorer") as demo:
        gr.Markdown("# VC-Bench Dataset Explorer")
        gr.Markdown(
            f"Browsing **{len(DF)} metadata entries** · "
            + _video_note + " · "
            f"**{len(MAIN_CAT_MAP)} main categories** · "
            f"**{len(ALL_SUBCATS)} subcategories**"
        )

        with gr.Tabs():
            # ── Browse tab ─────────────────────────────────────────────────────
            with gr.TabItem("Browse"):
                with gr.Row():
                    dd_main = gr.Dropdown(
                        choices=MAIN_CATS, value="All",
                        label="Main category", scale=2,
                    )
                    dd_sub = gr.Dropdown(
                        choices=["All"] + ALL_SUBCATS, value="All",
                        label="Subcategory", scale=2,
                    )
                    txt_search = gr.Textbox(
                        placeholder="Search filename / caption…",
                        label="Search", scale=3,
                    )

                count_md = gr.Markdown(f"**{len(DF)} videos** match the current filters.")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        tbl = gr.DataFrame(
                            value=DF[DISPLAY_COLS],
                            interactive=False,
                            wrap=False,
                        )
                    with gr.Column(scale=2):
                        video_out = gr.Video(label="Video", height=320)
                        meta_md   = gr.Markdown("*Select a row to see video and metadata.*")

                filtered_state = gr.State(DF[DISPLAY_COLS + ["local_path", "hf_url", "caption", "scene_start_frames"]])

                dd_main.change(get_sub_cats, inputs=dd_main, outputs=dd_sub)

                for trigger in [dd_main, dd_sub, txt_search]:
                    trigger.change(
                        on_filter,
                        inputs=[dd_main, dd_sub, txt_search],
                        outputs=[tbl, filtered_state, count_md],
                    )

                tbl.select(on_row_select, inputs=filtered_state, outputs=[video_out, meta_md])

            # ── Statistics tab ────────────────────────────────────────────────
            with gr.TabItem("Statistics"):
                gr.Markdown("### Dataset overview")
                with gr.Row():
                    gr.Plot(value=fig_cat, label="Category distribution")
                    gr.Plot(value=fig_aes, label="Aesthetic scores")
                with gr.Row():
                    gr.Plot(value=fig_dur, label="Duration")
                    gr.Plot(value=fig_scenes, label="Scene count")

                gr.Markdown(f"""
**Quick facts**
- Total metadata rows: **{len(DF):,}**
- Videos available locally: **{int(DF['has_video'].sum()):,}** ({DF['has_video'].mean()*100:.1f} %)
- Average aesthetic score: **{DF['aesthetic_score'].mean():.3f}**
- Average length: **{DF['length'].mean():.0f} frames**
- FPS values: {sorted(DF['fps'].dropna().unique().astype(int).tolist())}
""")

    return demo


# Build the demo at module level so HF Spaces can find it.
demo = build_ui()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VC-Bench Dataset Explorer")
    parser.add_argument("--share", action="store_true", help="Create a 72-hour public Gradio link")
    parser.add_argument("--port",  type=int, default=7860, help="Local port (default 7860)")
    parser.add_argument("--host",  default="0.0.0.0",    help="Bind address (default 0.0.0.0)")
    args = parser.parse_args()

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
