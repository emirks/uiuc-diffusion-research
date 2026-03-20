---
title: vc-bench-explorer
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.9.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# VC-Bench Dataset Explorer

Interactive Gradio app for browsing and analysing the [VC-Bench](https://huggingface.co/datasets/Kevinson-lzp/VC-Bench) video dataset.

## Features

- **Browse** all videos with filters by main category, subcategory, and free-text search
- **Watch** any video inline — streams directly from the HF dataset CDN (no local copy needed)
- **Inspect** per-video metadata: resolution, FPS, duration, scene count, aesthetic score, caption
- **Statistics** tab with distribution plots across categories, aesthetic scores, duration, and scene counts

## Usage

### HF Spaces (this page)

The app downloads `VC-Bench.csv` from the dataset hub on startup and streams all 1 261 videos via the HF CDN. No data upload required.

### Run locally

```bash
pip install -r requirements.txt
python app.py
```

To point at a local mirror of the dataset (for faster video playback):

```bash
VC_BENCH_HF_DIR=/path/to/vc-bench-hf python app.py
```

### Share a temporary public link

```bash
python app.py --share
```
