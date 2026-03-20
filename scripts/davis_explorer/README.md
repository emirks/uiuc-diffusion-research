---
title: DAVIS-Dataset-Explorer
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# DAVIS Dataset Explorer

Interactive browser for the **DAVIS 2017 Video Object Segmentation** benchmark (480p split).

## Features

| Tab | What it does |
|-----|-------------|
| 📋 Browse | Filter & search all 90 sequences; click a row to inspect metadata |
| 🔍 Viewer | Frame-by-frame scrubber with DAVIS palette mask overlay; single-sequence video playback |
| 🖼 Gallery | Thumbnail grid of all sequences; click a thumbnail to instantly play it |
| 📺 Multi-Video | Paged 3×3 video grid — page through all 90 sequences, 9 at a time |
| ⚖️ Compare | Up to 6 sequences side-by-side |
| 📊 Statistics | Distribution plots (frames, objects, splits, resolution) |

## First-run behaviour

On the first startup the app downloads **DAVIS-2017-trainval-480p.zip** (~800 MB)
from the official ETH Zurich server and extracts it to `DAVIS_ROOT`.

With HF Spaces **persistent storage** the data is stored under `/data/DAVIS` and
the MP4 cache under `/data/DAVIS_explorer_cache` — both survive restarts.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DAVIS_ROOT` | `/data/DAVIS` (Spaces) or local path | Dataset root directory |
| `DAVIS_CACHE_DIR` | `DAVIS_ROOT/../DAVIS_explorer_cache` | Where encoded MP4s are stored |

## Dataset

DAVIS 2017 — *The 2017 DAVIS Challenge on Video Object Segmentation*  
Pont-Tuset et al., arXiv:1704.00675  
Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
