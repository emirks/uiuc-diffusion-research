# VC-Bench Dataset Explorer — Research, Evaluation & Proposal

Goal: interact with VC-Bench samples — inspect statistics, browse samples, view metadata (CSV + 1,261 MP4s in `data/raw/vc-bench-hf`).

---

## 1. Research: Tools for Inspecting Datasets Like This

### A. FiftyOne (Voxel51)

- **What it is:** Open-source Python library + desktop-style App for CV datasets (images, **videos**, 3D). [GitHub](https://github.com/voxel51/fiftyone) ~9.5k stars.
- **Features:** Load from directory or **CSV**; attach arbitrary metadata to samples; filter/sort/query; play **video** in-app; run embeddings/models; no duplication (stores filepaths).
- **Fit for VC-Bench:** Native **video** + **CSV** support. `fo.types.CSVDataset` with `media_field` and custom `fields` can map our CSV (filename, resolution, length, fps, category, caption, etc.) to sample attributes. Video directory can be separate; we only need to resolve CSV `filename` → full path (e.g. under `vc-bench-hf` with backslash paths).
- **Pros:** Purpose-built for CV/video; rich UI (grid, filters, video player); no server if run locally; can add evaluations/embeddings later.
- **Cons:** Heavier dependency; CSV path resolution for our layout (filename vs `Category\subcat\file.mp4`) needs a small adapter or extra column.

### B. df-gallery

- **What it is:** Lightweight Python tool to build **filterable HTML image galleries** from a directory or from CSV/JSON. [PyPI](https://pypi.org/project/df-gallery/).
- **Features:** `dfg build annotations.csv --img-root ...` → single HTML with pandas-style filters, histograms, no server.
- **Fit for VC-Bench:** Designed for **images** (jpg, png, gif, webp). **No native video playback.** Would require extracting frames/thumbnails per video and pointing the gallery at those images (plus metadata from CSV).
- **Pros:** Very light; portable HTML; great for image + metadata audit.
- **Cons:** Not video-first; extra step to generate thumbnails; no in-app video inspection.

### C. Hugging Face Data Studio / DuckDB

- **What it is:** HF’s dataset viewer + SQL (DuckDB) over Parquet in the browser or locally.
- **Fit for VC-Bench:** Our data is **local** (CSV + MP4s), not loaded via `datasets` on HF. We could export CSV as Parquet and query with DuckDB, but **no built-in video player**; better for tabular analytics than for “click and watch this video.”
- **Pros:** Strong for SQL/aggregations if we only need stats.
- **Cons:** No integrated video browsing; workflow is separate from our folder layout.

### D. Custom app (Gradio or Streamlit)

- **What it is:** Small Python UI: load CSV (pandas), show table + filters, video player (Gradio `gr.Video` or Streamlit `st.video`), and summary stats (value_counts, describe, plots).
- **Fit for VC-Bench:** Full control: resolve `filename` → path under `vc-bench-hf`, filter by category/length/etc., play selected video, show caption and all CSV columns.
- **Pros:** Minimal dependencies (gradio or streamlit + pandas); exactly matches our schema; easy to add stats (e.g. by category, resolution, length).
- **Cons:** You maintain the code; less “dataset tooling” (e.g. embeddings, model runs) than FiftyOne.

### E. Other (VFRAME, Probe, etc.)

- **VFRAME:** Good for **batch** media attributes (resolution, duration) and plots; not an interactive sample browser.
- **Probe.video / FFprobe:** Single-file or API metadata; not a dataset-level explorer.

---

## 2. Evaluation & Reasoning

| Criterion              | FiftyOne | df-gallery | HF/DuckDB | Custom (Gradio/Streamlit) |
|------------------------|----------|------------|-----------|----------------------------|
| Video playback         | ✅ In-app| ❌ (images only) | ❌       | ✅                         |
| CSV + metadata         | ✅       | ✅ (if image path) | ✅ (tabular) | ✅                |
| Statistics / plots     | ✅       | ✅ (distributions) | ✅ (SQL) | ✅ (pandas + plots)       |
| Filter by category etc.| ✅       | ✅         | ✅        | ✅                         |
| Zero code (use as-is)  | ⚠️ Need 1 script to load | ⚠️ Need thumbnails | ⚠️ Export + DB | ❌ (you code) |
| Lightweight            | ❌       | ✅         | ✅        | ✅                         |
| Extensible (models etc.)| ✅       | ❌         | ❌        | ⚠️                         |

- **Best “tool” if you want a ready-made dataset workbench:** **FiftyOne** — video + CSV + filters + playback in one place; one-off script to build the dataset (resolve paths, map CSV columns to sample fields).
- **Best “minimal code” for “stats + pick sample + watch video”:** **Custom Gradio (or Streamlit) app** — single script, pandas + CSV, dropdown/filters, video component, and a small resolver from `filename` to `vc-bench-hf` path; no new concepts, easy to tweak.
- **df-gallery:** Only if you are happy with **image thumbnails** instead of video (e.g. for a quick visual audit); then you’d add a step to extract one frame per video and optionally merge with CSV.

---

## 3. Proposal

**Recommended: start with one of these two.**

### Option 1 — FiftyOne (recommended if you want a full dataset explorer)

- **Use case:** Browse and filter all samples, play videos, see all CSV fields, and later add embeddings or model runs.
- **Steps:**
  1. Add a small Python script that: (a) reads `VC-Bench.csv`, (b) for each row, resolves `filename` to the actual file path under `vc-bench-hf` (e.g. by scanning `rglob("*.mp4")` and matching basename or by building `Category\subcategory\filename` from CSV if available), (c) creates a FiftyOne dataset with `fo.Sample(filepath=..., ...)` and attaches CSV columns as sample fields (e.g. `category`, `length`, `caption`, `aesthetic_score`).
  2. Install FiftyOne: `pip install fiftyone`.
  3. Run the script once to build the dataset, then launch the app: `fiftyone app launch` (or from Python `session = fo.launch_app(dataset)`).
- **Result:** Interactive UI: grid of samples, click to play video, filter by category/length/etc., view metadata in the sidebar.

### Option 2 — Lightweight Gradio app (recommended if you want minimal and explicit)

- **Use case:** “I want to see stats, filter by category, pick a row, and watch the video with caption/metadata.”
- **Steps:**
  1. One Python file (e.g. `scripts/vc_bench_explorer.py` or under `experiments/`): load CSV with pandas, build a mapping `filename → full path` (same resolution logic as peek script: match basename against `Path(vc_bench_hf).rglob("*.mp4")`), then Gradio interface: `gr.Dataframe` (filterable) or dropdown for category + row selector, `gr.Video` for playback, `gr.JSON` or text for metadata, plus `gr.Plot` or markdown for simple stats (e.g. value_counts for category, describe for length/fps).
  2. Run: `gradio scripts/vc_bench_explorer.py` (or `streamlit run ...` if you prefer Streamlit).
- **Result:** Browser UI: table or filters → select sample → video + metadata; optional stats tab.

**If you only need aggregate statistics and no video:** Use pandas + Jupyter, or export CSV to Parquet and use DuckDB from the CLI/UI; no need for a dedicated explorer app.

---

## 4. Summary

- **Best ready-made tool for “inspect statistics, samples, metadata” with video:** **FiftyOne** — one loader script, then use the App.
- **Best “I’ll code it” option:** **Gradio (or Streamlit)** — single script, full control, video + table + stats.
- **df-gallery:** Only if you’re fine with **image thumbnails** and will generate them from the MP4s; not for direct video interaction.

Next step: either (1) add a FiftyOne dataset loader for VC-Bench + short instructions to launch the app, or (2) add a minimal Gradio app that loads the CSV, resolves paths, and provides filters + video player + stats. Say which you prefer and we can implement it.
