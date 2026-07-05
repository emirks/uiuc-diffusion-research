# exp_055 — Interactive HTML viewer for transition-eval-harness results

## Question

Can a human interactively explore an entire eval-harness run — every scored
pair and its M1–M6 metrics, core-frames labelled, videos playable inline, the
retrieval exam, figures, score tables, adversarial checks — from a single
self-contained page that a static file server can host? Pure front-end +
data-packaging (no GPU, no DINO/torch recompute).

## Setup

A **general bundle generator** (`build_viewer.py`) consumes any harness run and
emits a self-contained viewer under `<out>/`:

- `data.json` — everything the UI needs (schema below).
- `assets/videos/` — **relative symlinks** to generated / reference / condition
  / lerp-floor mp4s (big & numerous — symlinked, never copied; the bundle is
  11 MB on disk, ~200 MB dereferenced).
- `assets/filmstrips/*.jpg` — **labelled core-frame filmstrips** per clip:
  the mp4 is decoded, the harness's OWN `morph_profile` + `core_mask` (loaded
  directly from `src/diffusion/transition_eval/morph.py`) identify endpoint vs
  **core/effect** frames, and PIL burns colour-coded borders + captions
  (endpoint A / endpoint B / core / transition).
- `assets/figures/*.png` — copies of the harness figures (portable).
- `index.html` — copied verbatim from `viewer_template.html`: a single-page
  app, **inline CSS + vanilla JS, zero CDN/network**. It `fetch()`es
  `./data.json` and loads assets by relative path.

**No GPU / no torch.** Per-frame DINO features are read from the harness's
existing `outputs/eval/cache/dino_arr_*.npz` cache (resolved by reproducing
`features.file_key`); the diffusion package's heavy `__init__` is bypassed via
`importlib` so only numpy/PIL/PyAV load. The morph/core-frame labelling is thus
faithful to the harness, not a re-implementation.

### Viewer sections
- **Overview** — corpus summary (styles × clips, deduped/new badges), the exam
  accuracies per metric with **Wilson 95% error bars** + chance line, the
  HEADLINE + ANALYSIS score tables with †/‡ trust flags rendered & explained,
  the adversarial-checks verdicts (Check A/B/C, audit-aware).
- **Items** — filter by arm/style, sort by any metric, search item_id; each
  detail shows generated + reference (+ condition) videos playable inline
  (lazy-loaded), the labelled core-frame filmstrip, an interactive
  **morph-profile plot** (â/b̂ curves, endpoint windows + core band shaded,
  crosshair tooltip), every metric (raw / floor / ceiling / normalized,
  appearance, motion, morph-DTW, endpoint DINO+LPIPS, seam z, leakage) and the
  judge q1–q5 with evidence.
- **Exam & Pairs** — per-metric pairwise **distance heatmap** (hover, click to
  load a pair), **confusion matrix**, LOO 1-NN retrieval list (query→NN,
  flips-only toggle), and a **two-clip comparator** (distances + core-frame
  filmstrips side by side).
- **Figures** — all PNGs grouped, click-to-zoom lightbox.
- **About** — sources, M1–M6 glossary, dedup clusters, the run's `report.md`.

Light/dark neutral theme (validated dataviz palette; toggle with the header
button or `t`), keyboard nav (`[` / `]` between tabs, `t` theme), lazy videos.

## How to run

Build the **example bundle** from the existing exp_053 results:

```bash
$LAB/envs/diffusion/bin/python experiments/exp_055_eval_viewer/run.py
```

Then serve & open:

```bash
cd outputs/eval/exp_053/viewer && python -m http.server 8000
# open http://localhost:8000/
```

Validate integrity without a browser (asserts every asset path in data.json
exists & is non-empty, JSON parses, index.html present):

```bash
$LAB/envs/diffusion/bin/python experiments/exp_055_eval_viewer/validate_bundle.py \
    outputs/eval/exp_053/viewer
```

For a **fresh run** (e.g. a new full-corpus validation), call the general CLI
directly — nothing is hardcoded to exp_053; optional inputs degrade gracefully:

```bash
$LAB/envs/diffusion/bin/python experiments/exp_055_eval_viewer/build_viewer.py \
    --validation <val_dir> --items <items.jsonl> --manifest <manifest.json> \
    [--ceilings ...] [--judge-summary ...] [--judge-results ...] \
    [--checks <checks_dir>] [--report <report.md>] [--dedup ...] \
    [--controls <controls_dir>] [--figures-dir <dir> ...] \
    --out <out_dir> --label "<title>"
```
(only `--validation` is required; a validation-only run with no items still
renders the Overview/Exam/Figures/About tabs.)

## Outputs

- Generator: `experiments/exp_055_eval_viewer/build_viewer.py`
- Template: `experiments/exp_055_eval_viewer/viewer_template.html`
- Preset wrapper + config: `run.py`, `config.yaml`
- Validator: `validate_bundle.py`
- **Example bundle: `outputs/eval/exp_053/viewer/`** (`index.html`, `data.json`,
  `assets/`). 24 items, 41-clip exam, 61 filmstrips, 13 figures. Integrity:
  238/238 referenced assets resolve & non-empty.

### data.json schema (brief)
```
meta        {label, generated_at, generator, sources{...}, n_items, arms[], styles_in_items[]}
glossary    {metrics{M1..M6}, normalization, trust_flags, rubric, fail_if_true}
corpus      {styles[{style, n_clips, n_dup, clips[], in_exam, trust{motion_trusted,ceiling_trusted,n_ref_clips,motion_recall}}],
             total_clips, exam_n_clips, controls{style:[lerp mp4 rel paths]}}
exam        {names[41], styles[41], classes[9],
             metrics{metric:{accuracy_1nn, accuracy_wilson95[lo,hi], k_correct, n_total, chance,
                             per_class_recall{}, per_class_wilson{}, per_class_n{}, confusion{},
                             within_mean, cross_mean, separation_cohens_d}},
             matrices{metric:[[41x41 distances]]},
             retrieval_examples{metric:[{q,q_style,nn,nn_style,dist,correct}]},
             clip_records{name:{video, filmstrip, profile{a_hat[],b_hat[],core_idx[],cross,scalars,...}}},
             lerp_floor{}, clip_scalars{}}
score_tables{headline[{arm,n,appearance/motion/endpoint_dino/max_seam_z/leak_max_sim:{mean,std,n,flag}, judge_pass}],
             analysis[{arm, profile_dtw_norm{...}, depth/depart/arrive/core_frac/leak_excess{...}, cross_high_items}]}
judge       {summary{arm:{q1..q5,all_pass,n_parsed}}, rubric{}, fail_if_true[]}
checks      {checkA, checkB, checkC, checkC_audit}   adversarial[{item_id,arm,leak_max_sim_target,...}]
dedup       {method, clusters[]}   figures[{name,path,caption,group}]   report_md (verbatim)
items[]     {item_id, arm, style, n_endpoints, notes, videos{generated,prefix,suffix},
             filmstrip, profile{a_hat[],b_hat[],n_frames,n_prefix,n_suffix,core_idx[],cross,cross_high,scalars,columns[]},
             ref_clips[], ceiling{}, metrics{all M1–M6 raw+norm+floor+ceil}, judge{q1..q5{answer,evidence},all_pass,model_version}}
```

### Consumes (exp_053 example)
`outputs/eval/exp_052/validation/run_0001` (results.json + distance_matrices.npz
+ exam PNGs) · `outputs/eval/exp_052/ladder/run_0001/{items.jsonl,ceilings.json}`
· `experiments/exp_052_transition_eval_harness/manifest_exp051.json` ·
`outputs/eval/exp_053/judge_gemini_ladder/run_0004/{judge_summary,judge_results}.json`
· `outputs/eval/exp_053/checks/run_0001/` · `outputs/eval/exp_053/ladder_v2/run_0002/report.md`
· `outputs/eval/exp_053/{pair_examples,dedup}/` · `outputs/eval/exp_052/controls/`
· `outputs/eval/cache/dino_arr_*.npz` (read-only) · `data/processed/transitions/`.

### Notes / skipped assets
- 4 / 41 exam reference clips have no filmstrip (`melt/jump/display/flying_cam`
  clips moved to `_dup/` after the exp_052 exam → their content-keyed feature
  cache no longer resolves). Videos still symlink where the file exists; the UI
  shows "filmstrip unavailable" for the 3 that resolve and skips the 1 fully
  removed. Everything else (24/24 items, 37/41 refs) has a filmstrip.
- New styles `air_bending`, `firelava` show in the corpus with a **new** badge
  (uncached, not yet in the exam) — demonstrates the generator generalises to a
  growing corpus.
```
