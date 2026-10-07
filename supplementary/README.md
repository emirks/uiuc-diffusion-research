# SEGUE — supplementary website

A static video-results page for the SEGUE paper (double-blind ICLR 2027 submission),
built by adapting the [Video-As-Prompt](https://github.com/bytedance/Video-As-Prompt)
project page with minimal changes. The page is **anonymous**: no author names, no
institution, no external code/model/dataset links.

The deliverable is `site/` — self-contained and relocatable (every path relative). Copy
`site/` to any static host or a `gh-pages` branch and it works as-is.

```
supplementary/
├── build_site.py          the generator (python3.12, stdlib + ffmpeg subprocess)
├── template.html          VAP index.html, hero swapped, content -> {{SECTIONS}}, +one CSS block
├── site_manifest.json     the resolved rows/sections (tracked)
├── site/
│   ├── index.html         generated page (tracked)
│   ├── videos/…           re-encoded clips        (gitignored — regenerable)
│   ├── stills/…           endpoint frame JPEGs    (gitignored — regenerable)
│   └── media_manifest.json dest -> {source, sha256, frames/fps/w/h, bytes} (gitignored)
├── vap_gh_pages/          pristine VAP gh-pages clone (gitignored; NEVER edited)
└── .gitignore
```

## Rebuild

```bash
/usr/bin/python3.12 supplementary/build_site.py --all     # select + media + html
/usr/bin/python3.12 supplementary/build_site.py --check    # verify every ref resolves
```

Stages: `--select` (resolve rows -> `site_manifest.json`), `--media` (transcode ->
`site/videos`, `site/stills`, `media_manifest.json`; skip-if-present), `--html` (render
`site/index.html` from `template.html`), `--check`. Login node: uses the imageio_ffmpeg
binary when there is no system ffmpeg, `-threads 2`, at most 3 concurrent ffmpeg.

> **2026-09-25 — row source is now the owner's picks (this section describes the earlier
> win-all/6-row build; see "Site plan + curation overlay" below for the current behaviour).**
> A row is on the site **iff it has a pick** in
> `eval_ladder/viewer/collections/supplementary_picks.json`. Every section shows all its picked
> rows; the SEGUE clip in every section is the **picked clip** (`pick.video`, any arm/variant/
> seed the owner chose). A/B additionally take back the rows the 2026-09-24 judge win-all filter
> dropped — merged from the pre-filter snapshot exactly like the picker (live items first,
> snapshot extras appended, same ids); each row's `judge` verdict is kept as data for the
> overlay. Reference video + endpoint clips come from the collection item's `inputs`; A/B
> opponents come verbatim from the human study's `pairs.json` where the row was in the study,
> else from the item's prior-work `columns` (seed 42).

## Sections and where the rows come from

All four sections are generated from `eval_ladder/viewer/collections/neutral_effect_collections.json`.
The SEGUE system throughout is **arm `dualforce_dcg_w6`, NEUTRAL prompt, seed 42**
(grids `03_neutral_v3` for HF references, `04_neutral_v3ed81` for EffectData references).

- **A — Comparisons, Transition Effect Generation (both endpoints given).** The 24 rows of
  collection `teg_user_study`: the rows where SEGUE won the blind VLM-judge transition
  question against **every** opponent (`judge.transition_vs` all 1.0). The SEGUE clip, the
  two opponent clips (Base LTX-2, refVFX), the reference video and the endpoint clips are
  taken **verbatim from the human study** (`misc/2026-09-22_user_study/pairs.json`, task
  `two`, matched on endpoint + reference), so the page shows exactly what was judged.
- **B — Comparisons, Visual Effect Transfer (start given).** The 26 rows of
  `vfx_transfer_user_study`, same rule, opponents VAP / VFXMaster / refVFX (pairs.json task
  `one`).
- **C — SEGUE results, Transition Effect Generation.** The rows of `supplementary_teg` that
  are not already in A (matched on (endpoint, reference clip)); 33 candidates, **32 shown**
  (1 excluded — see below). Grouped by reference clip. The SEGUE clip is resolved through the
  viewer payload `outputs/reports/iclora_neutral_effect_v2/data.js`: the card whose key is
  the item's `card_key` base (`donor|endpoint|sided`, dropping any `|ref=` suffix), the
  generation in `slots["dualforce_dcg_w6_neutral_v3"|"…v3ed81"]` with `g.ref` == the
  reference, then `g.videos["42"]`. That path is a hardlink of the store file, so the
  **store path** (`store/gens/032_dualforce_dcg_w6/{03_neutral_v3,04_neutral_v3ed81}__dai/
  videos/…`) is what gets recorded — identical bytes, consistent with A/B. Reference video =
  the card's `ref_video`; endpoint clips = the card's `prefix_video` / `suffix_video`.
- **D — SEGUE results, Visual Effect Transfer.** The rows of `supplementary_vfx_transfer`
  not already in B; **199 shown**. Same reference-grouped layout, start clip only.

**Excluded (1):** `supplementary_teg` row `hero_flight|hero_flight_6|two|ref=shadow_smoke_0`
(a cross/mismatched reference) has no `dualforce_dcg_w6` neutral v3 generation — `present:
false` in the collection and no file on disk — so it cannot show a SEGUE clip and is dropped.
`build_site.py --select` records it under `excluded` and prints it loudly. Every other source
clip is asserted to exist; a genuinely missing source fails the build.

## Site plan + curation overlay

Order and hidden state live in **`eval_ladder/viewer/collections/site_plan.json`** (inside the
static server's POST allow-list), not in code. Schema:

```
{"schema":1,"updated":ISO,"sections":{
  "C":{"class_order":[cls…],"block_order":{cls:[block_id…]},"cell_order":{block_id:[item_id…]},
       "hidden_blocks":[block_id…],"hidden_cells":{block_id:[item_id…]},"merge_into":{src_block_id:dst_block_id}},
  "D":{ …same… },
  "A":{"class_order":[cls…],"row_order":{cls:[item_id…]},"hidden_rows":[item_id…]},
  "B":{ …same as A… }}}
```

`build_site.py --html` **seeds** the plan only if it is absent (automatic order + the migrated
code overrides — the old `BLOCK_DROP`/`BLOCK_MOVE_AFTER`/`BLOCK_MERGE_INTO`/`CELL_DROP` are gone
and their effect is baked into the seed by `item_id`), and **always reads** it: hidden entries
are not rendered on the public build, orders are applied, picked rows/blocks missing from the plan
are appended in automatic order, and plan entries whose row no longer exists are ignored. Ordering
is by transition class (one shared heading per class, references consecutive; class order by
picked-row count desc, ties alphabetical); within a class blocks/references are cells desc; within
a block the cell (endpoint) order follows `cell_order` (galleries C/D) when present, else tier order
seen → unseen → zero-shot (cells missing from `cell_order` append in that automatic order; unknown
ids ignored; hidden cells are still dropped on the public build). `merge_into` folds a block's cells into
another's; the old cross-class move (water_bending after flame_transition) is intentionally dropped
because per-class grouping cannot express it.

The **curation overlay** is local-only: the static HTML is byte-identical on every host and carries
no overlay markup; a script builds all controls at runtime **only** when
`location.hostname` is `localhost`/`127.0.0.1`. It fetches `site_plan.json` (via the symlink
`site/site_plan.json -> ../../eval_ladder/viewer/collections/site_plan.json`), un-hides the
`data-hidden` markup (dimmed + striped, with an unhide button), and offers per-class ▲▼, per-block/
row ▲▼ + hide/unhide, per-cell ◀ ▶ (reorder within its block, galleries only) + hide/unhide, and a
judge badge on A/B rows. Clicking **hide** sends the item to the END of its container (cell → end
of its block, block/row → end of its class), so it lands last in the saved order; **unhide** leaves
it in place. A sticky-bar **Show hidden / Hide hidden** toggle (persisted in `localStorage`, default
show) switches between the dimmed view and a public-page preview where hidden items vanish; the bar
also shows an "n hidden" count. "Save plan" POSTs the
plan rebuilt from the current DOM with optimistic concurrency (`X-Base-Updated`); on 409 it
re-fetches, rebuilds on top and retries once, else keeps a `localStorage` mirror and shows the
error. Hidden clips are emitted as inert markup with lazy `data-src` (never fetched on the public
page); the overlay loads them when it un-hides them. Rebuild any time with
`build_site.py --html` — the public build applies the same plan server-side.

## Re-encode recipe

Copied from `scripts/user_study/build_media.py` (the human-study recipe):

- clips: `libx264 -pix_fmt yuv420p -crf 26 -preset medium -an -movflags +faststart -threads 2`,
  `-map 0:v:0`, native fps and frame count, no audio.
- stills: `-vf select=eq(n\,N) -frames:v 1 -q:v 3` (mjpeg ≈ JPEG q90). N = 0 for the start
  frame, 8 (last of the 9-frame conditioning clip) for the end frame. Stills are **not shown
  on the page** (which plays the endpoint clips) but are generated for the registry `stills`
  link and for reuse; they use the same extractor as the study so they are pixel-consistent.

Media is deduped by source: each unique source file is encoded once and duplicate
destinations are hardlinked, so reruns and copies stay cheap.

## Layout (owner will restyle)

The page keeps VAP's look, fonts, section-title/description/footnote pattern, lazy-load
JS and slider code verbatim. All media are portrait 480×640. One added `<style
id="segue-layout">` block drives the cell layout (see the deviations list below). Sections
A/B are one horizontal **strip** per row; C/D are **blocks** grouped by reference (the
reference once on the left, then one cell per endpoint flowing right and wrapping).

## Publish

```bash
cp -r supplementary/site/ <somewhere>/            # or into a gh-pages worktree
```

Everything under `site/` is relative, so it serves from any root.

## Deliberate deviations from VAP's markup / JS

1. `<title>` and the whole `<header>` (title/authors/affiliation/buttons/bibtex) are swapped
   for the anonymous SEGUE hero; the arXiv/GitHub/HF buttons and the BibTeX block (author
   names) are dropped. One `href="#"` Paper button remains.
2. The content region between `</header>` and the two container-closing `</div>` is replaced
   by a single `{{SECTIONS}}` placeholder. VAP's own page has a stray extra `</div>` there
   (its markup is `div opens - closes = -1`); the template and the rendered page preserve
   that exact imbalance rather than "fixing" VAP.
3. VAP's `method.png` and all VAP demo videos are not used.
4. One **added** `<style id="segue-layout">` block after VAP's `</style>` (VAP's CSS is
   byte-verbatim). Its classes: `.seg-section` (`--seg-h` = generation height), `.seg-strip`
   (A/B rows), `.seg-block` + `.seg-cells` + `.seg-cell` + `.seg-cellwrap` (C/D grouped
   blocks), `.seg-unit`, `.seg-ref` / `.seg-gen` / `.seg-startfull` (full-height boxes),
   `.seg-stack` + `.seg-endpoint` (the start/end pair, together one generation tall),
   `.seg-noend` (hatched + diagonal-cross placeholder where no end is given, section D),
   `.seg-media` (portrait 3/4 video box), `.seg-badge` (start/end corner tag), `.seg-label`
   / `.seg-rowlabel` / `.seg-tier` (captions). Nothing in VAP's CSS was changed.
5. Our video cells use `<video class="lazy-video" preload="none" data-autoplay="true" muted
   playsinline loop>` with a plain `<source src="…">` — the same lazy-load contract VAP's
   IntersectionObserver script drives. We do **not** use VAP's before/after slider markup
   (`.prompt-wrap .container` with two `#media1`/`#media2` videos), so VAP's slider,
   overlay-card, and `srviump` transform scripts are retained verbatim but are inert on this
   page (their trigger elements/classes never appear). The lightbox markup and its script
   are kept so the retained JS finds its elements.

## Picker (internal)

`supplementary/picker/` is an **internal, unblinded** tool (not part of the public `site/`) for
choosing the best "ours" SEGUE clip per row before finalizing the public page. It reuses this
folder's template/CSS but plays the **original** store clips directly — **no transcode**.

```bash
/usr/bin/python3.12 supplementary/picker/build_picker.py            # render picker/site/index.html (default)
/usr/bin/python3.12 supplementary/picker/build_picker.py --check    # verify every media ref resolves
```

- **What it shows.** The same four sections/order as the public page, but **every row** of the
  four collections (A `teg_user_study` 24, B `vfx_transfer_user_study` 26, C `supplementary_teg`
  54 — including the one row with no w=6 gen, D `supplementary_vfx_transfer` 224) and, per row,
  **all candidate generations that were open when the row was saved**: every `columns[]` entry
  with `present:true`, every gen in `gens[]` (seeds 42/43), **except any column whose `arm` is in
  `EXCLUDE_ARMS`** (see below). 3,698 candidate clips total (3,398 own + 300 prior-work). A column
  is **pickable ("ours")** unless its `category` is one of the two prior-work categories `external`
  ("Visual effect transfer · prior work") or `teg` ("Transition effect generation (TEG) · prior
  work…"); prior-work clips are greyed and non-pickable.
- **Excluded arms (`EXCLUDE_ARMS`).** Owner 2026-09-24 ("i'll never choose base ltx 2, remove from
  all"): `build_picker.py` drops every candidate column whose `arm` id is in
  `{"base_cond","base_prompt"}` (category `baseline`, labels "Base LTX-2 …") from all four sections
  — keyed on the arm id, not the label. In this data only section A carried Base LTX-2 (as a
  pickable `baseline` column), so A drops from 382→334 candidates (own 238→190); B/C/D unchanged.
  No row loses all its pickable candidates. (The factual VLM-judge verdicts in A/B may still read
  "vs Base LTX-2" — that is study metadata, not a shown clip. `build_site.py` is unaffected: the
  public site keeps Base LTX-2 as a comparison system.)
- **How media resolves.** `build_picker.py` places relative symlinks in `picker/site/` — `store`,
  `outputs`, `data`, `eval_ladder` (start/end endpoint clips live in `eval_ladder/conds/`) — plus
  `collections.json` (read-only display source) and `picks.json` (the writable picks file). The
  repo-relative paths in the snapshot then resolve next to `index.html`, exactly like a viewer
  mount. `<source>` tags carry **`data-src` only** (no `src`), so VAP's IntersectionObserver
  assigns `src` only when a cell enters the viewport — the network is gated for all ~4,480 videos.
- **Picks file.** `eval_ladder/viewer/collections/supplementary_picks.json` (tracked), schema
  `{"schema":1,"updated":ISO,"picks":{"<collection_id>":{"<item_id>":{card_key,gen_id,video,arm,
  arm_label,variant,seed,picked_at}}}}`. The page saves on every pick through the static server's
  POST endpoint (`X-Base-Updated` optimistic concurrency, 409-merge of only the changed entries,
  localStorage mirror, "save FAILED — download instead" fallback) — the same client the
  collections page uses. Clicking a picked clip again unpicks it. The sticky bar has per-section
  `picked N / M` counters (they count **rows**, so the seed filter never changes them), an
  "Unpicked only" filter, a **seed 42/43/both filter (default 42**, persisted in
  `localStorage["segue_picker_seed"]`), Play/Pause all, a save-status indicator and "Download
  picks.json".
- **Per-row "↻ Restart all (synced)".** Each row header (next to the pick state) has a button that
  restarts every **visible** clip in that row — reference, start/end endpoints and all candidate
  clips — from frame 0 in sync for side-by-side judging (`restartRowSynced`: it assigns any deferred
  `data-src` with the same statement the lazy-loader uses, awaits `loadeddata`/`canplay` per video
  with a 4 s cap, then `pause()`+`currentTime=0` on all, then `play()` on all in one tick; cells
  hidden by the seed filter are skipped).
- **Seed filter mechanics.** Each `.pk-cand` carries a machine-readable `data-seed`; the filter is
  `document.querySelectorAll(".pk-cand")` → `c.hidden = (seedMode!=="both" && String(c.dataset.seed)
  !==seedMode)`. Hiding uses the `hidden` attribute backed by an explicit `[hidden]{display:none
  !important}` rule (without it `.pk-cand`'s inherited `display:flex` from `.seg-unit` beat the UA
  `[hidden]` rule — the reason the filter previously did nothing); a hidden cell is paused and, being
  `display:none`, never intersects, so the lazy-load observer does not fetch it.
- **Wiring picks into the public site.** `build_site.py --picks [path]` (default
  `eval_ladder/viewer/collections/supplementary_picks.json`) applies picks at `--select` time:
  for sections **C/D** the picked gen's `video` becomes the SEGUE clip (the cell is relabeled
  `owner pick: …`), else the current default (w=6 neutral s42). For **A/B** the judged clip stays
  unless `--apply-comparison-picks` is passed (the cell then says "owner pick, not the judged
  clip"). Because a pick changes the transcode **source**, apply picks end to end with
  `--all --picks <file>` (or `--select --media --html --picks <file>`); `--html` alone stays
  seconds and renders the current manifest. With an **empty** picks file the generated
  `site/index.html` is byte-identical to the default build; `site_manifest.json` rows gain
  `src_collection` / `item_id` / `card_key` / `pick` (`pick_applied` for A/B).
- **Hosting.** Registry slug `segue_supplementary_picker` (group `reports`, `featured:false`),
  mounted at `outputs/viewers/segue_supplementary_picker/` — `viewerctl mount|check|hub`.

Regenerable (gitignored): `picker/site/{store,outputs,data,eval_ladder,collections.json,picks.json}`
symlinks. Tracked-eligible: `picker/build_picker.py`, `picker/template.html`,
`picker/site/index.html`.

### Picker — 2026-09-24 rebuild (fixed grid, effect cells, unique pick key)

The picker now renders a **fixed per-section column grid** (CSS subgrid): every row shows the same
ordered `(arm, variant)` slots — one column each — so the correspondence lines up. A slot with no
clip for a row renders an explicit **hatched "absent" placeholder** (non-pickable, skipped by
Restart-all); a per-section column-header strip labels the slots. Two seeds sit side by side inside
a slot; the seed filter hides one.

- **Unique pick identity.** Cells are keyed on their **video path** (`data-video`, unique per
  arm x variant x seed) — NOT `data-gen`, which is the row-shared grid item_id (it even contains the
  token `ic_gen` for every row). `markRow`/`pickCand` match on video, and `build_picker.py` asserts
  per-row video-path uniqueness (fails the build otherwise) so one pick highlights exactly one cell.
- **Effect-prompt cells from the store.** For each row and each of `EFFECT_ARMS`
  (dualforce_control / dualforce_dcg_w1p5 / dualforce_dcg_w3 / dualforce_dcg_w6) the effect clip is
  pulled straight from `store/gens/<dir>/{05_effect_v3__dai,06_effect_v3ed81__dai}/videos/` by
  filename (`<cell>__<arm>_effect_v3[ed81]__<endpoint>__ref_<ref>__s<seed>.mp4`, `os.path.exists`
  per candidate — the partial w1.5/w3 entries have no grid.jsonl), deduped against the snapshot by
  basename. w6 effect stays from the snapshot.
- **Column set.** Fixed order: SEGUE (w=6) n · (w=3) n · (w=1.5) n · w/o NRG n · **ctt_v2 (r128) n**
  · (w=6) e · (w=3) e · (w=1.5) e · w/o NRG e · **ctt_v2 (r128) e** · other own (data order) · prior
  work. **Plain LoRA (`ic_gen`) is removed** (in `EXCLUDE_ARMS`). **ctt_v2** comes from the snapshot
  (prefer the non-regen tier over `*_regen`; dedup by path since ctt_v2 neutral/effect share a
  basename in different dirs); it has no grid-v3 gen, so grid-v3-only rows show the "absent"
  placeholder. The per-section slot list is the union of slots occurring in that section (A/B have no
  ctt_v2, so it is not shown there).
- **Counts (current):** 4,870 candidate clips (4,570 own + 300 prior) + 679 placeholders across
  328 rows; 5,604 lazy `<video>`. Per section: A 10 slots / 452 cand / 3 ph; B 9 / 437 / 5;
  C 10 / 592 / 222; D 10 / 3,389 / 449.
- **Picks migration.** When the pick key moved to the video path, existing saved picks were
  validated against the rebuilt page: a pick whose `video` resolves to exactly one cell is kept; a
  pick with no/unresolvable video is kept but flagged `needs_repick: true`, and the page shows that
  row unpicked with a "re-pick needed" tag (and it counts as unpicked). The picks file is backed up
  to `.history/` before any migration write. `build_site.py --picks` reads the `video` field, which
  may be a `store/…` path (handled — `pick_for`/`add_src` accept any repo-relative path).

## Two pages since 2026-09-25 22:55

`build_site.py --html` writes **`site/index.html` (production)** — the page that goes to GitHub Pages: hidden
items and classes are absent, no curation overlay — and **`site/curate.html` (internal)** — every picked item
present (hidden ones inert until the localhost overlay reveals them), reorder/hide controls, "Save plan" →
`eval_ladder/viewer/collections/site_plan.json`. Curate at http://localhost:8017/supplementary/site/curate.html,
save, re-run `--html`, review http://localhost:8017/supplementary/site/index.html, then push (owner-gated).
The overlay is delimited in `template.html` by `/*OVERLAY-CSS-BEGIN*/…/*OVERLAY-CSS-END*/` and
`<!--OVERLAY-JS-BEGIN-->…<!--OVERLAY-JS-END-->`; the production render strips both and asserts the page clean.
