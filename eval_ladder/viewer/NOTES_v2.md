# build_neutral_effect_v2 — fast-build notes

The arm-comparison viewer builder since 2026-09-22 (registry slug `iclora_neutral_effect_v2`,
featured). It started as a copy of the v1 builder `build_neutral_effect.py` and produces the SAME
payload far faster on a rebuild. The v1 builder + template are archived byte-identical under
`archive/neutral_effect_v1_2026-09-22/` (README + `run_v1.sh` there); the last v1 page,
`outputs/reports/iclora_neutral_effect/index.html`, stays openable from the dashboard's
"Earlier versions & archive" table (registry entry `iclora_neutral_effect`, `archived`).

Files:
- `build_neutral_effect_v2.py` — the builder (copy of v1 + the three changes below).
- `template_neutral_effect_v2.html` — copy of the v1 template with the data split out (see A).
- output: `outputs/reports/iclora_neutral_effect_v2/{index.html, data.js}` (two files, not one).
- cache: `outputs/cache/viewer_neutral_effect_v2/` (disposable; a wipe costs one cold rebuild).
- registry slug: `iclora_neutral_effect_v2` (mount pages = both index.html and data.js).

## Why the split (data ⟂ template)

The v1 page is one 35 MB `index.html`: a 39 KB template with a 35 MB JSON payload spliced into
`const D = /*__DATA__*/null;`. So every template tweak (a dropdown, a label) meant re-emitting all
35 MB, and there was no way to touch the page without also holding the whole payload.

v2 writes the payload to a sibling `data.js` (`window.__NE_DATA__=<compact JSON>;`) and the template
loads it with `<script src="data.js"></script>` right before the main inline script (scripts run in
order, so `D` is defined before use). The page is served by a plain static file server from the repo
root through the mount dir, and `data.js` is referenced by a bare relative path next to `index.html`,
so it resolves wherever the page is mounted. `index.html` is now ~39 KB and re-emitting it is free.

## Three modes (`--mode`, default `all`)

- `all` / `data` — run `build()` (with the cache), `check()` (the seatbelts), then write BOTH
  `data.js` and `index.html`. `data` and `all` are the same full rebuild.
- `page` — re-emit `index.html` from the template ONLY. No `build()`, no store reads, sub-second.
  Use it after any template/JS edit; `data.js` is left as-is.

`--out-dir` sets the output directory (default `outputs/reports/iclora_neutral_effect_v2`).

## What the cache keys on, and how to invalidate

The v1 profile showed `build()` spends ~95% of its time in `attach_external` — the external arms'
`items.jsonl` parse PLUS a per-generation filesystem `.exists()` storm (video and reference-clip
existence checks over the networked store). Both are per-arm and independent of which cards exist,
so v2 memoizes them per arm. It also memoizes the three score-set readers.

Cache units (one pickle file each, protocol 5, atomic tmp+`os.replace`):
- `attach__<arm_id>` — one per external arm: its parsed scores, provenance, and every pre-built
  generation dict (videos + reference clip resolved on disk). The cheap join onto cards (and the
  one card-dependent field, `prompt_hi`) is redone every build, so the join is always correct
  against the current card set even off a warm cache.
- `all_scores`, `instrument_delta`, `control_floors` — the SCORE_SETS directory reads.

Each unit's key is `sha256(CACHE_SCHEMA, <sha256 of build_neutral_effect_v2.py>, unit, fingerprint,
extra)` where the **fingerprint** is the sorted list of `(repo-relative path, size, mtime_ns)` for
EVERY file the unit reads — the `*/items.jsonl` and `results.json` under its score dir, the arm's
rows/grid/manifest file, and the registry + ceilings source files (their parsed dicts feed the join
and the pool-% arithmetic). The per-arm `extra` also carries a hash of the arm's media-directory
LISTING (which clips exist — the only thing the `.exists()` storm reads) plus the primary set's
corpus/env (which the provenance flags compare against). A miss (absent file, unreadable, or changed
key) re-reads that unit and rewrites it; every other unit is served from disk.

Invalidation is therefore automatic: touch a scored file, the registry, or the ceilings and only the
affected units re-read; **edit the builder source and everything re-reads** (its sha is in every
key). Manual: `--no-cache` ignores existing entries for this run (still rewrites them);
`--clear-cache` deletes the cache dir first (a true cold build). Assumption: the frozen std121
dataset (`data/processed/transitions_std121`, reference/endpoint clips) is treated as immutable and
is not fingerprinted; if it is ever regenerated, run `--clear-cache`.

The metric arithmetic is untouched — a cached value is byte-for-byte what the uncached function
returned, and the payload is proven identical to v1 (below).

## Identity proof (measured 2026-09-22)

Built v2 cold (`--clear-cache`), v2 warm, and the ORIGINAL v1 builder to a scratch path on the same
store state, then compared the compact JSON three ways (extracted with a script, not by eye — v1's
is the substring between `const D = ` and its closing `;`; v2's is `data.js` minus the
`window.__NE_DATA__=` … `;` wrapper):

- payload size: 35,455,253 bytes
- **v1 embedded JSON == v2 cold data.js == v2 warm data.js — byte-identical (sha256 870112c51a95…)**

`meta.rel` is set from the output depth exactly as v1's `emit()` did; the v2 output dir is at the
same depth (3) as v1's, so it is identical (`../../../`). There is no build timestamp in the payload.

Wall-clock (single GH200 login node, one process):

| build            | wall     |
|------------------|----------|
| v1 (original)    | 577.70 s |
| v2 cold cache    | 577.09 s |
| v2 warm cache    | 66.02 s |
| v2 page-only     | 0.47 s |

Cold ≈ v1 (a one-time cost that also populates the cache). Warm and page-only are the everyday paths:
a store-side change (new scores, a re-registered gen) re-reads only the arms whose files changed;
"move a dropdown" is `--mode page`. Note that adding an arm means editing this builder (an `EXTERNAL` /
`CATALOG` / `GRID_V3_TABLE` row), and the builder's sha is in every cache key, so that first rebuild is
cold (≈10 min) and the ones after it are warm again. (The residual warm time is the two run columns' own
card assembly and the per-unit fingerprint stats, which are not per-arm.)

## Status (2026-09-22 14:45)

The dashboard uses v2: `iclora_neutral_effect_v2` is the featured entry, `iclora_neutral_effect`
(the v1 page) carries `archived` and lists under "Earlier versions & archive". Everyday commands:

```
# data or arms changed (warm cache ~1 min; cold ~10 min):
$LAB/envs-aarch64/ltx2/bin/python eval_ladder/viewer/build_neutral_effect_v2.py
# template/JS edit only (sub-second, no store reads):
$LAB/envs-aarch64/ltx2/bin/python eval_ladder/viewer/build_neutral_effect_v2.py --mode page
# then (login-node python3 is 3.6 — use 3.12):
/usr/bin/python3.12 scripts/viewers/viewerctl.py mount iclora_neutral_effect_v2
/usr/bin/python3.12 scripts/viewers/viewerctl.py check iclora_neutral_effect_v2
/usr/bin/python3.12 scripts/viewers/viewerctl.py hub     # only if a registry field changed
```

To rebuild the archived v1 page for a comparison: `bash eval_ladder/viewer/archive/neutral_effect_v1_2026-09-22/run_v1.sh`.

## Collections (2026-09-22)

The page can bookmark INPUT rows into named collections (the TEG / VFX-transfer user studies, a
supplementary set) that snapshot the columns you have open — present OR absent for that row — plus
free-text notes. Two invariants: the **template stays generic** (no category/arm/tier/collection id is
hardcoded) and the **three presets are DATA**, seeded into the JSON, never template constants.

**Durable record — one git-tracked JSON.** `eval_ladder/viewer/collections/neutral_effect_collections.json`
(`eval_ladder/viewer/` is tracked; `outputs/` is gitignored). Pretty-printed (indent 2, ensure_ascii
False) so git diffs read cleanly. Schema `{schema, viewer, updated, collections:[{id,title,notes,
created,updated,items:[{id,card_key,added,updated,notes,view,inputs,columns:[{tier,category,
category_label,arm,arm_label,variant,label,present,gens:[{id,seed,video,ref_video,scored,pct,cond}]}]}]}]}`.
`columns` = the entries VISIBLE (selected in the arm panel) at save time, in panel order; `present` =
this card has ≥1 gen in that tier (any seed); absent columns are still recorded (`present:false`,
`gens:[]`) — that is what makes "refvfx + vap were open but only refvfx has an entry for this row"
obvious. Every snapshotted video path is repo-relative (no `meta.rel` prefix).

**Seeding + serving symlink (builder, EVERY mode incl. `--mode page`).** `seed_collections()` in
`build_neutral_effect_v2.py` writes the file with the three presets the first time it is missing,
writes `collections/.gitignore` (`.history/`), and refreshes a RELATIVE symlink
`outputs/reports/iclora_neutral_effect_v2/collections.json → ../../../eval_ladder/viewer/collections/
neutral_effect_collections.json` (relative so it survives a repo move). The registry mounts that page,
so the browser path `outputs/viewers/iclora_neutral_effect_v2/collections.json` is a symlink-to-symlink
that resolves to the tracked file. It is in the entry's `check_files` (so `check` is now LIVE 5/5).

**Save path — the static server does POST.** `scripts/viewers/viewerctl.py` `cmd_httpd` gained
`do_POST`/`do_PUT`: only for URLs whose `translate_path`→`realpath` lands inside
`<repo>/eval_ladder/viewer/collections/` and ends `.json` (else 403); body must parse as JSON with an
integer `schema` (else 400); 20 MB cap (413). **Optimistic concurrency:** the client sends
`X-Base-Updated: <the "updated" it loaded>`; if the file's current `updated` differs → 409 with
`{"error":"conflict","current":<file>}`, and the client merges (keep the server's collections, replace
only the ones it changed) and retries once. On success the server stamps top-level `updated`, writes
atomically (tmp + `os.replace`), rolls the PREVIOUS file into `.history/<name>.<UTC stamp>.json`
(newest 30 kept), and answers `{"ok":true,"updated":...,"bytes":N}`. Bind stays 127.0.0.1. GET + byte
ranges are untouched.

**The page** loads `fetch("collections.json?t="+Date.now())` (relative) at startup, mirrors the last
good state to `localStorage["iclne_v2_collections_mirror"]`, and if the fetch fails loads the mirror
and shows an "offline copy — server save unavailable" status. UI (all generic): a collections bar
(active-collection select, ＋new / rename / delete, "open collection" + "restore its columns" toggles,
"manage rows" drawer, a save-status indicator, and an export/import menu), a per-card bookmark button
opening an anchored add/edit dialog (collection select, columns-to-capture checklist with ✓ has /
— no-entry markers, notes textarea, Esc / Ctrl-Cmd+Enter), and a right-side management drawer
(collection notes, rows with input line + present/absent column chips + inline notes + ▲▼ reorder / go
/ ✕). Downloads use a Blob + `<a download>` (a normal browser page, not a Claude artifact).

**Exports.** In-page "Export Markdown / JSONL (active collection)" and the CLI
`eval_ladder/viewer/collections_export.py` (`--collection <id>` / `--all` / `--stdout` / `--check`,
stdlib only, writes `outputs/reports/iclora_neutral_effect_v2/collections/<id>.{md,jsonl}`) share the
SAME string ops and produce byte-identical output for the same data (verified 2026-09-22 by extracting
the template's export functions and running them under duktape against the CLI — MD and JSONL both
identical). `--check` validates the schema and that every snapshotted video path exists (exit 1 on any
miss). The CLI never reads `data.js` — everything it needs is in the JSON.

**If the server has no POST** (an old httpd, or a plain file server): saves fail, the status shows
"save FAILED — download instead", and the page keeps working off the localStorage mirror. Use the
menu's "Download collections.json" and drop the file in place at
`eval_ladder/viewer/collections/neutral_effect_collections.json`, then restart the server
(`viewerctl serve`) to pick up the POST handler.

**Note (shared live file).** The collections JSON is a single shared file the running server writes.
When testing the POST endpoint by hand (curl), test against a COPY — a POST to the live URL overwrites
whatever a live browser session is editing (the `.history/` backups and the 409-merge recover it, but
it is disruptive).

**Quick-add target (2026-09-23).** The collections bar has a `quick-add →` `<select>` (the collections in
file order with row counts, plus a `(none)` default) whose choice is stored per-browser in
`localStorage["iclne_v2_quick_target"]`; if that id later disappears the target silently falls back to none.
When a target is set, every card grows a second header button beside `＋ collection` reading `＋ <target
title>` (title ellipsized past ~18 chars). ONE click files that row into the target with the columns
currently open, producing the SAME item the dialog's Save would (same `snapshotInputs` / columns / `view`,
`notes` `""`), no dialog; a second click removes it (no confirm, but a 3 s "removed — undo" toast restores
the item with its notes). A filled button reads `✓ <title>`. EXCEPTION: if the row is also in the collection
currently OPEN in "open collection" mode, the new item copies that item's notes as its starting notes and
records a plain extra `source_collection:"<open id>"` field (`collections_export.py` reads item fields by
name and ignores unknowns — verified, so no exporter change was needed). Sub-cards use their `|ref=` key and
legacy base-key items resolve exactly as the dialog does (`itemMatchesDisplay`); a card never gets a
duplicate `card_key`. Saves reuse the dialog's autosave path (mirror + debounced/immediate POST, 409-merge)
— no second save mechanism. Keyboard nicety: `a` toggles quick-add on the HOVERED card (never while typing
in an input/textarea/select, never with the dialog open); cards are not focusable, so it is hover-only.
Everything is generic — no collection id is hardcoded (the target is a stored id). Template-only change;
`data.js` is untouched.

## Arm display names = the paper's names (2026-09-22)

Owner: "change the name of all the grid v3 arms to what we currently actually use, like plain lora,
SEGUE w/o guidance, SEGUE w=1.5 etc." The authority for the wording is the CURRENT paper source,
`papers_drafts/ctt_iclr2027/tables/*.tex` + `sections/03d_inference.tex` (the owner's 2026-09-22 rewrite
names the guidance step "Null-reference guidance (NRG)"; the tables say "\segue{} w/o NRG" in 9 places and
"w/o guidance" only in two older comments), so the viewer says **w/o NRG**. Store arm ids never change —
they are the join keys and stay visible in every entry's `sub` ("… · store arm <arm> (<gen dir>)").

| store arm (canonical) | gen dir | viewer label now | was |
|---|---|---|---|
| `base_prompt` | — (v2 grid only) | Base LTX-2 · prompt only | base · prompt-only |
| `base_cond` | 005_base_cond | Base LTX-2 (no reference) | base · +endpoints |
| `ic_gen` | 001_ic_gen | Plain LoRA | ic_gen (r32) / "IC-LoRA generalist" (run column) |
| `dualforce_control` | 013_dualforce_control | SEGUE w/o NRG | DUAL-FORCE control (plain FM) |
| `dualforce_dcg_w1` | — (v2 grid only) | SEGUE (w=1 · parity) | DCG w=1 (parity) |
| `dualforce_dcg_w1p5` | 030_dualforce_dcg_w1p5 | SEGUE (w=1.5) | DCG w=1.5 |
| `dualforce_dcg_w3` | 031_dualforce_dcg_w3 | SEGUE (w=3) | DCG w=3 |
| `dualforce_dcg_w6` | 032_dualforce_dcg_w6 | SEGUE (w=6) — the paper's plain "SEGUE" | DCG w=6 |

Category labels followed: `df_dcg` → "SEGUE · NRG weight w (test-time DCG on SEGUE w/o NRG)", `dualforce`
→ "SEGUE w/o NRG (dualforce_control) · DUAL-FORCE KD-crutch A/B siblings". NOT renamed on purpose: the
`dcg` category (test-time DCG on ctt_v2, not a paper arm) keeps "DCG w=…", and every non-paper arm keeps
its name. Label sites in the builder (all must agree for one canonical arm, because the panel takes the
FIRST label it meets for an arm id): `GRID_V3_TABLE` (4th field), the `CATALOG` rows, the v2-grid
`EXTERNAL` entry `label`s (the Ⓝ/Ⓔ/ⓝ strings feed the metrics-table row names via `TIER_LABEL`),
`FAM_LABEL` (legacy families), and the `RUNS` column `label`. A rename is a builder edit → one cold
rebuild (~10 min). Collections snapshot `arm_label` at save time, so rows saved before the rename keep the
old strings in their snapshot (the card_key/tier joins are unaffected).

## TEG arms: all rows + neutral twins (2026-09-22 17:58)

The TEG category (`teg`) is built from `_TEG_SYSTEMS` x (effect, neutral) in the builder. Since 17:58 every two-sided
grid-v3 row of each grid.jsonl joins (74 items x seeds 42/43 = 148 clips per arm; before, a `rows_keep` filter kept
the 38 zero-shot items): the effect arms keep their evals/042 zero-shot scores and show seen/unseen as videos only; the
three neutral twins (`refvfx_neutral_v3_teg`, `wan_flf2v_neutral_v3`, `wan_vace_neutral_v3`; gens 003/06, 042/02,
043/02, campaign misc/2026-09-21_neutral_baselines) are wholly unscored — their `scores` path points at an absent
entry (`store/evals/pending_teg_neutral__dai/<arm>`), which is the builder's documented "unscored, videos only" path
(no placeholder, no borrowed number). Scoring them later = drop the eval entry in, repoint `scores`, rebuild. The
`_TEG_SYSTEMS` tuple: (store shelf, frames, canonical arm, panel label, cond note, recipe text, (effect subentry,
neutral subentry), (effect harness_arm, neutral harness_arm)); harness_arm must equal the `arm` stamped in grid.jsonl
(`assert_arms`). Seatbelts that apply: every mp4 present (videos == exp_vids), off_grid 0, prompts differ from ours.

## Per-reference rows (2026-09-22)

A card is one INPUT row keyed `<donor>|<endpoint>|<sided>`, and 32 of the 365 cards have TWO
references of the target transition class (e.g. a transition demoed by clip `_1` AND clip `_2`). The
page used to pile every reference's generations into one card. It now splits such a card, at RENDER
time (template only — `data.js` is untouched), into one **display sub-card per reference**.

- **Sub-card key** = `<card.key>|ref=<refClip>` (a fourth `|`-segment appended to the base key).
  Single-reference cards are unchanged: same key, same object, same rendering — so existing
  collections keep working. Only cards with >=2 refs (or >=2 distinct non-null `g.ref`) split, into
  sub-cards adjacent in `card.refs` order.
- **Which generations land on a sub-card** (`subCardsFor` / the `keep` predicate): a reference-taking
  gen (`g.ref != null`) goes to the sub-card whose `refClip === g.ref`. A **reference-free** gen
  (`g.ref == null`, e.g. base LTX-2 `base_*`, the TEG text-only Wan arms) is matched by a `__ref_<clip>`
  token in its `g.id` (fallback: any `g.videos` path), word-bounded (the clip name runs until `__`,
  `.`, `-`, `/` or end). A reference-free gen with **no** `__ref_` token at all (the `spec_<donor>`
  specialist arm — 21 such gens across the 32 cards) is shown on **every** sub-card and its cell is
  tagged "reference-free" so it is not read as a per-reference result. Because those appear on more
  than one sub-card, the metrics table (`drawStats`) and the `selinfo` generation count **dedup by gen
  identity**, so the numbers are byte-for-byte what they were before the split (proven: per-tier
  gen-set identity, 101 tiers, 0 mismatches). The matrix counts are the payload's precomputed base-row
  counts and are unchanged. Donor/sidedness "all (N)" and the "N rows" text count DISPLAY rows (397 =
  365 + 32).
- **Row header** now states, on one line, the reference (split cards only) and the row's TIER —
  novelty × content — in the page's plain words plus the cell id(s) as a muted tag, e.g.
  "transition flying_cam_transition · two-sided · reference flying_cam_transition_2 · zero-shot · cross
  [G-zs-cross]". The vocab comes from `D.novelty_order`/`D.content_order` + their labels (no tier id is
  hardcoded); a row that mixes novelty (13 cards mix seen+unseen on the same reference) renders
  "seen + unseen". The label is derived from the SELECTED tiers' gens, falling back to all gens so it
  never blanks under a filter.
- **Backward-compat for old collection items.** An item whose `card_key` is the BASE key of a
  now-split card (saved before this change) is resolved to a sub-card by the reference in its snapshot
  (`inputs.refs`, else `__ref_` tokens in the saved `columns[].gens`); if ambiguous it attaches to the
  FIRST sub-card and shows a "saved before the per-reference split — re-save to pin the reference" note
  in the drawer and the add/edit dialog. The live JSON is **not** rewritten on load — the item's
  `card_key` becomes the sub-card key only when the owner next edits/updates it (normal save path).
  `collections_export.py` already treats `card_key` as an opaque string (no 3-part assumption), so a
  fourth `|ref=` segment needs no change there.
- **Snapshots + exports carry the tier.** `snapshotInputs()` now records
  `tier:{novelty:[...],content:[...],cells:[...]}` (raw ids, selection-independent) for NEW items;
  the drawer row line, the Markdown row heading and the JSONL manifest print it, and items saved
  before this change (no `tier`) render fine without it. In JSONL the row tier is emitted as
  **`row_tier`** because each row already merges a per-column `tier` (the arm-entry id). The in-page
  and CLI exports stay byte-identical (re-verified under duktape vs `collections_export.py`, incl. the
  live collections).
