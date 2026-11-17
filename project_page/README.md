# SEGUE project page

A simple, static landing page for the SEGUE paper. Pure HTML/CSS/JS, no build framework.
It reuses the already-encoded clips and stills from the supplementary site (via symlinks) and
the supplementary build's renderers (imported from `../supplementary/build_site.py`), so it
never re-encodes media and never writes anything under `supplementary/` or `eval_ladder/`.

Everything on the page is one idea per section:

1. **Header** — title, authors, link buttons (Paper / Code / Model / Dataset / Supplementary /
   BibTeX), and a line to the full supplementary page.
2. **Hero** — removed (owner 2026-10-07); the page starts with the TL;DR and the galleries. `picks.json` `"hero"` is ignored.
3. **TL;DR** — three sentences.
4. **Transition effect generation** — a gallery of SEGUE results (two endpoints given).
5. **Visual effect transfer** — a gallery of SEGUE results (only the start clip given).
6. **Comparison with previous work** — SEGUE vs prior work, TEG rows then VFX-transfer rows.
7. **How SEGUE works** — the method figure + the null-reference-guidance (NRG) sweep.
8. **Footer** — a link to the full supplementary page, then the BibTeX block.

## Two visual variants

The same page is rendered in a **dark** and a **light** look so the owner can view both and pick
one. Since 2026-10-07 **both themes use ONE template (`template_light.html`) and the identical
light-style markup / layout** (owner: "dark is better but the style of the light is better: make
the dark one just like the light, only colouring differs"). The two outputs differ in **exactly
two things**:

1. a theme class on `<html>` — `theme-dark` (→ `site/index.html`) or `theme-light` (→ `site/light.html`);
2. the injected accent values (see below).

Everything else — structure, spacing, fonts, radii, shadow geometry, media — is byte-identical.

| variant | output | `<html>` class | look |
|---------|--------|----------------|------|
| **dark** | `site/index.html` | `theme-dark` | `#0a0a0a` background, `#121212` cards with `#262626` borders, light text; the light layout recoloured by one palette block |
| **light** | `site/light.html` | `theme-light` | `#fcfcfc` background, white cards, accent title/authors (Bulma + Google Sans / Noto Sans) |

The single palette difference lives in `template_light.html` as one `<style id="pp-dark-palette">`
block scoped under `html.theme-dark`. It overrides **colour only** (page/hero/section backgrounds,
card surfaces + borders + shadow colour, text tiers, title/authors/affiliation chips, `btn-fancy`
pills, the `.seg-*` input/output frames, arrow, labels, footer box and BibTeX block) and touches
no layout, spacing, font, radius or shadow geometry. The accent inside that block is read through
`var(--accent)` / `var(--accent-dark)` (never a literal), so the block's text is identical on both
pages and only takes effect on the dark one.

Build one or both with `--theme`:

```bash
/usr/bin/python3.12 project_page/build_page.py --theme both   # default: builds + checks both
/usr/bin/python3.12 project_page/build_page.py --theme dark   # only site/index.html
/usr/bin/python3.12 project_page/build_page.py --theme light  # only site/light.html
```

**Accents** (all in `config.json`, injected per theme as the CSS variables `--accent` / `--accent-dark`):

| theme | `config.json` keys | defaults |
|-------|--------------------|----------|
| light | `accent` / `accent_dark` | `#2563eb` / `#1d4ed8` |
| dark  | `accent_dark_theme` / `accent_dark_theme_hover` | `#4d9fff` / `#2f7ff0` |

`template_dark_v1.html` is the **retired** original dark shell (the old supplementary-derived
markup). It is kept as a fallback and is **no longer built** — the builder never reads it.

**Switching the published default once the owner picks:** the deployed site root is whichever
file is served as `index.html`. Dark is already `site/index.html`. To publish the **light**
variant instead, serve `site/light.html` as the site's `index.html` (copy/rename it, or point
the deploy at it) and publish `site/assets/` alongside it. Nothing else changes — the two pages
reference the same `videos/`, `stills/` and `img/method.png`.

## Files

| file | what |
|------|------|
| `build_page.py` | renders the page(s), checks media, writes `CANDIDATES.md` (`--theme dark\|light\|both`) |
| `config.json` | title, authors, affiliations, link URLs, BibTeX, accents (`accent`/`accent_dark` light, `accent_dark_theme`/`accent_dark_theme_hover` dark) (owner-editable) |
| `picks.json` | what appears on the page (owner-editable) |
| `CANDIDATES.md` | generated menu of every selectable block/cell/row |
| `template_light.html` | **the** page shell for BOTH themes (theme class + one `html.theme-dark` palette block) |
| `template_dark_v1.html` | **retired** original dark shell — kept as a fallback, no longer built |
| `class_names.json` | local copy of the supplementary class-name map (do not hand-edit) |
| `site/index.html` | the rendered **dark** page |
| `site/light.html` | the rendered **light** page |
| `site/assets/css/` | vendored CSS for both pages (`bulma.min.css`, adapted `index.css`) |
| `site/videos` | symlink → `../../supplementary/site/videos` |
| `site/stills` | symlink → `../../supplementary/site/stills` |
| `site/img/method.png` | the method figure (rasterised from `fig3_method.pdf`); full-size link target + og:image |
| `site/img/method.webp`, `site/img/nrg.webp` | what the page shows (WebP q90, ~0.3 MB / ~0.1 MB); re-made from the PNGs by `build_page.py` (`ensure_webp`) whenever a PNG is newer |
| `site/supp/index.html` | the rendered **results gallery** (dark theme, all supplementary content) |
| `site/supp/videos` | symlink → `../../../supplementary/site/videos` |
| `site/supp/stills` | symlink → `../../../supplementary/site/stills` |
| `site/supp/assets` | symlink → `../assets` (so the vendored Bulma/`index.css` load unchanged) |
| `site/supp/img/nrg_start.jpg` | copy of `site/img/nrg_start.jpg` (the guidance row's start frame) |

## Build

Use the login-node Python 3.12 (`python3` on PATH is 3.6 and must not be used):

```bash
/usr/bin/python3.12 project_page/build_page.py              # render (both themes + supp) + check
/usr/bin/python3.12 project_page/build_page.py --html       # render the two theme pages only
/usr/bin/python3.12 project_page/build_page.py --supp       # render the results gallery only
/usr/bin/python3.12 project_page/build_page.py --check      # verify media + cleanliness only
/usr/bin/python3.12 project_page/build_page.py --candidates # regenerate CANDIDATES.md
```

`--check` fails if any `<source data-src>` / `<img src>` does not resolve inside `site/`
(through the symlinks); it prints how many clips/stills are referenced and the total MB.

View locally (static server on the repo root, port 8017):

```bash
# start it if it isn't running:
cd <repo> && nohup /usr/bin/python3.12 -m http.server 8017 --bind 127.0.0.1 >/dev/null 2>&1 &
```

→ dark:  http://localhost:8017/project_page/site/index.html
→ light: http://localhost:8017/project_page/site/light.html

## Editing what appears

Edit **`picks.json`**, then rebuild. Run `--candidates` first to regenerate `CANDIDATES.md`,
which lists every VISIBLE block / cell / row (with its `item_id`, endpoint, tier and SEGUE
clip label) that you can choose from. "Visible" means not hidden in the supplementary curation
plan `eval_ladder/viewer/collections/site_plan.json`.

`picks.json` schema:

```jsonc
{
  "hero": [ {"section": "C"|"D", "block": "<block_id>", "cell": "<item_id>"} ],
  "teg":  [ {"block": "<block_id>", "cells": null | ["<item_id>", ...]} ],  // null = all visible cells, plan order
  "vet":  [ {"block": "<block_id>", "cells": null | ["<item_id>", ...]} ],
  "teg_cmp": ["<item_id>", ...],   // comparison rows, Transition Effect Generation
  "vet_cmp": ["<item_id>", ...],   // comparison rows, Visual Effect Transfer
  "nrg": "nrg_color_rain_3_gfit"   // the guidance-sweep row id
}
```

- Gallery block ids are `teg_ref__<reference>` (TEG) and `vfx_ref__<reference>` (VFX transfer).
- If a pick names a block/cell/row that is **hidden** in the plan, it still renders but the
  build prints a `WARNING` so you know you reached past the curated set.

Edit **`config.json`** to fill in the link URLs (`paper`, `code`, `model`, `dataset`). An empty
URL renders that button in a muted "soon" style so the layout stays visible; fill it in and
rebuild to activate the button. The `supplementary` link is relative on purpose — see below.

## Deployment note

On deployment the **site root is this `index.html`** (publish the contents of `site/`). The
supplementary site is expected to live under **`/supp/`** at the same root.

Because of that, `config.json → links.supplementary` is set to the *local* relative path
`../../supplementary/site/index.html`, which points at the local supplementary site when you
view the page through the port-8017 server. **Before deploying, change that value to `supp/`**
(and place the supplementary build there) so the "Supplementary" button and the two
supplementary links resolve on the deployed host. `img/method.png` and the `videos` / `stills`
directories must be published alongside `index.html` (resolve the symlinks when you copy).

## Results gallery (`supp/`)

`build_page.py --supp` (and the default no-flag invocation, after the two themes) renders
`site/supp/index.html` — the **full** results gallery, the page served at
`https://segue-video.github.io/supp/`. It replaces the old VAP-derived dark supplementary page
with one that looks **exactly like the project page's dark showcase band**: the SAME
`template_light.html`, rendered in the dark theme (`<html class="theme-dark">`) with the config's
**band** accent (`accent_band` / `accent_band_dark`), so the dark palette, fonts, cards, grey
input frames, accent SEGUE output frame + chip, the arrow between inputs and output, square
corners and full-width comparison rows all match the project page. It carries **all** of the
supplementary's content (not the `picks.json` subset): a compact hero (SEGUE title + "Results
gallery" + *Project page* → `../` and *Paper* → `../paper.pdf` buttons), a contents card, then
the galleries and comparisons in the supplementary's order — Transition effect generation (C) and
its comparison (A), Visual effect transfer (D) and its comparison (B) — followed by Guidance (the
NRG sweep row) and Limitations (failure cases).

It is built by reusing the supplementary renderers imported from `../supplementary/build_site.py`:
galleries via `bs._render_gallery`, comparison rows via the project page's `cmp_row_html` over the
same plan iteration as `bs._render_comparison`, guidance via `guidance_row_html` and limitations
via `bs._limitation_row_html`, all driven by `supplementary/site_manifest.json` and the curation
plan `eval_ladder/viewer/collections/site_plan.json` (hidden blocks/cells/rows absent, `merge_into`
honoured, orders respected). The rendered content (blocks/cells/rows, in order) equals the current
supplementary page. Nothing is re-encoded.

Media resolve through symlinks created under `site/supp/` at build time: `videos` →
`../../../supplementary/site/videos`, `stills` → `../../../supplementary/site/stills`, and
`assets` → `../assets` (so the page's `./assets/...` CSS paths work unchanged). The guidance row's
start frame is a copy of `site/img/nrg_start.jpg` at `site/supp/img/nrg_start.jpg`. `--check`
verifies every `data-src` / `img src` resolves under `site/supp/` and prints the media count
(≈ 345 files) and total MB.

**Deployment note:** publish `supp/index.html` and everything it references *under* `supp/` in the
Pages repo — `supp/videos/`, `supp/stills/` (resolve the symlinks when you copy), `supp/assets/`
and `supp/img/nrg_start.jpg` — so it serves at `.../supp/`.

## Video loading and phone/tablet layout (2026-11-16)

Both pages (project page and `supp/`) share one loader in `template_light.html`. Since 2026-11-17
it works per ROW (a `.seg-blockwrap` = a reference with its samples, or a `.seg-strip` = a
comparison/guidance row), so the clips of a row start together. Each row is in one of three zones:
within one screen of the viewport its clips are attached and buffered; on screen, once every clip
of the row can play (at most 2.5 s wait), all start in the same moment, and off screen the row
pauses; more than 2.5 screens away the row is released (sources removed, decoder and buffer freed).
Only the start is lined up; clips of different lengths then loop on their own. A row is reloaded
only after a release, so scrolling back resumes it. Rows on a hidden pager page count as off screen and are released by the same rule; the
pager only warms the NEXT page, and only while that pager is on screen (before 2026-11-16 every
pager warmed its page 2 at page load, ~17 MB, and nothing was ever released). Swiping left/right on
a pager's rows changes page.

Below 1024px nothing scrolls sideways inside a card (`<style id="pp-responsive">`): gallery rows
show the reference on its own line, then one "start/end -> SEGUE" line per sample (one column on
phones, two or three on tablets); comparison, guidance and limitation rows are a grid of inputs, a
down arrow, then outputs (2 columns up to 600px, 4 columns up to 1023px). Inside a pager a thin
line separates the stacked rows (2026-11-17). On phones the header
buttons are an even two-column grid (the "soon" tag is hidden). The pager arrows sit in the dots bar
under the rows, and float beside the card only at 1341px and wider. Desktop layout (1024px+) is
unchanged. The clip files themselves are not touched (480x640 H.264, ~300 KB each, sharp at every
display size the page uses).

**Deployment:** publish `img/method.webp` and `img/nrg.webp` next to the PNGs.

## What is NOT on the page

By design the project page is a short teaser, not the full results. It does **not** include the
human study, the limitations / failure cases, a table of contents, or any internal wording
(curation, picks, blind-study verdicts, section codenames, item ids in visible text). All of
that lives on the full supplementary page, which the header and footer link to.

## Google Search Console

`site/googlef5044dc8e7b233be.html` is the ownership-verification file for https://segue-video.github.io/ (added 2026-10-09). It must stay at the site root in every deployment, or Search Console drops the property.
