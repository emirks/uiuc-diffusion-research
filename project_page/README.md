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

The same content is rendered in two interchangeable looks, so the owner can view both side by
side and pick one. **Both read the identical `config.json` / `picks.json` and the identical
media** (one `site/` directory, shared symlinks), so switching is purely cosmetic.

| variant | template | output | look |
|---------|----------|--------|------|
| **dark** (original) | `template.html` | `site/index.html` | black background, the supplementary site's dark theme |
| **light** | `template_light.html` | `site/light.html` | `#fcfcfc` background, white cards, accent-coloured title/authors; adapted from a light academic project-page template (Bulma + Google Sans / Noto Sans) |

Build one or both with `--theme`:

```bash
/usr/bin/python3.12 project_page/build_page.py --theme both   # default: builds + checks both
/usr/bin/python3.12 project_page/build_page.py --theme dark   # only site/index.html
/usr/bin/python3.12 project_page/build_page.py --theme light  # only site/light.html
```

The light variant's accent colour is read from `config.json` → `"accent"` (default `#4f46e5`,
a deep indigo) and `"accent_dark"` (default `#3730a3`); both are injected into the page as the
CSS variables `--accent` / `--accent-dark`. The dark page is unaffected by the accent key.

**Switching the published default once the owner picks:** the deployed site root is whichever
file is served as `index.html`. Dark is already `site/index.html`. To publish the **light**
variant instead, serve `site/light.html` as the site's `index.html` (copy/rename it, or point
the deploy at it) and publish `site/assets/` alongside it. Nothing else changes — the two pages
reference the same `videos/`, `stills/` and `img/method.png`.

## Files

| file | what |
|------|------|
| `build_page.py` | renders the page(s), checks media, writes `CANDIDATES.md` (`--theme dark\|light\|both`) |
| `config.json` | title, authors, affiliations, link URLs, BibTeX, light `accent`/`accent_dark` (owner-editable) |
| `picks.json` | what appears on the page (owner-editable) |
| `CANDIDATES.md` | generated menu of every selectable block/cell/row |
| `template.html` | **dark** page shell (copied from the supplementary template, overlay stripped) |
| `template_light.html` | **light** page shell (adapted from a light academic project-page template) |
| `class_names.json` | local copy of the supplementary class-name map (do not hand-edit) |
| `site/index.html` | the rendered **dark** page |
| `site/light.html` | the rendered **light** page |
| `site/assets/css/` | vendored CSS for the light page (`bulma.min.css`, adapted `index.css`) |
| `site/videos` | symlink → `../../supplementary/site/videos` |
| `site/stills` | symlink → `../../supplementary/site/stills` |
| `site/img/method.png` | the method figure (rasterised from `fig3_method.pdf`) |

## Build

Use the login-node Python 3.12 (`python3` on PATH is 3.6 and must not be used):

```bash
/usr/bin/python3.12 project_page/build_page.py              # render + check (default)
/usr/bin/python3.12 project_page/build_page.py --html       # render only
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

## What is NOT on the page

By design the project page is a short teaser, not the full results. It does **not** include the
human study, the limitations / failure cases, a table of contents, or any internal wording
(curation, picks, blind-study verdicts, section codenames, item ids in visible text). All of
that lives on the full supplementary page, which the header and footer link to.
