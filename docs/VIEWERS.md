# Viewers

Interactive HTML pages that show what the pipeline actually produced — clips,
filmstrips, scores, side-by-sides. They are how results get looked at rather
than described.

**Dashboard:** `http://localhost:8017/outputs/viewers/index.html`
**Start it:** `/viewer` in Claude Code, or `python3 scripts/viewers/viewerctl.py serve`

---

## The system in five rules

1. **A viewer is a directory, not a file.** `outputs/viewers/<slug>/index.html`,
   with the media it needs symlinked in beside it.
2. **Every path inside a page is relative to that directory.** No leading `/`,
   no `../../` climbing out of the repo. Media reaches the page through a
   symlink in its own directory, so the page does not care where the repo lives.
3. **`outputs/` is disposable; the registry is not.** `outputs/` is gitignored.
   Anything durable — what a viewer shows, how its media is wired, why it was
   retired — lives in `scripts/viewers/registry.json`, which is tracked. A viewer
   still *appears* without being registered (see Creating a viewer); the registry
   is for the parts worth keeping when `outputs/` is wiped.
4. **The generator is the artifact.** A viewer that cannot be regenerated is a
   liability. Generators live in `scripts/` and are tracked; their output is not.
5. **One page per family is current.** Older builds of the same viewer become
   `supersedes` entries under the current one, not siblings on the dashboard.

### Why relative-through-a-symlink

Three path conventions grew in this repo, and only one survives a server restart
or a repo move:

| Convention | Example | Verdict |
|---|---|---|
| relative through a local symlink | `media/clip_000.mp4` | ✅ **use this** — works from any server root |
| server-absolute | `/outputs/videos/run/clip.mp4` | ⚠️ works only when the server root is exactly the repo root |
| repo-relative, no slash | `outputs/videos/run/clip.mp4` | ❌ broken — resolves against the *page's* directory |
| over-climbing | `../../data/...` from a depth-1 page | ❌ escapes the repo entirely |

The last two are how the ladder and eval-ladder pages broke. They are now served
through mounts (below) that compensate, but new pages must use rule 2.

---

## Layout

```
diffusion-research/
├── scripts/viewers/
│   ├── viewerctl.py          the tool: mount · check · hub · serve · new   [tracked]
│   ├── registry.json         every viewer, its blurb, group, media wiring  [tracked]
│   ├── build_ctt_v2_corpora.py  builds the corpora viewer's data files     [tracked]
│   └── ctt_v2_corpora.html      that viewer's page source                  [tracked]
├── scripts/build_*_viewer.py generators, one per viewer family             [tracked]
├── outputs/viewers/
│   ├── index.html            the dashboard — generated, never hand-edited
│   ├── <slug>/index.html     a viewer + its media symlinks                 [disposable]
│   └── <slug>/media -> ...   symlink into outputs/videos/... or data/...
└── docs/VIEWERS.md           this file
```

Run-scoped viewers may stay next to their run
(`outputs/videos/<exp>/run_NNNN/viewer.html`) — that keeps the viewer with the
artifacts it describes. Register it by path; promote it into `outputs/viewers/`
with a mount only if it needs media wiring it does not already have.

---

## Creating a viewer

Nothing to register. Put a page where viewers live and it is on the dashboard.

```bash
python3 scripts/viewers/viewerctl.py new my_thing \
    --media outputs/videos/exp_090/run_0001 --group datasets \
    --title "exp_090 — what it shows" --blurb "over what data, what to look for"

# write the page (relative paths only — media/clip.mp4, never /outputs/...)
python3 scripts/viewers/viewerctl.py serve
```

`new` writes the page **and** a `viewer.json` beside it holding title, blurb,
group and `featured`. That sidecar is all the metadata the dashboard needs.

**Discovery.** `hub` scans these locations and shows anything it finds, with or
without a registry entry:

```
outputs/viewers/*/index.html            outputs/reports/*/index.html
outputs/videos/*/run_*/viewer.html      outputs/presentation/*/index.html
outputs/eval/*/viewer/index.html
```

Metadata comes from `viewer.json` if present, otherwise from the page's own
`<title>` and mtime; unlabelled pages land in an **Unsorted** group. Registry
entries always win, so promoting a viewer means moving its sidecar fields into
`registry.json` — worth doing for anything that needs a mount, cross-links, or a
curated blurb.

Multiple runs of one experiment fold automatically: newest is the card, the rest
become its earlier versions.

Write a real blurb either way. The dashboard is read months later by someone
deciding whether to open the page — "what it shows, over what data, what to look
for" beats "viewer for exp_090".

### What the dashboard shows

A **latest bar** across the top: one link per current viewer, newest and
featured first, and nothing else. Cards by category beneath it. Everything
that is not current — earlier builds, and anything whose data went away — sits
in one openable **Earlier versions & archive** block at the bottom, still
clickable, with the reason next to it.

Archiving is automatic: a viewer whose page or media stops resolving drops out
of the bar into the archive by itself, so the top of the page cannot rot.
Explicit `"archived": "<why>"` in the registry still wins.

### Mounts — for pages you should not edit

Certification records, frozen study instruments, and published campaign
references must keep their bytes. A **mount** builds a normal viewer directory
around such a page without touching it: the page is symlinked in (or copied with
its paths rewritten, if they are wrong), and the directories its paths expect are
symlinked beside it.

```json
"mount": {
  "dir": "outputs/viewers/cert_v3_draft8",
  "pages": [{"name": "index.html", "target": "outputs/eval/certification/3.0.0-draft.8/viewer.html"}],
  "links_from": "data/processed/transitions_std121"
}
```

`links` names individual symlinks, `links_from` symlinks every child of a
directory, and `rewrite_abs` writes a derived copy with path prefixes replaced
(used where a page climbs out of the repo). Mounts are rebuilt from the registry
on every `mount`/`serve`, so derived copies never go stale.

### The static server does byte ranges

`viewerctl httpd` (what `serve` runs) is not `python -m http.server` — that one
ignores `Range` and answers 200 with the whole file. Ours answers 206. Two things
depend on it: **seeking inside a video**, and **reading one member out of a tar
shard without downloading the shard**. Measured: 1.17 MB clip pulled out of a
12 GB WebDataset tar in 54 ms, `ffprobe`-valid H.264.

That second one matters for design — a corpus of video inside tars does *not*
force an application server. A static page can fetch
`[offset, offset+size)` from the shard and hand the bytes to `<video>` as a blob
URL, with the offsets coming from a `_viewer_index/*.jsonl.gz`.

That is not theoretical: the ctt_v2 corpora viewer (153,758 samples across
361 GB of tars) **was** an application on port 8799 and is now a static page on
8017. `build_ctt_v2_corpora.py` precomputes what its server did in memory —
axis grouping into `ids_<sub>_<axis>.bin`, rows into `rows_<sub>.jsonl` with a
`rowoff_<sub>.bin` offset table — and the page range-fetches the slice it needs.
Verified at parity against the old server: identical counts, axis groups, group
membership and row fields.

### Server-backed viewers — for pages that are an application

**Currently none.** The class is still supported, for a viewer that must
genuinely compute per request. Media living inside tars is *not* such a case.
These run on their own port and are registered under `"servers"` rather than
`"viewers"`:

```json
{"slug": "ctt_v2_dataset_viewer", "port": 8799,
 "cwd": ".claude/worktrees/bottleneck-branch",
 "cmd": ["python3", "scripts/ctt_v2/dataset_viewer/serve.py", "--port", "8799", "--bind", "127.0.0.1"]}
```

`viewerctl serve` starts each one alongside the static server (skipping any
already up) and the dashboard cards them with a running / not-running pill.
`--static-only` skips them.

A page that fetches its media in JS has no `src` attributes to scrape, so it
declares what it needs with `"check_files"` and `viewerctl check` verifies those
instead.

---

## Maintaining

| Situation | Do this |
|---|---|
| Added a new viewer | Nothing — `serve` finds it. Fill in its `viewer.json` for a proper title and group |
| New build of an existing viewer | Update its `path` and `date`; move the old path into `supersedes` |
| A viewer's data was cleaned up | Nothing — it self-archives with the reason. Add `"archived": "<why>"` to say it better |
| Media stopped resolving | `viewerctl check <slug>` names the first failing ref |
| Wiped `outputs/` | `viewerctl mount --all && viewerctl hub` rebuilds every mount and the dashboard |
| Before a demo | `viewerctl check --strict` exits 1 if anything current is broken |

`viewerctl check` samples 12 references per page and resolves them the way a
browser would, including one level of JS-composed subdirectory. It reports
`live` / `partial` / `broken` / `self-contained` / `missing`, and those badges
are what the dashboard shows.

---

## Current inventory

18 current viewers, all static, all resolving; 4 archived.
Generated view: `outputs/viewers/index.html`.

### Dataset strata & sources
| Viewer | What it shows |
|---|---|
| **S2 — procedural transition operators** ★ | 7,990 clips from 42 shader operators (CTT v2 synthetic stratum); + browse-all and retired/blacklisted pages |
| **S3 — 3D depth-parallax transitions** ★ | all 203 depth clips, three mechanisms side by side |
| **Luma-matte transitions** | 114 clips isolating matte source vs `step()` thresholding |
| **D2 — final synthetic dataset** | exp_077, 252 ref/target pairs with filmstrips (supersedes D1) |
| **HumanVid REAL (Pexels)** | 60-clip real-footage sample, streamed from Pexels (needs internet) |
| **ctt_v2 corpora — refVFX + VFXMaster** ★ | the external corpora along their counterfactual axes: refVFX code 136,800 · I2V_LoRA 6,995 · VFXMaster 9,963. Video read out of the tars by byte range — nothing extracted. Rebuild: `build_ctt_v2_corpora.py` |

### Eval instrument & ladder
| Viewer | What it shows |
|---|---|
| **ladder2 — single reference of truth** ★ | the clean campaign: design, seatbelts, prompt rendering; + DAVIS foreign generations |
| **eval ladder — results viewer** ★ | 1,902 generated transitions with scores, filterable by cell/tier/ontology |
| **Ladder v3 — paired side-by-side** | certified v3.0.0 rows, GT/demo/floors colour-coded, 1,800 clips |
| **Certification record 3.0.0-draft.8** | the certification verdict + results explorer (six metrics, one exam, 3,174 clips) |
| **Transition-eval harness · per-run viewer** | the viewer shipped with every eval run, shown on exp_058 |
| **Transition taxonomy v2 — validate** | click through 107 clips to confirm class assignments |

### Experiment runs
| Viewer | What it shows |
|---|---|
| **exp_083 — D3 pilot** ★ | 109 depth-parallax clips + filmstrips (supersedes exp_080, exp_076 ×2) |
| **exp_075 — procedural operator engine** | 73 operator demos; the engine behind S2 |

### Reports, decks & studies
| Viewer | What it shows |
|---|---|
| **Two-week progress report** | latent tricks → certified instrument, written up |
| **Week 2 — cross-ontology examples** | 5-row deck, inputs + outputs at two seeds |
| **Blind 2AFC study** | frozen forced-choice instrument, 93 clips, mounted read-only from `misc/` |
| **Transition-hardness playground** | interactive scatter over hardness signals (Plotly CDN) |

★ = current campaign work.

### Archived
| Viewer | Why |
|---|---|
| D1 synthetic pilot | superseded by D2 |
| HumanVid sample (40 clips) | `data/raw/humanvid_sample` was cleaned — only filmstrips resolve |
| VC Dissolve viewer (exp_021) | exp_021 run_0002 outputs deleted; only run_0004 survives |
| eval ladder v1 one-pager | superseded by ladder2 REFERENCE |

### Deliberately not registered
- `outputs/viewers/s2_dataset/shaders/*.html` — 42 per-shader detail pages, reached from the S2 browse page.
- `*/viewer_template*.html`, `eval_ladder/viewer/template.html` — templates consumed by generators, not pages to open.
- `.claude/worktrees/**` — duplicates of tracked pages on other branches.
- `misc/shotbridge/web/.next/**` — Next.js build output for a separate product.
- `data/raw/cifar10/**/readme.html` — vendor file.
- `status_vcbench.html` (workspace root, Apr 2026) — one-off status snapshot, outside the repo.
