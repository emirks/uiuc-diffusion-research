# EffectData (local mirror)

Full local copy of the HuggingFace dataset **[`ysy31415926/EffectData`](https://huggingface.co/datasets/ysy31415926/EffectData)**
(EffectMaker paper, arXiv:2603.06014 · project [effectmaker.github.io](https://effectmaker.github.io) · Apache-2.0).
A character-centric game-VFX dataset: an effect applied to a subject (person/animal),
start frame = the clean subject, end = after the effect.

Downloaded & verified 2026-08-26 (every zip sha256-checked vs the HF LFS manifest).

## What it is

| | |
|---|---|
| videos | **132,850** mp4 (1056×704) |
| effect classes | **3,061** (median 45 clips/class, min 2, max 135) |
| size | **821 GB**, shipped as **3,063 per-effect zips** (`Videos/<effect>.zip`, ~46 mp4/zip) |
| captions | per-video `prompt / vfx / instruction / abstract`, EN + ZH (`annotations.json`) |

## Directory layout (this folder = the dataset home)

```
data/raw/effectdata/
├── Videos/                 3,063 per-effect zips  (the raw corpus, 821 GB)
├── example_preview/        3,061 preview clips    (one per effect, for the viewer)
├── cf_media/               137 clips              (counterfactual-gallery media)
├── annotations.json        authors' per-video metadata (132,850 records)
├── effects_index.json      per-effect index        (viewer)
├── cf_data.json            counterfactual gallery  (viewer)
├── preview_manifest.json   preview index           (viewer)
├── videos_manifest.json    zip list + size + sha256 (integrity / re-download)
├── effect_names_list.csv   effect name list
├── axisA_degree.png        Axis-A distribution chart (see counterfactuality.md)
├── README.md               this file
└── counterfactuality.md    the counterfactual structure + how to build CF sample sets
```

All build/download/analysis code lives in **`scripts/effectdata/`** (not here) and is
repo-anchored, so it runs from any cwd.

## Filename scheme (undocumented by the authors — derived here)

Two generations coexist inside the zips:

- **tagged** — `<effect>,<subject-id>,<tag>.mp4`   (tag ∈ `F`/`M`/`Z` = woman/man/animal)
- **untagged** — `<effect>,<uuid>.mp4`   (831 clips, ~143 effects)

The **middle token is the subject/source id**: clips sharing it share the same start
frame (verified — codec-noise-close, ~0.7/255, not byte-exact). This is the key that
makes counterfactual sets possible; the authors document none of it (checked HF card,
paper, project page). See `counterfactuality.md`.

## Getting samples nicely

**A whole effect** — the zip *is* the effect's clips:
```bash
unzip -j data/raw/effectdata/Videos/Fireball_from_hands.zip -d /tmp/fireball
```

**One clip without unzipping the whole file** — extract a single member:
```python
import zipfile
z = "data/raw/effectdata/Videos/Fireball_from_hands.zip"
member = "Fireball_from_hands/Fireball_from_hands,02005942,F.mp4"   # == annotations video_path
with zipfile.ZipFile(z) as zf:
    zf.extract(member, "/tmp/one")
```

**One clip without downloading the zip at all** (HTTP range on the HF-hosted zip):
```bash
python scripts/effectdata/remote_zip.py    # pulls a single member via range reads (~4 MB, not ~250 MB)
```

**All clips of one subject = a counterfactual set** → see `counterfactuality.md` §"Building a counterfactual set".

## Reproduce / re-fetch

```bash
python scripts/effectdata/fetch_manifest.py     # rebuild videos_manifest.json from HF
python scripts/effectdata/download_videos.py    # resumable, sha256-verified, idempotent
```
`data/raw/` is gitignored — the 821 GB is never committed; re-downloadable from HF anytime.
