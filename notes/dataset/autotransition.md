# AutoTransition (yaojie-shen / HuggingFace)

HF: https://huggingface.co/datasets/yaojie-shen/AutoTransition (CC-BY-4.0, ~52.9 GB)

## What it is

Short vertical mobile-video templates (Douyin/Huoshan origin) annotated with **per-segment transition labels**. Used for transition recommendation / autoediting research.

- **Splits:** `train` 30,000 templates · `test` 5,000 templates · **35,000 total**
- **Annotations:** 187,644 transition events across **107 distinct named transitions**
- **Top types by frequency:** `direct_cut` (26k), `pull_in` (15.5k), `mix` (13.6k), `pull_out` (12.9k), `circle_1` (8.2k), `open`, `windmill`, `cube`, `switch`, `left`, `pane`, `circle_2`, `right`, `black_fade`, `turn_page`, …
- The taxonomy is *transition-style* labels (`direct_cut`, `clock_wipe`, `kaleidoscope`, `floodlight`…), **not** semantic transition categories like `shadow_smoke`. Our local `data/processed/transitions/shadow_smoke/` is from a different source — not present here.

## Annotation JSON

`transition_annotation_en_url_fixed.json` (53 MB, top-level keys `templates`, `statistic`):

```json
{
  "templates": {
    "train": {
      "<template_id>": {
        "url": "http://api.huoshan.com/hotsoon/item/video/_playback/?video_id=...",
        "transition": [
          {"type": "transition", "name": "pull_out", "start": 1002, "end": 1002, "duration": 0},
          ...
        ]
      },
      ...
    },
    "test": { ... }
  },
  "statistic": { "transition": {"direct_cut": 26293, ...} }
}
```

- `start` / `end` are integer milliseconds; `duration = end - start`. Many transitions have `duration = 0` (cut-style).
- `url` fields all point to internal `api.huoshan.com` endpoints — **assume unreachable**. The actual video bytes are in the `template_download.tar.gz.NN` parts on HF, keyed by `template_id`.

## Archive layout — IMPORTANT GOTCHA

The repo ships **13 split parts** named `template_download.tar.gz.00` … `.12` (twelve at 4.29 GB + one at 1.35 GB). Despite the `.tar.gz` extension, **the bytes are plain POSIX tar, not gzipped** — the parts are a single tar that was `split` into chunks. Verified via `file` (POSIX tar archive (GNU)) and direct `tar -tvf` on the first chunk.

Practical consequences:
- **Do not** `gunzip` them; just concatenate or stream as tar.
- Because tar is sequential, you can grab the first N bytes of `.00` via an HTTP Range request and `tar -x` will extract everything that fits (it'll error `Unexpected EOF` on the final partial file — expected and harmless).
- To extract the full set, `cat template_download.tar.gz.* | tar -xf -` (no `z` flag).

Internal layout once extracted:

```
template_download/
  <template_id>/
    <id_suffix>out_video.mp4   # one file per template; <id_suffix> = last 4 chars of template_id
```

Typical clip: **H.264 480×854 (vertical), 30 fps, ~11 s**, with AAC audio. File sizes ~250 KB – 1.6 MB. Template IDs in the archive match keys under `templates[split]` in the JSON.

## Partial download recipe (cheap sampling)

Useful when you don't want to commit to 52.9 GB:

```bash
DST=/workspace/diffusion-research/data/raw/AutoTransition
mkdir -p "$DST/extracted"

# 1. annotations (53 MB)
curl -L --fail -o "$DST/transition_annotation_en_url_fixed.json" \
  "https://huggingface.co/datasets/yaojie-shen/AutoTransition/resolve/main/transition_annotation_en_url_fixed.json"

# 2. first 500 MB of part .00 → ~485 complete template clips
curl -L --fail -H "Range: bytes=0-524287999" \
  -o "$DST/part00_first500MB.tar" \
  "https://huggingface.co/datasets/yaojie-shen/AutoTransition/resolve/main/template_download.tar.gz.00"

# 3. extract (EOF error on final partial file is expected)
tar -xf "$DST/part00_first500MB.tar" -C "$DST/extracted" || true
```

Scales linearly: ~1 MB per template → ~4,000 templates per `.00` part, ~35,000 total across all 13 parts.

## Local state (as of 2026-05-18)

- `data/raw/AutoTransition/transition_annotation_en_url_fixed.json` — full 53 MB JSON
- `data/raw/AutoTransition/part00_first500MB.tar` — 500 MB Range slice of `.00`
- `data/raw/AutoTransition/extracted/template_download/` — **485 template MP4s** (subset of the train split; first 3 IDs: `6734985427754765575`, `6740860015638220040`, `6742039444506823939`, all present in the JSON under `templates.train`)
