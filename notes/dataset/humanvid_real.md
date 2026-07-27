# HumanVid — the REAL (Pexels) subset

Investigated 2026-07-27. **Bottom line: located, fully characterised, and NOT usable
by us.** The real portion is a list of Pexels URLs, and the Pexels Terms of Service
prohibit exactly the thing we would need to do with them.

Companion notes: the synthetic HumanVid families (UE `3d_video_*` / `generated_video_*`)
are the 201.7 GB Apache-2.0 tree on HF `zhenzhiwang/HumanVid` — a different, separately
licensed thing that this note does not cover.

---

## 1. Where the real subset actually lives

Not on HuggingFace. The HF dataset tree contains only the two synthetic families.
The real portion is released as **URL lists + camera annotations** from the GitHub repo
[`zhenzhiwang/HumanVid`](https://github.com/zhenzhiwang/HumanVid):

| Artifact | Location | Contents |
|---|---|---|
| Training URLs | Google Drive folder `1UGEkOKXYX9BGUFz0ao6lOGXkZjQGoJcZ` | `pexels-horizontal-urls-new.txt` (11,411), `pexels-vertical-urls-new.txt` (7,851) |
| Camera params | same folder, `camera_tram.zip` (150 MB) | 19,429 per-clip TUM-format trajectories, **one line per frame** |
| Older/worse cameras | same folder, `camera_old.zip` (163 MB) | superseded 2025-01-03 by `camera_tram` |
| Stale URLs | `old_invalid_urls/` subfolder | pre-2024 URLs; Pexels renamed files, **use the `-new` lists** |
| Test set (71 clips) | in-repo `data/test_set/pexels-test-urls.txt` | subset of the original 40 vertical + 40 horizontal |

`gdown --folder <id>` retrieves the whole folder; no auth needed.

Fields carried: **URL, resolution, fps, per-frame camera pose**. There is *no* caption,
*no* clip start/end (segmentation is left to `tools/get_video_segments.py`), and pose must
be re-extracted locally with DWPose — poses are not shipped for the real half.

Useful accident: every Pexels CDN URL encodes its own geometry —
`.../video-files/<id>/<id>-hd_<W>_<H>_<fps>fps.mp4` — and each camera file has exactly one
line per frame. **Together those give resolution/fps/frame-count/duration for all ~19k clips
with zero media downloaded.** Verified against `ffprobe` on 10 random clips: resolution, fps
and frame count matched the manifest on 10/10.

## 2. Licence position — the blocker

The Apache-2.0 (HF) / CC-BY-4.0 (GitHub) covers HumanVid's *own* code, camera annotations
and UE renders. It **cannot** relicense third-party Pexels footage, and the authors say so:
*"The pexels video data is collected from the Internet and we cannot redistribute them."*

The governing terms are therefore Pexels'. The Pexels **License** page is permissive and
silent on ML, but the **Terms of Service** are not:

> "Data mining, extraction, scraping and the use of programs or robots for automatic data
> collection and/or extraction of digital data … is strictly prohibited for all unauthorised
> purposes, **including without limitation for machine learning purposes**."

> "Bulk, large-scale or systematic copying of Content is strictly prohibited unless explicit
> permission has been granted."

And the API terms close the "but we'd use the official API" loophole:

> "You may not use the API to collect Pexels photos/videos or metadata at scale to train,
> fine-tune, **evaluate**, or develop ML/AI models **or datasets**, unless you have explicit
> permission from Pexels."

That covers evaluation sets and dataset construction, not just training. **Conclusion: do not
fetch. Not 19k, not 60.** The 100 %-Apache-2.0 story on the HF card applies only to the
synthetic half. (HumanVid the paper is a research work under a different risk posture than a
product pipeline; their doing it is not cover for us doing it.)

## 3. What the corpus is (measured, no downloads)

`scripts/analyze_humanvid_real.py` → `data/manifests/humanvid_real/fitness_report.json`.

- **19,262 clips**: 11,411 landscape + 7,851 portrait. 19,173 have camera annotation.
- **90.2 h** total; median clip 13.8 s (p10 7.8 / p90 29.5 / max 155.8).
- Resolutions: 1920×1080 (8,886), 1080×1920 (5,959), 2048×1080 (2,505), 1080×2048 (1,883); all 1080p+.
- fps: 25 → 12,973 · 30 → 3,732 · **24 → only 2,538** · 60 → 19.
- Liveness: **120/120 sampled URLs return HTTP 200**; mean clip 8.9 MB @ 4.1 Mbps.
- **Projected full fetch ≈ 167 GB** (measured from `Content-Length`, not guessed).

HumanVid's own curation rules (paper §3): ~100 Pexels keywords; largest human bbox ratio
**r > 0.07**; **n ≤ 4 people**; keypoint motion Δp̄ > 0.01 (statics dropped); no exits /
entrances / occlusions; clips with **shot changes excluded**; SLAM-failed clips labelled
static-camera.

## 4. Fitness against our endpoint contract (480×640 · 121f · 24 fps)

| Criterion | Verdict |
|---|---|
| Resolution & length | **Pass.** 97.5 % hold ≥5.04 s; all crop to 480×640 with no upscale; ~54k distinct 121f endpoints in principle |
| fps | **Resample needed.** Only 13 % native 24 fps; 67 % are 25 fps |
| Single-shot | **Pass.** Shot changes were filtered out upstream — low boundary risk |
| Single subject | **Partial fail.** Their rule is "few people (n≤4)", not one; prominence floor r>0.07 is under half our 0.15 |
| Letterboxing | **Low risk.** Native stock resolutions throughout; no baked-in bars |
| Portrait crop | Landscape keeps only **~42 %** of frame area, vertical keeps **75 %** |
| Diversity | **None — volume only.** Pool is already 85 % `person`; this is 100 % human-centric by construction |

Non-obvious crop interaction: cropping a landscape frame to 0.75 AR *raises* a subject's
relative area (r ≈ 0.07 → ≈ 0.17), which can push clips over our 0.15 bar — but it also
discards 58 % of the framing and risks cutting limbs. Vertical clips invert this: they keep
more frame, so the subject stays proportionally smaller (0.07 → ≈ 0.09, still under our bar).

## 5. Viewer

`scripts/build_humanvid_real_viewer.py` → `outputs/viewers/humanvid_real/index.html`
(served at `http://localhost:8017/outputs/viewers/humanvid_real/index.html`).

60 clips, 30 vertical + 30 horizontal. **Streams straight from the Pexels CDN — nothing is
copied to disk**, which is what keeps the page compatible with §2. Filmstrips are rendered
client-side into a `<canvas>` from the already-streaming video (tiles downscaled to 200 px,
max 3 concurrent builds) so no frames are stored either. Pexels serves
`access-control-allow-origin: *`, so the canvas is untainted.

Two gotchas found while building it:

- **The repo HTTP server sends `Content-type: text/html` with no charset**, so a viewer
  without `<meta charset="utf-8">` mojibakes every `·`, `—` and `≥`. The older
  `outputs/viewers/humanvid_sample/index.html` has this bug.
- **Playwright's `chrome-headless-shell` has no H.264**, so video tiles render black in
  headless screenshots (`canPlayType('avc1.42E01E')` → `''`). This is not a page bug — the
  CDN returned 206 partial-content to the page and every clip probes as h264. Real browsers
  play them.
