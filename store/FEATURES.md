# Feature namespaces — the registry (contract v2, clause 10)

**One rule, two cases.** Inside `features/`: one folder per video (the video's stem), one file per namespace.
- Store gens: `features/` sits **next to** `videos/` — `<variant>/videos/<item>__s42.mp4` ↔ `<variant>/features/<item>__s42/<ns>.npz`.
- Clips not under a `videos/` folder (corpus `<class>/<clip>.mp4`, endpoint clips `eval_ladder/conds/<clip>_start9.mp4`):
  `features/` sits **inside** the clip folder — `<class>/features/<clip>/<ns>.npz`, `eval_ladder/conds/features/<clip>_start9/<ns>.npz`.

`FeatureStore.path(video, ns)` implements exactly this: `root = video.parent.parent if video.parent.name == "videos" else video.parent`;
`root / "features" / video.stem / f"{ns}.npz"`. Files are the truth; `features/manifest.jsonl` is an index rebuilt from them.
Design + migration record: `misc/2026-09-17_feature_store/PROPOSAL.md`.

A namespace tag is **frozen once written**. Changing any pin below (model, revision, preprocessing, tracking
protocol) means a NEW tag; existing files are never overwritten or reinterpreted.

| namespace | backbone (pin) | preprocessing | arrays | consumers |
|---|---|---|---|---|
| `dino_cls@dinov2b-r256` | `facebook/dinov2-base` @ `f9e44c814b77203eaa57a6bdbbd535f21ede1415` | decode short side 256 (`video_io.load_frames`); DINOv2 processor; CLS token; L2-normalized; fp16 model, fp32 output | `feats [T,768] f32` | transition-eval v4 (M1a/M2/M3; legacy `dino_arr_*`) |
| `cotracker3@g20-m384-v2` | CoTracker3 offline (`facebookresearch/co-tracker:cotracker3_offline`), ckpt sha256 `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834` | frames resized so max side = 384; grid 20×20 queried at frame 0 and the middle frame, backward tracking (tracking protocol tag `v2` = `motion.Tracker.CACHE_TAG`) | `tracks [T,N,2] f32`, `vis [T,N] f32` | v4 M1b/M1c; `det_motion_fidelity`; hand-off motion |
| `lpips_t@alex-r256` | LPIPS `alex` | consecutive-frame LPIPS d(t,t+1) on frames decoded at short side 256 (`endpoints.temporal_lpips`, cache tag `alex-v1`) | `d [T-1] f32` | v4 seams (`max_seam_z`, `prefix/suffix_seam_z`) |
| `clip_b32@r256` | `openai/clip-vit-base-patch32` | decode short side 256; `get_image_features` pooler output, projected 512-d, L2-normalized, per frame | `feats [T,512] f32` | competitor lenses `clip_sim_ref/input`, `motion_smoothness` |
| `videoprism@f16r288` | `MHRDYN7/videoprism-base-f16r288` @ `c5bb17adeb575aaf1d46b56c5b9dfa1b00465c80` (community torch port of `google/videoprism-base-f16r288`) | shipped preprocessor: 16 frames, 288×288, rescale 1/255, no normalize; spatial mean-pool of `last_hidden_state`; L2 per frame | `feats [16,768] f32` | competitor lens `videoprism_sim_ref/input` (refVFX headline) |
| `raft_mag@r256` | RAFT-large, torchvision `Raft_Large_Weights.DEFAULT` | decode short side 256, dims floored to multiples of 8, longest side ≤1024; per-step mean flow magnitude in pixels | `mag [T-1] f32` | dynamic degree (info column) |
| `clip_l14@r224` | `openai/clip-vit-large-patch14` | CLIP preprocessor (224); image features, L2-normalized, per frame | `feats [T,768] f32` | aesthetic quality (LAION head applied at scoring) |

## File format

Two files per (video, namespace):
- `<ns>.npz` — the arrays above, exactly as the extractor emits them (compressed npz). A **migrated** file is a
  hard link of the legacy cache file (same inode, zero extra bytes; legacy caches are kept, owner decision 2026-09-17);
  legacy files may carry an extra `src`/`fps` array — harmless, documented here.
- `<ns>.json` — the sidecar meta:
  `{"ns","video":"<path relative to repo root>","video_sha256","video_size","video_mtime_ns","host","code_sha","created","origin","shape","dtype","bytes"}`
  where `origin` is `"extracted"` or `"migrated:<legacy file path>"`. `host` names the machine that EXTRACTED the
  arrays (the host rule below guards cross-machine drift). Legacy caches carry no host, so a **migrated** file's
  extraction host is unknown: `host` is `null` and an extra `migrated_by_host` records the node that ran `migrate`.
  Writes are atomic (`<file>.tmp-<pid>` → `rename`); the `.json` is written last, so a `.npz` without its `.json` is an
  interrupted write and `fsck` reports it.

**Synthetic controls** (v4 harness, `eval/4.0.1`): the lerp / static-hold degenerate control for a gen is synthesized
at scoring and has no video of its own, so its features persist NEXT TO the gen's, as `<ns>.ctl-<name>.npz` +
`<ns>.ctl-<name>.json` in the gen's feature folder (e.g. `dino_cls@dinov2b-r256.ctl-lerp.npz`; `name` ∈ {`lerp`,
`hold`}). Same atomic + sidecar-last convention; the sidecar adds `"control": "<name>"`, sets `"origin":
"control:<name>"`, and carries the **gen video's** identity (`video`/`video_sha256`/`video_size`/`video_mtime_ns`) so
`fsck` — which stats each item folder's gen mp4 — reads a control as fresh. Written by
`diffusion.transition_eval.store_io.HarnessStore` (the scoring path), not by `scripts/store_features.py`.

`features/manifest.jsonl`: the concatenation of the sidecars, one per line. Rebuilt from the files by
`scripts/store_features.py fsck --rebuild-manifest`; never hand-edited.

`videos/SHA256SUMS` (tracked in git): standard `sha256sum` format, one line per mp4; `sha256sum -c` verifies a variant.

Each gen `meta.yaml` carries a `features:` block — `{<ns>: {have: n, of: n_videos, hosts: [...]}}` — refreshed by
`extract`/`migrate`; it is the human-readable coverage summary, the manifest is the authority.

## Tooling

- library: `src/diffusion/feature_store.py` (`FeatureStore.path/has/get/put/coverage/fsck/rebuild_manifest`)
- CLI: `scripts/store_features.py coverage | fsck | sha256sums | migrate | extract`
- coverage matrix: `store/FEATURES_COVERAGE.md` (generated by `coverage --md`; commit it when it changes)

## Host rule

Features record the extracting host. The store treats machine as identity-bearing (v4 does not reproduce
eps↔DeltaAI at the 0.005 bar); `fsck` warns when one variant mixes KNOWN extraction hosts, and an eval's `meta.yaml`
names the host it scored on. Migrated features have an unknown extraction host (`host: null`); a namespace whose
present features are all migrated is reported as `legacy` (not a mixed-host warning), and only known hosts count toward
the mixed-host guard.
