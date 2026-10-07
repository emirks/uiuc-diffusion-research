# Feature namespaces — the registry (contract v2, clause 10)

**One rule, two cases.** Inside `features/`: one folder per video (the video's stem), one file per namespace.
- Store gens: `features/` sits **next to** `videos/` — `<variant>/videos/<item>__s42.mp4` ↔ `<variant>/features/<item>__s42/<ns>.npz`.
- Clips not under a `videos/` folder (corpus `<class>/<clip>.mp4`, endpoint clips `eval_ladder/conds/<clip>_start9.mp4`):
  `features/` sits **inside** the clip folder — `<class>/features/<clip>/<ns>.npz`, `eval_ladder/conds/features/<clip>_start9/<ns>.npz`.
- Derived condition clips of one baseline live with their campaign, same layout: the 16-fps resamples of start9/end9 for the VACE
  first-last CLIP baseline (`misc/2026-09-20_teg_baselines/conds_16fps/<endpoint>_{start6,end4}.mp4`, features inside
  `conds_16fps/features/`; listed in `population_flowwin.json`; resolved by `scripts/store_eval_common.cond_clips` for grid type VACE16).

A **derived** namespace (built from another namespace's stored arrays, not from the video — e.g. `trackdesc@cotracker3-s64-v1`
off `cotracker3@g20-m384-v2`) names its source namespace in the registry row above and records it as `source_ns` in the sidecar.

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
| `raft_flow_win@r256` | RAFT-large, torchvision `Raft_Large_Weights.DEFAULT` (same pin as `raft_mag@r256`) | decode short side 256, dims floored to multiples of 8; dense flow FIELDS of the given windows only: `flow_start` = the 8 steps of frames 0..8, `flow_end` = the 7 steps of the last 8 frames; pixels of the resized frame, fp16 | `flow_start [8,H,W,2] f16`, `flow_end [7,H,W,2] f16` | endpoint motion (given clip vs the output's pinned window, per-step dense EPE); later subject-masked motion |
| `viclip@l14-f8` | ViCLIP ViT-L/14, InternVid-10M-FLT (`ViClip-InternVid-10M-FLT.pth` from `OpenGVLab/VBench_Used_Models`, code vendored from VBench under `src/diffusion/third_party/viclip`) | VBench `overall_consistency` recipe: 8 frames, middle of 8 equal segments, decoded at native resolution; bicubic short side 224 (no antialias), center crop, CLIP mean/std; L2-normalised video embedding | `feat [768] f32`, `frame_idx [8] i32` | text consistency (VBench overall consistency): cosine to the prompt embedding of the same model's text tower |
| `trackdesc@cotracker3-s64-v1` | **derived** from the stored `cotracker3@g20-m384-v2` tracks (source_ns) — no backbone | `dirs = _velocity_directions(tracks, vis, 64, 0.2, 0.1, 0.05)` (unit velocity directions of the kept tracklets, `diffusion.transition_eval.motion`); `px = step_features(tracks, vis)` (whole-field 31-step track descriptor, `misc/2026-09-02_temporal_dynamics_metric/run_motion_descriptors.py`) — exactly the `dirs_for` / `px_for` recipes of `score_v3_mf.py` | `dirs [M,64,2] f32`, `px [31,18] f32` | transport v5 PX channel (pixel transport) + MF fidelity |
| `flow_u32@raft-r256-g24x32` | RAFT-large, torchvision `Raft_Large_Weights.DEFAULT` (same pin as `raft_mag@r256`) | decode short side 256, dims floored to /8 (`RaftMagExtractor._resize`); T=32 uniform steps from `idx = round(linspace(0, T-1, 33))`; per step the dense RAFT flow field of the resized frames, divided by that frame's diagonal `sqrt(h^2+w^2)` (fraction of the frame diagonal), adaptive-average-pooled to a 24×32 grid; last axis `(dx, dy)` like `raft_flow_win`, fp16. T≥33 for the whole gridv3 population (idx strictly increasing); if T<33 the round repeats indices, recorded in the stored `idx` | `flow [32,24,32,2] f16` (fraction of frame diagonal per step), `idx [33] i32` | Flow MSE (eval 049; whole-video optical-flow fidelity of the reference vs the generated video) |
| `action@swin3db-k400-u32` | Video Swin-B, torchvision `Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1` (`swin3d_b_22k-7c6ae6fa.pth`, 81.6 top-1 K400) | decode short side 256; 32 uniformly sampled frames `idx = round(linspace(0, T-1, 32))` (the model's native clip length) through the shipped `VideoClassification` transform (resize short side 256 — a no-op here, center crop 224, ImageNet mean/std, permute to [C,T,H,W]); `swin3d_b(weights=...)` eval, no_grad, batch of 1 → logits [400]; prob = softmax(logits). T≥32 for the whole gridv3 population; if T<32 the round repeats indices, recorded in the stored `idx` | `prob [400] f32` (softmax), `logit [400] f32`, `idx [32] i32` | Action KL (eval 049; KL between the reference and the generated video's Kinetics-400 action-class distributions) |

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
`fsck` — which stats each item folder's gen mp4 — reads a control as fresh. Written through the library
(`FeatureStore.put(gen, ns, arrays, meta, variant="<name>")`) by the scoring path
(`diffusion.transition_eval.store_io.HarnessStore`), not by `scripts/store_features.py`. Controls are **first-class**
in the tooling: `fsck` pairs/sidecars/stale-checks each `<ns>.ctl-<name>` exactly like a namespace (but keeps it out of
the per-namespace host tracking — a control's host is the run that synthesized it, not an extraction host), and
`coverage` reports them in a **separate `controls` column** — the `<ns>` cells count only real-namespace files.

`features/manifest.jsonl`: the concatenation of the sidecars, one per line. Rebuilt from the files by
`scripts/store_features.py fsck --rebuild-manifest`; never hand-edited.

`videos/SHA256SUMS` (tracked in git): standard `sha256sum` format, one line per mp4; `sha256sum -c` verifies a variant.

Each gen `meta.yaml` carries a `features:` block — `{<ns>: {have: n, of: n_videos, hosts: [...]}}` — refreshed by
`extract`/`migrate`; it is the human-readable coverage summary, the manifest is the authority.

## Tooling

- library: `src/diffusion/feature_store.py` (`FeatureStore.path/has/get/read_meta/put/coverage/fsck/rebuild_manifest`;
  `put(..., variant="<name>")` writes a synthetic control, `put(..., video_sha256=…)` / `sha_from_sums(video)` skip
  re-hashing by reusing the sha from the clip's `SHA256SUMS`)
- CLI: `scripts/store_features.py coverage | fsck | sha256sums | migrate | extract`
- coverage matrix: `store/FEATURES_COVERAGE.md` (generated by `coverage --md`; commit it when it changes)

## Host rule

Features record the extracting host. The store treats machine as identity-bearing (v4 does not reproduce
eps↔DeltaAI at the 0.005 bar); `fsck` warns when one variant mixes KNOWN extraction hosts, and an eval's `meta.yaml`
names the host it scored on. Migrated features have an unknown extraction host (`host: null`); a namespace whose
present features are all migrated is reported as `legacy` (not a mixed-host warning), and only known hosts count toward
the mixed-host guard.
