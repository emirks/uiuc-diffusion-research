# P3b — scorers read/write through the feature store: implementation report

**Author:** opus48 implementer · **Date:** 2026-09-18 · **Branch:** `feature-store` · **Machine:** DeltaAI login (gh-login, aarch64) · **No GPU, no sbatch/srun.**
Commits: `975f471` (Part A, harness + reference artifact + SPEC/VERSION/tests) · `3df5afa` (Part B, lens scorer). Both pushed; `ahead=0 behind=0`.
Authority followed: `BRIEF_P3b.md`, `PROPOSAL.md` §2/§6, `store/FEATURES.md`, SPEC §9/§10, the certified `transition_eval` cache code, `score_v3.sbatch`, `SCORERS.json`.

## 1. What changed, per file

### Part A — harness (`src/diffusion/transition_eval`), I/O-only

| file | change |
|---|---|
| `store_io.py` (**new**) | The scoring-path bridge to `FeatureStore`. `HarnessStore(store, code_sha)` reads/writes the three eval namespaces by *identity*: `RealVideo(path)` → `store.get/put(path, ns)`; `Control(gen, name)` → `<ns>.ctl-<name>.npz`+`.json` next to the gen's features (atomic tmp+rename, sidecar last, gen video identity copied in so `fsck` reads them fresh). Namespace consts `DINO_NS`/`TRACK_NS`/`LPIPS_NS`. numpy + stdlib + `feature_store` at import (no torch of its own). |
| `pipeline.py` | `process_video_file` is now **dual-mode**: a `FeatureStore`/`HarnessStore` backend → store path (new `process_video_store`, decode-skip when dino+tracks present); a legacy `cache_dir` Path → the **byte-identical** old path (kept for `certify/` + `workbench/` and `test_certify_v3`). `process_video` (legacy) unchanged. New `process_video_store(frames, identity, hstore, …)` builds a `VideoBundle` reading/writing dino+tracks via the store; sets `bundle["identity"]`. |
| `score.py` | `--cache-dir` → `--store-root` (default repo root); `--lpips-cache` **removed**. Builds `HarnessStore(FeatureStore(store_root))`; `_ref_bundle_cache` + the gen loop go through it. Gens **always decode** (endpoints recompute fresh). Temporal-LPIPS via new `_temporal_lpips_stored` (store `lpips_t` hit/miss). Endpoints via `_cached_endpoint(…, cache_dir=None)` — fresh, no pair cache. Controls via `process_video_store(cframes, Control(gpath, "lerp"/"hold"), …)`. Results provenance gains `feature_store: {root, namespaces, code_sha}`. **Legacy `_cached_endpoint`/`_endpoint_key`/`lpips_warm` kept byte-identical** (imported by `test_certify_v3`; no longer on the deployed path). |
| `versioning.py` | `PINS['reference_v4_sha256']` → `459fd9a71bb5…` (was `e6ea4011…`); `INSTRUMENT_PATHS` gains `tests/test_transition_eval_store.py`. |
| `VERSION` | `4.0.0` → `4.0.1-draft.1`. |
| `reference_v4.npz` | replaced by the grid-v3 amendment build (sha256 `459fd9a7…`, verified) — the artifact that scored evals/028 + evals/030. Committed with the version bump. |
| `SPEC.md` | §9 implementation-map paragraph "Feature I/O — the store (4.0.1)"; spec-changelog entry `4.0.1-draft.1` (I/O-only; endpoint-LPIPS pair cache dropped; reference = 459fd9a7 grid-v3 build previously uncommitted in the eval-v4-cert worktree; numeric identity to be certified by bar-8 in P4; branch = `feature-store`, owner-approved §10 deviation). |
| `store/FEATURES.md` | one paragraph under **File format** documenting the synthetic-control files (`<ns>.ctl-<name>.npz`+`.json`, `origin: control:<name>`, gen identity in the sidecar). |
| `tests/test_transition_eval_store.py` (**new**) | 7 store-I/O contract tests (tmp `FeatureStore` roots + fake extractors): real-video put/get/has roundtrip, hit-skips-extract numeric identity, store-branch warm decode-skip + cold decode, control persists next-to-gen + `fsck`-clean, control get/has, `_temporal_lpips_stored` roundtrip. |

`features.py` / `motion.py` / `endpoints.py` are **untouched** — their legacy cache helpers (`array_features`, `cached_track`, `cached_temporal_lpips`, `lpips_cache_path`, …) stay for the legacy dual-mode branch + `workbench/` + `test_certify_v3`. No measurement code changed. `judge_gemini.py` is **untouched** (see open questions).

### Part B — competitor lenses (`misc/2026-08-13_baseline_metric_table/their_metrics`)

| file | change |
|---|---|
| `score_batch.py` | `--store` (default on, `--no-store` restores legacy) + `--store-root` + `--dry-run`. New `StoreLenses`: per-VIDEO `clip_b32@r256`/`videoprism@f16r288`/`raft_mag@r256`/`cotracker3@g20-m384-v2` read from / written to the store by video path, **misses filled by `diffusion.feature_extractors.REGISTRY`** (single preprocessing source). Input FRAME embedded fresh via the same REGISTRY model instances (`clip_frame`/`vp_frame`). `.track_cache` retired (tracks = the cotracker3 namespace). `score_one` forks array acquisition on `M.store`; the **metric arithmetic is unchanged**. |
| `SCORERS.json` | `impl_sha` `d63935f4` → `8a808635`; `impl_sha_prev` + a change note recording that the paper's `rows_v3` KEEP `d63935f4` and the 20-row numeric reproduction is P4. |
| `test_store_lenses.py` (**new**) | 6 fake-extractor CPU tests: `video()` hit-no-extract, miss extract+write-back+re-hit, tracks two-array, `clip_frame`/`vp_frame` shape+L2-norm, `parse_item` contract. |

## 2. pytest (CPU, `$LAB/envs-aarch64/ltx2/bin/python`)

Required set (SPEC §10 command), authoritative final run:
```
PYTHONPATH=$PWD/src pytest -q tests/test_transition_eval*.py tests/test_versioning.py
49 passed, 15 warnings in 34.54s
```
(= 42 pre-existing + 7 new `test_transition_eval_store.py`.) Full instrument suite incl. `test_certify_v3.py` (proves the legacy dual-mode path + retained helpers still pass):
```
pytest -q tests/test_transition_eval*.py tests/test_versioning.py tests/test_certify_v3.py
75 passed, 19 warnings in 91.38s
```
Part B: `pytest -q test_store_lenses.py` → **6 passed** (56.30s, torch import). (warnings = torch NVML/JIT noise from `diffusion/__init__`.)

## 3. CPU acceptance on real data (Part A)

3-item manifest built from `store/gens/013_dualforce_control/03_neutral_v3__dai` (item fields reconstructed from the gen's `grid.jsonl` + the evals/028 item ids; manifest at `acceptance/eval_accept3.json`), scored with the **same `--corpus`/`--reference-corpus` as `score_v3.sbatch`** (677-clip + 222-clip pin), `--controls off`, device auto=CPU. Pre-checked: all 677 corpus clips + the 3 gens have dino+tracks in the store (no CoTracker fires). Run: `3 rows, 0 error rows`; reference pin verified against the 222-clip corpus (`dc2e139a`). Compared to the SAME items in evals/028 `dualforce_control_neutral_v3/c*/items.jsonl` (all 3 onesided):

| field | max&#124;Δ&#124; (3 items) | status |
|---|---|---|
| app_ref, app_target, margin | 0.000e+00 | agree ≤1e-6 |
| copy_max, near_copy, copy_gen_frame, copy_ref_frame | 0.000e+00 | agree ≤1e-6 |
| cam_zpr, cam_zpr_saturated, cam_corr, cam_dtw, cam_valid | 0.000e+00 | agree ≤1e-6 |
| obj_csls, obj_match, obj_r_gen | 0.000e+00 | agree ≤1e-6 |
| max_seam_z, prefix_seam_z, suffix_seam_z, d_argmax | 0.000e+00 | agree ≤1e-6 |
| app_ref_v3, app_saturated, cross, cross_high | 0.000e+00 | agree ≤1e-6 |
| core_frames, core_frac_strict, core_degenerate | 0.000e+00 | agree ≤1e-6 |
| scalar_depth, scalar_depart, scalar_arrive, scalar_core_frac | 0.000e+00 | agree ≤1e-6 |
| **prefix_dino** | **1.413e-05** | **differs (fresh CPU endpoint DINO)** |
| **prefix_lpips** | **7.859e-06** | **differs (fresh CPU endpoint LPIPS)** |

Every metric derived from the **cached** features is bit-identical (0.0) — the store returns the same arrays the hashed cache held. Only the two endpoint-fidelity values differ, and both are the *fresh recompute* the owner asked for (endpoint pair cache dropped): `prefix_lpips` is CPU-vs-GPU LPIPS, `prefix_dino` is CPU-vs-GPU DINO on the condition clip (the endpoint DINO is computed on cond frames cropped to the gen's geometry — it is **not** a stored namespace, so it recomputes every run regardless of the store). The brief listed `prefix_dino` in the ≤1e-6 set; measured, it recomputes fresh and lands at ~1.4e-5 on CPU — expected, not a store-I/O effect. The gen's own cached DINO is used bit-identically (proven by app_ref/copy_max/margin = 0.0). `prefix_lpips`/`prefix_dino` agreeing to 1e-6 requires GPU (P4).

Controls were **not** exercised on the real store (`--controls off`, because CPU DINO/CoTracker on synthesized frames is prohibitively slow, per the brief's escape hatch). The control persistence path is covered by `test_transition_eval_store.py` (tmp store + fake extractors, incl. an `fsck`-clean assertion).

## 4. Acceptance on real data (Part B)

`--dry-run` lookup over 5 gens of `013_dualforce_control/03_neutral_v3__dai` (+ their reference clips), store paths resolved:
```
[dry] gen …animalization_3…s42.mp4: clip_b32=extract  videoprism=extract  raft_mag=extract  cotracker3=hit
[dry] ref animalization_1.mp4:      clip_b32=extract  videoprism=extract  raft_mag=extract  cotracker3=hit
… (5 gens + refs, all identical shape)
```
Exactly as specified: **cotracker3 = hit** (migrated), **clip_b32 / videoprism / raft_mag = extract** (new lens namespaces, absent). No GPU touched. The 20-row numeric reproduction of `rows_v3` to 1e-6 needs the real GPU features and is **P4**.

## 5. CHANGELOG entry text (do NOT commit CHANGELOG.md — it is dirty)

```
- `HH:MM` **Feature store P3b — scorers read/write through the store (harness 4.0.1-draft.1 + competitor lenses).** transition_eval scoring path (score.py, pipeline.py + new store_io.py) reads/writes dino_cls@dinov2b-r256 / cotracker3@g20-m384-v2 / lpips_t@alex-r256 through the FeatureStore keyed BY VIDEO PATH; --cache-dir→--store-root; endpoint-LPIPS pair cache DROPPED (endpoints recompute fresh, gens always decoded, --lpips-cache removed); synthetic controls persist next to the gen as <ns>.ctl-<name>. reference_v4.npz swapped to the grid-v3 amendment build 459fd9a7 (the artifact that scored evals/028+030; PINS follow); VERSION→4.0.1-draft.1 (UNCERTIFIED; numeric identity certified by bar-8 in P4). pipeline.process_video_file stays dual-mode (legacy cache_dir) for certify/+workbench/. their_metrics/score_batch.py gains --store (default on): clip_b32/videoprism/raft_mag/cotracker3 lens arrays through the store via feature_extractors.REGISTRY; .track_cache retired; impl_sha d63935f4→8a808635 (rows_v3 keep d63935f4). Tests: +test_transition_eval_store.py (7), +their_metrics/test_store_lenses.py (6); required suite 49 pass, +test_certify_v3 → 75 pass. CPU acceptance vs evals/028: every store-derived metric bit-identical (0.0); prefix_dino/prefix_lpips differ ~1e-5 (fresh CPU endpoint recompute). No GPU.
```

## 6. Upstream changes wanted in the FROZEN files

`src/diffusion/feature_store.py` (I did not edit it — frozen; `store_io.py` works around each):
1. **A public control-variant write.** `store_io.HarnessStore._put_control` reimplements the atomic tmp+rename + sidecar-last write because `FeatureStore.put` keys strictly by `(video, ns)`. A `put(video, ns, arrays, meta, *, variant="ctl-lerp")` (or a `control_path(gen, ns, name)` + a shared `_atomic_write` helper) would remove the duplication and keep the sidecar schema in one place.
2. **`fsck`/`coverage` awareness of control files.** `<ns>.ctl-<name>.npz` lives inside the gen's item folder, so `fsck` currently treats `dino_cls@dinov2b-r256.ctl-lerp` as an extra pseudo-namespace in `ns_hosts` and `rebuild_manifest` lists it. It is benign (I copy the gen's video identity into the control sidecar so the stale-check passes, verified in tests), but a first-class notion of "derived/control features" (e.g. skip `*.ctl-*` in the namespace loops, or a `controls:` sub-block in the meta) would be cleaner.
3. **Optional sha-free / precomputed-sha put.** `FeatureStore.put` hashes the whole video (`sha256_file`) on every extraction miss. For the lens scorer filling thousands of gens on GPU that is real I/O; an option to accept a precomputed sha (e.g. from the already-written `videos/SHA256SUMS`) or defer it would help P4 throughput.
4. Minor: expose `_now_iso` (store_io reimplements it) and consider making `_feat_dir` public for consumers that need the item folder.

`scripts/store_features.py`: no change needed for P3b. If control files become first-class (item 2), `coverage`/`fsck`/`extract` there would report them.

## 7. Open questions

- **`judge_gemini.py` left unchanged.** The brief's site list includes it, but its `cache_dir` is a **Gemini response cache keyed by `item_id`** (`<item_id>.json` text responses), not a per-`(video, ns)` array — it does not map onto the FeatureStore model, and the judge (M4, advisory) is not on `score.py`'s numeric path. I left it byte-identical and flag it here. Confirm this is the intended treatment (vs. a separate response-cache location).
- **`prefix_dino` in the ≤1e-6 set.** Measured, it recomputes fresh (endpoint DINO on cond frames cropped to gen geometry — not a store namespace) and differs ~1.4e-5 on CPU. Confirm the brief intended it to be reported as a fresh-recompute delta (like `prefix_lpips`), which is what it is.
- **`certify/run_certification.py` not updated.** It is instrument code (not in the brief's file list) that builds the `score.py` command with the now-removed `--cache-dir`/`--lpips-cache` and calls `process_video_file(vp, cache_dir, …)`. The direct calls still work (dual-mode legacy branch). The **subprocess flags must be updated in P4** (the bar-8 driver P4 owns anyway): `--cache-dir`→`--store-root`, drop `--lpips-cache`, and the warm/cold-anchor semantics need a store-model redesign (a "cold" cache = a fresh store root; the warm rerun would now read `lpips_t` from the store unless forced). Left for P4; flagged.
- **Reference-pin consistency.** I updated `PINS['reference_v4_sha256']` to `459fd9a7` to match the swapped file, so `scripts/provenance_rebuild_parity_v4.py`'s file==pin check passes. `certifications/v4.0.0.md` still records the old `e6ea4011` (a historical record of the 4.0.0 cert — left as-is).
- **Controls on the real store.** Not exercised on CPU (too slow); only unit-tested. First real control files land at P4 GPU scoring; watch that they don't drift `fsck`'s `manifest_drift` on the paper variants (rebuild the manifest after).
