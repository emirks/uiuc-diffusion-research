# Feature store — colocated per-video features (store contract clause 10)

**Since 2026-09-17/18.** Every metric feature (DINO CLS, CoTracker tracks, temporal LPIPS, CLIP-B/32, VideoPrism, RAFT magnitudes,
CLIP-L/14) lives next to the video it describes; scorers read and write through `diffusion.feature_store.FeatureStore`. There is no
`--cache-dir`, no hash-named cache, no per-campaign cache. Registry of namespaces + pins: `store/FEATURES.md`; coverage matrix:
`store/FEATURES_COVERAGE.md`; design + migration record: `misc/2026-09-17_feature_store/PROPOSAL.md` and `P3{a,b,c}_REPORT.md`.

## The rule

- Store gens: `<variant>/videos/<item>__s42.mp4` ↔ `<variant>/features/<item>__s42/<ns>.npz` + `<ns>.json` (sidecar meta).
- Clips not under a `videos/` folder (corpus `<class>/<clip>.mp4`, endpoint clips `eval_ladder/conds/*_start9.mp4`):
  `<folder>/features/<clip>/<ns>.npz`.
- Synthetic controls (lerp / static-hold, synthesized at scoring): next to the GEN as `<ns>.ctl-<name>.npz`.
- `features/manifest.jsonl` = index rebuilt from the files (files are the truth); `videos/SHA256SUMS` (tracked) pins video identity;
  each sidecar records the video sha256, extracting `host` (null + `migrated_by_host` for migrated files), `code_sha`, `origin`.
- A namespace tag (`name@tag`) is frozen once written; a pin change is a new tag.

## Tooling

`scripts/store_features.py coverage | fsck | sha256sums | migrate | extract` — all take `--population FILE`
(e.g. `misc/2026-09-17_feature_store/population_gridv3.json`) or `--gens GLOB --corpus --conds`. `--help` starts in <1 s (the CLI loads
`feature_store.py` by file path so the torch-heavy `diffusion/__init__` is not imported). Harness: `python -m diffusion.transition_eval.score
--store-root REPO_ROOT …`; lenses: `their_metrics/score_batch.py --store`; certification: `certify/run_certification.py` (warm rerun on the
shared store, cold anchors staged into `out/cold_store`).

## Facts worth remembering

- Legacy caches were keyed `sha1(resolved path | mtime_ns | size | model | short_side)`: unlistable and path-fragile. The paper's
  evals/028 used `misc/refvfx_baseline/probe/cache` (244k files); the externals' features (evals/030) were under paths that no longer
  exist and could not be recovered → re-extracted in P4. Legacy caches are kept (owner decision), untouched.
- Migration = hard links (same filesystem, device 1808112826): zero extra bytes. Population `gridv3` (19 variants / 7,242 gens,
  677-clip corpus, 354 endpoint clips): dino_cls/cotracker3 for every paper-arm gen + corpus clip, lpips_t for every paper-arm gen.
- The harness on branch `feature-store` (4.0.1-draft.1) reproduces evals/028 per-item numbers bit-identically on every store-derived
  metric; `prefix_dino`/`prefix_lpips` are recomputed fresh (endpoint pair cache dropped) and differ ~1e-5 CPU-vs-GPU.
- `reference_v4.npz` sha 459fd9a7 (the build that scored evals/028+030) was uncommitted in the eval-v4-cert worktree; it is now
  committed with 4.0.1-draft.1 and backed up in `misc/2026-09-17_feature_store/instrument/`.
- Login-node `python3` is 3.6 — always `$LAB/envs-aarch64/ltx2/bin/python`.
