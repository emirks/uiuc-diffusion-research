# BRIEF P3a — feature-store library + CLI + tests (implementer: opus48)

**Read first, in this order:** `CLAUDE.md` (repo rules) · `misc/2026-09-17_feature_store/PROPOSAL.md` §2–§5, §8 · `store/FEATURES.md`
(the frozen registry — authority on paths, namespaces, file format) · `store/README.md` clause 10 · `scripts/store_fsck.py` (style) ·
`src/diffusion/transition_eval/features.py` (`file_key`, `feature_cache_path`, `video_features`) · `motion.py` (`track_cache_path`,
`Tracker`, `CACHE_TAG`) · `endpoints.py` (`LPIPS_CACHE_TAG`, `lpips_cache_path`, `temporal_lpips`, `LpipsScorer`) · `score.py` lines 80–130.

## Deliverables (this round; nothing else is in scope)
1. `src/diffusion/feature_store.py` — library, no CLI, numpy + stdlib only at import (backbones imported lazily inside extractors).
   `FeatureStore(repo_root)` with `path(video, ns)`, `sidecar(video, ns)`, `has`, `get` (arrays dict), `put(video, ns, arrays, meta, *, link_from=None)`
   (atomic: `.tmp-<pid>` → rename; `.json` written last; `link_from` = hard-link a legacy file instead of writing arrays),
   `coverage(video_dir)` → per-ns `{have, of, hosts}`, `fsck(video_dir, rehash=False)` → report (orphan npz without json, json without npz,
   stale = sha/size/mtime mismatch when rehash or stat differs, mixed hosts, manifest drift), `rebuild_manifest(video_dir)`,
   `write_meta_block(variant_dir)` (refresh the `features:` block in a gen `meta.yaml` without touching other keys — text-level edit,
   preserve comments), `iter_videos(video_dir)` (`*.mp4`, sorted). Video paths in sidecars are **relative to repo root**.
   Legacy-key helpers (exact recipes, must match byte-for-byte):
   - `legacy_key(video) = sha1("|".join([str(video.resolve()), str(st_mtime_ns), str(st_size), "facebook/dinov2-base", "256"])).hexdigest()[:16]`
   - `dino_cls@dinov2b-r256` ← `dino_arr_{sha1(key)[:16]}.npz` (arrays `feats`[,`src`]); ALSO the older form `dino_{key}.npz` (arrays `feats`,`fps`,`src`); prefer `dino_arr_`.
   - `cotracker3@g20-m384-v2` ← `tracks_{sha1(key + ":tracks:v2")[:16]}.npz` (`tracks`,`vis`)
   - `lpips_t@alex-r256` ← `lpips_{sha1(key + ":tlpips:alex-v1")[:16]}.npz` (`d`)
   - endpoint-pair lpips files (`...:endp:...`) are NOT migrated (dropped by owner decision).
   Legacy dirs, search order: `misc/refvfx_baseline/probe/cache`, `outputs/eval/cache`, then any `--from` extras.
2. `scripts/store_features.py` — CLI: `coverage [--gens GLOB...] [--corpus] [--conds] [--md]` (matrix; `--md` writes `store/FEATURES_COVERAGE.md`),
   `fsck [...] [--rehash] [--rebuild-manifest]` (exit 1 on stale/orphan/mixed-host), `sha256sums [...]` (writes `SHA256SUMS` next to the videos —
   `videos/SHA256SUMS` for gens, `<folder>/SHA256SUMS` for corpus/conds; skip unchanged by size+mtime via an ignored `.SHA256SUMS.stat.json`),
   `migrate [...] --from DIR... [--dry-run]` (legacy lookup → validate `np.load` → hard link → sidecar → manifest → meta block; per-ns hit/miss table),
   `extract NS [...] [--shard i/n] [--dry-run] [--device]` (fill misses only; extractor registry keyed by namespace; DINO / tracks / lpips_t reuse
   `transition_eval` code verbatim: `DinoExtractor.extract`, `video_io.load_frames(short_side=256)`, `Tracker` at grid 20 / max_side 384 with the
   SAME query protocol as `Tracker.track`/`cached_track`, `temporal_lpips` via `LpipsScorer("alex")`; the four lens namespaces (`clip_b32@r256`,
   `videoprism@f16r288`, `raft_mag@r256`, `clip_l14@r224`) go in `src/diffusion/feature_extractors.py` ported faithfully from
   `misc/2026-08-13_baseline_metric_table/their_metrics/{clip_sim,videoprism_sim,dynamic_degree}.py` per `SCORERS.json` — if time is short,
   register them with a clear `NotImplementedError("P3a-2")` and SAY SO in your report). `--corpus` = `data/processed/transitions_std121/*/*.mp4`;
   `--conds` = `eval_ladder/conds/*.mp4`; `--gens` globs are variant dirs (`store/gens/<arm>/<variant>`), videos under `videos/`.
   Record `host` (`socket.gethostname()`), `code_sha` (git HEAD short sha of the repo; `+dirty` if the two feature-store files are modified), `created` ISO-8601 UTC.
3. `tests/test_feature_store.py` — CPU only, `tmp_path`, fake extractor injected: path rule (both cases), atomic put + sidecar-last, hard-link migrate
   (same `st_ino`), legacy-key recipe against a hand-computed sha1, manifest rebuild equals sidecars, fsck detects orphan/stale/mixed hosts,
   coverage counts, `write_meta_block` preserves unrelated yaml text, sha256sums skip logic. `pytest tests/test_feature_store.py` must pass.

## Acceptance on real data (allowed, CPU only, reversible)
- `python scripts/store_features.py migrate --dry-run --gens store/gens/013_dualforce_control/03_neutral_v3__dai` → expect ≈564/564 hits on
  dino, tracks, lpips_t (probe on 60 random videos hit 60/60 each). Then run it for real on THAT ONE VARIANT and on `--corpus` `--dry-run`.
  Report the tables. Do not migrate other variants (the operator runs P2).
- `python scripts/store_features.py coverage --gens store/gens/013_dualforce_control/03_neutral_v3__dai` shows the result; `fsck` exits 0.

## Hard rules
- NO GPU, no `sbatch`/`srun`, no extraction runs. Verify extractors by code reading + the fake-extractor tests only.
- Do NOT edit `src/diffusion/transition_eval/*` or `misc/.../their_metrics/*` (that is P3b). Do not touch the 66 pre-existing dirty files.
- Write only: the three deliverables, `src/diffusion/feature_extractors.py`, `CHANGELOG.md` (one entry at the end), `misc/2026-09-17_feature_store/P3a_REPORT.md`.
  If `store/FEATURES.md` must change (a format detail you find wrong), change it minimally and list the change in the report.
- Git: branch `feature-store` (exists). `export GIT_AUTHOR_NAME=emirks GIT_AUTHOR_EMAIL=emirks88@gmail.com GIT_COMMITTER_NAME=emirks GIT_COMMITTER_EMAIL=emirks88@gmail.com`.
  Stage by pathspec only; messages `infra(feature-store): …`; `git push` after every commit. End with `git status` clean for your paths.
- Python: `$LAB/envs-aarch64/ltx2/bin/python` (numpy present); run pytest with it. `LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa`.

## Report (`P3a_REPORT.md`): what was built, test output, the dry-run + real-migrate tables, deviations from the brief, open questions.
