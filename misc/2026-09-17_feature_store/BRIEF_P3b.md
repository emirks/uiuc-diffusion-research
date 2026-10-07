# BRIEF P3b — scorers read/write through the feature store (implementer: opus48)

Branch `feature-store` @ f0569bb or later (main checkout; `eval/v4-metrics` is merged in, so
`src/diffusion/transition_eval` here IS the code that scored evals/028 + evals/030). LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa;
python `$LAB/envs-aarch64/ltx2/bin/python` (never bare `python3` — it is 3.6 on the login node). No GPU, no sbatch/srun.

**Read first:** `CLAUDE.md` · `misc/2026-09-17_feature_store/PROPOSAL.md` §2, §6 · `store/FEATURES.md` (frozen registry) ·
`src/diffusion/feature_store.py` (public API: `FeatureStore.path/sidecar/has/get/read_meta/put/coverage/fsck/rebuild_manifest/iter_videos`;
`NAMESPACES`, `NS_ARRAYS`) · `src/diffusion/feature_extractors.py` (`REGISTRY`) · `src/diffusion/transition_eval/SPEC.md` §9, §10 and the
spec changelog · `versioning.py` · the harness files below · `misc/2026-09-07_eval_grid_v2/score_v3.sbatch` (the real scoring entrypoint) ·
`misc/2026-08-13_baseline_metric_table/their_metrics/{score_batch.py,SCORERS.json,clip_sim.py,videoprism_sim.py,dynamic_degree.py,motion_smoothness.py}`.

## FROZEN during your run (a migration chain is using them): `src/diffusion/feature_store.py`, `scripts/store_features.py`.
If you need an API addition, put it in a new small module (e.g. `src/diffusion/transition_eval/store_io.py`) and list the
desired upstream change in your report. Also do not touch: the ~66 pre-existing dirty files, `.claude/worktrees/*`, `CHANGELOG.md`.

## Part A — harness (`src/diffusion/transition_eval`) — I/O-only, numerics untouched
1. Every `cache_dir` / `lpips_cache_dir` path (score.py ~22 sites incl. `--cache-dir`, `lpips_warm`, `_ref_bundle_cache`, `_cached_endpoint`;
   pipeline.py; features.py `video_features`/`array_features`/`feature_cache_path`; motion.py `Tracker.cached_track`/`track_cache_path`;
   endpoints.py `cached_temporal_lpips`/`lpips_cache_path`; judge_gemini.py) is replaced by a `FeatureStore` (root = repo root; CLI
   `--store-root`, default REPO_ROOT). Persistence is BY VIDEO PATH (`store.get/put(video, ns, …)`) for the three namespaces
   `dino_cls@dinov2b-r256`, `cotracker3@g20-m384-v2`, `lpips_t@alex-r256`; the in-memory bundle `key` may stay a string identity.
   Real videos = gens, corpus references, pool references, condition clips (`eval_ladder/conds/*_start9|end9.mp4`).
2. Synthetic controls (lerp / static-hold frames synthesized at scoring; no file on disk): persist them next to the GEN's features as
   `<ns>.ctl-<name>.npz` + `.json` (e.g. `dino_cls@dinov2b-r256.ctl-lerp.npz`, `origin: "control:lerp"`), via your helper module using the
   same path/sidecar/atomic conventions. Document the form in `store/FEATURES.md` (one short paragraph under File format).
3. Endpoint-LPIPS pair cache (`…:endp:…` scalars) is DROPPED (owner decision): `_cached_endpoint` computes fresh every run. Simplify
   `lpips_warm`/`need_frames` accordingly (gens are always decoded; the decode skip may remain for corpus references when their three
   namespaces are present). Remove `--lpips-cache`.
4. `reference_v4.npz`: replace the package copy with `misc/2026-09-17_feature_store/instrument/reference_v4_459fd9a7.npz`
   (sha256 must equal 459fd9a71bb50ef81dcbd1d881aecf6a6c70e18855899b37c0a64b7167e606a8 — the build that scored evals/028/030; the
   committed e6ea4011… is the 4.0.0-certified build). Commit it WITH the version bump, and say so in the spec changelog.
5. `VERSION` → `4.0.1-draft.1`. SPEC.md: §9 implementation map (feature store replaces hash caches; where controls live), spec changelog
   entry stating: I/O-only; endpoint-LPIPS pair cache dropped; reference artifact = grid-v3 amendment build 459fd9a7 (previously
   uncommitted in the eval-v4-cert worktree); numeric identity to be certified by the bar-8 reproduction in P4; branch = `feature-store`
   (campaign branch, owner-approved deviation from §10 step 1). Results provenance gains `feature_store: {root, namespaces, code_sha}`.
6. Tests: adapt `tests/test_transition_eval*.py`, `tests/test_versioning.py` to the new API (tmp FeatureStore roots). Green with
   `PYTHONPATH=$PWD/src $LAB/envs-aarch64/ltx2/bin/python -m pytest -q tests/test_transition_eval*.py tests/test_versioning.py`.
7. **CPU acceptance on real data.** Build a 3-item manifest from `store/gens/013_dualforce_control/03_neutral_v3__dai` (its three
   namespaces are already in the store) using the same `--corpus/--reference-corpus/--manifest` conventions as `score_v3.sbatch`, and
   score with `--device cpu`. Compare each item's scalars with the SAME items in
   `store/evals/028_grid_v3_paper_arms__dai__2026-09-07/dualforce_control_neutral_v3/c*/items.jsonl`: metrics derived from the cached
   features (`app_ref`, `app_target`, `margin`, `copy_max`, `near_copy`, `cam_zpr`, `cam_corr`, `obj_csls`, `obj_match`, `max_seam_z`,
   `prefix_seam_z`, `prefix_dino`, `core_*`, `scalar_*`) must agree to 1e-6; `prefix_lpips` and any control-derived value may differ
   (CPU LPIPS / CPU DINO on synthesized frames) — report the deltas. If CPU DINO for controls is prohibitively slow, run with
   `--controls off` and say so. Tabulate the comparison in the report.

## Part B — competitor lenses (`misc/2026-08-13_baseline_metric_table/their_metrics/`)
1. `score_batch.py --store` (default on): CLIP-B/32 per-frame embeddings (`clip_b32@r256`), VideoPrism (`videoprism@f16r288`), RAFT
   magnitudes (`raft_mag@r256`) and tracks (`cotracker3@g20-m384-v2`) are read from / written to the store for every VIDEO (gen and
   reference). Use the `diffusion.feature_extractors.REGISTRY` classes as the single source of the preprocessing so arrays are identical by
   construction; metric arithmetic (`clip_sim`, `motion_smoothness`, `videoprism_sim`, `det_motion_fidelity`, `dynamic_degree`) untouched.
   The input FRAME (png) is not a video: embed fresh. `.track_cache` retired.
2. `impl_sha` changes (I/O files are in its set): update `SCORERS.json` (`impl_sha`, note that the paper's `rows_v3` keep `d63935f4`).
3. Acceptance here = unit tests with fake extractors (CPU) + a `--dry-run`/lookup check that store paths resolve for 5 gens of
   `013_dualforce_control/03_neutral_v3__dai` (tracks present → hit; clip/vp/raft absent → would extract). The 20-row numeric
   reproduction to 1e-6 needs GPU features and is P4 — say so.

## Git / report
Commit by pathspec as `emirks <emirks88@gmail.com>` (`export GIT_AUTHOR_NAME=emirks GIT_AUTHOR_EMAIL=emirks88@gmail.com GIT_COMMITTER_NAME=emirks GIT_COMMITTER_EMAIL=emirks88@gmail.com`),
`git push` after every commit, at least two commits: (1) `eval(4.0.1-draft.1): …` harness + reference artifact + SPEC + VERSION + tests;
(2) `eval(lenses): …` score_batch store integration. Report `misc/2026-09-17_feature_store/P3b_REPORT.md` (`git add -f`): what changed
per file, pytest output, the acceptance comparison table, the CHANGELOG entry text (do NOT commit CHANGELOG.md — it is dirty), the upstream
changes you want in the frozen files, open questions. Reply with a concise summary.
