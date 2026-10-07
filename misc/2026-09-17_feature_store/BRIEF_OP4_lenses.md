# BRIEF Op-4 — competitor-lens pass + aesthetic + gate check (code + fixtures now; the coordinator submits the GPU job)

Repo /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research, branch `feature-store`. Python `$LAB/envs-aarch64/ltx2/bin/python`
(never bare python3). **You do not run GPU jobs or sbatch** — write the scripts, the sbatch, unit tests on fixtures, and CPU dry-runs; the
coordinator submits. Read first: `misc/2026-08-13_baseline_metric_table/their_metrics/{score_batch.py,SCORERS.json,README*}` (the `--store`
path added in P3b; `parse_item` derives reference + input frame from the gen filename: reference under `data/processed/transitions_std121/<class>/`,
input frame `misc/refvfx_baseline/frames/<endpoint>__first.png`), `their_metrics/rows_v3/vap_author_native/*.json` (frozen paper rows,
impl_sha d63935f4), `store/FEATURES.md`, `src/diffusion/feature_store.py`, `src/diffusion/feature_extractors.py` (`clip_l14@r224` =
L2-normalized CLIP ViT-L/14 image features `[T,768]`), `store/README.md` (eval entry contract), `store/evals/038_handoff_gridv3__dai__2026-09-18/meta.yaml`
(an eval entry written tonight — copy its shape), `misc/2026-09-17_feature_store/population_gridv3.json`, and Op-5's expected lens schema in
`misc/2026-09-17_feature_store/BRIEF_OP5_tables.md` (input 3).

## Deliverables
1. `scripts/lens_pass_gridv3.py` — drives `their_metrics/score_batch.py --store` over the population's gens (all 19 variants; the
   store already holds `cotracker3` for every gen and, when the coordinator runs it, `clip_b32/videoprism/raft_mag`), sharded (`--shard i/n`),
   resumable (skip existing per-gen JSON), writing per-gen JSON under `store/evals/<next free NNN>_lenses_gridv3__dai__2026-09-18/<harness_arm>/rows/`;
   plus `--collect` that merges them into `<harness_arm>/rows.jsonl` (one row per gen: `item_id, seed, arm, motion_smoothness,
   videoprism_sim_ref, videoprism_sim_input, clip_sim_ref, clip_sim_input, det_motion_fidelity, dynamic_degree_mean_mag, impl_sha, warnings`),
   writes `meta.yaml` (instrument = score_batch impl_sha, `--store`, host, arms_scored) and appends one `store/INDEX.md` line. `item_id`/`seed`
   must join Op-2's per_gen rows (`(item_id, seed)` = grid item_id + seed, NOT score_batch's own item_id string — reconcile explicitly).
   Rows whose input frame png is missing (seen/unseen-tier endpoints may lack `frames/<endpoint>__first.png`) must still score every other
   lens (warning recorded), never crash.
   + `misc/2026-09-17_feature_store/jobs/lens_pass.sbatch` (copy the flags of `jobs/extract.sbatch`: bgjg-dtai-gh, ghx4, 1 GPU, 16 cpus, 110g,
   taiga, requeue, 8 s stagger, no srun; env NSHARDS; `--time=02:00:00`).
2. `scripts/aesthetic_from_store.py` — LAION aesthetic predictor: MLP 768→1024→128→64→16→1 (`$LAB/cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth`,
   keys `layers.{0,2,4,6,7}.*`; input = the L2-normalized CLIP ViT-L/14 image embedding — exactly our `clip_l14@r224` feats) applied per frame,
   `aesthetic` = mean over frames. CPU torch. Writes/merges an `aesthetic` column into the same lenses eval `rows.jsonl` (idempotent), NaN + `missing`
   when the namespace is absent. Include the standard LAION head loading code path (verify the state-dict shapes at load).
3. `scripts/lens_gate_check.py` — the port check: for `their_metrics/rows_v3/vap_author_native/*.json`, take every row whose gen AND reference
   have `clip_b32/videoprism/raft_mag/cotracker3` in the store (the gate job fills VAP + its 58 refs), recompute the lenses through
   `score_batch --store` for those gens, and report per-lens max|Δ| and the count over tolerance. Tolerances (coordinator-set): embedding lenses
   (`clip_sim_ref/input`, `motion_smoothness`, `videoprism_sim_ref/input`, `dynamic_degree_mean_mag`) **≤ 1e-4** (GPU float nondeterminism across
   nodes); `det_motion_fidelity` **≤ 1e-2** and reported separately (CoTracker re-tracking is not bit-reproducible; the store tracks for VAP were
   re-extracted tonight, the paper's came from `.track_cache`). Exit 1 if any embedding lens exceeds 1e-3 (a real port error). Needs GPU only if
   features are missing — with features present it must run on CPU (`--device cpu`; the input-frame CLIP embed on CPU is fine).
4. Tests: `tests/test_lens_pass_gridv3.py` (fixture: fake rows → collect/merge/join keys; aesthetic head on a random feature array →
   finite; gate-check comparator logic). Report `misc/2026-09-17_feature_store/OP4_REPORT.md` (`git add -f`) with the exact commands the
   coordinator runs: (a) gate check, (b) `sbatch` for the lens pass, (c) collect + aesthetic + INDEX.

## Rules
No GPU/sbatch/srun runs. Do not edit `feature_store.py`, `store_features.py`, `handoff_metrics.py`, `store_per_gen.py`, `build_metric_tables.py`,
the ~66 pre-existing dirty files, CHANGELOG.md, `.claude/worktrees/*`. Minimal edits to `their_metrics/score_batch.py` are allowed only if a
flag is missing (say so). Git: emirks <emirks88@gmail.com>, pathspec staging, push after every commit. Reply with a concise summary + the commands.
