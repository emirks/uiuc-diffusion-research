# BRIEF Op-2 — per-generation score rows from the existing v4 evals (`scripts/store_per_gen.py`)

Repo /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research, branch `feature-store`. Python `$LAB/envs-aarch64/ltx2/bin/python`
(never bare python3). CPU only, no GPU, no sbatch. Read first: `store/README.md` (contract), `store/evals/028_grid_v3_paper_arms__dai__2026-09-07/{meta.yaml,summary.json}`,
one `…/<arm>/c0/items.jsonl` row, `store/evals/030_external_zs_authornative__dai__2026-09-12/`, `misc/2026-09-07_eval_grid_v2/eval/REPORT.md`
and the scripts next to it that produced `summary.json`, `misc/2026-09-02_temporal_dynamics_metric/score_v3_zs.py` (the "capped pooled-%"
apples-to-apples slice: 183 one-sided zero-shot triples × 2 seeds), `eval_ladder/ceilings_v3.json`, the gens' `grid.jsonl`
(`store/gens/<arm>/<variant>/grid.jsonl`: `cell, content, endpoint, reference, sided, ref_novelty, pct_type, prompt`).

## Goal
One row per GENERATION (not per pool row) with every scalar the paper tables need, for the 19 gridv3 variants
(`misc/2026-09-17_feature_store/population_gridv3.json`), written as `store/evals/<eval>/<arm>/per_gen.jsonl` next to the
`items.jsonl` shards it is derived from (evals/028 for the 16 paper-arm variants, evals/030 for the 3 externals).

Row schema (exact keys): `item_id, seed, arm, variant_dir, gen_video (repo-relative), tier ∈ {seen, unseen, zero_shot}, sided ∈ {one, two},
cell, content, pct_type, endpoint, reference, ref_class, n_frames, transport_pct, transport_raw (app_ref), transport_ceiling,
transport_capped (bool), copy_max, near_copy, max_seam_z, prefix_seam_z, suffix_seam_z, prefix_dino, prefix_lpips, cam_zpr, obj_csls,
core_degenerate, cross_high, ref_in_v4_population`. `tier` comes from `ref_novelty` (ED grids are all zero_shot); externals: tier zero_shot,
sided one.

## The one thing that must be exactly right: `transport_pct`
It is the paper's headline number ("%_same", capped pooled-%: the generation's mean similarity to the real videos of its operator, divided by
the real videos' mean similarity to one another, as a percentage, capped at 100). **Find the canonical implementation** (the code that wrote
`summary.json` for evals/028, and `score_v3_zs.py`) and reproduce it per generation from the pool rows in `items.jsonl` + the ceilings.
**Verification (mandatory, in the report):** aggregating your `per_gen.jsonl` by arm × cell must reproduce every number in
`store/evals/028_…/summary.json` to the printed precision, and the 183-triple slice must reproduce the CHANGELOG 2026-09-17 23:23 numbers
(S3 column: VAP 81.1, VFXMaster 85.7, refVFX 62.2; ic_gen neutral 61.6, dualforce_control 87.6, dcg_w6 92.6). If anything does not
reproduce, do not tune — report the discrepancy with your best diagnosis. Note for the coordinator: the paper draft's refVFX zero-shot
value 75.5 matches the "Look_u" column, not S3 — flag which column your rows carry (they must carry S3/app_ref).

## Also
- A `--check` mode that prints the reproduction table (per_gen aggregate vs summary.json) and exits 1 on any mismatch.
- Idempotent; per-arm; fast (pure python/numpy over jsonl).
- Tests: `tests/test_store_per_gen.py` (tiny synthetic items.jsonl + ceilings → known pct).
- Git: identity emirks <emirks88@gmail.com>, stage by pathspec (`scripts/store_per_gen.py`, the test, `git add -f` your report
  `misc/2026-09-17_feature_store/OP2_REPORT.md`; per_gen.jsonl files are store artifacts — NOT committed), push after every commit.
  Do not touch the ~66 pre-existing dirty files, CHANGELOG.md, `.claude/worktrees/*`. Reply with a concise summary + the reproduction table.
