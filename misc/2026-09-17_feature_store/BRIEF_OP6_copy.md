# BRIEF Op-6 — copy rate vs the generation's OWN reference, from store features (`scripts/copy_metrics.py`)

Repo /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research, branch `feature-store`. Python `$LAB/envs-aarch64/ltx2/bin/python`
(never bare python3). CPU only; no GPU; no sbatch. Read first: `src/diffusion/transition_eval/m2_integrity.py` (`copy_score`, `mid_mask`,
`TAU_COPY = 0.858` — reuse VERBATIM, do not reimplement), `src/diffusion/transition_eval/pipeline.py` (`process_video`: how a bundle's
`core` mask is derived from DINO feats via `morph_profile` + `core_mask` with `n_prefix/n_suffix/n_endpoints`) and `s_structure.py`,
`src/diffusion/transition_eval/score.py` lines 150–210 (how `ref_core` and `gmid` are formed for the M2 call), `scripts/handoff_metrics.py`
(the n_pre/n_suf rules per grid type / external, and how it walks the population and writes a store eval — copy its shape),
`store/evals/038_handoff_gridv3__dai__2026-09-18/meta.yaml`, `data/processed/transitions_std121/corpus_manifest.json` (reference sidedness
→ `n_endpoints` for the reference's core mask, exactly as the harness does for reference bundles).

## Why
In evals/028/030 every pool clip is an eval item, so `copy_max`/`near_copy` were scored against POOL references, never against the
generation's own conditioning reference; `near_copy` is identical across all four paper arms on 100 % of 4,975 pool rows (coordinator check
2026-09-18). The table needs M2a against the OWN reference: "did the generation replay the reference's own shots?"

## Definition (FIXED)
For each generation g with own reference r (the `reference` field of the gen's grid.jsonl row; r's features are in the corpus store,
`data/processed/transitions_std121/<class>/features/<r>/dino_cls@dinov2b-r256.npz`):
- `gen_mid` = `mid_mask(T_g, n_pre, n_suf)` with the SAME n_pre/n_suf rules as handoff_metrics (HF 9/8|0 by sidedness; ED 1/0; externals 1/0).
- `ref_core` = the reference's core mask computed exactly as the harness computes a reference bundle (`morph_profile` + `core_mask` on r's
  DINO feats, with r's sidedness → n_endpoints, n_prefix=9, n_suffix=8). Use the harness functions; if `process_video` needs frames only
  for tracks, call it with tracker=None / feats already loaded (there is a store-backed path from P3b — `pipeline.process_video_store` or
  equivalent — prefer it).
- `copy_score(gen_feats, gen_mid, ref_feats, ref_core, TAU_COPY)` → `copy_max, near_copy, copy_gen_frame, copy_ref_frame`.
- Also emit `ref_core_frac` (fraction of reference frames in core) and `n_mid` (gen mid frames), and `missing` when a feature is absent.
Caveat to record in meta.yaml: `copy_max` is a max over mid frames, so longer generations (121 f) have more chances than the externals'
49 f / 33 f — the externals' copy rate is, if anything, under-estimated relative to ours; disclosed, not corrected.

## Output
Store eval `store/evals/<next free NNN>_copy_gridv3__dai__2026-09-18/<harness_arm>/rows.jsonl` (`item_id, seed, arm, n_pre, n_suf,
n_mid, ref_core_frac, copy_max, near_copy, copy_gen_frame, copy_ref_frame, missing`) for all 19 variants of
`misc/2026-09-17_feature_store/population_gridv3.json`, + `meta.yaml` (definition verbatim, why, caveat, tau, instrument sha, arms_scored)
+ one appended `store/INDEX.md` line. Idempotent.

## Tests + report
`tests/test_copy_metrics.py` (synthetic: a gen whose mid frames equal the reference's non-core frames → copy_max≈1, near_copy True; disjoint
features → low; mid_mask sizes per grid type). Run on the full population now (all DINO features exist). Sanity in the report: copy rate per
arm on the 366-row shared set, and confirm it is NOT identical across arms. Report `misc/2026-09-17_feature_store/OP6_REPORT.md` (`git add -f`).
Git: emirks <emirks88@gmail.com>, pathspec staging (script, test, new eval meta.yaml, INDEX line, report); push after every commit; don't touch
the other operators' scripts, the ~66 dirty files, CHANGELOG.md, `.claude/worktrees/*`. Reply with a concise summary + the per-arm copy rates.
