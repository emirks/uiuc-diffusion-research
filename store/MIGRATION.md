# Store contract v2 migration — 2026-08-13

One-time, owner-approved (2026-08-13) restructure of the `gens/` shelf from flat `NNN_<slug>`
entries to arm-first `NNN_<arm>/KK_<variant>__<machine>/` subentries. **Content bytes never moved
identity — only paths and aliases did.** Proposal + audits + scripts:
`misc/2026-08-13_store_restructure/` (PROPOSAL.md, migrate_gens.py, seed_prompts.py,
migration_report.json). The immutability rule survives into v2 unchanged; this table is frozen.

Every old gen id resolves forever via `gens/_legacy/<old>` symlinks. Eval metas (`arms_scored:`),
dossiers, and memories that name old ids stay valid through those shims and this table.

## Old → new (all 25 v1 entries)

| v1 entry | v2 subentry | harness_arm | mp4 | note |
|---|---|---|---|---|
| gens/001_ic_gen | gens/001_ic_gen/01_neutral__cc | ic_gen | 304 | |
| gens/010_ic_gen_effect | gens/001_ic_gen/02_effect__dai | ic_gen_effect | 304 | |
| gens/002_ctt_v2 | gens/002_ctt_v2/01_neutral__eps | ctt_v2 | 304 | |
| gens/005_ctt_v2_leaky | gens/002_ctt_v2/02_effect__dai | ctt_v2_leaky | 304 | |
| gens/024_ctt_v2_leaky_regen | gens/002_ctt_v2/03_effect__dai | ctt_v2_leaky_regen | 304 | materialized from outputs |
| gens/025_ctt_v2_plain_regen | gens/002_ctt_v2/04_neutral__dai | ctt_v2_plain_regen | 304 | materialized from outputs |
| gens/003_refvfx_A | gens/003_refvfx/01_effect__dai | refvfx_A | 304 | |
| gens/004_refvfx_B | gens/003_refvfx/02_neutral__dai | refvfx_B | 304 | |
| gens/006_base_prompt_ctt | gens/004_base_prompt/01_effect__dai | base_prompt_ctt | 304 | |
| gens/012_base_prompt_neutral | gens/004_base_prompt/02_neutral__dai | base_prompt_neutral | 304 | |
| gens/007_base_cond_ctt | gens/005_base_cond/01_effect__dai | base_cond_ctt | 304 | |
| gens/011_base_cond_neutral | gens/005_base_cond/02_neutral__dai | base_cond_neutral | 304 | |
| gens/008_bneck_frozen | gens/006_bneck_frozen/01_neutral__dai | bneck_frozen | 304 | materialized (was symlink) |
| gens/009_bneck_frozen_shufcode | gens/006_bneck_frozen/02_neutral_shufcode__dai | bneck_frozen_shufcode | 304 | materialized |
| gens/013_bneck_ctx_v2 | gens/007_bneck_ctx/01_neutral__dai | bneck_ctx_v2 | 304 | materialized |
| gens/014_bneck_ctx_v2_shufcode | gens/007_bneck_ctx/02_neutral_shufcode__dai | bneck_ctx_v2_shufcode | 304 | materialized |
| gens/015_surg1_wsd | gens/008_surg1/01_neutral__dai | surg1_wsd | 304 | `__ck4500` dir → `videos/` |
| gens/016_surg1_wsd_shufcode | gens/008_surg1/02_neutral_shufcode__dai | surg1_wsd_shufcode | 304 | same |
| gens/017_ctt_v2_pushA | gens/009_ctt_v3/01_neutral__eps | ctt_v2_pushA | 304 | stub → backfilled (media+grid+pins) |
| gens/019_ctt_v2_pushA_shufref | gens/009_ctt_v3/02_neutral_shufref__eps | ctt_v2_pushA_shufref | 39 | VOID, partial kept |
| gens/021_ctt_v2_pushA_effect | gens/009_ctt_v3/03_effect__dai | ctt_v2_pushA_effect | 304 | stub → backfilled |
| gens/023_ctt_v2_pushA_plain | gens/009_ctt_v3/04_neutral__dai | ctt_v2_pushA_plain | 304 | stub → backfilled |
| gens/018_ctt_v2_pushB | gens/010_ctt_v3_hs/01_neutral__eps | ctt_v2_pushB | 304 | stub → backfilled |
| gens/020_ctt_v2_pushB_shufref | gens/010_ctt_v3_hs/02_neutral_shufref__eps | ctt_v2_pushB_shufref | 39 | VOID, partial kept |
| gens/022_ctt_v2_pushB_effect | gens/010_ctt_v3_hs/03_effect__dai | ctt_v2_pushB_effect | 304 | stub → backfilled |

Verification at migration: per-entry mp4 counts matched pre-move counts (25/25); every `_legacy`
shim resolves; every subentry `grid.jsonl` prompt-sha matched its declared prompt family (25/25):
plain `0d708175fbfe`, effect `35930d7d7453`, V-neutral `f2ebeedf2187`, base-effect `d0460eaace93`,
refvfx `b88a248dfafc`/`11a50d24645a`. Each subentry keeps its v1 meta verbatim as `meta.v1.yaml`.

## Grid backfill sources (the 017–025 stubs had no grid.jsonl)

017/018/019/020 ← `misc/2026-08-11_ctt_v2_perf_push/build/registry_ctt_v2_push{A,B}{,_shufref}.jsonl`;
021/022/023/024/025 ← `eval_ladder/registry_ctt_v2_{pushA_effect,pushB_effect,pushA_plain,leaky_regen,plain_regen}.jsonl`.
(017's v1 meta claimed `eval_ladder/registry_ctt_v2.jsonl` "arm-stamped pushA" — wrong file; the
arm-stamped registry is the misc/ one. Corrected here; v1 meta preserved as-is.)

## Remediations executed with the migration

- **evals/001 retro symlinks removed** (9 forward links added 2026-08-12 pointing into evals/007+008
  — an immutability breach; verified consumed by nothing before removal). evals/001 is back to its
  as-registered five arms.
- **evals/007/_stray_shared_outroot/** (empty leftover of the pre-per-arm-outroot scoring pass) removed.
- **misc/2026-08-11_ctt_v2_perf_push/checkpoints_from_eps/ deleted (~15 GB)** after sha256-verify:
  armA 04500 `b6ed9789…` + 06000 `aa263cba…` and armB 06000 `76a0f66d…` all MATCH store/runs/008+009;
  armB 04500 + both `training_state_step_04500.pt` were unshipped intermediates (contract-deletable).
- **`_runner/` scratch** removed from the seven v1 entries that carried it (006/007/010/011/012/015/016).
- Outside locations that used to HOLD media now symlink INTO the store (13 links:
  `outputs/videos/{bneck_redesign,ctt_v2_push_effect,ladder2}/*__ck*`). Post-fix note: migrate_gens.py
  first wrote these against the transient `gens_v2` path; repointed to `store/gens/` in the same session.

## Errata corrected in v2 metas (v1 files untouched)

- 011/012 (`base_*_neutral`): v1 grid rows carry `prompt_variant: base_effect_no_token` — mislabel
  (no clause is rendered; the prompt is S1-only). v2 metas + `prompts/003_ctt152_vneutral` say `v_neutral`.
- 019/020: VOID (eps run_gen build ignored `code_source_reference` on the raw path — byte-identical
  to their matched twins; mechanism control unusable). Marked `void: true`.

## Same-day amendment: prompts shelf simplified to the 2 true sources

Owner observation (2026-08-13, verified 152/152 byte-exact): families C/D/E/F are deterministic
transforms of A/B — C = A−sksz, D = B−sksz, refvfx-neutral = A with sksz→token, refvfx-effect =
their template over (S1, clause). Entries `prompts/003-006` retired (numbers never reused); the
transforms live in `stamp_rows.py` (`--strip-token`, `--token`) and the derived shas in the two
family metas' `derived:` blocks. The six affected gen metas now cite `prompts/001|002 (<transform>)`.
