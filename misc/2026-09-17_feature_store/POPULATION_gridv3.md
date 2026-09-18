# Population `gridv3` — what gets featurized NOW (owner scope decision 2026-09-17)

Machine-readable twin: `population_gridv3.json` (the CLI's `--population misc/2026-09-17_feature_store/population_gridv3.json`
resolves to exactly these files). Everything not listed here is **deferred**: the other ~104 store variants, the legacy caches
(kept, untouched), the campaign caches, and the P6 cleanup.

| part | what | count |
|---|---|---|
| gen variants | the 4 paper arms × {neutral, effect} × {HF 121f, ED 81f} on grid v3, + the 3 author-native externals | 19 variants, 7242 mp4 |
| corpus | the 677-clip pool corpus `corpus_manifest.json` (references + same-class pools; NOT the whole `transitions_std121/` dir, which holds 4,536 mp4) | 677 clips |
| endpoint clips | `eval_ladder/conds/<endpoint>_start9.mp4` for all 204 endpoints the grids reference, plus `_end9.mp4` for the 150 that have one (54 endpoints — 51 EffectData-tier + 3 DAVIS — are one-sided-only and have no end clip). ED-tier rows condition on frame 0 of `start9` (`GEN_PREFIX_FRAMES=1`): identity-A is scored against that frame, motion-A is undefined | 354 clips |

Missing on disk: none. (Endpoint sidedness in the grids: 115 one-sided endpoints that still have both clips, 32 two-sided, 51 ED one-sided, 6 DAVIS.)

## Gen variants
- `store/gens/001_ic_gen/03_neutral_v3__dai`
- `store/gens/001_ic_gen/04_neutral_v3ed81__dai`
- `store/gens/001_ic_gen/05_effect_v3__dai`
- `store/gens/001_ic_gen/06_effect_v3ed81__dai`
- `store/gens/003_refvfx/03_author_native__dai`
- `store/gens/005_base_cond/04_neutral_v3__dai`
- `store/gens/005_base_cond/05_neutral_v3ed81__dai`
- `store/gens/005_base_cond/06_effect_v3__dai`
- `store/gens/005_base_cond/07_effect_v3ed81__dai`
- `store/gens/011_vap/05_author_native__dai`
- `store/gens/012_vfxmaster/05_author_native__dai`
- `store/gens/013_dualforce_control/03_neutral_v3__dai`
- `store/gens/013_dualforce_control/04_neutral_v3ed81__dai`
- `store/gens/013_dualforce_control/05_effect_v3__dai`
- `store/gens/013_dualforce_control/06_effect_v3ed81__dai`
- `store/gens/032_dualforce_dcg_w6/03_neutral_v3__dai`
- `store/gens/032_dualforce_dcg_w6/04_neutral_v3ed81__dai`
- `store/gens/032_dualforce_dcg_w6/05_effect_v3__dai`
- `store/gens/032_dualforce_dcg_w6/06_effect_v3ed81__dai`
