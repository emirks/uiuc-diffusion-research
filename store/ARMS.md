# Canonical arm registry

The naming authority for contract v2. An **arm** = one model/adapter lineage; a **variant** = the
prompt setting it was generated under. **`neutral` = that arm's leak-free prompt** (plain `sksz` for
adapter arms, S1-only V-neutral for base arms, the fixed token for refVFX); **`effect` = neutral +
the shared effect clause** (clauses: `misc/refvfx_baseline/reference_effects.json`, one per reference,
append-only). Controls suffix the variant (`_shufcode`, `_shufref`); machine suffixes the subentry
(`__cc` campus-cluster · `__eps` offsite fal-h100 · `__dai` DeltaAI).

**Canonical vs frozen:** `harness_arm` is the stamp inside already-scored `items.jsonl`/item_ids —
it can never be renamed without re-scoring. Joins (viewer, analysis) use `harness_arm`; directories,
the ledger, and humans use canonical names. New arms stamp `harness_arm = <arm>_<variant>` so the
two converge. Prompts have exactly TWO sources in `prompts/` (001 neutral, 002 effect — sha-pinned); base-arm
and external variants are `stamp_rows.py` transforms (`--strip-token`, `--token`), each with a
recorded derived sha in the family meta. A gen pins `prompt_family` + `prompt_sha` — never
hand-write a registry.

| arm | status | run | neutral family | effect family | harness_arm aliases (frozen) |
|---|---|---|---|---|---|
| `ic_gen` | active | runs/001 | prompts/001 | prompts/002 | ic_gen · ic_gen_effect |
| `ctt_v2` | superseded by ctt_v3 | runs/002 | prompts/001 | prompts/002 | ctt_v2 · ctt_v2_leaky · ctt_v2_leaky_regen · ctt_v2_plain_regen |
| `refvfx` | external baseline | runs/003 | prompts/001 ·token-swap | prompts/002 ·template | refvfx_A (=effect) · refvfx_B (=neutral) |
| `base_prompt` | floor (no adapter) | — | prompts/001 ·strip | prompts/002 ·strip | base_prompt_ctt (=effect) · base_prompt_neutral |
| `base_cond` | floor (no adapter) | — | prompts/001 ·strip | prompts/002 ·strip | base_cond_ctt (=effect) · base_cond_neutral |
| `bneck_frozen` | closed negative | runs/004 | prompts/001 | — | bneck_frozen · bneck_frozen_shufcode |
| `bneck_ctx` | closed negative | runs/006 | prompts/001 | — | bneck_ctx_v2 · bneck_ctx_v2_shufcode |
| `surg1` | closed refined negative | runs/007 | prompts/001 | — | surg1_wsd · surg1_wsd_shufcode |
| `ctt_v3` | **champion (provisional)** | runs/008 | prompts/001 | prompts/002 | ctt_v2_pushA · ctt_v2_pushA_shufref · ctt_v2_pushA_effect · ctt_v2_pushA_plain |
| `ctt_v3_hs` | retired negative | runs/009 | prompts/001 | prompts/002 | ctt_v2_pushB · ctt_v2_pushB_shufref · ctt_v2_pushB_effect |
| `vap` | external baseline (one-sided) | runs/010 | prompts/001 ·ext | prompts/002 ·ext ; prompts/008 ·authorcfg | vap_neutral · vap_effect · vap_authorcfg · vap_tgtfull_refempty |
| `vfxmaster` | external baseline (one-sided) | runs/011 | prompts/001 ·ext | prompts/002 ·ext ; prompts/008 ·authorcfg | vfxmaster_neutral · vfxmaster_effect · vfxmaster_authorcfg · vfxmaster_tgtfull_refempty |
| `dualforce_control` | dual-force plain-FM control | runs/012 | prompts/001 | prompts/002 | dualforce_control_neutral · dualforce_control_effect |
| `dualforce_kd` | dual-force text-crutch KD treatment | runs/013 | prompts/001 | prompts/002 | dualforce_kd_neutral · dualforce_kd_effect |

Adding an arm = one row here + `eval_ladder/arms.yaml` entry + `stamp_rows.py` for its registry.
Campaign nicknames (pushA/pushB…) never become directory names — canonical slugs only.
Probe stamps (new-artifact grammar): `ctt_v3_ctl` · `ctt_v2_ctl` · `base_ctl` (controllability probe, gens */probe_ctl, evals/009).
