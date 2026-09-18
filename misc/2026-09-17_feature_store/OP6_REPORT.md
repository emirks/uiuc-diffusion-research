# OP6 — copy rate vs the generation's OWN reference (`scripts/copy_metrics.py`)

**Date:** 2026-09-18 · **Operator:** Op-6 · **Machine:** DeltaAI login CPU (numpy over stored features; no GPU)
**Eval:** `store/evals/039_copy_gridv3__dai__2026-09-18/` (per-arm `rows.jsonl` + `meta.yaml` + one `store/INDEX.md` line)
**Instrument:** `scripts/copy_metrics.py` — reuses `src/diffusion/transition_eval/m2_integrity.py::copy_score` / `mid_mask` VERBATIM (`TAU_COPY = 0.858`).

## What was fixed
In evals/028/030 M2a was scored against POOL clips (every pool clip is itself an eval item), so a
generation's OWN conditioning reference was never in the pool it was measured against — and `near_copy`
came out identical across all four paper arms on 100% of the pool rows (coordinator check 2026-09-18).
This eval scores M2a the way the harness asks it of a single item: between each generation's **mid
frames** and its **OWN reference's non-core (endpoint) frames** — "did the generation replay the
reference's own shots?".

## Definition (FIXED, as implemented)
For each generation `g` with own reference `r` (the `reference` field of `g`'s `grid.jsonl` row; `r`'s
features are in the corpus store at `data/processed/transitions_std121/<class>/features/<r>/dino_cls@dinov2b-r256.npz`):

- `gen_mid = mid_mask(T_g, n_pre, n_suf)` with the SAME n_pre/n_suf rule as `scripts/handoff_metrics.py`:
  HF grid → `n_pre=9`, `n_suf = 8 if sided=="two" else 0`; ED grid → `1/0`; externals (refVFX/VAP/VFXMaster) → `1/0`.
- `ref_core` = the reference's core mask computed **exactly as the harness computes a reference bundle**
  (`transition_eval/score.py::_ref_bundle_cache`): `morph_profile(r_feats, n_prefix=9, n_suffix=8, n_endpoints=2)`
  then `core_mask_v3(profile, r's sidedness)`, with the reference's sidedness (`onesided`/`twosided`) read
  from `corpus_manifest.json`. The non-core frames `~ref_core` are the reference's own scenes A/B — the
  content that must never appear.
- `copy_score(gen_feats, gen_mid, ref_feats, ref_core, TAU_COPY)` → `copy_max`, `near_copy`,
  `copy_gen_frame`, `copy_ref_frame`. Row also carries `ref_core_frac` (fraction of reference frames in
  core), `n_mid` (gen mid frames), and `missing` (empty for the whole population — all DINO features exist).

Row schema (`rows.jsonl`, one per gen; gitignored store artifact):
`item_id, seed, arm, n_pre, n_suf, n_mid, ref_core_frac, copy_max, near_copy, copy_gen_frame, copy_ref_frame, missing`.

**Caveat (recorded in meta.yaml):** `copy_max` is a max over mid frames, so longer generations (121 f)
have more chances to match than the externals' 49 f / 33 f — the externals' copy rate is, if anything,
**under**-estimated relative to ours; disclosed, not corrected.

## Coverage
19 variants, **7,242 gens, 0 rows with a missing feature**. Every reference resolved to a corpus clip
with a sidedness (120 unique references; reference stems are unique in the manifest); every gen and every
reference has stored DINO. Copy is defined on 100% of rows (`copy_defined == rows` for all 19 arms).
`store_fsck` reports the same benign `no <label>/results.json` warn as the handoff eval 038 (these
feature-based evals write `rows.jsonl`, not `results.json`); the only fsck FAILs are pre-existing and
unrelated (dino shufsig gens 024/026/027).

## Per-arm copy rate — all 19 variants (whole variant)
Copy rate = `100·mean(near_copy)` over the arm's rows. `copy_max_mean` = mean copy_max.

| harness_arm | grid | n | near_copy | copy_rate % | copy_max_mean |
|---|---|---:|---:|---:|---:|
| ic_gen_neutral_v3 | HF | 564 | 6 | 1.06 | 0.2616 |
| ic_gen_neutral_v3ed81 | ED | 204 | 0 | 0.00 | 0.2689 |
| ic_gen_effect_v3 | HF | 564 | 8 | 1.42 | 0.3665 |
| ic_gen_effect_v3ed81 | ED | 204 | 0 | 0.00 | 0.4281 |
| refvfx_author_native | external | 366 | 4 | 1.09 | 0.3524 |
| base_cond_neutral_v3 | HF | 564 | 6 | 1.06 | 0.2070 |
| base_cond_neutral_v3ed81 | ED | 204 | 0 | 0.00 | 0.1944 |
| base_cond_effect_v3 | HF | 564 | 6 | 1.06 | 0.3537 |
| base_cond_effect_v3ed81 | ED | 204 | 0 | 0.00 | 0.4429 |
| vap_author_native | external | 366 | 6 | 1.64 | 0.3963 |
| vfxmaster_author_native | external | 366 | 2 | 0.55 | 0.4117 |
| dualforce_control_neutral_v3 | HF | 564 | 6 | 1.06 | 0.3365 |
| dualforce_control_neutral_v3ed81 | ED | 204 | 2 | 0.98 | 0.5009 |
| dualforce_control_effect_v3 | HF | 564 | 6 | 1.06 | 0.3863 |
| dualforce_control_effect_v3ed81 | ED | 204 | 4 | 1.96 | 0.5381 |
| dualforce_dcg_w6_neutral_v3 | HF | 564 | 7 | 1.24 | 0.4044 |
| dualforce_dcg_w6_neutral_v3ed81 | ED | 204 | 1 | 0.49 | 0.5655 |
| dualforce_dcg_w6_effect_v3 | HF | 564 | 7 | 1.24 | 0.4275 |
| dualforce_dcg_w6_effect_v3ed81 | ED | 204 | 1 | 0.49 | 0.5842 |

## Sanity — per-arm copy rate on the 366-row shared set (the bug that was fixed)
Shared set (OP5 Table-B definition): the intersection of `(endpoint, reference, cell, seed)` keys across
the three externals — the one-sided zero-shot frontier, **n = 366** (= 183 triples × 2 seeds) — matched
into each arm. Own arms pool their **neutral** HF (`_neutral_v3`) + ED (`_neutral_v3ed81`) variants (162 +
204 = 366); externals use their author-native variant. Every one of the 7 arms covers all 366 shared keys.

| arm | n | near_copy | copy_rate % | copy_max_mean |
|---|---:|---:|---:|---:|
| VAP (author-native) | 366 | 6 | **1.64** | 0.3963 |
| VFXMaster (author-native) | 366 | 2 | **0.55** | 0.4117 |
| refVFX (author-native) | 366 | 4 | **1.09** | 0.3524 |
| LTX baseline LoRA / ic_gen (neutral) | 366 | 2 | **0.55** | 0.2462 |
| SEGUE w/o guidance / dualforce_control (neutral) | 366 | 4 | **1.09** | 0.4057 |
| SEGUE / dualforce_dcg_w6 (neutral) | 366 | 4 | **1.09** | 0.4730 |
| LTX (no reference) / base_cond (neutral) | 366 | 2 | **0.55** | 0.1868 |

**NOT identical across arms.** On the 366 shared keys the seven arms produce **5 distinct `near_copy`
vectors** (1 would mean identical). Row-by-row the flags disagree pairwise by up to 6 rows (e.g. VAP vs
SEGUE = 6, VAP vs VFXMaster = 4). The continuous `copy_max_mean` separates every arm and spans 0.187
(base_cond, no reference in context → least copying) to 0.473 (SEGUE / dcg_w6, strongest reference pull) —
a monotone, sensible ordering. This is exactly the discrimination the pool-scored 028/030 version could
not produce (near_copy identical across the four paper arms on 100% of the pool rows).

Notes on the shared-set rates: near-copies are rare on the zero-shot set by construction (the reference is
a *different-class, unseen* clip, so replaying its shots is unlikely), so the rates are all small and a few
arms land on the same rate (0.55 / 1.09) despite flagging *different* rows — the `near_copy` vectors and
`copy_max_mean` are the arm-discriminating signal, not the coarse rate alone. The externals' rate is
computed on 49 f / 33 f clips (fewer mid frames → fewer chances), so per the caveat their copy is if
anything under-counted relative to our 121 f arms.

## Reproduce
```
$LAB/envs-aarch64/ltx2/bin/python scripts/copy_metrics.py          # writes eval 039 (idempotent)
$LAB/envs-aarch64/ltx2/bin/python -m pytest tests/test_copy_metrics.py -q
$LAB/envs-aarch64/ltx2/bin/python misc/2026-09-17_feature_store/op6_scratch/shared_set.py   # the table above
```

## Files
- `scripts/copy_metrics.py` — instrument (committed).
- `tests/test_copy_metrics.py` — synthetic copy / disjoint / window-size / schema tests (5 pass).
- `store/evals/039_copy_gridv3__dai__2026-09-18/{<arm>/rows.jsonl, meta.yaml}` — meta committed; rows.jsonl are gitignored store artifacts.
- `store/INDEX.md` — one appended evals line (#39).
- `misc/2026-09-17_feature_store/op6_scratch/shared_set.py` — the 366-row shared-set analysis behind the sanity table.
