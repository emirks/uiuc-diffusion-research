# Op-2 — per-generation score rows (`scripts/store_per_gen.py`)

**Deliverable:** `scripts/store_per_gen.py` writes one row per GENERATION to
`store/evals/<eval>/<arm>/per_gen.jsonl`, next to the `items.jsonl` shards it derives from —
evals/028 for the 16 paper-arm variants, evals/030 for the 3 externals (the 19 gridv3 variants in
`population_gridv3.json`). 7242 rows total across 19 files (= the population's `n_gens`). CPU-only,
pure python over jsonl (+ one cv2 frame-count decode per variant). Idempotent (byte-identical on
re-run, verified). `per_gen.jsonl` is a store artifact and is git-ignored — NOT committed.

## Which column `transport_pct` carries: **S3 / app_ref** (NOT Look_u)

`transport_pct` is the paper's "%_same" capped pooled-%, computed from the **stored `app_ref`** (the
deployed-kernel m1a_S3 appearance similarity), not the size-free `Look_u` recompute. So the
externals' zero-shot values my rows carry are VAP 81.1 / VFXMaster 85.8 / **refVFX 62.2** — the S3
column. The paper draft's refVFX zero-shot **75.5 is the Look_u column** and is deliberately NOT
what these rows carry (per the brief). Look_u cannot be derived from stored `app_ref`; it needs the
size-free feature recompute (score_v3_zs.py's `Look_u` path) and is out of scope here.

## Canonical implementation traced

`transport_pct` = "%_same" = min(100, generation's mean similarity to the real videos of its
operator ÷ that operator's ceiling). Reproduced from the code that wrote **evals/028
`summary.json`** — `eval_ladder/run_eval.py::{ceilings, pool_means, item_pct}` driven by
`scripts/grid_v3/closeout.py`:

- **pool mean per (grid item_id, seed)** = `st.mean` of `app_ref` over that generation's GT-pool
  references, deduped by full eval item_id, `app_ref=None` dropped — exactly `run_eval.pool_means`.
  This is `transport_raw` in each row.
- **ceiling per class** = mean within-class off-diagonal similarity of the **certified m1a_S3
  matrix** (`.claude/worktrees/eval-v4-cert/.../distance_matrices.npz`), with the
  `eval_ladder/ceilings_v3.json` overlay for classes absent from it (7 new HF zero-shot + all
  EffectData `ed.*`) — exactly `run_eval.ceilings()`. This is `transport_ceiling`.
- **summary.json `level`** = mean over the cell's grid items of the per-item UNCAPPED ratio, where
  the per-item ratio = mean over its seeds of (per-seed pool mean ÷ ceiling) — exactly
  `run_eval.item_pct`. My rows store the raw parts (`transport_raw`, `transport_ceiling`), so both
  the uncapped summary levels and the capped per-generation `transport_pct` = 100·min(ratio,1) are
  recoverable; `transport_capped` flags ratio>1.

Data facts established while tracing: evals/028 `items.jsonl` rows are (generation × GT-pool-ref)
comparisons (`app_ref` varies per pool ref, so per-gen `transport_raw` is the mean over them); a
"generation" = (grid item_id, seed), 282 grid rows × 2 seeds = 564 for HF arms, 102×2=204 for ED
arms, 183×2=366 for externals; the `control_hold`/`control_lerp` pool entries are diagnostic
pseudo-gens (not in grid.jsonl) and are excluded.

## Verification (mandatory)

`--check` re-aggregates the WRITTEN `per_gen.jsonl` and exits 1 on any summary.json mismatch.

**(a) per_gen → evals/028 summary.json — EXACT, 0 mismatches.** All 16 arms reproduce every printed
number: per-cell `level`, `n`, `sd`, `copy_max_mean`, plus `headline_pct_same` and `items_scored`
(spot: ic_gen_neutral_v3 headline 0.8509, base_cond_neutral_v3 0.6141, dualforce_control_effect_v3
1.0022, dualforce_dcg_w6_effect_v3 1.0191). `--check` prints `[OK]`, exit 0.

**(b) 183-triple one-sided zero-shot, S3 capped pooled-% vs CHANGELOG 2026-09-17 23:23:**

| arm | per_gen (certified ceilings) | CHANGELOG (222-based) | Δpp |
|---|---|---|---|
| VAP | 81.15 | 81.1 | +0.05 |
| VFXMaster | 85.76 | 85.7 | +0.06 |
| refVFX | 62.23 | 62.2 | +0.03 |
| ic_gen neutral | 61.64 | 61.6 | +0.04 |
| dualforce_control neutral | 87.62 | 87.6 | +0.02 |
| dualforce_dcg_w6 neutral | 92.60 | 92.6 | −0.00 |

At 1-decimal rounding this reproduces refVFX 62.2, ic_gen 61.6, dualforce 87.6, dcg 92.6, VAP 81.1
exactly; **VFXMaster is the one that does not: 85.76 → 85.8 vs the CHANGELOG's 85.7.**

## Discrepancy diagnosis (not tuned)

The ≤0.06 pp spread, and the single 1-decimal miss (VFXMaster), come entirely from the **ceiling
source**, which the two canonical implementations differ on by design:

- **summary.json** (and my rows) use the **certified deployed m1a_S3 ceilings** (`run_eval.ceilings()`).
- **the CHANGELOG 183-triple** (`score_v3_zs.py`) states "222-based ceilings" — the frozen 222-pin
  populations recomputed via `reference_stats`, which evals/028's own `meta.yaml` records as
  differing from the deployed kernel by **calibration ratio 0.9965**.

I confirmed this directly: reconstructing the 222-based S3 ceilings the way `score_v3_zs.py` does
gives VAP 81.12→81.1, **VFXMaster 85.73→85.7**, refVFX 62.22→62.2 (all matching the CHANGELOG),
while the certified ceilings give 81.15, 85.76, 62.23. No single `transport_ceiling` can reproduce
both sources to the last digit. I chose the **certified ceilings** because summary.json is the store
artifact the brief requires to reproduce exactly (it does, all numbers), and because it is the
deployed instrument; the 183-triple was a 222-based side-analysis. This is reported, not tuned:
`--check` prints the table and the note, and gates its exit code only on the summary.json
reproduction.

## Decisions made (documented in the script)

- **Per-pool-reference fields** vary across a generation's pool rows (measured: `copy_max`,
  `cam_zpr`, `obj_csls` vary in 100% of multi-ref gens; `near_copy` in a minority). Reduced per
  generation: `copy_max`/`cam_zpr`/`obj_csls` = **mean over pool references** (mean `copy_max` this
  way is exactly what reproduces summary.json `copy_max_mean`); `near_copy` = **OR** (a near copy of
  ANY real reference). The clip-property fields (`max_seam_z`, `prefix_seam_z`, `suffix_seam_z`,
  `prefix_dino`, `prefix_lpips`, `core_degenerate`, `cross_high`) are constant per generation and
  taken verbatim.
- **`ref_in_v4_population`** is a per-pool-ref flag in items.jsonl (whether each GT-pool clip is in
  the certified 222-clip population); reduced per generation as `all(...)` = "scored entirely
  in-population". It is orthogonal to `tier` (a zero_shot-tier gen can be all-in-population).
- **`ref_class`** = `gt_pool_class` (the operator whose real videos the generation is scored
  against; for cross/foreign this is the reference's class, not the endpoint's).
- **`pct_type`** for externals: evals/030 grid.jsonl has no `pct_type`, so it is derived from
  content (`same`→`same`, else `proxy`), matching the grid-v3 convention.
- **`n_frames`** decoded once per variant (cv2): HF 121, ED 81, VAP/VFXMaster 49, refVFX 33.

## Tests

`tests/test_store_per_gen.py` (pytest, 6 tests, all pass): item-id split; the capped/uncapped/
at-ceiling `transport` arithmetic; and `collect_generations` over a synthetic `items.jsonl` (dedup
of a repeated item_id, `app_ref=None` drop, mean of per-ref fields, OR of `near_copy`, `all()` of
`ref_in_v4_population`, verbatim const fields) → known pct.

## Not done / out of scope

- Only the gridv3 population (owner's 2026-09-17 scope). Other store variants / legacy caches
  deferred.
- The `Look_u` (size-free) column is not produced — my rows carry S3/app_ref by directive.
- evals/030 has no summary.json, so externals are verified only via the 183-triple slice (above).
