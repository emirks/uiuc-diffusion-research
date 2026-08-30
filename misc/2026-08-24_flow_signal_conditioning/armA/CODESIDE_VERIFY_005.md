# CODESIDE_VERIFY_005 — code-side ctt_v2plus_s6reshape invariant battery

`samples.jsonl` = **138,147** rows. Checks vs inventories / EFFECTDATA_r832 ROSTER (independent of build_samples). Invariants 1-5.

## 1 · Per-stratum counts vs independent predictions

| stratum | rows | predicted | source | ok |
|---|--:|--:|---|:--:|
| S0 | 385 | 385 | S0 inv ring | ✓ |
| S1 | 3,675 | 3,675 | S1 inv ring | ✓ |
| S2a | 22,731 | 22,731 | certified | ✓ |
| S2b | 23,577 | 23,577 | certified | ✓ |
| S4 | 6,000 | 6,000 | certified | ✓ |
| S6 | 81,779 | 81,779 | r832 ROSTER effect×grid ring | ✓ |
| **total** | **138,147** | **138,147** | | ✓ |

S6 reconciliation: **28,552 distinct targets + 92 drops = 28,644** (== 28,644 r832 ROSTER clips) — ✓

## 2 · Set-equality (no dup pairs; stem-sets match; same-grid + different-subject)

- duplicate pairs: **0** ✓
- S6 targets == r832 ROSTER non-singleton set, refs ⊆ it, targets∩drops=∅: ✓
- S6 pairs same-grid: ✓ · different-subject: ✓ (all 81,779 S6 rows)
- S1 distinct targets (1,225) == S1 inventory clips (1,225): ✓

## 3 · Shared-stub detector (source paths belong to exactly one clip except structural)

| stratum | distinct conditions paths | distinct caption_keys | ok (equal & >1) |
|---|--:|--:|:--:|
| S0 | 139 | 139 | ✓ |
| S1 | 350 | 350 | ✓ |
| S2a | 318 | 318 | ✓ |
| S2b | 785 | 785 | ✓ |
| S4 | 2,000 | 2,000 | ✓ |
| S6 | 2,000 | 2,000 | ✓ |

- latents 1:1 with target: ✓ · reference_latents 1:1 with reference: ✓

## 4 · Path-scheme gate (relative-under-root; no absolute; no `..`) + mask set

distinct paths: **100,301**; scheme violations: **0** ✓
mask set (used == disk == expected 5): ✓  used=['f11_h16_w26_p1_onesided.pt', 'f11_h26_w16_p1_onesided.pt', 'f16_h20_w15_p2_onesided.pt', 'f16_h20_w15_p2_twosided.pt', 'f5_h14_w26_p1_onesided.pt']

distinct source roots under `_src` (reported):
  - `_src/datasets/ctt_v2` — 99,879
  - `_src/eval_ladder/dataset` — 278
  - `_src/experiments/exp_058_ic_lora_diverse_retrain` — 96
  - `_src/experiments/exp_062_ladder_r2r3_specialists` — 24
  - `_src/experiments/exp_064_ic3_aligned_retrain` — 19
  - `_mask_store` — 5

## 5 · Existence (full) + shape (>=400 S6 rows spanning both grids; keyed independently)

- existence (FULL, 100,301 distinct paths): **PASS** ✓
- S6 shape sample: **420 rows** over grids {'[11, 16, 26]': 210, '[11, 26, 16]': 210} (target+reference keyed independently) ✓
- shape (all 620 sampled rows × target+reference): **PASS** ✓

## Overall: ALL INVARIANTS PASS
