# CODESIDE_VERIFY — code-side ctt_v2plus invariant battery

`samples.jsonl` = **114,215** rows. Checks vs inventories/ROSTER (independent of build_samples). fable-advisor invariants 1-5.

## 1 · Per-stratum counts vs independent predictions

| stratum | rows | predicted | source | ok |
|---|--:|--:|---|:--:|
| S0 | 385 | 385 | S0 inv ring | ✓ |
| S1 | 3,675 | 3,675 | S1 inv ring | ✓ |
| S2a | 22,731 | 22,731 | certified | ✓ |
| S2b | 23,577 | 23,577 | certified | ✓ |
| S4 | 6,000 | 6,000 | certified | ✓ |
| S6 | 57,847 | 57,847 | ROSTER shape-split ring | ✓ |
| **total** | **114,215** | **114,215** | | ✓ |

S6 reconciliation: **26,266 distinct targets + 2,378 drops = 28,644** (== 28,644 ROSTER clips) — ✓

## 2 · Set-equality (no dup pairs; stem-sets match)

- duplicate pairs: **0** ✓
- S6 targets == ROSTER-predicted non-singleton set, refs ⊆ it, targets∩drops=∅: ✓
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

## 4 · Path-scheme gate (relative-under-root; no absolute; no `..`)

distinct paths: **95,731**; violations (absolute / `..` / not `_src|_mask_store`): **0** ✓

distinct source roots under `_src` (reported):
  - `_src/datasets/ctt_v2` — 95,307
  - `_src/eval_ladder/dataset` — 278
  - `_src/experiments/exp_058_ic_lora_diverse_retrain` — 96
  - `_src/experiments/exp_062_ladder_r2r3_specialists` — 24
  - `_src/experiments/exp_064_ic3_aligned_retrain` — 19
  - `_mask_store` — 7

## 5 · Existence (full) + shape (sampled, keyed independently)

_(existence re-stat SKIPPED via --fast — proven PASS over all distinct paths in the prior full run)_
- existence (FULL, 95,731 distinct paths): **PASS** ✓
- shape (sampled 240 rows × target+reference, keyed to each stem's own shape): **PASS** ✓

## Overall: ALL INVARIANTS PASS
