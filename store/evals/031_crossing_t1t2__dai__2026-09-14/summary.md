# evals/031_crossing_t1t2 — headline (levels, no verdicts)

Two from-scratch IC-LoRA arms (crossed / uncrossed), T1/T2 test clips generated on eps, scored with transition-eval v4 on ONE machine (DeltaAI GH200). Readout = `app_ref` (v4 M1, HIGHER=more like the pool class) pooled over up to 8 same-class GT clips per row / class ceiling. Owner terms: shots, operators, pairs, references, crossed/uncrossed. Blocks: s0s1 (27 shots, 9 S1 manners), s2 (12 endpoints, 12 shaders); 2 generation seeds.

- instrument: reference_v4 sha256 `459fd9a7…` · corpus_manifest_xab sha256 `3e8bd29687b7` (1037 clips) · reference-corpus 222 pin `5a7a8be9f8a0` (amendment 2) · τ_copy 0.858 · harness transition-eval v4.0.0 (eval-v4-cert worktree @ 4e792c02).

- scored rows/arm (lines · real · control_lerp · control_hold · error): crossed 3744 · 1872 · 576 · 1296 · 0; uncrossed 3744 · 1872 · 576 · 1296 · 0.

## T1 — %_same toward the DESIGNATED operator (tie/level check)

| arm | block | n items | n item×seed | %_same | raw app_ref | %_same s42 | %_same s43 | copy_max mean | near_copy (item×seed) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crossed | s0s1 | 27 | 54 | 82.92 | 0.69265 | 85.75 | 80.09 | 0.3042 | 8/54 |
| crossed | s2 | 12 | 24 | 99.86 | 0.74016 | 99.73 | 99.98 | 0.3613 | 0/24 |
| uncrossed | s0s1 | 27 | 54 | 75.0 | 0.63035 | 75.69 | 74.32 | 0.2673 | 8/54 |
| uncrossed | s2 | 12 | 24 | 97.71 | 0.71545 | 95.9 | 99.53 | 0.3666 | 0/24 |

## T2 — margin = %_missing − %_designated (toward the MISSING/reference operator)

| arm | block | n items | n item×seed | %_missing | %_designated | margin (pp) | margin raw app | share margin>0 (item×seed) | copy_max mean | near_copy (item×seed) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crossed | s0s1 | 27 | 54 | 83.99 | 56.71 | 27.28 | 0.22198 | 0.815 | 0.2551 | 14/54 |
| crossed | s2 | 12 | 24 | 96.44 | 79.78 | 16.66 | 0.14839 | 0.708 | 0.3363 | 2/24 |
| uncrossed | s0s1 | 27 | 54 | 76.56 | 55.41 | 21.15 | 0.18178 | 0.759 | 0.242 | 17/54 |
| uncrossed | s2 | 12 | 24 | 96.99 | 86.06 | 10.93 | 0.08772 | 0.667 | 0.3461 | 2/24 |

## Lerp/hold-control level (raw app_ref mean; controls ignored in the headline %)

| arm | block | control | raw app_ref mean | n item×seed |
|---|---|---|---:|---:|
| crossed | s2 | lerp | 0.42058 | 72 |
| crossed | s0s1 | hold | 0.14661 | 162 |
| uncrossed | s2 | lerp | 0.42058 | 72 |
| uncrossed | s0s1 | hold | 0.14661 | 162 |

## Paired crossed − uncrossed T2 %margin (same T2 clip content + seed)

n paired = 78; mean(margin_crossed − margin_uncrossed) = **6.0 pp** (raw 0.04649); crossed>uncrossed on %margin 47/78 (on raw margin 48/78).

| block | n | mean diff (pp) | mean diff raw | crossed>uncrossed %margin | crossed>uncrossed raw |
|---|---:|---:|---:|---:|---:|
| s0s1 | 54 | 6.12 | 0.04019 | 33/54 | 33/54 |
| s2 | 24 | 5.73 | 0.06066 | 14/24 | 15/24 |

Full methods, ceilings, per-operator table, paths and "what was not done" in `misc/2026-09-13_crossing_ablation/eval/REPORT.md`; every number in `summary.json`; per-item CSVs beside this file.

