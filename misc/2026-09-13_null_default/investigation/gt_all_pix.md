# Per-frame PIX measures, ALL two-sided real clips vs base groups (same decoder)

| group | n | progress Gini | off-line distance | DR | change duration (frames) |
|---|---|---|---|---|---|
| GT:all-two-sided | 73 | 0.56 [0.47, 0.65] | 0.53 [0.48, 0.63] | 0.58 [0.51, 0.66] | 80 [48, 104] |
| S-GRID:NULLGEN | 68 | 0.84 [0.79, 0.89] | 0.25 [0.22, 0.34] | 0.21 [0.16, 0.31] | 12 [7, 26] |
| S-GRID-F:NULLGEN | 38 | 0.83 [0.70, 0.89] | 0.28 [0.14, 0.45] | 0.29 [0.11, 0.55] | 14 [2, 76] |
| S-PROBE:R3:high | 90 | 0.84 [0.77, 0.89] | 0.22 [0.17, 0.29] | 0.21 [0.11, 0.29] | 12 [8, 24] |
| S-PROBE:R1:high | 90 | 0.62 [0.56, 0.72] | 0.37 [0.27, 0.50] | 0.37 [0.23, 0.50] | 48 [32, 64] |
| S-PROBE:R2:high | 90 | 0.67 [0.58, 0.73] | 0.37 [0.27, 0.47] | 0.33 [0.23, 0.45] | 48 [24, 64] |
| S-PROBE:R3:inplace | 30 | 0.65 [0.58, 0.77] | 0.30 [0.21, 0.34] | 0.29 [0.17, 0.34] | 48 [24, 48] |

GT = all 73 real clips of the 15 two-sided classes (union of the grid's two-sided classes and the corpus's twosided_transitions folders; the 19 anchor twins are a subset), a_idx 8 / b_idx 113, same 128-px decoder as the base groups.
Effect sizes vs GT-all (Mann-Whitney AUC): grid default Gini higher 0.972 (p 2e-22), off-line distance lower 0.077 (p 2e-18), change duration shorter 0.058 (p 4e-20); probe R3 scene-change: 0.938 / 0.041 / 0.053 (p ≤ 4e-22).
