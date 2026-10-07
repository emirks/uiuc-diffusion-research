# Per-frame DINO measures, ALL two-sided real clips vs base groups

| group | n | progress Gini (DINO) | off-line distance (DINO) | DR (DINO) |
|---|---|---|---|---|
| GT:all-two-sided | 73 | 0.56 [0.52, 0.62] | 0.51 [0.45, 0.60] | 0.53 [0.44, 0.65] |
| S-GRID:NULLGEN | 68 | 0.79 [0.69, 0.85] | 0.28 [0.25, 0.37] | 0.24 [0.17, 0.35] |
| S-GRID-F:NULLGEN | 38 | 0.80 [0.73, 0.84] | 0.30 [0.22, 0.39] | 0.26 [0.16, 0.39] |
| S-PROBE:R3:high | 90 | 0.85 [0.79, 0.88] | 0.22 [0.18, 0.26] | 0.18 [0.15, 0.22] |
| S-PROBE:R1:high | 90 | 0.67 [0.62, 0.72] | 0.41 [0.34, 0.49] | 0.30 [0.23, 0.52] |
| S-PROBE:R2:high | 90 | 0.69 [0.62, 0.75] | 0.40 [0.33, 0.47] | 0.31 [0.21, 0.46] |
| S-PROBE:R3:inplace | 30 | 0.59 [0.52, 0.63] | 0.43 [0.36, 0.48] | 0.43 [0.36, 0.49] |

GT = all 73 real clips of the 15 two-sided classes; DINOv2-base CLS per frame (224 whole-frame resize), a_idx 8 / b_idx 113. New GT features cached at $LAB/cache/null_default/features_gt_all (CPU-extracted).
Effect sizes vs GT-all (MWU AUC): grid default Gini higher 0.938 (p 1e-19), off-line lower 0.119 (p 3e-15); probe R3 scene-change 0.977 (p 6e-26) / 0.031 (p 4e-25).
