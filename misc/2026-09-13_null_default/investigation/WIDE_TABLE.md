# Reals vs generateds vs R3 — pixel, per frame, no thresholds

Off-line distance = mean distance of interior frames from the line through the two anchor frames (gap units; 0 = a member of the lerp family). Progress Gini = concentration of per-frame progress along that line (0 = cross-fade, 1 = cut). Windows: both-anchor rows a=8,b=113; start-only rows and one-sided reals a=8,b=120 (own last frame). GENERATED = every unique, clean, 121-f clip of base_cond / dualforce_control / dualforce_dcg (v2+v3, neutral+effect, both+start) from the Sep-08 re-measure table. Medians [IQR].

## Three populations (pixel, per frame)

| population | n | off-line distance | progress Gini |
|---|---|---|---|
| REAL two-sided | 73 | 0.53 [0.48, 0.63] | 0.56 [0.47, 0.65] |
| REAL one-sided | 620 | 0.49 [0.40, 0.57] | 0.44 [0.36, 0.52] |
| GENERATED | 2466 | 0.50 [0.37, 0.66] | 0.52 [0.45, 0.62] |
| R3 (null, scene change) | 90 | 0.22 [0.17, 0.29] | 0.84 [0.77, 0.89] |
| R3 (null, in-place) | 30 | 0.30 [0.21, 0.34] | 0.65 [0.58, 0.77] |

## Generated, by arm / grid / text / anchors (pixel)

| arm | grid | text | anchors | n | off-line distance | progress Gini |
|---|---|---|---|---|---|---|
| base_cond | v2 | effect | both | 52 | 0.50 [0.38, 0.60] | 0.67 [0.59, 0.73] |
| base_cond | v2 | effect | start | 160 | 0.29 [0.22, 0.41] | 0.60 [0.49, 0.71] |
| base_cond | v2 | neutral | both | 30 | 0.25 [0.21, 0.33] | 0.83 [0.80, 0.89] |
| base_cond | v2 | neutral | start | 110 | 0.39 [0.28, 0.52] | 0.66 [0.48, 0.80] |
| base_cond | v3 | effect | both | 46 | 0.47 [0.38, 0.67] | 0.71 [0.67, 0.75] |
| base_cond | v3 | effect | start | 139 | 0.34 [0.26, 0.46] | 0.54 [0.46, 0.63] |
| base_cond | v3 | neutral | both | 8 | 0.26 [0.23, 0.43] | 0.88 [0.77, 0.91] |
| base_cond | v3 | neutral | start | 39 | 0.52 [0.45, 0.60] | 0.49 [0.41, 0.62] |
| dualforce_control | v2 | effect | both | 52 | 0.65 [0.53, 0.76] | 0.54 [0.49, 0.64] |
| dualforce_control | v2 | effect | start | 160 | 0.40 [0.33, 0.56] | 0.50 [0.43, 0.60] |
| dualforce_control | v2 | neutral | both | 52 | 0.59 [0.50, 0.73] | 0.56 [0.49, 0.65] |
| dualforce_control | v2 | neutral | start | 160 | 0.42 [0.33, 0.53] | 0.49 [0.43, 0.57] |
| dualforce_control | v3 | effect | both | 46 | 0.65 [0.55, 0.82] | 0.57 [0.51, 0.64] |
| dualforce_control | v3 | effect | start | 139 | 0.44 [0.37, 0.57] | 0.49 [0.39, 0.57] |
| dualforce_control | v3 | neutral | both | 46 | 0.57 [0.51, 0.71] | 0.56 [0.51, 0.62] |
| dualforce_control | v3 | neutral | start | 139 | 0.45 [0.35, 0.54] | 0.46 [0.38, 0.54] |
| dualforce_dcg_w1 | v2 | neutral | both | 26 | 0.56 [0.50, 0.68] | 0.59 [0.50, 0.70] |
| dualforce_dcg_w1 | v2 | neutral | start | 80 | 0.42 [0.34, 0.53] | 0.48 [0.42, 0.57] |
| dualforce_dcg_w1p5 | v2 | neutral | both | 26 | 0.60 [0.52, 0.73] | 0.56 [0.52, 0.67] |
| dualforce_dcg_w1p5 | v2 | neutral | start | 80 | 0.45 [0.36, 0.54] | 0.48 [0.43, 0.55] |
| dualforce_dcg_w3 | v2 | neutral | both | 26 | 0.73 [0.67, 0.81] | 0.53 [0.47, 0.62] |
| dualforce_dcg_w3 | v2 | neutral | start | 80 | 0.50 [0.40, 0.65] | 0.50 [0.45, 0.56] |
| dualforce_dcg_w6 | v2 | effect | both | 26 | 0.81 [0.69, 0.92] | 0.55 [0.48, 0.64] |
| dualforce_dcg_w6 | v2 | effect | start | 79 | 0.56 [0.46, 0.68] | 0.52 [0.44, 0.58] |
| dualforce_dcg_w6 | v2 | neutral | both | 26 | 0.82 [0.72, 0.95] | 0.52 [0.46, 0.59] |
| dualforce_dcg_w6 | v2 | neutral | start | 80 | 0.57 [0.46, 0.71] | 0.49 [0.45, 0.57] |
| dualforce_dcg_w6 | v3 | effect | both | 70 | 0.82 [0.70, 1.00] | 0.54 [0.47, 0.61] |
| dualforce_dcg_w6 | v3 | effect | start | 210 | 0.59 [0.45, 0.73] | 0.49 [0.44, 0.57] |
| dualforce_dcg_w6 | v3 | neutral | both | 70 | 0.83 [0.71, 1.03] | 0.53 [0.47, 0.60] |
| dualforce_dcg_w6 | v3 | neutral | start | 209 | 0.57 [0.45, 0.71] | 0.49 [0.43, 0.55] |

## Separation (probability that a random clip of the first population beats a random clip of the second)

| comparison | off-line: first lower | Gini: first higher |
|---|---|---|
| GENERATED vs REAL all | 0.523 (p 1e+00) | 0.663 (p 1e-39) |
| R3 scene-change vs REAL all | 0.069 (p 8e-41) | 0.966 (p 3e-47) |
| R3 scene-change vs GENERATED | 0.106 (p 2e-37) | 0.916 (p 2e-41) |
| GENERATED vs REAL two-sided | 0.452 (p 8e-02) | 0.443 (p 1e+00) |
| R3 scene-change vs REAL two-sided | 0.041 (p 4e-24) | 0.938 (p 4e-22) |
| GENERATED both-anchor only (602) vs REAL two-sided | 0.673 (p 1e+00) | 0.583 (p 1e-02) |
