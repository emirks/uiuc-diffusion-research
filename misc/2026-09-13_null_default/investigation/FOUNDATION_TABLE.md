# Foundation table — what the base model tends toward with both anchors and no transition text

Two numbers per clip, computed frame by frame, no thresholds, no fitting.
- **Off-line distance** = mean distance of the interior frames from the straight line through the two anchor frames, in units of the anchor-to-anchor distance. Every member of the lerp family (any schedule: cross-fade, cut, swap) lies ON that line, so this is the distance from the family. 0 = on the family.
- **Progress Gini** = concentration of the per-frame progress along that line. 0 = every frame advances equally (cross-fade), 1 = one frame carries all the progress (cut).
Pixel space (128 px, blur 1.0) is primary: the family is literally defined there. DINOv2 CLS per frame is the semantic check.
Anchors: frames 8 and 113 of 121 (exp_024: 24 and 168 of 193). GT = all 73 real clips of the 15 two-sided classes. Medians [IQR].
The pixel cross-fade reads 0.45 off-line in DINO because a blend is not on the semantic line; that row is the family's own footprint in DINO, not a flaw.

| group | n | off-line distance, pixel | progress Gini, pixel | off-line distance, DINO | progress Gini, DINO |
|---|---|---|---|---|---|
| pixel cross-fade (family member, linear schedule) | — | 0 | 0 | 0.45 | 0 |
| hard cut (family member, step schedule) | — | 0 | 0.99 | 0 | 0.99 |
| real transitions, all two-sided classes | 73 | 0.53 [0.48, 0.63] | 0.56 [0.47, 0.65] | 0.51 [0.45, 0.60] | 0.56 [0.52, 0.62] |
| base, both anchors, caption only (grid, real anchors) | 68 | 0.25 [0.22, 0.34] | 0.84 [0.79, 0.89] | 0.28 [0.25, 0.37] | 0.79 [0.69, 0.85] |
| base, both anchors, caption only (grid, DAVIS anchors) | 38 | 0.28 [0.14, 0.45] | 0.83 [0.70, 0.89] | 0.30 [0.22, 0.39] | 0.80 [0.73, 0.84] |
| base, both anchors, captions only (probe R3, scene change) | 90 | 0.22 [0.17, 0.29] | 0.84 [0.77, 0.89] | 0.22 [0.18, 0.26] | 0.85 [0.79, 0.88] |
| base, both anchors, empty prompt (exp_024, other config) | 10 | 0.54 [0.46, 0.59] | 0.59 [0.50, 0.72] | — | — |
| base, both anchors, prompt = "transition" (exp_024, other config) | 10 | 0.53 [0.46, 0.60] | 0.59 [0.50, 0.70] | — | — |
| base, both anchors, full transition prompt (R2) | 90 | 0.37 [0.27, 0.47] | 0.67 [0.58, 0.73] | 0.40 [0.33, 0.47] | 0.69 [0.62, 0.75] |
| base, start anchor only, full prompt (R1) | 90 | 0.37 [0.27, 0.50] | 0.62 [0.56, 0.72] | 0.41 [0.34, 0.49] | 0.67 [0.62, 0.72] |
| base, both anchors, captions only, in-place pairs | 30 | 0.30 [0.21, 0.34] | 0.65 [0.58, 0.77] | 0.43 [0.36, 0.48] | 0.59 [0.52, 0.63] |
