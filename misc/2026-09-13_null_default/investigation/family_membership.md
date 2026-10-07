# Family membership without fitting: concentration of A→B progress (Gini of per-frame progress increments)
Progress tau_t = projection of frame t onto the A→B line (0 at A, 1 at B). Increments |tau_{t+1} − tau_t| over the interior.
Gini = 1 → all progress in one frame (the cut family); Gini = 0 → equal progress every frame (the lerp). Landmarks: pixel LERP 0.00, CUT50 0.98.
Also: RMSE of tau(s) to the nearest step function (cut at any position) and to the lerp. Medians [IQR]. Per-frame DINO CLS (104 interior frames); PIX on the 48-point cached grid.

| group | n | Gini DINO | Gini PIX | dist to nearest cut (DINO) | dist to lerp (DINO) |
|---|---|---|---|---|---|
| base default, grid, real anchors | 68 | 0.79 [0.69, 0.85] | 0.84 [0.77, 0.88] | 0.13 | 0.27 |
| base default, grid, DAVIS anchors | 38 | 0.80 [0.73, 0.84] | 0.82 [0.65, 0.90] | 0.11 | 0.27 |
| base default, probe R3, scene change | 90 | 0.85 [0.79, 0.88] | 0.83 [0.75, 0.88] | 0.10 | 0.31 |
| probe R3, in-place | 30 | 0.59 [0.52, 0.63] | 0.65 [0.58, 0.75] | 0.19 | 0.17 |
| exp_024 empty prompt (other config) | 10 | 0.54 | 0.60 | 0.20 | 0.24 |
| exp_024 "transition" (other config) | 10 | 0.55 | 0.60 | 0.19 | 0.23 |
| real transitions GT | 19 | 0.56 [0.54, 0.60] | 0.55 [0.45, 0.59] | 0.25 | 0.12 |
| base + full prompt R1 | 90 | 0.67 [0.62, 0.72] | 0.62 [0.54, 0.71] | 0.23 | 0.19 |
| base + full prompt + both anchors R2 | 90 | 0.69 [0.62, 0.75] | 0.67 [0.57, 0.74] | 0.23 | 0.21 |

P(default Gini > GT Gini): grid DINO 0.943 (p 2e-9), grid PIX 0.972 (p 2e-10); P(R3 > R1) 0.923 (p 5e-23).
Per-clip values: family_membership.csv.
