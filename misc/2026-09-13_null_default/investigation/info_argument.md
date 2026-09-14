## Step C numbers

### (1) The pixel LERP read in the other spaces (landmark-source clips; medians)

| space | stratum | n | DR | M | nu_max | path_over_gap | step_share | swap_sharp | local_at_peak |
|---|---|---|---|---|---|---|---|---|---|
| DINO | S-GRID | 19 | 0.454 | 0.375 | 0.791 | 3.538 | 0.025 |  |  |
| DINO | S-GRID-F | 38 | 0.382 | 0.279 | 0.803 | 3.303 | 0.023 |  |  |
| DINO | S-GRID-START | 204 | 0.566 | 0.351 | 0.92 | 4.074 | 0.019 |  |  |
| DINO | S-PROBE | 120 | 0.403 | 0.284 | 0.787 | 3.305 | 0.026 |  |  |
| DINO | S-SWEEP | 60 | 0.369 | 0.259 | 0.804 | 3.317 | 0.017 |  |  |
| VAE | S-GRID | 19 | 0.279 | 0.615 | 0.573 | 3.481 | 0.081 |  |  |
| VAE | S-GRID-F | 38 | 0.272 | 0.538 | 0.574 | 3.872 | 0.083 |  |  |
| VAE | S-GRID-START | 204 | 0.285 | 0.615 | 0.576 | 4.037 | 0.083 |  |  |
| VAE | S-PROBE | 120 | 0.268 | 0.615 | 0.569 | 3.196 | 0.083 |  |  |
| VAE | S-SWEEP | 60 | 0.28 | 0.529 | 0.572 | 4.84 | 0.061 |  |  |
| TRANS | S-GRID | 19 |  |  | 0.283 |  |  | 0.124 | 0.0 |
| TRANS | S-GRID-F | 38 |  |  | 0.314 |  |  | 0.15 | 0.0 |
| TRANS | S-GRID-START | 204 |  |  | 0.141 |  |  | 0.109 | 0.0 |
| TRANS | S-PROBE | 120 |  |  | 0.266 |  |  | 0.154 | 0.0 |
| TRANS | S-SWEEP | 60 |  |  | 0.254 |  |  | 0.11 | 0.0 |

### (2) VAE latent geometry (interior latent timesteps; norms relative to the mean anchor-latent norm)

| stratum·group | n | gap / anchor-norm | d(enc-LERP, LATLERP) med (gap units) | at mid | LATLERP min norm | enc-LERP min norm | real clip min norm | real clip med norm | CUT50 min norm |
|---|---|---|---|---|---|---|---|---|---|
| S-GRID·GT | 19 | 1.343 [1.274, 1.386] | 0.380 [0.357, 0.495] | 0.375 [0.358, 0.488] | 0.742 [0.716, 0.766] | 0.892 [0.869, 0.903] | 0.876 [0.833, 0.924] | 0.968 [0.934, 0.987] | 0.980 [0.938, 0.987] |
| S-GRID·NULLGEN | 68 | 1.353 [1.293, 1.389] | 0.385 [0.357, 0.514] | 0.388 [0.362, 0.501] | 0.737 [0.716, 0.763] | 0.894 [0.875, 0.903] | 0.921 [0.894, 0.954] | 1.005 [0.981, 1.024] | 0.984 [0.945, 0.987] |
| S-PROBE·R3 | 120 | 1.408 [1.358, 1.486] | 0.376 [0.330, 0.464] | 0.378 [0.342, 0.463] | 0.709 [0.666, 0.733] | 0.856 [0.834, 0.894] | 0.907 [0.871, 0.943] | 1.005 [0.958, 1.064] | 0.948 [0.910, 0.993] |

### (3) DINO chord (feature-space lerp) midpoint norm — a unit-sphere CLS space; chord points are not CLS vectors of any image

| stratum·group | cos(A,B) med [IQR] | chord midpoint norm med [IQR] |
|---|---|---|
| S-GRID·GT | 0.164 [0.063, 0.293] | 0.763 [0.729, 0.804] |
| S-GRID·NULLGEN | 0.146 [0.049, 0.260] | 0.757 [0.724, 0.794] |
| S-PROBE·R3 | 0.036 [0.010, 0.289] | 0.720 [0.711, 0.803] |
