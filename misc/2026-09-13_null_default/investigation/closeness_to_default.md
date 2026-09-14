# Closeness to the observed default FAMILY vs to the full-length lerp

swapfam = RMSE between the clip's progress curve tau(s) and the nearest member of {hold A, ramp of 12 frames (0.115 of the interior) from s0, hold B}, s0 free (the default's fitted ramp length; position varies per clip). lerp = RMSE to the full-length lerp tau=s. offchord = mean distance of interior latents from the A-B line, gap units. Medians per group. Note: a FIXED template (s0=0.23) separated nothing (timing RMSE 0.27-0.30 for base defaults vs 0.34 GT) because swap position varies (IQR 0.07-0.39); the family distance is the right closeness.

| group | n | vae_swapfam | vae_lerp | pix_swapfam | pix_lerp | dino_swapfam | dino_lerp | vae_offchord |
|---|---|---|---|---|---|---|---|---|
| S-GRID:NULLGEN | 68 | 0.205 | 0.170 | 0.080 | 0.262 | 0.082 | 0.267 | 0.590 |
| S-GRID-F:NULLGEN | 38 | 0.264 | 0.158 | 0.100 | 0.239 | 0.088 | 0.261 | 0.707 |
| S-PROBE:R3:high | 90 | 0.207 | 0.171 | 0.066 | 0.301 | 0.055 | 0.306 | 0.580 |
| S-PROBE:R3:inplace | 30 | 0.217 | 0.131 | 0.133 | 0.194 | 0.153 | 0.171 | 0.574 |
| S-SWEEP:A_empty | 10 | 0.286 | 0.255 | 0.169 | 0.194 | 0.158 | 0.238 | 0.664 |
| S-SWEEP:A_word | 10 | 0.283 | 0.261 | 0.173 | 0.199 | 0.153 | 0.228 | 0.655 |
| S-GRID:GT | 19 | 0.300 | 0.119 | 0.223 | 0.133 | 0.212 | 0.119 | 0.688 |
| S-PROBE:R1:high | 90 | 0.270 | 0.109 | 0.148 | 0.212 | 0.190 | 0.194 | 0.647 |
| S-PROBE:R2:high | 90 | 0.259 | 0.113 | 0.140 | 0.235 | 0.192 | 0.211 | 0.641 |

## Share of clips whose progress curve is closer to the 12-frame swap family than to the full lerp

| group | n | PIX | DINO | VAE |
|---|---|---|---|---|
| S-GRID:NULLGEN | 68 | 0.94 | 0.94 | 0.47 |
| S-GRID-F:NULLGEN | 38 | 0.71 | 0.97 | 0.26 |
| S-PROBE:R3:high | 90 | 0.97 | 0.99 | 0.40 |
| S-PROBE:R3:inplace | 30 | 0.73 | 0.60 | 0.23 |
| S-SWEEP:A_empty | 10 | 0.50 | 0.80 | 0.40 |
| S-SWEEP:A_word | 10 | 0.50 | 0.60 | 0.50 |
| S-GRID:GT | 19 | 0.26 | 0.26 | 0.00 |
| S-PROBE:R1:high | 90 | 0.64 | 0.53 | 0.16 |
| S-PROBE:R2:high | 90 | 0.71 | 0.53 | 0.18 |
