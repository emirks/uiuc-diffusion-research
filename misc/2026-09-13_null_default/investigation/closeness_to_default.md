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

## Reportable continuous numbers (PIX; computed 2026-09-14)
Change duration d = ramp length of the best-fitting hold-A / linear change / hold-B curve, frames of the 104-frame interior (24 fps):
grid default 12 [8, 26] (n=68) · probe R3 scene-change 14 [8, 24] (n=90) · GT 80 [56, 104] (n=19) · R1 full prompt 48 [32, 64] · R2 48 [24, 64] · R3 in-place 48 [24, 48].
P(random default clip changes faster than random GT clip) = 0.957 (MWU p 5e-10); P(R3 faster than R1) = 0.857 (p 4e-17); paired R3−R1 same prompt+seed: median −32 frames, R3 shorter in 82 % of 90 pairs.
Swap index = d_lerp / (d_lerp + d_swap) with d = RMSE of the progress curve to the full lerp / to the nearest 12-frame swap (1 = pure swap, 0 = pure lerp, 0.5 = equidistant):
grid default 0.77 [0.70, 0.87] · DAVIS-anchor default 0.71 [0.49, 0.84] · R3 scene-change 0.82 [0.73, 0.87] · GT 0.36 [0.31, 0.50] · R1 0.58 [0.43, 0.73] · R2 0.63 [0.48, 0.74] · R3 in-place 0.61 [0.50, 0.76] · exp_024 empty 0.51 / "transition" 0.52 (n=10, other config).
P(default swap index > GT) = 0.959 (p 6e-10); P(R3 > R1) = 0.839 (p 2e-15). DINO gives the same medians within 0.05.
