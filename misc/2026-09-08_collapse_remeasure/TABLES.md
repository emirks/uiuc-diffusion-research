# Collapse re-measurement on existing store generations (2026-09-08)

Rows scored: 6663 · byte-identical duplicates: 584 · foreign/davis: 2186 · static (gap too small): 143 · **unique clean clips analysed: 4031**

Confinement residual DR = median normalised off-endpoint-line residual of interior frames (0 = on the line). On-line = DR ≤ 0.12 (descriptive cut point, kept from the Aug-24 campaign; sensitivity below). M = mid-band coverage of the projection coordinate (dissolve 0.5, cut/freeze 0). CIs are 95% cluster bootstraps over endpoint pairs.

## A. Levels per arm × grid × prompt × conditioning (unique clean clips)

| arm | grid | prompt | cond | n clips | n static (excl.) | n endpoints | DR median [IQR] | DR CI | M median | on-line share [CI] | on-line n | DISS/CUT/FRZ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_cond | v2 | effect | both | 52 | 0 | 15 | 0.385 [0.248, 0.598] | [0.277, 0.474] | 0.25 | 5.8% [0.0, 18.8] | 3 | 0/3/0 |
| base_cond | v2 | effect | start | 160 | 0 | 41 | 0.310 [0.205, 0.473] | [0.280, 0.350] | 0.23 | 10.6% [4.9, 16.9] | 17 | 0/17/0 |
| base_cond | v2 | neutral | both | 30 | 0 | 15 | 0.211 [0.136, 0.306] | [0.184, 0.301] | 0.06 | 26.7% [6.7, 46.7] | 8 | 0/8/0 |
| base_cond | v2 | neutral | start | 110 | 2 | 55 | 0.430 [0.300, 0.559] | [0.382, 0.473] | 0.22 | 6.4% [1.8, 12.7] | 7 | 0/7/0 |
| base_cond | v3 | effect | both | 94 | 0 | 19 | 0.385 [0.251, 0.575] | [0.306, 0.467] | 0.23 | 8.5% [0.0, 20.4] | 8 | 1/7/0 |
| base_cond | v3 | effect | start | 277 | 1 | 61 | 0.337 [0.232, 0.497] | [0.307, 0.372] | 0.26 | 7.9% [3.6, 13.2] | 22 | 0/22/0 |
| base_cond | v3 | neutral | both | 38 | 0 | 19 | 0.211 [0.166, 0.306] | [0.188, 0.288] | 0.06 | 21.1% [5.3, 39.5] | 8 | 0/8/0 |
| base_cond | v3 | neutral | start | 119 | 3 | 60 | 0.506 [0.366, 0.637] | [0.455, 0.552] | 0.34 | 3.4% [0.0, 7.6] | 4 | 0/4/0 |
| base_cond | v3ed81 | effect | start | 136 | 0 | 51 | 0.219 [0.119, 0.323] | [0.181, 0.259] | 0.16 | 26.5% [18.4, 35.2] | 36 | 1/35/0 |
| base_cond | v3ed81 | neutral | start | 32 | 70 | 22 | 0.100 [0.022, 0.209] | [0.029, 0.175] | 0.09 | 53.1% [34.4, 73.3] | 17 | 2/3/12 |
| dualforce_control | v2 | effect | both | 52 | 0 | 15 | 0.692 [0.570, 0.797] | [0.592, 0.764] | 0.45 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_control | v2 | effect | start | 160 | 0 | 41 | 0.446 [0.347, 0.597] | [0.416, 0.498] | 0.37 | 2.5% [0.0, 6.9] | 4 | 0/4/0 |
| dualforce_control | v2 | neutral | both | 52 | 0 | 15 | 0.642 [0.518, 0.748] | [0.560, 0.695] | 0.43 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_control | v2 | neutral | start | 160 | 0 | 41 | 0.457 [0.364, 0.579] | [0.410, 0.481] | 0.37 | 2.5% [0.0, 6.5] | 4 | 0/4/0 |
| dualforce_control | v3 | effect | both | 94 | 0 | 19 | 0.693 [0.589, 0.827] | [0.623, 0.770] | 0.37 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_control | v3 | effect | start | 277 | 1 | 61 | 0.480 [0.362, 0.609] | [0.443, 0.516] | 0.37 | 1.8% [0.0, 4.8] | 5 | 0/5/0 |
| dualforce_control | v3 | neutral | both | 94 | 0 | 19 | 0.616 [0.519, 0.738] | [0.564, 0.668] | 0.37 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_control | v3 | neutral | start | 277 | 1 | 61 | 0.467 [0.371, 0.585] | [0.438, 0.493] | 0.39 | 2.5% [0.0, 5.9] | 7 | 0/7/0 |
| dualforce_control | v3ed81 | effect | start | 136 | 0 | 51 | 0.462 [0.360, 0.551] | [0.420, 0.498] | 0.38 | 2.2% [0.0, 5.9] | 3 | 0/3/0 |
| dualforce_control | v3ed81 | neutral | start | 136 | 0 | 51 | 0.457 [0.370, 0.564] | [0.427, 0.491] | 0.37 | 2.2% [0.0, 5.8] | 3 | 0/3/0 |
| dualforce_dcg_w1 | v2 | neutral | both | 26 | 0 | 15 | 0.614 [0.516, 0.714] | [0.520, 0.672] | 0.42 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_dcg_w1 | v2 | neutral | start | 80 | 0 | 41 | 0.446 [0.356, 0.574] | [0.422, 0.487] | 0.36 | 2.5% [0.0, 6.7] | 2 | 0/2/0 |
| dualforce_dcg_w1p5 | v2 | neutral | both | 26 | 0 | 15 | 0.678 [0.552, 0.772] | [0.560, 0.722] | 0.44 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_dcg_w1p5 | v2 | neutral | start | 80 | 0 | 41 | 0.498 [0.393, 0.595] | [0.424, 0.517] | 0.40 | 2.5% [0.0, 6.7] | 2 | 0/2/0 |
| dualforce_dcg_w3 | v2 | neutral | both | 26 | 0 | 15 | 0.765 [0.650, 0.832] | [0.674, 0.811] | 0.44 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_dcg_w3 | v2 | neutral | start | 80 | 0 | 41 | 0.532 [0.426, 0.672] | [0.497, 0.590] | 0.38 | 1.2% [0.0, 4.3] | 1 | 0/1/0 |
| dualforce_dcg_w6 | v2 | effect | both | 26 | 0 | 15 | 0.828 [0.705, 1.024] | [0.760, 0.968] | 0.43 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_dcg_w6 | v2 | effect | start | 79 | 0 | 41 | 0.605 [0.492, 0.742] | [0.559, 0.636] | 0.40 | 2.5% [0.0, 6.6] | 2 | 0/2/0 |
| dualforce_dcg_w6 | v2 | neutral | both | 26 | 0 | 15 | 0.835 [0.735, 1.079] | [0.810, 1.072] | 0.41 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_dcg_w6 | v2 | neutral | start | 80 | 0 | 41 | 0.604 [0.487, 0.756] | [0.565, 0.647] | 0.43 | 1.2% [0.0, 4.2] | 1 | 0/1/0 |
| dualforce_dcg_w6 | v3 | effect | both | 94 | 0 | 19 | 0.834 [0.745, 1.039] | [0.790, 0.949] | 0.39 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_dcg_w6 | v3 | effect | start | 278 | 0 | 61 | 0.621 [0.485, 0.755] | [0.582, 0.658] | 0.42 | 2.5% [0.0, 6.2] | 7 | 0/7/0 |
| dualforce_dcg_w6 | v3 | neutral | both | 94 | 0 | 19 | 0.865 [0.750, 1.066] | [0.818, 0.992] | 0.38 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_dcg_w6 | v3 | neutral | start | 278 | 0 | 61 | 0.612 [0.496, 0.745] | [0.565, 0.656] | 0.43 | 1.4% [0.0, 3.5] | 4 | 0/4/0 |
| dualforce_dcg_w6 | v3ed81 | effect | start | 136 | 0 | 51 | 0.573 [0.467, 0.670] | [0.528, 0.599] | 0.46 | 0.0% [0.0, 0.0] | 0 | 0/0/0 |
| dualforce_dcg_w6 | v3ed81 | neutral | start | 136 | 0 | 51 | 0.583 [0.470, 0.685] | [0.544, 0.623] | 0.44 | 0.7% [0.0, 2.3] | 1 | 0/0/1 |

## B. Cut-point sensitivity, base_cond (share of unique clean clips with DR ≤ θ)

| grid | prompt | cond | n | θ=0.06 | θ=0.08 | θ=0.1 | θ=0.12 | θ=0.14 | θ=0.16 | θ=0.18 | θ=0.2 | θ=0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 | effect | both | 52 | 0% | 4% | 4% | 6% | 8% | 10% | 13% | 13% | 27% |
| v2 | effect | start | 160 | 4% | 5% | 9% | 11% | 14% | 17% | 19% | 24% | 38% |
| v2 | neutral | both | 30 | 7% | 7% | 10% | 27% | 27% | 27% | 27% | 43% | 57% |
| v2 | neutral | start | 110 | 1% | 2% | 5% | 6% | 6% | 8% | 11% | 12% | 20% |
| v3 | effect | both | 94 | 0% | 4% | 5% | 9% | 11% | 12% | 15% | 16% | 26% |
| v3 | effect | start | 277 | 3% | 4% | 6% | 8% | 10% | 13% | 16% | 19% | 27% |
| v3 | neutral | both | 38 | 5% | 5% | 8% | 21% | 21% | 26% | 26% | 45% | 58% |
| v3 | neutral | start | 119 | 0% | 1% | 3% | 3% | 3% | 4% | 5% | 6% | 11% |
| v3ed81 | effect | start | 136 | 12% | 16% | 19% | 26% | 29% | 33% | 40% | 47% | 58% |
| v3ed81 | neutral | start | 32 | 41% | 44% | 50% | 53% | 56% | 56% | 69% | 75% | 78% |

## C. Paired contrasts (same endpoint and seed; unique clean clips)

ΔDR = first − second (positive = first is further from the endpoint line). Cliff's δ = mean sign(Δ). Flips = on-line in first only / on-line in second only, with exact sign test.

| contrast | pairs | ΔDR median [CI] | Cliff δ | Wilcoxon p | on-line share first → second | flips (p) | note |
|---|---|---|---|---|---|---|---|
| END ANCHOR: base both − base start-only (v2 neutral, tier-2 regen) | 30 (15 ep) | -0.052 [-0.173, 0.018] | -0.33 | 0.0197 | 26.7% → 10.0% | 5/0 (p=0.062) | causal probe; anchor is the only difference |
| PROMPT: base effect − base neutral (v2, both) | 52 (15 ep) | 0.139 [0.106, 0.174] | 0.73 | 0.0000 | 5.8% → 28.8% | 1/13 (p=0.002) | text describes the transition vs neutral |
| PROMPT: base effect − base neutral (v2, start) | 156 (40 ep) | -0.120 [-0.191, -0.046] | -0.37 | 0.0000 | 9.0% → 5.1% | 10/4 (p=0.180) | text describes the transition vs neutral |
| PROMPT: base effect − base neutral (v3, both) | 94 (19 ep) | 0.148 [0.091, 0.185] | 0.66 | 0.0000 | 8.5% → 26.6% | 2/19 (p=0.000) | text describes the transition vs neutral |
| PROMPT: base effect − base neutral (v3, start) | 271 (60 ep) | -0.119 [-0.160, -0.045] | -0.34 | 0.0000 | 6.3% → 3.0% | 14/5 (p=0.064) | text describes the transition vs neutral |
| PROMPT: base effect − base neutral (v3ed81, start) | 37 (22 ep) | 0.079 [0.003, 0.177] | 0.51 | 0.0023 | 32.4% → 56.8% | 3/12 (p=0.035) | text describes the transition vs neutral |
| TRAINING: base − dualforce_control (v2, neutral, both) | 52 (15 ep) | -0.394 [-0.453, -0.284] | -1.00 | 0.0000 | 28.8% → 0.0% | 15/0 (p=0.000) | same grid row; base duplicates kept so every trained clip has its partner |
| TRAINING: base − dualforce_control (v2, neutral, start) | 156 (40 ep) | 0.005 [-0.042, 0.074] | 0.03 | 0.7063 | 5.1% → 1.3% | 6/0 (p=0.031) | same grid row; base duplicates kept so every trained clip has its partner |
| TRAINING: base − dualforce_control (v2, effect, both) | 52 (15 ep) | -0.265 [-0.367, -0.163] | -0.85 | 0.0000 | 5.8% → 0.0% | 3/0 (p=0.250) | same grid row; base duplicates kept so every trained clip has its partner |
| TRAINING: base − dualforce_control (v2, effect, start) | 160 (41 ep) | -0.103 [-0.161, -0.062] | -0.44 | 0.0000 | 10.6% → 2.5% | 13/0 (p=0.000) | same grid row; base duplicates kept so every trained clip has its partner |
| TRAINING: base − dualforce_control (v3, neutral, both) | 94 (19 ep) | -0.394 [-0.453, -0.294] | -0.98 | 0.0000 | 26.6% → 0.0% | 25/0 (p=0.000) | same grid row; base duplicates kept so every trained clip has its partner |
| TRAINING: base − dualforce_control (v3, neutral, start) | 270 (59 ep) | -0.003 [-0.035, 0.043] | -0.01 | 0.9265 | 3.0% → 1.5% | 6/2 (p=0.289) | same grid row; base duplicates kept so every trained clip has its partner |
| TRAINING: base − dualforce_control (v3, effect, both) | 94 (19 ep) | -0.265 [-0.330, -0.199] | -0.79 | 0.0000 | 8.5% → 0.0% | 8/0 (p=0.008) | same grid row; base duplicates kept so every trained clip has its partner |
| TRAINING: base − dualforce_control (v3, effect, start) | 277 (61 ep) | -0.088 [-0.125, -0.069] | -0.44 | 0.0000 | 7.9% → 1.8% | 17/0 (p=0.000) | same grid row; base duplicates kept so every trained clip has its partner |
| TRAINING: base − dualforce_control (v3ed81, neutral, start) | 37 (22 ep) | -0.333 [-0.478, -0.220] | -0.95 | 0.0000 | 56.8% → 2.7% | 20/0 (p=0.000) | same grid row; base duplicates kept so every trained clip has its partner |
| TRAINING: base − dualforce_control (v3ed81, effect, start) | 136 (51 ep) | -0.217 [-0.252, -0.163] | -0.75 | 0.0000 | 26.5% → 2.2% | 34/1 (p=0.000) | same grid row; base duplicates kept so every trained clip has its partner |
| GUIDANCE w=1: dcg − control (v2, neutral, both) | 26 (15 ep) | 0.001 [-0.008, 0.005] | 0.08 | 0.8028 | 0.0% → 0.0% | 0/0 (p=nan) |  |
| GUIDANCE w=1: dcg − control (v2, neutral, start) | 80 (41 ep) | 0.003 [-0.001, 0.008] | 0.15 | 0.0691 | 2.5% → 2.5% | 0/0 (p=nan) |  |
| GUIDANCE w=1.5: dcg − control (v2, neutral, both) | 26 (15 ep) | 0.052 [0.028, 0.070] | 0.92 | 0.0000 | 0.0% → 0.0% | 0/0 (p=nan) |  |
| GUIDANCE w=1.5: dcg − control (v2, neutral, start) | 80 (41 ep) | 0.029 [0.019, 0.043] | 0.72 | 0.0000 | 2.5% → 2.5% | 0/0 (p=nan) |  |
| GUIDANCE w=3: dcg − control (v2, neutral, both) | 26 (15 ep) | 0.133 [0.111, 0.171] | 0.92 | 0.0000 | 0.0% → 0.0% | 0/0 (p=nan) |  |
| GUIDANCE w=3: dcg − control (v2, neutral, start) | 80 (41 ep) | 0.099 [0.071, 0.121] | 0.68 | 0.0000 | 1.2% → 2.5% | 0/1 (p=1.000) |  |
| GUIDANCE w=6: dcg − control (v2, neutral, both) | 26 (15 ep) | 0.211 [0.165, 0.337] | 1.00 | 0.0000 | 0.0% → 0.0% | 0/0 (p=nan) |  |
| GUIDANCE w=6: dcg − control (v2, neutral, start) | 80 (41 ep) | 0.157 [0.133, 0.178] | 0.88 | 0.0000 | 1.2% → 2.5% | 0/1 (p=1.000) |  |
| GUIDANCE w=6: dcg − control (v2, effect, both) | 26 (15 ep) | 0.173 [0.142, 0.238] | 0.92 | 0.0000 | 0.0% → 0.0% | 0/0 (p=nan) |  |
| GUIDANCE w=6: dcg − control (v2, effect, start) | 79 (41 ep) | 0.148 [0.116, 0.184] | 0.87 | 0.0000 | 2.5% → 2.5% | 0/0 (p=nan) |  |
| GUIDANCE w=6: dcg − control (v3, neutral, both) | 94 (19 ep) | 0.253 [0.181, 0.329] | 0.91 | 0.0000 | 0.0% → 0.0% | 0/0 (p=nan) |  |
| GUIDANCE w=6: dcg − control (v3, neutral, start) | 277 (61 ep) | 0.123 [0.093, 0.150] | 0.72 | 0.0000 | 1.4% → 2.5% | 0/3 (p=0.250) |  |
| GUIDANCE w=6: dcg − control (v3, effect, both) | 94 (19 ep) | 0.169 [0.147, 0.200] | 0.77 | 0.0000 | 0.0% → 0.0% | 0/0 (p=nan) |  |
| GUIDANCE w=6: dcg − control (v3, effect, start) | 277 (61 ep) | 0.129 [0.108, 0.149] | 0.78 | 0.0000 | 2.5% → 1.8% | 2/0 (p=0.500) |  |
| GUIDANCE w=6: dcg − control (v3ed81, neutral, start) | 136 (51 ep) | 0.114 [0.089, 0.140] | 0.74 | 0.0000 | 0.7% → 2.2% | 0/2 (p=0.500) |  |
| GUIDANCE w=6: dcg − control (v3ed81, effect, start) | 136 (51 ep) | 0.110 [0.091, 0.133] | 0.76 | 0.0000 | 0.0% → 2.2% | 0/3 (p=0.250) |  |

## D. Unpaired both-endpoint vs start-only within arm × grid × prompt (different endpoints per stratum)

| arm | grid | prompt | n both / start | DR median both / start | Δmedian [CI] | Cliff δ (MWU) | on-line both / start | Δshare [CI] |
|---|---|---|---|---|---|---|---|---|
| base_cond | v2 | effect | 52 / 160 | 0.385 / 0.310 | 0.075 [-0.040, 0.183] | 0.20 (p=0.0311) | 5.8% / 10.6% | -4.9 pp [-15.2, 10.5] |
| base_cond | v2 | neutral | 30 / 110 | 0.211 / 0.430 | -0.220 [-0.281, -0.130] | -0.54 (p=0.0000) | 26.7% / 6.4% | +20.3 pp [2.2, 42.2] |
| base_cond | v3 | effect | 94 / 277 | 0.385 / 0.337 | 0.048 [-0.041, 0.144] | 0.11 (p=0.1033) | 8.5% / 7.9% | +0.6 pp [-9.9, 13.6] |
| base_cond | v3 | neutral | 38 / 119 | 0.211 / 0.506 | -0.296 [-0.343, -0.202] | -0.69 (p=0.0000) | 21.1% / 3.4% | +17.7 pp [1.3, 35.3] |
| dualforce_control | v2 | effect | 52 / 160 | 0.692 / 0.446 | 0.246 [0.122, 0.321] | 0.61 (p=0.0000) | 0.0% / 2.5% | -2.5 pp [-6.5, 0.0] |
| dualforce_control | v2 | neutral | 52 / 160 | 0.642 / 0.457 | 0.185 [0.082, 0.262] | 0.58 (p=0.0000) | 0.0% / 2.5% | -2.5 pp [-6.7, 0.0] |
| dualforce_control | v3 | effect | 94 / 277 | 0.693 / 0.480 | 0.213 [0.128, 0.306] | 0.59 (p=0.0000) | 0.0% / 1.8% | -1.8 pp [-4.9, 0.0] |
| dualforce_control | v3 | neutral | 94 / 277 | 0.616 / 0.467 | 0.149 [0.084, 0.210] | 0.51 (p=0.0000) | 0.0% / 2.5% | -2.5 pp [-6.0, 0.0] |
| dualforce_dcg_w1 | v2 | neutral | 26 / 80 | 0.614 / 0.446 | 0.169 [0.065, 0.253] | 0.56 (p=0.0000) | 0.0% / 2.5% | -2.5 pp [-6.7, 0.0] |
| dualforce_dcg_w1p5 | v2 | neutral | 26 / 80 | 0.678 / 0.498 | 0.180 [0.067, 0.273] | 0.56 (p=0.0000) | 0.0% / 2.5% | -2.5 pp [-7.0, 0.0] |
| dualforce_dcg_w3 | v2 | neutral | 26 / 80 | 0.765 / 0.532 | 0.234 [0.123, 0.285] | 0.61 (p=0.0000) | 0.0% / 1.2% | -1.2 pp [-4.4, 0.0] |
| dualforce_dcg_w6 | v2 | effect | 26 / 79 | 0.828 / 0.605 | 0.223 [0.148, 0.392] | 0.59 (p=0.0000) | 0.0% / 2.5% | -2.5 pp [-6.3, 0.0] |
| dualforce_dcg_w6 | v2 | neutral | 26 / 80 | 0.835 / 0.604 | 0.230 [0.170, 0.453] | 0.62 (p=0.0000) | 0.0% / 1.2% | -1.2 pp [-4.4, 0.0] |
| dualforce_dcg_w6 | v3 | effect | 94 / 278 | 0.834 / 0.621 | 0.213 [0.148, 0.327] | 0.57 (p=0.0000) | 0.0% / 2.5% | -2.5 pp [-6.2, 0.0] |
| dualforce_dcg_w6 | v3 | neutral | 94 / 278 | 0.865 / 0.612 | 0.253 [0.191, 0.392] | 0.63 (p=0.0000) | 0.0% / 1.4% | -1.4 pp [-3.4, 0.0] |

## E. Per-endpoint collapse propensity, base_cond both-endpoint (pooling seeds and both prompts)

| grid | endpoints | ≥1 on-line gen | all gens on-line | median per-endpoint on-line share |
|---|---|---|---|---|
| v2 (both prompts) | 15 | 33% | 0% | 0% |
| v3 (both prompts) | 19 | 26% | 0% | 0% |
| v2 (neutral only, 2 seeds) | 15 | 33% | 20% | 0% |
| v3 (neutral only, 2 seeds) | 19 | 26% | 16% | 0% |

## F. Endpoint covariates vs confinement (base_cond, unique clean clips)

Spearman ρ of DR with each covariate (per grid × prompt × cond); DINO/CLIP distances are between the two anchor frames (GT clip frames where the real transition exists, else the conditioned frames of the generation).

| grid | prompt | cond | n (with pair cov) | ρ DINO dist | ρ CLIP dist | ρ pixel gap | ρ motion prefix | ρ motion suffix | on-line share by DINO tercile (low/mid/high) |
|---|---|---|---|---|---|---|---|---|---|
| v2 | effect | both | 52 | -0.38 (p=0.005) | -0.16 (p=0.260) | -0.28 (p=0.042) | 0.26 (p=0.058) | 0.35 (p=0.011) | 0%/0%/19% |
| v2 | effect | start | 160 | -0.02 (p=0.789) | -0.20 (p=0.010) | -0.11 (p=0.178) | 0.21 (p=0.007) | – | 14%/12%/6% |
| v2 | neutral | both | 30 | -0.48 (p=0.007) | -0.22 (p=0.244) | -0.41 (p=0.024) | 0.44 (p=0.016) | 0.39 (p=0.034) | 20%/10%/50% |
| v2 | neutral | start | 110 | -0.15 (p=0.130) | 0.03 (p=0.783) | -0.26 (p=0.005) | 0.36 (p=0.000) | – | 5%/6%/8% |
| v3 | effect | both | 94 | -0.19 (p=0.065) | 0.04 (p=0.695) | -0.29 (p=0.004) | 0.18 (p=0.085) | 0.34 (p=0.001) | 0%/6%/20% |
| v3 | effect | start | 277 | 0.06 (p=0.311) | -0.05 (p=0.376) | -0.07 (p=0.226) | 0.29 (p=0.000) | – | 13%/9%/1% |
| v3 | neutral | both | 38 | -0.39 (p=0.016) | -0.08 (p=0.634) | -0.43 (p=0.006) | 0.41 (p=0.011) | 0.38 (p=0.019) | 0%/25%/42% |
| v3 | neutral | start | 119 | 0.06 (p=0.532) | 0.00 (p=0.960) | -0.02 (p=0.831) | 0.47 (p=0.000) | – | 5%/0%/5% |

**Regression R1 — neutral prompt, both vs start:** DR = b0 + b1·[both] + b2·DINO + b3·motion_prefix (n=157 clips, 79 endpoints; cluster-bootstrap 95% CIs over endpoints)

| term | coef | 95% CI |
|---|---|---|
| intercept | +0.446 | [+0.340, +0.538] |
| both-endpoint | -0.226 | [-0.305, -0.146] |
| DINO dist | +0.027 | [-0.106, +0.171] |
| motion_prefix | +0.464 | [+0.227, +1.003] |

**Regression R2 — both prompts with interaction:** DR = b0 + b1·[both] + b2·[effect] + b3·[both×neutral] + b4·DINO + b5·motion_prefix (n=554 clips, 80 endpoints; cluster-bootstrap 95% CIs over endpoints)

| term | coef | 95% CI |
|---|---|---|
| intercept | +0.503 | [+0.383, +0.654] |
| both-endpoint | +0.097 | [-0.008, +0.214] |
| effect prompt | -0.099 | [-0.150, -0.034] |
| both × neutral | -0.312 | [-0.425, -0.218] |
| DINO dist | -0.052 | [-0.298, +0.118] |
| motion_prefix | +0.326 | [+0.093, +0.991] |

**Regression R3 — within base both-endpoint neutral:** DR = b0 + b1·DINO + b2·motion_prefix + b3·motion_suffix (n=38 clips, 19 endpoints; cluster-bootstrap 95% CIs over endpoints)

| term | coef | 95% CI |
|---|---|---|
| intercept | +0.466 | [+0.141, +0.854] |
| DINO dist | -0.342 | [-0.741, +0.066] |
| motion_prefix | +1.055 | [-0.706, +6.144] |
| motion_suffix | +0.742 | [-0.153, +1.810] |
