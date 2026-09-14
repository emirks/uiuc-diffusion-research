## Extra numbers

### Swap duration: interior time with τ ∈ (0.2, 0.8), as a fraction of the interior (PIX/DINO: ×104 frames; VAE ×13 latents); per-clip spread = std of τ(s=0.5)

| space | group | n | transit frac med [IQR] | ≈ frames | share with transit ≤ 10% of interior | std τ(0.5) across clips | std ρ(0.5) |
|---|---|---|---|---|---|---|---|
| PIX | S-GRID:NULLGEN | 68 | 0.08 [0.04, 0.17] | 9 | 0.60 | 0.28 | 0.19 |
| PIX | S-GRID:GT | 19 | 0.48 [0.36, 0.56] | 50 | 0.00 | 0.17 | 0.33 |
| PIX | S-GRID-F:NULLGEN | 38 | 0.10 [0.02, 0.40] | 11 | 0.50 | 0.34 | 0.31 |
| PIX | S-PROBE:R3:high | 90 | 0.08 [0.04, 0.16] | 9 | 0.62 | 0.18 | 0.16 |
| PIX | S-PROBE:R1:high | 90 | 0.29 [0.17, 0.42] | 30 | 0.11 | 0.28 | 0.34 |
| PIX | S-PROBE:R2:high | 90 | 0.25 [0.15, 0.38] | 26 | 0.17 | 0.26 | 0.33 |
| PIX | S-PROBE:R3:inplace | 30 | 0.22 [0.12, 0.31] | 23 | 0.20 | 0.25 | 0.16 |
| PIX | S-SWEEP:A_empty | 10 | 0.34 [0.24, 0.44] | 36 | 0.10 | 0.26 | 0.12 |
| PIX | LM:LERP | 19 | 0.58 | 61 | 0.00 | 0.00 | 0.00 |
| PIX | LM:CUT50 | 19 | 0.00 | 0 | 1.00 | 0.00 | 0.00 |
| DINO | S-GRID:NULLGEN | 68 | 0.08 [0.06, 0.17] | 9 | 0.59 | 0.31 | 0.22 |
| DINO | S-GRID:GT | 19 | 0.40 [0.33, 0.51] | 41 | 0.05 | 0.15 | 0.16 |
| DINO | S-GRID-F:NULLGEN | 38 | 0.06 [0.04, 0.14] | 6 | 0.66 | 0.38 | 0.18 |
| DINO | S-PROBE:R3:high | 90 | 0.06 [0.04, 0.10] | 6 | 0.72 | 0.19 | 0.15 |
| DINO | S-PROBE:R1:high | 90 | 0.33 [0.21, 0.45] | 35 | 0.03 | 0.21 | 0.24 |
| DINO | S-PROBE:R2:high | 90 | 0.33 [0.21, 0.42] | 35 | 0.11 | 0.21 | 0.25 |
| DINO | S-PROBE:R3:inplace | 30 | 0.31 [0.15, 0.47] | 32 | 0.17 | 0.26 | 0.11 |
| DINO | S-SWEEP:A_empty | 10 | 0.28 [0.05, 0.51] | 29 | 0.30 | 0.34 | 0.21 |
| DINO | LM:LERP | 19 | 0.44 | 46 | 0.00 | 0.16 | 0.12 |
| DINO | LM:CUT50 | 19 | 0.00 | 0 | 1.00 | 0.00 | 0.00 |
| VAE | S-GRID:NULLGEN | 68 | 0.44 [0.27, 0.58] | 6 | 0.03 | 0.20 | 0.12 |
| VAE | S-GRID:GT | 19 | 0.71 [0.59, 0.77] | 9 | 0.00 | 0.10 | 0.08 |
| VAE | S-GRID-F:NULLGEN | 38 | 0.76 [0.22, 0.81] | 10 | 0.08 | 0.22 | 0.15 |
| VAE | S-PROBE:R3:high | 90 | 0.50 [0.29, 0.67] | 6 | 0.02 | 0.15 | 0.14 |
| VAE | S-PROBE:R1:high | 90 | 0.64 [0.46, 0.73] | 8 | 0.00 | 0.16 | 0.13 |
| VAE | S-PROBE:R2:high | 90 | 0.64 [0.48, 0.71] | 8 | 0.00 | 0.16 | 0.14 |
| VAE | S-PROBE:R3:inplace | 30 | 0.54 [0.33, 0.60] | 7 | 0.00 | 0.15 | 0.10 |
| VAE | S-SWEEP:A_empty | 10 | 0.86 [0.80, 0.88] | 11 | 0.00 | 0.15 | 0.07 |
| VAE | LM:LERP | 19 | 0.65 | 8 | 0.00 | 0.04 | 0.02 |
| VAE | LM:CUT50 | 19 | 0.08 | 1 | 1.00 | 0.02 | 0.06 |
| VAE | LM:LATLERP | 19 | 0.58 | 8 | 0.00 | 0.00 | 0.00 |

### Crossing-aligned mean curves (τ=0.5 at s=0.5): τ at aligned s = 0.3 / 0.4 / 0.6 / 0.7; ρ at aligned s = 0.5; ν at aligned 0.5

| space | group | τ_al(0.3/0.4/0.6/0.7) | ρ_al(0.5) | ν_al(0.5) | n aligned |
|---|---|---|---|---|---|
| PIX | S-GRID:NULLGEN | 0.06 / 0.11 / 0.88 / 0.95 | 0.64 | 0.72 | 68 |
| PIX | S-GRID:GT | 0.19 / 0.32 / 0.66 / 0.73 | 0.84 | 0.94 | 19 |
| PIX | S-PROBE:R3:high | 0.08 / 0.15 / 0.92 / 0.95 | 0.51 | 0.61 | 90 |
| PIX | S-PROBE:R1:high | 0.18 / 0.28 / 0.78 / 0.88 | 0.68 | 0.81 | 90 |
| PIX | S-GRID:LM:LERP | 0.30 / 0.41 / 0.60 / 0.70 | 0.00 | 0.48 | 19 |
| PIX | S-GRID:LM:CUT50 | 0.00 / 0.00 / 1.00 / 1.00 | 0.00 | 0.00 | 19 |
| DINO | S-GRID:NULLGEN | 0.10 / 0.15 / 0.87 / 0.95 | 0.68 | 0.76 | 68 |
| DINO | S-GRID:GT | 0.25 / 0.36 / 0.67 / 0.78 | 0.82 | 0.93 | 19 |
| DINO | S-PROBE:R3:high | 0.06 / 0.12 / 0.96 / 0.97 | 0.55 | 0.62 | 90 |
| DINO | S-PROBE:R1:high | 0.22 / 0.33 / 0.71 / 0.90 | 0.76 | 0.87 | 90 |
| DINO | S-GRID:LM:LERP | 0.21 / 0.34 / 0.66 / 0.78 | 0.62 | 0.78 | 19 |
| DINO | S-GRID:LM:CUT50 | 0.00 / 0.00 / 1.00 / 1.00 | 0.00 | 0.00 | 19 |
| VAE | S-GRID:NULLGEN | 0.25 / 0.35 / 0.76 / 0.81 | 0.76 | 0.84 | 68 |
| VAE | S-GRID:GT | 0.33 / 0.42 / 0.62 / 0.68 | 0.82 | 0.94 | 19 |
| VAE | S-PROBE:R3:high | 0.24 / 0.39 / 0.75 / 0.78 | 0.71 | 0.81 | 90 |
| VAE | S-PROBE:R1:high | 0.33 / 0.43 / 0.66 / 0.74 | 0.75 | 0.87 | 90 |
| VAE | S-GRID:LM:LERP | 0.34 / 0.45 / 0.61 / 0.70 | 0.30 | 0.55 | 19 |
| VAE | S-GRID:LM:CUT50 | 0.00 / 0.20 / 0.99 / 0.99 | 0.23 | 0.23 | 19 |
| VAE | S-GRID:LM:LATLERP | 0.32 / 0.44 / 0.63 / 0.73 | 0.00 | 0.45 | 19 |

### TRANS mean curves at s = 0.25 / 0.5 / 0.75 of the latent window

| group | swap progress | nu_t | local_t | inplace share/step | n |
|---|---|---|---|---|---|
| S-GRID:NULLGEN | 0.32 / 0.73 / 0.94 | 0.14 / 0.14 / 0.09 | 0.10 / 0.07 / 0.02 | 0.08 / 0.08 / 0.05 | 68 |
| S-GRID:GT | 0.23 / 0.59 / 0.89 | 0.26 / 0.42 / 0.15 | 0.24 / 0.16 / 0.05 | 0.08 / 0.09 / 0.06 | 19 |
| S-PROBE:R3:high | 0.32 / 0.87 / 0.96 | 0.17 / 0.12 / 0.07 | 0.18 / 0.03 / 0.00 | 0.10 / 0.06 / 0.05 | 90 |
| S-PROBE:R1:high | 0.28 / 0.62 / 0.89 | 0.33 / 0.41 / 0.15 | 0.11 / 0.09 / 0.04 | 0.09 / 0.08 / 0.05 | 90 |
| S-PROBE:R3:inplace | 0.25 / 0.59 / 0.83 | 0.13 / 0.15 / 0.11 | 0.03 / 0.01 / 0.01 | 0.08 / 0.07 / 0.06 | 30 |
| S-GRID:LM:LERP | 0.20 / 0.42 / 0.69 | 0.16 / 0.24 / 0.20 | 0.00 / 0.00 / 0.00 | 0.05 / 0.06 / 0.07 | 19 |
| S-GRID:LM:CUT50 | 0.00 / 0.00 / 1.00 | -0.00 / -0.00 / -0.00 | 0.00 / 0.64 / 0.00 | 0.00 / 1.00 / 0.00 | 19 |

### PIX, the 19 S-GRID endpoints with GT twins: GT vs base default vs trained control vs DCG w6 (Sep-08 re-measure rows, both anchors, neutral, non-foreign, md5-dedup, medians over seeds then endpoints)

| arm | n endpoints | DR med | M med | path/gap or PR med | paired ΔDR vs GT med (n_pos/n_neg) | paired ΔM vs GT med |
|---|---|---|---|---|---|---|
| GT twin | 19 | 0.604 | 0.404 | 22.13 | — | — |
| base_cond neutral (NULLGEN) | 19 | 0.239 | 0.048 | 7.59 | -0.366 (0/19) | -0.279 |
| dualforce_control | 19 | 0.575 | 0.351 | 4.07 | -0.019 (8/11) | -0.034 |
| dualforce_dcg_w1 | 15 | 0.559 | 0.413 | 3.89 | -0.037 (7/8) | -0.058 |
| dualforce_dcg_w1p5 | 15 | 0.627 | 0.409 | 4.49 | +0.001 (8/7) | -0.029 |
| dualforce_dcg_w3 | 15 | 0.774 | 0.442 | 4.48 | +0.115 (13/2) | +0.019 |
| dualforce_dcg_w6 | 19 | 0.942 | 0.346 | 4.77 | +0.296 (19/0) | +0.014 |
