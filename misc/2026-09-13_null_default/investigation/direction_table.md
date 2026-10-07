### S-GRID  (reference = GT, default = NULLGEN)
| space | descriptor | ref | default | LERP | verdict(LERP) | paired | LATLERP/DINOLERP | verdict | CUT50 verdict |
|---|---|---|---|---|---|---|---|---|---|
| PIX | DR | 0.604 | 0.239 | 0.000 | RIGHT | 1.00 | — | — | RIGHT |
| PIX | M | 0.404 | 0.048 | 0.500 | WRONG | 0.05 | — | — | RIGHT |
| PIX | cross | 0.452 | 0.404 | 0.500 | WRONG | 0.37 | — | — | WRONG |
| PIX | nu_max | 1.063 | 0.783 | 0.495 | RIGHT | 0.90 | — | — | RIGHT |
| PIX | path_over_gap | 22.128 | 7.593 | 1.000 | RIGHT | 0.95 | — | — | RIGHT |
| PIX | step_share | 0.025 | 0.081 | 0.009 | WRONG | 0.05 | — | — | RIGHT |
| DINO | DR | 0.511 | 0.269 | 0.454 | WRONG(N between default and ref) | 0.00 | 0.000 (DINOLERP) | RIGHT | RIGHT |
| DINO | M | 0.356 | 0.072 | 0.375 | WRONG | 0.05 | 0.500 (DINOLERP) | WRONG | RIGHT |
| DINO | cross | 0.471 | 0.394 | 0.577 | WRONG | 0.32 | 0.500 (DINOLERP) | WRONG | WRONG |
| DINO | nu_max | 0.974 | 0.913 | 0.791 | RIGHT | 0.32 | 0.495 (DINOLERP) | RIGHT | RIGHT |
| DINO | path_over_gap | 20.839 | 13.677 | 3.538 | RIGHT | 0.84 | 1.000 (DINOLERP) | RIGHT | RIGHT |
| DINO | step_share | 0.027 | 0.059 | 0.025 | WRONG | 0.16 | 0.009 (DINOLERP) | WRONG | RIGHT |
| VAE | DR | 0.741 | 0.585 | 0.279 | RIGHT | 0.79 | 0.000 (LATLERP) | RIGHT | RIGHT |
| VAE | M | 0.692 | 0.385 | 0.615 | WRONG(N between default and ref) | 0.21 | 0.538 (LATLERP) | WRONG(N between default and ref) | RIGHT |
| VAE | cross | 0.462 | 0.423 | 0.538 | WRONG | 0.26 | 0.538 (LATLERP) | WRONG | WRONG |
| VAE | nu_max | 0.972 | 0.897 | 0.573 | RIGHT | 0.63 | 0.500 (LATLERP) | RIGHT | RIGHT |
| VAE | path_over_gap | 9.002 | 7.003 | 3.481 | RIGHT | 0.95 | 1.000 (LATLERP) | RIGHT | RIGHT |
| VAE | step_share | 0.093 | 0.127 | 0.081 | WRONG | 0.10 | 0.071 (LATLERP) | WRONG | RIGHT |
| TRANS | nu_max | 0.495 | 0.336 | 0.283 | RIGHT | 0.47 | — | — | RIGHT |
| TRANS | swap_sharp | 0.232 | 0.566 | 0.124 | WRONG | 0.00 | — | — | RIGHT |
| TRANS | local_at_peak | 0.333 | 0.577 | 0.000 | WRONG | 0.21 | — | — | RIGHT |
| TRANS | trans_mean | 0.074 | 0.059 | 0.057 | no-gap(|ref-def|<0.02) | 0.53 | — | — | no-gap(|ref-def|<0.02) |

### S-PROBE high (reference = R1, default = R3)
| space | descriptor | ref | default | LERP | verdict(LERP) | paired | LATLERP/DINOLERP | verdict | CUT50 verdict |
|---|---|---|---|---|---|---|---|---|---|
| PIX | DR | 0.367 | 0.206 | 0.000 | RIGHT | 0.87 | — | — | RIGHT |
| PIX | M | 0.225 | 0.058 | 0.500 | WRONG | 0.09 | — | — | RIGHT |
| PIX | cross | 0.338 | 0.279 | 0.500 | WRONG | 0.39 | — | — | WRONG |
| PIX | nu_max | 0.819 | 0.713 | 0.495 | RIGHT | 0.63 | — | — | RIGHT |
| PIX | path_over_gap | 9.475 | 5.035 | 1.000 | RIGHT | 0.83 | — | — | RIGHT |
| PIX | step_share | 0.028 | 0.086 | 0.009 | WRONG | 0.07 | — | — | RIGHT |
| DINO | DR | 0.300 | 0.181 | 0.382 | WRONG | 0.14 | 0.000 (DINOLERP) | RIGHT | RIGHT |
| DINO | M | 0.288 | 0.048 | 0.260 | WRONG(N between default and ref) | 0.07 | 0.500 (DINOLERP) | WRONG | RIGHT |
| DINO | cross | 0.342 | 0.284 | 0.471 | WRONG | 0.28 | 0.500 (DINOLERP) | WRONG | WRONG |
| DINO | nu_max | 0.975 | 0.818 | 0.782 | RIGHT | 0.48 | 0.495 (DINOLERP) | RIGHT | RIGHT |
| DINO | path_over_gap | 18.711 | 8.928 | 3.244 | RIGHT | 0.98 | 1.000 (DINOLERP) | RIGHT | RIGHT |
| DINO | step_share | 0.030 | 0.071 | 0.025 | WRONG | 0.03 | 0.009 (DINOLERP) | WRONG | RIGHT |
| VAE | DR | 0.679 | 0.617 | 0.269 | RIGHT | 0.66 | 0.000 (LATLERP) | RIGHT | RIGHT |
| VAE | M | 0.615 | 0.308 | 0.615 | WRONG | 0.07 | 0.538 (LATLERP) | WRONG(N between default and ref) | RIGHT |
| VAE | cross | 0.462 | 0.308 | 0.462 | WRONG | 0.16 | 0.538 (LATLERP) | WRONG | WRONG |
| VAE | nu_max | 0.923 | 0.859 | 0.571 | RIGHT | 0.67 | 0.500 (LATLERP) | RIGHT | RIGHT |
| VAE | path_over_gap | 7.126 | 6.554 | 3.184 | RIGHT | 0.72 | 1.000 (LATLERP) | RIGHT | RIGHT |
| VAE | step_share | 0.098 | 0.123 | 0.081 | WRONG | 0.11 | 0.071 (LATLERP) | WRONG | RIGHT |
| TRANS | nu_max | 0.583 | 0.307 | 0.283 | RIGHT | 0.57 | — | — | RIGHT |
| TRANS | swap_sharp | 0.287 | 0.530 | 0.155 | WRONG | 0.12 | — | — | RIGHT |
| TRANS | local_at_peak | 0.338 | 0.587 | 0.000 | WRONG | 0.24 | — | — | RIGHT |
| TRANS | trans_mean | 0.053 | 0.053 | 0.056 | no-gap(|ref-def|<0.02) | 0.47 | — | — | no-gap(|ref-def|<0.02) |

### S-PROBE inplace (reference = R1, default = R3)
| space | descriptor | ref | default | LERP | verdict(LERP) | paired | LATLERP/DINOLERP | verdict | CUT50 verdict |
|---|---|---|---|---|---|---|---|---|---|
| PIX | DR | 0.278 | 0.286 | 0.000 | no-gap(|ref-def|<0.02) | 0.73 | — | — | no-gap(|ref-def|<0.02) |
| PIX | M | 0.239 | 0.183 | 0.500 | WRONG | 0.33 | — | — | RIGHT |
| PIX | cross | 0.333 | 0.351 | 0.500 | no-gap(|ref-def|<0.02) | 0.37 | — | — | no-gap(|ref-def|<0.02) |
| PIX | nu_max | 0.718 | 0.750 | 0.495 | WRONG | 0.50 | — | — | WRONG |
| PIX | path_over_gap | 5.357 | 4.909 | 1.000 | RIGHT | 0.60 | — | — | RIGHT |
| PIX | step_share | 0.032 | 0.048 | 0.009 | no-gap(|ref-def|<0.02) | 0.30 | — | — | no-gap(|ref-def|<0.02) |
| DINO | DR | 0.464 | 0.429 | 0.442 | negligible(|def-N|<0.02) | 0.37 | 0.000 (DINOLERP) | RIGHT | RIGHT |
| DINO | M | 0.257 | 0.202 | 0.351 | WRONG | 0.20 | 0.500 (DINOLERP) | WRONG | RIGHT |
| DINO | cross | 0.329 | 0.332 | 0.673 | no-gap(|ref-def|<0.02) | 0.63 | 0.500 (DINOLERP) | no-gap(|ref-def|<0.02) | no-gap(|ref-def|<0.02) |
| DINO | nu_max | 0.850 | 0.850 | 0.815 | no-gap(|ref-def|<0.02) | 0.37 | 0.495 (DINOLERP) | no-gap(|ref-def|<0.02) | no-gap(|ref-def|<0.02) |
| DINO | path_over_gap | 19.078 | 15.313 | 3.451 | RIGHT | 1.00 | 1.000 (DINOLERP) | RIGHT | RIGHT |
| DINO | step_share | 0.025 | 0.033 | 0.032 | no-gap(|ref-def|<0.02) | 0.30 | 0.009 (DINOLERP) | no-gap(|ref-def|<0.02) | no-gap(|ref-def|<0.02) |
| VAE | DR | 0.576 | 0.607 | 0.263 | WRONG | 0.40 | 0.000 (LATLERP) | WRONG | WRONG |
| VAE | M | 0.462 | 0.385 | 0.615 | WRONG | 0.10 | 0.538 (LATLERP) | WRONG | RIGHT |
| VAE | cross | 0.308 | 0.308 | 0.538 | no-gap(|ref-def|<0.02) | 0.27 | 0.538 (LATLERP) | no-gap(|ref-def|<0.02) | no-gap(|ref-def|<0.02) |
| VAE | nu_max | 0.818 | 0.831 | 0.565 | no-gap(|ref-def|<0.02) | 0.40 | 0.500 (LATLERP) | no-gap(|ref-def|<0.02) | no-gap(|ref-def|<0.02) |
| VAE | path_over_gap | 6.419 | 6.681 | 3.252 | WRONG | 0.40 | 1.000 (LATLERP) | WRONG | WRONG |
| VAE | step_share | 0.096 | 0.103 | 0.086 | no-gap(|ref-def|<0.02) | 0.30 | 0.071 (LATLERP) | no-gap(|ref-def|<0.02) | no-gap(|ref-def|<0.02) |
| TRANS | nu_max | 0.231 | 0.195 | 0.208 | negligible(|def-N|<0.02) | 0.40 | — | — | RIGHT |
| TRANS | swap_sharp | 0.214 | 0.299 | 0.153 | WRONG | 0.23 | — | — | RIGHT |
| TRANS | local_at_peak | 0.028 | 0.017 | 0.000 | no-gap(|ref-def|<0.02) | 0.23 | — | — | no-gap(|ref-def|<0.02) |
| TRANS | trans_mean | 0.061 | 0.061 | 0.058 | no-gap(|ref-def|<0.02) | 0.50 | — | — | no-gap(|ref-def|<0.02) |
