### Empirical check — DCG (null = pixel crossfade of the demo's endpoints, mechanism (a)) vs dualforce_control, neutral, both anchors, seed 42, paired by endpoint
| arm | n pairs | ΔDR med (dcg−ctrl) | n_pos/n_neg | ΔM med | n_pos/n_neg | ΔPR med | ctrl DR / M med | dcg DR / M med | GT PIX DR / M (S-GRID, 19) |
|---|---|---|---|---|---|---|---|---|---|
| dualforce_dcg_w1 | 56 | +0.001 | 29/27 | -0.010 | 24/29 | +0.02 | 0.647 / 0.433 | 0.646 / 0.442 | 0.604 / 0.404 |
| dualforce_dcg_w1p5 | 56 | +0.052 | 42/14 | +0.000 | 27/27 | +0.28 | 0.647 / 0.433 | 0.705 / 0.442 | 0.604 / 0.404 |
| dualforce_dcg_w3 | 56 | +0.123 | 46/10 | +0.029 | 32/21 | +0.43 | 0.647 / 0.433 | 0.757 / 0.442 | 0.604 / 0.404 |
| dualforce_dcg_w6 | 56 | +0.187 | 49/7 | +0.024 | 29/26 | +0.78 | 0.647 / 0.433 | 0.830 / 0.442 | 0.604 / 0.404 |
