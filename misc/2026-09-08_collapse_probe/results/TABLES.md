# Collapse-to-the-endpoint-line probe — TABLES

_n clips scored: 360 (120 R1 / 120 R2 / 120 R3)_

## DR_med by run x tier

| run | tier | median | IQR (25-75) | n |
|---|---|---|---|---|
| R1 | high | 0.3671 | [0.2336, 0.4990] | 90 |
| R1 | low | 0.2781 | [0.2286, 0.3877] | 30 |
| R2 | high | 0.3344 | [0.2335, 0.4477] | 90 |
| R2 | low | 0.2914 | [0.2078, 0.3595] | 30 |
| R3 | high | 0.2063 | [0.1115, 0.2908] | 90 |
| R3 | low | 0.2859 | [0.1727, 0.3380] | 30 |

## Paired DR deltas (per prompt x seed)

| delta | tier | median | n_pos | n_neg | n_zero | n | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| dDR_R2_minus_R1 | high | -0.0079 | 35 | 55 | 0 | 90 | 0.013 |
| dDR_R2_minus_R1 | low | -0.0247 | 10 | 20 | 0 | 30 | 0.0327 |
| dDR_R2_minus_R1 | all | -0.0103 | 45 | 75 | 0 | 120 | 0.00153 |
| dDR_R3_minus_R1 | high | -0.1000 | 12 | 78 | 0 | 90 | 9.63e-14 |
| dDR_R3_minus_R1 | low | -0.0288 | 8 | 22 | 0 | 30 | 0.00403 |
| dDR_R3_minus_R1 | all | -0.0621 | 20 | 100 | 0 | 120 | 1e-15 |
| dDR_R3_minus_R2 | high | -0.0703 | 11 | 79 | 0 | 90 | 5.55e-14 |
| dDR_R3_minus_R2 | low | -0.0049 | 7 | 23 | 0 | 30 | 0.0405 |
| dDR_R3_minus_R2 | all | -0.0349 | 18 | 102 | 0 | 120 | 3.27e-15 |

## Realized scene change (R1 DINOv2 CLS distance) by tier

| tier | median | IQR (25-75) | n |
|---|---|---|---|
| high | 0.9777 | [0.9582, 1.0075] | 90 |
| low | 0.4505 | [0.2902, 0.6371] | 30 |
| all | 0.9638 | [0.7118, 1.0005] | 120 |

