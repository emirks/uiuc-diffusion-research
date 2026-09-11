
## G. Endpoint distance vs confinement (added 2026-09-11)

Distances between the two anchor frames every clip is scored against (frame 8 = start anchor, frame 120 = R1's end): DINO = 1−cos of DINOv2 CLS (the eval's `realized_dino`), CLIP = 1−cos of ViT-B/32 image embeddings, pixel = anchor-to-anchor L2 in instrument units (`gap_rel`). Spearman ρ, negative = more distant endpoints → lower DR (closer to the endpoint line). Script `scripts/distance_corr.py`; per-pair values in `results/paired_with_distances.csv`, CLIP cache `results/clip_dist.csv`.

| tier | DINO median [IQR] | CLIP | pixel |
|---|---|---|---|
| scene change (90) | 0.98 [0.96, 1.01] | 0.50 [0.45, 0.57] | 0.93 [0.86, 1.03] |
| in-place (30) | 0.45 [0.29, 0.64] | 0.22 [0.16, 0.29] | 0.75 [0.67, 0.90] |

| subset | outcome | ρ DINO | ρ CLIP | ρ pixel |
|---|---|---|---|---|
| all 120 | R3 DR (both anchors, captions only) | −0.24 (p=0.009) | −0.19 (p=0.034) | −0.34 (p<0.001) |
| all 120 | R2 DR (both anchors, full text) | −0.08 (p=0.41) | −0.02 (p=0.80) | −0.26 (p=0.004) |
| all 120 | R1 DR (start only, full text) | −0.07 (p=0.43) | −0.01 (p=0.90) | −0.23 (p=0.013) |
| all 120 | ΔDR R3−R2 (text removed) | −0.24 (p=0.009) | −0.28 (p=0.002) | +0.07 (p=0.43) |
| all 120 | ΔDR R2−R1 (anchor added) | +0.06 (p=0.48) | +0.10 (p=0.27) | −0.16 (p=0.073) |
| scene change (90) | R3 DR | −0.12 (p=0.26) | −0.04 (p=0.68) | −0.29 (p=0.005) |
| scene change (90) | R2 DR | −0.33 (p=0.001) | −0.21 (p=0.049) | −0.32 (p=0.002) |
| scene change (90) | R1 DR | −0.36 (p<0.001) | −0.22 (p=0.041) | −0.31 (p=0.003) |
| scene change (90) | ΔDR R3−R2 | +0.16 (p=0.14) | +0.10 (p=0.37) | +0.19 (p=0.075) |
| in-place (30) | R3 DR | +0.23 (p=0.23) | +0.20 (p=0.30) | −0.18 (p=0.34) |
| in-place (30) | R2 DR | +0.18 (p=0.33) | +0.15 (p=0.43) | −0.26 (p=0.16) |

R3 by distance tercile, pooled 120 (on-line share = DR ≤ 0.12; ΔDR = R3−R2 median):

| distance | near | mid | far |
|---|---|---|---|
| DINO | 5% · DR 0.29 · Δ −0.01 | 28% · 0.21 · −0.11 | 30% · 0.20 · −0.05 |
| CLIP | 10% · 0.29 · −0.01 | 30% · 0.17 · −0.09 | 22% · 0.22 · −0.06 |
| pixel | 5% · 0.28 · −0.05 | 25% · 0.22 · −0.03 | 32% · 0.16 · −0.03 |

Within scene change, R3 by CLIP tercile (CLIP 0.20–0.46 / 0.46–0.54 / 0.54–0.68): on-line share 33% / 20% / 30%, DR median 0.16 / 0.24 / 0.20 — no gradient.

Reading (attributed, this campaign): pooled over both tiers the far endpoints collapse more in R3 (DINO/CLIP terciles: on-line share 5–10% near vs 22–30% mid/far), but that pooled relation is carried by the tier contrast (scene-change vs in-place), which is also the text-off contrast. Within scene-change endpoints the semantic distance (DINO saturated at ~1, CLIP 0.2–0.7) does not order the collapse at all; the pixel gap does weakly (ρ ≈ −0.3) in all three runs, R1 included, and the pixel-gap relation is partly arithmetic (DR is normalised by that gap). The text-removed delta ΔDR(R3−R2) is more negative for semantically distant pairs pooled (ρ −0.24 DINO, −0.28 CLIP) and unrelated to the pixel gap. Same picture as the grid re-measure Table F (neutral/both ρ DINO −0.48/−0.39, ρ pixel −0.41/−0.43, ρ CLIP n.s.), which likewise pools scene-change and in-place rows.
