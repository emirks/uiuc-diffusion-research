
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

## H. Which member of the null family? cut vs dissolve on the on-line clips (added 2026-09-11)

On-line = DR ≤ 0.12. Inside the endpoint line the mid-band share M separates the members: M ≈ 0 = frames sit at the two anchors (a cut), M ≈ 0.5 = frames spread along the line (a dissolve). τ = projection coordinate of a frame on the A→B line (0 = at the start anchor, 1 = at the end anchor); τ_med = median τ over the interior frames.

| tier | run | cut-like (M<0.15) | mixed | off-line |
|---|---|---|---|---|
| scene change | R1 start only, full prompt | 1% | 6% | 93% |
| scene change | R2 both anchors, full prompt | 2% | 4% | 93% |
| scene change | R3 both anchors, captions only | 26% | 2% | 72% |
| in-place | R1 / R2 / R3 | 0% | 0% | 100% |

Among the on-line clips: R3 scene change n=25, M median 0.05 [0.02, 0.07], 92% cut-like, 0% dissolve-like (M>0.30); τ_med of the cut-like clips median 0.99 (IQR 0.87–1.00): the end scene dominates the interior. Per-frame τ profiles (recomputed 2026-09-12 with the same instrument): the start scene is held for the first third, the crossing (first τ > 0.5) happens at 0.34 of the interior (median, IQR 0.23–0.40), and the end scene is held for the remaining two thirds (near-A share 0.30, near-B share 0.64; mean 10-bin profile 0.03 0.14 0.21 0.54 0.77 0.91 0.98 0.99 1.00 1.00). No on-line clip holds the start scene to the end (0 of 25 with near-A share > 0.8); 3 of 25 jump almost immediately. The 65 off-line R3 scene-change clips have the same shape (crossing at 0.27, near-B share 0.69) at residual DR 0.19–0.33: they also reach the end scene early, but the crossing or the holds carry off-line content. R1/R2 on-line clips (n=6 each) have M ≈ 0.2, partial blends rather than clean cuts.

Distance does not move the kind within scene change: ρ(M, DINO) −0.02, ρ(M, CLIP) −0.20 (p=0.06), ρ(M, pixel) −0.04 over the 90 R3 clips; cut-like share by CLIP tercile 30% / 17% / 30%.

Reading (attributed, this campaign): base LTX-2's operator-unaware default on scene-change endpoints is a cut at about one third of the clip (hold A briefly, cut, hold B), not the dissolve. Both are members of the interpolation family (α(t) a step at t≈1 vs linear) and both sit on the endpoint line, so DR alone calls them the same thing; M and τ tell them apart. No dissolve-like on-line clip was produced by the base model in this probe. The dissolve member has not been observed in any current-state arm; whether an operator-blind fine-tune lands there is untested.
