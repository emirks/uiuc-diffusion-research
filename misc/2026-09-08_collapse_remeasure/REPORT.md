# Null-family ("lerp collapse") re-measurement on existing store generations — REPORT

**Date:** 2026-09-08 · **Owner request:** re-measure the collapse observation behind paper Table 1 on the
generations already in the store (base_cond, dualforce, DCG), with continuous statistics instead of a
threshold, endpoint-distance covariates, and no new generation. · **Compute:** DeltaAI jobs 3114599 (scoring,
96 s) and 3114640 (covariates), 1 GPU slice each on `bgjg-dtai-gh`; analysis on the login node.
**Files:** `instrument.py` · `score_store.py` · `covariates.py` · `analyze.py` · `per_clip.csv` (raw layer, one row
per video) · `endpoint_covariates.csv` · `TABLES.md` + `summary.json` (all numbers) · `figs/fig1..4`.
Numbers below are levels; every reading is this campaign's and is marked as such.

## 1. Why re-measure

The Aug-24 proxy (`misc/2026-08-24_lerp_collapse`) that feeds paper Table 1 has two problems found in audit:

1. **Duplicates.** The base model receives no reference, and the neutral prompt is the start-scene caption
   only, so grid rows that share a target endpoint produce byte-identical videos (md5-verified: the three
   hero_flight_5 rows are one file, the three shadow_smoke_7 rows are one file, …). Table 1's n=52 is 30 unique
   generations over 15 endpoint pairs; the paired McNemar "10 vs 0, p<0.001" is 5 unique flips vs 0
   (p=0.031 one-sided, 0.062 two-sided). Point estimates barely move (28.8% → 26.7%).
2. **Threshold fragility.** The 15 counted cuts have residuals 0.104–0.119 against a cut point of 0.12
   borrowed from the static-clip guard; at 0.10 the count is 4/52. The DISSOLVE class also required
   Spearman monotonicity of exactly 1.0, unreachable for noisy outputs.

## 2. Method

**Instrument** (`instrument.py`, pixel space 128×128, blur σ=1, same geometry as Aug-24). For each clip,
a = last conditioned start frame, b = first conditioned end frame (or the final generated frame when there is no
end anchor), interior = frames strictly between. Per clip: confinement residual **DR** = median over interior
frames of the normalised off-line residual ‖(v_t−a) − proj_{b−a}(v_t−a)‖/‖b−a‖ (0 = a frame-wise blend of the
endpoints, i.e. inside the family of paper Eq. (lerp)); DR_mean; **online_frac** = share of interior frames with
residual ≤ 0.12; schedule descriptors on the projection coordinate τ: **M** = share of frames with τ∈[0.25,0.75]
(dissolve 0.50, cut/freeze 0.00), R = p95−p5 of τ, S = Spearman(τ, t); gap and gap_rel; motion energy inside the
prefix / suffix windows (nuisance: a held shot with camera motion reads as off-line). Windows: 121-f LTX arms
prefix 9 / suffix 8; EffectData 81-f tier prefix 1 (frame-0 anchor, one-sided only).
**Descriptive labels** (not used for inference): on-line = DR ≤ 0.12 and not STATIC; among on-line clips
DISSOLVE if M ≥ 0.25, CUT if R ≥ 0.5, else FREEZE. STATIC = estimated codec-noise floor / gap > 0.12
(noise level from the Aug-24 re-encoded dissolves).

**Data.** 6,664 clips from 21 store variants + the Aug-24 tier-2 regen: base_cond (v2 neutral/effect, v3
neutral/effect, ED81 neutral/effect), dualforce_control (same six), dualforce_dcg_w6 (same six),
dualforce_dcg_w1 / w1p5 / w3 (v2 neutral, seed 42). One clip failed to decode.
**Exclusions.** Foreign/davis endpoints (2,186 rows, kept out as in Aug-24), STATIC (143), byte-identical
duplicates within a variant (584; the base model's same-endpoint rows). Grid v3 hardlinks 139 v2 rows, so
pooled analyses use a second cross-variant dedupe. **4,031 unique clean clips analysed.**
**Statistics.** Unit = endpoint pair: every interval is a 95% cluster bootstrap over endpoints (2,000
resamples). Paired contrasts join on (endpoint, seed) or on the arm-stamp-stripped grid row + seed, and report
median ΔDR, Cliff's δ = mean sign(Δ), Wilcoxon signed-rank, and on-line flips with an exact two-sided sign test.
**Covariates** (`covariates.py`, per endpoint): DINOv2-base CLS cosine distance and CLIP ViT-B/32 distance
between the two anchor frames (GT transition clip frames 8 / T−8 where the real transition exists: 78
endpoints; else the conditioned frames of the base_cond two-sided generation), pixel gap, and the real
transition's own DR/M as the per-pair ceiling.

## 3. Results

### 3.1 Base LTX-2, neutral prompt: both endpoints vs start only (Table A, D; fig1)

| grid | cond | n unique | endpoints | DR median [IQR] | on-line share [CI] | on-line clips (DISS/CUT/FRZ) |
|---|---|---|---|---|---|---|
| v2 | both | 30 | 15 | 0.211 [0.136, 0.306] | 26.7% [6.7, 46.7] | 0 / 8 / 0 |
| v2 | start | 110 | 55 | 0.430 [0.300, 0.559] | 6.4% [1.8, 12.7] | 0 / 7 / 0 |
| v3 | both | 38 | 19 | 0.211 [0.166, 0.306] | 21.1% [5.3, 39.5] | 0 / 8 / 0 |
| v3 | start | 119 | 60 | 0.506 [0.366, 0.637] | 3.4% [0.0, 7.6] | 0 / 4 / 0 |

Unpaired both − start: Δmedian DR −0.220 [−0.282, −0.130] (v2), −0.296 [−0.343, −0.206] (v3); Cliff's δ −0.54 /
−0.69 (Mann–Whitney p < 1e-4); Δ on-line share +20.3 pp [2.0, 41.6] / +17.7 pp [1.8, 35.9]. Grid v3 is a superset
of v2 (139 hardlinked rows), so the two grids are not independent replicates.
Cut-point sensitivity (Table B): the both/neutral share is flat from θ=0.12 to 0.18 (27% v2, 21–26% v3) and
falls to 10% / 8% at θ=0.10; start/neutral stays ≤ 6% up to θ=0.16.
Per-endpoint propensity (neutral, 2 seeds; Table E): 33% (v2) / 26% (v3) of endpoint pairs have at least one
on-line generation, 20% / 16% have both seeds on-line.

### 3.2 Paired end-anchor probe (Aug-24 tier-2 regen, deduplicated; Table C)

Same 15 endpoints × 2 seeds, both-endpoint vs start-only with the suffix anchor dropped: 30 pairs.
ΔDR (both − start) median −0.052 [−0.173, 0.018], Cliff's δ −0.33, Wilcoxon p = 0.020; on-line 26.7% → 10.0%,
flips 5 vs 0, exact sign test p = 0.062 (two-sided; 0.031 one-sided). Direction consistent with 3.1; the
evidence is thin (15 endpoints).

### 3.3 Prompt: describing the transition in text (Table C, D; fig1)

Paired effect-prompt − neutral-prompt on the same endpoint and seed, base model:

| grid | cond | pairs | ΔDR median [CI] | Cliff's δ | on-line effect → neutral | flips |
|---|---|---|---|---|---|---|
| v2 | both | 52 (15 ep) | +0.139 [0.106, 0.174] | +0.73 | 5.8% → 28.8% | 1 / 13 (p=0.002) |
| v3 | both | 94 (19 ep) | +0.148 [0.091, 0.185] | +0.66 | 8.5% → 26.6% | 2 / 19 (p<0.001) |
| v2 | start | 156 (40 ep) | −0.120 [−0.191, −0.046] | −0.37 | 9.0% → 5.1% | 10 / 4 (p=0.18) |
| v3 | start | 271 (60 ep) | −0.119 [−0.160, −0.045] | −0.34 | 6.3% → 3.0% | 14 / 5 (p=0.064) |

With the effect clause in the prompt the both-vs-start gap is gone: unpaired Δ on-line −4.9 pp [−15.4, 10.5]
(v2) and +0.6 pp [−9.8, 13.4] (v3); Δmedian DR +0.075 / +0.048 with intervals spanning 0. Regression R2 (both
prompts, interaction term) puts the both×neutral interaction at −0.312 [−0.427, −0.214] and the main
both-endpoint term at +0.097 [−0.013, +0.219]. *Campaign reading:* the pull toward the null family appears only
when the model has no operator information from any channel; a text description removes it in these
generations. Under start-only conditioning the sign reverses (effect prompt slightly more on-line); note that
for start-only clips the "line" runs to the model's own last frame, so a smooth monotone morph reads as on-line
there — the start-only DR is not the same quantity as the both-endpoint DR (see §5).

### 3.4 Training (Table A, C)

dualforce_control: **0 of 292 unique clean both-endpoint clips on-line** across v2/v3 × neutral/effect (DR
median 0.62–0.69); start-only 1.8–2.5% (a few CUTs). Paired base − trained on the same grid row and seed,
neutral both: ΔDR −0.394 [−0.453, −0.294] (v3, 94 pairs, 19 endpoints), Cliff's δ −0.98, on-line 26.6% → 0.0%,
flips 25 vs 0. ED81 tier: base neutral 56.8% → trained 2.7% on the 37 non-static pairs (see 3.7 for why ED81 is
a different phenomenon).

### 3.5 Guidance weight (Table A, C; fig4)

dualforce_dcg − dualforce_control, paired on row and seed, neutral, both-endpoint: ΔDR median +0.001 (w=1, p=0.80),
+0.052 (w=1.5), +0.133 (w=3), +0.211 (w=6) on v2; +0.253 [0.181, 0.330] at w=6 on v3 (Cliff's δ 0.91–1.00).
Start-only moves the same way (+0.003 → +0.157). Medians for both-endpoint: no guidance 0.64 → w=6 0.84 (v2),
0.62 → 0.87 (v3). For reference, the real transitions between the same endpoints have DR median 0.56, IQR
[0.43, 0.64] (78 endpoints with GT clips). *Campaign reading:* guidance pushes the middle monotonically away from
the endpoint line, as its construction intends; at w=6 the median sits above the real-transition interquartile
range, so DR is not a higher-is-better score and this table cannot by itself say whether w=6 is better or
over-driven.

### 3.6 Endpoint covariates (Table F; fig3)

Within base_cond both-endpoint neutral: Spearman ρ(DR, DINO distance) = −0.48 (p=0.007, v2) and −0.39
(p=0.016, v3); on-line share by DINO-distance tercile low/mid/high = 20% / 10% / 50% (v2) and 0% / 25% / 42% (v3).
Same sign under the effect prompt (−0.38, −0.19). Regression R3 within that stratum: DINO coefficient −0.342
[−0.749, +0.084] with motion terms included (38 clips, 19 endpoints). No relation under start-only (ρ ≈ 0).
Motion inside the conditioning windows correlates positively with DR everywhere (ρ 0.2–0.5), as expected for a
frozen-frame reference line; regression R1 (neutral, both vs start, controlling for DINO distance and prefix
motion) keeps the both-endpoint term at −0.226 [−0.306, −0.150]. CLIP distance shows no consistent relation.
*Campaign reading:* the on-line outputs concentrate on endpoint pairs whose anchor frames are semantically far
apart — where a bridge has to be invented — rather than on near-identical pairs. Pixel gap correlates
negatively too, but that is partly arithmetic (gap-normalised residual).

### 3.7 Where in the family, and the EffectData tier (fig2)

Every on-line both-endpoint clip of the base model is CUT-shaped (M median 0.06; 0 DISSOLVE in 16 on-line HF
neutral clips, 1 in 8 under the v3 effect prompt). The DR×M scatter separates base both-endpoint outputs (low
DR, low M) from the trained arms and the real transitions (DR 0.4–0.8, M 0.2–0.6). The ED81 tier (81 f,
frame-0 anchor, start-only) behaves differently: base_cond neutral has 70 of 102 candidate clips STATIC (start
≈ end frame) and the remaining 32 are 53% on-line, 12 of 17 FREEZE — the base model given one frame and a bland
caption barely animates. This is I2V stillness, not the both-endpoint shortcut, and should stay out of the
collapse table.

## 4. What the paper can take from this (campaign reading, owner to decide)

- Table 1 should report **unique generations and endpoint pairs** (30 / 15 on v2; 38 / 19 on v3) and the
  corrected paired statistics (5 vs 0 flips, p≈0.03–0.06; Wilcoxon on ΔDR p=0.02). The current n=52 and
  p<0.001 are not defensible.
- The claim "with both endpoints and no operator information the base model tends to the null family" holds
  at the level measured, controlling for anchor motion, and the on-line outputs are cuts, not dissolves
  (consistent with the current §3.2 text).
- **New column worth adding:** the same base model with the transition described in text does not show the
  pull (both ≈ start). This directly supports the framing "collapse = operator not read from any channel"
  and connects §3.2 to the neutral-text argument of §5.2.
- Training row: 0/292 on-line for the dualforce control across two grids and two prompts.
- Guidance: DR rises monotonically with w; the paper should present this as "moves off the endpoint line",
  not as quality, and pair it with the endpoint-fidelity and past-peak evidence planned for Table sweep.

## 5. Limits of this measurement

1. **Start-only asymmetry.** With no end anchor, the reference line runs from the start frame to the model's own
   last frame; smooth monotone morphs score as on-line. Start-only rates are therefore an upper bound on
   shortcut behaviour and are not directly comparable to both-endpoint rates. The paired end-anchor probe (3.2)
   is the comparison that isolates the anchor.
2. **No end-only condition** and **no second untrained base model**; "training removes it" rests on one trained
   family (dualforce) here.
3. **Small endpoint pools for both-endpoint rows:** 15 (v2) and 19 (v3) clean endpoint pairs, 2 seeds; the v3
   pool contains the v2 pool.
4. **Frozen-frame reference line.** Held shots with camera motion read as off-line (motion ρ ≈ +0.4); the
   observed base cuts sit at DR ≈ 0.11 rather than at the synthetic-cut floor (0.002) for this reason.
5. **Cut point.** 0.12 is descriptive; the sensitivity table is the honest object. The DISSOLVE label no longer
   uses monotonicity; S is reported as a descriptor only.
6. Cross-arm joins strip the per-arm item-id stamp and the v2 DCG suffixes (`__dfw6`, `__dfw6_e`); every
   guidance contrast in Table C joins after that fix.

## 6. Next steps (from the design discussion, not started)

Paired regeneration probe on base LTX-2: (a) both-anchor regeneration of the existing start-only generations
using their own last 8 frames as the end clip (the start-only output is a witness that a non-degenerate
transition exists between exactly those anchors), (b) start-only regeneration of the two-sided rows on v3
(tier-2 replication), (c) an end-only condition in the same batch; same seeds; selection guard on REAL-class
sources; endpoint as the unit. Roughly 340 + 70 + 70 clips.

## 7. Reproduce

```
sbatch misc/2026-09-08_collapse_remeasure/job_score.sbatch      # score_store.py + covariates.py + analyze.py
python misc/2026-09-08_collapse_remeasure/analyze.py            # ~2–4 min on a login node (CPU only)
```
Store inputs: gens/005_base_cond/{01,02,04,05,06,07}, gens/013_dualforce_control/{01..06},
gens/032_dualforce_dcg_w6/{01..06}, gens/029–031 (01_neutral), misc/2026-08-24_lerp_collapse/tier2_gen/out.
