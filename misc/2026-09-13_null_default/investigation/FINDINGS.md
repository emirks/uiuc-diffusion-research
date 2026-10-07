# FINDINGS — which null should CTT / VFX-transfer guidance push away from? (advisor, 2026-09-14)

Scope: CPU re-analysis of the campaign's 2,513 feature files (PIX decoded serially; DINO/VAE/TRANS from
`$LAB/cache/null_default/features`), `results/per_clip.csv`, the Sep-08 re-measure rows of the trained arms, and the
mechanism sources (`eval_ladder/run_gen.py`, `ltx_trainer/validation_runner.py` DCG null kinds, `store/runs/020` meta,
`misc/2026-08-14_dcg_conditioning/DOSSIER.md`). No GPU, no Slurm, no commits. Readings are the advisor's; numbers carry
metric and bar. Tables: `direction_table.{csv,md}`, `dcg_empirical_check.md`, `info_argument.{csv,md}`,
`step_a_summary.md`, `extra_numbers.md`, `swap_tail.csv`, `clusters.csv`, `curves_*.csv`.

## 0. Verdict

**A lerp — in any of the four spaces — is not justified as the guidance null, for the base model or the trained CTT
arm.** "Adds no information beyond the endpoints" is exact only in the space the lerp is taken in; read in every other
space it encodes a *dissolve*, and "away from a dissolve" points the wrong way on exactly the descriptors where the
default deviates most from real transitions (timing: mid-line mass, swap sharpness, transit duration, spatial extent).
The one realisation of a lerp null on the trained arm (DCG-CN) shows the failure: off-chord content inflates (DR +0.30
past GT, 19/0 endpoints at w = 6) while timing does not move (M flat), and that content was demo intrusion.
**Base model: the null is the model's own text-dropped branch** (neutral caption as the CFG negative, both anchors) —
Step A shows that branch *is* the default distribution; Step B shows guiding away from it is right by construction.
**Trained arm: no test-time null on these descriptors** (its default already sits at GT on DR/M); the principled null is
a trained reference-drop, not a synthetic reference. The lerp stays as eval normaliser and descriptor floor.

## 1. Step A — the default as a trajectory

Groups: S-GRID NULLGEN (68) vs GT twin (19); S-PROBE R3 (90 scene-change / 30 in-place) vs R1 witness and R2 (90);
S-GRID-F (38); S-SWEEP A_empty/A_word (10 each, side panel). 48-point interior grid; `fig_traj_*.png`, `fig_clusters_*.png`.

**PIX.** Mean τ at s = 0.1/0.25/0.5/0.75/0.9: NULLGEN 0.11/0.35/**0.78**/0.97/0.99, R3 0.12/0.40/**0.89**/0.97/1.00 vs
GT 0.07/0.25/**0.52**/0.79/0.93, R1 0.12/0.37/0.69/0.92/0.98 — GT's mean progress lies on the lerp diagonal; the
default is three-quarters done at mid-clip. Crossing (first τ ≥ 0.5) median 0.37 / 0.29 vs 0.46 / 0.34.
**Transit** (interior time with τ ∈ (0.2, 0.8)): NULLGEN 8 % ≈ **9 frames** [IQR 4–17], R3 9 frames; GT 48 % ≈ **50
frames**, R1 30; clips with transit ≤ 10 %: 60 % / 62 % vs 0 % / 11 % (pixel LERP 61 frames). Crossing-aligned τ at
aligned s = 0.3/0.4/0.6/0.7: NULLGEN 0.06/0.11/0.88/0.95 (a 0.77 step within ±0.1 of the crossing), R3 0.08/0.15/0.92/
0.95; GT 0.19/0.32/0.66/0.73 (0.34); LERP 0.30/0.41/0.60/0.70 (0.19); CUT50 0/0/1/1. Off-chord ρ at the aligned crossing
0.64 / 0.51 vs GT 0.84 / R1 0.68 / LERP 0; novelty ν there 0.72 / 0.61 vs 0.94 / 0.81 / 0.48. **Abrupt** (τ rises ≥ 0.5
gap within 3 frames): NULLGEN **46 %**, R3 **43 %**, S-GRID-F 55 % vs GT 5 %, R1 6 %, R2 7 %.

**Clusters** (k-means k = 3 on [τ(s), ν(s)], 8 restarts): C0 *early swap* (crosses s = 0.26): NULLGEN 38 %, R3 70 %,
GT 0 %, R1 32 %. C2 *mid swap* (crosses 0.49, τ(0.25) = 0.09 → τ(0.75) = 0.96; all landmarks fall here): 48 %, 27 %,
5 %, 30 %. C1 *gradual, high-novelty* (τ(0.25) = 0.36, τ(0.75) = 0.86, ν_max 0.96): NULLGEN **13 %**, R3 **3 %**, GT
**95 %**, R1 38 %. DINO reproduces the three prototypes (35/50/15 %, 71/26/3 %, 0/21/79 %).
**The advisor reads:** the driver's "≈ 20 % swap tail + 80 % body" is refuted as stated — 87 % (grid) to 97 % (probe)
of the text-dropped default is swap-*shaped*; what varies is timing (early vs mid) and abruptness (46 % finish within 3
frames, the rest within ≈ 10–20). Only 13 % / 3 % resemble a real transition's gradual, novelty-rich course. R1 (full
prompt) is intermediate (transit 30 frames, 6 % abrupt, 38 % GT-like): text moves the base model about a third of the
way from the swap to GT geometry.

**DINO.** Same, sharper: transit 9 / 6 frames (59 % / 72 % ≤ 10 %) vs GT 41, R1 35; aligned ρ at crossing 0.68 / 0.55
vs 0.82 / 0.76. **The pixel LERP read in DINO** sits at ρ 0.62, ν 0.78 at mid-course (DR 0.45, path/gap 3.5) — as far
off the DINO chord as the default and at 84 % of GT's novelty: a dissolve's double exposures are new semantic content.
The true DINO chord has ρ = 0 but a midpoint norm of 0.72–0.76 on a unit-sphere feature space (`fig_info_latent.png`).

**VAE.** 8:1 temporal compression smears the swap: transit 6 / 13 latents (NULLGEN 44 %) vs GT 9 (71 %); aligned τ
0.25/0.35/0.76/0.81 vs GT 0.33/0.42/0.62/0.68 vs LATLERP 0.32/0.44/0.63/0.73 — in timing GT tracks the latent line, but
at ρ 0.82 off it (default 0.76; encoded pixel LERP a flat band at ρ ≈ 0.30; LATLERP 0). DR 0.585 / 0.741 / 0.279 / 0.000.

**TRANS.** Swap progress at s = 0.5: NULLGEN 0.73, R3 0.87 vs GT 0.59, R1 0.62, LERP 0.42. Novelty nu_t at mid-window
0.14 / 0.12 vs 0.42 / 0.41 — the pixel LERP's field reads **0.24**, more novelty than the default. swap_sharp 0.56 / 0.50
vs 0.23 / 0.28 (LERP 0.12); local_at_peak 0.59 / 0.49 vs 0.33 / 0.27 (LERP 0.00).

**Statement.** In pixel, DINO and transport space the default is a fast, spatially global A→B swap: three-quarters
complete by mid-clip, crossing in ≈ 9 frames (GT 50), 0.5–0.7 gap off the chord while crossing (GT 0.8–0.9), a third of
GT's novelty, twice its swap sharpness; a literal cut in ≈ 45 % of clips, a gradual transition in ≈ 10 %; in VAE the
swap fits in one latent step. The lerp of each space lies on the far side of GT from the default on every timing
descriptor (M 0.5 vs GT 0.40 vs default 0.05; transit 61 vs 50 vs 9 frames; swap_sharp 0.12 vs 0.23 vs 0.56) and on
the same side on the off-chord ones (DR 0 vs 0.60 vs 0.21; ν_max 0.5 vs 1.06 vs 0.78).

## 2. Step B — guidance-direction table (`direction_table.md`)

Criterion (pre-committed, stricter than the brief's): guiding away from N is RIGHT on d iff sign(ref − default) =
sign(default − N), i.e. the default lies between N and the reference. The brief's sign(ref − N) = sign(ref − default)
passes cases where N sits *between* default and reference (then guiding away moves away from GT); those are flagged.
Paired agreement = fraction of units (19 endpoints; 90 prompt×seed) with the right sign. Reference GT (S-GRID), R1
(S-PROBE; R2 gives identical signs).

| descriptor | S-GRID: GT / default / pixel LERP → verdict (paired) | S-PROBE high: R1 / R3 / LERP → verdict (paired) |
|---|---|---|
| PIX DR | 0.60 / 0.24 / 0.00 → RIGHT (1.00) | 0.37 / 0.21 / 0.00 → RIGHT (0.87) |
| PIX M | 0.40 / 0.05 / 0.50 → **WRONG** (0.05) | 0.23 / 0.06 / 0.50 → **WRONG** (0.09) |
| PIX cross | 0.45 / 0.40 / 0.50 → WRONG (0.37) | 0.34 / 0.28 / 0.50 → WRONG (0.39) |
| PIX nu_max | 1.06 / 0.78 / 0.50 → RIGHT (0.90) | 0.82 / 0.71 / 0.50 → RIGHT (0.63) |
| PIX path/gap | 22.1 / 7.6 / 1.0 → RIGHT (0.95) | 9.5 / 5.0 / 1.0 → RIGHT (0.83) |
| PIX step_share | 0.025 / 0.081 / 0.009 → **WRONG** (0.05) | 0.028 / 0.086 / 0.009 → **WRONG** (0.07) |
| DINO DR | 0.51 / 0.27 / 0.45 → **WRONG, LERP between** (0.00) | 0.30 / 0.18 / 0.38 → WRONG (0.14) |
| DINO M | 0.36 / 0.07 / 0.38 → WRONG (0.05) | 0.29 / 0.05 / 0.26 → WRONG (0.07) |
| VAE DR | 0.74 / 0.59 / 0.28 (LATLERP 0) → RIGHT (0.79) | 0.68 / 0.62 / 0.27 → RIGHT (0.66) |
| VAE M | 0.69 / 0.39 / 0.62 (LATLERP 0.54) → WRONG, N between (0.21) | 0.62 / 0.31 / 0.62 → WRONG (0.07) |
| TRANS nu_max | 0.50 / 0.34 / 0.28 → RIGHT, coin-flip (0.47) | 0.58 / 0.31 / 0.28 → RIGHT (0.57) |
| TRANS swap_sharp | 0.23 / 0.57 / 0.12 → **WRONG** (0.00) | 0.29 / 0.53 / 0.16 → **WRONG** (0.12) |
| TRANS local_at_peak | 0.33 / 0.58 / 0.00 → **WRONG** (0.21) | 0.34 / 0.59 / 0.00 → **WRONG** (0.24) |
| TRANS trans_mean | 0.074 / 0.059 / 0.057 → no gap | 0.053 / 0.053 / 0.056 → no gap |

LATLERP and the analytic DINO chord (DR 0, M 0.5, cross 0.5, ν_max 0.5, path/gap 1) show the same split: RIGHT on the
off-chord family (DR, ν_max, path/gap), WRONG on the timing family (M, cross, step_share). The transport lerp is WRONG
on swap_sharp and local_at_peak and indifferent on novelty (LERP nu 0.28 ≈ default 0.32). In-place tier: 13/22 cells
have no gap — nothing to push against. CUT50 is RIGHT on 8/10 gapped descriptors: the default is cut-like, so "away
from a cut" is the right *direction* — realised naturally by the model's own text-dropped branch (§4).

**Empirical check of the premise, mechanism (a)** (`dcg_empirical_check.md`, `extra_numbers.md`): DCG on
`dualforce_control` (null = VAE-encoded pixel crossfade of the demo's endpoints in the negative branch; neutral, both
anchors, seed 42, paired by endpoint, n = 56): ΔDR +0.001 (w1), +0.052 (w1.5, 42/14), +0.123 (w3, 46/10), **+0.187
(w6, 49/7)**; ΔM +0.00 / +0.03 / +0.02 (27/27, 32/21, 29/26). On the 19 GT endpoints: control DR 0.575 / M 0.351 vs GT
0.604 / 0.404 (ΔDR −0.019, 8/11 — *already at GT*); w3 0.774 (+0.115, 13/2), **w6 0.942 (+0.296 past GT, 19/0)**, M
0.35–0.44 throughout. The advisor reads: "RIGHT on DR" materialises as overshoot, "WRONG on M" does not materialise —
guidance does not act descriptor by descriptor; "away from a dissolve" adds off-chord content and leaves timing alone,
and the DCG dossier identified that content as demo intrusion (demo-copy exceedance 11 → 15 %, GT-exceed 27 → 34 % at
w3/w6; only w = 1.5 clean). The training-time version (runs/020, lose-reference = latent lerp of the demo) scored −6.0
neutral / −2.5 effect pooled-%same vs control (evals/026); the family was KILLed with the mechanism "the repel is
defined on states the sampler never visits". Two realisations, one outcome.

## 3. Step C — the information argument, made precise (`info_argument.md`)

"No information beyond the endpoints" holds only in the lerp's own space:
- **Pixel LERP** in DINO: DR 0.38–0.57, ν_max 0.79–0.92, path/gap 3.3–4.1; in VAE: DR 0.27–0.29, ν_max 0.57; in TRANS:
  nu_max 0.14–0.31, swap_sharp 0.11–0.15, local 0. In every non-pixel space it is a specific object — a smooth, global,
  mid-novelty dissolve — not a floor.
- **LATLERP**: exact in VAE (DR 1e-4). Not decodable here. Its minimum interior norm is **0.74** [0.72, 0.77] of the
  anchor norm (19 GT owners; 0.71 on R3 owners), where every real clip stays ≥ 0.88 (GT min 0.88, median 0.97; NULLGEN
  0.92; CUT50 0.98) and the encoded dissolve stays at 0.89; the encoded dissolve is 0.38 gap from the LATLERP at
  mid-course. The advisor reads the 26 % norm deficit as off-shell latent — a decoded LATLERP is most plausibly a
  washed-out composite, not an in-distribution frame (visual claim unverified).
- **DINO chord**: midpoint norm 0.76 (S-GRID) / 0.72 (R3) on the unit sphere — not any image's embedding, not
  renderable. The eval workbench's 223 nulls correctly embed the *rendered* pixel lerp.
- **Transport lerp**: definable as a signal; it is a real signal describing a dissolve, whereas the signal arms' trained
  exact null is zeros through a bias-free projection (p_drop 0.10) — the network's learned "no signal", a different object.
Honest endpoints-only reference per mechanism: (a) reference video — the pixel LERP (only renderable one); the model
sees VAE(dissolve), a real editing transition for the base prior but a demo type absent from the CTT corpus (Aug-24
Bar I: 0/223 GT clips are dissolves). (b) reference signal — the LERP field, but the trained null is zeros. (c) latent
target — LATLERP, exact but off-shell; the implemented `target_x0` kind uses the encoded pixel crossfade of the target.

## 4. Decisions (advisor)

1. **No lerp null for guidance — any space, either model.** Bar: a null is justified only if RIGHT with paired
   agreement ≥ 0.7 on a majority of gapped descriptors, with headroom (default between N and GT), realisable
   in-distribution for its mechanism. Every lerp fails the first (RIGHT only on the off-chord family; ≤ 0.24 on the
   timing family where the gap is largest); the trained arm fails the second (DR at GT; DCG overshoot +0.30); only the
   pixel lerp passes the third. Confidence ≈ 0.85; overturned if arm (b) of §5 beats arm (a) on timing without overshoot.
2. **Base model: the null is its own text-dropped branch** — CFG negative = the neutral caption "{S1}." with both
   anchors, replacing or composed with the deployed quality negative (scale 4.0, STG block 29). Step A shows R3/NULLGEN
   *is* the default; the R2 − R3 direction (scene-change tier: PIX DR +0.07, M +0.15, swap_sharp −0.21, TRANS nu +0.20,
   p < 1e-13) is right on every gapped descriptor and a no-op on in-place pairs (p ≥ 0.25), which is correct behaviour.
   Headroom is bounded: R1/R2 sit a third of the way to GT (transit 30 vs 50 frames, DR 0.35 vs 0.60) — this null
   fixes the swap, not the base model's missing transition vocabulary. Cheap proxy: negative prompt "hard cut, jump
   cut, abrupt scene change" (untested).
3. **Trained CTT arm: no test-time null on these descriptors.** Default DR 0.575 / M 0.351 vs GT 0.604 / 0.404 (PIX,
   19 endpoints; on-line 0 % in every both-anchor cell). Its failure modes (reference intrusion, ref-dependence) need a
   *reference* null, and the principled one is a trained drop: a short continuation of the raw-video IC-LoRA with
   reference dropout (p ≈ 0.1, as the signal arms do with zeros) so that v(ref) − v(∅) is ordinary in-distribution
   CFG. DCG's crossfade was a workaround for the missing drop and should not be promoted to "the null".
4. **The lerp remains**: the eval normaliser (rendered, DINO-embedded), the descriptor floor (PIX LERP; LATLERP as the
   VAE straight line), a calibration signal for the signal arms ("this field is a dissolve"). Not their CFG null.

## 5. The one smallest GPU test (not run)

Base LTX-2, deployed sampler, the 19 clean S-GRID endpoints with GT twins, both anchors, seeds 42/43 → 38 clips per new
arm; ≈ 1.5 GPU-h on one GH200 (45 s/clip; DCG ≈ 2×) plus minutes of `scripts/extract_features.py`.
- **(a) OWN-NULL**: effect prompt (grid clause, no `sksz`), CFG negative = the row's neutral caption (per-row negative:
  `ValidationConfig.negative_prompt` is global in `run_gen.py`; small change).
- **(b) LERP-NULL**: same prompt, deployed negative, plus x0-level guidance away from the target's own crossfade at
  w = 3 (`dcg_null_kind="target_x0"`, implemented; check `misc/2026-08-13_dcg_target_null/bnull_grid` first). Pure
  LATLERP = replace the encoded crossfade with the anchor-latent lerp (a few lines).
- **(c)** existing `005_base_cond/01_effect`, **(d)** `02_neutral` (the default): no regeneration.
Measure per clip, all four spaces: DR, M, cross, step_share, ν_max, path/gap; TRANS nu_max, swap_sharp, local_at_peak;
transit; 3-frame swap share; paired by (endpoint, seed) vs the GT twin. Gap set G = {PIX M, TRANS swap_sharp, PIX
transit, PIX DR, TRANS nu_max}.
Pre-registered: **own-null RIGHT** if (a) − (c) moves toward GT on ≥ 4/5 of G with paired agreement ≥ 0.65 (n = 38), no
descriptor overshoots GT by more than GT's IQR, and the 3-frame swap share drops ≥ 10 pp vs (c). **Lerp null
falsified** if (b) − (c) leaves M within ±0.03, swap_sharp ±0.05, transit ±3 frames while DR exceeds GT's median (the
DCG pattern). **Recommendation overturned** if (b) beats (a) on ≥ 3/5 of G with agreement ≥ 0.65 and no overshoot.
Seatbelt: eval-v4 %same of (a) not below (c) by > 3 pp; if it is, the follow-up is the composed negative, not a lerp.
If (a) does not move swap_sharp/M at all, the own-branch hypothesis fails (the deployed CFG branch is not what Step A
measured) and the text-proxy negative is the next cheapest probe.

## 6. Could not verify

- LATLERP's decoded appearance (no GPU); only the latent-norm deficit is measured.
- TRANS/DINO/VAE descriptors of trained arms (no features); only PIX DR/M/PR from the re-measure (seed 42 DCG cells).
- `misc/2026-08-13_dcg_target_null` cycle-0 results (gs 1 / STG off): no report found, unread.
- Per-row CFG negative in the base sampler without disturbing STG composition: code change assumed small.
- k-means shares ±5 % near boundaries are noise; transit and 3-frame-jump statistics carry the abruptness claim.
- S-SWEEP (n = 10, 193 f, CFG 3.2): A_empty crosses late (0.66), 60 % GT-like, 30 % abrupt — weaker swap; not pursued.
- The direction table is descriptor arithmetic; the DCG check shows it predicts the off-chord response, not the timing
  response. It is used as a first-order heuristic only.
