# READING — what the base model does by default between two anchors (driver, 2026-09-14)

This is the driver's (Fable) reading of `REPORT.md` / `TABLES.md`. It is an interpretation, attributed to the driver;
the numbers it rests on are in the operator's files. Strata are read side by side, never pooled (SPEC §1).

## 1. The default is not a member of the lerp / cut / freeze family
In all three point spaces the base neutral generations (S-GRID NULLGEN n=68; S-PROBE R3 n=120) sit nearest a REAL
trajectory, not a landmark, in ≥ 95 % of clips (landmark-nearest: PIX 0.00 / 0.00, DINO 0.04 / 0.04, VAE 0.03 / 0.00;
reference leave-one-out baseline ≈ 0, so the measure discriminates). No null clip is landmark-nearest in all three spaces.
Caveat (driver): the landmark set has ONE cut position (0.5); a cut at 0.2 is far from CUT50 in the τ-profile
coordinates, so the nearest-neighbour measure under-detects off-centre cuts. The position-free TRANS "swap-like" flag is
the better cut detector and it fires on 22 % of S-GRID nulls (15/68) and 17 % of R3 (20/120) vs 1 % of R2 (1/120) and
≈ 5 % of GT by construction — consistent with the 27 % / 21 % "on-line CUT" rates of the earlier campaigns
(`misc/2026-09-13_null_ledger/LEDGER.md` §2.1). So: a ~20 % tail of frank swaps, and an 80 % body that is not a cut.

## 2. What the body of the distribution looks like — the same signature in both strata
Relative to its own reference (GT twin for the grid, R1 witness for the probe), the text-dropped default is:

| descriptor (space) | S-GRID NULLGEN vs GT | S-PROBE R3 vs R1 / R2 | landmark bracket |
|---|---|---|---|
| straightness DR (PIX) | 0.21 vs 0.60 | 0.22 vs 0.35 / 0.32 | LERP/CUT 0 |
| mid-line mass M (PIX) | 0.06 vs 0.40 | 0.07 vs 0.23 / 0.21 | LERP 0.5, CUT 0 |
| path / gap (PIX) | 7.6 vs 22 | 5.0 vs 8.4 / 8.1 | landmarks 1 |
| novelty nu_max (TRANS) | 0.32 vs 0.50 | 0.29 vs 0.53 / 0.49 | CUT 0, LERP 0.28 |
| swap sharpness (TRANS) | 0.56 vs 0.23 | 0.50 vs 0.28 / 0.30 | CUT 1.0, LERP 0.12 |
| change extent at peak, local_at_peak (TRANS) | 0.59 vs 0.33 | 0.49 vs 0.27 / 0.30 | CUT 0.68, LERP 0 |
| transport energy (TRANS) | 0.058 vs 0.074 | 0.055 vs 0.055 / 0.054 | floor 0.053–0.057 |

Driver's reading: the default is a **fast, spatially global appearance swap from A to B** — half the novelty of a real
transition, twice its swap sharpness, change covering ~60 % of cells at the peak instead of ~30 %, a third of the total
path, and almost no time spent mid-line — but it is NOT a single-frame cut in most clips (step_share 0.08 vs 1.0), and
the frames do leave the A–B segment (nu_max 0.72–0.78 in PIX). Transport energy is at the landmark floor for EVERY base
generation in the probe (R1, R2, R3 alike) and only modestly above it for GT, so the transport channel separates
"real vs generated", not "text vs no text"; the swap/novelty/extent channels do the latter.
Both strata show the same signature although they differ in end-anchor provenance (real vs model-made), prompt form,
endpoint distances and seeds — that agreement is the robustness result the SPEC asked for.

## 3. The default needs the end anchor AND the missing text
- Text drop with both anchors (R2 → R3): PIX DR −0.07, swap_sharp +0.21, in the scene-change tier (n=90, p < 1e-13).
- End anchor with full text (R1 → R2): PIX DR −0.03 (from R3−R1 = −0.10 and R3−R2 = −0.07). Smaller.
- No end anchor, no text (S-GRID-START, n=204): swap_sharp 0.24 (= GT's 0.23), M 0.31, DR 0.50 — the base just drifts;
  no swap signature at all. So the swap is the model's answer to "reach this exact frame" when nothing says how.

## 4. The effect is conditional on the anchors demanding a scene change
In-place tier (n=30): R3 − R2 is non-significant on every metric (PIX DR −0.005 p=0.04, DINO DR −0.002 p=0.33,
VAE DR +0.001 p=0.60, swap_sharp +0.014 p=0.25). Scene-change tier (n=90): every metric moves, every p < 1e-13.
Driver's reading: when the two anchors share a scene, the captions-only branch and the full-prompt branch produce
the same trajectory geometry; the "default" degeneracy lives in the far-endpoint regime (consistent with the far-DINO
tercile having 50 % CUT in the Aug-24 addendum).

## 5. exp_024 side panel (different config; n=10 per arm) — text content barely moves geometry there
A_empty ≈ A_word ≈ B–E on DR (0.52–0.58), nu_max, step_share; only swap_sharp separates the null arms (0.45) from the
abstract-cue prompt C (0.29). DAVIS cross-video pairs, 193 f, CFG 3.2, 25 cond frames — not comparable to the grid.
Driver's reading: not evidence either way about the grid null; it says that in that config no prompt made the
trajectory straight, which is a config question (frame count / cond frames / guidance) worth one controlled check.

## 6. What this says about the null to guide away from (driver's recommendation, to be tested)
1. A synthetic cut / dissolve / freeze is the WRONG negative: the model almost never visits those points (§1).
2. The model's OWN captions-only-with-both-anchors branch IS the default distribution (§2), and it differs from the
   full-prompt branch in one consistent, measurable direction in all four spaces (more novelty, softer swap, more local
   change, longer path). Guidance = full-prompt − captions-only is aimed at the thing that actually happens. Structurally
   this is standard CFG with the neutral caption as the negative, instead of the current quality-word negative
   ("worst quality, inconsistent motion, distorted, jittery") that the base sampler guides against at scale 4.0.
3. A text proxy of the default is also available and cheap: the measured default reads as a "hard cut / jump cut";
   a negative prompt in that vocabulary is a one-line variant.
4. On in-place transitions this null has nothing to push against (§4). For the VFX-transfer task the null must be a
   different drop (the reference), and it must be measured on the trained arm — the trained arms' both-anchor outputs
   sat off the line in the re-measure, so the base default does not transfer to them by assumption.

## 7. Proposed next check (one small job, ~40 endpoints × 2 seeds × 3 arms ≈ 240 clips, ~3 GPU-h)
On the grid endpoints (seeds 42/43), base model, same descriptors:
 (a) neutral caption + negative "hard cut, jump cut, abrupt scene change";
 (b) effect prompt with the NEUTRAL caption as the CFG negative (scale 4.0);
 (c) effect prompt with the current quality negative (= existing 01_effect, no regen needed).
Read (a) vs neutral, (b) vs (c) on DR / nu_max / swap_sharp / local_at_peak; the null is "right" if (b) moves the
distribution toward GT on those four more than (c) does. Not run; owner's call.

## 8. Instrument caveats carried from REPORT.md §7
Single cut position in the landmark set (driver adds: fix = cut family at 0.2/0.35/0.5/0.65/0.8, or align τ-profiles by
crossing time); global nearest-neighbour pool across strata; FREEZE is not a stable TRANS landmark in the start-only
stratum; pixel LERP is not a straight line in DINO (DR 0.45) or VAE (0.28) — use LATLERP as the VAE straight-line
reference; VAE `explained` ≈ 0 for every group, so it does not separate there; S-SWEEP pooled measures are artefacts.
