# From-scratch collapse probe — REPORT

**Date:** 2026-09-08 (launched 17:42 CDT, scored 19:04 CDT). **Model:** base LTX-2 19B, no adapter, recipe of
`base_cond_neutral` (30 steps, guidance 4.0, stg 1.0, 480×640×121 f, prefix 9 f / suffix 8 f). **Compute:** DeltaAI
jobs 3115053 (R1), 3115054 (R2/R3), 3115382 (scoring), `bgjg-dtai-gh`, ≈5 GPU-h. **Numbers:** `TABLES.md`,
`per_clip.csv`, `paired.csv`. **Figures:** `fig_probe_paired.png`, `fig_strips_r1r2r3.png`, `fig_probe.png`.
Levels are stated as measured; every reading below is this campaign's and is marked as such.

## 1. Why this experiment

In the eval grid, two-sided rows are all scene-change classes (start→end DINO distance median 0.89) and one-sided
rows are all in-place effects (0.61); the class sets are disjoint. So "both anchors collapse, start-only does not"
in the grid is confounded with transition type. This probe controls the inputs: the same start clip, the same end
anchor and the same seed across conditions, so single factors can be changed one at a time.

## 2. Design

30 real 9-frame start clips from in-place classes (≤2 per class). 40 three-part prompts written for them: start
caption (verbatim from the grid) + change clause + end caption. 30 describe a scene change through a named
mechanism (30 distinct mechanisms); 10 are in-place controls on the same start clips. Seeds 42, 43, 44.

| run | anchors | text | role |
|---|---|---|---|
| R1 | start only | full (start + change + end) | witness: shows a non-degenerate transition exists; its last 9 frames become the end anchor |
| R2 | start + R1's end | full | end anchor added, text kept |
| R3 | start + R1's end | captions only (start + end, no change clause) | anchors kept, operator information removed from the text |

R1→R2 differ only in the end anchor; R2→R3 differ only in the text. Because the end anchor is R1's own last frames,
all three runs are measured against the same two endpoint frames (start anchor, R1 end), so the one-sided versus
two-sided asymmetry of the grid measurement does not exist here.

Instrument: `misc/2026-09-08_collapse_remeasure/instrument.py`. DR = median normalised off-endpoint-line residual of
the interior frames (0 = every frame is a blend of the two endpoint frames; the paper's Eq. lerp family). M = share
of frames whose projection lies in the middle half of the segment (dissolve 0.5, cut or freeze 0). Realized scene
change = DINOv2-base CLS distance between the start anchor frame and R1's last frame.

## 3. Results

### 3.1 Levels (median [IQR], all 3 seeds)

| run | tier | n | DR median [IQR] | DR mean | M median | share DR ≤ 0.12 |
|---|---|---|---|---|---|---|
| R1 | scene change | 90 | 0.367 [0.234, 0.499] | 0.400 | 0.23 | 6.7% |
| R2 | scene change | 90 | 0.334 [0.234, 0.448] | 0.371 | 0.21 | 6.7% |
| R3 | scene change | 90 | 0.206 [0.112, 0.291] | 0.219 | 0.06 | 27.8% |
| R1 | in-place | 30 | 0.278 [0.229, 0.388] | 0.316 | 0.24 | 0.0% |
| R2 | in-place | 30 | 0.291 [0.208, 0.360] | 0.298 | 0.21 | 0.0% |
| R3 | in-place | 30 | 0.286 [0.173, 0.338] | 0.285 | 0.18 | 0.0% |

Realized scene change of the R1 witnesses: scene-change prompts 0.978 [0.958, 1.008] (saturated: the model did
change the scene in essentially every case); in-place controls 0.451 [0.290, 0.637].

### 3.2 Paired shifts (same prompt and seed)

| contrast | tier | ΔDR median | negative / n | Wilcoxon p |
|---|---|---|---|---|
| R2 − R1: end anchor added, text kept | scene change | −0.008 | 55 / 90 | 0.013 |
| R2 − R1 | in-place | −0.025 | 20 / 30 | 0.033 |
| R3 − R2: text removed, anchors kept | scene change | −0.070 | 79 / 90 | 6e-14 |
| R3 − R2 | in-place | −0.005 | 23 / 30 | 0.041 |
| R3 − R1: both changes | scene change | −0.100 | 78 / 90 | 1e-13 |
| R3 − R1 | in-place | −0.029 | 22 / 30 | 0.004 |

Per seed, R3 − R2 on scene-change prompts: −0.094 (28/30 negative), −0.038 (24/30), −0.068 (27/30). 22 of the 30
scene-change prompts move toward the line in all three seeds; none moves away in all three. Largest single drops:
P10H seed 42 (1.54 → 0.22), P19H seed 42 (0.85 → 0.13), P05H seed 43 (0.57 → 0.05).

### 3.3 What the collapsed outputs look like (`fig_strips_r1r2r3.png`)

R1 and R2 execute the described mechanism (fog swallowing the frame, a dive through a puddle reflection, a plunge
through a window). R3 on the same inputs holds the start scene, then crosses to the end scene within a few frames
around the midpoint (a ghosted crossfade in P10H and P19H, a hard cut in P05H), then holds the end scene. This is why
M is low (0.06): the crossing is brief, so few frames sit mid-segment. Both cut-like and short-dissolve-like
crossings occur, which answers the draft's open remark that dissolve-like coverage had been seen but not scored.
One R3 output (P05H seed 43) also hallucinates on-screen text at the cut.

### 3.4 Agreement with the grid measurement

The grid's base model, neutral prompt, both anchors: DR median 0.211, M 0.06, 21–27% on-line. The probe's R3 on
scene-change endpoints: 0.206, 0.06, 27.8%. The grid's effect-prompt both-anchor rows: 0.385; the probe's R2: 0.334.
The controlled probe reproduces the grid levels.

## 4. Reading (this campaign's, for the owner to weigh)

1. **The end anchor by itself does almost nothing when the text describes the transition.** R2 − R1 is −0.01 on the
   median; detectable across 120 pairs, but the outputs remain full transitions.
2. **Removing the operator information from the text, with both anchors fixed, pulls the middle toward the
   endpoint line** on scene-change endpoints (−0.07 median, 79/90), and does not on in-place endpoints.
3. **So the collapse condition is a conjunction:** both anchors given, no description of the operator, and endpoints
   that are a different scene. The draft's §3.2 currently attributes it to the end anchor; the honest statement is
   "both endpoints fixed and the operator unspecified", which also connects §3.2 directly to the neutral-text
   argument of §5.2 (the reference channel is what supplies the operator when the text does not).
4. **Dose axis:** realized scene change is saturated (~0.98) for every scene-change prompt, so within that tier
   there is no gradient to read; the contrast is scene change versus in place. A graded axis would need prompts
   that change the scene partially, which the base model did not produce here even for the in-place controls
   (it tends to move or remove the subject).

## 5. Limits

- No start-only + captions-only run (R0), so the anchor's effect *under* a neutral prompt is not isolated here; the
  grid's paired tier-2 regen (5 vs 0 flips on 15 endpoints) is the only such evidence and it is thin.
- No end-only condition.
- One base model; the trained arms were not run on these inputs.
- The end anchor is synthetic (R1's frames through the VAE), 30 start clips, all from one base model's grid.
- In-place controls are imperfect: the base model often changed the subject even when asked not to (realized
  change 0.45), so "in place" means same place, not same content.

## 6. Reproduce

```
sbatch job_r1.sbatch
sbatch --dependency=afterok:<r1 jobid> job_r2r3.sbatch
sbatch --dependency=afterok:<r2r3 jobid> job_score.sbatch
```
Inputs: `prompts/prompts.jsonl`, `reg/r1.jsonl` (built by `build_r1.py`), `reg/r{2,3}_s<seed>.jsonl` (built by
`splice_r1.py` from the R1 outputs). Probe conditioning windows: `eval_ladder/conds/probe_<pid>_s<seed>_{start9,end9}.mp4`.
