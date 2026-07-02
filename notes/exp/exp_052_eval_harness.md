# exp_052 — Transition eval harness v1: metric suite + the harness's own exam

**Status: COMPLETE — harness validated on ground truth before use; effect appearance is the
style-ID workhorse (93% 1-NN vs 24% chance); the three metrics fail on COMPLEMENTARY classes;
exp_051 ladder findings reproduced quantitatively. 2026-07-06.**

## What was built (`src/diffusion/transition_eval/`)

Design premise: a transition is a program acting on content — every metric is computed
relative to the video's OWN endpoints/frames, so videos sharing a style but nothing else
stay comparable. Ten modules, morph/motion math pure numpy (11 unit tests, login-node
runnable):

- **M1 Morph Profile** (`morph.py`): per-frame DINOv2-base CLS cosine to own endpoint
  means (first 9 / last 8 px frames = the C2V conditioning windows), floor-normalized by
  `cross = cos(eA,eB)` so â=1 means "is endpoint A", â=0 means "unrelated content" —
  content-invariant thresholds. Comparison = z-norm + banded 2-channel DTW + Pearson.
  Scalars: transformation depth, departure/arrival timing, identity hold, core-frame mask
  (= frames that are neither endpoint; fallback argmin). 1-endpoint items drop b(t).
- **M2 Motion Fidelity** (`motion.py`): Yatim et al. tracklet-velocity correlation,
  CoTracker3. **Three non-obvious adaptations were REQUIRED for transition videos**
  (frame-0-only grids at chance accuracy before): (1) grid queries at frame 0 AND the
  middle frame with backward tracking — the effect medium doesn't exist at frame 0, so a
  frame-0 grid only samples scene content the effect occludes; (2) per-step visibility
  masking instead of whole-tracklet cuts — in transition videos MOST points get engulfed,
  a mean-visibility filter empties the set; (3) duration-normalized velocities
  (frame-fraction per full clip) + box-smoothed tracks, so one speed floor serves
  121- and 242-frame clips and sub-pixel jitter doesn't pass it.
- **M3 Effect Appearance / M6 Leakage** (`appearance.py`): DINO features of core frames
  only. Appearance = symmetric mean-of-max set similarity to reference core frames;
  leakage = single-max retrieval against ALL reference frames, contrasted with the same
  retrieval on the 8 unrelated styles (style similarity must not read as leakage).
- **M5 Endpoint fidelity + seams** (`endpoints.py`): LPIPS + DINO on conditioned frames
  vs condition clips (resize-cover+crop like the ValidationRunner); seam = robust-z spike
  of temporal LPIPS at the conditioning handoff indices (n_prefix−1, T−n_suffix−1).
- **M4 Rubric judge** (`judge.py`): 5-question evidence-required checklist, local Gemma 3
  12B-it, greedy. **EXPERIMENTAL — not human-validated yet** (needs ~50-100 labeled
  outputs, target Spearman ≥0.8). Env gotcha: transformers 4.57 Gemma3 needs torch≥2.6 →
  runs via `$LAB/LTX-2-official/.venv/bin/python` (torch 2.9), not the diffusion env (2.5).
- **Floors/ceilings** (`controls.py`, `report.py`): per-item lerp-crossfade floor +
  real-clip leave-one-out ceiling, identical pipeline; normalized = (raw−floor)/(ceil−floor).
  No composite number — per-arm three-axis table is the headline.
- Manifest schema is condition-count aware (`n_endpoints=1` portal items degrade gracefully).

## The exam (41 real clips, 9 styles, LOO 1-NN retrieval; chance 24%)

| metric | acc | Cohen's d | reads as |
|---|---|---|---|
| effect appearance | **0.93** | 2.04 | style/medium identity — the workhorse |
| motion fidelity | 0.46 | 0.92 | kinematics — perfect on motion-defined styles (display/jump 1.0, water 0.75), weak on medium-defined ones |
| morph DTW | 0.34 | 0.71 | transition FAMILY/timing — retrieves structurally distinctive styles (jump 1.0, display 0.75), confuses the sweep family |

The failures are COMPLEMENTARY (appearance's only weak class = flying_cam, which has no
effect medium; motion's strong classes are exactly those). Per pre-registration, morph
profile is scoped: depth/timing/lerp detection + same-endpoint comparative fidelity, NOT
standalone style retrieval.

**Lerp floor, empirically calibrated**: crossfade depth = 0.62 ± (max 0.76), NOT ~0 — a
pixel blend's double-exposure middle leaves DINO's endpoint neighborhoods. Real clips:
0.92 mean / 0.64 min. Separation holds; the pre-registered "≈0" idealization was wrong;
this is exactly why controls run through the identical pipeline. `lerp_below_02 = 0`.

## exp_051 ladder rescored (24 items; normalized 0=lerp floor, 1=real LOO ceiling)

| arm | appearance | motion | morph profile | seams (max z) | endpoint DINO |
|---|---|---|---|---|---|
| base | **0.56** | **0.47** | 0.94 | −0.64 | 0.98 |
| t2v | 0.79 | 0.90 | 0.98 | −0.69 | 0.98 |
| i2v_ff05 | 0.69 | 0.91 | 1.00 | −0.68 | 0.98 |
| c2v | 0.78 | 0.88 | **1.00** | −0.67 | 0.98 |

- Base separates cleanly on both style axes; LoRA arms cluster above — matches exp_051
  visual ranking. c2v tops profile fidelity (best curve match to real refs).
- Morph profile near-saturates for ALL arms (0.94–1.0): under endpoint conditioning the
  anchors dominate the curves — profile fidelity is necessary-not-sufficient there; the
  discriminative axes for conditioned generation are appearance/motion/judge.
- **No seams anywhere** (all robust-z < −0.55) and endpoint DINO ≈0.98 in every arm —
  quantitative twin of "anchoring is mechanism-robust".
- **No detectable trigger effect (n=3/cell — NOT "confirmed")**: appearance with vs
  without SHDWSMK is indistinguishable per arm (c2v raw 0.503 vs 0.500), one metric,
  3-vs-3 samples. If it holds at real n, the LoRA applies the style UNCONDITIONALLY
  (baked into weights; the trigger gates nothing) — which would kill the
  trigger-switched multi-style route before it starts. Needs a dedicated cheap probe
  (wrong-trigger + no-trigger inference over ~10 seeds; optional caption-ablation
  training arm) before any multi-style route decision.
- **No leakage**: max retrieval sim vs training/reference frames ≤0.78 (near-copy ≈0.9+);
  LoRA arms sit slightly above base = style-driven similarity, the contrast baseline
  absorbs it.

## Judge (experimental) — see judge_summary.json in the ladder run dir

Gemma 3 12B checklist on all 24 items (run `run_judge.py` via job_judge.sbatch).
Validate against human labels before headline use.

## Reuse

- Score any generated video: build a manifest (`manifest.py` schema) → `run_score.py
  --manifest ... --label ...`. All features/tracks cache under `outputs/eval/cache/`
  (content-hash keys; Tracker.CACHE_TAG bumps on protocol change).
- Re-run the exam after adding styles/clips: `run_validation.py` (figures + retrieval
  report regenerate).
- Ladder/validation runs: `outputs/eval/exp_052/{validation,ladder}/run_0001/`;
  W&B `exp052_validation`, `exp052_ladder` in `creative-transition-transfer`.

## Open questions / next steps

- Judge human-validation set (~50-100 items) → Spearman; until then judge is advisory.
- v2 ideas noted in code: SAM2 masks for region-level appearance; effect-region-restricted
  motion fidelity; per-style morph-profile prototypes (family-level grouping is real).
- 1-endpoint (portal) path implemented but unexercised — needs a portal-style manifest.
- flying_cam-style transitions have no medium: appearance is undefined there by
  construction; motion fidelity is the primary axis for that family.
