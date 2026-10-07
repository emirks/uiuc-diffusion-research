# Distance vs cut — addendum to the Aug-24 lerp-collapse campaign (2026-09-12)

Question asked by the owner: the first study and its Tier-2 causal check ran under the V-NEUTRAL prompt
(start-scene sentence only, `store/gens/005_base_cond/02_neutral__dai/meta.v1.yaml` `prompt_rule`). Can the
"far endpoints → abrupt cuts" failure be observed on the existing both-anchor base generations, and what
does the failure look like frame by frame? CPU-only re-measurement of clips already on disk; no new generation.

Data: all unique (md5-deduped) base LTX-2 clips in `misc/2026-09-08_collapse_remeasure/per_clip.csv`
(arm base_cond; 1238 clips), endpoint distances from `endpoint_covariates.csv` (DINOv2 CLS 1−cos, CLIP ViT-B/32
1−cos, pixel gap_rel between the two GT anchor frames). Clean = non-foreign cells (pre-registered denominator).
Instrument: `misc/2026-09-08_collapse_remeasure/instrument.py` (prefix 9, suffix 8; start-only scored to own last frame).

Detectors used here
- CUT class: on-line (DR ≤ 0.12) with step-like τ (Aug-24 classifier, stored `cls`).
- abrupt: largest adjacent-frame step > 6 % of the interior path AND > 56 % of the anchor gap (GT p95 thresholds, 240 clips).
- jump3: largest rise of the distance-to-start-anchor over 3 frames, in gap units (1.0 = the whole A→B distance in 3 frames).
- transit frames: interior frames whose distance to A lies in (0.15, 0.85) gap — frames spent "between" the two states.
Scripts: `cutstats_run.py` → `cutstats.csv`; `dist_analysis.py` → `both_anchor_with_dist.csv`; profiles in
`both_neutral_cut_profiles.csv`, `transit_profiles.csv`; frame strips in `strips/` (200 px, 14 frames; `zoom_*` around the cut).

## A. Cut rates by conditioning × prompt (clean cells, unique clips)

| conditioning | prompt | grid | n | CUT class (on-line) | abrupt (any) | DR med |
|---|---|---|---|---|---|---|
| both anchors | neutral | v2 | 30 | 27 % (8) | 50 % (15) | 0.21 |
| both anchors | neutral | v3 | 8 | 0 % (0) | 62 % (5) | 0.22 |
| both anchors | effect | v2 | 52 | 6 % (3) | 10 % (5) | 0.39 |
| both anchors | effect | v3 | 46 | 9 % (4) | 9 % (4) | 0.39 |
| start only (one-sided rows) | neutral | v2 | 82 | 5 % (4) | 13 % (11) | 0.46 |
| start only (one-sided rows) | neutral | v3 | 40 | 0 % (0) | 22 % (9) | 0.56 |
| start only (one-sided rows) | effect | v2 | 160 | 11 % (17) | 2 % (4) | 0.31 |
| start only, Tier-2 regen of the two-sided rows | neutral | v2 | 30 | 10 % (3) | 53 % (16) | 0.34 |

Paired Tier-2 (both vs start-only regen, same endpoint+seed, clean, 30 pairs): abrupt flag both-only 5 / start-only 6 /
both 10 / neither 9 — no anchor effect on abrupt cutting; on-line flips 5 → 0 (as in §G of the remeasure tables).
All-cells denominators are in `dist_analysis.py` output (same ordering; neutral v3 both: CUT 29 %, abrupt 71 % of 34).

## B. Transit geometry (clean, both-anchor clips of the 19 clean endpoints; GT = the real clip of the same endpoints)

| group | n | endpoints | transit frames med | transit ≤ 3 | jump3 med | jump3 ≥ 0.7 | frames near A | frames near B |
|---|---|---|---|---|---|---|---|---|
| GT clip | 19 | 19 | 26 | 0 % | 0.24 | 0 % | 0.06 | 0.04 |
| both / effect | 98 | 19 | 17 | 8 % | 0.31 | 11 % | 0.06 | 0.19 |
| both / neutral | 38 | 19 | 16 | 26 % | 0.45 | 29 % | 0.08 | 0.18 |
| start-only regen / neutral | 30 | 15 | 7 | 40 % | 0.69 | 47 % | 0.08 | 0.09 |

(start-only gap = start anchor to own last frame, so its jump3 is in its own units.)
Per-clip profiles (`both_neutral_cut_profiles.csv`): in every clean both/neutral clip the interior splits into a
pre-segment at A (median dist-to-A 0.05–0.5 gap) and a post-segment at ≈ 1.0 gap from A. Post-segment distance
to B is ≈ 0.1 for the CUT-class clips and 0.2–0.5 for the off-line abrupt ones: those cut into B's scene and then
move inside it (strip `display_transition_1_s42_both_v2.png`: hold A 48 frames → one-frame cut → B's scene, posing,
reaching the anchored end pose in the suffix). Its start-only twin (`..._start_v2.png`) cuts twice into unrelated scenes.

## C. Endpoint distance vs cutting (both anchors, clean, terciles over endpoints)

DINO (edges 0.84 / 0.94):

| prompt | tercile | n | endpoints | CUT class | abrupt | DR med |
|---|---|---|---|---|---|---|
| neutral | close | 20 | 10 | 15 % (3) | 50 % (10) | 0.26 |
| neutral | mid | 8 | 4 | 0 % (0) | 50 % (4) | 0.26 |
| neutral | far | 10 | 5 | 50 % (5) | 60 % (6) | 0.15 |
| effect | close | 54 | 10 | 2 % (1) | 6 % (3) | 0.43 |
| effect | mid | 18 | 4 | 0 % (0) | 6 % (1) | 0.40 |
| effect | far | 26 | 5 | 23 % (6) | 19 % (5) | 0.27 |

Spearman(DR, DINO dist): neutral ρ = −0.39 (p 0.016), effect ρ = −0.22 (p 0.032). Median DINO dist of CUT-class vs other
clips: neutral 0.94 vs 0.77 (Mann–Whitney p 0.039), effect 0.94 vs 0.83 (p 0.028). Abrupt vs other: neutral 0.84 vs 0.80
(p 0.61), effect 0.95 vs 0.83 (p 0.019). CLIP distance: CUT vs other 0.41 vs 0.33 (p 0.023) neutral only; pixel gap:
Spearman(DR) −0.43 (p 0.006) neutral, −0.29 (p 0.003) effect, CUT vs other n.s. Full tercile tables for CLIP and pixel in
the script output.

## D. Reading (attributed: operator, 2026-09-12; not a verdict)
- Under the start-scene-only prompt the base model does not traverse: it holds A, jumps within a few frames, and either lands
  on B (on-line CUT) or into B's scene (off-line abrupt, classified REAL by the Aug-24 instrument). Real clips of the same
  endpoints never jump (0/19).
- The end anchor does not change how often it jumps (Tier-2 paired abrupt 5 vs 6 discordant); it decides where the jump lands
  (on-line flips 5 vs 0). Naming the effect in the prompt lowers jumping (29 % → 11 % jump3 ≥ 0.7).
- On-line cutting increases with semantic endpoint distance in both prompt conditions; total abrupt cutting is flat vs distance
  under the neutral prompt (already ~50 %).
- Limits: 19 clean endpoints, 2 seeds, DINO range 0.50–1.03 only (all scene-change transitions; no near pairs), tercile bins of
  4–10 endpoints, one model. The "far endpoints → abrupt cuts" claim needs a designed distance spectrum (same-video pairs through
  cross-class pairs) with a text condition that does not conflict with the end anchor.
