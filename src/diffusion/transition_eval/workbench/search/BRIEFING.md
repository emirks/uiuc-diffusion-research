# Metric search — advisor briefing (ground truth, verbatim facts)

**Goal.** Find a distance metric that GENUINELY beats the incumbent `m1a__v3_sided`
on a fixed retrieval exam. Incumbent: **accuracy 0.6726, Cohen's d 1.522, coverage
1.0, misretrieved 73/223**. Target: accuracy **> 0.80**. Apples-to-apples (same judge),
no cheating.

## The data
- **223 clips**, each a 121-frame video of a "creative transition / effect."
- **39 style classes** (e.g. `portal`, `shadow_smoke`, `fire_element`, `shadow`,
  `gas_transformation`). Class sizes range 1–15 (dist: 1×2, 2×6, 3×2, 4×9, 5×4, 6×3,
  7×1, 8×3, 9×2, 10×4, 12×1, 13×1, 15×1). Retrieval task = "clips of the same effect
  style should be nearest neighbors."
- Per clip, the feature is a **[121, 768] time series of L2-normalized DINOv2-base CLS
  embeddings** (one unit vector per frame). This is ALL that m1a sees. Cheap to load,
  already cached (CPU, no GPU).
- A per-clip **"core mask"** selects the mid-transition frames (the effect medium):
  excludes the first 9 + last 8 conditioned frames, then keeps frames whose similarity
  to BOTH endpoints is low (< 0.5) — i.e. the frames that look like "neither endpoint,"
  the effect itself. "Sided": one-sided classes measure departure-from-endpoint-A only;
  two-sided use max(dist-from-A, dist-from-B). Typical core = a few dozen of the 104
  candidate frames.
- Also available per clip: endpoint anchors e_A, e_B (mean of the 9 head / 8 tail CLS
  frames, unit-normalized); the a_hat/b_hat "progress" curves.

## The exam (the FIXED judge — identical for incumbent and every candidate)
- Input: a symmetric **223×223 distance matrix** D.
- **Accuracy** = leave-one-out 1-NN: for each clip, nearest OTHER clip must be same-class.
- **Cohen's d** = (mean cross-class distance − mean within-class distance) / pooled std.
  Bigger = cleaner separation. m1a d = 1.522.
- **Coverage** = fraction of clips with at least one finite off-diagonal distance. NaN
  distances are read as "cannot retrieve" and DROPPED (a NaN row lowers coverage and
  counts as a miss). m1a coverage = 1.0.
- **Misretrieved** = n_clips − n_correct (uncovered rows count as misretrieved). m1a = 73.
- **Hubness gate** (must pass): no clip may be a "hub" that is everyone's NN. k=10;
  skew ≤ 3.0, normalized prediction-entropy ≥ 0.70, max single-class prediction share
  ≤ 0.25. m1a passes comfortably (skew 0.84, H 0.911, maxpred 0.081).
- Chance accuracy = 0.067.

## The incumbent m1a — exact formula
For clips i, j with sided-core CLS frame sets F_i, F_j (each [n_core, 768], unit rows):
```
set_similarity(F_i, F_j) = 0.5 * ( mean_a max_b cos(F_i[a], F_j[b])
                                 + mean_b max_a cos(F_i[a], F_j[b]) )
D[i,j] = 1 - set_similarity(F_i, F_j)
```
= symmetric **mean-of-max cosine** (soft-Chamfer). Order-agnostic bag-of-frames,
best-match, raw cosine, CLS-only.

## Where m1a FAILS (per-class recall, n, sidedness, top confusions)
Perfect (recall 1.00, 15 classes): portal(13), super_fast_run(12), shadow_smoke(10),
live_concert(8), earth_element(6), earth_wave(5), air_bending/raven/water_bending/
luminous_gaze(4), display_transition(3), flame/hole_transition/run_set_on_fire(2).

Fails hardest (biggest miss contributors):
- **shadow (recall 0.40, n=15)** → confuses with shadow_smoke, gas_transformation  [9 misses]
- **gas_transformation (0.20, n=10)** → shadow_smoke, giant_grab  [8 misses]
- **money_rain (0.14, n=7)** → shadow, color_rain  [6 misses]
- 9 classes at **0.00 recall**, mostly small: jump_transition(1), seamless_transition(1),
  nature_bloom(2), sakura_petals(2), wonderland(2), monstrosity(3), saint_glow(4),
  flying_cam_transition(4), giant_grab(5).
- mid: mystification(0.40,5), wireframe(0.56,9)→polygon, water_element(0.60,5)→fire_element,
  color_rain(0.62,8), cotton_cloud(0.67,6), firelava(0.67,6), polygon(0.78,9).

Confusion clusters: {shadow, shadow_smoke, gas_transformation} mutually; {wireframe,
polygon}; {water_element, fire_element}; {nature_bloom, saint_glow, wonderland}.
Most failures are one-sided classes.

## Compute reality
- **CPU (instant, already cached):** anything derived from the [223,121,768] CLS
  embeddings + core masks — set/distributional distances, whitening/decorrelation,
  temporal structure, frame weighting, kernels, pooling. A 223×223 matrix builds in
  seconds–minutes. THIS IS WHERE TO START.
- **GPU (Slurm job, ~hours):** genuinely NEW features — DINO patch/spatial tokens,
  other backbones (SigLIP, DINOv2-large/giant, CLIP), optical flow, video encoders.
  Feasible but a bigger step; escalate only if CLS-space ideas plateau.

## Constraints (no cheating)
- The exam judge is FIXED and imported (same function that judged m1a). I cannot change
  it. Only the distance matrix construction is open.
- No label leakage into the distance (no using class identity to build D).
- No NaN-ing hard classes to inflate coverage-adjusted accuracy — coverage is reported
  and must not shrink below 1.0 to "win."
- No test-set threshold tuning that wouldn't generalize; any hyperparameter must be
  justified a-priori or set by a principled/parameter-free rule.
- Certified code + shared cache are read-only.
