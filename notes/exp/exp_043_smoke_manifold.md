# exp_043 — Is there a "shadow_smoke transition manifold" in LTX-2 latent space?

Phase-1 (CPU-only) diagnostics on the 10 cached `smoke_z0` and 5 cached
`davis_gen_z0` packed-latent tensors from
`outputs/latent_pca/exp_043_latent_pca_inspection/run_0001/`.

Charts: `M1`..`M6` in that run's `charts/` directory.  Raw numbers in
`charts/manifold_summary.json`.

The motivating observation (from chart `03_smoke_unified_frame.png`): smoke
trajectories appear to "pass through a place" — i.e. converge through a
common region of latent space.  The question is whether this region is a
real shared manifold or a 2-D-PCA projection artefact.

---

## TL;DR

The chart-03 intuition has a real signal behind it, but the structure is
**weaker and more diffuse** than "shared manifold" suggests.

| Hypothesis                                                | Verdict       | Evidence                                                                            |
|-----------------------------------------------------------|---------------|-------------------------------------------------------------------------------------|
| Smoke clips share an **endpoint-displacement direction** `v_smoke = mean(z₁₅−z₀)` | **YES**       | M6: smoke proj 108±16, random 0±1, davis on same `v_smoke` only 15±11               |
| Smoke clips' **centroid trajectory** bulges out then returns (U-shape) | **YES**       | M1 drift: 0→167 at t=8 → 108 at t=15; davis monotone 0→175                          |
| Smoke clips agree **most at the middle frames** (t≈6-10)  | **YES, mild** | M1 σ(t) dips 1.02→0.84 at t=8; M4 heatmap shows clear dark square at t∈[6,10]       |
| Smoke clips share a **within-clip PC1 direction**         | **NO**        | M5 off-diag cos sim +0.003 vs davis +0.167; smoke directions are essentially independent |
| Smoke motion lives on a **low-dimensional manifold**      | **NO**        | M3 anchored-PCA top-5 EVR flat at 9/8/7/7/6%; davis has a sharp 26/21/14% structure |
| Smoke shows **more shared time-structure** than davis     | **NO — opposite** | M2 R²_time: smoke 0.13 < davis 0.22                                                 |

So the "place" you saw in chart 03 is real — it's the **centroid bulge / dispersion
dip at frames 6–10**.  But the individual smoke clips do *not* tightly trace a
shared trajectory through it; they just bunch a little closer in the middle
than at the ends.

DAVIS A_word, despite varied scene content, has **more** shared time-structure
than smoke does — almost certainly because the C2V mechanism pins the first
8 frames to the anchor clip, so all DAVIS Stage-1 generations share an
anchor-driven trajectory pattern.

---

## Diagnostic-by-diagnostic findings

### [M1] Per-frame across-clip dispersion σ(t) and centroid drift

`σ(t)` = RMS of `(z_k,t − μ(t))` across clips (one value per latent
frame).  `drift(t)` = `‖μ(t) − μ(0)‖₂`.

- **Smoke σ(t) is U-shaped**: 1.02 (t=0) → **0.84 (t=8)** → 1.04 (t=15).
  The middle frames are where clips are **least** dispersed.
- **DAVIS σ(t) is flat**: 0.80 (t=0) → ~0.72 thereafter.  DAVIS clips agree
  from t=1 onward and stay agreed.
- **Smoke drift(t) is also U-shaped**: 0 → **167 (t≈8)** → 108 (t=15).
  The smoke *centroid* moves out and partially returns.
- **DAVIS drift(t) is monotone**: 0 → 174.  The centroid moves out and stays out.

**Reading**: the smoke "place" the user saw is **the centroid bulge at t≈8**.
Smoke clips do bunch closer at this point — but only by ~20% (σ 1.02→0.84),
not by an order of magnitude.  The clip-by-clip trajectories cross through a
region; they don't superimpose.

### [M2] Variance explained by frame-index alone

`R²_time = 1 − Σ_{k,t}‖z_k,t − μ(t)‖² / Σ_{k,t}‖z_k,t − μ_global‖²`

- smoke: **0.134** (13.4%)
- davis: **0.221** (22.1%)

**Reading**: most of the variance in `z` is clip-specific, not time-driven —
and this is **more true for smoke than for DAVIS**.  This is the headline
finding that pushes back against the "shared smoke manifold" hypothesis.

### [M3] Anchored joint PCA — direction of motion away from t=0

Subtract `z_k,0` from each frame, then SVD over all 10 × 15 = 150 (smoke)
or 5 × 15 = 75 (davis) deltas.

| Component | smoke EVR | davis EVR |
|-----------|----------:|----------:|
| PC1       | **9.2 %**  | **26.4 %** |
| PC2       | 7.6 %     | 20.9 %    |
| PC3       | 6.9 %     | 14.3 %    |
| PC4       | 6.8 %     |  2.6 %    |
| PC5       | 6.3 %     |  2.1 %    |

**Reading**: davis motion concentrates in **3 dominant directions** (top-3
sum = 61.7 %), then drops sharply.  Smoke motion is **near-uniformly spread**
across many components (top-5 sum = 36.6 %, with each PC contributing
6–9 %).  In the anchored-PCA scatter plot (left of `M3.png`) the smoke
trajectories fan out radially with no shared structure; the davis ones
cluster into a few branches.

### [M4] Cross-clip frame-pair distance heatmap

`D[i,j] = mean_{k≠l} ‖z_{k,i} − z_{l,j}‖₂`

- smoke: diag mean = 298.9, off-diag = 308.4, **diag/off = 0.969**
- davis: diag mean = 243.4, off-diag = 261.1, **diag/off = 0.932**

Both ratios are close to 1 — i.e. cross-clip distance at matched `t` is
barely smaller than at mismatched `t`.  The heatmap structure is more
diagnostic than the scalar:

- **Smoke**: clear dark "square" in `i,j ∈ [6,10]` (≈270 vs ≈330 at the
  corners).  Middle-frame agreement across clips is real.
- **Davis**: a thin dark diagonal — agreement is concentrated on `i=j`
  (especially for t=2..4, the anchor frames per `notes/models/ltx2/conditioning.md`).
  Different time positions are NOT close to each other.

**Reading**: smoke clips agree at the middle frames *regardless of which
clip*.  Davis clips agree at matched-time only — the anchor-pinning
mechanism doesn't make the entire trajectory shared.  This is the
clearest visualisation of the "place" the user noticed.

### [M5] Per-clip PC1 direction cosine similarity

For each clip k, compute the PC1 of its 16 frames (sign-aligned with
`z_{k,15} − z_{k,0}`).  Compute pairwise cosine sim across clips.

- smoke off-diag mean: **+0.003** (std 0.126) — indistinguishable from null
- davis off-diag mean: **+0.167** (std 0.212) — ~35× null
- null σ for random unit vectors in R⁴⁶²⁰⁸ ≈ 0.0047

**Reading**: each smoke clip's within-clip motion direction is essentially
**independent of the others**.  Davis clips share a within-clip direction
(probably the anchor-to-free motion).

This is the strongest evidence against a "smoke direction in the PC1 sense".
The smoke transition does not reduce to one direction per clip aligned
across clips.

### [M6] Projection onto v_smoke = mean(z₁₅ − z₀)

`v_smoke` is the **mean endpoint-displacement direction**, normalised.
Project each clip's `(z_k,t − z_k,0)` onto `v_smoke`; plot per t.

- smoke clips: mean(proj at t=15) = **108.4 ± 16.2**  (CV 15%)
- davis clips: mean(proj at t=15) = 14.9 ± 11.2  (CV 75%, basically noise)
- smoke clips on random direction: -0.46 ± 1.19  (null)
- `v_smoke · v_davis` cos = 0.086 (nearly orthogonal)

**Reading**: `v_smoke` is a **real** direction — smoke clips all advance
along it strongly and consistently, davis clips don't, random directions
don't.  It's a **smoke-specific, low-CV summary** of the transition.

The picture from M3 + M5 + M6 is: per-clip *trajectory paths* in latent
space are mostly idiosyncratic, but the **net displacement** ends up in
a roughly common direction.  Different roads, same destination region.

---

## Synthesis — what is "the place" in chart 03?

Combining M1, M4, M6:

1. **The centroid of smoke clips' latents moves out then partially back**
   — drift peaks at t=8 (167 units) and falls to 108 at t=15.
2. **At t≈6–10, smoke clips bunch ~20% closer** (σ 1.0 → 0.84; cross-clip
   distance dark square in M4).
3. **The net `z₁₅ − z₀`** of each clip aligns to a smoke-specific direction
   `v_smoke` (CV 15%, well above noise floor and not shared with davis).

So the chart-03 visual is showing the **centroid bulge + middle-frame
dispersion-dip**, projected into 2-D.  In full-D the bulge exists, the
dispersion-dip exists, but **trajectories don't collapse onto a 1-D curve**
— smoke transitions remain high-dimensional and partly scene-specific.

The cleanest 1-D handle we have is `v_smoke`.  The cleanest *trajectory*
handle is the per-t centroid sequence `{μ(t)}_{t=0..15}`.

---

## "Disentangling smoke from background/objects"

The standard decomposition would be:

```
z_{k,t} = z̄_k          # per-clip mean = scene content
        + (μ(t) − μ̄)    # per-time centroid drift = shared transition signal
        + ε_{k,t}        # residual = clip-specific motion
```

For smoke:

- `z̄_k` carries the bulk of the variance (clip-specific scene, ~80 % of
  signal from M2 inverse: V_within is 87 % of V_total).
- `μ(t) − μ̄` is the shared time signal — about 13 % of variance.  Best
  summarised by `v_smoke` (mean displacement) + the U-shape modulator.
- `ε_{k,t}` is the residual within-clip motion that isn't shared (~63 %
  of V_within = ~55 % of V_total).  This is the "scene-specific motion" —
  e.g. one clip has a wind gust, another has a person walking.

If we want to **transfer smoke to a non-smoke clip**:
- Adding `α · v_smoke` to a target latent biases it toward the smoke
  endpoint region.  But the target's `ε` term will not pick up the
  middle-frame bulge structure.  Likely result: a flat "smoke-tinted"
  appearance, no actual transition over time.
- Adding the full per-t centroid offset `μ(t) − μ̄_smoke` to a target's
  per-t latents *might* induce a time-varying smoke-like change.  But this
  is uniformly applied across spatial tokens, so it'd probably wash out
  spatial structure (smoke is *localised*, not global).

A cleaner disentanglement would require **paired ablations** (same
scene with and without smoke), which the current shadow_smoke set
doesn't provide directly.

---

## Phase 2 — what to actually verify before claiming we can "use the manifold"

These are GPU experiments.  Order by cost.

### P2.a — Decode the centroid trajectory μ(t)

**Why**: if `{μ(t)}_t` is a meaningful "average smoke transition" in latent
space, decoding it directly should produce *a* smoke-like video.  Either
result is informative.

**How**: take the per-t centroid `μ(t)` (computed offline from the 10 smoke
clips, shape [16, 361, 128]).  Unpack to a 5-D latent.  Pass through
`pipe.vae.decode`.  Save the result.

**Cost**: one VAE forward — minutes on an A100.

**Predicted outcome** (low confidence): some grey-haze-like content where
smoke is dominant in the originals, but mushy / lacking texture because
averaging across clips smooths out spatial detail.

### P2.b — Decode `latent_baseline + α · v_smoke`

**Why**: does `v_smoke` carry smoke-like *visual* content, or is it just a
direction in latent space with no semantic meaning per se?

**How**: take a DAVIS A_word baseline latent (or a smoke clip's t=0 frame
replicated for all t).  Add `α · v_smoke` for α ∈ {0, 0.5, 1.0, 2.0, 4.0}.
Decode each.  Counter-check with the same α on a random unit vector
(per-clip mean magnitude as norm scaling).

**Cost**: ~10 decodes — minutes.

**Predicted outcome** (low confidence): non-trivial *something* appears at
α=1 or 2 (since v_smoke has ‖proj‖≈108 at t=15, α=1 should already give
a measurable effect).  If smoke-like haze emerges and the random-direction
control doesn't → v_smoke carries smoke-like content.

### P2.c — Token-level PCA (instead of frame-level)

**Why**: the frame-level analysis treats each 19×19×128 frame as a single
46208-D point.  But each token is a 128-D point in the VAE's per-pixel
channel space — the actual unit of LTX-2 representation.  A "smoke
direction" might be much cleaner at the token level (single direction per
token across many tokens) than at the frame level (one direction per
46208-D frame).

**How**: stack all smoke frame tokens [10 × 16 × 361, 128] = [57760, 128].
SVD.  Top components are interpretable as channel-wise smoke directions.
Same for davis, compare.

**Cost**: ~30 s of CPU.  No GPU needed.  **Worth doing in Phase 1.b.**

### P2.d — Per-time-band PCA

**Why**: the U-shape in M1 says the middle frames are where smoke clips
agree.  Do a separate PCA on just frames t ∈ [6,10] vs frames t ∈ [0,2] +
[13,15].  If the middle-band subspace is dramatically richer / lower-rank,
it confirms that the "smoke state" lives there and the endpoint states
are scene states.

**How**: stack smoke frames per band; SVD each.  Compare scree.

**Cost**: ~30 s of CPU.  **Worth doing in Phase 1.b.**

### P2.e — Subspace projection: how much of a non-smoke clip lies in the smoke subspace?

**Why**: pick the top-k anchored-PCA components from smoke (k=5 or 10).
For each davis A_word frame, compute the fraction of its (frame − davis
clip mean) variance that lives in this subspace.  If it's substantially
higher for the *middle* davis frames than for the endpoint davis frames,
the smoke subspace is picking up "general motion" structure not "smoke";
if not, the subspace is smoke-specific.

**How**: project, sum, compare.  ~10 s.  **Worth doing in Phase 1.b.**

---

## Phase 3 — "make the model create natural things around the manifold"

Only attempt after Phase 2.a–c clarify what `v_smoke` (or its
generalisation) actually represents.

### P3.a — `v_smoke` as a conditioning offset

Add `α · v_smoke[i]` to the mid-frame packed-latent conditioning tokens
of a C2V generation.  Eval: does the generated clip show a smoke-like
mid-clip transition while preserving start/end content?

### P3.b — Centroid trajectory as a target denoising path

Modify the sampler: at each diffusion step, push the per-t partial latent
toward `μ(t) − μ̄`.  Strength schedule could mirror the late-σ dominance
finding from exp_041 (most of the effect was at steps 27–39).

### P3.c — Feature injection at block_out

Re-use exp_041's block_out injection machinery but inject *smoke-feature*
deltas (derived from PCA of smoke-clip block_out activations on the same
clips).  This is the natural continuation of the exp_040/041 line and
inherits the +0.64 PSNR baseline we already have.

---

## What I am NOT claiming

- That there is a "smoke manifold" in the strict (low-D, well-defined,
  scene-invariant) sense.  M3 and M5 argue against it.
- That `v_smoke` carries enough information to *generate* smoke
  transitions.  We have only shown it exists as a direction with low CV
  across the 10 clips — its semantic content is unverified (needs P2.b).
- That the U-shape in M1 implies a return-to-start dynamic in the clips.
  The mean centroid returns part-way; the individual clips don't
  necessarily return.  This is an aggregate effect of how the centroid
  cancels out clip-specific noise.
- That 10 smoke clips is enough to estimate any of these directions
  reliably.  CV of 15% on `v_smoke` is OK; smaller subspaces would
  require more clips.

---

## Files

- `experiments/exp_043_latent_pca_inspection/manifold_diagnostics.py` — script
- `outputs/latent_pca/exp_043_latent_pca_inspection/run_0001/charts/M[1-6]_*.png`
- `outputs/latent_pca/exp_043_latent_pca_inspection/run_0001/charts/manifold_summary.json`
