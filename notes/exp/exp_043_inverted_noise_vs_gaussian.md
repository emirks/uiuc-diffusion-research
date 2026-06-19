# exp_043 — Inverted noise z1 vs matched Gaussian: where is the smoke signature?

Analysis of RF-inverted noise `z1` (and source latent `z0`) from
`exp_033_ltx2_rf_inv_drop1/run_0001`, against a matched white-Gaussian null.
Script: `experiments/exp_043_latent_pca_inspection/inverted_noise_vs_gaussian.py`.
Charts + numbers: `outputs/latent_pca/exp_043_inverted_noise_vs_gaussian/`
(`run_0002` = z1, `run_0003` = z0 control), per orientation group.

---

## TL;DR — the stated hypothesis is REFUTED

The hypothesis was: *z1's deviation from Gaussian = the shadow-smoke signature,
concentrated in the free-middle latent frames.* **The data says the opposite.**

- **z1's free-middle frames (4..12) are essentially WHITE GAUSSIAN.** RF-inversion
  has *erased* structure there, not concentrated it.
- **z1's deviation from Gaussian lives entirely in the ANCHOR frames (0-3, 13-15)**,
  which were hard-pinned to slices of `z0` during inversion. That "deviation" is
  just leftover `z0` (source-latent) structure, not a synthesised smoke prior.
- The actual smoke structure is a property of **`z0`** (the VAE encoding of the
  real clip), where it is present in *all* frames and is partly shared across clips.

So you cannot lift a "smoke signature" out of z1's free-middle — there is nothing
there beyond white noise. The signature must be sourced from `z0`.

Replicated identically across the portrait (n=5), landscape (n=4) and square
(n=1) orientation groups — i.e. it is not a geometry artefact.

---

## Geometry (verified, not assumed)

Packed `[1, N, 128]`, P=P_t=1, so the diffusers `_pack_latents` permute gives
token order `n = f·(H·W) + h·W + w`. Unpack = `reshape(F=16, H, W, 128)` with the
**clip's own** (H,W). Orientation groups (from `render_HxW/32`):
portrait 22×16 = clips 0,2,3,6,8; landscape 16×22 = 1,5,7,9; square 19×19 = 4.
Portrait and landscape **share N=5632 with swapped H,W** — grouping by N would
scramble spatial structure, so all spatial analysis is within-group only.

Free-middle latent frames = **4..12** (9 frames); anchors = **0-3, 13-15**.
Verified from `exp_033/run.py:end_clip_index`: n_lat=(121-1)//8+1=16,
k_lat=(25-1)//8+1=4 → start anchor 0-3, end anchor index 16-4=12, drop1 frees
frame 12 → it joins the free band.

Nulls: white N(0,1) of identical shape, plus a variance-matched white null
(scaled to z1's per-(frame,channel) std). Structural metrics use raw values
(mean-removal only); marginal-shape metrics standardize per channel.

---

## Numbers (z1 = inverted noise, z0 = source latent control)

Per-metric, free-middle vs anchor vs white-null. Three orientation groups agree;
portrait shown, landscape/square within ±0.05 unless noted.

| metric (portrait)                  | z1 free-mid | z1 anchor | z0 free-mid | z0 anchor | white null |
|------------------------------------|-------------|-----------|-------------|-----------|------------|
| excess kurtosis (per-channel mean) | **+0.078**  | +0.627    | +0.479      | +0.604    | −0.011     |
| spatial autocorr @ lag 1           | **+0.019**  | +0.450    | +0.544      | +0.451    | −0.002     |
| temporal adjacent-frame corr       | **+0.017**  | +0.617    | +0.458      | +0.620    | ~0.000     |
| low-freq power fraction            | **0.22**    | 0.65      | 0.70        | 0.72      | 0.135      |
| cross-clip free-struct cosine      | **+0.030**  | —         | **+0.258**  | —         | ~0.000     |

Landscape z1: free-mid kurt +0.102, anchor +0.814; free-struct cos +0.042 vs z0
+0.160. Square z1: free-mid kurt +0.039, autocorr +0.008.

### Reading

1. **Marginal (Gaussianity of values).** z1 free-middle is marginally Gaussian
   (kurt +0.08, skew 0.04 — indistinguishable from the white null's −0.01/0.03).
   z1 anchors are heavy-tailed (kurt +0.6–0.8). z0 is heavy-tailed *everywhere*.
   QQ-plots (chart 06) for z1's highest-|kurtosis| free-middle channels are
   near-straight; anchor channels would bow.

2. **Spatial structure.** z1 free-middle has near-zero neighbour autocorrelation
   (0.019) and a near-flat radial power spectrum — i.e. white. z1 anchors carry
   strong short-range correlation (0.45) and an excess of low-frequency power
   ("spectral collapse", the literature's signature of inverted noise). z0 carries
   that low-freq/correlated structure in every frame.

3. **Temporal structure.** z1 free-middle frames are temporally incoherent
   (adjacent-frame corr 0.017 ≈ white). z1 anchors are highly coherent (0.62) —
   but that is the clamped z0 content, not solver-synthesised dynamics. z0
   free-middle is coherent (0.46), consistent with real smoke evolving smoothly.

4. **Localization (chart 04).** The low-frequency power fraction per latent frame
   is a clean step function for z1: ~0.65–0.72 on anchor frames 0-3 & 13-15,
   crashing to ~0.22 (just above white 0.135) across free-middle 4-12. Frame 12
   (the drop1-freed end anchor) shows a partial bump (0.33) — it is freed but
   sits adjacent to the clamped end anchor and inherits a little structure.
   For z0 the fraction is flat-high (~0.7) across all 16 frames. **This single
   chart is the cleanest statement of the result.**

5. **Cross-clip sharing.** The free-middle structured-map cosine across clips is
   +0.030 for z1 (barely above the ~0 null) but +0.258 for z0. So whatever weak
   structure z1's free-middle has is mostly idiosyncratic; the *shared* component
   lives in z0. (Cross-clip radial-spectrum cosine is ~0.99 for both, but that is
   dominated by the trivially-shared spectral *envelope*, not a smoke-specific
   pattern — see "what would refute" below.)

---

## Why this happens (mechanism)

RF-Solver inversion integrates the free tokens along the flow ODE from data
(σ→0) toward noise (σ→σ_max). With CFG=1 and the anchors hard-pinned + x0-clamped
at every step (see `conditioning.md §14-b`), the free-middle tokens are driven to
the model's noise prior, which **is** standard Gaussian — that is exactly what
makes z1 a valid reconstruction seed. The transition information needed to
rebuild the middle is supplied at regen time by the *anchors* + the deterministic
solver trajectory, not stored as structure in the free-middle noise. This is the
same mechanism the RF-inversion postmortem named: the free-middle is reconstructed
from anchor velocity coupling, not from a structured middle seed.

This also explains exp_044's finding (`notes/exp/exp_044…`) that the recon→regen
gap is solver self-consistency: the free-middle of z1 carries no injectable smoke
prior, so a production sampler from fresh Gaussian + generic prompt has nothing
smoke-specific to latch onto.

---

## What would REFUTE this conclusion (self-challenge) — and whether it does

- *"z1 free-middle structure is real but only visible in higher-order / non-radial
  statistics."* Tested kurtosis (4th moment), spatial autocorr, temporal corr,
  and low-freq fraction — all at the white-null floor. A structured smoke blob
  would raise low-freq fraction and autocorr; it does not. **Not refuted.**
- *"The variance-matched null hides a variance signature."* Per-frame energy of
  z1 free-middle is ~1.0 (white), with mild bumps (1.1–1.15) on a few frames —
  not a localized smoke energy excess. **Not refuted.**
- *"Cross-clip radial-spectrum cosine is ~0.999 → shared signature after all."*
  This is the spectral *envelope* (all latents share a similar 1/f-ish falloff);
  the discriminating test is the spatially-aligned free-struct cosine, which is
  +0.03 (z1) vs +0.26 (z0). The shared component is in z0. **Not refuted.**
- The one place the hypothesis is *not* fully dead: frame 12 (drop1-freed) shows
  a small structure bump. If anything is injectable from z1 itself, it is the
  boundary frames adjacent to anchors — but the effect is small and clip-specific.

Conclusion stands: **z1's free-middle deviation is not the smoke signature.**

---

## Actionable recommendation for the downstream goal

The downstream goal (add a smoke signature to a fresh Gaussian sample at
production) should **NOT** try to extract the signature from z1's free-middle —
it is white there. Instead:

1. **Source the signature from `z0`, not z1.** z0's free-middle carries the
   structure (kurt +0.48, autocorr +0.54, temporal +0.46, low-freq 0.70) and it
   is partly shared across clips (free-struct cosine +0.26). The injectable smoke
   prior is a **structured, low-frequency, temporally-coherent field in z0's
   free-middle frames**, geometry-safe per orientation group.

2. **Represent it as a per-orientation, low-rank low-frequency field.** Because
   z0's structure is dominated by low spatial frequencies and short-range spatial
   + adjacent-frame correlation, a compact representation is the **mean structured
   field** `s(f,h,w,c) = mean_clips[ z0[free] − per-(frame,channel) DC ]`
   computed *within an orientation group*, optionally truncated to its top few
   low-freq / PCA components. This matches the +0.26 cross-clip cosine: there is a
   real shared component to capture.

3. **Inject at the right σ band, not as a raw additive on the seed.** A structured
   field added to a fresh Gaussian seed will be re-Gaussianized by the same flow
   dynamics that whitened z1's free-middle (this is precisely why z1 is white).
   The effective lever is to bias the *trajectory* in the late-σ content phase
   (steps ~27-39, per exp_041 / `denoising_schedule.md`), e.g. as a velocity /
   x0 guidance toward z0's structured free-middle field — consistent with the
   exp_047 velocity-guide and exp_046 perceptual-inject lines.

4. **Geometry rule for transfer.** Keep portrait and landscape signatures
   separate (they live on swapped grids). For cross-orientation transfer use only
   orientation-invariant summaries (radial low-freq band energy, per-frame
   coherence schedule), never a raw per-position field.

Net: the "smoke signature" is a real, partly-shared, low-frequency + temporally-
coherent structure — but it is in **z0's free-middle**, and z1's free-middle is
the wrong place to look (it is white by construction).

---

## Files

- `experiments/exp_043_latent_pca_inspection/inverted_noise_vs_gaussian.py`
- `outputs/latent_pca/exp_043_inverted_noise_vs_gaussian/run_0002/` — z1, all groups
- `outputs/latent_pca/exp_043_inverted_noise_vs_gaussian/run_0003/` — z0 control
- charts `01_marginal` / `02_spatial` / `03_temporal` / `04_localization` /
  `05_crossclip` / `06_qq` × {portrait,landscape,square}; `summary.json`

## Literature grounding (techniques chosen)

- Inverted/edited noise is known to deviate from Gaussian and that deviation is
  what carries source structure: DDIM-inversion non-Gaussianity & autocorrelation
  regularization (Pix2Pix-Zero; "Edit-Friendly DDPM Noise Space"; "There and Back
  Again", arXiv:2410.23530), RF-Solver-Edit (arXiv:2411.04746), DNAEdit
  (arXiv:2506.01430). Autocorrelation = the canonical editability/Gaussianity
  probe (white noise → Kronecker-delta autocorr) → motivates metrics 2 & 3.
- "Spectral Collapse in Diffusion Inversion" (arXiv:2602.13303): inverted noise
  shows **excess low-frequency power** vs flat white spectrum → motivates the
  radial power spectrum + per-frame low-freq-fraction localization (metric 4).
- Power spectrum alone is insufficient to characterise non-Gaussianity; need
  higher-order moments (skew/kurtosis) for phase/structure → motivates metric 1
  (PMC5050006 on high-order image statistics; 1/f² isotropy PMC2924385).
