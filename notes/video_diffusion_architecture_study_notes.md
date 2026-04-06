# Video Diffusion Model Architectures - Study Notes

## 1. Scope and Main Goal
These notes consolidate the discussion around latent video diffusion / flow models in an LTX-like setup, with special emphasis on:

- what the different latent spaces actually are,
- what the model predicts at each denoising step,
- which geometries were being mixed in the earlier questions,
- where semantic reasoning is most likely to make sense,
- and how to rigorously formulate the user's VC-oriented research ideas.

**Core thesis:** there is only **one true evolving latent state** during generation - the noisy VAE latent - while the patch space and transformer hidden space are auxiliary representation spaces used to compute how that latent should move.

---

## 2. The Three Geometries That Must Be Kept Separate
A major source of confusion is that the word *latent* often silently mixes three independent axes.

### 2.1 Video-time geometry
This is the frame axis inside the generated latent video.

- Index: `p = 1, 2, ..., P`
- Meaning: physical temporal evolution of the video
- Typical quantity:
  - `Δ_p z^(τ) = z_(p+1)^(τ) - z_p^(τ)`

This measures **frame-to-frame change** at a fixed denoising state.

### 2.2 Denoising-time geometry
This is the diffusion / flow timestep.

- Index: `τ` or `t`
- Meaning: how noisy vs clean the latent is
- Large `τ`: very noisy, coarse/global structure
- Small `τ`: near-clean, fine detail

Typical quantity:
- `Δ_τ z_p = z_p^(τ-Δτ) - z_p^(τ)`

This measures **trajectory evolution across denoising steps**, not across video frames.

### 2.3 Representation geometry
This is the geometry of the vector space being used.

- Examples: VAE latent space, patch space, transformer hidden space
- This is where norms, angles, directions, subspaces, and curvature live

**Important:** distances and directions are only meaningful relative to a specific representation space.

### 2.4 Main correction
When writing something like `Δu_p ≈ Δu_(p+1)`, one must specify:

1. which representation space `u` belongs to,
2. which denoising timestep is fixed,
3. whether the comparison is about video-time smoothness or semantic consistency.

Without those, the statement is ambiguous.

---

## 3. Spaces in a Direct Video Diffusion / Flow Model

## 3.1 Pixel space
Notation:
- `x ∈ R^[B x 3 x F x H x W]`

Meaning:
- actual video in pixel coordinates
- used at the input (encode) and output (decode)

This is **not** a latent space.

## 3.2 VAE latent space
Notation:
- clean latent: `z_0`
- noisy latent at denoising step `t`: `z_t`
- shape: `R^[B x C x F' x H' x W']`

Meaning:
- compressed video representation
- the same space in which the Video-VAE encodes videos
- the same space in which the diffusion / flow process evolves states

**Critical point:** denoising happens in this space. The evolving generative state is `z_t`.

## 3.3 Patch / token-pre-embedding space
Notation:
- patch vectors: `p_i`
- stacked patches: `P ∈ R^[B x N x patch_dim]`
- `patch_dim = C x pT x pH x pW`

Meaning:
- produced by patchifying the VAE latent
- exists before the embedding projection
- exists again after the output projection, before unpatchify

This is a **structural representation space**, not a deeply semantic one.

## 3.4 Transformer hidden space
Notation:
- input tokens after embedding: `h^0`
- hidden states: `h^l`, `l = 1,...,L`
- shape: `R^[B x N x D]`

Meaning:
- representation space where the transformer operates
- likely the most semantic space in the system
- style/object/content directions are more likely to emerge here than in raw VAE latent space

## 3.5 Prediction / vector-field space
Notation:
- `ε_θ(z_t, t, c)` for noise prediction
- `v_θ(z_t, t, c)` for velocity prediction
- same shape as VAE latent: `R^[B x C x F' x H' x W']`

Meaning:
- not a latent state
- not the next latent itself
- a field defined over latent space that tells the sampler how the current latent should move

**Deep point:** the model predicts an update direction / field, and the sampler integrates it to produce the next latent state.

---

## 4. End-to-End Inference Pipeline (LTX-like)

### 4.1 Initialization
Start from Gaussian noise in VAE latent space:

- `z_T ~ N(0, I)`

This is the initial state of the generative trajectory.

### 4.2 Denoising loop
For each timestep `t = T, ..., 1`:

1. Current noisy latent:
   - `z_t ∈ R^[B x C x F' x H' x W']`
2. Patchify:
   - `z_t -> P ∈ R^[B x N x patch_dim]`
3. Embed:
   - `h^0 = P W_e^T + b_e`
   - `h^0 ∈ R^[B x N x D]`
4. Add timestep / conditioning / positional information:
   - timestep embedding
   - text / image / keyframe conditioning
   - spatiotemporal positional encoding
5. Transformer stack:
   - `h^0 -> h^1 -> ... -> h^L`
6. Output head:
   - `P_hat = h^L W_o^T + b_o`
   - `P_hat ∈ R^[B x N x patch_dim]`
7. Unpatchify:
   - `P_hat -> prediction ∈ R^[B x C x F' x H' x W']`
8. Sampler update:
   - `z_(t-1) = Update(z_t, prediction)`

### 4.3 Final decode
After the last step:

- `z_0 -> VAE decoder -> x_hat`

### 4.4 Crucial correction
The transformer does **not** directly output the next latent. It outputs a prediction with latent-matching shape, but semantically that tensor is:

- predicted noise,
- or velocity,
- or residual,
- or occasionally an `x_0` estimate,

depending on the formulation.

The sampler then combines the current latent and the model prediction to produce the next latent.

---

## 5. Notation Table

| Symbol | Meaning | Typical shape | Notes |
|---|---|---:|---|
| `x` | pixel-space video | `[B, 3, F, H, W]` | real video |
| `z_0` | clean VAE latent | `[B, C, F', H', W']` | encoded video latent |
| `z_t` | noisy latent state | `[B, C, F', H', W']` | main evolving state |
| `p_i` | patch vector | `[patch_dim]` | raw latent patch |
| `P` | stacked patch vectors | `[B, N, patch_dim]` | patchified latent |
| `patch_dim` | patch size in flattened latent coords | `C x pT x pH x pW` | structured local chunk |
| `h^0` | embedded input tokens | `[B, N, D]` | transformer input |
| `h^l` | transformer hidden states | `[B, N, D]` | same dimensional space, different representations |
| `ε_θ(z_t,t,c)` | predicted noise | latent-shaped | DDPM-style |
| `v_θ(z_t,t,c)` | predicted velocity / update | latent-shaped | flow / v-pred style |
| `c` | conditioning | task-dependent | text, images, keyframes, etc. |
| `t` or `τ` | denoising timestep | scalar | noisy -> clean axis |
| `p` | video-frame index | scalar | frame-time axis |
| `W_e` | embedding projection | `[D, patch_dim]` | shared across all tokens |
| `W_o` | output projection | `[patch_dim, D]` | shared across all tokens |

### Recommended notation for VC work
Use:
- `z_t(p)` for VAE latent at frame index `p` and denoising step `t`
- `h_t^l(p)` for transformer hidden representation at frame `p`, step `t`, layer `l`
- `v_θ(z_t, t, c)` for predicted update field

This makes the geometry explicit.

---

## 6. Projection, Patchification, and Token Processing

## 6.1 Patchify
Patchify the latent into local spatiotemporal blocks:

- `p_i ∈ R^[C x pT x pH x pW]`
- flatten each patch to `R^[patch_dim]`

Stacking over all patches:
- `P ∈ R^[B x N x patch_dim]`

## 6.2 Embedding projection
Each patch is projected independently with shared weights:

- `h_i^0 = W_e p_i + b_e`

Batched form:
- `H^0 = P W_e^T + b_e`

**Important:** this is token-wise but parallel, not sequential.

## 6.3 Output projection
From final transformer representation:

- `p_hat_i = W_o h_i^L + b_o`

Batched form:
- `P_hat = H^L W_o^T + b_o`

Then unpatchify:
- `P_hat -> prediction ∈ R^[B x C x F' x H' x W']`

## 6.4 Compact user insight
**Correct insight:** after patchifying, a token has dimension `[N, C x pT x pH x pW]`; after embedding, it becomes `[N, D_model]`.

## 6.5 Another compact user insight
**Correct insight:** the same embedder is applied independently to every token; semantics do not come from this linear projection alone, but emerge after attention and deeper nonlinear processing.

---

## 7. What the Model Predicts vs What the Sampler Does

A central conceptual correction from the discussion is:

- wrong: `transformer -> next latent`
- correct: `transformer -> update field`, and `sampler -> next latent`

## 7.1 DDPM-style view
The model predicts noise:
- `ε_θ(z_t, t, c)`

The sampler computes:
- `z_(t-1) = f(z_t, ε_θ(z_t, t, c))`

## 7.2 Flow-matching / velocity view
The model predicts velocity:
- `v_θ(z_t, t, c)`

The sampler integrates:
- `z_(t-Δt) = z_t + v_θ(z_t, t, c) Δt`

## 7.3 Deep interpretation
The prediction lives in the same tensor geometry as the latent, but it is not itself the latent. It is best thought of as a **vector field over latent space** or a local tangent direction that tells the system how to move.

## 7.4 Compact user insight
**Correct insight:** the model output should indeed have the same shape as the VAE latent, because it parameterizes an update on that space.

---

## 8. Editing vs Direct Generation

## 8.1 Direct generation
- start from Gaussian noise in VAE latent space
- denoise to `z_0`
- decode to pixels

## 8.2 Video editing / image-to-video / video-to-video
Editing does **not** change only the initial starting point. It can also change:

1. initial latent state,
2. amount of injected noise,
3. conditioning signals,
4. guidance scale,
5. trajectory bias during denoising.

So a better decomposition is:

- different initial latent
- plus different conditioning
- plus different guidance forces during the denoising trajectory

---

## 9. Rigorous Evaluation of the Research Questions

## 9.1 Question 1: What does `Δu_p ≈ Δu_(p+1)` mean?
Suppose:
- `u_p ∈ R^d`
- `Δu_p = u_(p+1) - u_p`

Then `Δu_p ≈ Δu_(p+1)` means the trajectory is locally close to a straight line with near-constant first derivative in that specific representation space.

This supports:
- similar latent displacement,
- low local curvature,
- locally straight latent evolution.

It does **not** automatically imply semantic similarity.

### Proper decomposition
- `||Δu_p||` = magnitude of change
- `cos(Δu_p, Δu_(p+1))` = directional consistency
- `||Δu_(p+1) - Δu_p||` = discrete acceleration / curvature

### Verdict
- similar movement? **maybe**
- similar change? **yes, geometrically**
- similar meaning? **only if the space is semantically aligned**

### Best wording
> `Δu_p ≈ Δu_(p+1)` indicates local linearity / low curvature of the latent trajectory in the chosen space.

## 9.2 Question 2: Are there semantic directions / axes?
Likely yes, but one must be careful about **which** space.

### Most plausible answer
Semantic directions are more likely to be meaningful in:
- deeper transformer hidden representations `h^l`,
- or carefully chosen semantic bottleneck spaces,
than in raw VAE latents alone.

### Timestep dependence
A strong and likely correct intuition is that semantic sensitivity is timestep-dependent:
- early timesteps: coarse layout, global semantics
- late timesteps: texture, local detail

So the right object is not a single global direction `v`, but something like:
- `v(t)` or
- a local tangent subspace `T_(z,t) M`

### Can one increase variance along such directions?
In principle yes, but it can easily push samples off-manifold if the direction is only locally valid.

A better version is:
> bias the score / velocity / posterior update inside a trusted local subspace rather than globally increase variance.

### Best wording
> Semantic sensitivity appears anisotropic and timestep-dependent.

## 9.3 Question 3: Are concepts arranged as islands?
The “islands” picture is too crude for serious use.

A more defensible picture is:
- curved latent manifold,
- entangled factors,
- locally smoother directions for some attributes than others,
- representation-dependent and timestep-dependent structure.

### Styles vs objects
A reasonable prior:
- style-like attributes are often more globally editable,
- object identity/content is more entangled and data-dependent.

### Timestep-dependent structure
Not literally separate islands, but different timesteps can emphasize different factors:
- early: broad semantic / compositional aspects
- late: appearance / texture / fine detail

### Best wording
> Concept organization is better thought of as a curved, factor-entangled manifold with timestep-dependent emphasis, not a collection of isolated semantic islands.

## 9.4 Question 1/2(a): Minimize direction change of consecutive `Δu`s?
This is a legitimate smoothness prior.

Possible objectives:
- `1 - cos(Δu_p, Δu_(p+1))`
- `||Δu_(p+1) - Δu_p||^2`

### Scientific verdict
This can encourage:
- smooth latent evolution,
- low curvature,
- temporal coherence.

But it does **not** guarantee:
- semantic correctness,
- useful semantic bridging,
- interesting VC transitions.

### Main risk
Over-smoothing can collapse transitions into bland dissolves.

### Better hypothesis split
- H1: straight latent trajectories improve temporal coherence - plausible
- H2: straight latent trajectories improve semantic transition quality - not guaranteed

### Better VC objective
A richer objective should separate:
- endpoint consistency,
- temporal smoothness,
- semantic progress.

For example:
- `L_VC = λ1 L_endpoint + λ2 L_smooth + λ3 L_semantic_progress`

### More interesting extension
Penalize curvature only in a semantic subspace rather than globally:
- `L_curve = Σ_p || P_s (Δu_(p+1) - Δu_p) ||^2`

That is more defensible than globally forcing straight trajectories.

## 9.5 Question 4: Off-manifold issue and biased sampling
This was the strongest part of the user's intuition.

### Intuition
Instead of:
- sample on-manifold,
- then push the sample using an external gradient step that ignores the generative model geometry,

one can:
- bias the model's own predicted score / noise / velocity / posterior update,
- so that steering happens through the model's native dynamics.

### Rigorous verdict
This is fundamentally correct in spirit.

It does not guarantee perfect manifold preservation, but it is much more principled than blind latent optimization because it remains informed by the model's learned local geometry.

### Three steering regimes
1. **Post-hoc latent optimization**
   - `u <- u + η ∇J(u)`
   - least manifold-aware
2. **Guided model update**
   - `v_tilde(u,t) = v_θ(u,t,c) + λ g(u,t)`
   - better, because it modifies the model-defined field
3. **Projected / tangent-space guidance**
   - `v_tilde(u,t) = v_θ(u,t,c) + λ P_(T_u M_t) g(u,t)`
   - most principled

### Best wording
> Biased sampling should be viewed as manifold-aware steering, not guaranteed manifold preservation.

### Key conclusion
Your approach is aligned with an active research direction: use the model's own predicted dynamics as the steering substrate rather than blindly optimizing raw latent coordinates.

---

## 10. Where the User's Ideas Make the Most Sense

## 10.1 Measuring `Δ`
Three candidates:

1. `Δz_t(p) = z_t(p+1) - z_t(p)`
   - useful for compressed video geometry
   - likely more motion/appearance-oriented than semantic
2. `Δh_t^l(p) = h_t^l(p+1) - h_t^l(p)`
   - more promising for semantic trajectory analysis
3. `v_θ(z_t, t, c)`
   - best object for steering, guidance, and manifold-aware control

## 10.2 Strong practical recommendation
For semantic transition research, do **not** rely only on raw VAE latent differences. Prioritize:
- deep transformer hidden features for semantics,
- vector-field steering for control,
- timestep-conditioned analysis.

## 10.3 Compact user insight
**Correct insight:** if one wants to control generation, it is often better to intervene in the predicted update field or in the sampler, rather than directly manipulate the raw latent state.

---

## 11. Recommended First Experimental Program

Before control, start with measurement.

1. Generate many VC examples with a fixed pretrained model.
2. For each sample, extract trajectories in multiple spaces:
   - VAE latent `z_t`
   - transformer features `h_t^l`
   - possibly reduced PCA/SVD subspaces
3. Measure:
   - `||Δu_p||`
   - `cos(Δu_p, Δu_(p+1))`
   - curvature `||Δu_(p+1) - Δu_p||`
   - endpoint consistency
   - CLIP / DINO-like semantic progress
   - optical-flow smoothness
4. Test correlations against human judgments.
5. Slice everything by denoising timestep.

Only then move to control.

### Safer first control ideas
- weak early-step biasing,
- subspace-projected biasing,
- predictor-corrector or SMC-style steering,
- small scheduled guidance rather than large direct latent updates.

---

## 12. Compact Notes of Correct User Insights

- The denoising trajectory evolves in **the same VAE latent space** that the Video-VAE uses to encode videos.
- After patchifying, token dimension is `[N, C x pT x pH x pW]`; after embedding, it is `[N, D_model]`.
- The embedding is a **shared linear map** applied independently to all tokens.
- The transformer layers do **not** define totally different spaces; they define progressively transformed representations in the same hidden dimensionality.
- The output head must map back to latent-matching dimensions, because the prediction parameterizes an update over VAE latent space.
- The transformer output should be interpreted as a **prediction field**, not directly as the next latent.
- The sampler is an essential part of the dynamics: `prediction + current latent -> next latent`.
- Your off-manifold concern is scientifically real, and steering through the model's own predictive field is more principled than blind test-time gradient steps.

---

## 13. One-Page Cheat Sheet

### 13.1 The spaces
- Pixel space: `x ∈ R^[B x 3 x F x H x W]`
- VAE latent state space: `z_t ∈ R^[B x C x F' x H' x W']`
- Patch space: `P ∈ R^[B x N x patch_dim]`
- Transformer hidden space: `h^l ∈ R^[B x N x D]`
- Prediction field: `ε_θ(z_t,t,c)` or `v_θ(z_t,t,c)`

### 13.2 The three geometries
- Video-time geometry: frame index `p`
- Denoising-time geometry: timestep `t` or `τ`
- Representation geometry: the chosen vector space (`z`, `P`, `h`, etc.)

### 13.3 Canonical generation pipeline
1. `z_T ~ N(0, I)`
2. patchify `z_t`
3. embed to `h^0`
4. transformer `h^0 -> ... -> h^L`
5. output projection `h^L -> P_hat`
6. unpatchify `P_hat -> prediction`
7. sampler update `z_t -> z_(t-1)`
8. decode `z_0 -> x_hat`

### 13.4 Key formulas
- Embedding: `h_i^0 = W_e p_i + b_e`
- Output head: `p_hat_i = W_o h_i^L + b_o`
- Frame difference in latent space: `Δz_t(p) = z_t(p+1) - z_t(p)`
- Frame difference in semantic space: `Δh_t^l(p) = h_t^l(p+1) - h_t^l(p)`
- Curvature proxy: `||Δu_(p+1) - Δu_p||`

### 13.5 Interpretations
- `Δu_p ≈ Δu_(p+1)` means **low curvature / locally straight evolution** in the chosen space.
- It does **not** automatically mean semantic similarity.
- Semantic directions likely exist, but are more likely to appear in deeper hidden features than in raw VAE latent space.
- Semantic influence is likely timestep-dependent.

### 13.6 What to intervene on
- For **measurement of semantics**: use deeper `h^l`
- For **control / guidance**: use `v_θ` / `ε_θ` or the sampler
- For **raw geometry / motion**: use `z_t`

### 13.7 Strong final takeaway
There is only **one true latent state space** during generation: the VAE latent state `z_t`.

The patch space and transformer space are auxiliary representations used to compute **how `z_t` should move**.

The cleanest version of the user's research direction is:
> identify timestep-dependent semantic subspaces and steer generation through manifold-aware biasing of the model's own update field, rather than directly forcing raw latent paths by unconstrained optimization.
