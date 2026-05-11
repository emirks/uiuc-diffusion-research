# LTX-2: Spatial Locality, Token-to-Position Mapping, and Representation Analysis

> How does a token that has attended globally to 6,143 other tokens still end up
> "about" one specific 32×32 px × 8-frame cube of the video?
> This note answers that question and derives what it means for trajectory analysis.

---

## 1. Patch Size in LTX-2 19B (exp_021 onwards)

The patchification step (covered in `conditioning.md` §2) groups VAE latent cells into tokens.
The patch parameters differ between model variants:

| Model / config | Spatial patch P | Temporal patch p1 | Token count (121fr, 512×768) | Token embed dim |
|---|---|---|---|---|
| LTX-Video (smaller) | P=2 | 1 | 16 × 8 × 12 = **1,536** | 128 × 4 = 512 |
| **LTX-2 19B (exp_021)** | **P=1** | **1** | 16 × 16 × 24 = **6,144** | 128 × 1 = 128 |

**How to verify from data:** `hidden_states[τ][L].shape = [2, 6144, 4096]`.
- `6144 = F' × H' × W' = 16 × 16 × 24` — exactly one token per VAE latent cell.
- `4096` = transformer hidden dimension (after the linear projection 128→4096 that follows patchification).

The `denoising_schedule.md` note confirms this: "N = 16 × 16 × 24 = **6,144 tokens**" is used in the sigma shift calculation.

### What one token covers in pixel space (P=1, LTX-2 19B)

```
One token at latent position (p, h, w) represents:
  Temporal : latent frame p   → pixel frames  (p-1)×8 .. p×8 ≈ 8 consecutive pixel frames
  Height   : latent row h     → pixel rows     h×32  ..  (h+1)×32 = 32 px
  Width    : latent column w  → pixel columns  w×32  ..  (w+1)×32 = 32 px
```

Each token is one 32 px × 32 px × 8-frame spatio-temporal brick of the original video.

---

## 2. The Locality Question

Full self-attention means that after one block, token `(0,0,0)` has received weighted signals from
all 6,143 other tokens. After 48 blocks the receptive field is the entire video. Yet the final
representation of token `(0,0,0)` is still meaningfully "about" the top-left corner of the first
latent frame. Three mechanisms explain this.

---

### 2.1 3D RoPE — persistent spatial identity

RoPE (Rotary Position Embedding) is **not** an additive input embedding that gets diluted by
residual connections. It is applied **inside every attention layer** to the Q and K vectors before
the dot product:

```
attn_score(i, j)  =  RoPE3D(pos_i) · Q_i  ·  RoPE3D(pos_j) · K_j
```

Position `(p, h, w)` is encoded as a 3D rotation angle derived from the pixel-space midpoint of
the token's bounding box (see `conditioning.md` §5.1-a for the midpoint calculation).

**Key property:** The rotation angle grows with distance between positions.
Two tokens at identical positions get identical rotations → their Q·K alignment is maximised.
Two tokens far apart in (t, y, x) get divergent rotations → their Q·K alignment decays.

Because RoPE is re-applied at every block, each token carries its spatial identity all the way
to block L47, even though attention has globally mixed information between every pair of tokens.
This is why the Level 3 PCA in `exp021_trajectory_analysis.ipynb` can recover a clean temporal
gradient (blue frame 0 → red frame 15): the temporal axis of RoPE remains the dominant structural
signal even in deep layers.

---

### 2.2 Output constraint — locality enforced by the loss

The transformer's final operation is a **per-token linear projection** that reconstructs the
velocity field:

```
Transformer output:    [N=6,144,  D=4,096]   (one embedding per token)
      ↓  linear  (4,096 → 128)
Projected output:      [N=6,144,  C=128]
      ↓  reshape to spatial layout
Velocity prediction:   [C=128,  F'=16,  H'=16,  W'=24]   = v_pred
```

Token `(p, h, w)` → `v_pred[:, p, h, w]`. **One-to-one, no overlap.**

The flow-matching loss:

```
L = Σ_{p,h,w}  ‖ v_pred[:, p, h, w]  −  v_target[:, p, h, w] ‖²
```

Gradient for position `(p, h, w)` flows exclusively through token `(p, h, w)`.
This creates a persistent training pressure: even after 48 blocks of global mixing,
token `(p, h, w)` **must** compress its final 4,096-dim embedding into 128 numbers that
correctly predict the velocity at exactly that position.

The analogy: a weather forecaster reads the entire global map (global attention) but must still
produce a precise temperature forecast for one city (local output). The forecast responsibility
forces the forecaster's mental model of "what's happening at this city" to stay coherent.

---

### 2.3 Attention locality in practice

Even without architectural constraints, learned attention patterns in video DiTs are empirically
local for most heads. RoPE makes this natural: the relative rotation between nearby positions is
small, so their Q·K dot products are naturally high-magnitude. The model learns to exploit this
by having most heads attend within a spatiotemporal neighborhood, with a minority of "global"
heads attending broadly for semantic context.

The combined effect of §2.1 + §2.2 + §2.3 is that token `(p, h, w)` ends up with an embedding
that is:
- **Primarily local:** mostly encoding the content of the `(p, h, w)` VAE cell
- **Contextually global:** informed by the entire video via the global heads
- **Anchored by identity:** the RoPE rotation ensures the model always "knows" which position
  this token corresponds to

---

## 3. Unpatchify: the Output Side of the Same Coin

The "unpatchify" step at inference is the inverse of patchification:

```
v_pred packed:   [B, N=6144, C=128]   (linear projection output of transformer)
      ↓  reshape
v_pred spatial:  [B, C=128, F'=16, H'=16, W'=24]
```

With P=1, each token owns exactly one `[128]`-dim slice of the velocity field.
There is no aggregation, interpolation, or learned blending — a single token's last embedding
directly determines the velocity at that spatial-temporal position.

This is the mechanism the user identified: **"the velocity field has patches, and those patches
are created using a single token output of the transformer"**. For P=1 this means the patches are
individual VAE latent cells. For P=2 they would be 2×2 groups of latent cells. Either way, the
one-to-one (or one-to-patch) mapping creates soft locality via the loss.

---

## 4. Implications for Trajectory Analysis (exp_021)

Understanding the token-to-position mapping directly informs how to interpret the Level 3
hidden-state analyses in `exp021_trajectory_analysis.ipynb`:

### Per-frame activation norm `‖h^L(p)‖`

Token `(p, h, w)` in block L has a 4,096-dim embedding vector.
- We average the L2 norms of all 16×24 = 384 tokens within latent frame `p` → one scalar per frame.
- Because each token is responsible for predicting `v_pred[:, p, h, w]`, a high mean norm at frame
  `p*` in deep blocks (L35, L47) means the model is allocating representational capacity to regions
  that need a large velocity correction — i.e., the transition frame.

### PCA of frame mean-embeddings

For frame `p`, average the 384 token embeddings `h^L(p,h,w)` across `(h,w)` → one 4,096-dim
"frame mean-embedding" per frame.

This is a summary of what block `L` "thinks" about the entire temporal slice `p`.
With P=1, each token corresponds exactly to one VAE latent cell, so this average is
a direct spatial mean of the activated representation within that latent frame —
not an average over temporally mixed groups.

The clean blue→red temporal gradient visible in early/mid blocks is produced by RoPE preserving
the frame index as the dominant axis of variation. A kink or cluster break between frames `a` and
`a+1` reflects a representational discontinuity — what the transformer internally "perceives" as
a content boundary.

### Cosine similarity matrix

Entry `(p, q)` = cosine similarity between frame mean-embeddings of frames `p` and `q`.
Block-diagonal structure (high within-cluster, low between-cluster) means the model has
internally separated the video into two semantic halves. The boundary position gives a
purely representational estimate of the dissolve frame, independent of the curvature and
velocity signals.

---

## 5. Summary: Why Spatial Correspondence Survives Global Attention

| Mechanism | Where it acts | What it guarantees |
|-----------|---------------|-------------------|
| **3D RoPE** (§2.1) | Inside every attention layer | Token always carries its spatial identity via rotation encoding; cannot be "mixed away" |
| **Output constraint** (§2.2) | Loss function | Each token's final embedding must reconstruct velocity at its exact spatial position — creates gradient pressure for local accuracy |
| **Learned attention locality** (§2.3) | Attention patterns | In practice, most heads attend predominantly to nearby tokens; RoPE makes this the path of least resistance |

The result: global attention + RoPE + per-position output loss = **representations that are globally
informed but spatially anchored**. Analysing per-frame hidden states is meaningful exactly because
of this combination.

---

## 6. Cross-references

- `conditioning.md` §2 — patchification formula and P=2 token count (smaller model)
- `conditioning.md` §5.1 — RoPE mechanics, midpoint calculation, bounding box → fractional position
- `denoising_schedule.md` — confirms N=6,144 for 121fr 512×768 in sigma shift calculation
- `exp021_trajectory_analysis.ipynb` — Level 3 uses the P=1 token structure described here
