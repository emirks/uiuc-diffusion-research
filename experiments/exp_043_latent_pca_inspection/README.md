# exp_043 — VAE-latent PCA inspection

## Question

Three latent-space diagnostics for the LTX-2 / shadow_smoke / RF-inversion /
feature-injection pipeline:

1. **Domain test (cleans).** Are shadow_smoke z0's distributional outliers
   relative to natural-video latents (DAVIS A_word generations from exp_024)?
2. **Noise-shape test (inverted z1).** Do RF-inverted z1's cluster apart from
   i.i.d. Gaussian, i.e. does inversion leave structural residue?
3. **Injection-effect test (exp_041 run_0007).** Does block_out injection move
   the perturbed-inject latent toward the reference latent (good) or
   perpendicular to it (just noise), and which variants pull most?

## Setup

`run.py` loads the LTX-2 pipeline (bf16, CPU-offload, VAE tiling), then for
each input mp4 calls `vae.encode` + `_normalize_latents` + `_pack_latents`
exactly as the existing inversion scripts do.  All encodes are forced to
608×608 × 121 frames → packed shape `[1, 5776, 128]` so every artifact is
joinable in PCA.

Inputs (all paths relative to `REPO_ROOT`, listed in `config.yaml`):

| Group              | Source                                                  | Count |
|--------------------|---------------------------------------------------------|-------|
| `smoke_z0`         | `data/processed/transitions/shadow_smoke/ss{0..9}.mp4`  | 10    |
| `smoke_z1`         | copy of `exp_033/run_0001/<sample>/z1.pt`               | 10    |
| `davis_gen_z0`     | exp_024 Stage-1 A_word mp4s, 5 class-pair representatives | 5   |
| `exp_041_inject`   | exp_041 run_0007 ss4 variants × {reference_recon, perturbed_baseline, perturbed_inject} | 33 |
| `gaussian`         | `randn` N(0,1) at the same packed shape, seed 42        | 50    |

Total VAE encodes: 48 clips (~12 min on A100).  The notebook
`notebooks/exp_043_latent_pca.ipynb` consumes the resulting tensors and
renders the chart matrix described below.

## How to run

```bash
cd /workspace/diffusion-research
PYTHONPATH=src python -u experiments/exp_043_latent_pca_inspection/run.py
```

Each run lands under `outputs/latent_pca/exp_043_latent_pca_inspection/run_NNNN/`
with:

```
run_NNNN/
  run.log
  manifest.yaml          # group/name/source path for every saved tensor
  latents/
    smoke_z0/<sample>.pt
    smoke_z1/<sample>.pt
    davis_gen_z0/<pair>.pt
    exp_041_inject/<variant>__<role>.pt
    gaussian/sample_NNN.pt
```

After the encode finishes, open the notebook and point `RUN_DIR` at this
run directory.  All PCAs are computed on CPU (numpy SVD) from the cached
tensors — no GPU needed for plotting.

## Outputs

The notebook produces a chart matrix:

- `[1]` Per-sample frame trajectories — smoke (10 panels)
- `[2]` Per-sample frame trajectories — davis A_word (5 panels)
- `[3]` Smoke unified frame PCA (color by sample + color by frame index)
- `[5]` Whole-clip PCA — cleans only (smoke vs davis-gen)
- `[6]` Cross-domain frame PCA — smoke + davis-gen (cluster test)
- `[7]` Whole-clip PCA — smoke-z1 + gaussian
- `[8]` Frame PCA — smoke-z1 + gaussian
- `[9]` Per-variant injection PCA (11 panels: reference vs baseline vs inject)
- `[9b]` All-variants injection overlay

Per-channel z1 vs N(0,1) mean/std + scree plots accompany the joint PCAs.

---

# Addendum — `inverted_noise_vs_gaussian.py` (z1-vs-Gaussian deviation)

## Question

RF-inverted noise z1 reconstructs a shadow-smoke video; fresh generation starts
from white Gaussian N(0,I). **What in z1 deviates from a matched Gaussian, and is
that deviation the "smoke signature" concentrated in the free-middle frames?**
Goal (not implemented): at production, add the signature to a fresh Gaussian
sample. So the deviation must be characterized as (a) localized, (b) shared
across clips, (c) injectable.

## Setup

Standalone CPU/numpy analysis — reads the cached `z1.pt` / `z0.pt` from
`exp_033_ltx2_rf_inv_drop1/run_0001/<sample>/` directly (no pipeline, no GPU).

**Geometry (verified, not assumed):** packed `[1,N,128]`, P=P_t=1 → token order
`n = f·(H·W) + h·W + w`; unpack = `reshape(F=16, H, W, 128)` with the clip's own
(H,W). Orientation groups: portrait 22×16 (clips 0,2,3,6,8), landscape 16×22
(1,5,7,9), square 19×19 (4). **Never group by N** (portrait/landscape share
N=5632 with swapped H,W). Free-middle latent frames = **4..12** (drop1 frees
frame 12); anchors = **0-3, 13-15** (hard-pinned to z0 during inversion). Verified
from `exp_033/run.py:end_clip_index` (n_lat=16, k_lat=4).

**Nulls (per metric, plotted alongside z1):** white N(0,1) of identical shape, and
a variance-matched white null scaled to z1's per-(frame,channel) std (isolates
structure from variance). All structural metrics (power spectrum, autocorr,
temporal corr) use raw values with mean-removal only; distributional metrics
(skew/kurtosis) standardize per channel.

**Battery:** (1) per-channel marginal moments + KS/QQ; (2) radial power spectrum
+ isotropic spatial autocorrelation; (3) adjacent-frame temporal correlation;
(4) per-frame / per-channel energy + low-frequency power fraction (signature
localization); (5) within-group cross-clip cosine of the free-middle structured
map + cross-clip radial-spectrum cosine; (6) free-middle vs anchor split
throughout. Cross-orientation comparisons use only radial / per-frame scalars.

## How to run

```bash
# on the GPU pod (CPU-only analysis):
python experiments/exp_043_latent_pca_inspection/inverted_noise_vs_gaussian.py \
    --tensor z1 --groups portrait,landscape,square
# z0 control (source latent, same battery):
python experiments/exp_043_latent_pca_inspection/inverted_noise_vs_gaussian.py \
    --tensor z0 --groups portrait,landscape,square
```

Args: `--data_dir` (default exp_033 run_0001), `--tensor {z1,z0}`, `--groups`,
`--seed`, `--out_subdir`. Uses `next_run_dir` + `TeeLogger`.

## Outputs

`outputs/latent_pca/exp_043_inverted_noise_vs_gaussian/run_NNNN/`:
`summary.json` + `charts/{01_marginal,02_spatial,03_temporal,04_localization,
05_crossclip,06_qq}_{portrait,landscape,square}.png` — each chart overlays z1 vs
the Gaussian null, per orientation group.

## Headline result (HYPOTHESIS REFUTED)

The "smoke signature" is **NOT** in z1's free-middle — it has been **erased**
there. RF-inversion re-Gaussianizes the free-middle and leaves structure only in
the clamped anchors (which are just slices of z0). See
`notes/exp/exp_043_inverted_noise_vs_gaussian.md` for the full report and the
production recommendation (the signature lives in z0, not z1).
