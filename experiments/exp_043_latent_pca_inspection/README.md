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
