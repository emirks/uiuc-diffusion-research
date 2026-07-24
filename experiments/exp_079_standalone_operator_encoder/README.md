# exp_079 — Standalone operator encoder: "SupCon-T" (composite class × temporal-manipulation contrastive)

Advised campaign (bottleneck branch). Advisor ruling V3, **locked** 2026-07-24. Dossier:
`$LAB/misc/bottleneck_branch/DOSSIER.md` (🔒 ADVISOR RULING V3).

## Question

Can we train a **standalone** operator-token encoder — no generator in the loop — whose K=72 code
provably carries the transition's **operator structure** (direction, timing) and generalizes across
classes, on the data we already have (139 clips, 26 classes), *before* the factorial dataset lands?

This is the owner's "make a good encoder, contrastive loss, train now" directive, realized in the
strongest form that survives the class≈operator≈content confound.

## Setup

- **Objective:** one supervised-contrastive loss (τ=0.1) over **composite labels (class,
  manipulation)**. Positives = same class AND same manipulation (different instances + augmented
  views); negatives = everything else — so a clip's own time-reverse/time-warp is a NEGATIVE with
  *byte-identical content but a different operator*. This is the only content-identical /
  operator-different signal available pre-dataset. **No decoder, no endpoint side-inputs** (owner
  directive; the encoder must *extract* the operator, not be handed the endpoints).
- **Encoder:** the built-and-tested Perceiver head (`operator_encoder.py`, 8.6M, K=72, shape
  (6,4,3), segment embeddings), **normal init** (no zero-init output, no skip). Mean-pool the 72
  tokens → 2-layer MLP → L2-normalized z. Loss on z; probes on both z and the raw tokens. K=72 is
  kept deliberately so all post-dataset coupling plumbing stays valid.
- **Ablation arm** = the owner's exact proposal: identical, but labels = **class only**.
  Pre-registered prediction: class-decode ≥95 %, reversal margin ≈0 (an info-free 26-way classifier).
- **Temporal manipulations** (pixel-space, endpoints fixed for warps, nearest-frame resample, then
  VAE re-encoded — the LTX VAE is causal, so reversing *latents* is not reversing video):
  TRAIN `{identity, reverse, ease-in γ=2, ease-out γ=0.5}`; HELD-OUT `{γ=3, γ=0.33, γ=1.5, γ=0.67}`
  used only for the generalization probes.
- **Split (PRIMARY, contamination-safe; `split.json`, frozen before training):**
  - **train** = 26 held-in classes' 139 clips − 1 held-out instance/class = **113 clips** × 4 train manips.
  - **heldout_instance** = 26 clips (1/class) → class-separation & instance-ID probes.
  - **heldout_class** = **45 zero-shot-class clips (10 zs classes)** → the load-bearing temporal
    probe. **Never trained** — encoded only for probe features.

## Pre-registered probes & bars (frozen before any result)

1. **Non-collapse** (B1 cross-demo sensitivity, verbatim): ≥0.2 (B1 0.0018, b1r 0.0075). Sanity.
2. **Class separation** (linear probe, held-out instances): discriminativeness only — NOT operator evidence.
3. **Temporal generalization (LOAD-BEARING, confound-valid)** — on held-out zs classes:
   margin = d(z(V), z(m(V))) / median d(z(V), z(same-class other instances)).
   **Bars (median seed, none collapsed):** reverse margin ≥1.0; held-out-γ margin ≥0.5;
   γ-monotonicity Spearman ρ ≥0.7.
4. **Content-leak guards** (one-sided, non-gating): within-class instance-ID decode (low = content-light);
   endpoint-appearance R² from z vs raw pooled-feature baseline (report ratio). Never claimed as disentanglement.
5. **Corpse controls:** run the battery on the B1 & b1r encoder outputs — must reproduce B1 ~0.002
   sensitivity and FAIL every temporal probe (certifies the harness on known-dead encoders).

**Contingency (one, pre-authorized):** trained-class temporal pass but held-out-class fail → one
retry swapping frozen VAE latents → frozen VideoMAE-v2 features. **KILL:** if held-out temporal still
fails → the pre-dataset standalone line STOPS (finding: temporal structure doesn't generalize from
this data/feature stack). **Budget:** ≤4 GPU-h, ≤2 days, single GPU (L40S OK). **No generator
coupling pre-dataset under any outcome.**

## How to run

```bash
# 1. freeze the split (deterministic; commit split.json)
python experiments/exp_079_standalone_operator_encoder/build_split.py

# 2. encode manipulations -> frozen VAE latents (GPU; resumable)
sbatch --partition=secondary --account=campusclusterusers --gres=gpu:1 --time=02:00:00 \
       experiments/exp_079_standalone_operator_encoder/job_encode.sbatch

# 3. train E1 SupCon-T + plain-class ablation, 3 seeds each (GPU; L40S OK)
sbatch --array=0-5 --partition=secondary --account=campusclusterusers --gres=gpu:1 --time=01:00:00 \
       experiments/exp_079_standalone_operator_encoder/job_train.sbatch

# 4. probe battery + corpse controls -> probe table
python experiments/exp_079_standalone_operator_encoder/probes.py --all
```

## Outputs

- `dataset/manip_latents/<split>/<class>/<clip>__<manip>.pt` — 916 frozen VAE latents.
- `outputs/exp_079/<arm>_seed<k>/encoder.pt` — trained encoder + projection head.
- `outputs/exp_079/probe_table.json` — the pre-registered metrics per arm/seed + corpse controls.
