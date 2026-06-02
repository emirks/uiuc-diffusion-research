# exp_034 — LTX-2 RF-Solver inversion: anchor-quality interventions

## Question

Under the §0 deployability constraint (no source middle frames may enter
the recipe), does a higher-quality deployable anchor at the end-sub-clip's
first latent position improve real-clip RF inversion vs simply dropping
that pin (exp_033)? And does the cumulative end-side anchor mismatch
across all four end positions hurt more than it helps?

## Setup

Fork of exp_033 with a recipe knob (`inversion.recipe` in config.yaml):

- **recipe=A (scaffold_pad).** Build an 8-frame static-replay clip
  `[end_frames[0]] × 8`, VAE-encode, and substitute the resulting single
  latent frame as the anchor at source latent frame 12 (re-enabling cmask
  there). The static-replay produces an 8-pixel-collapse latent that
  matches the full-clip encoder's semantics at this position, instead of
  the original sub-clip encoder's 1-pixel-collapse asymmetry that drove
  the 3.5× larger error in exp_030.
- **recipe=B (drop_all_end).** Zero cmask + clean_latents at all four
  end-sub-clip latent positions {12, 13, 14, 15}. Keep only start anchors
  {0..3}.

Both recipes use only endpoint sub-clip pixels — pass the §0 mechanical
test (delete middle source frames → recipe still produces clean_latents).

It-3's CPU diagnostic
(`scripts/anchor_error_localization.py` →
`scripts/anchor_error_localization.csv`) measured per-latent-frame
|z0 - z0_recon|² across exp_030/032/033 and found:

- ~60% of exp_030's total round-trip cost lives in free middle frames;
- the dominant single-position anchor mismatch is end-sub-clip frame 12
  (3.5× the per-frame error of {13, 14, 15});
- exp_033's drop1 zeroed frame-12 cost (98% local reduction) but left
  ~57K cumulative start-side mismatch and 1.5M middle-truncation cost.

The hypothesis under test: a deployable anchor proxy at frame 12 (A) or
removing the end-side anchor entirely (B) shifts the velocity field in
the right direction during inversion.

## How to run

```bash
# Pilot (Phase B1) — 2 clips × 2 recipes, ~25 min total on A100
python experiments/exp_034_ltx2_rf_inv_anchor_quality/run.py \
  --config experiments/exp_034_ltx2_rf_inv_anchor_quality/config_pilot_A.yaml
python experiments/exp_034_ltx2_rf_inv_anchor_quality/run.py \
  --config experiments/exp_034_ltx2_rf_inv_anchor_quality/config_pilot_B.yaml

# Full batch (Phase B2) — only after pilot CONFIRMS, on the winning recipe
python experiments/exp_034_ltx2_rf_inv_anchor_quality/run.py
```

## Outputs

`outputs/videos/exp_034_ltx2_rf_inv_anchor_quality/run_NNNN/<sample_id>/`:

- `z0.pt`, `z1.pt`, `z_t_25/50/75.pt`, `z0_recon.pt`, `z0_regen.pt`
- `source_video.mp4`, `recon_video.mp4`, `regen_video.mp4`
- `step_diag_invert/reconstruct/regenerate.csv`
- `inv_meta.yaml` (now includes the `recipe` field)

Plus per-run `summary.yaml`, `config_snapshot.yaml`, `run.log`.

## Decision rule (pre-registered, see Ledger It-4)

- **CONFIRMED** for a variant: median PSNR across {ss0, ss5} ≥ exp_033's
  ss0+ss5 baseline median + 3 dB AND no clip regresses below 18 PSNR →
  promote to full 10-clip batch.
- **REJECTED**: both variants regress vs exp_033 by > 2 dB on either
  pilot clip → no full batch; design It-5.
- **INCONCLUSIVE**: mixed results → run a 3-clip mini-batch (ss0, ss5,
  ss9) on the less-regressed variant.
