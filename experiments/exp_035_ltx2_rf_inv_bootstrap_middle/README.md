# exp_035 — LTX-2 RF-Solver inversion with model-bootstrap middle anchors

## Question

Under §0 (no source middle frames), can a forward C2V generation pass —
running the model itself on just the start+end sub-clips — produce middle
latent anchors that improve real-clip inversion vs exp_033's "free middle"
baseline?

## Setup

Per-clip flow:
1. **Bootstrap.** Call `pipe(conditions=[start_sub_clip, end_sub_clip], ...,
   num_inference_steps=N_boot, guidance_scale=4.0, output_type="latent",
   return_dict=False)` → produces `z_bootstrap` (full-source-shaped latent
   conditioned only on the two endpoint sub-clips and the model — both §0-
   permitted inputs). Re-normalize and re-pack to match the inversion's
   `clean_latents` format.
2. **Build conditioning anchors.**
   - frames {0..3}: sub-clip-encoded anchors (exp_033 baseline, unchanged).
   - frames {4..11}: NEW — `z_bootstrap` slices at these positions, strength
     1.0.
   - frame 12: dropped (exp_033 baseline, unchanged — frame-12 single-pixel-
     collapse asymmetry was confirmed in It-4 to prefer "no pin" over any
     mismatched pin).
   - frames {13..15}: sub-clip-encoded anchors (exp_033 baseline, unchanged).
3. **Invert + recon + regen.** Same RF-Solver midpoint pipeline as exp_033.
4. **Save** `z_bootstrap.pt` per clip for diagnostic comparison vs `z0.pt`.

## How to run

Three configs corresponding to the three sub-phases of It-5:

```bash
# B0 — mini-falsification (1 clip, reduced steps, ~8 min on PCIe A100)
python experiments/exp_035_ltx2_rf_inv_bootstrap_middle/run.py \
  --config experiments/exp_035_ltx2_rf_inv_bootstrap_middle/config_mini.yaml

# B1 — pilot (2 clips, full steps, ~30 min)
python experiments/exp_035_ltx2_rf_inv_bootstrap_middle/run.py \
  --config experiments/exp_035_ltx2_rf_inv_bootstrap_middle/config_pilot.yaml

# B2 — full batch (10 clips, full steps, ~2 hours) — only after B1 confirms
python experiments/exp_035_ltx2_rf_inv_bootstrap_middle/run.py
```

## Outputs

`outputs/videos/exp_035_ltx2_rf_inv_bootstrap_middle/run_NNNN/<sample_id>/`:

- `z0.pt`, `z1.pt`, `z_t_25/50/75.pt`, `z0_recon.pt`, `z0_regen.pt`
- **`z_bootstrap.pt`** — the new bootstrap-derived latent
- `source_video.mp4`, `recon_video.mp4`, `regen_video.mp4`
- `step_diag_invert/reconstruct/regenerate.csv`
- `inv_meta.yaml` (now includes `bootstrap.{num_steps, guidance_scale,
  wall_s, z_bootstrap_norm, middle_token_range}`)

Plus per-run `summary.yaml`, `config_snapshot.yaml`, `run.log`.

## Decision rule (pre-registered, see Ledger It-5)

**B0 mini-falsification gate:**
- bootstrap mp4 not structurally broken (no NaN, not all-black) AND
- recon PSNR on ss0 ≥ 22 dB (3 dB tolerance vs exp_033's 24.97 at full
  steps) → PROCEED to B1; otherwise REJECTED, exit ②.

**B1 pilot gate:**
- median PSNR across {ss0, ss5} ≥ exp_033 baseline median (20.68) + 3 dB →
  full batch (B2).
- regression > 2 dB on either pilot clip → REJECTED.

**B2 full batch:** exit ① if recipe meets the target.
