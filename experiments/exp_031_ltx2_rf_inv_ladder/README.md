# exp_031 — LTX-2 RF-Solver inversion: the R0→R5 localization ladder

Iteration **It-0** of the RF-inversion research loop (`notes/rf_inversion_loop.md`).

## Question

exp_029 inverts *generated* latents almost perfectly (`inv_recon_rel ≈ 0.01`).
exp_030 collapsed on *real* clips (`≈ 0.68`, 0/10 pass) — but it moved ~6
variables at once, so the failure is unattributed. **Where, exactly, does the
degradation enter?** This ladder moves *one* variable per rung from the
known-good exp_029 baseline and watches for the cliff.

## Setup

Six rungs, one shared `run.py`. Each rung changes exactly one variable from a
well-defined predecessor:

| Rung | z₀ source | conditioning | single Δ | isolates |
|------|-----------|--------------|----------|----------|
| R0 | `gen_latent` — exp_029 `z0.pt` direct | external | — | baseline + **harness validation** |
| R1 | `decode_encode` — `encode(decode(z0.pt))` | external | provenance: transformer-out → encoder-out | VAE round-trip / on-manifold-ness |
| R2 | `decode_encode` | none | conditioning off | C2V cost (vanilla inversion) |
| R3 | `decode_encode` | self | cond source: external → self | C2V conditioning-source cost |
| R4 | `encode_clip` — real DAVIS clip, edge-padded | self | content: gen → real natural | off-manifold (natural) |
| R5 | `encode_clip` — shadow_smoke clip | self | content: natural → stylized | off-manifold (stylized) |

Held **fixed** across every rung (so never a confound): resolution 512×768,
121 frames, 40 invert + 40 recon midpoint steps, CFG=1, zeros audio, seed 42,
3 samples/rung, invert+recon only.

- **`gen_latent` / `decode_encode`** reuse exp_029 run_0002's `z0.pt` (the
  decode-as-fake-real-clip trick) — content held identical across R0–R3.
- **`self` conditioning**: `clean_latents` = z₀'s own endpoint positions, so
  the conditioning invariant (`z₀_cond == clean_latents`) holds exactly.
- **Primary metric: FREE-positions-only** latent rel/cos. Conditioned positions
  are hard-pinned every solver step in both directions → their round-trip error
  is trivially ~0; including them (exp_030's all-positions metric) only adds a
  constant offset. all/cond latent + perceptual PSNR/SSIM/LPIPS also logged.

## How to run

On a GPU pod (A100-80GB), from repo root, with the env initialised
(`source /workspace/cache/pod_init.sh`):

```bash
python experiments/exp_031_ltx2_rf_inv_ladder/run.py
# or a subset:
python experiments/exp_031_ltx2_rf_inv_ladder/run.py --rungs R0,R1
```

Loads the pipeline once, runs all 6 rungs × 3 samples. Est. ≈ 1–1.5 GPU-hours.

## Expected outcome

A `rung → free_rel` table where the **cliff** (first ≥3× jump) localizes the
dominant cause. Pre-registered prior: cliff at **R0→R1** (VAE-encoder latents
are off the flow-matching manifold). Decision tree → It-1 is in the loop Ledger.

**R0 HALT gate:** R0 `free_rel` mean must reproduce exp_029 within ~2× (≤ 0.05).
If not, the unified harness diverges from exp_029 and R1–R5 are untrustworthy —
`summary.yaml.r0_halt_triggered` flags this and the run log warns loudly.

## Outputs

`outputs/videos/exp_031_ltx2_rf_inv_ladder/run_NNNN/`:

- `summary.yaml` — per-(rung,sample) metrics + per-rung `ladder` rollup + `r0_halt_triggered`
- `config_snapshot.yaml`, `run.log` (with the final ladder table)
- `<rung>/<sample_id>/` — `z0.pt`, `z1.pt`, `z0_recon.pt`, `z_t_25/50/75.pt`,
  `source_video.mp4`, `recon_video.mp4`, `step_diag_{invert,reconstruct}_n40.csv`,
  `meta.yaml`
