# exp_044 — Smoke-transition regen CFG sweep (token-localized CFG)

## Question

The shadow_smoke "transition" lives in the **free-middle** latent frames
(4–12, mask=0). Under production regen (Euler, CFG=4, +negative prompt) these
frames reconstruct at only **11–18 dB**, even though the inverted z1 *encodes*
the smoke well — RF-solver reconstruction (CFG=1, midpoint) reaches **27–38 dB**
in the free middle for 6/10 clips (see `notes/exp/exp_043_smoke_manifold.md`
and `project_smoke_transition_regen_collapse` memory).

**Hypothesis:** the free-middle collapse is caused by CFG. The free-middle
tokens are never clamped (mask=0), so they evolve purely under
`v = v_uncond + s·(v_cond − v_uncond)`. At s=4 with a negative prompt, this
amplifies the generic text direction over z1's encoded source smoke. Reducing
CFG on the free-middle tokens (while keeping full CFG on the visible anchors)
should recover the recon-quality smoke.

**Target:** free-middle regen PSNR median > 18 (currently 14.6 across 10 clips).

## Setup

Regen-only: loads the cached exp_033 z1 (`z1_source.run_dir`) — **no
re-inversion**. Rebuilds the exact exp_033 conditioning (sub-clip-encoded
start/end anchors + drop1 of the end-clip first latent frame). For each clip,
sweeps the regen guidance:

- `cfg4_global` (V0): scalar CFG=4 + neg prompt — reproduces exp_033 regen.
- `cfg1_global` (V1): scalar CFG=1, positive-only — should approach recon.
- `cfg2_global` (V2): scalar CFG=2.
- `loc_f1_a4` (V3): **token-localized** — CFG=1 on free-middle tokens (mask=0),
  CFG=4 on anchor tokens (mask>0). The deployable production recipe.
- `loc_f1.5_a4` (V4): localized, free_cfg=1.5.

Token-localized CFG is implemented via `RFInverter._call_transformer`'s new
`guidance_per_token` arg ([1,N,1] per-token scale, broadcast over channels in
the CFG mix). Deployable: uses only z1 + endpoints, no source-middle leak.

Metric: **free-middle PSNR/SSIM/LPIPS** (pixels [25:97], = latent frames 4–12)
plus full-frame and per-latent-frame PSNR, vs the decoded source.

## How to run

```bash
# pilot (ss0 z1-rich, ss7 near-threshold, ss5 z1-poor control)
PYTHONPATH=src python experiments/exp_044_smoke_regen_cfg/run.py
# full 10-clip batch: point --config at config_full.yaml (created on promotion)
```

GPU: regen is ~40 Euler steps/clip/variant; CFG>1 variants do 2 transformer
calls/step. Loads LTX-2 in bf16 with cpu-offload + vae tiling.

## Expected outcome

- V1/V3 lift the z1-rich pilot clips' (ss0, ss7) free-middle PSNR by ≥ +6 dB
  over V0 → CONFIRMS the CFG-collapse mechanism → promote to full batch.
- ss5 (z1-poor) stays flat → confirms it needs the Phase-2 cross-clip smoke
  prior (z1 itself lacks the middle).

## Outputs

`outputs/videos/exp_044_smoke_regen_cfg/run_NNNN/`:
- `<sample>/source_video.mp4`, `<sample>/regen_<variant>.mp4`
- `summary.yaml` — per-clip per-variant full + free-middle metrics, per-latent-frame PSNR
- `config_snapshot.yaml`, `run.log`
