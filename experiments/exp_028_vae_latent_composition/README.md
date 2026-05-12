# exp_028 — VAE Latent Composition with Held Boundaries + Pixel-Anchored Bridge

Fork of [exp_023](../exp_023_vae_latent_lerp/) that fixes two compounding
problems with the original VAE-only dissolve probe.

## Question

exp_023 produces a dissolve that **starts at output frame 1** (rather than
holding the start clip cleanly for ~1 sec before transitioning), and the
dissolve region itself shows flicker. Are these caused by:

1. The lerp ramping `alpha = t / (T_total - 1)` across the **entire** timeline,
   so the start clip never gets a clean hold window? and
2. The middle blends mixing `start_lat[-1]` (a **motion**-latent encoding
   pixel frames 17-24 of the start clip) with `end_lat[0]` (a **key-frame**
   latent encoding pixel frame 0 of the end clip) — i.e. arithmetic blends
   of two latents with incompatible temporal semantics?

This experiment tests both fixes by running three composition strategies per
sample and saving the decoded videos side by side.

## Setup

| Mode | Timeline structure | Tests fix |
|---|---|---|
| `naive` | `out[t] = (1−α)·start_lat[clamp(t)] + α·end_lat[clamp(t−offset)]` across the whole timeline. exp_023 baseline. | — (control) |
| `hold_lerp_hold` | `[start_lat \| latent_lerp(start_lat[-1], end_lat[0], M anchors) \| end_lat]` — pure start in head, pure end in tail, **latent-space** lerp only in the middle. | (1) only |
| `hold_bridge_hold` | `[start_lat \| lerp(VAE_encode([last_start]), VAE_encode([first_end]), M anchors) \| end_lat]` — pure boundaries + a lerp between **single-frame VAE encodings** in the middle. | (1) + (2) |

The 2026-05-12 revision replaced the original "pixel cross-fade + VAE
encode" bridge with this single-frame-encoding lerp. The previous version
double-anchored `first_end_pixel`: the cross-fade clip ended on
`first_end_pixel` *and* `end_lat[0]` was already an encoding of the same
pixel. Adjacent latent slots therefore described the same pixel and the
second-clip onset came in twice. The new bridge does not re-encode
`first_end_pixel` at the bridge tail — the M middle slots are a latent-
space lerp between two *single-frame key-frame latents*, both produced
by encoding their source pixel alone.

For the length sweep `num_frames_sweep = [121, 89, 65]` with
`num_clip_frames = 25` (T_clip = 4):

```
N=121 → T_total = 16 → M = 8   (long bridge,    ~2.67s @ 24fps)
N= 89 → T_total = 12 → M = 4   (mid bridge,     ~1.33s)
N= 65 → T_total =  9 → M = 1   (single-latent bridge — fastest transition
                                while keeping both clips fully held)
```

The single-latent (M=1) bridge writes a single lerp(A, B, α=0.5) slot
between the held start and end clips — the minimum-length transition
possible without collapsing the hold-the-clips structure.

### Why decoded middle frames look "stuck"

The LTX-2 causal VAE decoder treats latent slot 0 as a *key-frame* (decodes
to 1 pixel) and slots ≥1 as *motion* (decodes to 8 pixels each). The M
middle slots all sit in motion positions, but they were synthesised from
single-frame *key-frame* latents (and lerps thereof). The decoder
therefore interprets these single-frame-flavored latents as motion and
produces 8 pixel frames per slot that read as a near-static "hold" of the
encoded image. This is intentional: the only true single-frame anchor in
the decoded output is `composed[0] = start_lat[0]`, and the user wants
the bridge region to read as a held interpolation rather than as
synthesised motion.

Samples: 3 DAVIS pairs (easy / mid / hard) matching exp_027's subset.

## How to run

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/envs/diff
cd /workspace/diffusion-research
python experiments/exp_028_vae_latent_composition/run.py
```

VAE-only (~9 GB GPU; tiling on). No transformer, no text encoder, no audio
model. First run downloads the VAE shard of `Lightricks/LTX-2` into
`HF_HOME`; subsequent runs reuse the cache.

## Outputs

```
outputs/videos/exp_028_vae_latent_composition/
└── run_NNNN/
    ├── run.log
    ├── config_snapshot.yaml
    ├── summary.yaml
    └── {sample_id}/
        ├── s42_K25_N121_M8_mode-naive.mp4
        ├── s42_K25_N121_M8_mode-hold_lerp_hold.mp4
        ├── s42_K25_N121_M8_mode-hold_bridge_hold.mp4
        ├── s42_K25_N89_M4_mode-naive.mp4
        ├── s42_K25_N89_M4_mode-hold_lerp_hold.mp4
        ├── s42_K25_N89_M4_mode-hold_bridge_hold.mp4
        ├── s42_K25_N65_M1_mode-naive.mp4
        ├── s42_K25_N65_M1_mode-hold_lerp_hold.mp4
        ├── s42_K25_N65_M1_mode-hold_bridge_hold.mp4
        └── config_snapshot.yaml
```

Three videos per sample, identical filename pattern except for the `mode-*`
suffix — open them in pairs and compare:

- **naive vs hold_lerp_hold** → does holding the start/end pure restore
  the "1 sec of clean playback before the dissolve" behaviour? If yes, fix
  (1) is enough on its own.
- **hold_lerp_hold vs hold_bridge_hold** → does the VAE-coherent bridge
  reduce flicker in the dissolve region? If yes, fix (2) is also needed.
- If `hold_bridge_hold` is qualitatively the best, that becomes the new
  reference for any future "what would a perfect rectified-flow output
  look like in latent space" baseline.

## Why this matters for exp_027 and downstream

exp_023's dissolve is the *VAE-only* upper bound on quality — any
rectified-flow output that goes through a straight latent-space path
inherits the same dissolve. If `hold_bridge_hold` gives a cleaner
straight-path result, the downstream story changes: the dissolve we see
in exp_020 is *not* a fundamental property of "straight in latent space",
it's a property of "naive straight-line latent arithmetic". Anchoring the
endpoints and routing the middle through a VAE-coherent bridge is a
strictly different (and possibly better) straight-path baseline.

For Step 7 (feature injection, building on exp_027's cached `z₁` and σ
checkpoints), this matters because the editing recipe may want to inject
features that target a *better* straight-path latent than exp_023's
naive lerp. exp_028's `hold_bridge_hold` output is a candidate target for
that.

## Sources

- exp_023 — original VAE latent lerp probe (`../exp_023_vae_latent_lerp/`).
- LTX-2 VAE temporal scale / latent count formula:
  `notes/models/ltx2/conditioning.md` (and the inline note in `config.yaml`).
- exp_027 sample subset (matched for cross-experiment parity).
