# exp_016 — LTX-2 C2V Sequential Batch Evaluation

## Question

Can the exp_015 C2V clip-guiding baseline generalise across visually diverse VC-Bench
categories?  This experiment evaluates the same pipeline on samples from two distinct
categories — `action` (outdoor sports) and `conversation` (indoor people) — using a
single pipeline load to amortise model-init cost.

## Setup

- **Pipeline**: `KeyframeInterpolationPipeline` (same as exp_015), bf16, no quantisation.
- **Conditioning**: `ClipConditioningInput` — 24 pixel frames from each end of the source
  video encode to 3 VAE latent tokens.  Start clip anchors latent tokens 0–2; end clip
  anchors tokens 10–12.  Tokens 3–9 (≈ pixel frames 17–73) are generated freely.
- **Prompt source**: VC-Bench `caption` field for each video.
- **Samples** (2):

| sample_id | VC-Bench category | Description |
|---|---|---|
| `action_4927323_surfer` | `action` | Surfer riding a wave, turquoise water, outdoor |
| `conversation_3044674_office` | `conversation` | Two women walking through a modern office |

- **Inference**: 97 frames @ 24 fps (≈4 s), 512×768, 40 steps, seed 42.
- **Key difference from exp_015**: pipeline is loaded *once* and iterated over N samples.

## How to run

```bash
cd /workspace/diffusion-research
source src/LTX-2/.venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python experiments/exp_016_ltx2_c2v_batch_eval/run.py
```

## Outputs

```
outputs/videos/exp_016_ltx2_c2v_batch_eval/
└── run_NNNN/
    ├── run.log                          # full stdout + library logs
    ├── config_snapshot.yaml             # full config at run time
    ├── summary.yaml                     # per-sample paths + elapsed_s
    ├── action_4927323_surfer/
    │   ├── s42_K24_steps40.mp4
    │   └── config_snapshot.yaml
    └── conversation_3044674_office/
        ├── s42_K24_steps40.mp4
        └── config_snapshot.yaml
```

To add more samples, append entries to `config.yaml → samples`.  The pipeline
re-uses the same loaded weights for each sample — no extra cost beyond inference.
