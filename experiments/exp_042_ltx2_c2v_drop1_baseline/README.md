# exp_042 — Production C2V Baseline (drop1, two-stage, audio)

The full LTX-2 production pipeline (Stage 1 base + Latent Upsampler + Stage 2
distilled LoRA + model-generated audio muxed via `encode_video`) on the
shadow_smoke samples, with the only deviation from a vanilla production call
being a one-time monkey-patch on `apply_visual_conditioning` that **drops the
FIRST latent frame of the end-clip anchor** in `cmask`, `clean_latents`, and
the initial `latents`. Both Stage 1 and Stage 2 outputs are saved per sample.

Mirrors the mechanics of `exp_020` but keeps this experiment's content
invariants: shadow_smoke 0..9, the smoke transition prompt, CFG=3.2, and
`max_area=393216`.

## Question

What does straight production LTX-2 C2V generation of shadow_smoke 0..9 look
like at CFG=3.2 with drop1 — at both the Stage-1 base resolution and the
Stage-2 ×2 upscaled distilled refinement?

## Setup

- `LTX2ConditionPipeline.from_pretrained(Lightricks/LTX-2, bf16)`,
  `enable_model_cpu_offload`, `vae.enable_tiling()`.
- Distilled LoRA loaded as adapter `stage_2_distilled` (weight
  `ltx-2-19b-distilled-lora-384.safetensors`, strength configurable).
- `LTX2LatentUpsamplerModel` + `LTX2LatentUpsamplePipeline` (×2 spatial,
  latent → latent).
- Stage-2 scheduler: `FlowMatchEulerDiscreteScheduler` with
  `use_dynamic_shifting=False, shift_terminal=None` (per ltx2 docs).
- **drop1 monkey-patch** on `pipe.apply_visual_conditioning`. After the
  original call runs (overwriting both anchor slabs in-place), the slot
  `[end_latent_idx·tpf : (end_latent_idx+1)·tpf]` is zeroed in `cmask`,
  `clean_latents`, and `latents`. With `cmask=0` at the dropped slot the
  pipeline's own `prepare_latents` fills it with pure prior noise; the
  denoise loop's `denoised·(1-cmask) + clean·cmask` then treats it as a
  free position. The patch is a no-op when `condition_indices=[]` (Stage 2
  is called with `conditions=None`, so drop1 fires only in Stage 1).
- Stage 1: 40 steps, CFG=3.2, sigmas=None (dynamic shift), seeded
  generator, `output_type="latent"` → unpacked + denormalized 5D latents.
  Stage-1 video is decoded manually via `pipe.vae.decode`; Stage-1 audio
  via `pipe.audio_vae.decode → pipe.vocoder`.
- Stage 2: 3 steps, CFG=1.0, `noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0]`,
  `sigmas=STAGE_2_DISTILLED_SIGMA_VALUES`, `conditions=None`,
  `latents=upscaled_latent`, `audio_latents=audio_latent_from_stage1`,
  `output_type="np"`.

## How to run

```bash
source /workspace/cache/pod_init.sh
conda activate /workspace/envs/diff
cd /workspace/diffusion-research
python experiments/exp_042_ltx2_c2v_drop1_baseline/run.py
```

## Outputs

`outputs/videos/exp_042_ltx2_c2v_drop1_baseline/run_NNNN/<sample_id>/`:

- `stage1_seed42_steps40_cfg3p2_drop1.mp4` — Stage 1 output at base
  resolution (audio muxed).
- `stage2_seed42_steps40_cfg3p2_drop1.mp4` — Stage 2 distilled output at
  ×2 resolution (audio muxed).

Plus `run_dir/summary.yaml`, `config_snapshot.yaml`, `drop1_log.yaml`,
`run.log`.
