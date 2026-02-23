# exp_006 — Wan 2.1 T2V 1.3B clip-to-clip connecting

Same clip-to-clip pipeline as [exp_005](../exp_005_wan21_clip_connect/) but uses the **1.3B** text-to-video model for lower VRAM and faster iteration.

## Model

- **Repo:** [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
- **VRAM:** ~8.2 GB (consumer GPUs, e.g. RTX 3080/4090)
- **Components:** All loaded from the 1.3B repo:
  - **VAE:** `AutoencoderKLWan` (same architecture as 14B)
  - **Transformer:** `WanTransformer3DModel` (1.3B params)
  - **Text encoder:** `UMT5EncoderModel`
  - **Tokenizer:** T5-style
  - **Scheduler:** Config loaded from repo; pipeline uses `FlowMatchEulerDiscreteScheduler` for API compatibility

The 1.3B Diffusers repo uses the same folder layout and component types as the 14B repo; only the transformer weights differ. No code changes are required in `src/diffusion/wan_clip_connect.py`.

## Run

From repo root (with `diff` env and package installed):

```bash
# Dry run (validate paths/config only)
python experiments/exp_006_wan21_clip_connect_1_3b/run.py
```

Set `dry_run: false` in `config.yaml` to run full inference. Outputs go to `outputs/videos/exp_006_wan21_clip_connect_1_3b/`.
