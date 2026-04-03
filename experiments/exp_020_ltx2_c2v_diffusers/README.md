# exp_020 — LTX-2 C2V (HuggingFace Diffusers)

## What this is

Clip-to-video (C2V) on DAVIS pairs using **`diffusers`** pipelines only — not the vendored
`src/LTX-2/` Python package or the [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
application repo. Weights are loaded from the Hugging Face Hub checkpoint
[`Lightricks/LTX-2`](https://huggingface.co/Lightricks/LTX-2) via `from_pretrained`, but
**the API, call signatures, and behaviour** follow the official Diffusers documentation:

- [LTX-2 pipelines (diffusers)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2)
- Doc layout this experiment follows: [**Condition Pipeline Generation**](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md#condition-pipeline-generation) (conditions API, offload → tiling, Stage 2 spatial) plus [**Two-stages Generation**](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md#two-stages-generation) (non-distilled Stage 1 + Hub distilled LoRA for Stage 2). `run.py` uses **`enable_model_cpu_offload`**, not sequential offload.
- Source in-tree: `diffusers.pipelines.ltx2` (installed via `pip install diffusers`)

---

## Question

Can **`LTX2ConditionPipeline`** in **diffusers** reproduce the same C2V-style behaviour as
our vendored `KeyframeInterpolationPipeline` (exp_016)? What differs between the two
conditioning APIs?

---

## Setup (matches `run.py` + `config.yaml`)

- **Library**: HuggingFace **`diffusers`** (+ **`accelerate`** for CPU offload, **`peft`** for LoRA loading).
- **Pipeline**: `LTX2ConditionPipeline` — see docs link above.
- **Conditioning**: `LTX2VideoCondition(frames=<list[PIL]>, index=<latent_idx>, strength=…)` as
  `conditions=[start_cond, end_cond]` on **Stage 1** only; Stage 2 refines upsampled latents
  (same pattern as the FLF2V two-stage example in the diffusers docs).
- **Inputs**: DAVIS **`start_clip` / `end_clip` MP4s** under `data/processed/DAVIS/` (same pairs as
  exp_016 `config_davis.yaml`).
- **Inference** (defaults in `config.yaml`):
  - **121** frames @ 24 fps (official LTX-2 default in the diffusers docs), **512 × 768** Stage 1;
    latent upsampler **×2** → **1024 × 1536** Stage 2 decode.
  - Stage 1: **40** steps, `guidance_scale=4.0`, `sigmas=None`.
  - Stage 2: **3** steps, `guidance_scale=1.0`, `STAGE_2_DISTILLED_SIGMA_VALUES`, distilled LoRA;
    **`height=height*2`, `width=width*2`** (required for `LTX2ConditionPipeline` — see `run.py` comments).

---

## Diffusers vs. vendored LTX-2 (exp_016)

| Aspect | exp_016 (`KeyframeInterpolationPipeline` in `src/LTX-2/`) | exp_020 (`LTX2ConditionPipeline` in **diffusers**) |
|---|---|---|
| Conditioning type | `ClipConditioningInput(path, frame_idx_px, strength, K)` | `LTX2VideoCondition(frames=list[PIL], index=lat_idx, strength)` |
| Coordinate system | Pixel frame offset | **Latent** frame index — `index=0` (start), `index=N_lat − K_lat` (end) |
| Input format | Path → MP4 decoded inside pipeline | MP4 → `torchvision` → `list[PIL]` in `run.py` |
| Stage 2 | Conditioning re-applied at ×2 | Latents only; spatial size passed as `height*2`, `width*2` |
| Model loading | `StateDictRegistry` / custom ledger | `from_pretrained("Lightricks/LTX-2")` + Hub LoRA / upsampler subfolders |
| Code reference | Vendored `src/LTX-2/` fork | **`diffusers`** pip package + HF docs |

### End-clip index (with `num_frames=121`, `num_clip_frames=25`)

```
LTX temporal scale = 8
latent_num_frames   = (121 - 1) // 8 + 1   = 16
clip_latent_frames  = (25 - 1) // 8 + 1   =  4
end_clip_index      = 16 - 4  = 12
```

`index=-1` is **not** used for multi-frame end clips in the VC setup.

---

## How to run

### 1. Environment

Use the conda env that has **diffusers**, **accelerate**, and **peft** (same as `run.py` docstring):

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/envs/diff
pip install "diffusers>=0.37" accelerate peft   # one-time if needed
```

### 2. Model cache

```bash
export HF_HOME=/workspace/cache/huggingface
```

First run downloads diffusers-format shards from the Hub into `HF_HOME`.

### 3. Launch

```bash
cd /workspace/diffusion-research
export HF_HOME=/workspace/cache/huggingface
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python experiments/exp_020_ltx2_c2v_diffusers/run.py
```

---

## Outputs

```
outputs/videos/exp_020_ltx2_c2v_diffusers/
└── run_NNNN/
    ├── run.log
    ├── config_snapshot.yaml
    ├── summary.yaml
    └── {sample_id}/
        ├── s{seed}_K{K}_steps{steps}.mp4
        └── config_snapshot.yaml
```

---

## Notes

- **Primary reference** for parameters and two-stage flow: **diffusers** [`ltx2.md`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2), not the Lightricks GitHub app repo.
- Outputs are comparable in spirit to exp_016; conditioning indices and guidance recipes differ
  (exp_020 follows diffusers defaults in `config.yaml`).
