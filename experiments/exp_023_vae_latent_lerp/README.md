# exp_023 — VAE Latent Linear Interpolation (dissolve-cause probe)

## Question

In **exp_020** (LTX-2 C2V via `LTX2ConditionPipeline`), the transition between the start
and end clips manifests as a **dissolve / cross-fade** rather than a hard cut. LTX-2 is a
**rectified-flow** model: it is trained to travel in a *straight line* from noise to data
in latent space. If the model simply "goes straight" from the start-clip latent to the
end-clip latent, the decoded video should look identical to **a VAE-only linear
interpolation** between the same two clip latents.

**Hypothesis:** the dissolve artefact is a consequence of the straight-path nature of
rectified flow — not of the transformer, the text conditioning, or the noise schedule.

**Test:** skip diffusion entirely. Encode start/end clips with the LTX-2 VAE, build the
full video by linearly interpolating along the temporal axis, decode directly.

- ✅ If the decoded video shows the *same* dissolve, the straight-path mechanism is a
  sufficient explanation.
- ❌ If it does not, the dissolve must arise from something else (text conditioning,
  free-middle frames being generated rather than blended, transformer dynamics, …).

---

## Setup

- **Library:** `diffusers` — but only `AutoencoderKLLTX2Video` (subfolder `"vae"` of
  [`Lightricks/LTX-2`](https://huggingface.co/Lightricks/LTX-2)). **The 19B transformer,
  text encoder, audio models, and Stage-2 upsampler are never loaded.**
- **Samples:** identical to exp_020 / exp_021 (10 DAVIS pairs across semantic classes
  1, 2, 5, 6, 8). Prompts are kept in the config for cross-experiment parity, but `run.py`
  does **not** read them.
- **Inputs:** `num_clip_frames=25` pixel frames per clip → `T_clip = (25-1)/8 + 1 = 4`
  latent frames per clip.
- **Inference:** height=512, width=768, frame_rate=24 fps. `num_frames` swept across
  **{121, 73, 57}** to vary the size of the "free middle" region:

  | `num_frames` | `T_total` | `T_clip` | free-middle latent frames | semantics |
  |---|---|---|---|---|
  | 121 | 16 | 4 | 8 | matches exp_020/021 — longest blend region |
  | 73 | 10 | 4 | 2 | tight blend region |
  | 57 | 8 | 4 | 0 | start clip → end clip directly, every frame is a blend |

  Constraint `(num_frames − 1) % 8 == 0` is required by the LTX-2 VAE temporal scale.

### Interpolation scheme

For each output latent frame `t ∈ [0, T_total)`:

```
alpha   = t / (T_total - 1)               # 0 → 1 linearly
s_idx   = min(t, T_clip - 1)              # clamp into start-clip range [0, T_clip)
e_idx   = max(t - (T_total - T_clip), 0)  # clamp into end-clip   range [0, T_clip)
out[t]  = (1 - alpha) * start_lat[s_idx] + alpha * end_lat[e_idx]
```

Boundary behaviour: `t=0` returns pure `start_lat[0]`; `t=T_total-1` returns pure
`end_lat[T_clip-1]`; the middle is a smooth linear cross-blend between `start_lat[-1]`
and `end_lat[0]`.

---

## How to run

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/envs/diff
export HF_HOME=/workspace/cache/huggingface
cd /workspace/diffusion-research
python experiments/exp_023_vae_latent_lerp/run.py
```

First run downloads only the VAE shard of `Lightricks/LTX-2` into `HF_HOME`; subsequent
runs reuse the local cache. Single GPU (`cuda:0`), bfloat16, with `vae.enable_tiling()`
to keep memory bounded.

---

## Outputs

```
outputs/videos/exp_023_vae_latent_lerp/
└── run_NNNN/
    ├── run.log
    ├── config_snapshot.yaml
    ├── summary.yaml
    └── {sample_id}/
        ├── s{seed}_K{num_clip_frames}_N{num_frames}_lerp.mp4
        └── config_snapshot.yaml
```

The MP4 filename encodes the seed, clip-frame count `K`, and total frame count `N` so
sweeps across `num_frames` are distinguishable on disk.

---

## Outcome (per existing runs)

- `run_0003` — `num_frames=121` (matches exp_020 geometry)
- `run_0004` — `num_frames=73`
- `run_0005` — `num_frames=57` (current default)

Visual comparison against the corresponding exp_020 videos is the qualitative check for
the hypothesis above.
