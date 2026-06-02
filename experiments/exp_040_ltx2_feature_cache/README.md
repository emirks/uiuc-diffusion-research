# exp_040 — LTX-2 RF-Solver Inversion + Comprehensive Feature Cache

Fork of [`exp_033`](../exp_033_ltx2_rf_inv_drop1) — the **§0-deployable floor**
of the closed RF-inversion research loop (see
[`notes/rf_inversion_postmortem.md`](../../notes/rf_inversion_postmortem.md)).

The inversion recipe (drop1: production sub-clip anchors with the end-clip
first-latent-frame token dropped from the mask) is **byte-identical** to
exp_033. The only addition is hook-based **feature caching** during all three
phases (invert / reconstruct / regenerate). Hooks are pure observers — no
gradients, no behavioral change. PSNR / SSIM / LPIPS between exp_033 and
exp_040 should match to numerical precision.

This is a **wide gathering experiment**: the cache surface is broad enough to
seed multiple downstream injection-style edits (PnP / MasaCtrl / DiTCtrl /
FreeControl / TokenFlow). It is **not** itself an editing experiment.

---

## Question

Once the inversion recipe is frozen, what intermediate tensors do we need to
capture so a downstream feature-injection regen can replay them without
re-running the inverter?

## Setup

* Inversion: RF-Solver midpoint 2nd-order, CFG=1, 40 steps, drop1 anchors
  (= exp_033 recipe verbatim).
* Reconstruction: same midpoint solver, CFG=1, 40 steps (solver self-
  consistency).
* Regeneration: Euler, CFG=4, 40 steps (production-trajectory recovery).
* Cache attached to `pipe.transformer` once at startup; gated per (phase,
  step, substep, layer, site) at hook time.

## How to run

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/envs/diff
cd /workspace/diffusion-research
python experiments/exp_040_ltx2_feature_cache/run.py
```

A single sample can be run by truncating `samples:` in `config.yaml` to the
desired entry — the cache config is otherwise unchanged.

## Outputs

Per sample under `outputs/videos/exp_040_ltx2_feature_cache/run_NNNN/<sample_id>/`:

* All standard exp_033 artifacts: `z0.pt`, `z1.pt`, `z_t_25/50/75.pt`,
  `source_video.mp4`, `recon_video.mp4`, `regen_video.mp4`, `step_diag_*.csv`,
  `inv_meta.yaml`.
* `feature_cache/` — the new artifact (see schema below).

---

## What the cache contains and why

### Field landscape (what people inject / where)

| Method family | Site cached for replay | Notes |
|---|---|---|
| Plug-and-Play (Tumanyan '23) | `block_out` (early-mid layers, early steps) | Structure preservation |
| MasaCtrl / DiTCtrl | self-attn `K`, `V` (mid layers, all steps) | Identity transfer w/ layout adaptation |
| Prompt-to-Prompt | cross-attn maps (derive from `attn2_q · attn2_k`) | Text-conditioned editing |
| FreeControl | `block_out` PCA bases | Generic semantic control |
| TokenFlow | self-attn `Q`, `K`, `V` across frames | Video consistency |
| Pivotal/inversion | `z_T`, `v_pred` per step | Trajectory replay, score-guided edits |

The cache surface in this experiment covers all of the above:

* **`z_t_in`** and **`v_pred`** per outer step (full packed latent
  `[B, N, 128]`) — for trajectory replay and score-guided methods.
* **`block_out`** per (step, layer) — for PnP, FreeControl.
* **`attn1_q/k/v`** per (step, layer) — for MasaCtrl, DiTCtrl, TokenFlow.
* Optional **`attn2_q/k/v`**, **`ff_out`** sites for P2P / FreeControl PCA.
* σ-schedule and clamp anchors (`clean_latents`, `conditioning_mask`) — so a
  downstream regen can reproduce the exact integration grid.

### Hook sites (`src/diffusion/feature_cache.py`)

For each block index `l` in `layer_indices`, forward hooks are registered on:

| Site name | Module hooked | Output shape | Notes |
|---|---|---|---|
| `block_out` | `transformer_blocks[l]` | `[B, N_video, 4096]` | Block returns `(video, audio)` tuple; we keep `[0]`. |
| `attn1_q` | `transformer_blocks[l].attn1.to_q` | `[B, N_video, 4096]` | **Pre-RMSNorm, pre-RoPE.** |
| `attn1_k` | `transformer_blocks[l].attn1.to_k` | `[B, N_video, 4096]` | **Pre-RMSNorm, pre-RoPE.** |
| `attn1_v` | `transformer_blocks[l].attn1.to_v` | `[B, N_video, 4096]` | V is not norm'd / RoPE'd. |
| `attn2_q` | `transformer_blocks[l].attn2.to_q` | `[B, N_video, 4096]` | Video → text cross-attn Q. |
| `attn2_k` | `transformer_blocks[l].attn2.to_k` | `[B, N_text≈128, 4096]` | Text-side K. |
| `attn2_v` | `transformer_blocks[l].attn2.to_v` | `[B, N_text≈128, 4096]` | Text-side V. |
| `ff_out` | `transformer_blocks[l].ff` | `[B, N_video, 4096]` | Video FFN output. |
| `audio_attn1_q/k/v` | `transformer_blocks[l].audio_attn1.*` | `[B, N_audio, 2048]` | Audio self-attn. |
| `a2v_q/k/v` | `transformer_blocks[l].audio_to_video_attn.*` | Q: `[B, N_video, ·]`, K/V: `[B, N_audio, ·]` | Audio → video cross-attn. |

**B (batch dim)** is `1` for invert/recon (CFG=1) and `2` for regen with
CFG>1 (uncond cat cond, in that order). Split at inject time.

**`N_video`** = `F' × H' × W'` = packed latent token count. For the default
exp_040 setup (121-frame 512×768): `F'=16, H'=16, W'=24` → **N_video = 6144**.

**Hidden dim D = 4096** (32 heads × 128 head_dim, from `transformer/config.json`).

### Loop-recorded ("cheap") payload per outer step

Recorded explicitly by the invert/recon/regen loops in `run.py` whenever the
current step is in the save grid:

| Key | Shape | Notes |
|---|---|---|
| `z_in` | `[B, N_video, 128]` bf16 | Packed latent into the transformer at this substep. |
| `v_pred` | `[B, N_video, 128]` bf16 | Raw post-CFG-mix output of the transformer (matches what the loop integrates). |
| `sigma` | float | σ value at which this substep was evaluated. |

For midpoint phases (invert, recon) this dict has both `predictor` and
`corrector` keys when `substeps_midpoint = ["predictor", "corrector"]`.
For regen the only key is `euler`.

### Per-step file schema

`<sample_dir>/feature_cache/<phase>/step_NNN.pt` is a single dict containing
both cheap and heavy payload for that outer step:

```python
{
    "phase":      "invert" | "recon" | "regen",
    "step_idx":   int,
    "sigma_curr": float,
    "sigma_next": float,
    "sigma_mid":  float | None,   # midpoint phases only
    "dtau":       float,
    "t_value":    float,           # = sigma_curr * 1000
    "step_payload": {
        "predictor": {"z_in": Tensor, "v_pred": Tensor, "sigma": float},
        # "corrector": {...} when enabled
        # OR for regen only:
        # "euler":     {"z_in": Tensor, "v_pred": Tensor, "sigma": float},
    },
    "blocks": {
        "predictor": {
            layer_idx: {
                "block_out": Tensor[B, N_video, 4096]  bf16,
                "attn1_q":   Tensor[B, N_video, 4096]  bf16,
                "attn1_k":   Tensor[B, N_video, 4096]  bf16,
                "attn1_v":   Tensor[B, N_video, 4096]  bf16,
                # plus any of attn2_*, ff_out, audio_*, a2v_* that you enabled
            },
            ...
        },
        # "corrector": {...} when enabled
        # OR for regen:
        # "euler":     {...}
    },
}
```

### Static (per-sample) payload

`<sample_dir>/feature_cache/static.pt` is a single dict containing the
sample-invariant tensors needed to reproduce the integration grid and the
clamp / conditioning state:

```python
{
    "sample_id":         str,
    "prompt":            str,
    "negative_prompt":   str,
    "clean_latents":     Tensor[1, N_total, 128]  bf16,  # production anchors
    "conditioning_mask": Tensor[1, N_total, 1]    bf16,  # 1 = pinned, 0 = free
    "z0_packed":         Tensor[1, N_total, 128]  bf16,  # source latent
    "audio_context":     Tensor[1, N_audio, *]    bf16,  # encoded silent mel
    "latent_num_frames": int,
    "latent_height":     int,
    "latent_width":      int,
    "num_frames":        int,
    "height":            int,
    "width":             int,
    "frame_rate":        float,
    "drop_token_range":  (int, int),  # masked-out token slice (end-clip first lat frame)
    "end_latent_idx":    int,
}
```

### Manifest

`<sample_dir>/feature_cache/manifest.yaml` is a small index emitted at
sample completion:

```yaml
schema_version: 1
model_id:        Lightricks/LTX-2
num_blocks:      48
layer_indices:   [3, 11, 19, 28, 37, 46]
step_indices:
  invert: [0, 4, 9, 14, 19, 24, 29, 34, 39]
  recon:  [0, 4, 9, 14, 19, 24, 29, 34, 39]
  regen:  [0, 4, 9, 14, 19, 24, 29, 34, 39]
substeps_midpoint: [predictor]
sites:           [block_out, attn1_q, attn1_k, attn1_v]
dtype:           bfloat16
saved_steps:
  invert: [0, 4, 9, 14, 19, 24, 29, 34, 39]
  recon:  [0, 4, 9, 14, 19, 24, 29, 34, 39]
  regen:  [0, 4, 9, 14, 19, 24, 29, 34, 39]
```

---

## Sizing and tradeoffs

Per-tensor size (bf16, N_video=6144, D=4096): `6144 × 4096 × 2 = 50 MB`.

**Default config** (`config.yaml` shipped in this directory):

| Quantity | Value |
|---|---|
| Layers per save slot | 6 spread (3, 11, 19, 28, 37, 46) |
| Steps per phase | 9 (every 5 + final) |
| Substep (midpoint) | predictor only |
| Sites | block_out + attn1_q/k/v (4 sites) |
| Heavy tensor count per phase | 6 × 9 × 4 = 216 |
| Heavy MB per phase | 216 × 50 ≈ 10.8 GB |
| Phases per sample | 3 (invert, recon, regen) |
| **Per-sample disk** | **≈ 33 GB** |
| **10-sample run total** | **≈ 330 GB** |

The `feature_cache.dtype: "bfloat16"` setting is essentially required —
float16 has less range than the activations sometimes need, and float32
doubles disk.

### Widening — common variants

* **All 48 blocks, current step grid** — set `layer_indices: "all"`. Disk
  scales 8× (≈ 264 GB per sample, 2.6 TB for 10 samples).
* **All 40 steps, current 6 layers** — set every `step_indices_*: "all"`. Disk
  scales 40/9 ≈ 4.4× (≈ 145 GB per sample).
* **Add corrector substep** — set `substeps_midpoint: ["predictor", "corrector"]`.
  Doubles invert + recon heavy disk (regen unchanged).
* **Add text cross-attn** — add `attn2_q`, `attn2_k`, `attn2_v` to `sites`.
  Adds ~50 MB / (step, layer) for `attn2_q` and ~1 MB each for `attn2_k/v`
  (text seq length only ~128 tokens).
* **Add FFN** — add `ff_out` to `sites`. Adds another 50 MB / (step, layer).

### Narrowing — common variants

* **Pilot run on 2 samples** — truncate `samples:` to two entries
  (e.g. `shadow_smoke_0` and `shadow_smoke_4`, the two best on exp_033).
* **Drop regen capture** — set `step_indices_regen: []`. Regen is the most
  expendable phase (no Q/K/V replay site uses it).
* **Drop block_out** — keep only Q/K/V. Halves heavy per-(step, layer).

---

## Reloading the cache

```python
import torch, pathlib, yaml

sample_dir = pathlib.Path("outputs/videos/exp_040_ltx2_feature_cache/run_0001/shadow_smoke_4")

# Manifest tells you what was captured.
manifest = yaml.safe_load((sample_dir / "feature_cache/manifest.yaml").read_text())

# Static state — clean anchors, mask, σ-grid prerequisites.
static = torch.load(sample_dir / "feature_cache/static.pt", weights_only=False)
clean_latents     = static["clean_latents"]
conditioning_mask = static["conditioning_mask"]

# A single invert step.
step = torch.load(sample_dir / "feature_cache/invert/step_009.pt", weights_only=False)
sigma_curr = step["sigma_curr"]
predictor_payload = step["step_payload"]["predictor"]   # {"z_in", "v_pred", "sigma"}
predictor_blocks  = step["blocks"]["predictor"]         # {layer_idx: {site: Tensor}}

# MasaCtrl-style K, V at layer 19, step 9.
k_layer19 = predictor_blocks[19]["attn1_k"]   # [B=1, 6144, 4096]
v_layer19 = predictor_blocks[19]["attn1_v"]   # [B=1, 6144, 4096]
```

### Mapping invert step → regen step for replay

Both invert and regen use the **same σ grid** (regen descends, invert
ascends). Step `k` of invert visits the same σ pair as step `(S-1-k)` of
regen — so to inject at regen step `j`, load from invert step `S-1-j`.

In code:

```python
S = 40                      # num inversion steps
def invert_for_regen_step(j):
    return S - 1 - j
```

Sigma equivalence holds exactly because exp_033 / exp_040 reuse
`stage1_scheduler` and `_build_sigma_grid` for both directions.

---

## Caveats

1. **Q,K,V are pre-RMSNorm and pre-RoPE.** LTX-2's `LTX2Attention.forward`
   applies `qk_norm = "rms_norm_across_heads"` to Q and K after the linear
   projection, then rotary embeddings, then enters scaled dot-product
   attention. Hooks on `to_q`/`to_k`/`to_v` capture the **rawest** state.
   This is the most reusable cache point (RoPE positions are deterministic;
   re-applying RMSNorm + RoPE at inject time is trivial), but if you want
   the *exact* tensors that SDP consumed, apply
   `block.attn1.norm_q(q) / norm_k(k)` then the layer's rotary embedding
   yourself. V is not norm'd or rotary'd inside attn1, so V is identical
   pre/post.

2. **CFG batching is preserved.** Regen at CFG=4 runs the transformer with
   `[B=2, N, ...]` (uncond, cond in that order — matches
   `LTX2ConditionPipeline.__call__` line 1208). All captured regen tensors
   inherit `B=2`. Split via `t[0]` (uncond) and `t[1]` (cond) at inject
   time. Invert/recon at CFG=1 have `B=1`.

3. **Block output is video-stream only.** `LTX2VideoTransformerBlock.forward`
   returns `(video_hidden_states, audio_hidden_states)`. The hook keeps
   `out[0]`. If you want the audio stream too, enable `audio_*` sites.

4. **AdaLN modulation parameters are NOT cached.** The per-block scale/shift
   /gate values for AdaLN-zero modulation are derived from `temb` (a
   transformer-top input) and the per-block `scale_shift_table` parameter.
   The `scale_shift_table` is a static `nn.Parameter` (snapshot once with
   `pipe.transformer.transformer_blocks[l].scale_shift_table`); `temb` is
   derivable from `sigma_curr` and the timestep MLP. Both are recoverable
   without a hook. If you need them inline, hook `block.scale_shift_table`
   readers or just save `temb` at the transformer-top forward.

5. **Hooks share state across calls.** The cache holds at most one
   (step × substep) worth of heavy tensors in memory; `flush_step` writes
   the per-step file and clears the buffer. Memory peak is bounded by
   `n_layers × n_sites × per-tensor-size`. For the default config this is
   `6 × 4 × 50 MB ≈ 1.2 GB` CPU residency between substeps — well under any
   pod's RAM ceiling.

6. **`v_pred` cached is *raw* (post-CFG-mix, pre-x0-clamp).** The `_x0_clamp_velocity`
   step that the loop runs *after* the transformer call is NOT applied to the
   cached `v_pred`. If you want the clamped velocity, apply the same clamp:
   ```python
   x0_pred = z_in - v_pred * sigma
   x0_pred_clean = x0_pred * (1 - mask) + clean_latents * mask
   v_clamped = (z_in - x0_pred_clean) / sigma  # for sigma > 1e-4
   ```
   Storing raw `v_pred` is the right choice — the clamp is a deterministic
   function of the mask + anchors, both of which live in `static.pt`.

7. **One per-step .pt per phase per sample.** With 9 saved steps × 3 phases
   the file count per sample is 27 (+ 1 static + 1 manifest = 29). Across
   10 samples = 290 files. `torch.load` is fast enough that this is fine;
   if you find yourself doing bulk-load patterns, consider switching to
   a single `.zarr` or single concatenated `.pt` per phase — easy to add.

8. **Disk usage scales fast with widening.** At `layer_indices: "all"` +
   `step_indices_*: "all"` + `substeps_midpoint: ["predictor", "corrector"]`
   + every video site enabled (block_out, attn1_q/k/v, attn2_q/k/v, ff_out)
   per-sample disk reaches roughly:
   `48 layers × 40 steps × 2 substeps (mp) × 8 sites × ~50 MB ≈ 1.5 TB / sample (invert+recon)`,
   plus another `48 × 40 × 1 × 8 × 50 ≈ 750 GB` for regen.
   Total ≈ **2.3 TB per sample** at full bore. Don't enable everything at
   once; widen one axis at a time as injection experiments demand it.

9. **Hooks are pure observers.** Numerical results (recon PSNR/SSIM/LPIPS,
   regen ditto) should be bit-identical to exp_033 modulo non-determinism
   sources. Any drift > LSB indicates a bug in the hook path.

---

## What this enables next

* **exp_041_*** — first injection experiment. Run a regen on a target
  (edited prompt / replaced anchor), inject cached `attn1_k` and
  `attn1_v` from the source's invert path at mid-layers. Test MasaCtrl-style
  identity transfer.
* **exp_042_*** — PnP-style `block_out` injection at early layers, early
  steps. Test whether replacing the early structural commitment of the
  regen pass with the inversion's early-step block outputs cleans up the
  transition zebra-artifact regime documented in the postmortem.
* **FreeControl PCA basis** — fit PCA on the captured `block_out` tensors
  across all 10 samples; the resulting subspace is a candidate semantic
  basis for the transition feature.
* **Trajectory replay** — use the `step_payload` series alone (no hidden
  states) to reconstruct the σ-by-σ velocity field and drive a
  score-guidance correction at regen time.
