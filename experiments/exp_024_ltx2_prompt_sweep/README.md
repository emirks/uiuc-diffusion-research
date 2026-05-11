# exp_024 — LTX-2 Prompt Optimization Sweep

## Question

How much does prompt variation alone move transition quality on LTX-2 C2V, given fixed source/target conditioning clips from the DAVIS subset?  This is a pure text-variable experiment — everything else held constant.

## Setup

**Baseline:** exp_020 (`LTX2ConditionPipeline`, two-stage pipeline, DAVIS pairs).

**Independent variable:** Prompt category (6 levels).

**Fixed conditions:**

| Parameter | Value |
|-----------|-------|
| Model | `Lightricks/LTX-2` |
| DAVIS pairs | 10 (same subset as exp_020) |
| Clip conditioning | 25 px frames (4 lat), strength 1.0 each end |
| Duration | 193 frames ≈ 8 s at 24 fps |
| Stage 1 steps | 40 |
| CFG (guidance\_scale) | 3.2 |
| Seed | 42 |
| Stage 1 resolution | 512 × 768 |
| Stage 2 resolution | 1024 × 1536 (×2 upsampler) |
| enhance\_prompt | False for all categories (not supported by LTX2ConditionPipeline in Diffusers) |

**Prompt categories (6):**

| Category | Description | enhance\_prompt |
|----------|-------------|----------------|
| `A_empty` | Empty string — model floor | False |
| `A_word` | `"transition"` — minimal signal | False |
| `B` | Generic cinematic structure, no named mechanism | False |
| `C` | Abstract transition descriptor, object-agnostic (identical text for all pairs) | False |
| `D` | Typed creative transition with explicit source/target object reference | False |
| `E` | Timed version of B with explicit timing language (`"at 4s"`, `"by 6s"`) | False |

**D\_enhanced (enhance\_prompt=True)** was planned but is not supported by `LTX2ConditionPipeline` in Diffusers — the parameter lives only in the vendored `ltx-pipelines` stack.  Omitted.

**Total runs:** 6 categories × 10 DAVIS pairs = 60.

**Prompt format (B, C, D, E):** Single flowing paragraph, present tense, 4–8 sentences, ≤200 words.  Cinematographic: shot scale, lighting, action arc (chronological), subject (physical cues only), one camera movement with lens/aperture, ambient audio.  No lists, no headers, no line breaks within paragraph.

**Category C prompt is universal** (same text for all pairs): describes abstract visual flow and motion-blur transition mechanics without naming any object.

**Negative prompt** (per LTX-2 Step 4 specification, different from exp_020):
> morphing, distortion, warping, flicker, jitter, stutter, shaky camera, erratic motion, temporal artifacts, frame blending, low quality, jpeg artifacts, text, watermark, logo, cartoon, anime, CGI

## How to run

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate /workspace/envs/diff
cd /workspace/diffusion-research
python experiments/exp_024_ltx2_prompt_sweep/run.py
```

## Outputs

```
outputs/videos/exp_024_ltx2_prompt_sweep/run_0001/
  summary.yaml                              # all 60 runs with paths and timing
  config_snapshot.yaml                      # full config
  {sample_id}/{category}/
    s42_cat{category}_steps40.mp4           # 1024×1536, ≈8s
    config_snapshot.yaml                    # per-run config + prompt text
```

## Expected outcome

- `A_empty` and `A_word` establish the floor — raw conditioning-only behavior.
- `B` tests whether correct LTX-2 structure alone beats the floor.
- `C` tests whether abstract cinematic language generalizes across pairs.
- `D` tests whether naming the transition mechanism and both objects raises quality.
- `E` tests whether temporal language (`"at 4s"`, `"by 6s"`) controls transition timing.

Results feed Step 4 of the research roadmap: establishing how much headroom prompting provides before architectural interventions (Steps 6–8).
