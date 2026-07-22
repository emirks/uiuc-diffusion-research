# exp_075 — Procedural transition operator engine

## Question

Can we manufacture transition training data at a scale the 49 real clips cannot reach,
by applying a large bank of **procedural operators** to arbitrary endpoint pairs?

Three things this is supposed to buy, none of which the real corpus provides:

1. **Task diversity** — thousands of distinct operators instead of ~11 style classes.
2. **Operator ⊥ content factorisation** — the operator is sampled independently of the
   endpoints, so nothing about the content predicts the effect.
3. **Counterfactuals** — the *same* endpoint pair under many different operators. The
   real corpus has exactly one transition per endpoint pair, so a model trained on it
   alone can satisfy the data by memorising content→effect associations.

This experiment is the **feasibility + sample-quality check**, not the pretraining run.

## Setup

**Input.** The latest 9-frame endpoint clips,
`experiments/exp_062_ladder_r2r3_specialists/dataset/cond/*_{start9,end9}.mp4`
(44 complete pairs, 480×640, 24 fps). Verified to be exact slices of the 121-frame
corpus clips (MAE 0.42/0.61 vs `data/processed/transitions_std121`, i.e. re-encode noise).

**Operator bank.** [gl-transitions](https://github.com/gl-transitions/gl-transitions) —
125 MIT-licensed GLSL shaders, cloned to `$LAB/misc/gl-transitions`. Each shader
declares its tunable parameters as uniforms with the default in a trailing comment
(`uniform float bounces; // = 3.0`), which `engine/shaders.py` parses into a typed
parameter space per shader. That is what makes the bank *parameterised* rather than a
fixed set of 125 effects.

An **operator** is the tuple

```
(shader, sampled uniforms, easing curve, spatial flip, direction swap,
 layer-extension policy, auxiliary map)
```

- 122 usable shaders, 85 of which have 1–10 tunable uniforms
- 12 easing curves (`engine/streams.py`), including `snap_late`, `snap_early`, `mid_hold`
- 4 spatial flips × direction swap → turns a left wipe into a right wipe, an "open" into
  a "close", without touching the shader
- `luma.glsl` / `displacement.glsl` take an auxiliary image sampler, fed from 7 families
  of procedural map (`engine/maps.py`) — each map is a distinct operator

Estimated distinguishable operators: **~1.7 × 10⁶** (`operators.bank_capacity`).

**Rendering.** `moderngl` on an EGL standalone context. On this cluster that resolves to
Mesa **llvmpipe (GL 4.5 core)** — software rendering, so the whole engine runs on plain
CPU nodes and never competes with training for the H100/H200 pools. On a node with the
NVIDIA driver visible the same code picks up `libEGL_nvidia` instead.

**Output contract.** 121 frames @ 480×640/24 fps, `(121-1) % 4 == 0`. Frames 0–8 are the
`start9` endpoint and frames 112–120 are the `end9` endpoint, so each clip is a drop-in
sample for the existing transition task.

### Two design decisions worth flagging

**Layer extension.** We are given 9 frames per endpoint but need both layers for all 121.
`hold` freezes them — which would teach the model that *all motion stops during a
transition*, exactly the wrong prior. `boomerang` ping-pongs the 9 frames; `flow`
extrapolates the terminal Farneback flow with a decay. The run renders all three on one
fixed (pair, operator) so the difference is directly visible. Default: `boomerang`.
At corpus scale the right answer is to feed full-length source clips and skip this
entirely — extension only exists because the input here is 9 frames.

**Endpoint-identity gate — and why it has to run twice.** Progress is pinned to 0 across
frames 0–8 and 1 across 112–120, so the conditioning blocks reproduce the inputs
*provided* the shader satisfies `transition(uv, 0) == from` and `transition(uv, 1) == to`.
Not all of them do, and a shader that fails silently corrupts the conditioning frames of
every sample it generates — the model would be trained on endpoints that disagree with
its own inputs.

*Gate 1, per shader, at default parameters* (`operators.validate_bank`). All 125 compile;
**3 are rejected**: `tangentMotionBlur` (MAE 57.1), `AdvancedMosaic` (42.9),
`InvertedPageCurl` (2.01, just over the 2.0 tolerance).

*Gate 2, per operator, at sampled parameters* (`operators.sample_valid_operator`).
Gate 1 is **not sufficient** — this run found it empirically. The identities are
**parameter-dependent**: `undulatingBurnOut` is clean at its defaults but left MAE 8.1 on
the end block at a sampled `smoothness`, and `colorphase` reached **MAE 53.8**. Measured
over 400 draws, **3.8% of sampled operators violate the identity despite their shader
passing gate 1**. Every operator is therefore rejection-sampled against the actual
endpoint frames it will be rendered with (2 extra renders, ~16 ms).

Gate 1 is also **resolution-dependent** — `InvertedPageCurl` passes at 120×160 and fails
at 480×640 — so it must be run at the production resolution, not a cheap proxy.

Measured endpoint MAE on accepted operators: **0.000**.

## How to run

```bash
# CPU-only, ~1 s per 121-frame clip
sbatch job_render.sbatch                    # partition=secondary, 16 cpus, no GPU

# then build the viewer and serve from the REPO ROOT so relative paths resolve
python build_viewer.py outputs/videos/exp_075_procedural_transition_engine/run_NNNN
cd $LAB/diffusion-research && python -m http.server 8000
# → /outputs/videos/exp_075_procedural_transition_engine/run_NNNN/viewer.html
```

Do **not** run `run.py` on a login node: llvmpipe, swscale and x264 each try to open a
per-core thread pool and trip the per-user thread limit (`EAGAIN` out of swscale init).
`job_render.sbatch` caps `LP_NUM_THREADS` / `OMP_NUM_THREADS` for this reason.

## Outputs

`outputs/videos/exp_075_procedural_transition_engine/run_NNNN/`

| File | Contents |
|---|---|
| `videos/*.mp4` | 121-frame procedural transitions |
| `filmstrips/*.jpg` | 11-frame contact sheet per clip |
| `manifest.json` | full operator spec + endpoint MAE + render time per clip |
| `bank_validation.json` | gate 1 — per-shader compile / endpoint-identity result |
| `operator_gate.json` | gate 2 — sampled/rejected counts + rejection by shader |
| `viewer.html` | grouped side-by-side viewer |

Sample blocks, matching the viewer's sections:

- **counterfactual** — 1 endpoint pair × 12 operators (content fixed, operator varies)
- **sharedop\<k\>** — 3 operators × 4 endpoint pairs (operator fixed, content varies)
- **diverse** — 8 pairs × 5 random operators; half the pairs draw their two endpoints
  from *different* source clips
- **ext_{boomerang,hold,flow}** — layer-extension ablation
- **REAL** — the human-made transition for the same endpoints, for calibration

## Expected outcome / what to look for

The engine is viable if (a) endpoint MAE stays at ~0 across the bank, (b) the
counterfactual block reads as many genuinely different transitions over identical
content, and (c) the shared-operator block reads as one recognisable effect over
different content.

**Not settled here:** whether these clips are good enough to pretrain on. Shader
compositing produces *2D overlay* transitions; the real corpus is dominated by *semantic
morphs* (animalization, gas transformation, portal). Whether the former transfers to the
latter is the open question, and it is a question about pretraining, not about the engine.

## Next

1. **UV pre-transform (deferred, with a caveat).** Rendering in a rotated UV frame would
   turn every axis-aligned wipe into an arbitrary-angle wipe — nominally ×8–16 on the
   whole bank. But it cannot be done by transforming `_uv` before `transition()`: the
   shaders call `getFromColor(uv)` themselves, so rotating the effect frame also rotates
   the sampled content and breaks the endpoint identity. Doing it correctly means
   rotate-input → render → rotate-back, which costs two resamples and needs a zoom-crop to
   avoid black corners. The 4 spatial flips are the exact, free subset of this and are
   already in.
2. **Easing bank ×43.** Lift the closed-form easing curves from `scriptituk/xfade-easing`
   (MIT) to replace our 12. Cheapest large multiplier, and perceptually salient in video.
3. **Held-out operator split.** ffmpeg's 57 `xfade` built-ins are a separate, exactly
   reproducible family — never pretrain on them, evaluate operator generalisation on them.
4. **Depth-reprojection family.** Depth Anything V2-Small (Apache-2.0) + parameterised
   camera paths as a single `grid_sample`; no inpainting needed because the incoming
   stream covers disocclusions. ~250 operators. Label as a separate, content-dependent
   family and exclude from any K sweep.
5. **The K sweep** — the headline. K ∈ {2⁴ … 2¹⁴} operators at *fixed total clip count*,
   evaluated on held-out operators and on the 49 real transitions.

Background, citations and the tooling survey: `notes/dataset/procedural_operators.md`.
