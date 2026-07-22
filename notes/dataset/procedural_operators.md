# Procedural transition operators — tooling survey and design notes

Background for the **procedural operator data engine**: synthesise (start9, end9,
transition) triples by applying a large bank of procedural operators to arbitrary
endpoint pairs, to get task diversity, operator ⊥ content factorisation, and
counterfactuals (same endpoints, many operators) that the 49 real clips cannot provide.

Facts marked **[verified]** were checked on this cluster (exp_075, 2026-07-22). The rest
is from the cited sources.

---

## 1. The shader bank: gl-transitions

`https://github.com/gl-transitions/gl-transitions` → cloned to `$LAB/misc/gl-transitions`.

- **125 shaders**, not the ~80 the website suggests. **[verified]** 123 MIT, 1 BSD-2, 1 BSD-3
  (per-file `// license:` headers; the repo shows `NOASSERTION` on GitHub for that reason).
- **189 tunable uniforms across 86 shaders**; 39 have none. **[verified]**
  Distribution of uniform count → shader count: `0:39 1:34 2:27 3:15 4:4 5:3 6:1 9:1 10:1`.
- Each `.glsl` defines only `vec4 transition(vec2 uv)` and may read the host-provided
  `progress`, `ratio`, `getFromColor(uv)`, `getToColor(uv)`.
- Parameters carry their default in a trailing comment — this is the whole reason the bank
  is *parameterised*:
  ```glsl
  uniform float bounces;   // = 3.0
  uniform vec2  direction; // = vec2(0.0, 1.0)
  uniform ivec2 size;      // = ivec2(4)
  ```
  Regex: `^\s*uniform\s+(\w+)\s+(\w+)\s*;\s*(?://|/\*)\s*=\s*(.+)$`. Two shaders
  (`luma`, `displacement`) additionally take a `sampler2D` and need an auxiliary map.

### Pitfalls (all three cost real time)

1. **Unset uniforms default to 0**, which silently turns `bounces=3` into `bounces=0` — a
   degenerate operator that looks like a duplicate. Always set every parsed parameter.
2. **Texture y-flip.** GL's texture origin is bottom-left, video frames are top-left, so
   `getFromColor` must sample at `vec2(uv.x, 1.0 - uv.y)`. Getting it wrong *looks fine*
   but aliases `wipeup` ↔ `wipedown`, silently halving directional diversity.
3. **`#version 330` vs WebGL-1 source.** 3 of the 125 use the `texture2D` spelling; a
   `#define texture2D texture` in the preamble fixes them. **[verified]** all 125 compile
   under 330 core with that one define.

### Endpoint-identity gate — do not skip

The spec requires `transition(uv, 0) == from` and `transition(uv, 1) == to`, but not every
shader honours it. Since our training samples pin progress to 0/1 across the conditioning
blocks, a violating shader silently corrupts the endpoints of every sample it generates.

Measured against random-noise image pairs, MAE on a 0–255 scale **[verified]**:

| shader | MAE @ p=0 | MAE @ p=1 | verdict |
|---|---|---|---|
| `tangentMotionBlur` | 57.12 | 57.08 | reject |
| `AdvancedMosaic` | 42.85 | 42.85 | reject |
| `InvertedPageCurl` | 2.01 | 0.00 | reject at tol 2.0 (borderline) |
| other 122 | ≤ 2.0 | ≤ 2.0 | accept |

Accepted shaders render real clips at **0.000** endpoint MAE. **[verified]**

---

## 2. Runners — what works headless

**Use `moderngl` + EGL.** `pip install moderngl glcontext`, no root, no X server.

```python
ctx = moderngl.create_standalone_context(backend="egl")   # works as-is on cc
```

**[verified]** on `cc-login5` this yields `llvmpipe (LLVM 21.1.7) | 4.5 (Core Profile) Mesa
25.2.7` — i.e. **software rendering, which is a feature**: the engine runs on ordinary CPU
nodes and never competes with training for the H100/H200 pools. Measured **~1 s per
121-frame 480×640 clip** on a CPU node.

⚠️ **If you want the GPU, assert it.** `/usr/share/glvnd/egl_vendor.d/` contains both
`10_nvidia.json` and `50_mesa.json`, and enumeration order is not guaranteed, so an EGL
context on a GPU node can silently land on llvmpipe and be ~100× slow. Assert
`"NVIDIA" in ctx.info["GL_RENDERER"]`, or pin with
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
EGL device indices are independent of `CUDA_VISIBLE_DEVICES`.

### Runners that do NOT work here

| Runner | Why not |
|---|---|
| `transitive-bullshit/ffmpeg-gl-transition` | Source patch to ffmpeg's internal filter API; last push 2024-07, "ffmpeg 7.0 support?" still an open issue. The only 8.x port (PR #82) was closed unmerged. |
| `livingbio/ffmpeg-gl-transition` | archived 2025-09 |
| `editly`, `gl-transition-scripts` | Node + `headless-gl` → needs `xvfb-run` + software Mesa + node-gyp |
| `vispy`, `glumpy`, `pyglet` | vispy's EGL backend is a chronic headless failure (issues #1718, #2510); glumpy unmaintained; pyglet wants a window |
| pure-Python/torch port of the shader set | **does not exist** — searched, no maintained port |

---

## 3. ffmpeg paths (fallback / oracle)

- `xfade` in our static ffmpeg 7.0.2 has **57 named built-ins** plus `custom`. **[verified]**
  ~0.36 s per 49-frame 768×512 clip. Zero install. Good as a *held-out operator split* with
  exactly reproducible semantics.
- **`xfade_opencl` is NOT in our build** (no `--enable-opencl`). **[verified]** Don't plan on it.
- `transition=custom:expr=` is a per-pixel expression engine with `X Y W H P A B`, plane
  accessors `a0..a3`/`b0..b3` (so *warps*, not just masks), and 10 `st()/ld()` registers.
- Also present and useful: `blend` (~30 modes), `displace`, `remap`, `morpho`, `tblend`.
  **`remap` + numpy-generated flow maps is a warp-operator engine with no GPU code at all.**

### `scriptituk/xfade-easing` — worth knowing about

MIT, actively maintained (2026-03). Ships **pre-generated `expr=` strings for 123
transitions (64 of them gl-transitions ports) × 43 easing curves** that run on *stock*
ffmpeg with no rebuild, plus **15 original shaders not in gl-transitions**.

- **[verified]** a BOUNCE-OUT ⊗ GL_SWIRL expression runs correctly on our ffmpeg 7.0.2.
- Composition rule: `st(0,P);` + `<easing expr>` + `;` + `<eased transition expr>`.
- `-filter_complex_threads 1` is **mandatory** — slice threading corrupts the `st()/ld()`
  registers, and empirically the job also stalls without it. **[verified]**
- **But it is 128× slower than a built-in**: 46 s vs 0.36 s for 49 frames @768×512. **[verified]**

**Verdict:** use it as (i) a correctness oracle for our moderngl renders of the 64 shared
shaders, (ii) a source of the 43 easing curves (all closed-form one-liners), (iii) a
CPU-only fallback. Not as the engine.

---

## 4. Diversity multipliers beyond raw shader count

Define the operator as a tuple and make the tuple the label:

```
(family, primitive, param_vector, easing, direction, spatial_frequency,
 phase/seed, temporal_profile, composition_op)
```

- **Easing** — cheapest large multiplier and perceptually salient in video (bounce/elastic
  overshoot reads very differently from linear), unlike a 4th-decimal parameter change.
- **UV pre-transform** — applying the shader in a rotated/scaled/polar-remapped UV frame
  turns every axis-aligned wipe into an arbitrary-angle wipe. **A clean ×8–16 on the whole
  bank for ~20 lines of code.** Not yet implemented in exp_075.
- **Spatial flips + direction swap** — ×4 × ×2, free.
- **Composition** — sequential (op_A on [0,0.5], op_B on [0.5,1]), masked (op_A inside a
  mask, op_B outside), blend-mode, per-channel offset.
- **Quantize, don't sweep continuously**: 3–5 perceptually separated levels per parameter.
  Supported by Visual Atoms (arXiv:2303.01112), where generator parameterisation — not
  sample count — controlled transfer.

**Distinguishability audit before training:** render all K operators on one fixed endpoint
pair, embed with a frozen video encoder, cluster, and merge operators that are
indistinguishable across ≥3 endpoint pairs. Report *effective* K, not nominal K.

---

## 5. Content-dependent families — keep separate

Flow-morph (RAFT via `torchvision.models.optical_flow`, RIFE/MIT, Farnebäck) and
depth-reprojection operators are **content-conditioned by construction**: the same nominal
operator produces very different results on different endpoint pairs. They break the
operator ⊥ content property, so label them as a separate family and **do not count them
toward K** in any diversity sweep. Their value is realism — they look like what a human
editor produces, which shader compositing cannot.

**Cheap depth path worth trying before any inpainting network:** a transition does not need
disocclusion inpainting, because the incoming stream covers the holes. Estimate depth on
the last start frame and the first end frame (Depth Anything V2 — **Small is Apache-2.0,
Base/Large/Giant are CC-BY-NC**), reproject both along a parameterised camera path, and
cross-dissolve. `~8 path types × 4 amplitudes × 4 axes × 2 directions ≈ 250 operators`
as a single `grid_sample`. Prefer `sniklaus/3d-ken-burns` (CuPy, headless by construction)
over `vt-vl-lab/3d-photo-inpainting` (vispy → the same headless EGL failures as above).

---

## 6. Endpoint-pair sources

We use DAVIS (150 sequences — fine quality, far too small for endpoint diversity).

| Source | Scale | Notes |
|---|---|---|
| **OpenVid-1M** (arXiv:2407.02371) | 1M clips ≥512², CC-BY-4.0 | **Best default** — HF-hosted, already shot-segmented, aesthetic-filtered |
| Pexels / Pixabay API | 10⁴–10⁵ | Underrated: high production value, clean shots, direct MP4, no YouTube scraping |
| Panda-70M (arXiv:2402.19479) | 70.8M | Non-commercial; you must scrape YouTube yourself; contains flicker/blur, needs filtering |
| WebVid-10M | 10M | **Avoid** — watermarks would be learned as content |

Sampling policy that matters more than the source:

1. **Sample clips of length ≥ 9 + N_transition + 9** so both endpoint streams keep *moving*
   through the transition. Freezing the last start frame teaches the model that motion
   stops during a transition — the wrong prior. (exp_075 works around this with boomerang /
   flow extension because its input is only 9 frames.)
2. Pair endpoints **across** videos for most of the corpus; keep a stratum of same-video,
   distant-in-time pairs — those are the cuts a real editor makes.
3. Shot-boundary check each endpoint (`ffmpeg select='gt(scene,0.4)'`) so an endpoint is
   not itself a cut.
4. Store counterfactual blocks with a shared `pair_id` (fix a pair, render M = 8–64
   operators) so operator-contrastive batches are buildable.

---

## 7. Why "number of operators" is the scaling axis — the phase-transition claim

- **Raventós, Paul, Chen, Ganguli, NeurIPS 2023** — *Pretraining task diversity and the
  emergence of non-Bayesian in-context learning for regression* (arXiv:2306.15063).
  **The load-bearing citation.** Below a **task-diversity threshold** the transformer acts as
  a Bayesian estimator whose prior *is* the pretraining task distribution (= memorises the
  operator bank); above it, it solves **unseen** tasks. Threshold ≈ **2¹⁴–2¹⁵ tasks** for
  linear regression, with a sharp transition and a break in the scaling law.
- **Nguyen & Reddy 2024** (arXiv:2412.00104) — mechanistic account: memorising and
  generalising circuits are largely independent and their *relative learning rates* set
  where the threshold lands. Supports framing the threshold as (diversity × training
  budget), not model size.
- **Chan et al., NeurIPS 2022** (arXiv:2205.05055) — ICL emerges only under specific data
  properties (burstiness, many rare classes, skewed labels). Argues for designing the
  *sampling distribution* over operators, not just the count.
- **Power et al. 2022** (arXiv:2201.02177) — grokking, the canonical delayed
  memorisation→generalisation transition. Cite as the phenomenon; Raventós as the
  diversity-controlled version.
- Also: arXiv:2212.04458 (Kirsch et al.), arXiv:2410.05448 (task diversity shortens the ICL
  plateau).

**Procedural-pretraining precedent:** FractalDB / *Pre-training without Natural Images*
(arXiv:2101.08515 — categories *are* generator parameters, the direct structural analogue);
*Learning to See by Looking at Noise* (arXiv:2106.05963); *Procedural Image Programs for
Representation Learning* (arXiv:2211.16412 — ~21k programs, more programs ⇒ better
transfer, the closest existing "many procedural operators" scaling result);
*Scaling Backwards* (arXiv:2408.00677 — counterweight: more is not monotonically better,
measure it); AutoFlow (CVPR 2021 — 4 well-parameterised examples beat 22,872 FlyingChairs
examples, i.e. generator parameterisation ≫ raw sample count); Infinigen (arXiv:2306.09310);
Kubric (arXiv:2203.03570); domain randomisation (arXiv:1703.06907).

**The experiment this motivates:** sweep K ∈ {2⁴, 2⁶, 2⁸, 2¹⁰, 2¹², 2¹⁴} operators at
**fixed total clip count** (so compute is constant and only diversity varies — Raventós'
protocol), and evaluate on (a) held-out operators, (b) the 49 real human transitions.
A knee in (b) as a function of K is the headline result.
