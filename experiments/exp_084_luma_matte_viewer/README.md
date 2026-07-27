# exp_084 — luma-matte transitions: was it the map, or was it `step()`?

## Question

The aux-map operator family — a static greyscale matte in the `luma` sampler,
with `progress` sweeping a threshold across it — was judged fake-looking and
shipped at **0%**. Two things were confounded in that judgement and were never
varied independently:

1. **the maps.** `exp_075/engine/maps.py` emits seven *stationary, isotropic*
   fields (`fbm, radial, linear, stripes, checker, spiral, voronoi`). Ink, paint
   and liquid are neither stationary nor isotropic: they have a source, they
   propagate, they branch.
2. **the compositor.** `gl-transitions/transitions/luma.glsl` is one line —

   ```glsl
   return mix(getToColor(uv), getFromColor(uv), step(progress, texture2D(luma, uv).r));
   ```

   `step()`. No feather, no rim colour, no glow. Ink and paint read as real
   almost entirely *because of the advancing boundary*, and this compositor has
   no boundary at all.

So: **was the family killed by bad maps, or by a broken compositor?**

## Setup

A 2×2 over real playing footage, 3 content pairs, 19 mattes, 114 clips.

|                    | hard `step()`      | feathered `luma_soft` |
| ------------------ | ------------------ | --------------------- |
| shipped maps (×7)  | `A1_current`       | `A2_soft_same_map`    |
| new maps (×12)     | `A4_new_map_hard`  | `A3_new_map_soft`     |

A1/A2 share matte, matte seed, footage and easing, so the only difference is
the compositor. A4/A3 likewise. `luma.glsl` is used byte-for-byte unmodified;
the new compositor is a **separate** file, `shaders/luma_soft.glsl`.

**Footage.** Both transition layers are real frames from the curated 227-clip
endpoint bank (`data/processed/synth_endpoints/bank_tightened.json`): the
outgoing shot plays its last 61 frames, the incoming shot its first 61. Nothing
is held, boomeranged or extrapolated, and the 6-frame anchor blocks are
verbatim (asserted).

**`luma_soft.glsl`** keeps the same static-matte plumbing and changes only the
compositing: `smoothstep` feather with a per-operator width, a rim colour
painted into the advancing band, and an additive glow lobe biased ahead of the
front. The threshold is remapped to `p = progress·(1+2f) − f` and rim/glow are
gated by an envelope, so the endpoint identities are exact. Widths, rim colours
and glow amounts are per-operator (`mattes/styles.py`, five presets: `ink`,
`paint`, `leak`, `frost`, `burn`).

**New mattes** (`mattes/newmaps.py`), all emitted as arrival-time fields so they
drop into the existing static-aux plumbing unchanged:

* **eikonal / fast marching** (`eikonal_ink, _burst, _streak, _drip`) — first-order
  FMM (heapq; no `scikit-fmm` on this cluster) from seed points through a
  *ridged* multifractal speed field. The ridging matters: plain fbm has fat fast
  regions and gives round blobs, ridged fbm has thin fast channels, so the front
  runs along them and branches.
* **invasion percolation** (`invasion_ink, _frost, _finger`) — the standard model
  for ink wicking into paper / viscous fingering / frost. Invasion *order* is
  natively an arrival time. Grey opening removes the late sites stranded inside
  the cluster (the morphological form of trapping), which is what separates ink
  from grain.
* **brush stamping** (`brush_wipe, _scribble, _splat`) — CC0 Krita brush alphas
  stamped along parametric paths with pressure and jitter, stamp index written
  into the arrival time. This is how commercial paint-wipe packs are made.
* **content-aware boundary draw** (`edge_draw, edge_draw_fine`) — the same
  stamping, but the path follows a Canny edge extracted from the *incoming*
  frame, then bleeds outward. No stock matte can do this, because a stock matte
  never sees the footage.

## How to run

```bash
sbatch job_render.sbatch                     # 114 clips, ~2.5 min, CPU-only (HCESC-L40S)
python build_viewer.py                       # -> outputs/viewers/luma_matte/index.html
python sample_audit.py sheets                # anonymised blind contact sheets + AUDIT_KEY
#   ... grade into outputs/.../run_0001/GRADES.json ...
python sample_audit.py score                 # joins grades with the key
python probe_orientation.py                  # one-off: aux-sampler orientation
```

Not `secondary` — another session sweeps that partition. `mattes/glctx.py`
walks the EGL device list, which an HCESC node without `--gres` needs.

## Outputs

* `outputs/videos/exp_084_luma_matte_viewer/run_NNNN/` — videos, filmstrips,
  matte PNGs, `manifest.json`, audit sheets/key/grades/result
* `outputs/viewers/luma_matte/index.html` — the paired A/B viewer
* `AUDIT/` — the blind grades and the scored result, committed

## Result

Blind audit: 16 clips per arm, sampled at a fixed seed *before* anything was
looked at, cut into anonymous 3-frame strips at progress 0.3/0.5/0.7, shuffled
across arms, graded with no arm/map/compositor label, key joined afterwards.

|                    | hard `step()` | feathered   |
| ------------------ | ------------- | ----------- |
| shipped maps       | **88% BAD**   | 56% BAD     |
| new arrival maps   | **88% BAD**   | **31% BAD** |

* Better maps through the old compositor buy **nothing**: 14/16 BAD either way
  (Fisher p = 1.00). `step()` gates everything.
* The residual 56% is not spread evenly. Split the shipped maps:
  **aperiodic** (`fbm, radial, linear`) 6/8 → **1/8** BAD; **geometric**
  (`stripes, checker, spiral, voronoi`) 8/8 → **8/8** BAD (p = 0.0014). The
  feather rescues the aperiodic maps outright and cannot touch the geometric
  ones — a feathered checkerboard is still a checkerboard.
* New maps + fixed compositor is the best cell (p = 0.003 vs the shipped
  baseline) but is **not** separable from rescued-aperiodic-shipped (p = 0.62).
  Their value is variety and content-awareness, not a higher ceiling.

**Answer: it was `step()` first, and the maps second — and only three of the
seven shipped maps were ever salvageable.**

### Side findings

* `step(progress, m)` returns 1 when `m == progress`, so at `progress = 1` every
  pixel sitting at the matte's *maximum* keeps showing the outgoing shot. Any
  matte normalised to [0,1] has such pixels: every hard-compositor clip here
  leaks 5–6 stale pixels into the final conditioning block. `luma_soft.glsl` is
  exactly 0.
* The `luma` sampler is read with a bare `texture2D(luma, uv)` while
  `getFromColor` flips y, so **the matte is vertically flipped relative to the
  image** (`probe_orientation.py`). Irrelevant for the seven isotropic shipped
  maps; load-bearing for any content-aware matte.

### Caveats

Single grader, n = 16 per arm, and that grader also rendered the clips —
mitigated by the anonymised shuffled sheets and the pre-registered rubric, not
eliminated. Grading was from three stills per clip, not from motion. Three
content pairs only. Part of the A2 gain may come from the glow rather than the
feather; the two were not separated. **Treat the ordering as the finding and the
exact percentages as soft.**

## Licences

CC0 only. Brush alphas: David Revoy *Krita brushes 2025-01*, CC-0 / public
domain (davidrevoy.com/article1060). No commercial or ML-restricted matte pack
was downloaded — Pixabay, Pexels, ProductionCrate and ActionVFX all prohibit ML
training and Shadertoy defaults to CC BY-NC-SA. See `PROVENANCE.json`.
