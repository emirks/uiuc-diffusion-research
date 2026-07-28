# exp_083 — D3 / S3 pilot: depth-parallax 3D procedural transitions

## Question

The D2 procedural stratum (exp_077) — 3,072 gl-transitions GLSL shaders over real
endpoint clips — trained a generalist and **missed its win bar** (pooled cross +2.0pp
against a ≥+5pp bar, 13/23 donors positive). The losses were not spread evenly. They
concentrated on the **content-coupled** donors:

| donor | Δ |
|---|---|
| saint_glow | −12.3 |
| shadow | −12.1 |
| display_transition | −10.3 |
| wireframe | −8.5 |
| polygon | −7.0 |
| … | |
| flying_cam | +14.0 |
| luminous_gaze | +24.1 |

The diagnosis: **D2's design assumed operator ⟂ content.** A screen-space shader applies
to any pair, so the stratum only ever taught manners that are independent of what is in
the frame — and those are exactly the manners it won on. Real corpus transitions where
shadow or smoke *attaches to a foreground object and travels with it* cannot be expressed
by a screen-space wipe at all, and the stratum actively hurt there.

exp_076's depth renderer can express them: its dissolve field is sampled at **unprojected
scene positions**, so the pattern sticks to surfaces, parallaxes and foreshortens.

**So: is a depth-parallax 3D stratum worth building?** This is a ~110-clip pilot for the
owner to look at and decide. It is not a stratum.

## Setup

**Vehicle.** exp_076's `engine3d/` (via exp_082's copy, which is a superset), reused rather
than rewritten. Three additive changes, all in `engine3d/ops3d.py`:

1. `subject_anchor(view_z, rgb)` — the foreground subject's pixel, as the
   saliency-weighted centroid of the nearest depth quartile. No external metadata, so it
   works on any endpoint.
2. `dissolve_field(..., center=)` and two new families, `subject` (an expanding shell
   centred on the subject's own world position) and `subject_fbm` (that shell modulated by
   fBm, so the erosion front reads as smoke rather than as a geometric surface).
3. `render_transition(..., coverage_out=)` — the per-frame disocclusion audit exp_082 added
   to the stream driver, ported to the frozen-endpoint driver. Free: `den` already exists
   inside `composite`.

**Endpoints — our bank, not the old corpus.** Previous exp_076 runs used
`exp_062/dataset/cond`, whose honest critique was dark/weak B endpoints. This pilot draws
from `data/processed/synth_endpoints/` — 331 standardised 480×640×121 clips, filtered by
`bank_tightened.json` to the **227** that pass subject presence (bbox area ≥ 0.15, detector
score ≥ 0.7, allowed labels). Anchors are **real consecutive frames sliced out of those
real clips**:

```
A-role  start9 = frames[112:121]   transition-facing frame = the last  (120)
B-role  end9   = frames[  0:  9]   transition-facing frame = the first (  0)
```

Nothing is generated, extended, held, boomeranged or optical-flowed. Both anchors are
copied through **verbatim**, so conditioning fidelity is exact by construction and is
verified numerically per clip rather than gated.

**Varying lengths — no padding.** `n_middle ∈ {7, 15, 23, 31}` with 9+9 anchors:

| n_middle | total | F = 8k+1 |
|---|---|---|
| 7 | 25 | 8·3+1 |
| 15 | 33 | 8·4+1 |
| 23 | 41 | 8·5+1 |
| 31 | 49 | 8·6+1 |

Asserted per clip at render time, not just in the config.

**Depth.** Two frames per tuple, read out of exp_082's cached per-frame stabilised stacks
(`outputs/analysis/ctt_v2_s3_depth_cache`, one `.npy` per bank clip) — the same
Depth-Anything-V2-Small (Apache-2.0) maps, already computed, so the pilot spends no time on
depth. A clip missing from the cache falls back to computing and caching its own map.

**Operator recipes** (`RECIPES` in `run.py`), labelled by what the effect is coupled to:

| coupling | recipes |
|---|---|
| `none` — camera only | `bare_move`, `dolly_zoom` |
| `depth` — tied to the depth field | `depth_wipe`, `rack_defocus`, `atmos_travel` |
| `world` — field at scene positions | `world_fbm`, `world_worley`, `sweep_plane`, `shell_sphere` |
| `subject` — field centred on the subject | `subject_shell`, `subject_smoke` |

**Blocks.** `family` (7 camera paths, bare move, one pair) · `axisop{0,1}`
(**same content × different operator**: all 11 recipes on each of two pairs) ·
`axiscontent{0..3}` (**same operator × different content**: one operator instance across 6
pairs) · `length` (4 recipes × the 4 legal totals, one pair) · `amp` (2 paths × 5
amplitudes, deliberately on `bare_move` so the disocclusion probe measures geometry and not
the dissolve's own erosion) · `diverse` (30 random recipe × pair × length).

## How to run

```bash
sbatch job_render.sbatch                      # CPU-only, HCESC-L40S-normal, no GPU
python build_viewer.py  outputs/videos/exp_083_d3_pilot/run_NNNN
python summarize.py     outputs/videos/exp_083_d3_pilot/run_NNNN
```

## Outputs

`outputs/videos/exp_083_d3_pilot/run_NNNN/` — `videos/`, `filmstrips/`, `manifest.json`,
`viewer.html`, `run.log`, `PILOT_RESULT.json`.

Per clip the manifest records the operator spec, coupling class, camera path, `n_frames`,
both seam ratios, the parallax index, the in-array endpoint max-abs-diff, the codec
round-trip max-abs-diff, and the disocclusion audit (`uncovered_*`, `hole_radius_max`).
