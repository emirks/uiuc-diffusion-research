# exp_080 — 3D camera transitions at 121 frames over real playing streams

## Question

exp_076 proved the 3D operator family (real-depth parallax, dolly-zoom, world-space
dissolves) but only at 33 frames with a FROZEN world: one depth mesh per endpoint frame,
all motion from the camera. Can the family be promoted to the full 121-frame format on
the D2 contract — both source clips PLAYING throughout, pure phases verbatim — so it can
join the ctt_v2 training mix (stratum S3) format-identical with every other stratum?

## Setup

Fork of exp_076. Differences:

1. **Inputs are full 121-frame clips** (`transitions_std121`, train classes only),
   paired across classes. No more 9-frame buckets, no layer extension of any kind.
2. **D2 timing manifold**: onset = 8 + u·0.2·104, release = 112 − u·0.2·104 ⇒
   transition duration ≥ ~62 frames. `out[0..onset] = A verbatim`,
   `out[release..120] = B verbatim` — pure-phase identity is byte-exact BY
   CONSTRUCTION and asserted per clip.
3. **Per-frame depth, temporally stabilised** (`depth.disparity_stack`): percentile
   normalisation + least-squares scale/shift alignment to the running estimate + EMA.
   Full-clip stacks cached as float16 (`outputs/analysis/exp_080_depth_cache`).
   A `flicker` metric is logged per stack for the future gate calibration.
4. **`ops3d.render_transition_stream`**: every rendered frame unprojects the CURRENT
   frame of both live streams through that frame's depth — the world never freezes.
   Dissolve fields are re-evaluated at each frame's world positions so the pattern
   sticks to moving surfaces. Camera easings stay restricted to zero-endpoint-velocity
   (`PATH_EASINGS`) — the D2 audit independently confirmed boundary-velocity easings
   manufacture C1-discontinuity seams.
5. **`join_ratio`** replaces bucket seam error: frame delta across each join relative
   to the local internal motion (1.0 = the join moves like the content around it).
6. `amplitude_scale` config knob (default 1.6): the ramp is ~5× longer than exp_076's
   15 frames, so the same excursion runs slower; a longer move can travel further.
   This is the main audit knob the sample run exists to calibrate.

## How to run

    sbatch experiments/exp_080_depth3d_realstream_121/job_render.sbatch

One L40S (depth model per-frame; EGL binds the NVIDIA GL on that node). The job
renders the sample plan, then builds `viewer.html` inside the run dir.

## Outputs

`outputs/videos/exp_080_depth3d_realstream_121/run_NNNN/`:
`videos/*.mp4` (121f, 24 fps, 480×640) · `filmstrips/*.jpg` · `manifest.json`
(operator params, timing window, join ratios, parallax index, per-stack depth
flicker in run.log) · `viewer.html` (grid, autoplay-on-scroll, filmstrip toggle).

Sample plan: camera-family showcase (7) · optical-effect showcase (6) ·
counterfactual block (1 pair × 8 ops) · shared-operator block (2 ops × 3 pairs) ·
diversity (4). ~31 clips.

## Expected outcome

If join ratios sit near the content's own motion level (~1–2), depth flicker is
visually invisible, and the owner's visual read is positive, S3 of the ctt_v2 data
plan is promoted to this 121-frame format (300 ops × 6 contents draft) and the
mixed-length trainer question disappears.
