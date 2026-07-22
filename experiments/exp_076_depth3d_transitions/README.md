# exp_076 — 3D-plausible procedural transitions

## Question

exp_075 showed the procedural operator engine works, but its gl-transitions bank produces
**2D overlay** effects — wipes, dissolves, shader warps pasted over the frame. The real
corpus is dominated by transitions that read as *geometrically plausible*: something moves
through a real three-dimensional scene.

Can we generate procedural transitions that are **3D-plausible** — where parallax,
occlusion, atmosphere and focus behave the way a real camera in a real scene would?

## Setup

**Format change from exp_075.** No more padding to 121 frames. Each sample is

```
start9 (9 frames, verbatim)  +  rendered middle (15 frames)  +  end9 (9 frames, verbatim)
= 33 frames   ((33-1) % 4 == 0, so it is still a legal LTX length)
```

The two given buckets are **copied through unmodified**, so conditioning fidelity is exact
by construction rather than by gate. This also kills the layer-extension problem exp_075
had to work around: with a short middle, the motion during the transition is supplied by
the *camera*, not by faking continued scene motion.

**The renderer.** For each endpoint we take the single frame facing the transition
(`start9[8]`, `end9[0]`), estimate per-pixel depth with **Depth Anything V2-Small**
(the Apache-2.0 checkpoint; Base/Large/Giant are CC-BY-NC), unproject it into a displaced
grid mesh, and re-render it from a moving virtual camera in OpenGL — same headless
moderngl + EGL path as exp_075, so it still runs on CPU nodes.

The realism comes from three things being *optically motivated* rather than decorative:

| Effect | Why it looks real |
|---|---|
| **Parallax** | Real per-pixel depth, so foreground and background separate at the correct relative rates under camera motion. This is the thing a 2D shader fundamentally cannot fake. |
| **Fog** | Beer–Lambert extinction `T = exp(-density·z)` along the view ray — the actual physical law, which is why a depth-ramped haze reads as distance. |
| **Defocus** | Circle of confusion from depth and focus distance, so a rack focus lands on the right plane instead of blurring uniformly. |
| **Handheld** | Band-limited 6-DoF jitter, tapered to zero at both ends. A perfectly rigid virtual camera reads as CGI. |
| **Motion blur** | 180° shutter, 3–4 sub-frames accumulated per output frame. The largest single realism gain available on a camera move, and it softens disocclusion smear for free. |
| **Dolly-zoom** | `f(t) = f₀·z_subj(t)/z_subj(0)` holds the subject's projected size constant while translating. Geometrically impossible under any 2D warp, which is exactly why it reads as unmistakably 3D. |
| **World-space dissolve** | The noise field is sampled at **unprojected scene positions**, not screen UVs, so the pattern sticks to surfaces, parallaxes with the camera and foreshortens on oblique geometry. A screen-space noise dissolve slides over the image and instantly reads as an overlay. Families: fBm, Worley, sweeping plane, expanding sphere. |

**One continuous trajectory.** Both layers ride the *same* camera path: layer A leaves
from rest, layer B arrives at rest, and B's excursion is offset by the full travel so the
two halves join into a single move. The eye reads one camera flying out of one scene and
into the next, not two clips being blended.

**Disocclusion.** A camera move reveals geometry the source frame never saw. The fragment
shader computes |det J| of the UV derivatives — texels-per-pixel, exactly 1 at the identity
camera — and fades out fragments where one texel is smeared over many pixels, so the other
layer shows through instead of a rubber sheet. What neither layer covers is filled by
normalised convolution (push-pull) at three scales. Measured hole fraction on a 0.5 dolly:
**0.4%**.

**Camera families** (`engine3d/cameras.py`): `dolly`, `truck`, `orbit`, `crane`, `roll`,
`shear`, `spiral`. Crossed with amplitude, sign, pivot depth, FOV (35–75°), depth range and
gamma, easing (8), blend mode (crossfade window / depth-ordered wipe), fog, rack focus and
handheld — order 10⁴ combinations, of which a few thousand are plausibly distinguishable.

## Validation

- **Bucket fidelity: MAE 0.000** — the buckets are copied, so this is exact by construction.
- **Identity-camera exactness: MAE 0.083** with alpha 255 everywhere. The projection
  inverts the unprojection regardless of depth, which is what allows a rendered frame at
  rest to reproduce its source. This is the property that makes the join possible at all.
- **Seam ratio** — the metric that actually matters here. Raw seam MAE is not comparable
  across clips: a near-static bucket has an internal frame-to-frame delta of ~1.7 while a
  fast one runs ~25, so the same absolute step is invisible in one and a jump cut in the
  other. We report `seam step / the bucket's own mean frame delta`; **≈1 means the join is
  as smooth as the content's natural motion.**
- **Parallax Index** (`engine3d/metrics.py`) — the certificate that a clip really is a 3D
  camera move. A translating camera moves near pixels more than far ones, by a ratio the
  depth distribution predicts. Every 2D operator in the exp_075 bank has PI ≈ 1 *by
  construction*, because it has no notion of depth at all. We report PI, the predicted PI,
  their ratio, and the Spearman ρ between `1/z` and flow magnitude. One OpenCV DIS flow
  call plus the cached depth — ~50 ms/clip, cheap enough to gate every generated sample.
  `compare_2d_3d.py` runs it across the exp_075 shader bank, this bank, and the real
  human-made transitions.
- Render cost: **~1–3 s per 33-frame clip** on CPU (3–4× with motion blur); depth is
  ~5 s/frame cold, cached to `outputs/analysis/exp_076_depth_cache` and reused.

### Three bugs the seam metric caught

The metric earned its keep immediately — the first run had a median seam ratio of 7.6 and
a max of 2364 (i.e. visible jump cuts), from three independent causes:

1. **Camera easing must have zero velocity at both endpoints.** The middle samples the
   open interval, so an easing still moving at u=1 leaves the camera off-rest on the last
   rendered frame. `in_out_cubic` scored 0.46 while `linear`/`accel`/`in_cubic` ran 20–160×
   worse. Camera paths are now restricted to `PATH_EASINGS`; the *blend* easing stays
   unrestricted, which also makes it an independent diversity axis.
2. **The blend must close inside the rendered range.** It now runs on `k/(n_middle-1)`,
   which hits exactly 0 on the first middle frame and exactly 1 on the last, instead of
   inheriting whatever fraction of the outgoing layer it had not yet retired.
3. **The dissolve threshold must sweep the full field range plus the edge band** — and the
   B-layer mask was inverted, so B was fully *absent* precisely when it should have been
   fully present. Cost a seam ratio of ~190 vs ~1.8 without a dissolve.

After all three: **worst-case seam ratio 0.14** across every camera family × dissolve
family × motion-blur setting.

## How to run

```bash
sbatch job_render.sbatch          # CPU-only, partition=secondary
python build_viewer.py outputs/videos/exp_076_depth3d_transitions/run_NNNN
cd $LAB/diffusion-research && python -m http.server 8077     # serve from the repo root
```

## Outputs

`outputs/videos/exp_076_depth3d_transitions/run_NNNN/` — `videos/`, `filmstrips/`,
`manifest.json` (full operator spec + seam ratios), `viewer.html`.

Blocks: **family** (one operator per camera family, matched amplitude/easing, same pair),
**counterfactual** (one pair × many operators), **sharedop** (one operator × several
pairs), **diverse** (random operators × cross-clip pairs).

## Known limits

- Depth is estimated on **one frame per endpoint** and held while the camera moves. With a
  15-frame middle this reads fine, but it does mean the scene itself is frozen mid-transition.
- 2.5D, not 3D: a single depth layer cannot represent what is *behind* an object, so large
  excursions expose disocclusions that push-pull fill only approximates.
- Monocular depth is relative, not metric, so translation amplitudes are in arbitrary scene
  units and their perceptual strength varies with the scene's depth distribution.
