# DATASET REGISTRY — training strata for the transition-transfer IC-LoRA

**Purpose.** One place that says, for every data source: *where the latest valid version lives,
how many samples it has, what its specification is, and whether it is cleared for training.*
When a stratum is rebuilt, update the row here — this file is the pointer of record.

Status vocabulary: **BUILT** (rendered, audited, cleared) · **PARTIAL** (rendered, not audited or
not complete) · **PLANNED** (spec frozen, nothing rendered) · **DEFERRED** (not this round, not killed)
· **EVAL-ONLY** (exists, must never be trained on).

Last verified against disk: **2026-07-27**.

---

## 1. The strata

| # | stratum | status | location | samples | spec |
|---|---|---|---|---|---|
| **S0** | real corpus (anchor) | **BUILT** | `eval_ladder/dataset/roots/ic_gen/` | **385** unique, 26 donor classes | the real VFX corpus; never replaced, present in every mix |
| **S1** | spec counterfactuals | **DEFERRED** | — | ~660 planned (11 manners × ~30 pairs) | exact same-op × diff-content in the real visual domain |
| **S2** | 2D shader procedural | **BUILT** | `outputs/videos/ctt_v2_s2/full/` | **7,990** clips · **799** ops · 56 shaders | 121f · 480×640 · 24fps · 10 clips/op |
| **S2r** | S2 retired/blacklisted | **PARTIAL** — awaiting blind audit | `outputs/videos/ctt_v2_s2/full/retired_blacklisted/` | **420** clips = 42 complete op blocks, 6 shaders | same spec; candidates for reinstatement |
| **S3** | 3D depth-parallax | **PLANNED** — approved design **not built** | see §3 for the 203 pre-fix clips | **4,000** target (400 ops × 10 pairs) | 121f · 9f anchors · exp_080 + 3 structural fixes |
| **S4** | refVFX I2V-LoRA | **PARTIAL** — raw on disk, not reshaped | `data/raw/refvfx/data/I2V_LoRA/` (12 GB, 1 shard) | ~2,000 target of 6,995 · 48 effect types | webdataset tars; external, counts toward the ≤50% cap |
| **S5** | refVFX code-based | **DEFERRED** | `data/raw/refvfx/data/code_based_edits/` (349 GB, 16 shards) | ~2,000 target of 136,800 | inputs are 8 fps — unusable as playing streams |

**Mix directive for the first retrain (owner, 2026-07-25):** **S0 + S2 + S3 + S4**. S1 and S5 deferred, not killed.
**Advisor-ruled epoch sampling mix (2026-07-27):** **50% real / 25% S3 / 25% S2**, by sample.

### S2 counting — read this before quoting a number
`ops_shard*.jsonl` has **809 rows**, but 10 are `dropped:true / n_slots:0` (they exhausted the
pair-swap budget and shipped nothing). The dataset is **799 operators × 10 clips = 7,990**.
Quote 799, not 809.
**Recomputed overdraw: 1.2506×** (9,992 renders → 7,990 accepted), from the append-only ops log.
Two metadata files are **stale/non-authoritative** and must not be used:
`S2_ACCEPTANCE.json` reports the pre-blacklist state (7,550 / 755 / 62 shaders), and
`summary_shard*.json` were overwritten by the backfill pass (they describe only 86 backfill ops,
giving a misleading 1.93 overdraw). `accept_s2.py` warns about the latter trap in a comment.

### Retired S2 shaders (the 420)
Each of the 42 op blocks is complete (10 clips), so reinstatement requires **no regeneration**.
**Zero overlap with `HOLDOUT_S2`** — verified. The blacklist was **purely economic** — a 0.50
reject-rate bar on *renders*, never a judgement on clip quality — and it splits into two clear tiers:

| shader | render reject rate | complete op blocks | tier |
|---|---|---|---|
| `PuzzleRight` | 52.6% | 10 | marginal — just over the bar |
| `splitSlideOutHorizontal` | 53.9% | 10 | marginal |
| `StripDatamoshGlitch` | 54.5% | 9 | marginal |
| `SimpleZoomOut` | 55.4% | 9 | marginal |
| `SimpleZoom` | 92.8% | 2 | **pathological** (11 of 13 ops dropped) |
| `swap` | 93.4% | 2 | **pathological** (11 of 13 ops dropped) |

Reinstating all 42 blocks → 8,410 clips / 841 ops / 62 shaders at zero render cost (+3,780 ref-target
pairs). Reinstating only the four marginal shaders → 38 blocks / 380 clips.
Reinstatement bar: blind-rate all 420; a clip enters training iff rated GOOD; a shader re-enters
the generator iff **≥60%** of its retired clips rate GOOD. Cap 2,000 new clips at 14 ops/shader
(the measured S2 density — *not* the 5 the advisor initially assumed).

### S2 audit caveat
The n=64 blind two-rater audit **predates the blacklist**. Six of its 64 sampled stems are now in
`retired_blacklisted/`, including `s2_0229_c06` (PuzzleRight) — one of the two adjudicated BADs.
Against the roster that actually shipped it reads **1 BAD / 58**, which is better on its face but is
**no longer a clean pre-registered 64-sample audit** of this dataset. The remaining in-roster BAD is
`s2_0270_c08` (Slides). Treat S2 as needing a fresh n=64 audit before any claim rests on it.

---

## 2. Content (endpoint) pool

| item | value |
|---|---|
| authority | `data/processed/ctt_v2_strata/CONTENT_POOL.json` |
| training endpoints | **291** |
| reserved eval-only | **20** (appear in no training grid) |
| total ids | 311 · letterboxed exclusions 20 · trim log 12 |
| spec | 480×640 portrait · 121 frames · 24 fps · single subject (bbox area ≥0.15, score ≥0.7) |
| v1 bank | `data/processed/synth_endpoints/` — 331 clips, `bank_tightened.json` = 227 |
| sources | OpenVid · vc-bench · DAVIS |
| **known skew** | **~85% label `person`** — logged limitation |

**Hard rule:** corpus clips are **never** used as procedural contents — the class effect unfolds
inside them, so the manner would leak into "content".

**Expansion ruling (2026-07-27):** 291 is sufficient for the sizes above (~84k possible ordered
pairs). **HumanVid rejected for this round** — 100% human-centric, deepens the skew, and the
Apache-2.0 HF drop is entirely UE-rendered/generated synthetic. Any expansion must be
**anti-skew**: +80–120 non-person, training-safe clips, target person share ≤60%.

---

## 3. S3 — what exists today (all of it pre-fix)

The approved S3 design has **zero clips rendered**. Three earlier generations exist:

| group | location | clips | mechanism | measured |
|---|---|---|---|---|
| **A** | `outputs/videos/exp_080_depth3d_realstream_121/run_0001` | **31** | **approved**: 121f, both streams playing, per-frame stabilised depth | join median **0.94** / max 1.86 (bar 2.0) · parallax **3.31** · 11 s/clip |
| **B** | `outputs/videos/ctt_v2_s3` | **63** | **same engine as A, byte-identical, same amplitude 1.6** | **62% defective** (39 BAD / 24 GOOD) |
| **C** | `outputs/videos/exp_083_d3_pilot/run_0001` | **109** | superseded: 25/33/41/49f, static depth, frozen endpoints | **47% BAD** (14/30 sampled; 79 unlabelled) · seam monotone in length (25f 1.02 → 49f 0.02) |

> **⚠ A's clean numbers are a CONTENT artifact, not a mechanism win.** A and B are the *same engine
> at the same amplitude on the same 121f contract*. A covers only **4 content pairs / 8 source clips**,
> all dark stock-VFX footage; B covers **63 pairs / 91 endpoints** of real DAVIS/VCBench/OpenVid and
> came back 62% defective. **A does not establish that the mechanism survives real content — B is the
> evidence on that question, and it is negative.** The defect is present in A too, just rare: its
> highest-parallax clip (`shear_crossfade_sphere`, PI 8.63) carries a hard black polygon mid-ramp.
> A calibrated novel-hard-black probe (`scripts/s3_novel_black_probe.py`, measured against each clip's
> own byte-identical pure phases) flags 49% of B's BAD and 4% of B's GOOD, and flags **2 of 31 (6%)**
> in A — i.e. A sits with B's GOOD clips. The probe is one-sided (it sees black voids, not the
> smear/melt mode) so it is a **lower bound**.
> **Operational consequence: the 200-clip gate pilot must run on B-difficulty real content**, never on
> A's easy content, or it will produce another false positive.

**The shared defect.** `composite()` computes total alpha `den` and hands it to `_fill_holes`,
a push-pull inpainter with ~40px reach. Where the camera looks past the mesh edge `den` collapses:
small holes inpaint fine, medium smear/melt, large stay black. The world-space dissolve compounds
it by thresholding layers A and B against **independent** fields, so both can be absent at the same
pixel. Median hole radius by dissolve family: `none` 59px · `fbm` 87 · `worley` 144 · `subject` 150
· `plane` 154 · `sphere` 159.

**The three fixes that define the approved build** (prevention, not post-hoc detection):
1. **Frustum stays inside both meshes** every frame — analytic per-frame test; makes edge holes impossible.
2. **One shared arrival-time field** for the dissolve: `mask_A = step(progress, T)`, `mask_B = 1 − mask_A`
   — mutually exclusive and collectively exhaustive, so the dissolve cannot collapse `den`.
3. **Pre-inpaint invariant:** per-frame max hole radius (distance transform on `den < ε`, *before*
   `_fill_holes`) **≤ 32px**; violation → resample amplitude downward. Amplitude sampled in [0.5, 2.0];
   the invariant is the ceiling, not a fixed amplitude cap.

Why pre-inpaint: post-inpaint, badness is a semantic property (four detectors failed out-of-sample);
**pre**-inpaint it is exactly measurable, because holes are precisely where `den < ε`. The earlier
"defect is unmeasurable" conclusion measured after the evidence had been painted over.

**Gate before the full build:** 200-clip pilot (~40 GPU-min). Must show hole-radius p99 ≤40px,
invariant-triggered resamples ≤20%, and a blind two-rater n=64 audit at **≤5% BAD**.
5–10% → tighten to 24px and re-pilot once. >10% → halt.

---

## 4. Eval stock and holdouts — never train on these

| item | location | why |
|---|---|---|
| **D2 build** | `outputs/videos/exp_077_synth_stratum_d2/d2full/` — 6,144 clips | **EVAL-ONLY.** Spans all 72 shaders including the 10 held out, so it supplies the unseen-op × seen-content cell for free. Training on it voids the split. |
| S2 holdout | `experiments/exp_081_s2_stratum/HOLDOUT_S2.json` | 10 shaders, each keeping a same-genre cousin in training (tests unseen-operator generalisation, not removed capability) + ffmpeg-xfade as a cross-engine probe |
| S3 holdout | `experiments/exp_082_s3_stratum/HOLDOUT_S3.json` | the `spiral` family (= dolly ∘ orbit, both primitives trained → tests *compositional* generalisation) + 30 exact ops |
| reserved endpoints | `CONTENT_POOL.json` | 20 clips in no training grid |

Together these give the factorised eval grid free: {seen, unseen operator} × {seen, unseen content}.

---

## 5. Contracts a dataloader must honour

**No reference clip on disk (S2, and S3 by the same design).** An operator's 10 clips **are** its
reference pool. For a target, draw the reference from a *different* clip of the same `op_id`, and
**resample every epoch** — 90 ordered combos per op. That recurrence, the same target appearing
against different references, is the signal that the reference's *content* carries no information.
Two invariants hold by construction and are worth asserting as corruption checks: an op's 10 pairs
use **20 distinct endpoint clips**, and every clip of an op shares (shader, uniforms, easing,
onset/release, flip, swap) exactly — **timing is part of operator identity**.

**Endpoint fidelity.** Pure phases are byte-identical to source by construction (measured max abs
diff **0**). No frame fabrication — no boomerang, hold, or flow extension.

**Baselines.** A comparator arm is *prompt only* or *prompt + endpoint conditioning*. **Never
reference conditioning** — a no-adapter model given an in-context reference is a copier, not a
baseline. See `notes/models/ltx2/conditioning.md` §5.4.

**Unique-content honesty.** Report unique-content shares, not just sampled shares. The D2 build was
50/50 by sample but **89% synthetic by unique content** (385 real clips replicated 8×).

---

## 6. Scoreboard and current bars

Frozen eval v4, four cells. Incumbent generalist: **72.9 / 72.8 / 88.7 / 90.8**
(G-unseen-cross, G-zs-cross, G-unseen-same, G-zs-same).

Pre-registered bars for the next generalist:
- pooled cross **≥ +5.0pp**
- donors positive **≥ 15/23**
- **same-content non-inferiority:** both same cells ≥ incumbent **−1.0pp**
- **coupled-donor recovery:** median Δ over `saint_glow`, `shadow`, `display_transition`,
  `wireframe`, `polygon` **≥ 0** — the specific test of whether S3 earned its existence
- guards: `near_copy` and `ref_dominated` at or below incumbent levels

Reference result — the D2-trained generalist **missed** its bar: pooled cross +2.0pp, donors 13/23,
same-content −2.0 / −4.6, with losses concentrated in the coupled donors. Guards were clean
(`near_copy` 0.00%, `ref_dominated` 0.7% vs incumbent 2.2%), so the small gain was real transfer,
not copying. That run stands as the **no-coupling control arm** for the next one.

---

## 7. Viewers

| dataset | URL (server rooted at repo root, port 8017) |
|---|---|
| S2 (7,990) | `/outputs/viewers/s2_dataset/` |
| S3 (203, all three mechanisms) | `/outputs/viewers/s3_dataset/` |
| D2 eval stock (252 shown) | `/outputs/viewers/d2_dataset/` |
| exp_080 native | `/outputs/videos/exp_080_depth3d_realstream_121/run_0001/viewer.html` |
| exp_083 native | `/outputs/videos/exp_083_d3_pilot/run_0001/viewer.html` |
| HumanVid sample | `/outputs/viewers/humanvid_sample/` |

---

## 7b. Luma-matte operator family — revived, with conditions

Settled 2026-07-27 by a 2×2 (`experiments/exp_084_luma_matte_viewer/`, viewer
`/outputs/viewers/luma_matte/`). Blind-graded, 16 clips/arm, anonymised shuffled sheets, rubric
pre-registered:

| | hard `step()` | feathered |
|---|---|---|
| shipped maps | **88% BAD** | 56% BAD |
| new arrival-time maps | **88% BAD** | **31% BAD** |

**The compositor is the gate, the maps are second.** Better maps through the hard threshold buy
*nothing* (14/16 BAD either way, Fisher p=1.00). Compositor alone 88→56% (p=0.11); both 88→31%
(p=0.003). The 56% residual is not uniform: aperiodic shipped maps (`fbm`, `radial`, `linear`)
go 6/8 → **1/8** BAD, while geometric ones (`stripes`, `checker`, `spiral`, `voronoi`) stay
**8/8 → 8/8** (p=0.0014) — a feathered checkerboard is still a checkerboard. **Drop the four
geometric maps from this family; keep the three aperiodic ones.** New maps are not separable from
rescued-aperiodic (p=0.62) — their value is variety and content-awareness, not a higher ceiling.

**Two engine bugs found, both real:**
1. `step(progress, m)` returns 1 when `m == progress`, so **every** hard-compositor clip leaks 5–6
   stale pixels of frame A into the final conditioning block. Any matte normalised to [0,1] has
   max-valued pixels. **exp_075's MAE gate cannot see this**; `luma_soft` measures exactly 0.
2. The `luma` sampler is read with a bare `texture2D(luma, uv)` while `getFromColor` flips y, so
   **the matte is vertically flipped relative to the image**. Harmless for isotropic maps,
   load-bearing for anything content-aware (`probe_orientation.py`).

Caveats: single grader who also rendered the clips (mitigated by anonymised sheets + pre-registered
rubric, not eliminated), n=16/arm, graded from 3 stills rather than motion, and glow was not
separated from feather. Treat the ordering as the finding and the percentages as soft.

Assets: only **CC0** David Revoy Krita brush alphas (`data/raw/cc0_brush_alphas/`, SHA-256 in
`PROVENANCE.json`). Nothing from Pixabay/Pexels/ProductionCrate/ActionVFX/Shadertoy.

---

## 8. Open items

1. **S3 approved build** — not started. Needs the three fixes, then the 200-clip gate pilot.
2. **420 retired clips** — blind audit pending; also pending: which gate rejected them, and whether
   the coupled fraction among the 6 shaders is elevated (would confirm the instrument, not the
   shaders, was defective).
3. **S4 reshape** — refVFX raw tars on disk, not yet turned into training pairs.
4. **Endpoint anti-skew expansion** — optional, non-blocking, +80–120 non-person clips.
5. **Real-corpus growth** — the advisor's view is that this is the highest-leverage action available
   (the same-content regression says real data is the binding constraint), but it is out of scope
   for this round.
