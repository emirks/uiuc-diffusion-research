# exp_077 — synthetic → training pipeline smoke + operator-bank audit

## Question

Before mass-rendering procedural transitions (exp_075's engine) for IC-LoRA pretraining,
two things must be de-risked:

- **0d (pipeline smoke):** does a synthetic clip flow end-to-end
  `render → VAE-encode → cond-clean + text → assemble ic_gen root → 50-step LoRA train`
  with **no silent sample drops** and a **finite, descending** loss? (proves the plumbing
  before we spend compute at scale)
- **0e (operator-bank audit):** how many *distinguishable* operators does the bank really
  emit (**effective-K** vs nominal), and are the **medium-bearing aux-map families**
  (particle/smoke/ink-like) being disproportionately rejected by the endpoint gate? (they
  are the most valuable operators and must not be silently filtered out)

Nothing here is a full run — a 20-tuple smoke + two audits.

## Setup

- **Engine.** exp_075's procedural engine, imported through the `engine/` **symlink**
  (exp_075 is never modified). The only new render logic is a **timing-aware progress ramp**
  (`render_tuples.timed_progress_ramp`) so onset/duration/curve inside the `num_frames`
  window are a genuine, continuously-sampled degree of freedom — exp_075's ramp fixes the
  transition to the whole `[8, 112]` window.
- **Endpoints.** `experiments/exp_062_ladder_r2r3_specialists/dataset/cond/*_{start9,end9}.mp4`
  (44 pairs, 480×640).
- **Length parameter.** `inference.num_frames` (default **121**) is plumbed through the whole
  render path (ramp, timing sampling, VAE bucket `WxHxF`, mask frame count) so future rounds
  can vary length without re-plumbing.
- **Training.** frozen ic_gen (generalist IC-LoRA) recipe — rank32/α32 attn+FFN, lr 2e-4,
  flexible strategy, reference-latents + mask conditioning, cond_clean — on the
  `cond-bleed-fix` trainer. Validation is **disabled for the smoke** (nothing emerges in 50
  steps; a real run must restore the ID+OOD+control triad — `lora-train` skill).

### A tuple

`(reference clip, target clip)` sharing the **same operator** — shader, continuously-sampled
uniforms, easing, flip/swap, aux-map, **and** the transition timing (onset/duration/curve) —
rendered on **two different endpoint pairs**. This is the operator-fixed / content-varying
counterfactual axis the IC-LoRA generalist trains on: target latents + reference latents +
neutral text + mask, exactly like `assemble_roots.py::assemble_generalist`.

## How to run

```bash
cd $LAB/diffusion-research/experiments/exp_077_synth_stratum

# TASK 0d — pipeline smoke (CPU render -> GPU encode/assemble -> GPU 50-step train)
sbatch job_render_tuples.sbatch                        # CPU: 20 tuples + metadata + determinism
RUN=<outputs/videos/exp_077_synth_stratum/run_NNNN>    # the render run_dir
sbatch --export=ALL,RUN=$RUN job_build.sbatch          # GPU: encode + cond-clean + text + assemble
python make_train_config.py --run $RUN                 # emit configs/synth_smoke.yaml
sbatch --export=ALL,RUN=$RUN job_train_smoke.sbatch    # GPU: 50-step IC-LoRA smoke

# TASK 0e — operator-bank audit
sbatch job_bank_audit.sbatch                           # CPU: broad sample + per-family gate stats
RUN=<outputs/videos/exp_077_synth_stratum_audit/run_NNNN>
sbatch --export=ALL,RUN=$RUN job_embed.sbatch          # GPU: DINOv2 embed + cluster -> effective-K
```

GPU jobs run on the cluster-wide `secondary` partition (`--gres=gpu:H100:1`) because the HCESC
H100/H200 nodes were 8/8 allocated; they are idempotent + `--requeue`-safe.

## Outputs

- `outputs/videos/exp_077_synth_stratum/run_NNNN/`
  - `videos/*.mp4` — 40 tuple clips (`tup{NN}_{ref,tgt}.mp4`) + `clips_manifest.json`
  - `meta/tuple_NN.json` — per-tuple metadata (operator, timing, endpoints, gate, commit)
  - `tuples.json`, `determinism_check.json`
  - `dataset/` — `latents/`, `cond_clean/`, `conditions/`, `roots/synth_smoke/`, reports
- `outputs/videos/exp_077_synth_stratum_audit/run_NNNN/`
  - `videos/*.mp4`, `audit_report.json` (per-family gate acceptance), `effective_K.json`
- `outputs/training/exp_077_synth_stratum/synth_smoke/` — LoRA checkpoints + train log

---

# D2 — the REAL-STREAM redesign (2026-07-24)

## Why

**D1 (3,072 tuples) was rejected on visual review**: "endpoints are nice but the transitions
are NOT — a big proportion completely breaks the scene." Root cause: the renderer only ever
saw 9-frame endpoint snippets and **fabricated** the other ~112 frames of each layer via an
extension policy — `boomerang` (motion runs BACKWARD), `hold` (freeze) or `flow` (Farneback
flow re-applied with decay 0.97, so error compounds and the frame MELTS over 30-60 frames).
D1 used `flow`, so long pure phases melted. But the endpoint-bank clips are **full 121-frame
standardised clips** — the renderer was inventing frames we already had.

## What changed (LOCKED D2 spec)

| | D1 | D2 |
|---|---|---|
| layer streams | `start9`/`end9` + extension policy (`flow`) | **REAL frames**: A-layer = clip A[0:121], B-layer = clip B[0:121], lockstep |
| `extension` | `"flow"` | `"none"` — **the extension code path does not exist in the D2 renderer** |
| timing | `onset ∈ t0+[0,0.60]·win`, `duration ≥ 12` | `onset = 8 + u1·0.20·104`, `release = 112 − u2·0.20·104`, u1,u2 ~ U[0,1] indep ⇒ duration ≥ 62.4, pure phase ≤ 20.8/side |
| easings | 12 | **10** (12 minus `snap_early`, `snap_late` — they manufacture near-cuts) |
| aux maps | 50% of draws | **ZERO** (`aux_kind: null` always) |
| shaders | 122 gate-1 pass, minus 8 holdout | **112**: gate-1 pass − aux-sampler (`luma`,`displacement`) − hard blacklist − D1's 8 holdout |
| params | `sample_params(p_vary=0.85)` | unchanged |
| endpoint gate | gate-2, tol 0.5, 200 tries | gate-2, tol 0.5, **20 tries**, production res 480×640 |

Pinned anchors now fall out for free: progress is 0 for `t ≤ onset` and 1 for `t ≥ release`
with `onset ≥ 8`, `release ≤ 112`, so `out[0:9] == A[0:9]` and `out[112:121] == B[112:121]`
**exactly** (measured: 856/896 audit clips at MAE 0.0000, max 0.238, zero failures).

## Files

- `streams_real.py` — real-stream rendering, D2 timing/ramp, D2 easings, D2 (plain-only) bank
- `d2_metrics.py` — Assert1 (pure-phase identity), Assert2 (seam, exp_076 `ops3d.seam_error`
  semantics), M1 (mush: p10 over the ramp of `max(zNCC(F,A_src), zNCC(F,B_src))` @96×72 gray,
  compared at the SAME t), M2 (near-cut: max single-frame jump of min-max-rescaled `q(t)`)
- `render_d2.py` — `MODE=sanity` (Task 1) / `MODE=audit` (Task 2, shardable, resumable JSONL)
- `calibrate_d2.py` — `PHASE=select` (labeling sheets) / `PHASE=tau` (freeze τ, write audits)
- `owner_sheet_d2.py`, `d2_sheets.py` — the owner contact sheet
- `config_d2.yaml`, `job_render_d2.sbatch`, `job_d2_sheets.sbatch`
- `D2_LABELS.json` (visual labels + criteria), `D2_TAU.json` (frozen gate), `D2_AUDIT.json`

## How to run

```bash
cd $LAB/diffusion-research/experiments/exp_077_synth_stratum
sbatch --export=ALL,MODE=sanity job_render_d2.sbatch                        # Task 1: 6 tuples
sbatch --array=0-11%12 --export=ALL,MODE=audit,NSHARDS=12 job_render_d2.sbatch  # Task 2: 448 tuples
sbatch --export=ALL,STEP=select job_d2_sheets.sbatch                        # Task 3a: label sheets
PHASE=tau python calibrate_d2.py                                            # Task 3b: freeze τ
sbatch --export=ALL,STEP=owner  job_d2_sheets.sbatch                        # Task 4: owner sheet
```

All rendering is **CPU-only** (moderngl + EGL/llvmpipe, `secondary` partition, **no `--gres`**).
NOTE: PyAV/swscale cannot allocate on the login nodes (`BlockingIOError` in `reformat`), so any
step that reads or writes an mp4 must run inside a batch job.

## Result (see `D2_AUDIT.json`)

448 tuples / **896 clips**, 112 shaders × 4 tuples, 0 gate-2 exhaustions.

- **Assert1 pure-phase identity: 856/896 clips are EXACTLY 0.0000**; 40 (4.5%) fail. Assert1 is
  gated as a **MAX** condition (`max_pure`), not a mean: the mean-MAE reading passed all 896,
  but 40 clips are locally 29–241 off the source while their mean MAE is only ~0.2. Six shaders
  do it (`BowTieVertical`, `undulatingBurnOut`, `BowTieHorizontal`, `BlockDissolve`, `Radial`,
  `StarWipe`) — gate-2 misses them because it checks ONE frame pair, not every pure-phase frame.
  Real-stream endpoint pinning itself is exact; this is a residual shader-identity defect.
- **Assert2 seam ≤ 2.0: 211/896 fail (23.5%)**, concentrated in `out_expo` (45.7%) / `in_expo`
  (34.9%) / `out_cubic` (30.2%) — easings whose progress rises fastest right at a handoff.
- **M2 near-cut ≤ 0.5: 19/896 fail (2.1%)**.
- **M1 τ = 0.2543 (FROZEN, `D2_TAU.json`)**: 211/896 fail. Whole-gate pass **504/896 clips = 56.3%**;
  a training *tuple* needs BOTH its clips, so the operative yield is **202/448 tuples = 45.1%** —
  a full 3,072-tuple D2 build must plan for ~2.2× overdraw (or per-shader pre-filtering).
- **Visual bad rate on real streams: 7/40 labeled clips = 17.5%** (Wilson 95% CI 8.8–32.0%) —
  **below** the 30% pre-registered overturn threshold. Every labeled-bad clip is caused by the
  **shader/params** (extreme-zoom destruction, coordinate-warp shredding, flat-colour/black
  matte domination, chromatic glitch), never by fabricated frames.
- **τ rule INFEASIBLE**: no τ both rejects all labeled-bad and passes ≥90% of labeled-good.
  τ = 0.2543 is the lowest value that rejects all 7 bad clips; it retains 84.8% of good (M1 leg)
  / 81.8% (full gate). The 5 sacrificed good clips are all **geometric-recomposition**
  transitions (`cube`, `DirectionalScaled`, `splitSlideInOutHorizontal`, `TopBottom`,
  `splitSlideInHorizontal`) — zNCC-vs-source penalises spatial *displacement*, which M1 cannot
  distinguish from *destruction*. **This is the known limitation of the M1 instrument.**
- **43/112 shaders blacklisted** at >50% rejection; **39/112 pass at ≥80%**. The blacklist
  sweeps in the geometric-split/cube family for the reason above — an owner/advisor call.
- **Residual M1 leak**: M1 is a p10 over the ramp, so a SHORT destruction burst survives it —
  6.4% of gate-passing clips have `min s(t) < 0.15`. A per-frame floor would close it; not added
  because the spec froze M1 as a p10.

## Outputs

- `outputs/videos/exp_077_synth_stratum_d2/sanity/` — `SANITY.json`, videos, filmstrips
- `outputs/videos/exp_077_synth_stratum_d2/audit/` — `videos/` (896 mp4), `filmstrips/`,
  `meta/rows_shard*.jsonl` (per-clip metrics), `meta/plan.json`, `meta/bank_info.json`,
  `label/labelsheet_01..20.png`, `verdicts.json`
- `outputs/endpoint_candidates/d2_audit/owner_sheet.png` — **the owner deliverable** (20 random
  gate-PASSING clips) + `filmstrips/`, `videos/`, `owner_sheet_index.json`
- `outputs/viewers/d2_dataset/index.html` — browsable video viewer (`build_viewer_d2.py`); serve
  from the REPO ROOT so `/outputs/...` resolves: `python -m http.server 8017`

**The training run is BLOCKED on the owner's OK on `owner_sheet.png`.**

---

# D2-FULL — the FINAL 3,072-tuple build (2026-07-24)

## Structure (LOCKED)

3,072 passing tuples = **384 target pairs × exactly 8 operators**, ≥6 distinct shaders per pair,
768 ref pairs (content-disjoint from their target pair) each reused ~4×, 7 easings, **aux 0%**,
40-shader blacklist applied at **sampling** time (72-shader bank), 8 holdout shaders held out,
`extension: "none"`, fixed 121 frames, timing redrawn per attempt.

## Two rulings that shaped this build

### 1. Parameter clamping is ABANDONED PERMANENTLY

`param_clamp.py` stays on disk as a record but **never runs** (`param_filter=None`, the
byte-identical validated path; `sampling.param_clamp: false`, asserted by the renderer).

Its rule 2 took `[0.5d, 2d] ∩ |v| ≤ 3.0`. That intersection is **EMPTY** for
`EdgeTransition.edge_brightness` (d = 8.0) and a **single point** for `ColourDistance.power`
(d = 5.0), so the absolute cap won on **100% of draws including the canonical default** and
collapsed both parameters to the constant 3.0 — the *destructive* direction (dimmer edge map =
near-black frames; lower power = white blowout). The pre-committed first-chunk check failed
accordingly (25.0% BAD raw, 16.7% uniform-projected vs a 17.5% baseline) and array 9659414 was
killed. Its clips are discarded. See `D2_FIRSTCHUNK_VISUAL.json`.

### 2. A PIXEL-level gate instead — the degenerate-frame gate (DFG)

Every observed failure mode — near-black frames, white blowout, flat mattes (brown / lavender /
orange / olive), saturated washes, geometry blank-outs — is a **degenerate frame**: extreme mean
luma or near-zero spatial variance. `dfg.py` detects it there.

This also dodges the retired per-frame-floor failure: a zNCC floor could not separate
decorrelation-by-NOISE (legitimate `StaticFade` grain) from decorrelation-by-FLATNESS (a junk
matte). **Raw luma statistics separate them trivially — grain has high pixel variance, a matte has
none.**

Per frame in the **transition window only** (`ramp = range(i0+1, j0)`, identical to M1's ramp; the
pure phases are byte-identical real frames and are never flagged), on the 96×72 grayscale M1
already computes plus a 96×72 downsampled RGB:

| test | condition | guard |
|---|---|---|
| near-black | `L(t) < θ_black` | only if `min(L_Asrc(t), L_Bsrc(t)) > 0.15` |
| near-white | `L(t) > θ_white` | only if `max(L_Asrc(t), L_Bsrc(t)) < 0.85` |
| flat | `S(t) < θ_flat` | **none** — real frames are never near-zero variance |
| sat wash | `sat(t) > θ_sat` and `S(t) < m·θ_flat` | optional; added only if flatness misses it |

A clip is REJECTED at **≥ K flagged frames** (a 1–2 frame stylistic flash is tolerated; sustained
degeneracy is junk). **No shader exceptions** — a fade *through* a held solid counts as BAD.

The DFG is an **additive** criterion evaluated only on clips that already passed the frozen gate,
i.e. **AND-composed downstream** of it. The accepted set is therefore a strict SUBSET of the frozen
gate's, so **τ = 0.2543, the frozen gate and the blacklist are UNTOUCHED — no recalibration**.

## Files

- `dfg.py` — the gate (features + decision), reused by the calibration and the renderer
- `calibrate_dfg.py` — `PHASE=features` (batch: re-render every graded clip from its recorded
  operator so the features are measured on RAW frames) / `PHASE=grid` (the pre-committed search)
- `DFG_CALIB.json` — the calibration record: full grid table, chosen thresholds, per-clip features
- `render_d2_full.py` — the mass render (frozen gate AND DFG, per-slot rejection sampling)
- `config_d2full.yaml` — `sampling.param_clamp: false`, `dfg:` = the calibrated config
- `contact_sheet.py --blind` — index-only captions + a separate `blind_key.json`, for blind grading

## How to run

```bash
cd $LAB/diffusion-research/experiments/exp_077_synth_stratum
sbatch job_dfg_calib.sbatch                    # CPU: DFG features on every graded clip
PHASE=grid python calibrate_dfg.py             # the pre-committed grid + the bar
python plan_d2_full.py                         # d2full_plan.json (384 x 8)
sbatch --array=0-15%16 --export=ALL,NSHARDS=16 job_render_d2full.sbatch
python contact_sheet.py --sub d2full --role target --limit 64 --blind --out <dir>
python audit_d2full.py                         # -> D2_BUILD_AUDIT.json (stage2_render)
sbatch --array=0-5%6 --export=ALL,NSHARDS=6 job_encode_d2full.sbatch     # L40S
sbatch job_assemble_d2full.sbatch                                        # ONE combined root
python make_d2_train_config.py                 # configs/d2_gen.yaml  (TRAIN IS HELD)
python build_viewer_d2full.py
```

All rendering is **CPU-only** (`secondary`, never `--gres`); PyAV cannot allocate on login nodes,
so any step touching an mp4 must run inside a batch job.

## Result 1 — the DFG calibration ESCAPED (`DFG_CALIB.json`)

Features were measured on **RAW** frames by re-rendering all 960 already-rendered clips from
their recorded operators (the mp4 codec moves the features by at most |ΔL| 0.007 / |ΔS| 0.002 /
|Δsat| 0.015, so it is not a confound). **No config in the declared grid met the declared bar.**

| leg | bar | best reachable |
|---|---|---|
| baseline BAD recall | ≥ 5/7 | **4/7** |
| round-5 BAD recall | 5/5 | **3/5** |
| false positives | ≤ 2/38 GOOD | 2 (at the configs that reach only 4/7 · 3/5) |
| StaticFade passes | 5/5 | **2/5** (3/5 at θ_flat 0.02) |

**The failure is structural, not a grid artifact:**

1. The premise *"grain has HIGH pixel variance, a matte has NONE"* is **false at 96×72
   INTER_AREA** — area-averaging averages the grain away. StaticFade's downsampled luma std is
   **0.012–0.040**, only one order above the true mattes (**0.000–0.002**) and *below every
   declared θ_flat* {0.02, 0.03, 0.05}, so every grid value flags StaticFade.
2. 3 of 7 baseline BAD and 2 of 5 round-5 BAD are **texture/geometry destruction with entirely
   normal luma statistics** (S_min 0.083–0.186: extreme zoom, coordinate shredding, chromatic
   glitch, saturated flash, StereoViewer). Reaching them by flatness needs θ_flat above the
   lowest labeled-GOOD S_min (0.012), which false-positives GOOD clips including all 5 StaticFade.

Per the pre-committed escape: **no detector ships**; the render is **unclamped at baseline**
(frozen gate only) and the residual is documented. The bar was not softened.

## Result 2 — the first-chunk BLIND check PASSED (`D2_FIRSTCHUNK64_BLIND.json`)

64 target clips = the first 64 tuple_ids = 8 target pairs × their 8 operators, **uniform
allocation, no offender oversampling**, graded from index-only captioned filmstrips.

**2 BAD (3.1%), 11 MARGINAL, 51 OK** — Wilson-95 [0.9%, **10.7%**], well under the 17.5%
baseline and under the 5/64 bar ⇒ **PASS**, array ran to completion. The two BAD are
`Overexposure` (white blowout) and `PuzzleRight` (solid black/white block matte); **neither is
visible to `m1_min`** (+0.733 / +0.304). The drop from 17.5% is explained: 6 of the 7
baseline-BAD clips came from shaders the 40-shader blacklist now excludes at sampling time.

## Result 3 — the delivered build (`D2_BUILD_AUDIT.json::stage2_render`)

**3,072 tuples / 6,144 clips**, 384 target pairs at **exactly 8** operators and **8 distinct
shaders** each, all 72 shaders used (allocation 3–48; `swap` gets only 3 at a 98.6% gate-rejection
rate), aux `null`, extension `none`, no blacklisted or holdout shader, easings within the kept 7.

- **pure-phase identity EXACT: max abs diff 0.0000** over all 6,144 clips (anchor-9 MAE 0.0 too)
- realized overdraw **1.5869×** (ceiling 2.5) over 9,750 renders; attempts/slot mean 1.79, max 22
- timing law survives rejection sampling: u1 0.485±0.282, u2 0.494±0.293; onset [8.01, 28.79],
  release [91.20, 112.00]
- non-gating `m1_min_flag`: 18 clips / 14 tuples
- DFG: 0 clips reached it (it never shipped), so its per-shader rejection log is empty **by
  construction**

### Gap-fill (`D2_GAPFILL.json`)

4 of 3,072 slots exhausted the 25-attempt ladder, and re-running reproduced it exactly. The
blocker is the slot's **reference pair**, which the ladder never redraws: 3 of the 4 share one
reference whose clip has a **perfectly static pure phase**, so assert2's ratio denominator
collapses to its 1e-3 floor (seam 1.8–1560 against a ≤2.0 gate) and the degenerate `q(t)` pins m2
at ~0.64 against ≤0.5 — unpassable for any shader. Those slots were re-run with a **substituted
content-disjoint reference pair** (least-used first, never a pair that already proved
unpassable): this spends the slack the spec marks as *"reused ~4×"* to protect the invariant it
states exactly, *"384 target pairs × **exactly 8** operators"*. Gate, τ, bank, easings, ladder and
timing law are unchanged; all 4 passed on the first or second attempt.
