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
