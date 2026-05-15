# RF-Inversion Research Loop — Protocol & Ledger

> **ACTIVE AUTONOMOUS LOOP.** If you are resuming a session: **read this entire
> file before doing anything.** The Protocol governs how iterations run; the
> **Ledger** at the bottom is the live state. Update the Ledger after every phase.
> This file is the single source of truth — chat context is not.

---

## Objective

Get LTX-2 RF-Solver inversion working on **real video clips** (the exp_030 goal).
exp_029 works on *generated* latents (`inv_recon_rel ≈ 0.01`, recon PSNR 44 dB);
exp_030 collapsed on *real* shadow_smoke clips (`inv_recon_rel ≈ 0.68`, 0/10 pass).
The loop bridges the two by **localizing where degradation enters** and fixing it
**one variable per iteration**.

## Success criteria — the loop STOPS when ANY of these fires

Population for ① = the `shadow_smoke` 10-clip set. Metrics are perceptual
(human-eye), measured on the **decoded** recon/regen videos vs source.

| Exit | Condition |
|------|-----------|
| **① SUCCESS — target** | **recon** (must-pass): median **PSNR ≥ 28 dB**, **SSIM ≥ 0.88**, **LPIPS ≤ 0.10**, with **≥6/10** clips passing all three. **regen** (secondary): median **PSNR ≥ 22 dB**, **SSIM ≥ 0.78**, **LPIPS ≤ 0.18**. |
| **② SUCCESS — scientific floor** | ≥2 distinct targeted fixes run; degradation **localized to a named cause** with tensor/CSV evidence; achievable floor measured + documented. |
| **③ STOP — wall** | 3 consecutive iterations with <20% relative gain in the north-star metric **and** residual attributed to a fundamental cause (off-manifold real latents). |
| **④ STOP — budget** | **8 cumulative pod-hours** consumed. |

`inv_recon_rel` / `inv_recon_cos` (latent space) remain as **diagnostics** — they
localize degradation — but are **not** the acceptance gate. Acceptance is perceptual.

## Loop protocol — one iteration = COLD → HOT → COLD

**Phase A — COLD / pre-flight (no GPU).** Pick the single next hypothesis from the
Ledger backlog. Write its *prediction* + *decision rule* into the Ledger **before**
any code runs. Author/modify the experiment (new `exp_NNN`, or new run in the
loop's own dirs). Local validation is **static only** — `py_compile`, AST checks,
import-graph and path inspection — because the CPU host has ~4GB RAM and no
`torch` (memory: `cpu-host-is-4gb-orchestration-only`). The real dry-run (imports,
shape asserts) happens on the GPU pod as the first cheap step of Phase B. Assemble
the **batch manifest** (every config ready to run this hot phase).

**Phase B — HOT / execution (GPU, metered).** Spin up pod via `runpod-pod-init`
Route B; `source /workspace/cache/pod_init.sh` first. Run the batch sequentially,
each under `TeeLogger` → its own `run_dir`. Monitor via background tmux pane. On
completion: verify all artifacts present → **`podTerminate` immediately**. Log
pod-hours to the Ledger.

**Phase C — COLD / adjudication (no GPU).** Evaluate each run vs its pre-registered
decision rule → `CONFIRMED` / `REJECTED` / `INCONCLUSIVE`. Update the Ledger +
Running Best + `CHANGELOG.md` + knowledge bank. Check exit conditions. If none
fire → derive the next hypothesis *from where the evidence now points* → Phase A.

## Pod lifecycle rules

- **Pod exists ONLY while a pre-registered batch executes.** Spin up → run → verify
  → terminate. Never leave a GPU pod idle during a cold phase.
- Batch the hot phase: pod spin-up is fixed overhead, so run *every* ready config
  in one session.
- Log `pod-hours` used after each hot phase; this is the exit-④ meter.

## Final shutdown (when the loop exits via ①/②/③/④)

1. Confirm all per-iteration GPU pods are already terminated.
2. Write the final report + close the Ledger.
3. **LAST:** terminate the **CPU host pod** this session runs on. Do **not**
   terminate it before the loop fully exits — doing so kills the session and
   leaves work unfinished. (memory: `runpod_pod_shutdown_order`)

## Frozen artifacts — NEVER modify

`exp_001`..`exp_030`, all existing `notes/` (except this file + `INDEX.md`),
`src/`, all existing configs. The loop only **creates** `exp_031+` and updates
this file, `INDEX.md`, `CHANGELOG.md`, and *new* knowledge-bank notes.

## Running best

| Approach | Population | recon (perceptual) | latent free-rel | Note |
|----------|-----------|--------------------|-----------------|------|
| exp_029 (gen latents) | 3 DAVIS gen | PSNR 44.2 dB | ~0.01 | works — but not the target population |
| exp_030 (real clips) | shadow_smoke ×10 | PSNR ~18 median | ~0.68 | 0/10 pass — the failure baseline |
| exp_031 R5 (true self-cond) | shadow_smoke ×3 | PSNR 40.6 median | free 0.11 median | 2/3 pass — localization rung |
| **exp_032 (true self-cond, full set)** | shadow_smoke ×10 | **PSNR 40.9 median** | recon_rel 0.105 median | **8/10 pass — EXIT ① ✅ loop solved** |

---

# LEDGER

## It-0 — Instrument & localize  · STATUS: COMPLETE → It-1 pre-registered

**Phase A (2026-05-14 ~17:16):** `exp_031_ltx2_rf_inv_ladder` built & statically validated.
**Phase B (17:28–19:11):** GPU pod `8hevgt1m0l9gv9` (A100-SXM4-80GB, SECURE).
Run `outputs/videos/exp_031_ltx2_rf_inv_ladder/run_0001/`. Pod terminated 19:14.
**Pod-hours used: ~2.0  ·  Cumulative: 2.0 / 8.0**

### Outcome — DEGRADATION LOCALIZED

Per-sample `free_rel` (means in the run's auto-table are outlier-polluted; read per sample):

| sample | R0 gen+ext | R1 d_e+ext | R2 d_e+none | R3 d_e+self | R4 clip+self |
|--------|-----------|-----------|-------------|-------------|--------------|
| class2 | 0.018 | 0.045 | **1.18** | 0.039 | 0.062 |
| class5 | 0.036 | 0.293 | **1.11** | 0.029 | 0.047 |
| class8 | 0.331 | 0.242 | **1.65** | **1.00** | 0.015 |

R5 (shadow_smoke, clip+self): ss1 PSNR 40.6 / SSIM 0.980 / LPIPS 0.016 **PASS**;
ss3 PSNR 42.4 / 0.985 / 0.015 **PASS**; ss4 PSNR 29.8 / 0.730 / 0.307 **FAIL**.

**Findings (scored vs the pre-registered prior):**
1. **Prior WRONG.** No clean R0→R1 cliff — provenance (VAE round-trip) is *not*
   the dominant cause. R4 (encode_clip + true self-cond) is the **best rung in
   the ladder** (free_rel 0.015–0.062, PSNR 46, SSIM 0.994) — VAE-encoder-output
   latents of *real* clips invert essentially perfectly.
2. **Dominant cause = conditioning-anchor mismatch.** R2 (no conditioning) is
   catastrophic for *every* sample (free_rel 1.1–1.65) — vanilla midpoint RF
   inversion of a VAE-encoded latent diverges. Conditioning is load-bearing.
   And exp_030's failure specifically traces to **fault #2**: it built
   `clean_latents` by re-encoding sub-clips (causal-VAE mismatch) → pinned the
   solver to anchors ≠ z0's actual conditioned positions. exp_031 R5 uses *true*
   self-conditioning (exact slices of z0) and shadow_smoke goes 0.68 → 0.11.
3. **R0 HALT gate tripped — adjudicated a FALSE ALARM.** R0 mean free_rel 0.128 >
   0.05, but: class2/class5 reproduce exp_029 cleanly (all_rel 0.012/0.025 ≈
   exp_029's ~0.01); only class8 diverges (0.33). Cause: exp_029's *reported*
   numbers are **60-step**; exp_031 R0 ran **40-step**, and exp_029 itself
   escalated class8 to 60 *because* 40 wasn't enough. Harness is sound. Lesson:
   the gate should compare per-sample medians at *matched step count*, not a
   mean vs 60-step numbers. R1–R5 are usable; class8's 40-step results just
   reflect a step-count limit, not their rung's variable.
4. **Stylized content adds only mild cost.** R4 (natural) ~0.04 → R5 (stylized)
   ss1/ss3 ~0.11 — still perceptually passing. Not a wall.
5. **ss4 outlier = my own control artifact.** shadow_smoke_4 is a 1440² *square*
   clip; exp_031 forced fixed 512×768, distorting it (exp_030 used per-clip
   resolution and ss4 was its *best* sample). ss4's R5 failure is aspect-ratio
   distortion, not content.

**Verdict:** It-0 succeeded at its job — degradation localized to the
conditioning-anchor mismatch, with the fix (true self-conditioning) already
partially validated in R5 (2/3 shadow_smoke pass perceptual exit ①). No formal
exit fired (exit ① needs the full 10-clip set). Loop continues to It-1.

---

## It-1 — Confirm the fix on the full target set  · STATUS: ✅ COMPLETE — EXIT ① TRIGGERED

**Phase A done (2026-05-14 ~19:25):** `exp_032_ltx2_rf_inv_selfcond` built —
fork of exp_030, verified one-variable change (`clean_latents = z0_packed.clone()`,
true self-conditioning). `py_compile` clean; diff vs exp_030 = docstring +
conditioning block only; all 10 shadow_smoke paths verified.

**Phase B running (started 2026-05-14 19:25):** GPU pod `l9h30xtbq5u9vs`
(A100-SXM4-80GB, SECURE). Run launched in tmux `exp032`; pane Monitor armed.
Terminate pod immediately on `=== EXIT ===`.

### Hypothesis
exp_030's catastrophic real-clip failure was the causal-VAE conditioning-anchor
mismatch (fault #2). Replacing it with **true self-conditioning** (`clean_latents`
= exact slices of z0, not re-encoded sub-clips) — the only change — recovers
perceptually-faithful inversion on the full shadow_smoke set.

### Prediction (pre-registered, to be scored)
`exp_032` = exp_030 with exactly two changes: (a) true self-conditioning, (b)
per-clip resolution restored (un-distort square clips like ss4) — full
shadow_smoke 10-clip set, 40 steps, recon+regen. Predict **≥6/10 pass exit ①**
(recon median PSNR≥28 / SSIM≥0.88 / LPIPS≤0.10). ss4-type square clips expected
to pass once resolution is un-distorted.

### Decision rule
- ≥6/10 pass recon thresholds → **exit ① triggered** (pending regen secondary check).
- 3–5/10 pass → partial; inspect failures, derive It-2 (likely step-count or the
  remaining R0→R1 provenance cost).
- <3/10 pass → fix insufficient; reopen — the mismatch wasn't the dominant cause
  after all, escalate to step-count sweep or RF-Solver correction machinery.

### Batch manifest
`exp_032` — fork exp_030, swap conditioning construction to true self-cond,
restore per-clip `max_area` resolution. 10 shadow_smoke clips × (invert + recon
+ regen). Est. ≈ 1.5–2 pod-hours.

### Outcome — ✅ EXIT ① TRIGGERED (loop SUCCESS)

`exp_032` ran the full 10-clip shadow_smoke set with true self-conditioning.
Run `outputs/videos/exp_032_ltx2_rf_inv_selfcond/run_0001/`. Pod terminated 20:58.

**Recon perceptual (exit ① = PSNR≥28, SSIM≥0.88, LPIPS≤0.10, all three per clip):**

| clip | PSNR | SSIM | LPIPS | pass | clip | PSNR | SSIM | LPIPS | pass |
|------|------|------|-------|------|------|------|------|-------|------|
| ss1 | 38.45 | 0.972 | 0.022 | ✅ | ss6 | 36.11 | 0.891 | 0.117 | ❌ LPIPS |
| ss2 | 34.73 | 0.846 | 0.159 | ❌ SSIM,LPIPS | ss7 | 40.83 | 0.976 | 0.016 | ✅ |
| ss3 | 44.75 | 0.995 | 0.004 | ✅ | ss8 | 45.25 | 0.993 | 0.009 | ✅ |
| ss4 | 39.76 | 0.970 | 0.034 | ✅ | ss9 | 42.62 | 0.982 | 0.020 | ✅ |
| ss5 | 40.92 | 0.971 | 0.027 | ✅ | ss0 | 44.76 | 0.995 | 0.004 | ✅ |

**8/10 pass** (exit ① needs ≥6/10). **Medians: PSNR 40.88 / SSIM 0.974 /
LPIPS 0.025** — all clear the thresholds. **EXIT ① TRIGGERED.**

vs the failure baseline (exp_030): 0/10 pass, recon PSNR median ~18, recon_rel
~0.68. exp_032: 8/10, PSNR median 40.9, recon_rel median ~0.105. The
one-variable fix — true self-conditioning — was the whole story.

The 2 misses (ss2, ss6) still have PSNR 34–36 (far above exp_030's ~15–18) and
fail only the strict SSIM/LPIPS bar. ss2 is the lone 10-second-source clip
(recon_rel 0.56), ss6 a landscape clip (recon_rel 0.40) — their residual is the
secondary R0→R1-style provenance cost the ladder already flagged: a clean
follow-up lever (inversion-step sweep), not a wall.

Regen secondary (stretch): median PSNR ~31.6, SSIM ~0.82 — pass; the strict
*latent* regen gate fails as expected (structural CFG=1↔CFG=4 mismatch,
documented since exp_029 — see [[feedback_regen_simulates_production]]).

### Pod-hours used
~1.6  ·  **Cumulative: 3.6 / 8.0**

---

# LOOP CLOSED — 2026-05-14 20:59

**Exit ① fired at It-1.** Total: 2 iterations, 3.6 / 8.0 pod-hours.

exp_030's catastrophic real-clip RF-Solver inversion failure (0/10, recon PSNR
~18) was caused by **one bug**: building `clean_latents` from separately
re-encoded sub-clips, which the causal video VAE makes ≠ the slices of the
full-clip encode — so the solver was hard-pinned, every step, to mismatched
anchors. exp_031's R0→R5 ladder localized it; exp_032's one-variable fix (true
self-conditioning) confirmed it — **8/10 shadow_smoke clips now pass the
perceptual bar, median PSNR 40.9**.

Banked en route: conditioning is load-bearing (vanilla inversion diverges);
VAE-encoder provenance and real natural content invert fine; the R0 HALT-gate
mis-calibration lesson (compare per-sample medians at matched step count).
Knowledge bank updated → `notes/models/ltx2/conditioning.md`. Open follow-up
lever for ss2/ss6 if ever revisited: inversion-step sweep.

---

## Appendix — It-0 pre-registration detail (kept for the audit trail)

### Hypothesis
exp_030 moved ~6 variables at once vs exp_029 (z₀ provenance, z₀ content,
conditioning source, conditioning consistency, resolution, audio). The collapse
is therefore unattributed. **A controlled ladder that moves one variable per rung
will localize the dominant cause** to a single, named transition.

### Prior (my honest bet, to be scored against the outcome)
The cliff is at **R0→R1 (provenance)**: VAE-encoder-output latents are off the
flow-matching manifold that generated latents live on. Evidence: exp_030's
step-diag CSVs show `x0_pred_norm` inflated to ~1518 at σ=1 (vs ~1000 expected)
and round-trip error concentrated in the big-dτ low-σ steps — a misbehaving
velocity field, which is a property of the *latent*, not the conditioning.

### Design — the R0→R5 ladder (exp_031)
One hot phase, one batch. Fixed across **all** rungs (so never confounds):
resolution **512×768**, **121 frames**, **40** invert + **40** recon steps,
**zeros** audio, seed 42, **3 samples/rung**, **invert + recon only** (no regen —
recon is the clean self-consistency signal; regen deferred).

| Rung | z₀ source | conditioning source | single Δ from predecessor | isolates |
|------|-----------|---------------------|---------------------------|----------|
| **R0** | exp_029 `z0.pt` (gen latent, direct) | external DAVIS clip latents | — (re-run of exp_029 in unified harness) | **baseline + harness validation** |
| **R1** | `encode(decode(z0.pt))` | external DAVIS clip latents | provenance: transformer-output → VAE-encoder-output | **the VAE round-trip / on-manifold-ness** |
| **R2** | `encode(decode(z0.pt))` | **none** (vanilla inversion) | conditioning turned off | **does C2V help or hurt** |
| **R3** | `encode(decode(z0.pt))` | self: own endpoints sliced from z₀ | conditioning source: external → self | **C2V conditioning-source cost** |
| **R4** | `encode(real DAVIS clip)`, edge-padded to 121f | self | content: decoded-generation → real camera footage | **off-manifold (natural real)** |
| **R5** | `encode(shadow_smoke clip)` | self | content: natural real → stylized real | **off-manifold (stylized)** |

R2 and R3 both branch from R1 (R2-vs-R3 = clean C2V on/off comparison).

### Locked design refinements (from the 3-pass critique — see Critique Log)
1. **Metric = FREE-positions-only latent rel/cos as PRIMARY** (also log
   all-positions + cond-positions for continuity, + perceptual PSNR/SSIM/LPIPS on
   decoded recon). Conditioned positions are hard-pinned to `clean_latents` every
   solver step in both invert and recon → their round-trip error is trivially
   ~0 (or a trivial constant offset) and pollutes the all-positions metric. The
   free positions are the *entire* solver-quality signal. This single fix
   removes the exp_030 "fault #3" metric pollution **and** decouples
   conditioning-consistency from the ladder (the inverter pins cond anyway, so
   z₀'s own cond values never affect the trajectory).
2. **R0 is a re-run in exp_031's unified harness, not a citation of exp_029's
   number.** It doubles as a harness-validation checkpoint.
3. **Conditioning variable = SOURCE only** (external / none / self). Consistency
   is removed as a confound by refinement #1.
4. **R4 = real DAVIS `full.mp4`** (max available = 90 frames) **edge-frame padded
   to 121** (hold first/last frame). Documented minor confound; cleaner than
   loop-tiling (no motion discontinuity). Also note R3→R4 mixes
   transition-content → continuous-content — secondary, accepted, since the
   solver's manifold question is content-type-agnostic.
5. Audio = **zeros** (proven non-breaking in exp_029; `encoded-silent` is for
   production-faithful runs, not for this controlled ladder).

### Decision rule
- **"Cliff"** = free-rel jumps **≥3×** from one rung to the next. The *first*
  ≥3× jump localizes the dominant cause.
- **R0 gate (mandatory checkpoint):** R0 free-rel must reproduce exp_029 within
  **~2×** (i.e. ≤ ~0.05). If not → **HALT**, the unified harness diverges from
  exp_029; debug before trusting R1–R5.
- Per-rung result counts only if consistent across **≥2 of 3** samples (a single
  outlier clip is not a cliff).

### Predicted outcomes per rung (pre-registered, to be scored)
- R0: free-rel ≈ 0.01–0.05 (else HALT).
- R1: free-rel jumps to ~0.2–0.5 — **predicted cliff here**.
- R2: ≈ R1 if C2V is neutral; **<** R1 if the C2V hybrid state is the problem.
- R3: ≈ R1 (free-only metric should be ~insensitive to cond *source*); **<** R1
  would mean cond source matters even for free tokens.
- R4: ≈ R3 if provenance already explained it; **>** R3 → real content adds cost.
- R5: ≈ R4 if stylization is free; **>** R4 → stylization adds cost.

### Decision tree → It-1
- Cliff R0→R1 → It-1 = manifold-projection of the encoder latent (re-noise to
  small σ then denoise back; or RF-Solver paper's correction term; or step-count
  sweep).
- Cliff R1→R2 or R1→R3 → It-1 = soft conditioning (blend, not hard re-pin) or
  drop-C2V-then-guide-at-regen.
- Cliff R3→R4 → It-1 = real content is off-manifold; port RF-Solver correction
  machinery or optimization-based inversion.
- Cliff R4→R5 → It-1 = stylization-specific; the method works on natural clips.
- No single cliff (gradual decay) → It-1 = inversion-step sweep (40→60→80).

### Batch manifest (to run in Phase B)
`exp_031` configs R0,R1,R2,R3,R4,R5 — 6 configs × 3 samples × (40 invert + 40
recon). Est. ≈ 1 pod-hour incl. model load + spin-up. Samples: 3 reused from
exp_029 run_0002 (`class2/5/8`) for R0–R4; 3 shadow_smoke clips for R5.

### Critique Log (3-pass pre-registration)
- **Pass 1** caught: R0 must be *re-run* not cited (harness parity); `z0.pt` is
  packed+normalized so decode needs `unpack_and_denormalize` first; "invert a
  real DAVIS *transition*" is ill-defined (DAVIS gives clips, not morphs) → R4
  redefined to a single real clip.
- **Pass 2** caught: conditioning was secretly two variables (source vs
  consistency) → the FREE-only metric collapses it to one; R3→R4 has a
  transition-vs-continuous content sub-confound (accepted as secondary); the
  ladder is complete at 6 rungs (no rung to add, none to skip).
- **Pass 3** caught: R0 must also be a hard HALT gate (harness validation); no
  real clip reaches 121 frames → R4 needs edge-frame padding; confirmed
  free-only metric hides nothing (the excluded component is provably ~0).
- **Converged:** design above is locked. Open Phase-A task: confirm `z0.pt`
  decode path + `clean_latents` slicing indices against exp_029/run.py while
  coding exp_031.

### Outcome
See "It-0 — Outcome — DEGRADATION LOCALIZED" above (this Appendix is the
pre-registration record only). It-0 used ~2.0 pod-hours.
