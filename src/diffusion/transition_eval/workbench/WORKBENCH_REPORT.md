# Metric Workbench — Report

**Branch** `eval/metric-workbench` · **Run** 2026-07-14 · **Executor** implementation
agent (build → test → run → report neutrally, OPERATIONS §8).

This is a **neutral data package**. Every §7 adoption condition below is a **computed
pass/fail fact**. There is **no adoption recommendation, no interpretation, and no
strategy** in this document — those belong to owner-side review.

Both tracks terminated on pre-registered rules. **Two owner-reserved matters are
escalated** (§7 below).

---

## 0. Outcome at a glance

| track | rung | pre-registered rule that fired | outcome |
|---|---|---|---|
| Phase 2 (appearance) | **E1** | §4.1 KILL RULE | **KILL** — E2/E3 do not run |
| Phase 1 (motion) | **§3.4 acceptance** | "the exam is not run on a metric that fails constructed truth" | **FAIL** — the exam is not run |

Kill rules honored are results, not failures (OPERATIONS §7).

---

## 1. Freeze verification (step 0)

- All six incumbent metrics **reproduce RUNBOOK §B exactly** from the frozen
  `distance_matrices.npz` (sha256 verified) through the frozen exam kernel.
- `m1a__v3_sided` rebuilt from warm bundles with deployed code: **max|Δ| = 0.0**
  (bitwise).
- The workbench wrote **nothing** to the certified shared cache. Corpus reads go
  through a `ReadOnlyExtractor` whose `extract()` raises, so polluting it is
  impossible by construction rather than by discipline; the GPU job printed a
  before/after cache canary and both read **1933 entries, 0 files touched**.

  **Disclosed for accuracy:** the shared cache *did* grow later in the day — 1933 →
  2233 entries, 300 files written between 12:58 and 13:03. That is **not** the
  workbench. It coincides exactly with the owner's own `exp060_score` job
  (12:57:43 → 13:03:18), and the certified harness writes feature caches into
  `outputs/eval/cache` by design when scoring new generations. **All 223 corpus cache
  entries this run depends on are untouched** (mtimes 2026-07-08), and the incumbent
  bitwise round-trip **still reproduces `max|Δ| = 0.0`** against the frozen npz after
  the fact. Every number in this report therefore stands.

### Amendment A4 — the drafted stratum recalls do not exist

RUNBOOK §3.5 cites incumbent stratum recalls of **0.62 / 0.44** from memory. **No
constructible definition reproduces them.** All five readings were computed and
recorded:

| definition (m1b_camera, camera stratum) | value |
|---|---|
| clip-pooled, uncovered = miss | 0.31373 |
| clip-pooled, uncovered dropped (deployed descriptive) | 0.33684 |
| macro per-class, all classes | 0.26929 |
| **macro per-class, eligible (FROZEN as the gate)** | **0.34623** |
| restricted pool | 0.40000 |

A4 pre-registered this backfill and demoted the memory figures, so the backfilled
values are the numbers to beat. The frozen definition reproduces the certified exam's
own persisted `o7_conditional.camera_stratum_mean_recall` **bit-for-bit**. Object
stratum (m1c_object): **0.03426**.

---

## 2. Phase 2 — appearance · E1 (§4.1 KILL TEST)

**Rule (verbatim):** *"if delta fails to beat raw M1a on **both** Cohen's d and
misretrieved count, the endpoint-normalization program is dead at the appearance
level. One appendix paragraph, full stop. E2/E3 do not run."*

| | E1 delta | pinned incumbent `m1a__v3_sided` | beats? |
|---|---|---|---|
| Cohen's d | 0.358190 | 1.522006 | **NO** |
| misretrieved | 209/223 | 73/223 | **NO** |
| accuracy | 0.0628 (chance 0.067) | 0.672646 | — |
| coverage | 1.0000 | 1.0000 | — |
| §1.4 hubness | **FAIL** (skew 4.300, entropy 0.322, max-pred 0.650, sink `mystification`) | PASS | — |

**VERDICT: KILL. E2/E3 do not run.** Coverage is 1.0000, so this is not a
shrunken-support artifact.

### Instrument diagnostics (facts; they do not revise the verdict)

The control is the identical representation with **no null subtraction at all** —
i.e. containing zero endpoint-normalization:

| arm | accuracy | Cohen's d | misretrieved | hubness entropy |
|---|---|---|---|---|
| RAW `v_clip`, no subtraction | 0.6054 | 0.9875 | 88/223 | 0.882 |
| WHITENED `v_clip`, no subtraction | 0.0628 | 0.1068 | 209/223 | 0.042 |
| WHITENED delta (**the candidate**) | 0.0628 | 0.3582 | 209/223 | 0.322 |
| RAW delta (unwhitened) | 0.1345 | 0.6211 | 193/223 | 0.524 |

ZCA spectrum: 768 dims, λ ∈ [1.848e-09, 3.637e-02], condition ≈ 1.97e7. The frozen
eigenvalue floor (1e-6 · λ_max = 3.64e-08) floors **1 of 768** dimensions. Whitened
norm means: null 35.52 vs clip 14.61 (raw: 0.912 vs 0.867). Plumbing verified: nulls
are genuinely distinct from their clips (cos mean 0.700, min 0.113), 223/223 deltas
defined.

**Floor-sensitivity diagnostics** (grid pre-declared before computing; no verdict
column, no floor recommended): across `eig_floor_ratio ∈ {1e-6 … 1e-1}` the whitened
no-subtraction control stays at accuracy 0.0628 from 1e-6 through 1e-3 and reaches
0.3722 only at 1e-1 (704 of 768 dims floored), still below the raw control's 0.6054.
The whitened delta does not exceed accuracy 0.0942 or fall below 202/223 misretrieved
at any floor in the grid.

### E0 (§4.2, non-gating; ran regardless of the kill)

Chord detachment (clip residual magnitude / its own rendered null's): most detached
flying_cam_transition 1.217, shadow_smoke 1.191; most chord-hugging live_concert
0.694, polygon 0.726.

---

## 3. Phase 1 — motion · §3.4 acceptance

Full detail: `outputs/eval/workbench/phase1/RECORD.md`.

**Backbone:** SEA-RAFT (amendment A2's timeboxed attempt **succeeded**, so the
RUNBOOK's primary choice shipped; the torchvision fallback was staged and not
needed). Flow at 432×320.

**Test 2 — injected-trajectory: FAIL.** 35 verdict cells (all rung p90; `wireframe_2`
excluded-with-reason as texture-gated). **29 PASS** (corr 0.9618–0.9998, amp_err
0.0004–0.0996). **6 FAIL.** The pre-committed **post-hoc oracle** (σ re-estimated from
each probe's own realized noise; `recovered = truth + N(0,σ)` pushed through the
unchanged grader) splits them: **3 noise-limited** (an oracle could not pass them
either → construction) and **3 fail at rungs the oracle passes with margin** →
**the metric failing constructed truth**. The verdict is **monotone** under any
reclassification of the 3 noise-limited cells.

**Test 1 — reversal: FAIL.** 102 camera-tagged → 12 insensitive (Bar-5 screen, floor
0.5 inherited verbatim from the certified `bars.yaml`) → 57 undefined-ungradable
(§1.5: counted, never failed) → **33 graded**. Leg A1 (closer-to-negated,
threshold-free) **33/33**. Leg A2 (0.90 negation-correlation floor) **17/33**. Leg B
(descriptor distance > median within-class) **22/33** (failing ratios 0.360–0.880).
Joint row pass **11/33**. Leg B's form is frozen verbatim; the construction was
re-checked for a third defect and none was found. **Reversal fails on leg B alone.**

**Consequence (§3.4):** the exam is **not run**; no §3.6 adoption question arises.
M1b_flow and M1c_flow remain **analysis-tier**. No second attempt this cycle
(§3.6, §9).

### Pre-exam facts (computed before any distance existed)

- **Coverage under the frozen gates:** m1b_flow **130/223 (0.5830)**, m1c_flow
  **118/223 (0.5291)** — against incumbent m1b_camera **0.9686** and m1c_object
  **0.9955**. The dominant loss (223 → 143) comes from RUNBOOK-**pinned** rules (40%
  inlier floor, 30% clip cap), **before** either open threshold was set; the two
  calibrations frozen this cycle are second-order (143 → 130 / 118).
- **Mechanism, measured on the corpus:** effect-area fraction (pixels the camera fit
  calls outliers) median **0.875** on undefined core frames vs **0.160** on defined
  ones.
- **§3.6 stratum-recall ceilings implied by coverage alone:** camera **0.3571** (9 of
  14 eligible classes have zero covered clips; incumbent target 0.34623); object
  **0.6111** (7 of 18; incumbent target 0.03426).
- **Corpus camera-motion scale:** median per-pair translation **tx 0.297 px / ty 0.421
  px**, within an order of magnitude of the flow fit's own per-pair parameter noise
  (σ ≈ 0.015–0.017 px). Vigorous decile: tx 3.11 px / ty 2.31 px. **An oracle fails
  §3.4's peak-amplitude criterion at p50.**
- **Huber breakdown (constructed truth):** max parameter error vs contaminated area —
  5% → 0.05 px, 15% → 0.24, 25% → 0.52, 33% → 0.75, 40% → 1.68, 45% → 2.92. Past ~40%
  the inlier fraction collapses and the frame exits **undefined** rather than
  confidently wrong.

---

## 4. §1.4 hubness gate (mandatory per candidate; terminal)

The RUNBOOK mandates this gate but states **no numbers**. They were derived, not
chosen: each threshold is the **midpoint of the empty gap** between the pass and dead
incumbent populations — a deployed convention in the certified instrument
(`probes.grade_splices` fixes `tau_copy` the same way). Frozen: skew ≤ 3.0, prediction
entropy ≥ 0.70, max-pred-class share ≤ 0.25, gating k = 10.

Authority for where the gate sits is RUNBOOK §0 itself ("M1c is hub-collapsed
(polygon column = sink artifact)"; "M1b is merely weak"). The frozen gate reproduces
that diagnosis on all six incumbents (fires on `m1c_object`/polygon and
`m_incumbent`; silent on the m1a family and `m1b_camera`), and a regression fixture
asserts it.

**Candidate verdicts:** E1 delta **FAIL** (sink `mystification`). M1b_flow / M1c_flow:
**not evaluated** — the §3.4 gate is upstream of the exam.

---

## 5. §5 pre-registered prediction checks (descriptive, non-gating)

| # | prediction | result |
|---|---|---|
| P1 | sibling γ-distance < clip-to-own-null distance for every eligible class | **NOT CHECKABLE** in its registered (γ-signature) form — E2 did not run. On the nearest available analogue, **8/29** eligible classes have every clip closer to a sibling than to its own null. |
| P2 | sidedness (26/13) recoverable from s(σ) asymmetry | rank **AUC 0.523** (one-sided mean +0.360, two-sided +0.347). 0.5 = no separation. |
| P3 | nature_bloom remains lerp-adjacent | detachment ratio **0.893** vs corpus median 0.948; rank 13/39 low→high. n=2 class, never gating (§6). |
| P4 | one-sided classes concentrate excursion mass in early σ | one-sided centroid **0.456** vs two-sided **0.499**; one-sided earlier: **True**. |

---

## 6. §7 adoption conditions — computed pass/fail FACTS

§7 governs the **appearance** candidate. E1 is the only appearance candidate that
ran; E2/E3 did not (per the §4.1 kill).

| # | §7 condition | computed |
|---|---|---|
| 1 | Cohen's d improves by ≥ 0.25 over 1.522006 (⇒ ≥ 1.772006) **and** misretrieved < 73 | **FAIL** — d 0.358190; misretrieved 209 |
| 2 | probe battery passes unchanged (splices, siblings > controls, twins 11/11, lerp at floor) | **NOT RUN** — §4.1 kill fired first; the probe battery was never reached |
| 3 | hubness gate passes (§1.4) | **FAIL** — skew 4.300 > 3.0; entropy 0.322 < 0.70; max-pred 0.650 > 0.25 |
| 4 | definedness coverage reported and not materially narrower than the incumbent's (1.0000) | coverage **1.0000** — **reported**. *"Materially narrower" is **not computable**: no frozen materiality threshold exists. The number is stated; the adjudication is owner-side.* |

Phase 1's gate is **§3.6**, not §7, and §3.6 carries no coverage condition. No §3.6
adoption question arises because the exam was not run.

---

## 7. Owner-reserved matters (escalated; both tracks stopped)

**(a) The E1 whitening confound.** RUNBOOK §1.1 mandates whitening but **does not pin
its regularization**. `eig_floor_ratio = 1e-6` was an **executor-chosen** free
parameter, frozen in good faith (`694afc7`) before any candidate ran. The
identically-preprocessed **no-subtraction control** — containing zero
endpoint-normalization — produces the same retrieval statistics as the candidate
(0.0628 accuracy, 209/223). Whether the frozen parameter confounds the §4.1 test, and
whether the recorded verdict bears on the endpoint-normalization hypothesis, are
outcome-aware threshold and kill-rule-interpretation questions — **owner-reserved**
(OPERATIONS §1(5), §8). **No workbench action revised the recorded verdict.**
Floor-sensitivity diagnostics are attached so this can be adjudicated **without a
re-run**.

**(b) Reversal leg A2.** The 0.90 correlation floor is frozen in `gates.yaml` only
under `injected_trajectory`; applying it to reversal negation correlations is an
executor operationalization, applied to real-clip channels whose amplitudes are
uncontrolled. 20 of 132 graded clip×channel cells fall below it (rotation 11, ty 5,
log_scale 3, tx 1); rotation is the corpus's lowest-amplitude channel (per-pair p50
0.00089). No oracle-validity guard was pre-registered for reversal, and constructing
one after the leg failed is precluded by the C5-R pre-commitment. **No mechanism claim
is attached.** Note the reversal verdict does **not** rest on A2 (leg B fails
independently), and §3.4 does not rest on reversal (the injected test fails
independently).

---

## 8. What was NOT run, and why

| not run | why |
|---|---|
| **E2** (γ-signature), **E3** (within-video Gram) | §4.1 recorded **KILL**. `gates.yaml` makes this structural: the driver refuses E2/E3 unless E1's recorded verdict is a pass. |
| §4.5 ablations | they apply to "whichever rung survives" (§4.5); none did. |
| **Phase 1 exam** (§3.5), §3.6 verdict | §3.4 acceptance FAILED: "the exam is not run on a metric that fails constructed truth." |
| M1b_flow / M1c_flow hubness verdicts | the §3.4 gate is upstream of the exam. |
| Re-run of the 3 noise-limited injected cells | the verdict is **monotone** under any reclassification of them; no outcome could change it (C5, C6; §9). |

---

## 9. Integrity of the pre-registration

- `gates.yaml` was frozen **before any candidate computation** and **no frozen number
  was changed at any point**: §3.4's 0.9/10%, the `max|·|` amplitude statistic, the
  all-graded-must-pass aggregation, §3.2's 40%/30% caps, §4.1's kill thresholds,
  §1.4's hubness thresholds, §7's deltas, `gates.yaml` itself.
- The two open Phase-1 calibrations were **derived** from §1.2's in-document
  5th-percentile convention and frozen in their own commit **before the exam**, with
  the construction **pre-declared before the numbers were computed**.
- The §3.4 second construction's amplitude ladder, oracle, and verdict-rung rule were
  **pre-declared before any corrected probe flow was computed**;
  `select_verdict_rung()` reads **only** the oracle simulation and the frozen-gate
  definedness — the metric's recovered parameters are **not an argument to it**.
- Six advisor consultations are logged (`CONSULTATIONS.md`, C1–C6) with question,
  direction, and action. **No consultation overrode a frozen rule.** Three errors of
  the executor's own were caught and are logged as corrections rather than silently
  repaired: a retroactively-logged consultation (C2), and three count misreports in
  the §3.4 summary (C6).
- Bugs found and fixed during the run, two of which had been **flattering** the
  metric: a z-scale mismatch in the reversal descriptor leg, a unit-mixing degeneracy
  in the compound probe (e⁶ ≈ 403× zoom), a NaN→FAIL §1.5 violation, and a reversal
  definedness-mask misalignment.

## Artifacts

`outputs/eval/workbench/{step0,e0,e1,phase1}/` · `CONSULTATIONS.md` ·
`IDEAS_NEXT_CYCLE.md` (non-authoritative parking lot; referenced by no record and
gating nothing).
