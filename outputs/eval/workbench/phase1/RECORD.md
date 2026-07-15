# Phase 1 — motion (M1b_flow / M1c_flow) · §3.4 ACCEPTANCE FAILED · TERMINAL

Run 2026-07-14. Gates frozen at `694afc7` (before any candidate) and `8c38833`
(the two Phase-1 calibrations, before the exam). Backbone pinned at `970ed9f`.

---

## 1. The rules that fired

RUNBOOK §3.4, verbatim:

> **1. Reversal probe** (inherited from old Bar 5): time-reversed reference must be
> distinguished from the original — camera parameter trajectories negate; descriptor
> distance to reversed self must exceed the median within-class distance.
> **2. Injected-trajectory recovery:** apply known synthetic pans/zooms/rotations to
> static clips; recovered parameter trajectories must match ground truth
> (per-parameter correlation ≥ 0.9 and relative amplitude error ≤ 10%).
>
> **Failure of either = fix or stop; the exam is not run on a metric that fails
> constructed truth.**

Aggregation (frozen, unchanged across both constructions): all graded cells must
pass. RUNBOOK §3.6: *"Misses → stays analysis-tier. **No second attempt this
cycle.**"* §9: no second motion attempt.

C5-R pre-commitment, quoted verbatim (made **before** the second construction ran):

> If the verdict-rung cells FAIL, that is the metric failing constructed truth →
> STOP, no further construction. If a post-hoc oracle sim on the NEW probes' own σ
> shows a verdict cell was noise-limited after all, that is a CONSTRUCTION failure
> and returns to the fork — it does NOT convert into a metric failure.

**VERDICT: §3.4 FAILS. The exam is NOT run. Phase 1 is closed.**

---

## 2. Two constructions; both are in the record

**First construction** — committed unmodified at `f5d2790`. It failed, and three
defects in it were identified afterwards:
- a **z-scale bug**: the reversal descriptor leg compared a corpus-z-scored curve
  against a raw one (`clip_descriptor` never applies the scaler; only
  `corpus_descriptors` does). Raw-vs-z-scored is a finite but meaningless distance,
  and it *flattered* the metric;
- **pan_zoom unit-mixing**: one scalar applied across channels with different units
  put 0.3 × 20 = 6.0 into **log**-scale — e⁶ ≈ **403× cumulative zoom**;
- a **§1.5 violation**: 68 undefined descriptor distances were converted NaN→False→FAIL.

**Second construction** — per C5-R. Amplitudes **derived** from the corpus (not
invented), a pre-declared ladder {p50, p75, p90}, a noise-limited **oracle**, exact
border-validity masks, the inherited Bar-5 sensitivity screen (floor **0.5**, taken
verbatim from the certified `bars.yaml`).

**Never touched across either construction:** the §3.4 thresholds (corr ≥ 0.9,
amp_err ≤ 10%), the `max|·|` amplitude statistic, the all-graded-must-pass
aggregation, §3.2's 40%/30% caps, the texture gate, `gates.yaml`.

**The anti-gaming guard is structural.** `probe_ladder.select_verdict_rung()` takes
as input **only** the oracle simulation and the frozen-gate definedness. The metric's
recovered parameters are not an argument to it and cannot be.

---

## 3. Test 2 — injected-trajectory recovery: **FAIL**

8 substrates × 5 kinds. `wireframe/wireframe_2.mp4` **excluded with reason** (100%
of its core pairs sit below the frozen texture gate; the metric itself declares it
undefined, so it is not a valid constructed-truth substrate). 35 verdict cells, all
at rung **p90**.

**29 / 35 PASS** — corr **0.9618 – 0.9998**, amp_err **0.0004 – 0.0996**. (One
passing cell sits within 0.0004 of the 0.10 bound.)

**6 / 35 FAIL:**

| substrate | kind | failing channel(s) | amp_err | post-hoc oracle |
|---|---|---|---|---|
| wireframe_5 | pan_x | tx | 0.121 (corr 0.982) | **INVALID** → construction |
| wireframe_5 | pan_y | ty | 0.180 (corr 0.965) | **INVALID** → construction |
| gas_transformation_4 | pan_zoom | ty | 0.192 | **INVALID** → construction |
| **mystification** | **zoom** | log_scale | **0.118** (corr 0.996) | **VALID** → metric |
| **mystification** | **pan_zoom** | ty, log_scale | **0.117 / 0.138** | **VALID** → metric |
| **saint_glow** | **pan_zoom** | ty | **0.119** | **VALID** → metric |

**Post-hoc oracle** (the pre-committed instrument; σ re-estimated from each probe's
own realized noise, then `recovered = truth + N(0,σ)` pushed through the **unchanged**
grader): three failing cells are at rungs an oracle **could not** pass either
(construction), and **three fail at rungs the oracle passes with margin** —
mystification zoom (metric 0.118 vs oracle median 0.055), mystification pan_zoom
(0.117/0.138 vs 0.069/0.084), saint_glow pan_zoom (0.119 vs 0.067).

Under the C5-R pre-commitment those three are **the metric failing constructed
truth**, and the frozen all-graded-must-pass aggregation makes **test 2 FAIL on
validly-constructed cells alone**.

**The verdict is monotone under any reclassification of the three noise-limited
cells:** reclassifying any as metric failures adds failures; excluding all three
leaves 3 failures among 32 validly-constructed cells. Both directions STOP. This is
stated because two of the noise-limited calls are razor-thin (post-hoc oracle median
amp_err **0.1025** for gas_transformation_4 ty and **0.1047** for wireframe_5 tx,
against the 0.10 line). **The verdict does not depend on that classification.** The
three noise-limited cells are therefore **not re-run**: no outcome of that work could
change the verdict.

Over all 35 verdict cells, 3 of the 29 **passes** also sat at rungs the realized σ no
longer supports. A pass is still a pass and is not relabelled; the fact is recorded.

**σ-estimator fact.** C5-R predicted the identity control would *over*-estimate σ. It
under-estimated it for the warped probes: identity-control σ_tx median **0.0174 px**
vs realized **0.187 px** for wireframe_5 tx at p90 (valid area 0.63).

---

### 3b. Item-5 readout — which channels the 3 oracle-valid failures failed on

Added by the E1′ cycle's Part-1 amendment. **Pure readout of the persisted
`acceptance.json` + `posthoc_oracle.json`; no re-run, no recomputation, no new
probe.** Persisted as `item5_channel_readout.json`.

| cell | failing channel(s) | corr | amp_err | oracle median amp_err |
|---|---|---|---|---|
| mystification_5 · zoom | `log_scale` | 0.9962 | 0.1182 | 0.0554 |
| mystification_5 · pan_zoom | `ty`, `log_scale` | 0.9898 / 0.9917 | 0.1172 / 0.1376 | 0.0691 / 0.0844 |
| saint_glow_3 · pan_zoom | `ty` | 0.9940 | 0.1187 | 0.0674 |

Channel tally across the three cells: **`ty` x 2, `log_scale` x 2**. No `tx`
failure and no `rotation` failure among them.

**Every failure is an AMPLITUDE failure.** On all four failing channel-cells the
correlation floor (>= 0.90) is met with margin 0.9898-0.9962, and only the
relative-amplitude bound (<= 0.10) is breached (0.1172-0.1376). Across all SIX
failing cells (including the three noise-limited ones) the lowest correlation on
any failing channel is **0.9655**, and every failure is `amp_ok = false`; not one
is a correlation failure.

Stated without interpretation: the fitter recovered the SHAPE of every injected
trajectory it was graded on, and missed its SCALE by 12-18%. No mechanism claim
is attached. The §3.4 verdict is unchanged and remains terminal.

## 4. Test 1 — reversal: **FAIL**

Waterfall: **102** camera-tagged clips → **12** insensitive (Bar-5 screen, DTW 0.000
< floor 0.5, inherited verbatim from `bars.yaml`) → **57** undefined-ungradable
(§1.5: counted, never failed; single reason — "descriptor or within-class median
undefined") → **33 graded**.

| leg | form | result |
|---|---|---|
| **A1** closer-to-negated (threshold-free) | is the reversed fit closer to the NEGATED trajectory than to the direction-blind one? | **33 / 33** |
| **A2** negation-correlation floor (0.90) | per Bar-5-sensitive channel | **17 / 33** |
| **B** descriptor distance | d(clip, its reverse) > median within-class distance | **22 / 33** |
| **joint row pass** | A ∧ B | **11 / 33** |

**Leg B fails on 11 of 33 gradable clips** (ratio d_self_vs_reversed / median
within-class = **0.360 – 0.880**). Leg B's form is frozen verbatim
(`gates.yaml: descriptor_distance_exceeds: median_within_class_distance`). The
construction was re-checked for a third defect and none was found: the reversed flow
is extracted from actually-reversed frames, both descriptors now pass through the
same corpus scaler, the within-class median uses finite same-label distances, and the
§1.5 NaN discipline is correct. **Reversal FAILS on leg B alone, independent of A2.**

**Leg A2 — disclosed, not adjudicated.** The 0.90 floor is frozen in `gates.yaml`
only under `injected_trajectory`; applying it to reversal negation correlations is an
executor operationalization, applied to real-clip channels whose amplitudes are
uncontrolled. Of 132 graded clip×channel cells, 20 fall below the floor; by channel:
**rotation 11, ty 5, log_scale 3, tx 1**. Context: rotation is the corpus's
lowest-amplitude channel (per-pair |Δ| p50 = **0.00089**). No oracle-validity guard
was pre-registered for reversal, and constructing one after the leg failed is
precluded by the C5-R pre-commitment. **No mechanism claim is attached.** The A2
sensitivity question is assigned to owner-side review.

First construction's reversal waterfall, for the permanent record: 10/101 under the
pre-fix grader.

---

## 5. Consequences (rule text + numbers)

- §3.4 FAILS → **the exam is NOT run** ("the exam is not run on a metric that fails
  constructed truth").
- No §3.6 adoption question arises: there is no exam result to compare against the
  pinned incumbents (m1b_camera d 0.519962 / camera-stratum recall 0.34623;
  m1c_object d 0.247601 / object-stratum recall 0.03426).
- **M1b_flow and M1c_flow remain analysis-tier.** No second attempt this cycle
  (§3.6, §9).
- **Phase 1 is closed.** The appearance track (Phase 2) is separately closed by the
  §4.1 KILL recorded at `72a5bd4`.

## 6. Recorded pre-exam facts (computed before any distance existed)

- Coverage under the frozen gates: **m1b_flow 130/223 (0.5830)**, **m1c_flow 118/223
  (0.5291)**, against incumbent m1b_camera 0.9686 and m1c_object 0.9955. The dominant
  loss (223 → 143) comes from RUNBOOK-**pinned** rules (40% inlier floor, 30% clip
  cap), before either open threshold was set.
- Mechanism, measured on the corpus: effect-area fraction median **0.875** on
  undefined core frames vs **0.160** on defined ones.
- §3.6 stratum-recall ceilings implied by coverage alone: camera **0.3571** (9 of 14
  eligible classes have zero covered clips; incumbent target 0.34623); object
  **0.6111** (7 of 18; incumbent target 0.03426).
- Corpus camera-motion scale: median per-pair translation **tx 0.297 px / ty 0.421
  px**, within an order of magnitude of the flow fit's own per-pair parameter noise
  (σ ≈ 0.015–0.017 px); the vigorous decile is tx 3.11 px / ty 2.31 px. **An oracle
  fails §3.4's peak-amplitude criterion at p50.**

## Artifacts

`acceptance.json` (second construction) · `acceptance_first_construction.json`
(first, unmodified) · `probe_ladder.json` · `posthoc_oracle.json` ·
`reversal_and_oracle_tables.json` · `definedness_report.json` ·
`calibration_frozen.json` · `camera_fits.npz`.
