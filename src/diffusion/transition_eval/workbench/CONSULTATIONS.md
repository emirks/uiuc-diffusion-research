# Advisor consultation log (OPERATIONS §8)

Every consultation: timestamp, question asked, direction received, action taken.
The advisor steers EXECUTION. It cannot override a frozen gate, kill rule,
threshold, or adoption rule, authorize a rescue variant, or make the §7 call;
those are owner-reserved and stop the track. Advisor input never softens or
reframes a committed record — verdicts stay rule text + numbers.

**Channel note (2026-07-14 10:55).** The Claude Code `advisor` tool is
unavailable in this session (API-level). The advisor channel is therefore
implemented as a Fable reasoning consult through the Agent tool, given the
RUNBOOK, OPERATIONS, the deployed seams, and the run's intermediate results.
Same may/may-not boundary; same logging requirement.

---

## C1 — 2026-07-14 11:00 · pre-work plan + step-0 findings

**Asked.** (Q1) RUNBOOK A4: which reading of "within-stratum recall" gates §3.6,
given that the drafted 0.62/0.44 reproduce under NO definition I can construct,
and is that a stop? (Q2) endpoint/whitening mechanics — RUNBOOK §1.2 ("mean
whitened embedding") vs OPERATIONS §5 (`endpoint_vecs` then whiten), which
differ by a nonlinear L2-renorm. (Q3) GPU cache-build design: dense-flow storage
(~15 GB) vs downsampling; ZCA fit population. (Q4) run E1 immediately after the
cache job, before Phase 1's acceptance tests?

**Direction received.**

- **Q1 — pin definition (iv), `macro_per_class_eligible`; proceed on artifact
  truth; NOT an owner-stop.** A4's "the same stratum-grouping code that will
  judge the candidates" is a *symmetry* requirement (identical code both sides),
  not a pointer to `diagnostics.tag_accuracy`. §3.5 says the incumbent's
  *recorded* stratum recalls, and the only stratum recall the artifact actually
  recorded is `o7_conditional.camera_stratum_mean_recall` — which (iv)
  reproduces bit-for-bit. RUNBOOK §6 ("n≤3: reported, never gating") positively
  forbids the clip-pooled variants, which let ineligible clips move a gating
  number. 0.62/0.44 were never validly pinned — A4 pre-registered their
  replacement before any candidate ran, so this is A4 operating as designed (the
  A1 pattern), not baseline re-pinning. Requirements: freeze the definition and
  the targets in gates.yaml before any candidate descriptor is computed;
  enumerate the eligible class lists as frozen corpus facts; record all five
  definitions; pin the NaN-support edge case pre-run; disclose the 0.62/0.44
  finding loudly in baselines.json, the step-0 commit, and WORKBENCH_REPORT.md.
  Hard line carried forward: if a candidate's §3.6 verdict flips between (iv)
  and any recorded alternative, the frozen (iv) cell decides the verdict and the
  sensitivity is reported neutrally beside it — re-adjudicating the cell choice
  after candidate numbers exist is owner-only.
- **Q2 — follow OPERATIONS: `endpoint_vecs` verbatim, whiten after.** Not a
  science conflict: §1.2's phrase is underdetermined on renorm, and the
  conditioned windows *are* the flanking stable frames outside the S-mask.
  `endpoint_vecs` is the deployed endpoint definition everywhere the certified
  instrument reasons about endpoints; a divergent workbench anchor would be
  worse than the renorm wart. The renorm cancels in E1 (no anchors in the
  formula) and is self-consistent in the min-D guard; it bites only E2's
  coordinates. Plus two E1 choices to pre-commit: (1) pool v_null over **the
  clip's own core frame indices**, not the null's own S-mask (a lerp's envelope
  stays ≥ threshold throughout, so its strict core is degenerate by construction
  and the fallback valley would pick an arbitrary sliver); (2) low_D excludes
  from E2's D-normalized channels, NOT from E1 (E1's delta contains no D).
- **Q3 — dense float16 flow at full res, ~15 GB, `savez_compressed`.** Anything
  less makes the §3.3 ε recalibration a GPU re-run, which is what the front-load
  rule exists to prevent. **Trap flagged:** torchvision RAFT needs H,W divisible
  by 8, and the deployed `RAFTExtractor._resize` rounds to /8 **only when it
  downscales**; decoding at short_side=320 gives 320×427 and passes through
  unrounded. Feed native 480×640 with `max_size=432` — the wrapper's own
  arithmetic then yields exactly 320×432. Cache all 120 pairs per clip (seam
  exclusion is descriptor-time, not cache-time). Pin resolution, weights enum,
  `num_flow_updates`. Never edit `signals/flow.py`; batch in `flowcache.py` if
  needed. **ZCA fits on S-mask core frames** — §1.1's own parenthetical ("S-mask
  frame selection") pins it; fitting on all frames would be quiet drift on a
  frozen scientific choice. Deterministic eigenvalue floor, recorded, frozen
  pre-candidate. Nulls: deployed `make_lerp` verbatim, endpoints decoded at
  short_side=256, lerp null for one-sided clips too (not `make_static_hold`),
  own `lerpnull-v1` tag into `$WB_CACHE`; make the two-cache split structural so
  a corpus-side miss inside the GPU job still raises rather than writes.
- **Q4 — E1 immediately after the cache job: endorsed.** §6 sanctions
  interleaving; E1 gates ~4 days of downstream work and depends only on nulls +
  ZCA + the frozen kernel.
- **Step-0 stress tests.** Stub extractor confirmed necessary (`need_frames=False`
  alone does NOT protect the cache: on a miss the pipeline decodes and
  `array_features` computes *and writes*). Add a before/after cache
  `mtime_ns` audit so non-pollution is a recorded fact, not an argument. Pin BLAS
  threads for the bitwise round-trip as a pre-declared determinism control.

**Action taken.** All directions adopted; none required overriding a frozen rule.
gates.yaml pins definition (iv) with targets 0.34623 / 0.03427, the
`misretrieved` convention, the NaN-support convention, and the two E1 choices,
all before any candidate computation. baselines.json enumerates the frozen
eligible class lists and records all five stratum definitions plus the
0.62/0.44 disclosure. bundles.py gains the cache mtime_ns audit and the BLAS
thread pin (the round-trip had already come out bitwise-exact without it; the
pin is now declared rather than discovered). The ZCA OOD concern is parked in
IDEAS_NEXT_CYCLE.md, unregistered and non-authoritative.

---

## C2 — 2026-07-14 11:40 · the §1.4 hubness gate has no numbers

**LOGGED RETROACTIVELY 2026-07-14 12:20, disclosed.** This consultation happened
and its direction was acted on (gates.yaml, commit 694afc7), but it was not
written into this log at the time — a §8 violation, recorded here rather than
quietly backfilled. Nothing below is reconstructed from memory: it is the
consultation as it occurred, and every number in it is reproducible from
`outputs/eval/workbench/step0/hubness_incumbents.json`.

**Asked.** §1.4 mandates a hubness gate, names its two statistics, and names the
failure condition ("a sink column (M1c/polygon pattern)") — but states NO
numbers, and I must freeze them before any candidate runs. (1) Do both statistics
gate, or does k-occurrence skew merely report while the prediction column gates?
(2) Are the thresholds 3.0 / 0.70 / 0.25 acceptable? (3) Is k=10 right, given mean
class size ~6? And the boundary question: is this a threshold only the owner may
set, or an open calibration §1.4 delegates?

**Direction received.** DELEGATED, not owner-reserved: prohibition 5 reserves
thresholds gating an outcome the RUNBOOK did not register, whereas §1.4 registers
this gate completely at the level a pre-registration operates (mandatory, both
statistics named, failure condition named, exemplar named) — what is open is the
numeric operationalization, structurally identical to §3.3's ε percentile, which
OPERATIONS §8 lists verbatim as advisor-appropriate. BOTH statistics gate: §3.6's
kill rule says "if M1c_flow still exhibits a HUB after the energy gate → the
descriptor is dead", which is k-occurrence language and would have no operational
referent if skew only reported; and the two are not redundant, because the
prediction column is a k=1 phenomenon (rank-2..10 absorption casts no 1-NN votes,
which is exactly how the incumbent m1b decouples: skew 2.52 but entropy 0.917).
The three numbers are the MIDPOINTS of the empty gap between the pass and dead
populations — and gap-midpoint placement is already a deployed convention in the
certified instrument (`probes.grade_splices` sets τ_copy at the midpoint between
known-positive and known-negative populations), so freeze the DERIVATION RULE, not
three free parameters. Where the gate sits is DERIVED FROM FROZEN TEXT, not
chosen: §0 calls M1c "hub-collapsed (polygon column = sink artifact)" and M1b
merely weak, so the gate must fail m1c/MFS and pass m1a×3/m1b. k=10 endorsed
(it deliberately reaches past the class boundary and measures beyond-class
absorption — the polygon failure exactly); persist k ∈ {1,5,10} as non-gating
diagnostics, only k=10 gates. Loss asymmetry accepted deliberately: a false KILL
is terminal and irreversible (§3.6/§9), a false PASS is caught in owner-side
review — so never tighten these, and never build a carve-out (that would be a
changed bar form, owner-reserved).

**Action taken.** All adopted. gates.yaml (694afc7) freezes the derivation rule,
both populations, the calibration band, and the four attached requirements
(regression fixture, gap-band disclosure, coverage interaction with NO coverage
correction, gate applies to appearance candidates too). The frozen gate reproduces
§0's diagnosis on all six incumbents; tests/test_workbench.py asserts it as a
regression fixture.

---

## C3 — 2026-07-14 12:20 · the §4.1 kill fired, and the control is dead too

**Asked.** E1 fails BOTH §4.1 kill conditions (d 0.358 vs 1.522; 209/223 vs
73/223; accuracy 0.0628 against chance 0.067) at full coverage 1.0, and
independently fails §1.4 hubness. But a below-chance result has the shape of a
plumbing bug, so I ran the control — the identical representation with NO null
subtraction, i.e. containing zero endpoint-normalization:

    RAW v_clip, no subtraction ....... acc 0.6054  d 0.9875  mis  88/223  entropy 0.882
    WHITENED v_clip, no subtraction .. acc 0.0628  d 0.1068  mis 209/223  entropy 0.042
    WHITENED delta (the candidate) ... acc 0.0628  d 0.3582  mis 209/223  entropy 0.322
    RAW delta (unwhitened) ........... acc 0.1345  d 0.6211  mis 193/223  entropy 0.524

The whitened control is as dead as the candidate: §1.1's whitening destroys the
representation before the delta is formed. Mechanism verified — DINO's core-frame
covariance spans λ 3.637e-2 … 1.848e-9, and the frozen floor (1e-6·λ_max) floors
only 1 of 768 dims, so ZCA amplifies ~700 near-null directions by up to ~2e4.
`eig_floor_ratio` is FROZEN in gates.yaml and I did not touch it. Is this an
instrument defect (blown experiment) or a valid negative? Does the kill bind?
May I record a floor-sensitivity curve?

**Direction received.**
- **The kill is PROCEDURALLY VALID and FINAL.** The registered candidate is not
  "the delta idea" — it is the delta *in the registered pipeline*, which includes
  §1.1 whitening and the floor frozen at 694afc7. This is not a §3.4
  implementation bug: whitening.py's own docstring names the hazard, the code does
  what it documents, and the frozen parameter behaved as frozen. The specification
  was inadequate; the code was not wrong. There is therefore NO fix-and-rerun
  authority — changing the floor is prohibition 5.
- **The four-arm table is legitimate diagnostic FACT, but "therefore the kill says
  nothing about the hypothesis" is an INFERENCE about a fired kill rule —
  owner-reserved.** Neither executor nor advisor makes it. The in-family template
  is gates.yaml's own `band_disclosure`: frozen verdict stands, terminal, no
  executor re-adjudication; the report flags the sensitivity, prints the evidence,
  and assigns the question to owner-side review.
- **E2/E3 DO NOT RUN, and are recorded as killed-per-rule, NOT as
  blocked-pending-owner.** A confounded-but-as-specified run is a valid run of the
  registered procedure; pre-registration's escape hatch is disclosure plus owner
  review, never executor-side invalidation.
- **E0 (§4.2) is still owed** — it "rides the E1 run, no gating numbers" and is NOT
  conditioned on E1 passing.
- **Do not repeat the inference** ("a kill test that also kills its own control has
  not tested its hypothesis") in any record. It is already committed inside
  e1_cause_diagnostics.json's `finding` field; that artifact is not rewritten, but
  the sentence is not carried forward. The facts carry themselves.
- **The floor-sensitivity sweep is AUTHORIZED as labelled instrument diagnostics**
  — OPERATIONS §8's completeness mandate ("the reviewer never needs a re-run")
  affirmatively requires this evidence to exist now — under mandatory guardrails,
  the first of which is that the grid is PRE-DECLARED IN THIS LOG BEFORE IT IS
  COMPUTED.

**PRE-DECLARED SWEEP GRID (declared here BEFORE computing; this declaration is what
makes it a diagnostic rather than a search):**

    eig_floor_ratio ∈ {1e-6 (REGISTERED), 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}

Only this scalar is swept — the one parameter the RUNBOOK left free. The ZCA fit
population is NOT swept (RUNBOOK-pinned by §1.1's parenthetical, ruled in C1,
parked in IDEAS_NEXT_CYCLE — sweeping it would relitigate a frozen scientific
choice). No new whitener families (shrinkage etc. = new instrument design →
IDEAS_NEXT_CYCLE). Two arms per floor (whitened no-subtraction control, whitened
delta), with the raw arms as floor-independent reference rows. NO verdict column,
no pass/fail language anywhere in the artifact; the 1e-6 row is marked REGISTERED
and is the only row that is the §4.1 candidate. Nothing from the sweep feeds E2/E3
or any gate. The line not crossed: no floor is recommended or selected, no sweep
row is scored against §4.1/§1.4/§7, no downstream rung runs on a sweep
configuration, gates.yaml is not touched, and the recorded verdict is not revised.

**Action taken.** Kill recorded as final and terminal (72a5bd4). E2/E3 not run.
E0 produced. Sweep computed under the guardrails above and committed as
diagnostics. Owner-reserved blocker escalated with provenance: RUNBOOK §1.1
mandates whitening but does not pin its regularization; `eig_floor_ratio` was an
executor-chosen free parameter, frozen in good faith at 694afc7 before any
candidate ran. Phase 1 continued without waiting — it is an independent track.

---

## C4 — 2026-07-14 12:45 · the two open Phase-1 calibrations, and a coverage collapse

**Asked.** §3.2's low-texture gate and §3.3's energy gate are both mandated by the
RUNBOOK with NO numbers, and the driver refuses the Phase 1 exam until both are
frozen. (1) Is §1.2's in-document 5th-percentile convention the right derivation
for both? (2) Before either open threshold is set, §3.2's PINNED rules (40% inlier
floor + 30% clip cap) already leave only 143/223 clips defined — coverage 0.641 vs
the incumbent's 0.9686/0.9955, with 16 clips having no defined core pair at all. Is
that a bug, a reportable property, or an owner-reserved blocker? (3) Should the
exam run at all at this coverage? (4) What else to record now?

**Direction received.**
- **Both slots are DELEGABLE** (C2's boundary test: the gates are registered —
  mandatory, named, purpose stated, §3.3 even names its exemplar failure; only the
  numeric operationalization is open, and OPERATIONS §8 lists "§3.3's ε percentile"
  verbatim as advisor-appropriate).
- **Adopt §1.2's 5th-percentile convention for both** — it is the RUNBOOK's only
  in-document precedent for turning a corpus distribution into a floor, and it is
  already frozen as `min_d.percentile: 5.0`. The alternatives are rejected on frozen
  text: a C2-style gap-midpoint is unimplementable (no labelled no-object-motion
  population exists; the 6 camera-tagged-without-object classes are NOT one, because
  their effects still move — a mislabelled negative is worse than none), and a
  noise-floor estimate is a new instrument-design element found nowhere in frozen
  text (→ IDEAS_NEXT_CYCLE, not this cycle). §3.3's own text pins the construction
  family — "set ε from the corpus residual-magnitude distribution (report chosen
  percentile)" — so the only open parameter is the percentile.
- **ε MUST BE CALIBRATED ON THE POPULATION IT OPERATES ON.** The texture gate is
  upstream of the energy gate (§3.2 definedness feeds m1c's frames; "energy gate
  first" is first *within* m1c). So the p5 quoted from the pre-texture population
  (5.5e-5) is not the right realization.
- **The coverage collapse is an honest measured property of PINNED rules** — not a
  §3.4 fix-or-stop (nothing failed, nothing is buggy: the estimator is the
  registered one, the no-spatial-mask fact is a certified-tree property, and the
  gate refusing large-effect frames is the gate working as designed), and not
  owner-reserved (every number involved — 40%, 30% — is RUNBOOK-pinned; a computed
  consequence of frozen rules is a fact, not a decision). Report it.
- **TWO CORRECTIONS to my framing.** (a) §7 is the APPEARANCE adoption rule; Phase
  1's gate is §3.6, which has NO coverage condition. Coverage enters Phase 1 only
  through the frozen conventions (an uncovered row is a misretrieved one; an
  eligible class with zero covered clips contributes 0.0 to the macro stratum
  recall) and through §1.5's report-beside-accuracy discipline. (b) Coverage NUMBERS
  are computable facts; **"not materially narrower" is NOT computable** — no frozen
  materiality threshold exists, and inventing one would itself be prohibition 5.
  Report 0.583 / 0.529 beside 0.9686 / 0.9955 and stop there. Those words never
  appear with a verdict attached.
- **Proceed to §3.4 and, if both probes pass, the exam.** No frozen rule conditions
  the exam on pre-exam coverage; stopping on coverage would be an executor-side
  inference about an outcome, which C3 prohibited. The frozen conventions already
  price definedness into §3.6's own numbers without anyone editorializing.

**PRE-DECLARED ε CONSTRUCTION (declared here BEFORE the number is computed — the
declaration is what makes this a derivation and not a search):**

    1. freeze min_pair_texture = p5 of the corpus pair-texture distribution,
       taken over CORE PAIRS ONLY (both frames core; seams excluded) = 0.0753.
       This gate applies to BOTH m1b and m1c (it is a §3.2 frame-definedness rule
       and m1c inherits frame definedness).
    2. THEN recompute the residual-energy distribution over core frames that
       survive BOTH the camera fit (inlier >= 40%) AND the texture gate
       (pair_texture >= 0.0753), and freeze
           energy_gate_epsilon = p5 of THAT population
           energy_gate_percentile = 5.0
    3. Report both the pre-texture realization (5.5e-5) and the frozen post-texture
       one, so the construction is auditable.

**Action taken.** All adopted. Both thresholds frozen in their own commit before any
exam. Coverage waterfall, the tex x eps grid (as a no-verdict diagnostic, disclosed
as computed BEFORE the freeze, with the threshold derived from §1.2's convention
rather than selected from the grid), the per-clip definedness report, the pre-exam
enumeration of zero-covered eligible stratum classes, the Huber breakdown table as
a committed artifact, and the incumbent-vs-candidate definedness overlap are all
recorded. Phase 1 proceeds to §3.4 and the exam.

---

## C5 — 2026-07-14 13:10 · §3.4 acceptance FAILED: my probe, or the metric?

**Asked.** Both §3.4 tests failed as first constructed (committed unmodified at
f5d2790). Injected: zoom/rotate pass, translation fails (amp_err 0.14–0.55),
pan_zoom catastrophic (2.3–82×). Reversal: 10/101. Is this the metric failing
constructed truth (→ STOP, no second attempt) or a defect in my probe construction
(→ fixable; debugging is not a rescue variant)? May I change the injected amplitude,
the amp_err statistic, the sensitivity filter?

**Direction received.** FIXABLE BRANCH — probe construction, not the metric. And the
advisor found **two defects I had missed**, both of which had been *flattering* the
metric:
- **The z-scale bug.** Leg B compared `m1b_z[i]` (corpus z-scored) against
  `rdesc["curve"]` (raw — `clip_descriptor` never applies the scaler; only
  `corpus_descriptors` does). Raw-vs-z-scored is a meaningless distance that happens
  to be finite. Mandatory fix.
- **pan_zoom was physically degenerate.** One scalar applied across channels with
  different units put 0.3×20 = 6.0 into LOG-scale — e⁶ ≈ **403× cumulative zoom**.
  Its late frames were 400× magnifications of nothing, and its "translation" truth
  was dominated by scale-coupling. The 2.3–82× errors were the probe grading its own
  degeneracy.
- **The reversal failure profile was not what I said.** 101/101 graded clips pass the
  threshold-free closer-to-negated check (margins 15–350×); the dominant failure is
  68/101 NaN `d_self_vs_reversed` and 65/101 NaN `median_within_class` being
  converted NaN→False→FAIL. That is a §1.5 violation ("undefined ≠ zero"), not a
  metric failure.

Rulings: `max|·|` LEFT ALONE (changing a grading statistic after watching it fail is
goalpost-moving). §3.4's 0.9/10%, the §3.2 gates, the aggregation rule and gates.yaml
all UNTOUCHED. NaN → ungradable-with-reason, counted, never failed. Reversal
sensitivity uses the certified Bar-5 screen — z-unit banded DTW of a trajectory vs
its own negated-reverse, floor **0.5 INHERITED VERBATIM from certify/bars.yaml**
(§3.4 itself says the probe is "inherited from old Bar 5", so completing the
inheritance is implementing the spec, not choosing a number), plus the same statistic
at channel granularity. Border mirror pixels excluded through §3.2's own "fit on its
complement" pathway, the invalid set known exactly from the warp. **ONE corrected
construction, ONE run**; if it fails, the presumption is STOP.

---

## C5-R — 2026-07-14 13:35 · RECONCILE: the rule contradicted its own rationale

**Asked (before running anything).** C5's derivation rule — "amplitude peak = p50 of
the corpus per-pair |Δc|" — yields tx 0.297 px/pair, which is essentially the
*original invented* amplitude (0.314 px/pair). It does not raise the SNR at all,
which was the rationale the fix rested on. Applied literally it reconstructs the
failing probe and would kill Phase 1 on a probe artifact. p90 would give 3.11 px/pair
and the SNR the rationale assumed. But "advisor said p50, executor prefers p90
because p50 would fail" is exactly the shape of goalpost-moving — so I brought it
back rather than choosing.

**Direction received — the error was the advisor's, and the fix is better than either
option.** The decisive fact is not about the metric at all: **at p50 a PERFECT
estimator fails too.** With per-pair parameter noise σ ≈ 0.02–0.05 px, `max|·|` over
~120 pairs adds ≈ 2.5–3σ at the peak; against a 0.297 px peak that is amp_err
≈ 0.2–0.5 for *any* estimator whose error is the flow's own — including an oracle
that fits the flow exactly. **A test an oracle fails is not a test of M1b; it grades
the instrument's noise floor.** So the quantile stops being a judgment call: build a
LADDER and make the probe *prove* it can test what §3.4 asks.

**PRE-DECLARED, WITH EVERY REALIZED NUMBER, BEFORE ANY CORRECTED PROBE FLOW WAS
COMPUTED (this declaration is what makes it a derivation and not a search):**

Corpus per-pair |Δc| quantiles (defined + texture-gated core pairs of camera-tagged
clips, n = 3,566):

| channel | p50 | p75 | p90 |
|---|---|---|---|
| tx | 0.296625 | 0.650124 | 3.113154 |
| ty | 0.421367 | 1.166700 | 2.305041 |
| log_scale | 0.003916 | 0.011151 | 0.024682 |
| rotation | 0.000894 | 0.002165 | 0.004176 |

Ladder amplitudes (total, per channel, own units; ×76.394 = (T−1)/(π/2)):

| rung | tx | ty | log_scale | rotation |
|---|---|---|---|---|
| p50 | 22.6605 | 32.1901 | 0.2992 | 0.0683 |
| p75 | 49.6658 | 89.1293 | 0.8518 | 0.1654 |
| p90 | 237.8274 | 176.0922 | 1.8856 | 0.3190 |

Measured σ per substrate × channel (identity control — the base clips' own corpus
fits; it OVER-estimates, since the "most-static" clips are not perfectly static,
which is conservative): tx median 0.01743 (range 0.0064–0.0233), ty 0.01460
(0.0092–0.0315), log_scale 0.00011, rotation 0.00007.

**The noise-limited ORACLE (200 draws): `recovered = truth + N(0, σ)` pushed through
`grade_injection` UNCHANGED** (same `max|·|`, same frozen 0.9/10%). A cell is
oracle-valid at rung q iff the MEDIAN draw has amp_err ≤ 0.10 AND corr ≥ 0.90 — no
new threshold is introduced. Result, confirming the diagnosis: **at p50 the oracle
FAILS** on pan_x for 5/8 substrates, rotate for 6/8 and pan_zoom for 6/8; at p75 and
p90 it is valid nearly everywhere.

**VERDICT RUNG, per (substrate, kind) cell:** the HIGHEST rung that is *both*
oracle-valid *and* DEFINED under the frozen §3.2 gates. If no rung qualifies, the
cell is excluded-with-reason (as wireframe_2 already was). **THE ANTI-GAMING GUARD IS
STRUCTURAL:** `select_verdict_rung()` takes as input ONLY the oracle simulation and
the frozen-gate definedness — the metric's recovered parameters are not an argument
to it and cannot be. That is what makes "highest valid rung" a construction-validity
rule rather than "the rung where it passes". Every rung of every cell is graded and
reported regardless; the verdict is taken at the declared cell and the rest is the
amplitude-resolution curve.

**PRE-COMMITTED NOW, so it cannot be invoked opportunistically later:** if the
verdict-rung cells FAIL, that is the metric failing constructed truth → STOP, no
further construction. If a post-hoc oracle sim on the NEW probes' own σ shows a
verdict cell was noise-limited after all, that is a CONSTRUCTION failure and returns
to the fork — it does NOT convert into a metric failure.

**A FIRST-CLASS PHASE-1 FINDING, to be recorded beside coverage (facts only, no
inference — the inference is owner-side):** in this corpus the MEDIAN per-pair camera
translation (tx 0.297 px, ty 0.421 px) is within an order of magnitude of the flow
fit's own per-pair parameter noise (σ ≈ 0.015–0.017 px), while the vigorous decile is
tx 3.11 px / ty 2.31 px. An oracle fails §3.4's peak-amplitude criterion at p50.

**Action taken.** Ladder + oracle implemented and pre-declared (probe_ladder.json).
Probes rebuilt at all three rungs with per-channel amplitudes in their own units, the
rung in the cache key (a key that ignored it would silently pair the first
construction's flow with the second's ground truth), and exact border-validity masks.
`max|·|`, 0.9/10%, the §3.2 gates, the aggregation rule and gates.yaml: untouched.
