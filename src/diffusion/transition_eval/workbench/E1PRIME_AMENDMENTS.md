# Record amendments — E1′ cycle, Part 1

**Authority:** `E1PRIME_DIRECTIVE.md` (owner directive, committed verbatim in this
same commit). **Written and committed BEFORE any E1′ computation**, per the
directive's own ordering ("write these before any computation").

These amendments ADJUDICATE the two matters the previous cycle escalated as
owner-reserved, and enter the owner's own spec errors into the record as design
provenance. The executor did not decide any of them; each is transcribed from the
directive.

---

## A. Escalation (a) — ADJUDICATED (owner)

> Escalation (a) — adjudicated. The E1 KILL verdict binds the candidate-as-specified
> (whitened delta vector under eig_floor_ratio 1e-6). The whitened no-subtraction
> control (acc 0.0628 ≈ chance, vs raw control 0.6054) is entered as evidence of
> instrument failure independent of the hypothesis. The endpoint-normalization
> hypothesis is recorded as unadjudicated, prior lowered (raw-delta collapse 0.1345,
> P1 analogue 8/29, P2 AUC 0.523 are the descriptive wounds). E1-as-specified is not
> re-run under any circumstances.

**Status of the E1 record:** the §4.1 KILL recorded at `72a5bd4` **stands** and is
**procedurally valid**. What changes is its SCOPE: it binds the candidate as
specified, not the hypothesis. The hypothesis is now carried by E1′ (Part 2).

**Executor disclosure attached to this amendment (new, and it bears on one of the
three "descriptive wounds" the owner cites).** The P2 figure quoted above
(AUC 0.523) was computed in `e0_anatomy.py`, whose curves are resampled by
`curves.resample(v[:, None], 64)` — i.e. **each scalar channel is reparameterized by
its own arc length**. For a monotone scalar channel `v(t)`, arc length is
`σ(t) = cum|Δv| / total|Δv|`, and therefore `v(σ) = v₀ + σ·(v_T − v₀)` — **a straight
line, for every clip**. `â` is progress along the chord and is near-monotone by
construction, so E0's `â` curves are near-linear ramps for every clip, and P2's
statistic `mean(â) − mean(1 − b̂)` was taken over them. This is demonstrated, not
asserted: `tests/test_workbench_e1prime.py::test_per_channel_arclength_linearizes_monotone`.
E0 is explicitly descriptive and non-gating, so **no verdict ever rested on it** and
nothing is retracted. But the owner should know that one of the three wounds cited
in the adjudication is measured through a degenerate reparameterization. **P2 is
recomputed in E1′'s registered form** (a single shared σ; see the pre-registration),
and both numbers are reported side by side. **No mechanism claim is attached**, and
this disclosure does not reopen the E1 kill.

## B. Escalation (b) — ADJUDICATED (owner)

> Escalation (b) — adjudicated. Leg A2's 0.90 floor is recorded as an
> invalid-in-context operationalization (amplitude-controlled threshold applied to
> uncontrolled real-clip channels). No mechanism claim attached. Reversal and §3.4
> verdicts unaffected (leg B and the injected test fail independently).

Recorded. The Phase-1 verdict (`2de4835`) is unchanged: reversal fails on **leg B**
alone (22/33) and the injected test fails independently, both without leg A2.

## C. Owner spec errors — entered as design-error provenance for E1

> **a.** The RUNBOOK §3.5 recalls 0.62/0.44 were tag-group accuracies misread as a
> recall definition; superseded by the A4 backfill (0.34623 frozen).
>
> **b.** The derivation "curves are integrals of what delta summarizes, so if delta
> fails the curves must" is wrong: the delta vector is net displacement — out-and-back
> excursions annihilate in it; the signature channels take magnitudes before
> integrating. E1 therefore structurally erased the signal class it proxied. This is
> the premise error under the (valid) E1 kill.

Item (a) confirms amendment A4 (`gates.yaml`, `stratum_targets`) as the governing
definition; the memory figures 0.62/0.44 are demoted, as A4 already recorded.

Item (b) is the premise error that makes E1′ a **different candidate** rather than a
re-run: `‖Σ ρ(t)‖` (what the delta measures) and `Σ ‖ρ(t)‖` (what the m̃ channel
measures) are **not** related by an inequality that lets the first bound the second.
A clip whose residual excursion leaves the chord and returns contributes ~0 to the
delta and its full path length to m̃. This is why the E1 kill does not transfer to
the signature, and it is the entire licence for E1′.

## D. Phase 1 closure — corpus facts recorded as findings

> Phase 1 closure. Corpus facts recorded as findings: motion-scarce corpus (median
> per-pair tx 0.297 px / ty 0.421 px, rotation p50 0.00089), full-frame effects
> (outlier-area median 0.875 on undefined core frames). Consequences: M1b's realistic
> ceiling on this corpus is a presence/absence flag, not a similarity metric;
> M1c_flow moves to the v-next roster with acceptance tests aimed at object-side
> constructed truth (synthetic residual patterns), not camera reversal. No motion
> re-attempt this cycle (§3.6 stands).

Recorded. These are the numbers already committed in `phase1/RECORD.md` §6; the
directive elevates them from "pre-exam facts" to **corpus findings**. No motion code
runs this cycle.

## E. Item-5 readout — the 3 oracle-valid injected-trajectory failures

Pure readout of persisted `acceptance.json` + `posthoc_oracle.json`. **No re-run, no
recomputation, no new probe.** Appended to `phase1/RECORD.md` §3 and persisted as
`phase1/item5_channel_readout.json`.

| cell | failing channel(s) | corr | amp_err | oracle median amp_err |
|---|---|---|---|---|
| mystification_5 · zoom | `log_scale` | 0.9962 | 0.1182 | 0.0554 |
| mystification_5 · pan_zoom | `ty`, `log_scale` | 0.9898 / 0.9917 | 0.1172 / 0.1376 | 0.0691 / 0.0844 |
| saint_glow_3 · pan_zoom | `ty` | 0.9940 | 0.1187 | 0.0674 |

Channel tally across the three cells: **`ty` × 2, `log_scale` × 2**. No `tx` failure
and no `rotation` failure among them.

**The one fact the readout establishes: every failure is an AMPLITUDE failure.**
On all four failing channel-cells the correlation floor (≥ 0.90) is met with enormous
margin (0.9898 – 0.9962) and only the relative-amplitude bound (≤ 0.10) is breached
(0.1172 – 0.1376). Extending to all six failing cells including the three
noise-limited ones, the same holds: the lowest correlation on any failing channel
anywhere is **0.9655**, and every single failure is `amp_ok = false`.

Stated without interpretation: the fitter recovered the **shape** of every injected
trajectory it was graded on and missed its **scale** by 12–18%. No mechanism claim
is attached, and this readout changes no verdict — §3.4 remains FAILED and terminal.

## F. OPERATIONS amendment — template rule (from escalation (b))

> Any threshold ported across contexts requires a pre-registered validity guard in
> the destination context, or the leg it gates is advisory.

Entered into `OPERATIONS.md` as §8.5. It is binding on this cycle: E1′ pre-registers
its instrument-validity preconditions (IV1/IV2) **before** the candidate runs, which
is this rule applied to E1′'s own kill rule — the ported object being the §4.1 kill
form itself.

---

## Advisor channel — UNAVAILABLE this cycle (disclosure)

The previous cycle ran with an advisor channel (six consultations, C1–C6, logged in
`CONSULTATIONS.md`). **This cycle the advisor tool returned "unavailable" on first
call and was not retried.** No consultation exists for E1′.

The compensating discipline, applied throughout: every reading of the directive that
was not fully determined by frozen text is **pre-declared in the Part-2
pre-registration commit, with its derivation, before any candidate number exists**,
and the one reading that remains genuinely open (the σ-parameterization) carries a
**pre-declared, non-gating sensitivity column** so that it is adjudicable in owner
review without a re-run. That is the codified lesson of escalation (a), applied
pre-emptively rather than after the fact.
