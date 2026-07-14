Executor Directive — E1′ Final Appearance Cycle + Workbench Close-Out

Branch: eval/metric-workbench · Authority: owner directive, supersedes nothing in eval/v3.0 · Mode: OPERATIONS §8 (build → test → run → report neutrally). Every underdetermined parameter is escalated BEFORE running, never chosen. That is the codified lesson of escalation (a).

Budget: one day, cached-corpus CPU work. This is the final workbench cycle. Every outcome below terminates the workbench.


Part 1 — Record amendments (write these before any computation)


Escalation (a) — adjudicated. The E1 KILL verdict binds the candidate-as-specified (whitened delta vector under eig_floor_ratio 1e-6). The whitened no-subtraction control (acc 0.0628 ≈ chance, vs raw control 0.6054) is entered as evidence of instrument failure independent of the hypothesis. The endpoint-normalization hypothesis is recorded as unadjudicated, prior lowered (raw-delta collapse 0.1345, P1 analogue 8/29, P2 AUC 0.523 are the descriptive wounds). E1-as-specified is not re-run under any circumstances.
Escalation (b) — adjudicated. Leg A2's 0.90 floor is recorded as an invalid-in-context operationalization (amplitude-controlled threshold applied to uncontrolled real-clip channels). No mechanism claim attached. Reversal and §3.4 verdicts unaffected (leg B and the injected test fail independently).
Owner spec errors — entered into the record as design-error provenance for E1:
a. The RUNBOOK §3.5 recalls 0.62/0.44 were tag-group accuracies misread as a recall definition; superseded by the A4 backfill (0.34623 frozen).
b. The derivation "curves are integrals of what delta summarizes, so if delta fails the curves must" is wrong: the delta vector is net displacement — out-and-back excursions annihilate in it; the signature channels take magnitudes before integrating. E1 therefore structurally erased the signal class it proxied. This is the premise error under the (valid) E1 kill.
Phase 1 closure. Corpus facts recorded as findings: motion-scarce corpus (median per-pair tx 0.297 px / ty 0.421 px, rotation p50 0.00089), full-frame effects (outlier-area median 0.875 on undefined core frames). Consequences: M1b's realistic ceiling on this corpus is a presence/absence flag, not a similarity metric; M1c_flow moves to the v-next roster with acceptance tests aimed at object-side constructed truth (synthetic residual patterns), not camera reversal. No motion re-attempt this cycle (§3.6 stands).
One readout from existing Phase 1 artifacts (no re-run): for the 3 injected-trajectory cells that fail where the oracle passes with margin — report which channels they failed on. Pure readout of persisted records; append to phase1/RECORD.md.
OPERATIONS amendment (template rule, from escalation (b)): any threshold ported across contexts requires a pre-registered validity guard in the destination context, or the leg it gates is advisory.



Part 2 — E1′ pre-registration (freeze this section verbatim, with commit hash, before any candidate computation)

2.1 Candidate: γ-scalar signature (R0 rung, direct)

Per clip, three scalar channels over arc-length σ ∈ [0,1] within the S-mask:


â(σ), b̂(σ) — the endpoint-progress coordinates S already computes.
m̃(σ) = m(σ) − m_lerp(σ) — sided residual magnitude, null-calibrated:

m(σ) = ‖ρ(σ)‖/D where ρ = frame embedding minus its projection onto span{e_A, e_B} (two-sided classes) or onto e_A alone (one-sided classes).
m_lerp computed identically on the clip's rendered null (frame-aligned by construction; σ-alignment = index alignment; same anchors e_A, e_B — the null shares the pair's endpoints).



Geometry: RAW embedding space. No ZCA anywhere in the gating arm. Anchors = flanking-frame means. Min-D guard at the frozen 5th-percentile floor (flag, exclude, never zero).
Resample 64 per channel → per-channel corpus z-norm → banded DTW (≤10%) per channel → equal-weight mean of the three channel distances (frozen combination rule).


2.2 Pre-declared arms (closed list — no other variant may be computed)

armrole(â, b̂, m̃) raw geometrythe candidate — the only gating arm(â, b̂, m) — no null subtractiondiagnostic control (calibration on/off)(â, b̂, m̃) Ledoit-Wolf shrinkage-whiteneddiagnostic column, non-gatingm̃ channel alonediagnostic (where the signal lives)

2.3 Instrument-validity preconditions (owner-chosen, frozen now: 0.90 both)

The kill verdict binds the hypothesis only if both hold; otherwise the recorded verdict is INSTRUMENT-INVALID (workbench still closes — see 2.6):


IV1 (effect vs nothing): binary LOO 1-NN over pooled signatures {223 real clips, 223 rendered nulls} ≥ 0.90 accuracy.
IV2 (snap vs nothing): binary 1-NN {hard-cut splice, rendered lerp}, reusing the existing Bar-6 splice construction, ≥ 0.90. Report pair coverage if incomplete.
Rationale on record: a signature that cannot distinguish an effect from its absence, or a cut from a crossfade — the objects it was designed to make identifiable — cannot issue findings about class identity.


2.4 Kill rule (same form as §4.1, verbatim)

If the candidate fails to beat pinned m1a__v3_sided (d 1.522006, misretrieved 73/223) on both Cohen's d and misretrieved count → KILL, one appendix paragraph, workbench closed. Binding per 2.3.

2.5 Gates and reporting (unchanged from RUNBOOK)

Frozen exam kernel, LOO + Cohen's d, per-clip margins. Hubness gate at the frozen numbers (skew ≤ 3.0, entropy ≥ 0.70, max-pred ≤ 0.25, k = 10). Coverage reported next to accuracy. §7 adoption conditions computed as facts (d ≥ 1.772006 ∧ misretrieved < 73 ∧ probe battery ∧ hubness ∧ coverage). Predictions re-checked in registered form: P1 (sibling γ-distance < clip-to-own-null, per eligible class — now checkable), P2 (sidedness AUC from this signature's s-asymmetry), P4 (one-sided early-σ mass). Descriptive, non-gating.

2.6 Terminal outcomes (all close the workbench; no third cycle under any outcome)


KILL (IV pass): endpoint-normalization program dead at appearance level, adjudicated this time. Appendix paragraph.
INSTRUMENT-INVALID (IV fail): program closes unadjudicated; recorded as such; no repair attempts.
Survives kill, misses §7: appendix analysis, workbench closed.
Meets §7 in full: eligible for v3.1 re-cert through the standard door; workbench closed.


2.7 Prohibitions

No E2/E3 resurrection. No motion work. No parameter search outside 2.2's closed list. No threshold may be derived after seeing any candidate number. Ambiguity → escalate and halt that leg.


Part 3 — Deliverable

Same neutral-report format as the previous workbench report: outcome table first, freeze verification, computed pass/fail facts per rule, escalations (if any) owner-reserved, integrity section with commit hashes, artifacts listed. No recommendations, no interpretation.
ArtifactsMetric workbench runbookDocument · MD ContentCold Start Summary
Context
Emir is building the evaluation harness for his "Creative Transition Transfer" research project (IC-LoRA on LTX-2 for style-transferable video transitions, paper freeze August 16, 2026, targeting ICLR/CVPR 2027). This conversation covered the full arc of designing, red-teapastedThe agent's diagnostics are largely correct and its taxonomy read (gas_transformation↔shadow_smoke at distance ratio 0.99, the tag-table error breakdown) genuinely adds value. But two of its four proposals fail the test we've been applying all along — could this edit have been reverse-engineered fropastedLet me build this from the ground up, being explicit about every assumption, because the current M1a has a documented wound that this program is trying to heal — and I want to be precise about what the cure actually requires.
1. What object are we actually comparing?
A transition is an operator, notpasted# Metric Workbench — Report

**Branch** `eval/metric-workbench` · **Run** 2026-07-14 · **Executor** implementation
agent (build → test → run → report neutrally, OPERATIONS §8).

This is a **neutral data package**. Every §7 adoption condition below is a **computed
pass/fail fact**. There is **npasted