# E1′ — PRE-REGISTRATION (FROZEN)

**Authority:** `E1PRIME_DIRECTIVE.md` Part 2, committed verbatim at **`2ac90d7`**
(the Part-1 amendments commit).

**FROZEN BEFORE ANY CANDIDATE COMPUTATION.** When this file is committed, no E1′
signature, no E1′ distance, no E1′ IV number and no E1′ exam exists. Nothing below
may be changed after any candidate number is seen. Changing a number here after the
fact is threshold-changing — owner-reserved, and terminal for the track.

---

# Part A — the directive's §2, transcribed

## 2.1 Candidate: γ-scalar signature (R0 rung, direct)

Per clip, three scalar channels over arc-length σ ∈ [0,1] within the S-mask:

- **â(σ), b̂(σ)** — the endpoint-progress coordinates S already computes.
- **m̃(σ) = m(σ) − m_lerp(σ)** — sided residual magnitude, null-calibrated:
  - `m(σ) = ‖ρ(σ)‖/D` where ρ = frame embedding minus its projection onto
    span{e_A, e_B} (two-sided classes) or onto e_A alone (one-sided classes).
  - `m_lerp` computed identically on the clip's rendered null (frame-aligned by
    construction; σ-alignment = index alignment; same anchors e_A, e_B — the null
    shares the pair's endpoints).

**Geometry: RAW embedding space. No ZCA anywhere in the gating arm.** Anchors =
flanking-frame means. Min-D guard at the frozen 5th-percentile floor (flag, exclude,
never zero).

Resample 64 per channel → per-channel corpus z-norm → banded DTW (≤10%) per channel
→ **equal-weight mean of the three channel distances** (frozen combination rule).

## 2.2 Pre-declared arms (closed list — no other variant may be computed)

| arm | role |
|---|---|
| **(â, b̂, m̃) raw geometry** | **the candidate — the only gating arm** |
| (â, b̂, m) — no null subtraction | diagnostic control (calibration on/off) |
| (â, b̂, m̃) Ledoit-Wolf shrinkage-whitened | diagnostic column, non-gating |
| m̃ channel alone | diagnostic (where the signal lives) |

## 2.3 Instrument-validity preconditions (owner-chosen, frozen: 0.90 both)

The kill verdict binds the hypothesis **only if both hold**; otherwise the recorded
verdict is **INSTRUMENT-INVALID** (workbench still closes — see 2.6):

- **IV1 (effect vs nothing):** binary LOO 1-NN over pooled signatures {223 real clips,
  223 rendered nulls} **≥ 0.90** accuracy.
- **IV2 (snap vs nothing):** binary 1-NN {hard-cut splice, rendered lerp}, reusing the
  existing Bar-6 splice construction, **≥ 0.90**. Report pair coverage if incomplete.

> Rationale on record: a signature that cannot distinguish an effect from its absence,
> or a cut from a crossfade — the objects it was designed to make identifiable —
> cannot issue findings about class identity.

## 2.4 Kill rule (same form as §4.1, verbatim)

If the candidate fails to beat pinned `m1a__v3_sided` (**d 1.522006**, **misretrieved
73/223**) on **both** Cohen's d and misretrieved count → **KILL**, one appendix
paragraph, workbench closed. Binding per 2.3.

## 2.5 Gates and reporting (unchanged from RUNBOOK)

Frozen exam kernel, LOO + Cohen's d, per-clip margins. Hubness gate at the frozen
numbers (skew ≤ 3.0, entropy ≥ 0.70, max-pred ≤ 0.25, k = 10). Coverage reported next
to accuracy. §7 adoption conditions computed as facts (d ≥ 1.772006 ∧ misretrieved
< 73 ∧ probe battery ∧ hubness ∧ coverage). Predictions re-checked in registered form:
P1, P2, P4. Descriptive, non-gating.

## 2.6 Terminal outcomes (all close the workbench; no third cycle under any outcome)

1. **KILL (IV pass)** — endpoint-normalization dead at appearance level, adjudicated.
2. **INSTRUMENT-INVALID (IV fail)** — program closes unadjudicated; no repair attempts.
3. **Survives kill, misses §7** — appendix analysis, workbench closed.
4. **Meets §7 in full** — eligible for v3.1 re-cert through the standard door.

## 2.7 Prohibitions

No E2/E3 resurrection. No motion work. No parameter search outside 2.2's closed list.
**No threshold may be derived after seeing any candidate number.** Ambiguity →
escalate and halt that leg.

---

# Part B — executor readings, PRE-DECLARED

The directive is precise, but three of its terms admit more than one implementation.
Each is settled below **by derivation from frozen text wherever the text determines
it**, and where it does not, the open reading is named as open and given a
**pre-declared non-gating sensitivity column** so it is adjudicable in owner review
**without a re-run**. That is escalation (a)'s lesson applied before the fact instead
of after it.

**The advisor channel was unavailable this cycle** (`CONSULTATIONS.md`). These are
executor readings, reviewed by no one. They are declared here, before any number, so
that owner review can catch what consultation would have.

## P1. σ-parameterization — THE ONE GENUINELY OPEN READING

**GATING (frozen):** σ is the normalized cumulative chordal arc length of the
**3-channel signature curve** γ(t) = (â(t), b̂(t), m̃(t)) ∈ ℝ³ over the clip's core
frames, computed in **native chord units, before the corpus z-norm**. All three
channels are resampled onto that **single shared σ grid** at 64 points.

Derivation from frozen text:

1. §1.3: *"Parameterize by normalized arc length σ ∈ [0,1] within the S-mask … Resample
   **all signature channels** to 64 points."* — **one** σ, and all channels resampled
   onto it. σ is therefore a property of the signature curve **as a whole**, not of
   each channel separately.
2. §1.3 explicitly rejects progress as the parameter (*"not progress — progress can be
   non-monotone"*). â **is** progress along the chord. So σ ≠ â. Satisfied.
3. §1.3: *"No differential invariants (curvature/torsion)."* Curvature and torsion are
   properties of a curve **in a multi-dimensional space**. The sentence only parses if
   the signature IS such a curve — γ ∈ ℝ³.
4. Directive 2.1: *"three scalar channels over arc-length σ ∈ [0,1]"* — one σ, shared.
5. Order of operations is fixed by the directive itself (*"Resample 64 per channel →
   per-channel corpus z-norm"*): resampling **precedes** z-norm, so σ must be
   computable before z-norm, i.e. in native units. All three channels are in chord
   units (each divides by D), hence commensurate, so a Euclidean arc length in ℝ³ is
   well-defined.

**Rejected reading — per-channel arc length** (each channel reparameterized by its own
total variation). Rejected **by derivation, not by taste**: for a monotone scalar
channel v(t), arc length is σ(t) = cum|Δv| / total|Δv|, whence
**v(σ) = v₀ + σ·(v_T − v₀) — a straight line, for every clip.** â is near-monotone by
construction, so this reading annihilates the â channel corpus-wide. §1.3's stated
purpose is to *compare shapes*; a reading that erases a channel's shape cannot be the
one implementing it. Proved in
`tests/test_workbench_e1prime.py::test_per_channel_arclength_linearizes_monotone`.
(This is the reading `e0_anatomy.py` used. E0 is descriptive and non-gating, so no
verdict rested on it — see the disclosure in `E1PRIME_AMENDMENTS.md` §A.)

**Open alternative — σ_emb, arc length of the raw 768-d embedding path** over the same
core frames. This reading is **also consistent with §1.3's text**; it is not chosen,
because §1.3's vocabulary (*"signature channels"*, curvature/torsion) locates the curve
in signature space, not in the ambient DINO space that the directive's raw-geometry
choice deliberately declines to correct. But the text does not exclude it.

**→ PRE-DECLARED NON-GATING SENSITIVITY COLUMN:** the gating arm is **also** computed
under σ_emb, and reported beside the verdict. **It cannot change the verdict under any
outcome.** Its sole purpose is to let owner review see whether the E1′ verdict was
σ-sensitive, without a re-run. Declared here, before any number exists.

## P2. ρ — the sided residual (DETERMINED; linear span, not the affine chord)

`gates.yaml` (frozen) pins it: `e2.coords: sided  # project out span{e_A, e_B}
two-sided; e_A alone one-sided`. With the directive's 2.1 this determines:

- two-sided: P = orthogonal projector onto **span{e_A, e_B}** — the 2-D **linear**
  subspace through the origin.
- one-sided: P = orthogonal projector onto **span{e_A}** — 1-D.
- ρ(f) = f − P f;  m = ‖ρ‖ / D;  D = ‖e_B − e_A‖.

**Why linear and not the affine chord line** (the residual `anchors.endpoint_progress`
already returns): the **one-sided clause decides it.** "Projection onto e_A alone" has
no affine reading — the affine hull of a single point is that point, and f − e_A is not
a residual (nothing is projected out); it would merely restate the progress
coordinates. Only the projection onto the **line spanned by** e_A is a projection at
all. `gates.yaml`'s "project **out** span{…}" (remove a subspace) says the same.
Decisively: the directive spells out a **new formula** for m, where for â and b̂ it
simply says "the coordinates S already computes" — had it meant the existing residual,
it would have said so there too.

**Geometric note, disclosed:** â/b̂ (affine, measured from e_A) and m̃ (linear-span
residual, measured from the origin) therefore use **different projections**. This is
what both frozen texts specify, and it is implemented as registered. The executor does
not "fix" it.

**Rank guard:** the 2-D basis is built by Gram–Schmidt; if the second vector's norm
falls below 1e-8·‖e_B‖ the basis is degenerate → the clip is **UNDEFINED with a
reason** and counted (never silently demoted to rank-1). DINO CLS features are
L2-normalized and `endpoint_vecs` re-normalizes, so the origin is the unit sphere's
centre — a content-independent origin.

## P3. Anchors and geometry (DETERMINED)

`e_A, e_B = certify.probes.endpoint_vecs(bundle)` — the deployed endpoint definition,
the convention pinned in the previous cycle (C1), **not whitened** (directive: RAW).
`D = ‖e_B − e_A‖` in raw space.

## P4. Min-D guard (percentile FROZEN; value is corpus-only calibration)

The **percentile is already frozen**: `gates.yaml min_d.percentile: 5.0`. The floor
**value** must be recomputed on the **raw-space** D distribution — the previously
persisted floor was computed in whitened space and is meaningless under raw geometry.

This is **corpus-only calibration** (it reads the frozen corpus and nothing else; no
candidate signature or distance exists when it is computed), the class `gates.yaml`'s
own header explicitly permits pre-freeze. It is computed and **frozen in its own
commit before any candidate distance**.

low_D clips are **FLAGGED and EXCLUDED** from the γ-signature — every channel of it
divides by D — and **never zeroed** (§1.2, §1.5). Coverage is reported next to
accuracy, always.

## P5. IV structure — TWO DISCLOSURES, MADE BEFORE THE IV RUNS

### (a) The m̃ ≡ 0 tautology

`controls.make_lerp` returns `concat([prefix, mid, suffix])`. A rendered null's own
first-9 / last-8 frames are therefore **exactly** the prefix/suffix it was built from,
so **the rendered null of a rendered null is that null, bit-for-bit** — `make_lerp` is
idempotent on its own endpoints. (Test:
`test_workbench_e1prime.py::test_make_lerp_is_idempotent_on_its_own_endpoints`.)

**Consequence, stated before any number exists:** for every "nothing" object — the
corpus null in IV1, the matched lerp in IV2 — `m̃ = m_lerp − m_lerp ≡ 0` **exactly, on
every frame, by construction**. The "nothing" class is therefore a **constant** on the
m̃ channel, and any real clip with non-zero m̃ separates from it there for structural
reasons. **IV1 and IV2 certify less than their names suggest.**

The IV is nonetheless **computed exactly as registered** — the executor does not
redesign the owner's precondition — and its registered verdict is the one that binds
per 2.3. Note the tautology does **not** make IV1 vacuous: if real clips *also* have
m̃ ≈ 0 (real effects do not depart from their own rendered null), **IV1 fails**. It is
a real test, of a weaker proposition than its name implies.

**→ PRE-DECLARED NON-GATING DISCLOSURE COLUMN:** both IVs are additionally computed on
the **(â, b̂) channels only** — the two channels that are *not* structurally constant
for the "nothing" class. If the full signature passes while the (â,b̂) column sits at
chance, the pass rests on the tautological channel. Reported; **does not alter the
verdict.**

### (b) IV pools and coverage

The IV1 pool is the **defined** signatures among {223 clips} ∪ {223 nulls}; low_D
clips (and their nulls) are undefined by P4, dropped, **counted**, and their coverage
reported. Same discipline for IV2 (2.3 asks for pair coverage explicitly).

## P6. IV2 construction (DETERMINED, with one disclosed engineering choice)

- **Cuts:** `certify.probes.build_hard_cut` — the deployed Bar-6 construction,
  **imported, never reimplemented** — on each n≥2 class's `bar_pair`, selected by the
  deployed `certify.probes.sibling_pairs` (deterministic, corpus-only). Prefix of A
  (9 frames) + cover-cropped body of B: a content discontinuity exactly at the
  conditioning handoff.
- **The matched "nothing":** `nulls.render_null(cut_frames)` = `make_lerp(cut[:9],
  cut[-8:], 121)` — the crossfade with the **cut's own endpoints**. Same §4.0 per-pair
  null object, same deployed builder. This is what makes IV2 "snap vs nothing" rather
  than "snap vs some other video".
- The **cut's own** S-profile and anchors are used for **both** members of the pair —
  the frozen `v_null_pooling: clip_core_frame_indices` convention (the null is a
  calibration object *for that clip* and inherits its structure). Sidedness = the
  class's.
- **DEVICE CONSISTENCY (disclosed engineering choice).** Cut and lerp features are
  extracted **together, on the same device, in one job**, into `$WB_CACHE`. The
  certification's own probe features are **deliberately not reused**: cuts from a
  prior GPU run paired against lerps embedded now would let a **device-drift
  signature** separate IV2's two classes, and IV2 could then pass for a reason that
  has nothing to do with the signature. Nothing is written to the shared cache; the
  corpus is still read through `ReadOnlyExtractor`.
- This is the cycle's only GPU work (74 videos × 121 frames). The directive's budget
  ("cached-corpus CPU work") is a budget, not a prohibition, and IV2 is not
  constructible from cache — its videos have never been embedded into `$WB_CACHE`.

## P7. Distances (DETERMINED)

- **Corpus z-norm:** `curves.fit_channel_scaler`, fitted **once** on the defined
  signatures of the **223 real corpus clips**, and applied **unchanged** to nulls,
  cuts and IV2 lerps. (§1.3: *"Z-score per channel over the corpus."* The corpus is the
  223 real clips; fitting on a pool that included the calibration objects would let
  the nulls move the scale they are supposed to be measured against.)
- **Distance:** `curves.banded_dtw` per channel on the 64-point z-scored channel
  (band = 10% → 6), then the **equal-weight mean of the three channel distances** —
  the directive's frozen combination rule.
- **NaN** if either signature is undefined → the frozen kernel reads NaN as "cannot
  retrieve" and drops it from coverage (§1.5). `misretrieved` follows the frozen
  convention (`n_clips − n_correct`; an uncovered row counts as misretrieved, so
  definedness never buys accuracy).

## P8. The four arms (2.2's closed list — nothing else is computed)

- **ARM A — GATING:** (â, b̂, m̃), raw geometry. **The only arm that gates.**
- **ARM B — diagnostic:** (â, b̂, m), no null subtraction (calibration on/off).
- **ARM C — diagnostic, non-gating:** (â, b̂, m̃), **Ledoit–Wolf** shrinkage-whitened.
  LW is the **analytic closed form** (Ledoit & Wolf 2004): Σ* = δ·μI + (1−δ)·S with
  μ = tr(S)/p and δ = b²/d² — it has **no free parameter**, which is exactly why it
  answers escalation (a)'s question: *does a principled whitener also kill the signal,
  independent of the executor-chosen eigenvalue floor?* `sklearn` is absent from the
  env, so LW is implemented from the published formula and **unit-tested** (δ ∈ [0,1];
  trace exactly preserved; δ → 0 as n → ∞; Σ* PSD and better-conditioned than S). Fit
  population = §1.1's: the S-mask core frames of the 223 clips, pooled.
- **ARM D — diagnostic:** m̃ alone (where the signal lives).

## P9. What is NOT computed

No fifth arm. No σ grid other than the two pre-declared here (one gating, one
non-gating disclosure). No threshold derived from any candidate number. No E2, no E3,
no motion. If anything else turns out to be needed, the leg **halts and escalates**;
it is not chosen.
