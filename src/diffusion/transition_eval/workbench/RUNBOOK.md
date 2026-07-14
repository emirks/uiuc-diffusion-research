# Metric Workbench Runbook — Motion Rebuild + Endpoint-Normalized Appearance

**Project:** Creative Transition Transfer — eval harness v-next research track
**Branch:** `eval/metric-workbench`, forked from `origin/main` @ `9b1a4cb` (contains
tag `eval/v3.0.0`; the certified package `src/diffusion/transition_eval` is
byte-identical to the tag at this base — verified by diff before fork).
**Status:** PRE-REGISTERED AND FROZEN at the commit that introduces this file.
Frozen before any Phase 1/2 computation. Edits after freeze only as dated,
disclosed amendments in §A — never in place.
**Authority chain:** `SPEC.md` governs the certified instrument (v3.0.0 —
untouched by this document). This RUNBOOK governs the workbench science.
`OPERATIONS.md` (same directory) governs execution mechanics. Conflicts:
SPEC > RUNBOOK for anything certified; RUNBOOK > OPERATIONS for anything
scientific.
**Hard boundary:** eval/v3.0 tag is NOT reopened by anything in this document.
Candidates enter only via v3.1 re-cert through the adoption rules in §7.
Nothing past E3 runs before the August 16 paper freeze under any outcome.

---

## A. Amendments (pre-freeze, owner-disclosed)

### A1 — Incumbent baseline pinned from the certification artifact, not from memory

The draft text below cites the incumbent as "LOO 1-NN accuracy 0.673, 71/221
clips misretrieved" and "M1a defined on 221". The persisted certification
artifact (the draft.8 exam that the 3.0.0 record regrades) says **73/223
misretrieved at coverage 1.0**. The 71/221 figures are a drafting error; the
artifact is the record. All baseline references in §0, §4.1 (kill rule), and
§7 (adoption rule) resolve to the **pinned values in §B below**, which were
extracted from the artifact and written here BEFORE any candidate ran.
Wherever the draft says "71/221", read "the pinned incumbent misretrieved
count (73/223)".

### A2 — Flow backbone fallback made concrete

§3.1 names SEA-RAFT with RAFT-small fallback. SEA-RAFT is a new dependency in
this environment. Concretized: attempt SEA-RAFT install once, timeboxed
(~30 min); on failure fall back without ceremony to torchvision RAFT
(`raft_large` or `raft_small`; `src/diffusion/signals/flow.py` already wraps
`raft_large`). The chosen backbone + weights version is a pin recorded in
`baselines.json` and in every Phase 1 artifact. The backbone choice is made
ONCE, before any Phase 1 acceptance test runs, and never revisited this cycle.

### A3 — "Masked DINOv2 embeddings" reading fixed

§1.1's "per-frame masked DINOv2 embeddings" means **temporal masking**:
per-frame DINOv2 CLS embeddings, frame-selected by the sidedness-aware S-mask
(`core_mask_v3`) — exactly the substrate certified M1a operates on, and
exactly what the existing feature cache holds. It does NOT mean spatially
masked patch tokens (the cache holds CLS only; no patch tokens exist). ZCA
(§1.1) is fit on per-frame CLS embeddings across the corpus.

### A4 — Within-stratum baseline recall backfill

The 0.62 / 0.44 within-camera-stratum recalls cited in §3.5 predate the
persisted by-tag machinery. Before the Phase 1 exam runs, these baselines are
recomputed ONCE from the frozen incumbent distance matrices
(`distance_matrices.npz`, sha pinned in §B) using the same stratum-grouping
code that will judge the candidates, and the recomputed values are recorded
in `baselines.json`. Those recorded values — not 0.62/0.44 from memory — are
the numbers to beat. This is corpus-only, outcome-independent calibration
(the candidate has not run), permitted pre-freeze under the two-kind
calibration rule.

---

## B. Pinned incumbent baselines (extracted 2026-07-14, before any candidate ran)

Source of record: the 3.0.0-draft.8 certification run (job 9465002, commit
`31dd07e`), regraded to 3.0.0 (record commit `dfc1901`, tag `eval/v3.0.0`).
Artifact paths are absolute and read-only (see OPERATIONS.md §Paths):

- `…/eval-v3-spec/outputs/eval/certification/3.0.0-draft.8/analysis/distance_matrices.npz`
  sha256 `f96934c65fdc95f9a4709e5673ba39b00f3c257aba191c8c8a14889ceb31483b`
- `…/eval-v3-spec/outputs/eval/certification/3.0.0-draft.8/analysis/analysis.json`
- `…/eval-v3-spec/outputs/eval/certification/3.0.0-draft.8/exam/trust_map.json`
- `…/eval-v3-spec/outputs/eval/certification/3.0.0/record.json` (regrade record)

| metric id | LOO 1-NN acc | Cohen's d | coverage | misretrieved |
|---|---|---|---|---|
| `m1a__v3_sided` (**incumbent appearance**) | 0.672646 | 1.522006 | 1.0000 | **73/223** |
| `m1a__v2_envelope` | 0.578475 | 1.319097 | 1.0000 | 94/223 |
| `m1a__all_frames` | 0.538117 | 1.271417 | 1.0000 | 103/223 |
| `m_incumbent` (v2 MFS) | 0.062780 | 0.368167 | 1.0000 | 209/223 |
| `m1b_camera` (**incumbent camera**) | 0.268519 | 0.519962 | 0.9686 | 165/223 |
| `m1c_object` (**incumbent object**) | 0.076577 | 0.247601 | 0.9955 | 206/223 |

Chance = 0.067 (39 classes, 223 clips). Within-stratum recalls: backfilled
per A4 into `baselines.json` before the Phase 1 exam.

Step-0 obligations (OPERATIONS.md): regenerate these numbers from the
artifacts, assert equality with this table, verify the npz sha256, and verify
one incumbent matrix reproduces bitwise from the warm cache before any
head-to-head is trusted.

---

## 0. Goal and context

The certified harness (eval/v3.0) measures reference-based transition
transfer. Its certified core is M1a (appearance) + M2a/M2b (copy/intrusion) +
M3 (endpoints/seam) + pipeline determinism. Two known weaknesses motivate
this workbench:

1. **Motion is dead.** M1b (camera) exam accuracy 0.269 with discrimination
   only inside camera-tagged strata; M1c (object) at 0.077 ≈ chance,
   hub-collapsed (polygon column = sink artifact from near-zero residual
   descriptors). Both are analysis-tier. This workbench rebuilds them on
   optical flow.
2. **Appearance is content-contaminated.** M1a compares masked core frames
   absolutely; the 0.82 content-invariance finding is the bill. This
   workbench tests whether endpoint-normalized representations (quotient by
   the clip's own endpoints, calibrated against a rendered degenerate null)
   improve class identity.

**Corpus (frozen):** 223 real transition clips, 39 style classes (26
one-sided / 13 two-sided), mean class size ~6, two singletons. Existing
definedness baselines: see §B coverage column (per A1, artifact values
supersede the drafted 221/213/220).
**Incumbent baseline to beat:** `m1a__v3_sided` — pinned in §B (acc 0.672646,
d 1.522006, 73/223 misretrieved). Chance = 0.067.
**Exam machinery (frozen, reused as-is):** leave-one-out 1-NN retrieval over
the corpus + Cohen's d between within-class and cross-class distance
distributions; per-clip margins (nearest-cross − nearest-within); trust maps
per class; mandatory hubness diagnostic (see §1.4).

---

## 1. Shared infrastructure (build once, both phases use it)

### 1.1 Whitening
Fit corpus-level ZCA on per-frame masked DINOv2 embeddings (per A3: CLS
embeddings, S-mask frame selection). Fit once, freeze, persist the matrix.
**All** inner-product geometry downstream (anchors, projections, residuals,
distances) operates in whitened space. Raw DINO is anisotropic; unwhitened
chords and angles are measured with a bent ruler.

### 1.2 Endpoint anchors and guards
- e_A, e_B = mean whitened embedding of the flanking stable frames outside
  the S-mask (never single frames).
- Chord length D = ‖e_B − e_A‖. Persist the corpus D distribution.
- **Min-D guard:** floor at the 5th percentile of the corpus D distribution.
  Below floor → clip flagged `low_D`, excluded from normalized scores, never
  zeroed.

### 1.3 Curve conventions
- Parameterize by normalized arc length σ ∈ [0,1] within the S-mask (not raw
  time, not progress — progress can be non-monotone).
- Resample all signature channels to 64 points (motion descriptors: 32).
  Z-score per channel over the corpus.
- Distance: L2 default; banded DTW with ≤10% band as fallback alignment.
  Never widen the band — timing is M1d's property.
- No differential invariants (curvature/torsion): noise amplifiers at 20–100
  jittery frames. Integral quantities only.

### 1.4 Hubness gate (mandatory per candidate space/descriptor)
Compute prediction-column entropy and k-occurrence skew on the exam's
distance matrix. A candidate with a sink column (M1c/polygon pattern) fails
regardless of accuracy. Persist stats with the distance matrices.

### 1.5 Definedness discipline
Undefined ≠ zero, everywhere. Frames or clips failing gates are excluded from
fits and averages and counted in a definedness report per metric. A
candidate's definedness coverage is reported next to its accuracy (an
accuracy win on a shrunken support is not a win).

---

## 2. Phase 0 — gates

1. eval/v3.0 tagged (7-bar cert passing, deterministic reproduction).
   **STATUS: SATISFIED 2026-07-14** — tag `eval/v3.0.0` exists, record
   committed (`dfc1901`), bars frozen.
2. σ_seed measured: 12 stratified items × 5 seeds on the adapter arm. Gates
   the first model report, **not this workbench** (per its own annotation —
   not a workbench blocker; still PENDING as of freeze).
3. Only then: Phases 1 and 2 may start, in parallel. Both are cached-corpus
   work and interleave with LoRA training babysitting.

---

## 3. Phase 1 — motion workbench (~2–3 days)

### 3.1 Flow backbone
- SEA-RAFT (fallback torchvision RAFT, per A2) at ~320px, adjacent frame
  pairs.
- Seam frames (from M3b) excluded from all fits.

### 3.2 M1b_flow — camera metric
- **Model:** 4-parameter similarity (tx, ty, log-scale, rotation) fit to the
  dense flow field per frame pair.
- **Fit:** Huber IRLS, δ ≈ 1.5 px. Where S provides a spatial effect mask,
  fit on its complement; else rely on Huber inlier weighting.
- **Definedness:** inlier fraction < 40% or low-texture frame → frame
  undefined. > 30% of core frames undefined → clip undefined.
- **Descriptor:** the 4 parameter trajectories over core frames → resample 32
  → z-score per dim → L2 (banded DTW fallback).

### 3.3 M1c_flow — object metric
- **Residual:** flow − fitted camera field, inside the effect region, core
  frames only.
- **Energy gate first:** mean |residual| < ε → frame undefined. Set ε from
  the corpus residual-magnitude distribution (report chosen percentile). This
  is the designed kill for the polygon sink: near-static residuals must exit
  as undefined, not collapse to a shared descriptor.
- **Descriptor per frame:** 8-bin magnitude-weighted orientation histogram on
  a 3×3 grid + mean divergence + mean curl + normalized magnitude. Magnitudes
  normalized by image diagonal (resolution must not leak).
- **Clip descriptor:** per-frame vectors → resample 32 → z-score → L2.

### 3.4 Acceptance tests (constructed truth — both must pass before the exam runs)
1. **Reversal probe** (inherited from old Bar 5): time-reversed reference
   must be distinguished from the original — camera parameter trajectories
   negate; descriptor distance to reversed self must exceed the median
   within-class distance. (Known blind spot to design against, recorded in
   `certify/probes.py`: per-channel z-norm makes time-antisymmetric
   trajectories self-identical — the probe must check parameter negation,
   not only descriptor distance.)
2. **Injected-trajectory recovery:** apply known synthetic pans/zooms/
   rotations to static clips; recovered parameter trajectories must match
   ground truth (per-parameter correlation ≥ 0.9 and relative amplitude
   error ≤ 10%).

Failure of either = fix or stop; the exam is not run on a metric that fails
constructed truth.

### 3.5 Exam and success criteria
- LOO + Cohen's d through the frozen machinery; hubness gate; definedness
  report.
- **Realistic target:** within-camera-stratum recall beats incumbent M1b's
  recorded stratum recalls (backfilled per A4 into `baselines.json`; drafted
  from memory as 0.62 / 0.44). Pooled accuracy may stay modest — the trust
  map scopes the claim.

### 3.6 Adoption / stop rules
- Beats incumbent M1b/M1c on Cohen's d **and** within-stratum recall →
  one-hour v3.1 re-cert, tier upgrade (analysis → certified for the scoped
  strata).
- Misses → stays analysis-tier. **No second attempt this cycle.**
- If M1c_flow still exhibits a hub after the energy gate → the descriptor is
  dead; no rescue variants pre-freeze.

---

## 4. Phase 2 — appearance ladder (~3–4 days, hard stops)

### 4.0 Pre-work (one afternoon)
- ZCA matrix (§1.1). Anchors + D distribution + min-D floor (§1.2).
- **Rendered-lerp null:** for every corpus clip, synthesize the degenerate
  video for its own endpoint pair (alpha-blend at matched progress; reuse
  core_degenerate machinery), embed and cache its whitened curve. This is the
  per-pair calibration object — the geometric chord is kept only as the
  coordinate frame, never as the null.

### 4.1 E1 — effect-delta vector (THE KILL TEST — runs first)
- **Definition:** v_clip = mean whitened masked embedding over core frames;
  v_null = same pooling over the clip's rendered lerp; **delta = v_clip −
  v_null**. One vector per clip. Distance: L2.
- **Exam:** LOO + d, head-to-head vs `m1a__v3_sided` on the frozen corpus and
  splits.
- **KILL RULE (verbatim, pre-registered; counts per A1):** if delta fails to
  beat raw M1a on **both** Cohen's d and misretrieved count (pinned: 73/223),
  the endpoint-normalization program is dead at the appearance level. One
  appendix paragraph, full stop. E2/E3 do not run.

### 4.2 E0 — anatomy plots (rides the E1 run, no gating numbers)
γ-curves per clip: â(σ), b̂(σ) endpoint-progress coordinates (from S), m(σ) =
‖ρ(σ)‖/D residual magnitude. Eyeball checks: flame-family detaches from the
chord, bloom-family hugs it; the 26/13 sidedness split is recoverable from
s-asymmetry.

### 4.3 E2 — γ-signature (only if E1 pays)
- **Coordinates:** sided — project out span{e_A, e_B} for two-sided classes,
  e_A alone for one-sided.
- **Channels:** â(σ), b̂(σ), and **m̃(σ) = m(σ) − m_lerp(σ)** — residual
  magnitude calibrated per endpoint pair against the rendered null (removes
  the endpoint-pair-dependent crossfade bump; without this the signature
  re-imports content through the channel it claims to quotient).
- **Compare:** resample-64, z-norm, banded DTW. Exam: LOO + d vs E1's delta
  and vs raw M1a.

### 4.4 E3 — within-video Gram (only if E2 adds over E1)
- PCA residuals to 20–50 dims, then Gram(ρ̃): normalized inner products
  ⟨ρ̃(σᵢ), ρ̃(σⱼ)⟩/D² — direction-blind choreography of the excursion.
  Distance: Frobenius or elastic.
- Plus measurement-only: principal angles between per-class residual
  subspaces (Level-3 direction feasibility). **No adoption path for direction
  this cycle.**

### 4.5 Ablation flags (on whichever rung survives)
- null ∈ {rendered lerp (default), geometric chord}
- coords ∈ {sided â/b̂ (default), chord-decomposition}
- space ∈ {whitened DINOv2 (default), CLIP}; generator VAE-latent as
  diagnostic column only, never a headline space.

---

## 5. Pre-registered predictions (write into the record now, check later)
1. Sibling γ-distance < clip-to-own-rendered-null distance for every
   n≥4-eligible class ("real effects are farther from nothing than from each
   other").
2. Sidedness (26/13) recoverable from s(σ) asymmetry.
3. nature_bloom remains lerp-adjacent — blind-spot theorem (smooth-growth
   effects genuinely live near the chord), not a metric failure.
4. One-sided classes concentrate excursion mass in early σ.

---

## 6. Class-scope discipline
All class-level claims restricted to the n≥4-eligible scope already adopted
by the merged Bar 2+3 (eligibility is a corpus fact, frozen pre-run; never
scope by outcome-dependent trust status). Singletons and n≤3 classes:
reported, never gating.

---

## 7. Adoption rule (verbatim, frozen before any Phase 2 run)
A candidate replaces or joins M1a only if ALL of:
1. LOO Cohen's d improves by ≥ 0.25 over `m1a__v3_sided` (pinned d
   1.522006), **and** misretrieved count drops below the pinned 73/223;
2. The probe battery passes unchanged: splices don't fool it; siblings rank
   above controls per n≥4-eligible class; twins stay 11/11 through the M2a
   channel (which it must not degrade); rendered lerp scores at the floor for
   non-bloom classes;
3. Hubness gate passes (§1.4);
4. Definedness coverage reported and not materially narrower than the
   incumbent's (pinned: 1.0000).

Anything less = appendix analysis, not a ruler swap. Entry is via v3.1
re-cert; the v3.0 tag and both failed-draft records stay permanent. If the
exam shows raw M1a out-retrieving the quotient on texture-saturated classes,
the recorded structure is: signature as primary defensible metric, raw M1a as
flagged higher-power secondary — never a silent revert.

---

## 8. Artifacts to persist (standard outputs of every run)
Distance matrices + per-clip margins; hubness stats; trust maps; definedness
reports; ZCA matrix; rendered-null cache; corpus D distribution;
acceptance-test records; the pre-registration text of §4.1, §5, §7 with run
timestamps (satisfied by this file's freeze commit + per-run stamps).

---

## 9. Budget and stop rules
- Phase 1 + Phase 2 through E2 ≈ one week interleaved.
- E1 dies → four days back, clean negative result (paper text), workbench
  closed.
- No second motion attempt, no direction-channel adoption, nothing past E3
  before August 16, regardless of outcome.
- Any result feeding a paper claim goes through the v3.1 re-cert door; no
  number skips certification.
