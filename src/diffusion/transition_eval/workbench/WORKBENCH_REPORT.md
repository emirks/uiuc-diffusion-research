# Metric Workbench — Final Report (WORKBENCH CLOSED)

**Branch** `eval/metric-workbench` · **Run** 2026-07-14 (two cycles) · **Executor**
implementation agent (build → test → run → report neutrally, OPERATIONS §8).

This is a **neutral data package**. Every gate, kill rule and §7 adoption condition below
is a **computed pass/fail fact**. There is **no adoption recommendation, no
interpretation and no strategy** in this document — those belong to owner-side review.

**All three tracks terminated on pre-registered rules. The workbench is closed. No third
cycle under any outcome (E1PRIME_DIRECTIVE §2.6).**

---

## 0. Outcome at a glance

| cycle | track | pre-registered rule that fired | outcome |
|---|---|---|---|
| 1 | Phase 1 (motion) | §3.4 — *"the exam is not run on a metric that fails constructed truth"* | **FAIL** — exam not run; M1b/M1c_flow stay analysis-tier |
| 1 | Phase 2 (appearance) · **E1** delta | §4.1 KILL RULE | **KILL** — E2/E3 do not run |
| 2 | Phase 2 (appearance) · **E1′** γ-signature | §2.4 KILL RULE, **binding per §2.3** | **KILL (IV pass)** — **adjudicated**; workbench closed |

Kill rules honored are results, not failures (OPERATIONS §7).

**Both owner-reserved matters escalated in cycle 1 were adjudicated by the owner before
cycle 2 ran.** Cycle 2 escalates **one** new matter (§7 condition 4; moot for the call).

---

## 1. Freeze verification

**Cycle 1.** All six incumbent metrics **reproduce RUNBOOK §B exactly** from the frozen
`distance_matrices.npz` (sha256 verified) through the frozen exam kernel.
`m1a__v3_sided` rebuilt from warm bundles with deployed code: **max|Δ| = 0.0** (bitwise).

**Cycle 2 (E1′).** Ordering was enforced and is verifiable from the git history:

| step | commit | what existed at that commit |
|---|---|---|
| Part-1 record amendments | **`2ac90d7`** | no E1′ computation of any kind |
| **Part-2 pre-registration FROZEN** | **`aace78d`** | **no E1′ signature, distance, IV number or exam** |
| code + 16 unit tests | `c4fce9b` | still no candidate number |
| **raw min-D floor FROZEN** (corpus-only) | **`6930eae`** | **no candidate distance** |
| the run | *(after all of the above)* | — |

`gates.yaml` — the RUNBOOK's frozen gates — was **not touched by cycle 2**. Its last
commit is `8c38833` (the cycle-1 Phase-1 freeze). Every number cycle 2 reuses (kill rule
1.522006 / 73, hubness thresholds, §7, min-D percentile 5.0) is **quoted** from it, never
restated with a different value.

**Shared-cache canary.** Both GPU jobs printed a before/after cache count. Cycle 2's IV2
job: **2233 → 2233 entries, 0 files touched.** The corpus is read through a
`ReadOnlyExtractor` whose `extract()` raises, so polluting the certified cache is
impossible by construction, not by discipline.

*Disclosed for accuracy (unchanged from cycle 1):* the shared cache did grow earlier in
the day, 1933 → 2233 entries between 12:58 and 13:03. That is **not** the workbench — it
coincides exactly with the owner's own `exp060_score` job (12:57:43 → 13:03:18), and the
certified harness writes feature caches by design when scoring generations. All 223
corpus entries this report depends on are untouched (mtimes 2026-07-08), and the
incumbent bitwise round-trip still reproduces **max|Δ| = 0.0**.

---

## 2. Cycle 2 — E1′, the γ-scalar signature (the final candidate)

Full record: `outputs/eval/workbench/e1prime/RECORD.md`.

### 2.1 §2.3 instrument validity — BOTH PASS, so the kill BINDS THE HYPOTHESIS

| precondition | accuracy | floor | verdict | (â, b̂)-only column *(non-gating)* |
|---|---|---|---|---|
| **IV1** effect vs nothing — {223 clips, 223 rendered nulls} | **0.9357** | 0.90 | **PASS** | 0.8476 |
| **IV2** snap vs nothing — {37 Bar-6 hard cuts, 37 matched lerps} | **1.0000** | 0.90 | **PASS** | 1.0000 |

The (â, b̂)-only column was **pre-declared before the IV ran**, because the
pre-registration disclosed in advance that every "nothing" object has **m̃ ≡ 0 exactly**
(`make_lerp` is idempotent on its own endpoints — proved by unit test), which makes the
IV partly tautological on that channel. **The passes do not rest on the tautology:** with
m̃ removed entirely, the signature still separates effect-from-nothing at **0.85** and
cut-from-crossfade at **1.00**. The instrument is valid on its merits.

IV2 pair coverage **1.0000** over all 37 n≥2 classes.

### 2.2 §2.4 kill rule — FIRES on both legs

> *"If the candidate fails to beat pinned `m1a__v3_sided` (d 1.522006, misretrieved
> 73/223) on **both** Cohen's d and misretrieved count → KILL, one appendix paragraph,
> workbench closed."*

| | candidate | incumbent | beats? |
|---|---|---|---|
| Cohen's d | **0.797854** | 1.522006 | **NO** (×1.91 short) |
| misretrieved | **183 / 223** | 73 / 223 | **NO** (×2.51 worse) |

### 2.3 The arms (§2.2's closed list — nothing else was computed)

| arm | role | acc | Cohen's d | coverage | misretr. | hubness |
|---|---|---|---|---|---|---|
| **A (â, b̂, m̃) raw** | **GATING** | 0.1905 | **0.7979** | 0.9417 | **183** | PASS |
| B (â, b̂, m) no null subtraction | control | 0.2143 | 0.6794 | 0.9417 | 178 | PASS |
| C (â, b̂, m̃) Ledoit-Wolf | non-gating | 0.2048 | 0.8047 | 0.9417 | 180 | PASS |
| D m̃ alone | diagnostic | 0.0905 | 0.4336 | 0.9417 | 204 | PASS |
| *A under σ_emb* | *non-gating σ column* | *0.2048* | *0.7763* | *0.9417* | *180* | *PASS* |
| `m1a__v3_sided` (incumbent) | — | 0.6726 | **1.5220** | 1.0000 | **73** | PASS |

Chance = 0.067. The signature carries **real class signal** (2.8× chance) and is far
below the incumbent. **The hubness gate PASSES on every arm** — the failure is not a hub
collapse and no sink is suppressing the score.

### 2.4 The σ reading — pre-declared as open, now closed by its own column

The pre-registration named exactly one genuinely open reading (the σ-parameterization)
and pre-declared a **non-gating sensitivity column** for the alternative **before any
number existed**, so a σ-sensitive verdict would be adjudicable without a re-run.

Gating σ: **d 0.7979**. Alternative σ_emb: **d 0.7763**. Both are less than **half** the
incumbent's 1.522006 and both fail the kill rule on both legs. **The open reading cannot
change the verdict.**

---

## 3. §7 adoption conditions — computed pass/fail FACTS

| condition | value | threshold | pass |
|---|---|---|---|
| 1. Cohen's d ≥ 1.772006 | 0.797854 | 1.772006 | **FAIL** |
| 2. misretrieved < 73 | 183 | 73 | **FAIL** |
| 3. hubness gate | skew 0.41 / H 0.920 / max-pred 0.076 | 3.0 / 0.70 / 0.25 | **PASS** |
| 4. coverage not *materially* narrower | 0.9417 vs 1.0000 | — | **OWNER-RESERVED** |
| 5. full probe battery | not run | — | n/a (kill test) |

**§7 ALL-PASS: FALSE**, on conditions 1 and 2, independently and terminally.

---

## 4. §5 pre-registered predictions, in registered form (DESCRIPTIVE — never gating)

| prediction | result |
|---|---|
| **P1** sibling γ-distance < clip-to-own-null, per n≥4-eligible class | **9 / 28** classes (clip level: 155/189 = 82%). **Fails as registered.** |
| **P2** sidedness recoverable from s-asymmetry | rank AUC **0.1017** — see below |
| **P4** one-sided mass early in σ | m̃-centroid **0.3263** (one-sided) vs **0.4626** (two-sided) → **HOLDS** |

**P2, stated precisely.** AUC 0.1017 is **not** "no separation" (which would be 0.50).
The two sidedness populations are **strongly separated** — a classifier on the *negation*
of the registered statistic scores **0.8983** — but in the direction **opposite** to the
one the registered statistic's phrasing implies. Sidedness **is** recoverable from the
asymmetry; the registered statistic's sign is inverted relative to the prediction. **No
mechanism claim is attached.**

---

## 5. Cycle-1 tracks (unchanged; both terminal)

**Phase 2 · E1 (delta vector) — §4.1 KILL** (`72a5bd4`): d **0.358190** vs 1.522006;
**209/223** misretrieved vs 73; coverage 1.0000; **hubness FAIL**. E2/E3 did not run.

**Phase 1 · motion — §3.4 ACCEPTANCE FAILED** (`2de4835`): injected trajectories
**29/35** pass; a post-hoc oracle splits the 6 failures into 3 noise-limited
(construction) and **3 real metric failures at oracle-VALID rungs**. Reversal: 102 → 12
insensitive → 57 undefined-ungradable → **33 graded**; leg A1 33/33, leg A2 17/33, leg B
**22/33**, joint 11/33 → **fails on leg B alone**. The exam was **not run**. M1b_flow /
M1c_flow remain analysis-tier. No second attempt (§3.6, §9).

**Item-5 readout** (added this cycle; pure readout of persisted records, no re-run): the
3 oracle-valid failures failed on **`ty` × 2, `log_scale` × 2**. **Every failure across
all six failing cells is an AMPLITUDE failure, never a correlation failure** — the lowest
correlation on any failing channel anywhere is **0.9655**, against a 0.90 floor, while
amplitude misses by 12–18%.

---

## 6. Owner-reserved matters

### Adjudicated by the owner before cycle 2 (both from cycle 1)

- **(a) E1's whitening confound** — adjudicated: the E1 KILL binds the
  candidate-as-specified; the hypothesis was recorded **unadjudicated**, prior lowered.
  Cycle 2 (E1′) carried the hypothesis and **adjudicated it**.
- **(b) Reversal leg A2** — adjudicated: an invalid-in-context operationalization. Phase-1
  verdicts unaffected (leg B and the injected test fail independently). Codified as
  OPERATIONS **§8.5**: *a ported threshold needs a validity guard in its destination
  context, or the leg it gates is advisory.*

### Escalated by cycle 2 (one, and it is moot)

- **§7 condition 4 — "not *materially* narrower" coverage.** "Materially" is not a
  threshold and the executor did not invent one (§2.7: ambiguity → escalate, never
  choose). **The fact:** coverage **0.9417** = 210/223, with 13 undefined rows — **12
  low_D + 1 empty core mask**. The shortfall is entirely a consequence of the **frozen
  §1.2 min-D guard**, which the γ-signature triggers and E1's delta did not (the delta
  contains no D; every γ channel divides by it). **Moot for the §7 call:** conditions 1
  and 2 fail independently and terminally.

### Facts bearing on already-adjudicated matters (recorded, reopening nothing)

- **Arm C (Ledoit-Wolf) and escalation (a).** E1's whitener used an **executor-chosen**
  `eig_floor_ratio = 1e-6`, and its whitened no-subtraction control scored **0.0628**
  (chance) against a raw control's **0.6054**. Arm C substitutes Ledoit-Wolf, which has
  **no free parameter** (δ = 0.007576, computed from the data; condition number 1.968e7 →
  4.062e3; no eigenvalue floor). Under it the whitened arm scores **d 0.8047 / acc
  0.2048** — **indistinguishable from the raw gating arm** (0.7979 / 0.1905), not
  collapsed to chance. The mechanism is isolated in a unit test: an eigenvalue floor
  divides near-null directions by `sqrt(floor)` and amplifies pure noise to unit variance;
  Ledoit-Wolf divides them by `sqrt(δ·m)` and leaves them small. **Stated without
  extension:** a parameter-free whitener does not flatten this signature, while the
  floored whitener flattened E1's delta to chance. **Escalation (a) is already
  adjudicated and E1-as-specified does not re-run; nothing here changes that.**

- **e0's descriptive curves are degenerate.** `e0_anatomy.py` resampled **each channel by
  its own arc length**, which maps any **monotone** channel to a straight line for every
  clip (proved by unit test). â is near-monotone by construction, so E0's â curves are
  near-linear ramps corpus-wide. **E0 is non-gating and no verdict ever rested on it** —
  but the owner's adjudication of escalation (a) cited **"P2 AUC 0.523"** as one of three
  descriptive wounds, and that figure was measured through this reparameterization. P2 is
  recomputed above in E1′'s registered form (0.1017 → 0.8983 on the negated statistic).

---

## 7. Advisor channel — UNAVAILABLE in cycle 2 (disclosure)

Cycle 1 ran with an advisor channel (six consultations, C1–C6, logged in
`CONSULTATIONS.md`, including three self-corrections). **In cycle 2 the advisor tool
returned "unavailable" on the first call and was not retried. No consultation exists for
E1′.** Its readings are **executor readings, reviewed by no one.**

Compensating discipline actually applied, and verifiable in the git history: every
reading not fully determined by frozen text was **pre-declared with its derivation in the
Part-2 pre-registration commit (`aace78d`), before any candidate number existed**; the one
reading that remained genuinely open (σ) carried a **pre-declared non-gating sensitivity
column** (which subsequently closed it); and the two structural weaknesses of the
registered IV design (the m̃ ≡ 0 tautology) were **disclosed before the IV ran**, not
after its result was known.

---

## 8. What was NOT run, and why

- **E2, E3** — forbidden by the §4.1 kill (cycle 1) and by §2.7 (cycle 2).
- **The Phase-1 exam** — §3.4: *"the exam is not run on a metric that fails constructed
  truth."*
- **Any motion work in cycle 2** — §2.7. The item-5 readout is a readout of persisted
  artifacts; no probe was rebuilt and no flow recomputed.
- **The full §7 probe battery** — E1′ is a kill test, not an adoption run.
- **Any rescue variant, any repair, any threshold adjustment** — terminal by rule.

---

## 9. Integrity

- **No frozen number was changed at any point, in either cycle.** `gates.yaml` is
  byte-identical since `8c38833`; `certify/bars.yaml` is untouched; the `eval/v3.0.0` tag
  (`50c0270`) was never reopened.
- **The certified package is byte-identical.** Inside
  `src/diffusion/transition_eval/`, the number of files this branch modifies **outside**
  `workbench/` is **zero** (verified: `git diff --name-only origin/main...HEAD --
  src/diffusion/transition_eval/ | grep -v workbench/` → empty). The branch's complete
  footprint is exactly four paths: `src/diffusion/transition_eval/workbench/`,
  `outputs/eval/workbench*/` (artifacts), `tests/test_workbench*.py`, and the repo
  `CHANGELOG.md`.
- **155 tests pass** (139 from cycle 1 + 16 new), 1 skipped.
- **Determinism:** the full E1′ run was executed **twice**; every gating number (all four
  arms' accuracy, Cohen's d, coverage, misretrieved; both IV accuracies) is
  **bit-identical across re-runs**.
- **Two unit tests are evidence for claims the pre-registration rests on**, not shape
  checks: per-channel arc length linearizes any monotone channel (why that σ reading was
  rejected **by derivation**), and `make_lerp` is idempotent on its own endpoints (why
  every "nothing" object has m̃ ≡ 0, disclosed **before** the IV ran).

### Commit hashes (cycle 2)

| commit | content |
|---|---|
| `2ac90d7` | Part-1 record amendments (both escalations adjudicated; owner spec errors; item-5 readout; OPERATIONS §8.5) |
| `aace78d` | **Part-2 pre-registration FROZEN** — before any candidate number |
| `c4fce9b` | `e1prime.py`, `iv.py`, `lw.py`, `build_iv2.py` + 16 tests |
| `6930eae` | **raw min-D floor FROZEN** (corpus-only) — before any candidate distance |

---

## Artifacts

`E1PRIME_DIRECTIVE.md` (owner authority, verbatim) · `E1PRIME_AMENDMENTS.md` (Part 1) ·
`E1PRIME_PREREG.md` + `gates_e1prime.yaml` (frozen Part 2) · `CONSULTATIONS.md` (C1–C6 +
the cycle-2 advisor-unavailable disclosure) · `outputs/eval/workbench/e1prime/RECORD.md`
+ `e1prime.json` + `mind_raw_frozen.json` + five distance matrices ·
`outputs/eval/workbench/phase1/RECORD.md` + `item5_channel_readout.json` ·
`outputs/eval/workbench/phase2/` (E1) · `$WB_CACHE/iv2/` (37 hard cuts + 37 lerps).
