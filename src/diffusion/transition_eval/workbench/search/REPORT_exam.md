# Exam-design deliverable — an instrument datasheet for retrieval metrics

**The reframing.** The goal here is not a score but the exam's *explanatory, diagnostic,
measurement, and representation power*. So the exam is redesigned as an **instrument
validation datasheet**, not a leaderboard grade. A metric is a measuring instrument; the
exam is its calibration certificate. A real datasheet reports *range, resolution,
systematic bias, per-region trust, and failure modes* as **separate, non-collapsible
fields** — never one composite number. That is the whole design.

## 1. What the current exam misses (the foundations)

Instrument validation has four **orthogonal** axes that the current exam (LOO 1-NN
accuracy + Cohen's d) conflates or omits:

| axis | current tool | the gap |
|---|---|---|
| **Reliability** | (workbench hubness gate) | not in the certified exam |
| **Construct validity** | 1-NN accuracy | one point on the ranking curve; discards the rest; variance explodes at n≤5; chance is size-dependent and uncorrected |
| **Causal validity** | content-invariance audit | **fires at 0.82 (alarm 0.4) but is NON-GATING** — the certified metric's "style" score is documented to be confounded with raw content, and nothing fails on it |
| **Sensitivity / power** | — | σ_seed/MDE unmeasured: the exam cannot say how large a delta must be to be real |

Cohen's d is **confound-blind**: content-driven separation inflates it exactly as much as
style-driven separation. And the fundamental tension: the prior workbench proved that
hard-gating content-invariance via endpoint-normalization **collapsed** discrimination
(E1/E1′ KILLs). So causality must be controlled *without* representation surgery.

## 2. The design — six datasheet fields

The barred quantity is **content-controlled discrimination, gated by its own resolution
floor**, built in four moves:

**(a) Ranking statistic = mAP@R** (sees the full top-weighted ranking, not just top-1),
always reported as **skill = (mAP − null)/(1 − null) ∈ [0,1]** (0 = chance, 1 = perfect),
where `null` is a **size-preserving label-permutation** mean (2000× shuffles preserving
the class-size histogram). Skill is comparable across strata, metrics, and class sizes —
which raw mAP@R and 1-NN accuracy are not — and it kills small-n chance inflation.

**(b) Causality via content-matched HARD NEGATIVES** (the key mechanism, replacing the
owner's positive-deleting mask). For each anchor, the gallery = all same-class positives
**+ the R hardest content-confounded negatives** (the R most content-similar
different-class clips, content-sim = DINO endpoint cosine). The controlled contrast asks
the deployment-relevant causal question directly: *does the metric rank a same-style,
different-content clip above a same-content, different-style clip?* A content-shortcut
metric ranks the content-twins first and scores at chance; a style metric survives.
**Positives are never deleted → no coverage collapse** (dissolving the owner's hole #2).
Only the *gallery* is reshaped, never the representation → fully respects the
endpoint-normalization KILL.

**(c) PASS = conjunctive & resolution-derived.** PASS iff the **Politis–Romano subsample
lower-95%-CI of the controlled excess exceeds the resolution floor** (= 2× the
permutation-null std) **AND** the permutation **p < 0.01**. The margin comes from the
exam's own null geometry, not from what our metrics happen to score (anti-overfit). The
subsample CI (delete-d, √(m/n)-rescaled) avoids the with-replacement bootstrap's
distance-0 self-retrieval artifact.

**(d) Mask-validity honesty (owner hole a).** DINO is itself a content proxy. The
datasheet scopes its causal claim explicitly: *"genuine style discrimination under
DINO-measurable content."* Multi-proxy triangulation (add color/luma + raw-pixel proxies)
and copy-twin proxy-calibration are **specified as PENDING fields** (the copy-twins are
generation-domain, outside the 223-clip corpus) — scoped, never silently claimed.

**The six datasheet fields:** Reliability (hubness + hub identity), Validity (uncontrolled
skill U), Causal-bias (controlled skill Cn + PASS), Shortcut (Δ = Cn − U per class),
Scope/Trust-map (per-class TRUSTED/WEAK/UNRATABLE), Failure-modes (hub class, confusion,
UNRATABLE ledger).

## 3. Validating the certifier BEFORE trusting it

Per the discipline "validate the certifier first," three controls ran before any verdict:
- **Chance calibration:** a random distance matrix scores **at null** (excess −0.002,
  z −0.1, p 0.81) and its PASS gate is **FALSE** — the exam does not certify noise.
- **Convergent validity:** the exam independently reproduces m1c's **polygon hub** (0.58)
  and m1a's **content confound** (corr 0.65 vs the cert's 0.82 partial).
- **Bug caught:** the controls exposed a real methodological bug (with-replacement
  bootstrap inflating CIs via duplicate self-retrieval) *before* it could produce a false
  PASS — fixed with the Politis–Romano subsample CI. The acceptance test confirms:
  random-D → PASS **False**, m1a boot_lo (0.547) **< observed** (0.577).

## 4. The datasheet, run blind on all 6 metrics

| metric | PASS | causal skill Cn | raw skill U | shortcut Δ | z | hubness | hub class | TRUSTED/WEAK/UNRAT |
|---|---|---|---|---|---|---|---|---|
| m1a incumbent | ✓ | 0.545 | 0.460 | +0.085 | 56.6 | PASS | shadow_smoke | 23 / 6 / 0 |
| **m1a improved** | ✓ | **0.682** | 0.545 | **+0.138** | 59.7 | PASS | shadow | **27 / 2 / 0** |
| m1b incumbent | ✓ | 0.412 | 0.137 | +0.275 | 10.9 | PASS | portal | 7 / 6 / 1 |
| **m1b improved** | ✓ | **0.533** | 0.186 | **+0.347** | 14.3 | PASS | gas_transf | **10 / 3 / 1** |
| **m1c incumbent** | **✗** | 0.094 | 0.011 | +0.084 | 3.1 | **FAIL** | polygon | 2 / 16 / 0 |
| **m1c improved** | ✓ | **0.326** | 0.081 | **+0.245** | 10.9 | PASS | polygon | **6 / 12 / 0** |

**What the exam explains (its whole point):**
1. **The improved metrics win on the CAUSAL axis, not just raw** — Cn: m1a 0.545→0.682,
   m1b 0.412→0.533, m1c 0.094→0.326. The improvements are content-robust discrimination,
   not shortcut-inflation.
2. **Every metric's Δ (Cn − U) is POSITIVE** — discrimination survives (sharpens under)
   content control; none is a pure content shortcut. The improved metrics have **larger
   Δ** → they are *more* content-robust than the incumbents. The exam was built to be able
   to expose the opposite and didn't have to — the strongest evidence it isn't overfit to
   our metrics.
3. **The exam correctly fails the one broken metric** — m1c incumbent is the only
   PASS=False (z 3.1 doesn't clear the resolution floor) *and* hubness-FAIL (polygon sink).
   It independently confirms our m1c repair (m1c improved PASSes, z 10.9, hub PASS).
4. **Failure modes are surfaced per metric** — distinct hub classes (m1a: shadow_smoke/
   shadow appearance cluster; m1b: portal/gas; m1c: polygon), and per-class trust maps show
   every improved metric gains trusted classes.
5. **Corpus-vs-metric:** the ensemble ceiling flags **no eligible class as fully
   corpus-limited** — every eligible class is discriminable by at least one metric.

## 5. Honest limitations (scoped, never silent)

- **Mask validity:** the causal claim is scoped to **DINO-measurable content**. Multi-proxy
  triangulation (color/luma, pixel) and copy-twin proxy-calibration are **specified PENDING
  fields** — the copy-twins are generation-domain (outside the corpus), the exam's one
  toe-hold on the domain gap, awaiting their cached features.
- **Domain gap:** like the current exam, this certifies discrimination on **real corpus
  clips**, not on generations (untestable without generation-domain labels).
- **Power in perturbation units, not model-ELO:** with no seed data, the honest power
  claim is the resolution floor + permutation significance + subsample CI. A pre-registered
  perturbation-response curve (ε_MDE) is specified as the seed-free MDE proxy; mapping ε to
  real model deltas needs seed pairs (the one deferred calibration).

## 6. Why this is a better exam
- **Causal:** the content control is *inside* the barred quantity (gating), not a
  disclosed 0.82 footnote — and done without the endpoint-normalization that killed
  discrimination.
- **Powered:** every field is chance-corrected (permutation null) with uncertainty
  (subsample CI) and a resolution floor — deltas are *callable*, not just displayed.
- **Explanatory:** a datasheet of raw, paired fields (validity / causal-bias / shortcut /
  trust-map / reliability / failure-modes) — a metric is certified *per region*, with its
  failure mode named.
- **Honest:** UNRATABLE taxonomy over silent drops; scoped mask-validity; no composite
  number to game; and it validated itself (and caught its own bug) before issuing a verdict.

Reference implementation: `workbench/search/exam_design.py`; datasheet artifact:
`outputs/eval/workbench/search/exam_datasheet.json`.
</content>
