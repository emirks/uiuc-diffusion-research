# Health-validation of the 3 metrics + exam (pre-certification)

**Process.** Operator/advisor campaign (Opus executes + reports; a fresh fable-advisor at
each fork makes every judgment call). Goal: from far apart, decide whether m1a/m1b/m1c and
the redesigned datasheet-exam are *artificial? grounded? overfit to the 223-clip corpus?
as simple as they can be?* — before the owner fixes on them and runs certification. Three
consultation rounds, all bars pre-declared before results. Verdict: **STOP — all three
metrics health-validated; the exam had a proven defect that was fixed and re-verified.**

Every number below was reproduced this session (base-touch m1a bit-exact, max|Δ|=0.0);
scripts + JSON in the campaign dossier (`$JOB/tmp/DOSSIER.md`, `round2_*.json`,
`round3_*.json`, `exam_datasheet_fixed.json`).

---

## 1. The exam had a real, provable defect — now fixed

The redesigned exam's headline **causal PASS gate was a mathematical no-op for any
content-monotone metric.** The controlled gallery selects the R hardest negatives *by
content_sim*; for `D_cont = 1−content_sim` the controlled top-R is identical to the
uncontrolled top-R for the observed AND every permuted labeling, so its controlled skill
≡ uncontrolled skill (measured: `U = Cn = 0.2068` exactly, Δ=0.0 exactly — a theorem, not
noise). On this corpus content carries class signal, so the above-chance gate could
**never fail a pure-content metric**. Confirmed empirically: pure DINO-content `D_cont`
scored Cn 0.207 and **PASSED** the old gate.

The gate machinery itself is sound — the content-matched negatives genuinely bite
(dose-response Δ(w) is perfectly monotone, Spearman ρ=−1.0; hard negatives cost every
appearance metric more skill than random negatives, D_cont most at −0.269). The **gate
threshold** was the problem: it tested "better than permutation chance" when the claim
needs "better than content alone on the same galleries."

**Fix (advisor-directed, ported into `exam_design.py`, verified):** the causal gate now
tests **causal excess over an explicit content baseline** —
`causal_PASS iff causal_excess = Cn − Cn_B ≥ max(0.10, 2·floor) AND paired-bootstrap
lo95(controlled-mAP difference) > 0`, with `B = D_cont`. Both arms are load-bearing
(random-D passes the CI arm; a sub-floor metric passes the excess arm) and must not be
simplified to one. The old binary PASS is demoted to a descriptive `above_chance` field.

**Three-falsifier validation of the fixed gate** (all pass as required):

| through the fixed gate | causal_excess | causal_PASS |
|---|---|---|
| pure DINO content `D_cont` | 0.000 | **FAIL** ✓ |
| independent color-histogram content `B′` (Spearman vs DINO content = 0.09) | 0.072 | **FAIL** ✓ |
| random distance matrix | −0.226 | **FAIL** ✓ |
| m1c *incumbent* (broken) | −0.076 | **FAIL** ✓ |

The ported gate reproduces every round-3 verdict exactly at n_perm=1000.

**Exam scope the owner must carry (honest limitation):** the content baseline is
**proxy-relative.** An independent color-histogram proxy scores *higher* controlled skill
than the DINO baseline on the m1a and m1c strata (color 0.279/0.229 vs DINO 0.207/0.172),
so the DINO gate **understates the content ceiling** there. Certification must
**pre-declare** DINO-only vs per-stratum max-over-proxies as its baseline — this choice
changes m1c's verdict (below). Both excesses ship in the datasheet
(`causal_excess`, `causal_excess_maxproxy`).

---

## 2. Metric verdicts

### m1a appearance — **HEALTHY, and simplifiable. Deliverable → S3 (4-channel).**
A pre-declared simplicity ladder (adopt the simplest variant within 1 SE accuracy, d≥1.522,
hubness, causal within-floor) selected **S3 = `0.5·App(centered, endpoint-debiased) +
0.5·Dyn(velocity-EMD, velocity)`** over the 6-channel `D_STACK`: horizon-4 and acceleration
add nothing (acc 0.807 = D_STACK within 1 clip; d 1.73; Cn 0.684 within-floor of D_STACK
0.682). **`D_FINAL` (k-reciprocal re-rank) is dropped** — it adds zero accuracy and amplifies
instability (a 1-clip input drift became a 5-clip output drift). Causal excess **0.478**
(0.406 even under the strengthened color baseline). Class-half stability degradation **0.003**;
S3≥S1 ordering held in 9/10 splits.
> Provenance note: the report headline 0.8117 is the *recipe's* fresh output; the cached
> `stack_cls.npz` (0.8072) was a stale build. Deliverable = the recipe.

### m1b camera — **HEALTHY. Deliverable → D_ZPR (all 3 views earned).**
Dropping either redundant rigid view costs ~0.07 recall (D_ZR 0.335, D_PR 0.345 vs D_ZPR
0.411) — below the −0.03 tolerance, so the 3 views (Z shape, P amplitude, res turbulence)
are now **verified-earned, not asserted**. Causal excess **0.307** (0.292 under the color
baseline — passes both). Class-half degradation −0.004 (out-of-sample slightly *better*).
> Usage note baked into the datasheet: never rank m1b variants by sub-floor Cn differences
> (Cn mildly prefers dropping P, but retrieval recall — the metric's job — governs).

### m1c object — **HEALTHY MACHINERY; SCOPED causal stamp (the one caveat).**
The CSLS de-hub repair is validated as real, not an artifact: CSLS-on-random-D manufactures
no recall (mean 0.022, max 0.065, both under bar); the hubness gate PASSES at k=5/10/15/20
(the k=10-circularity worry is closed); fresh rebuild reproduces recall 0.2179; class-half
degradation 0.031 (largest of the three, still within bar). It **passes the pre-declared
DINO-baseline causal gate** (excess 0.156, 3.5× floor). **But** its controlled skill (0.328)
only marginally exceeds a pure color proxy on the object stratum (0.229): excess over the
*strongest* content proxy is **0.099 — right at the 0.10 bar** (F2: the raw mAP excess is
statistically real, paired lo95 0.043 > 0, but marginal).
- **Ship with a scoped stamp**, not a full one: *"content-controlled versus the DINO
  endpoint baseline; object-motion signal is real but faint, and its margin over the
  strongest content proxy is at the practical bar."*
- Advisor **refused iteration** as a pre-refuted dead end: magnitude was already refuted 3
  ways and the ~0.22 recall ceiling is a property of this motion-scarce corpus (median
  per-pair translation 0.297px). A stronger m1c needs motion-richer data, not more metric
  engineering — an owner decision, not a campaign fix.
- Labeled hypothesis (plausible, not proven): the color/CSLS overlap is corpus confound
  correlation (color-separable classes coincide with motion-separable ones), not a color
  shortcut — m1c's inputs are camera-residualized velocity *directions* with no color access.

---

## 3. Answers to the four questions the owner asked

- **Artificial?** No. Each metric decomposes into real per-clip measurements (frame
  appearance, velocity, rigid-transform trajectory, tracklet directions) plus a
  corpus-relative fusion/correction layer (ECDF, CSLS). The corpus-relativity is the one
  deployment consequence: **certification must pin the 223-pair reference population + the
  ECDF/CSLS procedure as part of each metric's definition** (scoring a new clip requires them).
- **Overfit to this corpus?** The *answerable* part is clean: refitting every
  corpus-dependent statistic on a disjoint half of the classes barely changes performance
  (degradation 0.003 / −0.004 / 0.031). This means **not fragile to corpus composition** —
  which is weaker than "generalizes to new data" and should be stated as such. True external
  validity is unprovable on 223 clips; if it matters, pre-register a small fresh-clip set.
- **Robust on theory?** Yes for m1a/m1b (excess many multiples of the resolution floor,
  stable across fresh builds + high-precision nulls + disjoint class halves). Marginal-but-
  honest for m1c (faint signal, scoped).
- **Simplest possible?** Now yes, by pre-declared rule: m1a trimmed 6→4 channels + dropped
  the re-rank; m1b's 3 views proven each necessary; m1c's CSLS is already minimal.

---

## 4. What changed / what the owner carries into certification

- **Code:** `exam_design.py` — the causal gate replaced (old→`above_chance`; new
  causal-excess gate + `content_baseline_Cn`/`causal_excess`/`causal_excess_maxproxy`/
  `confound_susceptibility`); `random_gallery_mask` added. This is workbench (writable)
  code, **not** the certified `eval/v3.0.0` instrument. Re-emitted datasheet:
  `outputs/eval/workbench/search/exam_datasheet_fixed.json`.
- **Deliverable metrics:** m1a = **S3** (4-channel, DINOv2-base, no re-rank); m1b = **D_ZPR**;
  m1c = **CSLS(incumbent)** with a scoped causal stamp.
- **Two decisions the owner must pre-declare before certifying:** (1) the causal baseline
  (DINO-only vs per-stratum max-over-proxies — the latter flips m1c to fail by 0.001);
  (2) whether faint m1c is shippable for its use, or whether motion-richer data is warranted.
- **Do not** simplify the two-arm causal gate to one arm; **do not** rank variants by
  sub-floor Cn differences (both written into the datasheet notes).
