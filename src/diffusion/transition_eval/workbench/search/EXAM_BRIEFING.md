# Exam-design exploration — advisor briefing (foundations-first)

**Owner directive (verbatim intent).** Design the exam ITSELF. The goal is **NOT to
maximize a score** — it is to **maximize the exam's explanation / diagnosis / measurement
/ representation power**. Work close to foundations; be comprehensive and explanatory;
this is deep design, not fast experimentation. The owner's proposal below is a *starting
idea, not a constraint* — don't limit yourself.

## What the exam is FOR (the construct)

The exam certifies the **RULER, not the model** (SPEC §6). A "metric" (m1a appearance,
m1b camera, m1c object) is a pairwise distance on clips. The exam asks: **does this
deployed pairwise statistic validly discriminate the construct "effect-transition class,"
and can we trust each per-class number?** Today that's operationalized as leave-one-out
1-NN retrieval accuracy + Cohen's d (within vs cross-class separation) + coverage +
per-stratum macro recall + (workbench-only) a hubness gate. Inferential stance is **paired
& raw** — no composite score, deltas on identical inputs are the unit.

## The current exam — mechanics + the two "exams"

- **Certified (`eval/v3.0.0`):** `report.retrieval_eval` → accuracy_1nn, Wilson95,
  coverage, chance, per-class recall, confusion, within/cross means, **Cohen's d**; plus
  an R2 pool-margin readout (M2b), a **trust map** (per-class per-metric boolean; recall
  ≥0.5 on n≥4; n<4 auto-untrusted; 2 singletons permanently untrusted), and per-tag macro
  recall. **NO hubness gate in the certified exam.** 8 frozen certification **bars**
  (Bar 1 = M1a d≥1.5; Bars 2–7 = sibling>control, splice/copy detection, camera
  reversal-sensitivity, endpoint-swap + hard-cut, copy-twins; Bar 8 = determinism).
- **Workbench (`eval/metric-workbench`, CLOSED):** added the k=10 hubness gate + §7
  adoption conditions + the IV1/IV2 causal preconditions (effect-vs-nothing,
  snap-vs-nothing). Its E1/E2/E3 appearance ladder tried to build a content-invariant
  metric by subtracting a rendered-lerp null — **all KILLed** (endpoint-normalization
  destroyed discrimination on this corpus).

## Why Cohen's d exists: accuracy saturates
1-NN accuracy on 39 classes / chance 0.067 saturates and the old 0.80 floor was
calibrated on a different (47-clip/11-style) corpus and never re-derived. d is the
saturation-immune effect size `(cross−within)/pooled_std`.

## The FOUNDATIONAL HOLES (what a better exam should fix)

1. **Content shortcut — the biggest under-instrumented axis.** The certified
   content-invariance audit (within-class partial corr of M1a style-similarity vs raw
   endpoint/content similarity) fires at **0.82** (alarm 0.4) but is **NON-GATING**. So
   the certified appearance metric's "style" score is strongly confounded with raw content
   similarity, disclosed but never failed on. Poster child: nature_bloom's lerp control
   scored *above* its sibling. **A better exam should make the causal content-control part
   of the barred quantity, not a side-note.** (This is exactly the owner's proposal.)
   BUT — the fundamental tension: hard content-invariance via endpoint-normalization
   KILLED discrimination (E1/E1′). So the content control must be a **gallery/mask**
   mechanism (control what you retrieve against), NOT representation surgery.

2. **Measurement power is unquantified (σ_seed/MDE PENDING).** The exam certifies the
   ruler discriminates styles on real clips, but **cannot yet say how large a model A-vs-B
   delta must be to be real** (no seed-noise floor measured). A "measurement power" exam
   should quantify a minimum-detectable-effect / give calibrated uncertainty (permutation
   null, bootstrap CI).

3. **Small-class / coverage / chance honesty.** 223 clips, 39 classes, **23 of 39 have
   n≤5, 10 have n<4, 2 singletons** (unretrievable). NaN rows dropped from retrieval.
   1-NN throws away the full ranking. A better exam should: use the full ranking
   (top-weighted), be robust to class imbalance + small-n chance inflation (permutation
   null), and return **UNRATABLE** honestly rather than a fake number when support is
   too thin — not silently drop.

4. **Motion metrics near-dead on this corpus.** m1b/m1c live on a **motion-scarce**
   (median per-pair translation 0.297px), **full-frame-effect** (outlier-area median
   0.875) corpus — camera motion is at the instrument's noise floor. A measurement-power
   exam should distinguish "metric is bad" from "corpus can't test this."

## The corpus reality (bearing on exam validity)
- 223 clips / 39 classes, class sizes {1:2, 2:6, 3:2, 4:9, 5:4, 6:3, 7:1, 8:3, 9:2, 10:4,
  12:1, 13:1, 15:1}; long-tailed, small. 26 one-sided / 13 two-sided.
- Strata are overlapping **class-level tags** (camera 18, object 24, style 10) — not
  per-clip labels.
- Motion-scarce, full-frame effects → a content-MASK exam risks **coverage collapse**
  (little static-background complement to mask against; endpoints semantically close,
  which is the source of the 0.82 confound).

## Label reality
- **Exactly ONE label per clip** — its style class. camera/object/style are overlapping
  class-tags, no per-clip motion labels.
- **No pair-level similarity judgments, no human ratings** except the **11 human-verified
  copy-twins** (Bar 7). All other "truth" is constructed (splices/swaps/hard-cuts/
  reversals/lerp-nulls). The M4 rubric judge is **uncalibrated** (O9 open) → no headline.

## The owner's proposal (a starting idea — the "null-excess masked-gallery mAP@R")
1. Cache the metric's distance matrix. 2. **Content mask** (frozen pre-run): per anchor,
drop gallery items above a fixed DINO content-similarity quantile. 3. **mAP@R** per anchor
on the masked gallery, macro over n≥4 surviving-positive classes. 4. **Null:** ~1k label
permutations on the same masked galleries → empirical chance. 5. **Barred scalar** =
excess over null, clip-level bootstrap lower-CI > frozen margin. 6. Coverage < floor →
**UNRATABLE**, not pass/fail. Free diagnostics same run: unmasked mAP@R → **retention
ratio** (shortcut dependence), per-class excess → trust map, within-family vs
cross-stratum confusion, hubness skew from k-occurrence.
Owner's honest **3 residual holes**: (1) **mask validity** — DINO similarity is itself a
proxy; a shortcut in color-grade/codec-cadence DINO doesn't encode slips through; the exam
is only as causal as its mask. (2) **coverage collapse** — template-heavy corpus may
delete most same-class positives → UNRATABLE means the corpus, not the metric, is the
blocker. (3) **domain gap** — it certifies discrimination on corpus clips, not generations.

## Assets available to implement/validate any design (executor side)
- All 6 distance matrices (m1a/m1b/m1c incumbents + my improved m1a 0.81 / m1b 0.41 /
  m1c 0.22), 223×223, apples-to-apples on the frozen kernel.
- Per-clip DINO CLS features [121,768] (for the content mask / an alternative content
  channel), endpoint anchors, core masks, CoTracker tracks.
- The frozen retrieval kernel; ability to build mAP@R, permutation nulls, bootstrap CIs,
  masked galleries, retention ratios, CSLS/hubness, on CPU in seconds–minutes.
- The prior workbench's KILL evidence (endpoint-normalization dead) as a design constraint.

## What "better" means here (the design targets)
A single, foundations-grounded exam (+ its free diagnostics) that maximizes:
- **Explanation/diagnosis:** says *why* a metric wins/loses (shortcut-dependence,
  per-class trust, confusion structure, hub behavior, corpus-limited vs metric-limited).
- **Measurement power:** quantifies uncertainty (null + CI + MDE) so an A-vs-B delta is
  callable, not just displayed.
- **Causal validity:** the content-shortcut control is *inside* the barred quantity, done
  in a way that doesn't kill discrimination (gallery/mask, not representation surgery),
  with the mask-validity hole honestly bounded.
- **Representation:** one honest scalar + a rich diagnostic panel; UNRATABLE when the
  corpus can't support a verdict; robust to small-n/imbalance/chance.
</content>
