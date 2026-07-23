# Eval Ladder v3 — Comprehensive Report

*(exp_061–exp_066 · campaign of 2026-07-16/17 · results certified under
transition-eval v3.0.0 + amendments 1–3 · ic3 sections finalized 2026-07-17)*

## 1 · Philosophy: the tier-first ladder

The question behind the ladder: **where does transition ability actually come
from — the base model, endpoint conditioning, per-class weight specialization,
or in-context demonstration?** v3 reorganizes everything around one principle:
*one generation manifest per trained model, tier = (was the class trained?) ×
(which split band do the endpoints come from)*:

- **base** (no tiers): P = prompt-only · PE = prompt+endpoints, keyed by
  owner-final sidedness (prefix-only iff one_sided).
- **specialists** (11 per-class LoRAs): SEEN (own train clips) · UNSEEN·own
  (own test clips) · UNSEEN·foreign (donor classes' test clips, prefix-only,
  sidedness-matched — "endpoints carry no information about the transition").
- **ic3** (in-context generalist): A held-in (trained class × train clips) ·
  B unseen (trained class × test clips) · C zero-shot (holdout class) ·
  X foreign (same donor pairs as the specialists → paired twins).

Universal rules, frozen before any adapter generation: no adapter ever trains
on a test-band clip; conditioning keyed by the owner-validated taxonomy
everywhere; seeds 42/43/44; paired items across all models; legacy rung names
(R0…R5, R3X/R4X) survive only as output-directory aliases. Discipline:
pre-registration + fail-forward (nothing adjusted after outcomes), certified
instrument only, owner decides all instrument matters.

## 2 · Data

- **Corpus**: 222 clips / 39 classes, std121 contract (480×640×121@24),
  `corpus_manifest.json` as single truth. live_concert_2 removed as an
  owner-ruled duplicate of live_concert_0 (cert amendment-3).
- **Split v1.1** (tag `split/v1.1`): 182 train / 40 test, 29 test-bearing
  classes; per-class seeded draw (`random.Random(f"split_v1:{cls}")`), 38
  classes byte-identical to v1; live_concert redraw only.
- **Taxonomy v2** (owner-validated 39/39): exclusive mechanism as primary
  projection (transform/overlay/cover/traverse/cut); sidedness owner-final
  (24 A-only / 15 two-sided; giant_grab & hero_flight relabeled two-sided —
  cert amendment-2).
- **IC holdout** (7 classes, verbatim from ic2): hero_flight,
  illustration_scene, gas_transformation, raven_transition, hole, seamless,
  jump — tier-C-eligible only the 4 with test clips.

## 3 · Models & training

| model | recipe | data | status |
|---|---|---|---|
| base | LTX-2 19B dev, no adapter | — | frozen |
| 11 specialists | exp_051 c2v LoRA recipe, keyed conditioning (prefix tb=2 p=1.0; +suffix iff two_sided), ckpts 250+2000 | own class split-train band | pre-existing, validated against plan |
| **ic3** (exp_064) | exp_058 recipe verbatim: rank 32/α32 attn+FFN, lr 2e-4, 5000 steps, batch 1, bf16, seed 42, reference-concat + per-pair mask (prefix 2 latent frames always, +final frame iff two_sided) | split-v1.1 train band of 32 classes = 151 clips / 403 pairs (95 two-sided / 308 one-sided), owner-final keying (only flip vs ic2: giant_grab) | trained this campaign |
| ic2 (frozen) | exp_058 | pre-alignment data (2 known violations: 12/16 "unseen" items actually trained; giant_grab mis-keyed) | comparison arm only, never headlines |

ic3 training details: 132/151 clip latents reused from exp_058's precompute,
19 fresh; inline validation ID + OOD + **control** (no trigger/reference →
must not produce a transition), per the lora-train directive. Validation
cadence was cut mid-campaign to final-round-only (observability-only change;
optimizer trajectory untouched; documented in PLAN as-run). Training ran as a
resume chain on 1h59/3h55 preemptible slots, checkpoint every 500 steps.

## 4 · Generation (all H100, uniform contract)

Contract everywhere: 480×640×121@24, 30 steps, CFG 4.0, STG 1.0 stg_v[29],
negative "worst quality, inconsistent motion, distorted, jittery", prompt =
`ICTRANS ` + type-blind caption, skip-if-exists (requeue-safe). Volumes:

| arm | videos | runner |
|---|---|---|
| base P / PE / PE-keyed / PE-ext | 150+150+54+30 | exp_061 / exp_065 (nullable-adapter grid) |
| specialist SEEN + UNSEEN·own | 132+132 (2 ckpts) | exp_062 keyed |
| specialist FOREIGN (R3X) | 132 (96 B8 + 36 ext) | exp_062 `--r3x` |
| ic3 A/B/C | 45+99+21 | exp_065 manifest grid |
| ic3 X (R4X twins) | 132 | exp_065 |
| ic2 R4/R5 (frozen) | 48+12 | exp_063 (pre-existing) |

Grid of record `ladder_items_v3.json`: deterministic builder, frozen donor
draws (`random.Random(f"ladder_v3:donors:{recipient}")`), X-extension rows
prefix-only on **both** twin sides.

## 5 · Scoring (certified instrument)

- **Code**: clean worktree at tag `eval/v3.0.0` exactly (stamp `certified:
  true` on every headline row). Amendments 1–3 authorize σ_seed/τ_copy pins
  and the corrected corpus.
- **Manifests**: exp_066 `build_eval_manifests.py` → 1,142 rows over 20
  labels. Reference conventions (pre-registered): base/specialists → own GT
  clip; ic tiers A/B/C → the demo reference actually fed; X arms → recipient's
  frozen grid reference, identical across twins (delta apples-to-apples).
  Conditions point at full endpoint clips (score.py slices 9/8).
- **Statistics**: no composite score; paired per-item deltas (twins
  co-resident per class-hashed chunk); exact binomial sign tests over trusted
  classes; σ_seed MDEs attached to every delta (amendment-1); trust map from
  the certification exam (†-classes never back a claim); `near_copy`
  reflagged at τ_copy = 0.858; synthesized control arms in the same table.
- **GPU purity**: all headline labels generated *and* scored on H100. A
  multi-pool contingency lane (A100/A30/A40/L40S, per-label caches) ran as
  insurance and is not used in headline numbers.

## 6 · Operational trajectory (what it took)

One night, ~750 videos + retrain + 1,540 scored rows, on a preemptible shared
cluster. The five lessons worth keeping:

1. **Short walltime beats PLANNED windows** — 3h55 jobs starved while a
   59-min job backfilled; resubmitting everything at 1h59 started in minutes.
2. **Validation cadence vs chunked walltime** — a 30-min initial validation
   round per resumed chunk nearly ate the training; cut to final-round-only.
3. **The certified stamp is unforgiving by design** — scoring from a
   code-identical *branch* stamped UNCERTIFIED ("HEAD tag is None");
   everything rescored from the exact tag.
4. **Zombie nodes** — ccc0423/24 accepted jobs and produced zero bytes
   (hung project-FS mount); detected via log mtimes, excluded, canary-probed.
5. **Cold shared caches race** — concurrent scorers corrupted npz entries
   (non-atomic writes); per-label caches fixed it structurally.

## 7 · Results

### Contrasts (certified; Δ over trusted classes; MDE-gated)

**C1 — endpoint conditioning (R1−R0, 50 paired items): large, near-universal.**
app_ref +0.229 (21/22 classes, p<0.001), margin +0.114 (18/22, p=0.004),
camera match better 9/10 (p=0.021). Every Δ ≫ MDE.

**C3 — specialist overfit gap (SEEN−UNSEEN @2000, class-level):**
app_ref 0.845 vs 0.706, margin 0.306 vs 0.237 — the seen-item ceiling is
memorization-adjacent (near-copy ~98–100% on own items, definitional).

**C4 — specialist value over conditioned base (R3−basePE, same items):**
appearance channels null (app_ref −0.020, 2/9 classes, p=0.18); **seam
integrity transformed: max_seam_z −10.8, 10/11 classes improved, p=0.012.**
Specialists make transitions *coherent*, not more class-like, at unseen
endpoints.

**Specialist FOREIGN collapse (tier table):** app_ref 0.205 (below the 0.36
control floor), margin −0.141, near-copy 0% — specialist weights do not
transfer their effect onto foreign endpoints.

**C5 (PRIMARY) — ic3-B vs specialist UNSEEN, same items:** on the
cross-family-comparable channel the generalist **matches the specialist:
margin Δ −0.018 (2/5 classes, p=1.0, below the 0.037 MDE) — parity on
identical unseen items.** What differs is *how* each gets there: the
specialist is memorization-adjacent (near-copy 100%, copy_max Δ +0.647,
p=0.031) while ic3 synthesizes (near-copy 3%); and the specialist keeps its
seam-integrity edge (max_seam_z Δ +18.0 against ic3, though sign-weak 4/6,
p=0.69, heavy-tailed). app_ref Δ −0.415 is **not interpretable as a quality
gap**: the arms score against different references by pre-registered
convention (own GT vs demo reference) and r3's app_ref rides its own-GT
near-copy inflation — margin is the comparable column.

**C6/C7 — zero-shot (descriptive, n≤4 classes):** in-context zero-shot onto
held-out classes does not clear the conditioned base — C6 margin −0.048
(1/3 classes), and ic3-C margin 0.038 sits barely above the control floor
(−0.01). Specialists beat zero-shot on their own classes (C7 margin +0.148,
n=2 trusted). One-reference in-context learning is not (yet) class-free.

**C8 — ic3-B over conditioned base (same items):** appearance-family
channels null-to-negative (margin −0.016, 7/19, p=0.36, <MDE), but two
transformations: **seam integrity max_seam_z −19.4 (21/25 classes improved,
p=0.001)** and **synthesis instead of copying (copy_max −0.591, 0/25,
p<0.001; near-copy 3% vs 100%)**. The generalist's value over conditioning
is not "more class-like" — it is *coherent, non-degenerate transition
structure* on content it never saw.

**C9 — effect transfer onto foreign endpoints (R3X vs ic3-X paired twins,
B8 confirmatory, pre-registered direction R3X>R4X): CONFIRMED.** app_ref
Δ +0.042 and margin Δ +0.094, both **6/6 recipients, p=0.031, ≫MDE** (twins
share an identical frozen reference — apples-to-apples). Both arms collapse
on foreign endpoints (r3x margin −0.141, ic3-x −0.240, both far below every
in-class arm), but the specialist retains a small, universal appearance
residue. The exploratory extension agrees (margin +0.108, 3/3). In this
prefix-only foreign regime, *neither* weights nor one demo re-apply the
effect — and in-context re-application loses even the residue.

**C10 — value of split alignment (ic3 vs ic2):** the sharpest reading is
C10b: **ic3 on genuinely-unseen items matches ic2 on items ic2 had trained
on** — app_ref Δ +0.010, margin Δ −0.002 (4/7, p=1.0, <MDE). Decontaminating
the training split cost nothing measurable; ic2's seam looks better in the
mean (Δ +12.4) but is sign-weak (5/8, p=0.73, heavy-tailed). Descriptive by
pre-registration (contaminated baseline).

**C11 — ic3 held-in vs unseen (A−B, class-level, descriptive):** margin
0.240 vs 0.187, app_ref 0.347 vs 0.408 (different class sets; tier-A includes
7 test-less stand-ins). The generalist's held-in advantage is **small and
channel-inconsistent** — nothing like the specialist's overfit gap (C3:
0.845→0.706 app_ref, 0.306→0.237 margin on matched conventions). In-context
training does not memorize its training items the way per-class weights do.

### Near-copy diagnostic (τ=0.858)
Own-item arms ~100% (their GT is in the corpus — definitional); ic2 tiers and
specialist-FOREIGN 0% — generalists synthesize rather than reproduce, and no
content paste-on is flagged on foreign endpoints. **ic3: A 0% · B 3% (3/99)
· C 0% · X 0%** — the aligned generalist synthesizes across every tier,
including its own held-in items. Controls flag ~47% (the detector sees the
degenerate arms).

## 8 · Limitations & flags

- obj_match / cam channels have tiny trusted-class counts (†-heavy) — sign
  claims rest on app_ref/margin/seam.
- seam_z has heavy tails in control/hold arms (means dominated by outliers;
  medians tell the same ordering).
- C6/C7/C10 are descriptive (n small or contaminated baseline); C9 extension
  is exploratory; the hero_flight σ_seed recheck (amendment-2) shifted no
  conclusion.
- Every headline label (all 20, including all five ic3 labels) was generated
  *and* scored on H100 against the exact `eval/v3.0.0` tag — 2,134 rows, 0
  error rows. The L40S/mixed-pool insurance lanes ran in parallel and are not
  used in any headline number. Cross-GPU check: the v4 lane's carried
  `app_ref_v3` reproduces certified v3 `app_ref` to 0.00000 on shared H100
  rows (A100 lane drift 7e-5 ≈ MDE/300).
- C5/C8 app_ref crosses reference conventions (own GT vs demo reference) —
  margin is the pre-registered cross-family channel; app_ref deltas there are
  reported but not claim-bearing.

*Full numbers: `outputs/eval/ladder_v3/_contrasts/{contrasts.md,json,
tier_table.md}`. Compact presentation version: `SUMMARY_v3.md`.*
