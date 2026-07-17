# Eval Ladder v1 — Canonical Plan & Pre-Registration

**Status:** FROZEN upon commit. This document is the pre-registration for ladder
generation and analysis. Changes after generation begins require a dated amendment
section, never an edit.
**Authorship:** designed by the campaign advisor (fable) under the `/advised`
operator(Opus)/advisor(fable) working mode; executed by the operator. 2026-07-16.
**Scope of this batch:** persistence + split freeze + scaffolds + specialist
training + R4/R5 generation. **No scoring** — all scoring blocked until the
sidedness re-annotation is owner-validated.
**Deadline context:** first-model report Aug 16.
**Frozen item grid:** [`ladder_items_v1.json`](ladder_items_v1.json) (re-derivable
via `build_ladder_items.py` against the recorded split sha256).

---

## Decisions (advisor rulings)

**D1 — Ordering: ratified, with corrections.**
(a) Item 1 code commit already done (`32b1546` on `eval/metric-workbench`, pushed);
remaining was to force-add the 3 output files (see D6, done `393f093`). (b) Items 2,
3 proceed now. (c) Commit the plan + frozen item grid BEFORE submitting any
generation — that commit is the pre-registration timestamp. (d) Item 4: all 11
specialist trainings submit now (blind conditioning makes them taxonomy-immune,
D2/D3). (e) Item 5: R4 (48 videos) and R5 gas/illustration (12) generate now; **only
R5 hero_flight (6 videos) waits for tomorrow's sidedness validation** because its
keyed suffix mask depends on the label. R2/R3 generation waits on R2 training only.
Scoring stays blocked (owner rule).

**D2 — R2/R3 conditioning: sidedness-BLIND** (prefix tb=2 p=1.0 + suffix tb=1 p=1.0,
always on). Reasons: (1) exp_051-validated recipe verbatim — keyed specialist
training (prefix-only c2v) has never been validated and is a silent new variable;
(2) makes all 11 trainings taxonomy-independent — hero_flight trains today with zero
re-train risk; (3) holds conditioning identical to R1, so R3−R1 is a pure adapter
effect. The "cleaner R3−R4" argument for keyed is illusory: R3−R4 is pre-registered
as native-vs-native anyway, and running the generalist off-mode is the known
exp_058 §5 mistake in reverse.

**D3 — hero_flight: train its specialist NOW.** Blind conditioning makes it immune
to the sidedness outcome. R5 classes need specialists most: R3(specialist) is the
per-class ceiling against which R5 zero-shot is judged (contrast C7).

**D4 — Specialists for all 11 test-bearing roster classes; skip live_concert.**
live_concert has 0 test clips (all-dup remediation) → feeds no pre-registered delta.
R4-tier classes still get specialists: they supply the R3 side of R3−R4.

**D5 — Adapter pin: exp_058 `ic2` step_05000, run in its NATIVE keyed mode.** The
roster's A/B/C exposure tiers are defined by exp_058 — pinning exp_056 `ic` would
invalidate the R4/R5 class assignment. Keying rule: **R4 uses exp_058 training-time
sidedness (the adapter's actual distribution); R5 uses the validated taxonomy
label** (no training-time label exists for held-out classes — this is why
hero_flight waits). exp_056 `ic` is not in the ladder; only a labeled post-hoc
robustness probe if the owner asks.

**D6 — Commit scope: force-add the outputs** (3 files, 88K: `results_log.jsonl`,
`exam_datasheet.json`, `exam_datasheet_fixed.json`). 335a83d reproducibility
precedent with no size risk. Done as `393f093`.

**D7 — Items / references / seeds / outputs: pinned (§4–§6).** Seeds 42/43/44
everywhere. R3/R4/R5 targets = the class's 2 split-v1 test clips (== exp_061 items,
so R0/R1 pairing already exists). R2 items = 2 deterministic train clips per class
(seeded, disjoint from reference). R4/R5 reference = 1 fixed train clip per class for
all items and seeds. Both specialist checkpoints (step_00250, step_02000) generate R2
and R3 (checkpoint-sensitivity → pre-registered robustness check). Every rung emits
SPEC §2 manifests with `twin_of` links + per-item `endpoint_seen_by_ic2` flags.

**D8 — Done-criteria (§8) and pre-registration (§7).** The confirmatory analysis is
fully pre-registered so later scoring is confirmatory, not exploratory.

---

## 1. The ladder

Six rungs, all at 480×640×121 @ 24 fps, 30 steps, CFG 4.0, STG 1.0 `stg_v`
blocks [29], negative prompt "worst quality, inconsistent motion, distorted,
jittery", prompt = `ICTRANS ` + type-blind endpoint caption — the exp_060/061
contract, identical across every rung. Report **deltas, never absolutes**;
cross-class sign tests are primary.

| rung | model | adapter | conditioning | endpoints | class signal |
|---|---|---|---|---|---|
| R0 | 19B dev | none | none (T2V) | — | prompt only |
| R1 | 19B dev | none | prefix 9f + suffix 8f (blind) | item clip | prompt only |
| R2 | 19B dev | per-class specialist | prefix 9f + suffix 8f (blind) | HELD-IN (train clip) | weights |
| R3 | 19B dev | per-class specialist | prefix 9f + suffix 8f (blind) | UNSEEN (test clip) | weights |
| R4 | 19B dev | ic2 step_05000 | reference + prefix 9f + suffix 8f **iff two-sided (exp_058 training label)** | test clip | weights + reference |
| R5 | 19B dev | ic2 step_05000 | reference + prefix 9f + suffix 8f **iff two-sided (validated taxonomy label)** | test clip | reference ONLY (class never trained) |

Interpretation pin: "reference-only" in R5 means the class is conveyed solely by the
reference clip (zero-shot class transfer) — endpoints are still conditioned,
identically to R4, or the rung would neither pair with R3/R4 nor be scorable by the
endpoint-referenced harness.

Conditioning windows (exp_060 `make_conds.py` ffmpeg recipe, exp_051 causal-VAE
rule): prefix = first 9 frames → 2 latents at position 0; suffix = last 9 frames
consumed as `num_frames=8` → final latent only. Reference (R4/R5) = full 121-frame
train clip, latents concatenated before target, clean/ts0/loss-excluded.

## 2. Adapters

- **Specialists (R2/R3), NEW — 11 trainings.** exp_051 c2v recipe verbatim: rank
  32/α32, video-attention targets, lr 1e-4 linear, 2000 steps, bs 1, bf16, ckpt
  every 250; conditions = prefix tb=2 p=1.0 + suffix tb=1 p=1.0 (both always on —
  sidedness-blind). Training set = ALL split-v1 train clips of the class. Captions:
  `ICTRANS ` + type-blind captions (reuse exp_056/057/058/060 caption files; fill
  gaps with the exp_058/060 Gemini captioner, temp 0). **This is the one pinned
  deviation from exp_051** (which used SHDWSMK + effect-specific captions): the
  conditioning mechanism is what exp_051 validated; the caption convention is changed
  to buy exact prompt parity across all six rungs. Queue: `HCESC-H100-secondary`,
  exp_051 resume-aware chain-safe sbatch pattern (2 chained copies per class).
  ~1h19m each ≈ 14.5 H100-h total.
- **Generalist (R4/R5), EXISTING.** exp_058 `ic2`
  `lora_weights_step_05000.safetensors` — pinned; record the file sha256 in every
  manifest row. exp_056 `ic` step_03000 is NOT in the ladder.

## 3. Class → rung coverage matrix

| class | R0/R1 | R2/R3 specialist | R4 | R5 | test clips | ic2 exposure of test clips |
|---|---|---|---|---|---|---|
| shadow | done | yes | yes | — | shadow_10, shadow_3 | _10 unseen / **_3 SEEN** |
| portal | done | yes | yes | — | portal_1, portal_4 | **both SEEN** |
| super_fast_run | done | yes | yes | — | super_fast_run_0, _3 | _0 unseen / **_3 SEEN** |
| shadow_smoke | done | yes | yes | — | shadow_smoke_7, _0 | **both SEEN** |
| polygon | done | yes | yes | — | polygon_4, polygon_8 | **both SEEN** |
| wireframe | done | yes | yes | — | wireframe_0, _7 | **_0 SEEN** / _7 unseen |
| animalization | done | yes | yes | — | animalization_0, _2 | _0 unseen / **_2 SEEN** |
| color_rain | done | yes | yes | — | color_rain_0, _2 | **both SEEN** |
| gas_transformation | done | yes | — | yes | gas_transformation_6, _7 | both unseen (class held out) |
| hero_flight | done | yes (blind → immune) | — | yes (WAITS on sidedness) | hero_flight_5, _6 | both unseen (class held out) |
| illustration_scene | done | yes | — | yes | illustration_scene_7, _4 | both unseen (class held out) |
| live_concert | done (train-clip item) | **no** — 0 test clips, feeds no delta | — | — | — | — |

**Contamination finding (computed 2026-07-16 from `experiments/exp_058_ic_lora_diverse_retrain/dataset/pairs.json`
× `split_v1.json`):** exp_058's B-tier exclusions were keyed to its own eval quads,
not to split v1 (which post-dates it). Result: only **4 of 16** R4 items have
endpoints truly unseen by ic2 — `shadow_10`, `super_fast_run_0`, `wireframe_7`,
`animalization_0`. The other 12 test clips were ic2 training clips. R4 still
generates ALL 16 items (pairing with R3 is non-negotiable); the analysis is
stratified (§7, C5). Every R4 manifest row carries `endpoint_seen_by_ic2:
true|false`. R5 is fully clean — A-tier classes have zero ic2 exposure in any role.

Sidedness keying for R4 (from exp_058 training labels, ALLOCATION.md): shadow_smoke
**two-sided** (suffix ON); shadow, portal, super_fast_run, polygon, wireframe,
animalization, color_rain one-sided (prefix-only). None of the 8 is in the
sidedness-conflict set, so these are stable. R5: gas_transformation and
illustration_scene are `A_only` (one-sided, prefix-only) and not in the conflict set
→ stable → generate now; hero_flight is `two_sided` + in the conflict set → waits for
validation.

## 4. Item, reference, and seed rules (deterministic, frozen)

- **Seeds: 42, 43, 44** for every item at every rung. Same item ⇒ same seeds across
  rungs. Never extend or substitute seeds mid-campaign.
- **R3/R4/R5 targets:** the class's 2 split-v1 test clips — identical to exp_061's
  R0/R1 items, so base-rung twins already exist on disk.
- **Selection RNG** (mirrors split provenance style):
  `random.Random(f"ladder_v1:{class_name}:{role}").sample(sorted(pool), k)`. Draw
  reference first, then R2 items.
  - `role="reference"`, k=1. R4 classes: pool = the class's split-v1 train clips that
    WERE in ic2 training (a trained clip in the reference role is the adapter's native
    distribution). R5 classes: pool = all split-v1 train clips (none trained; any is
    equally zero-shot).
  - `role="r2_items"`, k=2, pool = split-v1 train clips minus the chosen reference.
- **Reference is FIXED per class** — same clip for both test items and all three
  seeds, in both R4 and R5.
- The realized grid is frozen as `ladder_items_v1.json` and committed BEFORE any
  generation job is submitted.

## 5. Per-rung generation recipe and outputs

Two new experiments (repo numbering applies; expected `exp_062` specialists + R2/R3,
`exp_063` R4/R5):

- **R2/R3** — exp_051 `run_c2v_inference.py` path (trainer ValidationRunner), seeds
  42/43/44, specialist checkpoint per class. Generate at **both step_00250 and
  step_02000**. Counts: R2 = 11×2×3×2 = 132; R3 = 11×2×3×2 = 132; ≈ 8 GPU-h. Verify
  the script's STG blocks match the exp_060 contract ([29]); if exp_051 used
  different blocks, pin [29].
- **R4/R5** — exp_058 `run_ic_inference.py` path, ic2 step_05000, native keyed mode
  per §3. Counts: R4 = 8×2×3 = 48; R5 = 3×2×3 = 18 (12 now; 6 hero_flight after
  validation); ≈ 2.2 GPU-h.
- All generation jobs resumable (exp_061 skip-if-exists pattern) — required for
  `secondary`.
- **Outputs per rung:**
  1. Videos: `outputs/videos/exp_06N_<slug>/<rung>/<rung>__<class>__<clip>__s<seed>[__ckpt<step>].mp4`
     + `config_snapshot.yaml`.
  2. Manifests: `eval_manifest_<rung>[_ckpt<step>].json`, SPEC §2 schema, one row per
     video, carrying: class, clip id, seed, rung, adapter path + sha256, checkpoint
     step, conditioning mode actually applied (prefix/suffix/reference flags),
     reference clip id, `endpoint_seen_by_ic2`, sidedness key used and its source
     (exp_058-training vs taxonomy), prompt string, and `twin_of` chains linking the
     same (item, seed) across R0→R5. `reference_video` for scoring = the item's
     ground-truth clip, as in exp_061.
  3. Training side: 11 checkpoint dirs with all step-250…2000 checkpoints retained,
     W&B runs, config snapshots.
  These manifests must let `score.py` run with ZERO additional decisions once
  sidedness lands.

## 6. Dependency / ordering graph

```
DONE   32b1546 metric-search code+reports committed & pushed (eval/metric-workbench)
DONE   393f093 A1 force-add 3 search output files (88K) → pushed  [eval/metric-workbench]
NOW    A2 SPLIT_V1_FINAL declaration (stamp file + sha256, water_element_5 note) → main
NOW    A3 commit THIS PLAN + ladder_items_v1.json → main        ← pre-registration timestamp
       ── A3 must precede A5/A6 ──
NOW    A4 scaffold exp_062/exp_063 (configs, manifests, sbatch, per-class precompute)
NOW    A5 submit 11 specialist training chains (secondary, resume-aware)
NOW    A6 submit R4 (48) + R5 gas/illustration (12) generation
NOW    A7 exp_061 pairing audit: 22 roster test items × 3 seeds × 2 arms present & nonzero
DAY+1  B1 after sidedness validation: R5 hero_flight (6 videos) with validated key
LATER  C1 per class as training lands: R2+R3 generation (ckpt 250 + 2000)
BLOCKED D scoring of everything — waits on sidedness validation (owner) + harness mask
```

Safe before taxonomy validation: A1–A7, C1. Waits: B1 (hero_flight R5 keying) and all
scoring. If hero_flight's sidedness FLIPS: nothing retrains (specialists are blind);
regenerate only B1 with the flipped key; scoring masks come from the validated
taxonomy by construction.

Split-final rationale (A2): taxonomy labels are not split inputs, so label edits
cannot move it; the only open curation flag (water_element_5) touches a non-roster
class whose test band is invariant to the pull, and the per-class RNG streams
(`split_v1:{class}`) localize any future rebuild to that class alone. Declare final:
`SPLIT_V1_FINAL.md` beside the split with its sha256 + this rationale, plus tag
`split/v1`.

## 7. Pre-registered contrasts and analysis (scoring, when unblocked)

Metrics: **certified harness v3.0.0 (+ amendment-1 σ_seed/τ_copy) only** for all
primary claims; the metric-search M1 variants are uncertified (two owner decisions
pending) and may enter only as labeled secondary if certified before scoring. Metric
trust map applies (e.g., polygon's M1c-only trust). Deltas reported in metric units
with the amendment-1 σ_seed/MDE context. Tests: one-sided binomial sign test across
classes on per-class mean deltas, α = 0.05, direction pre-declared below. No post-hoc
exclusions beyond those written here.

| id | contrast | pairing | isolates | pre-declared direction / test |
|---|---|---|---|---|
| C1 | R1 − R0 | item+seed, 50 items | endpoint conditioning on base prior (suppression / lerp-shortcut) | per exp_061 pre-registration (suppression expected) |
| C3 | R2 − R3 | class-level: same specialist, same seeds, matched counts (2v2) | content overfit (held-in vs unseen endpoints) | R2 > R3; sign test, 11 classes; at BOTH ckpts → overfit trajectory |
| C4 | R3 − R1 | item+seed, 11 classes × 2 × 3 | specialist adapter value over conditioned base | R3 > R1; sign test, 11 classes |
| C5 | R3 − R4 | item+seed, 8 classes | specialist vs generalist ("interference"), native-vs-native | **PRIMARY: sign test across all 8 classes, direction R3 > R4 — conservative, because endpoint contamination (12/16 items ic2-seen) inflates R4 and biases AGAINST this hypothesis.** Clean subset (4 unseen items) = pre-registered effect-size estimate with seed-level bootstrap, no test (n=4 cannot reach α=0.05 by sign). |
| C6 | R5 − R1 | item+seed, 3 classes | zero-shot generalist value over conditioned base | descriptive: effect sizes + seed-level uncertainty (n=3, no test) |
| C7 | R3 − R5 | item+seed, 3 classes | zero-shot gap vs specialist ceiling | descriptive, same treatment |
| C8 | R4 − R1 | item+seed, 8 classes | generalist value on trained classes | secondary; contamination-stratified |

Robustness rules, pre-committed: (i) C5's verdict is "robust" only if its sign agrees
at specialist ckpt 250 and 2000; disagreement is reported as checkpoint-sensitive,
not resolved post-hoc. (ii) Headline specialist numbers = step_02000; step_00250 is
the pre-registered sensitivity row. (iii) R3−R4 mode asymmetry on one-sided classes
(specialist consumes suffix, generalist doesn't) is a declared property of the
native-vs-native design, not a bug; the S depart-side mask governs scoring in both.
(iv) Generation keying = exp_058 training sidedness for R4 / validated taxonomy for
R5; scoring masks = validated taxonomy everywhere.

## 8. Batch DONE criteria (all must hold)

1. `eval/metric-workbench`: outputs force-add commit pushed; `git status` clean;
   ahead=0 behind=0.
2. `main`: SPLIT_V1_FINAL stamp (with split sha256) + this plan + `ladder_items_v1.json`
   committed & pushed; grid JSON re-derivable from the §4 rules against the recorded
   sha256.
3. 11 specialist training chains submitted on `secondary` with resume+chain (verified
   in queue or running); W&B runs alive; per-class precomputed latents present.
4. 60 R4/R5 videos (48 + 12) on disk, each ffprobe-verified 121 frames nonzero,
   manifests complete per §5 including `endpoint_seen_by_ic2`, reference ids, keying
   source, adapter sha256.
5. Pairing audit report: 22/22 roster test items × 3 seeds × 2 arms present in
   exp_061 outputs.
6. CHANGELOG entries for A1–A7 per repo rules.

Deferred by design, NOT part of done: hero_flight R5 (6 videos, next day), R2/R3
generation (on training completion), all scoring, metric-search certification (2
owner decisions), any clean-generalist retrain.

**Registered contingent follow-up (no compute committed):** if C5's clean subset
proves too noisy to support any statement and the owner wants a clean interference
estimate, the named remedy is a split-v1-train-only generalist retrain (new exp;
circulant pairing minus test clips). It is deliberately NOT in this batch: its
sidedness-keyed training mask depends on labels still under validation
(earth_element, earth_wave, giant_grab, water_bending, water_element are all trained
classes in the conflict set), so it only becomes safe after validation.

---

## Amendment 1 — Side-keyed specialists, R1K base rung, R3X/R4X cross-class rungs (2026-07-16)

**Status of this amendment:** dated addition per the §0 rule ("Changes after generation
begins require a dated amendment section, never an edit"). Authored under `/advised`
(advisor = fable) on owner directive. Realized grid: [`ladder_items_v2.json`](ladder_items_v2.json)
(sha256 `087206d7708b8b9237ebaa62ef1b58158af9eaa060974fe3224e60e75c78d752`), derived
verbatim from frozen `ladder_items_v1.json` (`afe17a3f…`) — **no v1 item is re-drawn or
edited**; v2 only layers the additions below. Split sha256 UNCHANGED.

**Honesty anchor (why this is legitimate, not data-dependent).** At the time this
amendment is committed: the 11 *blind* specialist trainings are complete or in flight
(fallback checkpoints on disk), but **zero R2/R3 generations exist, zero R4/R5 scoring
has occurred, and no metric has been computed on any specialist output.** The amendment
is therefore outcome-blind with respect to every specialist rung. It is committed and
pushed BEFORE any keyed training or any R2/R3/R1K/R3X/R4X generation is submitted (same
pre-registration-timestamp discipline as A3).

**D2 is SUPERSEDED.** D2 chose sidedness-BLIND specialists (prefix+suffix always) to
keep training taxonomy-immune. That is now judged wrong for the PRIMARY contrast. On the
one_sided classes SPEC §3 defines the effect's terminal state AS endpoint B; a blind
specialist is fed the true arrival endpoint via the suffix, while the keyed generalist
(R4) is not — so on 7 of 8 C5 classes the blind design hands R3 the answer and biases
C5 (R3−R4) **toward** its pre-declared direction R3>R4. A declared bias favoring the
headline hypothesis is exactly what this pre-registration exists to prevent. Under the
amendment every conditioned rung applies **identical class-keyed conditioning**
(R1K/R2/R3/R4/R5), so C5 becomes conditioning-matched (adapter-vs-adapter only).

**What CHANGES:**
1. **Specialists are side-keyed** per `specialist_conditioning`: one_sided → prefix
   (tb=2, p=1.0) ONLY; two_sided → prefix+suffix (unchanged from blind). ⇒ the 9
   one_sided specialists RETRAIN prefix-only into new dirs `outputs/training/
   exp_062_ladder_r2r3_specialists/<cls>_keyed/`. The two_sided classes (shadow_smoke,
   hero_flight) keep the blind==keyed run; `<cls>_keyed` symlinks to it. Blind
   one_sided checkpoints are RETAINED as a labeled sensitivity artifact (regenerated
   into results only if keyed training misbehaves or the owner asks).
2. **New rung R1K** = base 19B dev, NO adapter, prefix-only, for the 9 one_sided
   classes' 2 test items × 3 seeds (54 videos). two_sided R1K ≡ R1 (reuse). Contrasts
   **C4, C6, C7, C8 re-baseline onto R1K** (so "adapter value" is not confounded with
   suffix removal); **C1 (R1−R0) stays on the original blind R1** per exp_061.
3. **§7 robustness rule (iii) is superseded** — the one-sided R3−R4 "mode asymmetry"
   it declared acceptable is now removed by keying, not tolerated.
4. **New rungs R3X / R4X (contrast C9), SECONDARY.** For each recipient in eligibility
   block **B8** = {one_sided ∧ scene_swap=false} = animalization, color_rain,
   gas_transformation, illustration_scene, polygon, portal, shadow, wireframe
   (super_fast_run excluded: scene_swap=true ⇒ no matched donors; two_sided excluded):
   feed the recipient specialist (R3X) and the generalist-with-recipient-reference
   (R4X twin) prefix endpoints borrowed from N=4 other B8 classes (donor rule frozen in
   the grid). step_02000 only; seeds 42/43/44; 96 R3X + 96 R4X videos. No ground truth
   (foreign endpoints) — scored as class-effect transfer, never item reconstruction;
   M2c copy runs vs the recipient's training manifest (the signature failure mode of
   weight-baked identity). Scoring contract per grid: `reference_video` = recipient's
   grid reference clip (identical across the twin ⇒ delta apples-to-apples),
   `style`/mask = recipient class. **C9: R3X > R4X, one-sided sign test across the 8
   recipients, α=0.05.**

**What is UNCHANGED:** all R0/R1 rows and the 300 exp_061 videos; the 60 R4/R5 videos
already generated (their keying is identical in v1 and v2 — this amendment does not
touch the generalist); C1; the split and its sha; seeds; the certified-harness scoring
plan (§7) except the C4/C6/C7/C8 re-baseline and the added C9.

**hero_flight contingency.** hero_flight is two_sided (taxonomy_v1) AND in the
sidedness-conflict set → its specialist R2/R3 generation AND its R5 stay DEFERRED
(bundle B1) pending owner sidedness validation. Under two_sided keying its blind
training already equals the keyed recipe (no retrain). IF validation flips it to
one_sided: retrain prefix-only, add its R1K rows, regenerate R2/R3, key its R5
prefix-only (~2 GPU-h). shadow_smoke is exp_058-labeled two_sided and conflict-free →
fully in.

**σ_seed caveat.** amendment-1's σ_seed/MDE was calibrated on suffix-anchored
generations; prefix-only rungs (R1K, one_sided R2/R3/R3X) may show higher seed
variance. This is reported context accompanying those deltas, NOT a gate.

## Amendment 2 — Unified tier grid, split-aligned generalist retrain (ic3), live_concert restoration (2026-07-16)

**Honesty anchor.** At commit time, **zero adapter-rung scores exist** (no R2–R5/R1K/R3X/R4X
item has ever been scored; base R0/R1 scoring may run concurrently — base arms are untouched
by this amendment). This amendment is therefore outcome-blind with respect to every claim it
re-bases. It becomes **binding only after a fresh-context gate PASS**; no ic3 job is submitted
before that. Owner authorized the retrain and the live_concert ruling 2026-07-16 (chat).

**Amends:** D5 (ic2 stays pinned — but as the *comparison* arm; headline generalist becomes
ic3), D4's live_concert skip (its "0 test clips" rationale dissolves; the specialist roster
nonetheless stays 11 — adding a 12th specialist is declined scope, owner may reopen), §3's
tier bookkeeping (contamination stratification → designed tier A). Amendment 1 is unchanged.

### A2.1 The unified tier grid (design restatement)

One generation manifest per trained model; a row's tier is just
**(class trained?) × (endpoint clip band)**:

| tier | class trained | endpoints | meaning |
|---|---|---|---|
| A | yes | train band | held-in ceiling / overfit |
| B | yes | test band | within-class generalization |
| C | no  | test band (ref from its train band) | zero-shot class transfer |
| X | — | another class's test clips (sidedness-matched) | effect transfer vs content memorization |

Universal rules: **no adapter ever trains on a test-band clip**; conditioning keyed by
owner-validated taxonomy sidedness everywhere (base, specialists, generalist); foreign (X)
endpoints sidedness-matched (owner ruling: endpoints carry no transition information, so
matched-sidedness foreign endpoints are well-posed for every class); same items + seeds
42/43/44 across all models → every comparison is a paired delta.

### A2.2 Corpus repair (owner ruling): live_concert_2 removed

Owner visual review 2026-07-16: **live_concert_0 ≡ live_concert_2** (true duplicate);
remaining live_concert clips ruled distinct — this *downgrades* the automated all-mutual-dup
finding (which had forced all-train) to "high similarity, one true duplicate."
Action: quarantine `live_concert_2.mp4` (keep-lowest-index convention) to
`data/processed/transitions_std121/_removed/` (manifest builder skips `_` dirs); corpus
223 → 222 clips; manifest rebuilt (corpus_sha changes; a one-paragraph **certification
amendment-3** on `eval/v3-spec-versioning` records the new sha, same data-plane pattern as
amendment-2).

### A2.3 split v1.1 (localized re-draw, one class)

Rebuild live_concert's band **under the frozen v1 rule, unchanged**: n=7 → **1 test clip**,
drawn by the same seeded stream `random.Random("split_v1:live_concert")` over the 7 surviving
sorted clip ids (remediation stream applies as in v1 if flagged). **Every other class's
assignment is asserted byte-identical to split_v1.** Output: `split_v1.1.json` +
`SPLIT_V1.1.md` (sha256 + this rationale) + tag `split/v1.1`. Housekeeping in the same batch:
force-fix `split/v1` → 262aa10.

### A2.4 ic3 — split-aligned generalist retrain (exp_064)

- **Data:** split-v1.1 **train band only**, of the 32 non-holdout classes.
  **Holdout verbatim ic2** (owner: no extension): hero_flight, illustration_scene,
  gas_transformation, raven_transition, hole_transition, seamless_transition, jump_transition.
- **Keying:** owner-final sidedness for every class (fixes ic2's giant_grab / earth_wave /
  water_bending mis-keys).
- **Recipe pinned = exp_058 verbatim:** rank 32/α32 attn+FFN, lr 2e-4, 5000 steps, ckpt/500,
  seed 42, 480×640×121@24, single mask condition (prefix 2 latent frames always; +final
  latent frame iff two_sided); pairs rebuilt from the train band by the exp_058 builder.
- ic2 is retained, demoted to **labeled comparison arm** — it never headlines again.

### A2.5 Generation grid v3 (exp_065 — ONE manifest, all tiers at once)

| model | tier rows |
|---|---|
| ic3 | **A**: 8 C5-classes × designated train item + the 10 test-less classes' stand-in items · **B**: every trained ∩ test-bearing class (now incl. live_concert) × its test items · **C**: holdout classes with ≥1 train + ≥1 test clip (hero_flight, gas, illus, raven, hole; seamless/jump are single-clip → no rows) × test items · **X**: 11 recipients × 4 donors |
| specialists | **X extension only**: the 3 recipients Amendment 1 excluded (shadow_smoke, hero_flight, super_fast_run) × 4 sidedness-matched donors, ckpt 2000 (+36 videos); frozen B8 donor sets kept verbatim |
| base | keyed-endpoint arm extended to all one_sided classes beyond the R1K 9 (+≈45) + live_concert's new test item rows (+9) |

References = the class's frozen grid reference clip (train band; for tier C, the holdout
class's train band — never trained). Exact item lists + donor assignments frozen by
`build_ladder_items_v3.py` → `ladder_items_v3.json` (sha recorded at gate time); ic3 volume
≈ 270–300 videos, total new ≈ 360.

### A2.6 Contrasts

C1/C3/C4/C9 unchanged. **C5 (PRIMARY) re-based:** specialist UNSEEN·own (R3) vs **ic3 tier
B**, same items/seeds, sign test across the same 8 classes — tier B is clean **by
construction**, so the contamination stratification becomes unnecessary for the headline
(`endpoint_seen_by_ic2` still emitted for the ic2 comparison arm). C6/C7/C8 re-based on ic3;
C6/C7 upgrade from descriptive n=3 to **n=5 tier-C classes** (5/5 unanimity → p≈.031;
labeled fragile). New secondary contrasts: **C10 = ic3 − ic2** on shared rows (the measured
value of split alignment + correct keying); **C11 = ic3 tier A − tier B** (designed
generalist overfit gap). σ_seed/MDE context and prefix-only caveat carry unchanged.

### A2.7 What does NOT change

split_v1 assignments for the other 38 classes; all specialist trainings (blind + keyed) and
their R2/R3/R3X/B1 rows; the 300 R0/R1 videos; certified harness v3.0.0 + amendments;
taxonomy v2 and its strata; sidedness semantics; seeds; the frozen B8 donor grid.

### A2.8 Execution order

1. Fresh-context gate on this amendment → PASS required.
2. Quarantine live_concert_2 → rebuild corpus manifest → cert amendment-3.
3. Build split_v1.1 (+byte-identity assertions) → tag.
4. `build_ladder_items_v3.py` → freeze grid v3.
5. Submit exp_064 (ic3 train; secondary, ≤4h windows, resume-aware).
6. Submit exp_065 grid (skip-if-exists) after ic3 DONE.
7. Score all rungs — certified checkout, corrected manifest. (Base-arm scoring independent;
   may run before the gate.)
