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
