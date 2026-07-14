# Transition Eval Harness — Specification

**Version: `transition-eval/3.0.0`** (see `VERSION`; stamped by `versioning.py`)
**Status: CERTIFIED on the committed record `certifications/v3.0.0.md` + annotated tag `eval/v3.0.0`** — a checkout is certified only when `versioning.stamp()` says so (clean tree ∧ exact tag ∧ non-draft).
v2 (exp_053 conventions) is retired; v2↔v3 numbers are not comparable.

The two rules above all sections: **pin everything, stamp everything.**

---

## 0. OPEN register (lock-severity items must be empty before tagging — **satisfied 2026-07-14; `eval/v3.0.0` exists**)

| id | item | severity | where resolved |
|----|------|----------|----------------|
| O1 | ~~Create corpus_manifest.json~~ **RESOLVED 2026-07-10**: built by `build_corpus_manifest.py` — 39 classes / 223 clips, all std-contract-verified (portrait 480w×640h×121f@24), sidedness+tags from the labeled tree, dedup provenance per class, 2 raw filename quirks recovered via logged fuzzy match, 0 problems | ~~lock~~ done | `data/processed/transitions_std121/corpus_manifest.json` (force-added; data/ is gitignored) |
| O2 | ~~Freeze dep pins + stage checkpoints~~ **RESOLVED 2026-07-10**: DINOv2 rev `f9e44c81…`, CoTracker code pinned by content hash `868059fa…` + ckpt sha256 `2670d456…`; all staged in `$LAB/cache/` since 2026-07-06 | ~~lock~~ done | `versioning.PINS` |
| O3 | ~~Pre-register certification bars~~ **NUMBERS FINAL 2026-07-13** (freeze delegated by the owner; forms from the three-pass design review; per-number rationale in bars.yaml comments). The `frozen: true` flip is its own commit, before the first run | ~~lock~~ done | §6 / `certify/bars.yaml` |
| O4 | ~~Implement `certify/`~~ **IMPLEMENTED 2026-07-13**: `exam.py` (R1+R2, sign-test adoption, trust map, O7 conditional), `probes.py` (sibling selection + audit, splices + perturbation, swap, hard-cut, reversal enumeration/grading), `blockc.py` (archive conversion, copy-twin bar, bridge, distributions), `run_certification.py` (§6.5 driver), `stability.py`, `seeds.py`. Synthetic contract tests in `tests/test_certify_v3.py`. First execution = the certification run | ~~lock~~ done | `certify/` |
| O5 | ~~v3 metrics + lifecycle~~ **RESOLVED 2026-07-13/14**: full stack (`s_structure`, `m1_transfer`, `m2_integrity`, `controls`, `manifests_v3`, `plan`, `score.py`) executed end-to-end on real data by the draft.8 certification run (job 9465002 — every planned A–D item scored, zero error rows); `score.py` is the one scorer for all v3 numbers, experiment forks are retired from use (legacy scripts under `experiments/` are historical artifacts, never mutated per repo rules) | ~~lock~~ done | §3/§9 |
| O6 | σ_seed measurement (12 stratified items × 5 seeds, adapter arm) — **gates the first model report, not the tag** (§6.4) | cert | §4/§6.4 |
| O7 | ~~M1b robust weighting choice~~ **RESOLVED 2026-07-13 by pre-registration**: median-trim ships as primary; Huber variant examined under the same adoption rule *iff* camera-stratum recall < trust floor (executes mechanically at certification) | ~~cert~~ done | §6.1 / `bars.yaml` |
| O8 | Δ-novelty / difference-from-lerp candidate roster — **DEFERRED to post-lock (v3.1)**: certification scope = shipping headline metrics; candidates enter via a future exam cycle under the same adoption machinery | post-lock | §3 M1d / §6.1 |
| O9 | Judge human-calibration set (~50–100 labels, Spearman ≥0.8) + q1 name-the-mechanism sharpening decision | post-lock | §3 M4 |
| O10 | Rescore exp_056–058 archived items under certified v3 (continuity bridge) | post-lock | §7 |
| O11 | Update repo guidelines (CLAUDE.md / skills) to the v3 workflow | post-lock | §10 |
| O12 | Repo hygiene: commit the exp_044–059 backlog in the main checkout; single-writer discipline for CHANGELOG.md | advisory | out of spec |

Severity: **lock** = blocks the `eval/v3.0` tag · cert = resolved during first certification run · post-lock = scheduled after tagging.

---

## 1. Purpose & scope

Judges **reference-based creative transition transfer under endpoint conditioning**: did the generated video execute its single reference demo's transition — appearance (M1a), camera+object motion (M1b/c), timing (M1d) — on its **own** endpoints (M3), without copying reference content, importing a wrong style, or regurgitating training data (M2).

Does **not** judge: absolute perceptual quality, text/prompt alignment, human preference (M4 advisory until calibrated), styles absent from the corpus.

Decision fed: **"is arm/checkpoint A better than B on identical inputs; does capability X exist vs the base twin"** — paired, never absolute.

## 2. Inputs — the contract

**Videos** (generated, reference, conditions): **width 480 × height 640 (portrait)**, 121 frames, 24 fps, H.264 mp4 (`std121`; probed uniform across all 223 corpus clips 2026-07-10). Condition clips: prefix `num_frames % 8 == 1`, suffix `% 8 == 0`.
**Malformed input → reject loudly, never adapt**: wrong resolution/fps/length, `T < n_prefix + n_suffix + 4`, unknown manifest keys. Resizing is a metric decision, never input handling (the 320p-upscale confound is documented precedent).

**Three documents; no fact stored twice:**

| document | owns | lives with | changes |
|---|---|---|---|
| `corpus_manifest.json` (O1) | per-clip truth: class, sidedness, tags, provenance, native resolution, dedup status | the data (`transitions_std121/`) | rarely; every change → dedup gate + re-certification |
| `training_manifest.json` | per-adapter truth: clip keys, pairs, per-pair conditioning mode | the adapter checkpoint | once per training run |
| eval manifest | per-generation truth (below) | the eval suite | per suite |

Eval manifest row:

```
item_id, generated_video, reference_video, style,
n_endpoints (1|2), condition_prefix{video,num_frames}|null,
condition_suffix{...}|null, arm, twin_of|null, notes
```

**Derived at scoring, never stored:** sidedness & tags (corpus lookup via reference), tier A/B/C (join vs training manifest), trust flags (from certification). `n_endpoints` is a run fact (was a suffix pinned); sidedness is a class fact — distinct on purpose (the exp_058 conditioning-mode-mismatch lesson is only expressible because they differ).

## 3. Metrics

Substrate (pins §7): DINOv2-base CLS per frame, L2-normed, short_side 256 → `f_t`; CoTracker3 tracklets (dual grid at frame 0 + middle, backward tracking, per-step visibility); temporal LPIPS(alex) `d(t)`. All cached content-keyed.

**ID taxonomy = task anatomy:** *execute the reference's transition (M1) on your own endpoints (M3) without cheating (M2), and look right overall (M4).* Legacy v2 IDs (old M1–M6) are retired; IDs are interpreted per the stamped version.

**S — Structure (infrastructure, not scores).**
`eA = norm(mean f[:n_pre])`, `eB = norm(mean f[-n_suf:])`, `cross = eA·eB`;
`â = clip((f·eA − cross)/(1−cross), −0.25, 1.25)` (guard band; inert — curves are z-normed downstream), ditto `b̂`.
**Core mask:** two-sided → `max(â,b̂) < 0.5`; one-sided → `â < 0.5` (the effect's terminal state *is* endpoint B there); conditioned windows always excluded. Fallback: if `|core| < k` (k=8, frozen) → frames within `env_min + δ` (δ=0.05, frozen) and `core_degenerate=true` — fallback output is always flagged, so k/δ are sizing constants, not gates (§6.4). Emits core size, `cross`, `cross_high (>0.85)` → trust flags. Scalars: depth, depart, arrive (two-sided only), core_frac.
*Blind to:* effects below CLS sensitivity; semantically-close endpoints degrade it (flagged, not hidden).

**M1a — Appearance transfer.** Symmetric mean-of-max cosine between gen-core and **reference**-core features. [−1,1], ↑. *Blind to:* copy-vs-transfer (M2a disambiguates); camera classes (taxonomy-flagged advisory); inherits core-mask health.

**M1b — Camera motion match.** Per-step robust visibility-weighted similarity fit `(dx, dy, log s, θ)` on tracklets (median-trim primary; Huber conditional per §6.1, O7) → 4-channel trajectory, resample 64, z-norm → banded DTW (band 0.15) + correlation vs reference. ↑corr/↓DTW. *Blind to:* parallax/3D (SfM rejected: scenes are non-rigid/dissolving); flagged when trackable points < N.

**M1c — Object motion match.** MFS (bidirectional mean-of-max velocity-direction correlation; 64 steps, speed floor 0.1, min_vis 0.2, min_moving_frac 0.05) on **residual** velocities after removing the M1b global fit. ~[−1,1], ↑; NaN reported, never imputed. *Blind to:* magnitude, spatial arrangement; requires per-class exam trust.

**M1d — Timing (analysis tier).** depart / arrive / depth from S. Candidate Δ-novelty (residual off span{eA,eB}) enters only by displacing depth + profile-DTW (O8, deferred post-lock — candidates enter via a future exam cycle).

**M2a — Copy (reference → generation).** `max cos(gen_mid, ref_noncore)`; gen_mid = all frames outside conditioned windows (near-endpoint bleed is in scope — the exp_056 briefcase lesson). `near_copy` flag at τ_copy (0.88 draft, recalibrated O3). ↓. Argmax provenance (gen frame, ref frame) recorded. *Gray zone:* in-style layout tracking (0.85–0.9) — the base twin arbitrates.

**M2b — Intrusion (wrong style → generation).** `app_k` = set-sim(gen core, class-k core pool) for **every** corpus class; `margin = app_style − max_{k≠style} app_k`, argmax class named. ↑/positive. Detects cross-class texture leakage (smoke into gas). *Blind to:* textures absent from the corpus (M4's job); camera classes.

**M2c — Memorization (audit, off-headline).** `max cos(gen_mid, training-clip frames)` + clip/tier attribution; read **only** via tier B↔C contrasts; skipped without a training manifest.

**M3a — Endpoint fidelity.** Per conditioned side: aligned-frame DINO cos ↑ + LPIPS ↓ vs condition clip (cover-crop, ValidationRunner-mirrored). Suffix N/A when `n_endpoints=1`.

**M3b — Seam flag.** Robust-z of `d(t)` at the two handoff indices + raw ΔLPIPS; headline `max_seam_z`. **A flag, not a ranker** (conflates failure snaps with handoff artifacts; MAD-relative; z ≳ 3 saturating). Valid only within matched conditioning modes.

**M4 — Judge (ADVISORY; certification-exempt).** Gemini rubric ×5 questions, native video 8 fps, temp 0, response-schema-pinned, per-item response cache, model_version recorded per response. Headline entry gated on human calibration (Spearman ≥ 0.8, ~50–100 labels; O9).

**Deleted from v2:** floor/ceiling normalization, `excess`/`mean_max_other`, undifferentiated `max_sim_target`, `hold` scalar, Pearson, silent 1-frame core fallback, per-experiment scorer forks.

## 4. Lifecycle, aggregation & reporting

**Lifecycle — the harness never runs the model:**
1. **`plan`** (harness, CPU): corpus manifest + training manifest + suite design → `suite.json` (endpoints, reference, caption, conditioning per item; **base twins and controls auto-included**; lerp/hold controls are synthesized at scoring, no GPU generation) + eval-manifest skeleton.
2. **`infer`** (model side, external, any infra): fulfill `suite.json` to the §2 contract.
3. **`score`** (harness): verify completeness vs `suite.json` — missing items listed, run stamped `partial`, never silently subset — then compute, aggregate, stamp.

**Inference & reporting:** paired deltas on identical inputs (`twin_of`, item-matched arms) are the inferential unit; sign tests at small n; **all scores raw** — nothing is divided by corpus statistics. Controls are arms in the same table: lerp (two-sided), static-hold (one-sided & prefix-only: prefix + endpoint-A held + suffix as given), base twin (**mandatory for capability claims**). Context row: real-sibling distribution (same-class real clips vs this item's reference; n≥2 classes only). Grouping: arm × tier × sidedness × tag; mean±std with n in every cell; no composite score, ever.

**Seeds:** routine suites = 1 seed (42), fully paired. σ_seed certified once per model family (12 stratified items × 5 seeds on the adapter arm, O6; **gates the first model report, not the instrument tag** — §6.4); every reported delta carries the σ_seed-derived minimum detectable effect; within-noise decisions escalate to 3 seeds, targeted.

**Headline table:** `arm | n | M1a | M2b (margin + intruder) | M2a | M1b | M1c | M3a | M3b` + flags (`†` exam-untrusted, `core_degenerate`, `near_copy`, `cross_high`, `camera-advisory`, `partial`, `UNCERTIFIED`). Analysis table: M1d, M2c, M4.

## 5. Splits & data

`corpus_manifest.json` (O1) is the single source of truth. Corpus today: `data/processed/transitions_std121/` — 223 clips / 39 classes; 320p-source classes excluded (upscale confound). Exam split = whole corpus (self-supervised, zero generations). Tiers (A held-out / B trained-eval-clips-excluded / C seen) are derived per adapter from the training manifest.

**Rules:** corpus additions require the order-invariant dedup gate (DINO set-sim ≥ 0.90 + dHash-bag confirm ≤ 10) and an exam re-run before first use; any corpus/split change → §6 re-certification; the corpus hash is stamped on every result.

## 6. Health assessment & certification (`certify/`; bars in `certify/bars.yaml`)

The health system tests the **ruler, never the model**: generated videos enter it only as realistic inputs. **Certification = one stamped execution of the health system against bars frozen beforehand** (own commit, before any outcome is computed), producing the committed record that authorizes the version tag. It re-runs on every §7-relevant change; model evals never re-run it — they inherit its trust map and MDE.

Three rules bind every bar:

- **Hard bars only on constructed or human-verified truth.** Statements about model behavior (flag rates, margins on archived generations) are descriptive, never gating — a generation that fails to depart endpoint A *should* flag `core_degenerate`; gating on flag rates would punish the flag for working.
- **Fail-forward.** Failed certification → diagnose → new draft version → re-freeze → full re-run. Bars and thresholds are never adjusted after seeing an outcome.
- **Two-kind calibration.** *Corpus-only* calibration (core-fallback δ/k from envelope statistics, reversal-sensitivity enumeration, sibling-pair selection) may run **pre-freeze** — it reads real clips only, never a graded outcome. *Outcome-coupled* calibration (τ_copy) runs post-freeze under a frozen setting rule: the bar is on the gap, the rule sets the value.

### 6.1 Block A — EXAM (validity + variant selection; corpus only, cached features)

Two readouts, both computed with **imported deployed metric code** (never reimplemented; pytest guard: exam statistic ≡ `score.py` statistic on the same item):

- **R1 clip-level:** LOO 1-NN retrieval + Cohen's d for M1a/M1b/M1c — the metrics deployed as clip-to-clip comparisons.
- **R2 pool-level:** LOO class-pool margin classification for M2b — deployed as pool comparisons, clip excluded from its own pool. R1 trust does **not** transfer to M2b (different estimator); R2 exists because of that.

**Trust map** (consumed by every model report): per-class recall on n≥4 classes only — n<4 auto-untrusted, the 2 singletons excluded from denominators and permanently untrusted; M1c trust additionally requires per-class definedness rate ≥ bar (NaN prevalence is a trust fact).

**Variants (only these):** core mask {v2_envelope, v3_sided, all_frames}; motion {v2 MFS incumbent, v3 decomposed}. **Adoption (pre-registered, `exam.py` applies it mechanically):** one-sided exact binomial sign test (α=0.05) on per-class recall over the non-tied n≥4 classes of the target stratum ∧ overall accuracy within tolerance of the incumbent ∧ no trusted class regresses below the trust floor ∧ (core mask only) non-degenerate STRICT cores on ≥ bar fraction of real one-sided clips. **Motion contingency (pre-registered):** decomposition additionally requires the M1c median recall on definedness-eligible classes to trail the incumbent by ≤ bar; if not adopted, the incumbent MFS ships as the single headline motion metric and M1b/M1c demote to analysis tier. **O7 pre-registered conditional:** median-trim ships as primary; the Huber variant is examined under the same rule *iff* camera-stratum recall < trust floor (new draft version).

**Bar 1 (3.0.0):** winning M1a variant clears Cohen's d ≥ 1.5 — the d conjunct only. The accuracy conjunct (≥ 0.80) was **deleted by owner decision at the draft.8 inspection with the outcome known**: the observed accuracy was 0.673, against a floor calibrated on the 47-clip / 11-style v2 corpus (chance 0.213) and never re-derived for the 223-clip / 39-class exam (chance 0.067). The deletion is outcome-aware and does not count as pre-registration for the draft.8 data; the surviving d ≥ 1.5 conjunct was pre-registered in draft.7/draft.8 and passed there (d = 1.522) before the change was made. Accuracy remains a reported, non-gating exam statistic.

### 6.2 Block B — PROBES (constructed truth; zero GPU inference)

All probes are scored by the same `score.py` as real items — a probe path through special code would certify the special code, not the instrument.

- **Siblings + controls (Bar 2, merged — 3.0.0).** All within-class pairs of n≥2 classes feed the **content-invariance audit** via cached-feature statistics (deployed set-similarity + endpoint vectors — no full scoring; ~500 pairs would waste hours of LPIPS for numbers the audit doesn't need). The hard bar attaches only to the **max-endpoint-distance pair per class** — deterministic from cached features; same style, maximally different content, the constructional known-positive — which runs through full `score.py` together with its auto-synthesized control arm (lerp two-sided / static-hold one-sided). (Lexicographic pairing rejected: filename adjacency correlates with shared source scenes, inflating M1a and legitimately firing M2a.) **Bar 2:** per **n≥4-eligible** class, sibling M1a > its control's M1a ∧ M2a silent on the sibling; **all eligible classes must pass**. Rationale for the merge (owner decision, 2026-07-14, draft.8 inspection): draft.8's bars 2 and 3 gated the *same inequality* from opposite sides, and their count floors (35/37, 37/39) were arbitrary headroom; eligibility reuses the exam's existing n≥4 trust convention — a 2-clip class yields exactly one pair and no distributional basis for a hard claim; ineligible classes stay scored and reported, never graded. **Disclosed plainly:** this rule is outcome-aware — nature_bloom (n=2), draft.8's only floor inversion and only bar-2 miss, leaves the denominator under it; the residual risk it represented is documented in the record's nature_bloom / content-confound note. `core_degenerate` is removed from the certification bar path **entirely** — no conjunct, no silent per-class logging (it was smoke-only; dead code in a certified instrument is worse than no code). The flag itself remains a live output of S — per-row flag, mask-adoption criterion, Block C descriptive rates — none of which are certification gates. There is no bar 3 in 3.0.0; bars 4–8 keep their historical ids.
- **Copy splices.** Reference **non-core** frames spliced into a centered gen-mid segment (bar-length frozen in bars; core-frame splices sit outside M2a's comparison pool and would fail spuriously — construction pinned); verbatim + exactly **one** pinned deterministic perturbation (center-crop + fixed per-channel color gains, no RNG — real model copying is re-rendered, never bit-exact). Honest set = the **bar-pair** sibling M2a values (independent real footage provably does not copy its reference; max-distance pairs avoid shared-scene contamination). **Bar 4:** 100% of splices (incl. perturbed) ≥ τ_copy ∧ gap(splice min − honest max) ≥ bar minimum in cosine units. **τ_copy := gap midpoint** (frozen rule); τ is *set* here and *tested* in Block C — a C miss means new draft, never a τ nudge.
- **Reversal.** Reversal-sensitive camera bar pairs only: sensitivity is enumerated **analytically** from the deployed trajectories (per-step params time-reversed and negated; self-reversal DTW ≥ bar — palindromic camera moves score ≈0 and stay out of the denominator; corpus-only calibration). The probe itself re-tracks **real reversed videos** through the identical pipeline (`process_video_file` + `camera_match`); the reversed reference is intentionally non-corpus, so this is the one probe graded outside `score.py`'s manifest wrapper — the metric code path is byte-identical, only the I/O shim differs. **Bar 5:** unreversed match beats reversed match per pair — sign test (α=0.05) if the enumerated n ≥ 8, else every pair must drop; the rule is frozen now, parameterized by n.
- **M3 panel. Bar 6:** endpoint-swap — true prefix beats a wrong-**class** prefix (cross-class swaps only: unambiguous truth; same-class swaps can tie via shared scenes), count-form — ∧ hard-cut — a constructed abrupt cut at the handoff index fires `max_seam_z > 3`, count-form.

### 6.3 Block C — REALISM (the ~150 archived exp_056–058 generations; zero GPU inference)

**Bar 7:** the 11 human-verified copy-regime base twins (exp_057) flagged 11/11 (`near_copy` ∨ `max_seam_z > 3`) — τ_copy's re-rendered test. Everything else is descriptive: per-arm distributions (saturation/degeneracy check), flag rates, margins, and the per-item v2↔v3 bridge (O10 head start). No bars on model behavior.

### 6.4 Block D — STABILITY + CALIBRATION

**Bar 8:** every planned A–D item scores without crash — scoring isolates per item (draft.8), so a bad item yields a loud ERROR ROW instead of a dead stage, and error rows count against this clause exactly as crashes do — ∧ warm-cache rerun within 1e-6 per headline metric (LPIPS is recomputed every run — byte identity across GPU kernels is not a property this instrument has; v2 twin-job precedent: agreement at 1e-7, across separate jobs; same-node back-to-back reruns may agree bitwise) ∧ the 6 anchor items reproduce raw scores within the single reproduction tolerance against a fresh cache directory. Anchors are a frozen **rule**, not a hand-picked list (deterministic picks: strata filled in order — two-sided, one-sided, camera — each taking the first lexicographic n≥4 class *not already picked* (draft.8 dedup; draft.7's wording picked air_bending for two strata and its own assertion refused), the two-sided control, first exp_057 base + ic items). **Calibration outputs** (constants with provenance, not pass/fail): τ_copy · core-fallback δ/k (frozen 0.05/8 — fallback output is always flagged, so these are low-stakes sizing constants, not thresholds that gate anything) · per-sidedness control floors · the reversal-sensitive set · **σ_seed** (12 stratified items × 5 seeds on the decision-generating adapter arm; base assumed same-family, documented; ≈48 df, stated as *pooled*; **gates the first model report, not the tag** — the tag certifies the ruler, σ_seed calibrates the measured thing's noise).

### 6.5 Sequence, record, gate

Freeze bars (own commit) → A adjudicates variants → B and C run under the winners → D → `certifications/v<X.Y>.md`. Mechanical throughout — no human decision between freeze and record.

The record must contain: bars.yaml sha · per-bar verdicts · trust map (recall + definedness, per class per metric) · **content-invariance audit** — within-class partial correlation of style-similarity (core frames) vs endpoint/content-similarity across all sibling pairs; expectation ≈ 0 (DINO is content-heavy and M1a claims style); alarm level noted, non-gating · archive distributions + bridge · calibration constants with provenance · anchor values · corpus hash + artifact pointers · the claims paragraph below, verbatim.

**What certification claims, exactly:** *the instrument separates the styles it has seen (per-class trust map attached), refuses known-degenerate and copied inputs, is direction-sensitive on motion, runs deterministically end-to-end on real generations, and its blind spots are enumerated.* It does **not** claim: that metrics track human judgment (M4 exempt until O9); that pools/masks behave identically on generated-domain frames (untestable without labels — named limitation); M2c validity (validated at the first real training manifest); M1b absolute validity (injected-trajectory test = post-lock appendix).

**Gate:** any §7-relevant change → full re-run before any model number is reported. M4 re-enters via O9 through this section.

## 7. Versioning & invariants

`versioning.py` is the enforcement point. Every `results.json` and every `items.jsonl` row embeds `versioning.stamp()`: harness version, git commit + instrument-scoped dirty state, SPEC hash, corpus hash, declared pins + observed env, certified flag with reasons.

- **Certified** ⟺ clean instrument tree ∧ HEAD tagged `eval/v<version>` ∧ non-draft version ∧ no OPEN pins. Everything else renders `UNCERTIFIED` in every table.
- Pins (`versioning.PINS`): DINOv2 revision (O2), CoTracker3 commit (O2), lpips net, judge model string, short_side, all thresholds. Checkpoints staged under `$LAB/cache/huggingface/` so old tags stay runnable offline.
- Any change to metrics, checkpoints, thresholds, preprocessing, or splits bumps `VERSION`; cross-version comparison is invalid — the bridge is rescoring old items under the new tag (features cached; cheap). Cache tags bump with protocol changes.
- **Invariants:** raw-only scores · paired-first inference · provenance on every retrieval-type number · reject-don't-resize · no composite · flags never dropped · no fact stored twice · **one instrument, one location** (experiments never contain metric code) · uncertified/dirty runs stamped as such.

## 8. Interface

- `python -m diffusion.transition_eval.plan --design D --corpus C --out suite.json`
- `python -m diffusion.transition_eval.score --manifest M --corpus C [--training T] [--suite suite.json] --label L → {results.json, items.jsonl}` (results embed the stamp, completeness, paired twin table); `--from-items`-style re-reporting = login node, numpy-only.
- `python src/diffusion/transition_eval/versioning.py [--corpus PATH] [--check]` — print/verify the stamp (standalone stdlib; the `-m` form also works inside the diffusion env).
- `python -m diffusion.transition_eval.certify.run_certification --corpus C --main-root R` — the §6.5 certification driver (refuses unfrozen bars; GPU node, ~35–40 min warm; writes record + figures + explorer under `outputs/eval/certification/<version>/`). One-time regrade tooling: `scripts/regrade_draft8_to_v3.py` (3.0.0 record provenance).
- Judge: separate CLI, login node, resumable via response cache.
- Runtime: 1× H100/L40S; ~50 items ≈ 10–15 min warm cache (measured 8:39/46); cold corpus featurization dominates first runs.
- `results.json` schema: `{provenance (the stamp), completeness, items[], tables, flags}`.

## 9. Implementation map & code versioning

**One instrument, one location:** all metric code in `src/diffusion/transition_eval/`; experiments contain only manifests, suite designs, and sbatch wrappers calling the pinned CLI. The v2-era `run_score*.py` forks (3 divergent variants across 4 experiment dirs, md5-verified 2026-07-10) are retired; history stays in git.

```
src/diffusion/transition_eval/
├── SPEC.md  VERSION  versioning.py            # this spec + enforcement stamp
├── video_io.py features.py morph.py motion.py # substrate (v2, unchanged)
│   endpoints.py appearance.py pipeline.py     #   endpoints = M3; appearance feeds M1a/M2b
├── s_structure.py                             # S: sidedness-aware core mask + flags
├── m1_transfer.py                             # M1a appearance · M1b camera · M1c object
├── m2_integrity.py                            # M2a copy · M2b intrusion · M2c memorization
├── rubric.py judge_gemini.py                  # M4 (advisory; Gemma backend retired)
├── controls.py                                # lerp + static-hold degenerate controls
├── manifests_v3.py                            # 3 schemas + tier/sidedness derivations
├── plan.py  score.py                          # lifecycle CLIs (score = the ONE scorer)
├── build_corpus_manifest.py                   # corpus manifest builder (O1 tooling)
├── certify/                                   # §6 as code:
│   ├── bars.yaml                              #   pre-registered bars (FROZEN)
│   ├── run_certification.py                   #   §6.5 driver (A→B→C→D → record)
│   ├── exam.py  probes.py  blockc.py          #   Block A · Block B · Block C
│   ├── stability.py  seeds.py                 #   Block D compare + σ_seed protocol
│   ├── diagnostics.py                         #   exam analysis persistence (non-gating)
│   └── figures.py  explorer.py                #   auto figures + results_explorer.html
└── certifications/                            # committed records, one per attempt
```

Regeneration: annotated tag `eval/vX.Y` per certified release; `git worktree add ../eval-vX.Y <tag>` reproduces any old harness — no snapshot folders, no vendored copies. Rejected anti-patterns: per-version package copies, metric code in experiment dirs, floating `torch.hub`/HF `main` references.

## 10. Harness-change protocol (the operational loop — to be encoded in repo guidelines, O11)

1. Branch `eval/<slug>` (worktree if other work is live in the checkout).
2. Edit **code and SPEC.md together** — a metric change without its §3 block updated is an incomplete change. Update the §0 OPEN register.
3. Bump `VERSION` (iterate as `X.Y.Z-draft.N`; certified releases drop the suffix).
4. `PYTHONPATH=$PWD/src pytest -q tests/test_transition_eval*.py tests/test_versioning.py` — green. (PYTHONPATH matters in worktrees: the env's editable install points at the main checkout and silently shadows worktree code; every harness test file also carries a sys.path shim for the same reason.)
5. `python -m diffusion.transition_eval.versioning --corpus <corpus_manifest>` — review the stamp.
6. Run certification (§6) → committed record `certifications/v<X.Y>.md` + artifacts under `outputs/eval/certification/<version>/`. *(While the version is a draft it cannot produce headline numbers.)*
7. `CHANGELOG.md` entry (repo rule).
8. Commit package + SPEC + VERSION + certification record together. On PASS: drop `-draft`, annotated tag `eval/vX.Y`, push branch + tag, open PR. On FAIL: stays draft, no tag.
9. Cross-version comparisons only via rescoring old items under the new tag.

---

## Spec changelog

- **3.0.0** (2026-07-14): certification revision from the draft.8 joint inspection — owner's closed list, nothing else rides along. (1) **Bar 1 → d ≥ 1.5 only**; the accuracy conjunct is deleted, with the outcome known (0.673 vs a floor calibrated on the 11-style v2 corpus) — outcome-aware by construction, disclosed verbatim in §6.1 and the record; the surviving d conjunct was pre-registered and passed (1.522) before the change. (2) **Bars 2+3 merge into bar 2**: per n≥4-eligible class, sibling > control ∧ M2a silent on the sibling, all eligible must pass — the two bars gated the same inequality from opposite sides and their count floors were arbitrary headroom; eligibility reuses the exam's n≥4 trust convention, and the record discloses plainly that nature_bloom (n=2, draft.8's only miss on both bars) leaves the denominator under this rule. `core_degenerate` exits the certification bar path entirely (no conjunct, no silent logging — smoke-only; dead code in a certified instrument is worse than no code); the flag stays live in S, mask adoption, and Block C descriptive rates. There is no bar 3; bars 4–8 keep historical ids. **Verdicts produced by REGRADE of the draft.8 run artifacts (job 9465002), not a re-run — owner directive** ("don't redo already-done calculations"): the two changed bars are pure grading-rule changes over data score.py already produced under frozen pins; graders for bars 4–8 are byte-identical, their verdicts carry over; the regrade script is committed (`scripts/regrade_draft8_to_v3.py`). On PASS: version drops the draft suffix, record `certifications/v3.0.0.md`, annotated tag `eval/v3.0.0`. O5 marked resolved (draft.8 executed the full stack on real data).
- **3.0.0-draft.8** (2026-07-13): MINIMAL reliability revision after the draft.7 FAILED certification (record `certifications/v3.0.0-draft.7.md`; owner directive: get a complete-data run first, reason about bar forms after inspecting it). Instrument fixes: `object_match`'s empty tracklet keep-filter now returns NaN instead of crashing `_smooth_tracks` (the one bug that killed bars 4/6/7's data), zero-tracklet guard in `camera_trajectory`; `score.py` isolates per item — a failing item emits a loud ERROR ROW (no metric keys → graders count a documented miss) instead of killing the stage, and error rows gate bar 8's no-crash clause exactly as crashes do; the certification driver funnels every stage/grader failure into the record so it always writes (draft.7's record needed post-hoc assembly). Two disclosed bar edits, thresholds untouched: bar 3 drops its `core_degenerate` conjunct (owner decision — vacuous: true-by-construction on holds, legitimately false on lerps since DINO blends sit far from both endpoints; flag stays descriptive + in mask adoption), and the anchor rule gains the dedup that makes it executable. Bars 1 and 3 are EXPECTED to fail again on known draft.7 arithmetic (0.673 < 0.80; 36/37 < 37) — deliberate: this run exists to produce bars 4/6/7/8 data (splices, M3 panel, copy twins, cold anchors) that has never been graded. Deferred to the post-inspection session: bar-1 form, motion headline-eligibility, content-invariance null reframe, Huber O7 examination.
- **3.0.0-draft.7** (2026-07-13): certify/ fully implemented to the locked §6 — `exam.py` two readouts + exact-binomial sign-test adoption (replaces the count-margin draft; α=0.05) + motion contingency (decomposition not adopted → incumbent MFS ships as headline motion) + O7 conditional; `probes.py` sibling selection (max-endpoint-distance bar pairs) + all-pairs content-invariance audit via cached-feature statistics (full scoring of ~500 pairs rejected as pure LPIPS waste) + splices (24f segment, deterministic crop+color-gain perturbation, honest set = bar-pair siblings) + endpoint-swap + hard-cut + reversal (analytic self-reversal enumeration ≥0.5; probe re-tracks real reversed videos; rule frozen parameterized by n: sign test if n≥8 else all-must-drop); `blockc.py` (archive conversion with loud exclusions, 11-twin bar, v2↔v3 bridge, distributions); `run_certification.py` = §6.5 driver. Bar 8 warm rerun: byte-identity → ≤1e-6 (LPIPS recomputed per run; GPU kernels don't owe byte equality; v2 precedent 1e-7). Anchors = frozen deterministic rule, not hand-picked ids. Core-fallback k=8/δ=0.05 frozen as flagged-only sizing constants. `score.py` gains `--controls off` + `--cache-dir` (cold-anchor rerun) + CPU fallback; `versioning.py` dirty-scope now includes the v3/certify test files and uses path-only git queries (own implementation; the prior session's parsing fix superseded and removed). bars.yaml: NUMBERS FINAL with per-number rationale (freeze delegated by the owner); `frozen` flips in the next commit. Contract tests: `tests/test_certify_v3.py`.
- **3.0.0-draft.6** (2026-07-13): §6 rewritten as the full health-assessment spec after a three-pass design review (proposal → red-team → external review). Final shape: two-readout exam (R1 clip-level LOO 1-NN for M1a/M1b/M1c, R2 pool-level margin classification for M2b — R1 trust does not transfer to M2b, different estimator; exam must import deployed metric code), 8 hard bars on constructed/human-verified truth only, sibling probes hard-barred on max-endpoint-distance pairs (all pairs descriptive; lexicographic pairing rejected as scene-adjacent), splice perturbation level + minimum-gap requirement (τ_copy midpoint rule; set in B, tested in C, never nudged), reversal-sensitivity enumeration pre-freeze, M3 panel reinstated (endpoint-swap + hard-cut), content-invariance audit as required record artifact, two-kind calibration rule, σ_seed gates first model report not the tag. Killed with reasons: cross-label probe (≡ exam pool readout on real clips; circular on generations), self-memorization probe (unit test), enforcement probes (pytest), Block-C degeneracy bar (gated on model behavior), O8 roster (deferred v3.1). O7 resolved by pre-registered conditional. `bars.yaml`: forms locked, numbers DRAFT until the freeze session. `probes.py`: cross-label builder/grader removed; docstrings aligned.
- **3.0.0-draft.5** (2026-07-10): stale v2 surface pruned per §3 "Deleted from v2" — `judge.py` (Gemma backend, superseded by Gemini), `manifest.py` (v2 schema, superseded by manifests_v3), `appearance.leakage`/`effect_similarity` (superseded by m2_integrity), `report.normalize_score`/`score_tables` (normalization removed); report.py reduced to exam machinery + trust flags. Package is now fully two-sided AND one-sided capable (sidedness-aware core, sidedness-appropriate controls, prefix-only support, no normalization dependency) — ready for the health-design session.
- **3.0.0-draft.4** (2026-07-10): O1 resolved (corpus_manifest.json, 39/223, contract portrait-corrected to 480w×640h); O3/O4/O5 advanced to implemented-draft — `s_structure/m1_transfer/m2_integrity/manifests_v3/plan/score` + `certify/{bars.yaml DRAFT, exam, probes, stability, seeds}` + static-hold control + v3 synthetic test suite; §9 map updated to actual filenames.
- **3.0.0-draft.3** (2026-07-10): consolidated in-repo; added §0 OPEN register, §9 implementation map + code versioning, §10 change protocol; `versioning.py` + `VERSION` land; package brought under VCS.
- 3.0.0-draft.2: three-manifest model (sidedness/tier derived, not stored); plan→infer→score lifecycle; σ_seed-once seeds policy; metric IDs → task anatomy (S/M1/M2/M3/M4).
- 3.0.0-draft.1: first full spec from the v3 design sessions (leakage 3-axis split, normalization demoted to controls, M2 camera/object decomposition, sidedness-aware core mask, Δ-novelty roster).
