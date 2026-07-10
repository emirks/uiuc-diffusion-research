# Transition Eval Harness — Specification

**Version: `transition-eval/3.0.0-draft.4`** (see `VERSION`; stamped by `versioning.py`)
**Status: DRAFT — NOT CERTIFIED.** No number produced under a draft counts as a result.
v2 (exp_053 conventions) is retired; v2↔v3 numbers are not comparable.

The two rules above all sections: **pin everything, stamp everything.**

---

## 0. OPEN register (must be empty of `lock`-severity items before `eval/v3.0` is tagged)

| id | item | severity | where resolved |
|----|------|----------|----------------|
| O1 | ~~Create corpus_manifest.json~~ **RESOLVED 2026-07-10**: built by `build_corpus_manifest.py` — 39 classes / 223 clips, all std-contract-verified (portrait 480w×640h×121f@24), sidedness+tags from the labeled tree, dedup provenance per class, 2 raw filename quirks recovered via logged fuzzy match, 0 problems | ~~lock~~ done | `data/processed/transitions_std121/corpus_manifest.json` (force-added; data/ is gitignored) |
| O2 | ~~Freeze dep pins + stage checkpoints~~ **RESOLVED 2026-07-10**: DINOv2 rev `f9e44c81…`, CoTracker code pinned by content hash `868059fa…` + ckpt sha256 `2670d456…`; all staged in `$LAB/cache/` since 2026-07-06 | ~~lock~~ done | `versioning.PINS` |
| O3 | Pre-register certification bars — **ADVANCED**: full bar structure drafted in `certify/bars.yaml` (`status: DRAFT`, `frozen: false`); the health-design session replaces placeholder numbers and flips `frozen` in its own commit. `certify/exam.py` refuses to run against unfrozen bars | **lock** | `certify/bars.yaml` |
| O4 | Implement `certify/` — **ADVANCED (skeleton-complete)**: `exam.py` (variant retrieval exam, grading, refuses unfrozen bars), `probes.py` (splice/cross-label builders + graders), `stability.py` (rerun comparator), `seeds.py` (σ_seed→MDE). First execution = the certification run itself | **lock** | `certify/` |
| O5 | v3 metrics + lifecycle — **ADVANCED (implemented, GPU-unexecuted)**: `s_structure.py` (sidedness-aware core + flagged fallback), `m1_transfer.py` (M1a/M1b/M1c), `m2_integrity.py` (M2a/b/c + provenance), `controls.make_static_hold`, `manifests_v3.py` (3 schemas + tier/sidedness derivation), `plan.py`, `score.py` (stamped, completeness-checked, control arms, paired table). Synthetic contract tests in `tests/test_transition_eval_v3.py`. First real-data run = certification; scorer forks retire at that moment | **lock** | §3/§9 |
| O6 | σ_seed measurement (≈12 stratified items × 5 seeds) | cert | §4/§6 |
| O7 | M1b robust global-fit weighting choice (Huber/IRLS/RANSAC-lite) | cert | §3 M1b |
| O8 | Δ-novelty profile displacement test (vs depth + profile-DTW) | roster | §3 M1d / §6 |
| O9 | Judge human-calibration set (~50–100 labels, Spearman ≥0.8) + q1 name-the-mechanism sharpening decision | post-lock | §3 M4 |
| O10 | Rescore exp_056–058 archived items under certified v3 (continuity bridge) | post-lock | §7 |
| O11 | Update repo guidelines (CLAUDE.md / skills) to the v3 workflow | post-lock | §10 |
| O12 | Repo hygiene: commit the exp_044–059 backlog in the main checkout; single-writer discipline for CHANGELOG.md | advisory | out of spec |

Severity: **lock** = blocks the `eval/v3.0` tag · cert = resolved during first certification run · roster = pre-registered test candidate · post-lock = scheduled after tagging.

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
**Core mask:** two-sided → `max(â,b̂) < 0.5`; one-sided → `â < 0.5` (the effect's terminal state *is* endpoint B there); conditioned windows always excluded. Fallback: if `|core| < k` (k=8 draft, O3) → frames within `env_min + δ` (δ OPEN, O3) and `core_degenerate=true`. Emits core size, `cross`, `cross_high (>0.85)` → trust flags. Scalars: depth, depart, arrive (two-sided only), core_frac.
*Blind to:* effects below CLS sensitivity; semantically-close endpoints degrade it (flagged, not hidden).

**M1a — Appearance transfer.** Symmetric mean-of-max cosine between gen-core and **reference**-core features. [−1,1], ↑. *Blind to:* copy-vs-transfer (M2a disambiguates); camera classes (taxonomy-flagged advisory); inherits core-mask health.

**M1b — Camera motion match.** Per-step robust visibility-weighted similarity fit `(dx, dy, log s, θ)` on tracklets (weighting O7) → 4-channel trajectory, resample 64, z-norm → banded DTW (band 0.15) + correlation vs reference. ↑corr/↓DTW. *Blind to:* parallax/3D (SfM rejected: scenes are non-rigid/dissolving); flagged when trackable points < N.

**M1c — Object motion match.** MFS (bidirectional mean-of-max velocity-direction correlation; 64 steps, speed floor 0.1, min_vis 0.2, min_moving_frac 0.05) on **residual** velocities after removing the M1b global fit. ~[−1,1], ↑; NaN reported, never imputed. *Blind to:* magnitude, spatial arrangement; requires per-class exam trust.

**M1d — Timing (analysis tier).** depart / arrive / depth from S. Candidate Δ-novelty (residual off span{eA,eB}) enters only by displacing depth + profile-DTW (O8).

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

**Seeds:** routine suites = 1 seed (42), fully paired. σ_seed certified once per model family (≈12 stratified items × 5 seeds at certification, O6); every reported delta carries the σ_seed-derived minimum detectable effect; within-noise decisions escalate to 3 seeds, targeted.

**Headline table:** `arm | n | M1a | M2b (margin + intruder) | M2a | M1b | M1c | M3a | M3b` + flags (`†` exam-untrusted, `core_degenerate`, `near_copy`, `cross_high`, `camera-advisory`, `partial`, `UNCERTIFIED`). Analysis table: M1d, M2c, M4.

## 5. Splits & data

`corpus_manifest.json` (O1) is the single source of truth. Corpus today: `data/processed/transitions_std121/` — 223 clips / 39 classes; 320p-source classes excluded (upscale confound). Exam split = whole corpus (self-supervised, zero generations). Tiers (A held-out / B trained-eval-clips-excluded / C seen) are derived per adapter from the training manifest.

**Rules:** corpus additions require the order-invariant dedup gate (DINO set-sim ≥ 0.90 + dHash-bag confirm ≤ 10) and an exam re-run before first use; any corpus/split change → §6 re-certification; the corpus hash is stamped on every result.

## 6. Certification (bars pre-registered at the health-design session — O3; numbers frozen *before* running)

1. **Exam:** LOO 1-NN retrieval + Cohen's d on the full current corpus, per-class recall → trust flags, for M1a/M1b/M1c under every contested variant. Adoption rule: ≥ incumbent overall ∧ strictly better on target stratum (one-sided classes for the core mask; camera classes for M1b) ∧ no previously-trusted class regresses below threshold.
2. **Adversarial probes:** M2a — ref-segment splice ground-truth copies all ≥ τ_copy with honest-set max clearly below (gap form per exp_053 check C); M2b — deliberately cross-labeled references yield negative margin naming the true source; controls land at the transfer floor and trip degeneracy flags.
3. **Twin sanity:** copy-regime base twins detected 100% (near_copy + seam), per the exp_057 11/11 precedent.
4. **Stability:** warm-cache rerun bit-identical; cold-cache within stated tolerance; anchor items reproduce raw within ±0.04. σ_seed measured here (O6).
5. **M4 exempt** until human-calibrated (O9); re-enters via this section.
6. **Gate:** any §7-relevant change → this suite re-runs before any model number is reported.

Each certification writes a committed record `certifications/v<X.Y>.md` (bars, numbers, verdicts, corpus hash, artifact pointers) — the tag is authorized by that record.

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
├── rubric.py judge_gemini.py judge.py         # M4 (advisory)
├── controls.py                                # lerp + static-hold degenerate controls
├── manifests_v3.py                            # 3 schemas + tier/sidedness derivations
├── plan.py  score.py                          # lifecycle CLIs (score = the ONE scorer)
├── build_corpus_manifest.py                   # corpus manifest builder (O1 tooling)
├── certify/                                   # §6 as code: bars.yaml (DRAFT), exam,
│                                              #   probes, stability, seeds
└── certifications/                            # committed records, one per certified version
```

Regeneration: annotated tag `eval/vX.Y` per certified release; `git worktree add ../eval-vX.Y <tag>` reproduces any old harness — no snapshot folders, no vendored copies. Rejected anti-patterns: per-version package copies, metric code in experiment dirs, floating `torch.hub`/HF `main` references.

## 10. Harness-change protocol (the operational loop — to be encoded in repo guidelines, O11)

1. Branch `eval/<slug>` (worktree if other work is live in the checkout).
2. Edit **code and SPEC.md together** — a metric change without its §3 block updated is an incomplete change. Update the §0 OPEN register.
3. Bump `VERSION` (iterate as `X.Y.Z-draft.N`; certified releases drop the suffix).
4. `pytest -q tests/test_transition_eval.py tests/test_versioning.py` — green.
5. `python -m diffusion.transition_eval.versioning --corpus <corpus_manifest>` — review the stamp.
6. Run certification (§6) → committed record `certifications/v<X.Y>.md` + artifacts under `outputs/eval/certification/<version>/`. *(Until O4 lands: version stays draft; drafts cannot produce headline numbers.)*
7. `CHANGELOG.md` entry (repo rule).
8. Commit package + SPEC + VERSION + certification record together. On PASS: drop `-draft`, annotated tag `eval/vX.Y`, push branch + tag, open PR. On FAIL: stays draft, no tag.
9. Cross-version comparisons only via rescoring old items under the new tag.

---

## Spec changelog

- **3.0.0-draft.4** (2026-07-10): O1 resolved (corpus_manifest.json, 39/223, contract portrait-corrected to 480w×640h); O3/O4/O5 advanced to implemented-draft — `s_structure/m1_transfer/m2_integrity/manifests_v3/plan/score` + `certify/{bars.yaml DRAFT, exam, probes, stability, seeds}` + static-hold control + v3 synthetic test suite; §9 map updated to actual filenames.
- **3.0.0-draft.3** (2026-07-10): consolidated in-repo; added §0 OPEN register, §9 implementation map + code versioning, §10 change protocol; `versioning.py` + `VERSION` land; package brought under VCS.
- 3.0.0-draft.2: three-manifest model (sidedness/tier derived, not stored); plan→infer→score lifecycle; σ_seed-once seeds policy; metric IDs → task anatomy (S/M1/M2/M3/M4).
- 3.0.0-draft.1: first full spec from the v3 design sessions (leakage 3-axis split, normalization demoted to controls, M2 camera/object decomposition, sidedness-aware core mask, Δ-novelty roster).
