# exp_053 — Eval harness v2: adversarial validation, trust flags, decision-grade numbers

**Status: COMPLETE 2026-07-07. Headline: the core mask survived its ablation (all-frames M3
fails the bar), M6 leakage is adversarially validated on 12 ground-truth copies (min 0.926 vs
honest max 0.78), the ladder's base-vs-LoRA separation survives paired testing while
within-LoRA differences do not, and the Gemini native-video judge is functional (24/24,
degeneracy fixed) but measures cleanliness, not transfer — advisory pending human labels.**

Jobs: checks 9364990 (L40S-normal) + twin 9364995 (secondary H100) — both EXIT 0 in ~6 min
(cache-hits); twin runs agree to the 7th decimal, verdict-identical.

## 1. Check A — can M3 drop its M1 dependency? **NO (KEEP_CORE_MASK)**

| variant | 1-NN acc | Wilson 95 | Cohen's d | lerp sep_frac | sep_d |
|---|---|---|---|---|---|
| core-mask | 0.927 | [0.81, 0.97] | 2.04 | 0.976 | 2.00 |
| all-frames | 0.780 | [0.63, 0.88] | 1.11 | 1.000 | 1.82 |

All-frames missed the pre-registered 0.88 bar by a distance: endpoint content dilutes the medium
signal by ~15 points of accuracy and halves the separation d. The mask earns its keep; M3 keeps
its M1 dependency (which check C shows is fine — the whole chain works end-to-end).

### Check A addendum — "isn't all-frames more robust?" (re-analysis 2026-07-07)

Follow-up to the objection that a metric with no mask dependency should be more reliable even at
lower accuracy. Re-ran check A's two similarity matrices on the login node (cached feats, pure
numpy: `build_pair_examples_notebook.py`). Numbers below are on the **deduped 37-clip corpus** (§6);
the pre-dedup 41-clip version reached the same conclusion — dedup only sharpens it. The robustness
argument favors the mask, not all-frames:

- **Same measurement, denoised — not a different one.** Core-mask vs all-frames pair similarity
  rank-correlate **Spearman ρ=0.79** across all 666 pairs. Endpoints don't corrupt M3, they dilute it.
- **Strict (Pareto) improvement.** Going core→all-frames, **7 clips flip correct→wrong and 0 flip
  the other way** — no clip all-frames retrieves that the mask misses. (Appearance exam: core
  **0.865** vs all-frames **0.676**.)
- **Wider decision margin = more robust to noise.** Mean (best same-style − best diff-style sim):
  **core +0.169 vs all +0.028** — the mask ~6× the headroom, the robustness that actually matters
  against per-frame/seed/decode noise.
- **The mask never misfires on real data.** `cross_high` (endpoint cross-sim > 0.85 → unstable
  normalization) fires **0/37** (max cross-sim 0.53, median 0.12); **0** clips hit the 1-frame
  fallback (kept frames min 4 / median 49 / max 133); retrieval is threshold-flat (0.865 for thresh
  0.40–0.50, 0.838 at 0.55–0.60) — not a knife-edge.
- **Targeted, not global.** flame/raven/shadow_smoke/water are a literal no-op (1.00 either way);
  the mask only helps the fluid styles where generic scenery swamps the effect (melt +0.75,
  earth_wave +0.40, display +0.33, flying_cam +0.25) — melt's all-frames recall collapses to 0.00
  once its trivial resolution-twin is removed.

Conclusion: keep the core mask; the only legitimate hedge is a `cross_high → all-frames` fallback
for M3, a **no-op on every current style** (0/37) → insurance for future portal-class styles, not
needed by today's numbers. Worked examples (dilution + the 7 flips + negatives/confusions):
`experiments/exp_053_eval_harness_v2/pair_examples.ipynb`, montages under
`outputs/eval/exp_053/pair_examples/`.

## 2. Check B — motion sanity: within-style > cross-style for **8/9 styles** (per-clip 0.88)

Only raven_transition inverts (within 0.052 < cross 0.080) — consistent with its exam recall
(0.25). Passes the pre-registered ≥7/9 bar; the exam-weak styles are exactly the inverting ones.

## 3. Check C — adversarial near-copies: **M6_OK after ground-truth audit**

26-item manifest from exp_046/047/048 (`manifest_adversarial.json`): src recons, donor-pin
splices, donor blends, self-recon injections, live-branch negatives.

| arm | n | leak max-sim mean | min | verdict |
|---|---|---|---|---|
| src_copy (recon of the ref itself) | 4 | 0.940 | **0.926** | screams ✓ |
| donor_pin (another ref's smoke spliced in) | 8 | 0.963 | **0.943** | screams ✓ |
| donor_blend 0.7 | 3 | 0.866 | 0.847 | gray zone (softened splice) |
| self_inject_g1 / g08 (exp_048) | 5 | 0.876/0.848 | 0.742 | see audit |
| neg_tempblend (contains real donor smoke) | 2 | 0.892 | 0.869 | flags — CORRECTLY |
| neg_velguide (generations) | 4 | 0.830 | 0.685 | below near-copy regime |

- **Hard bar (0.88) passed by all 12 unambiguous copies** (src_copy + donor_pin, min 0.926);
  honest exp_051 generations max 0.78 → clean near-copy threshold at ~0.88.
- **Audit**: the initial run flagged `self_inject_g1` ss1 (0.742) as a miss. exp_049's record
  (z1-poor clips "have nothing to recover"; ss1 PSNR 11.88) shows that output genuinely diverges
  from the source — a mislabeled test item, not a broken metric. The harness independently
  reproduced the z1-rich/poor dichotomy (ss0 0.951 / ss2 0.934 vs ss1 0.742 with appearance
  collapsed to 0.509). Hard-bar arms re-scoped to unambiguous copies; audit trail in
  `outputs/eval/exp_053/checks/run_0001/checkC_audit.json`.
- tempblend flagging at 0.9 is the metric being RIGHT (it literally contains donor smoke) —
  a warning for any future "borrow real reference latents" recipe: the harness will call it
  copying.

## 4. Uncertainty + the standard report shape (ladder_v2)

`report.score_tables` is now the standard output: HEADLINE (appearance / motion / judge pass /
endpoint DINO / seam z / leak max-sim) vs ANALYSIS (M1 scalars — they saturate under endpoint
conditioning); mean±std everywhere; per-style trust flags († motion not exam-certified for the
style, e.g. shadow_smoke at recall 0.4; ‡ <4-clip ceilings: flame/jump); Wilson CIs on exam
accuracies; Pearson dropped from tables. `run_score.py --from-items` re-reports any old
items.jsonl in seconds.

exp_051 ladder re-reported (`outputs/eval/exp_053/ladder_v2/run_0001/`):

- Unpaired: appearance base 0.56±0.25 vs LoRA 0.69–0.79 (±0.27-0.31) — stds overlap because they
  mix endpoint-pair difficulty.
- **Paired per cell (the honest test): every LoRA arm beats base 6/6 on appearance (sign test
  p≈0.016/arm; min Δ +0.006..+0.063) and 5-6/6 on motion.** Base separation is real.
- **Differences among t2v/i2v/c2v are NOT resolvable at n=6** (overlapping paired deltas) — do
  not gate route decisions on e.g. i2v 0.69 vs c2v 0.78 without more samples.
- Trigger claim reworded everywhere: "no detectable trigger effect (n=3/cell)" — if it holds,
  the style is always-on and the trigger-switched multi-style route dies; needs a dedicated
  probe (wrong-trigger + no-trigger × ~10 seeds) before Week-3 decisions.

## 5. Judge backend swap (Gemini API, native video)

`judge_gemini.py`: pinned model, native video both-clips @ 8 fps, temperature 0, JSON out,
every raw response disk-cached (item-keyed, model_version recorded). Rubric moved to
backend-agnostic `rubric.py`; q2/q5 got severity-calibration clauses; `all_pass` (all-5) is the
headline judge number. Env: runs on the login node, no GPU/torch. Key at
`~/.config/gemini/api_key`.

- `gemini-3.5-flash` (GA, first choice) 503'd persistently on video requests 2026-07-07
  ("high demand" — text requests fine); **pinned `gemini-3-flash-preview`** which serves video
  reliably. Revisit the GA pin when capacity settles.
- Free-tier quota (20/day for gemini-3-flash) stalled the first pass at 15/24; user enabled
  billing → **COMPLETE 24/24, 0 parse errors** (run_0004; the scheduled quota-reset finisher
  job 9365640 was cancelled). Two backend fixes landed during the pass: response **schema
  pinned** via structured output (one item had answered in an invented score format —
  `response_mime_type` alone does not pin shape), and 429 handling honors the server's
  `retry in Ns` hint.
- **Final results (24/24)**: q1 same-type, q2 dynamics, q3 endpoints, q4 no-leakage = 1.00 for
  EVERY arm including base. The only discriminative question is **q5 artifacts: base 1.00 >
  t2v 0.83 > c2v 0.67 > i2v 0.50** (all_pass identical). Read: the q2/q5 all-false degeneracy
  is FIXED (differentiated, timestamp-cited answers), but the judge now measures visual
  CLEANLINESS, not style transfer — base makes clean wrong-style videos and tops the judge
  ranking, exactly inverted from the style axes. Its q5 signal (LoRA arms carry more visible
  artifacts than base) is plausible and worth keeping as an advisory quality axis.
- Verdict: judge is functional but NOT a transfer metric on this ladder; stays ADVISORY.
  Concrete questions for the human-validation set: (1) is q1 too lenient on base ("any smoke
  mechanism present → true"; exp_051's visual read said partially)? (2) does q5's artifact
  ranking track human perception of the LoRA arms?

## 6. Reference-corpus dedup + new styles landing (2026-07-07)

**Dedup — 4 clusters, corpus 41 → 37.** `detect_duplicates.py`, layered: SHA-256 exact →
**order-invariant** DINO set-similarity (gate ≥0.90) → independent dHash-bag confirm (≤10),
calibrated against the distinct-pair distribution (cross-style set-sim maxes at 0.42, same-style
non-dup at 0.43; dups sit ≥0.977 — a clean gap):
- 3 **resolution-twins** (same clip @600px + ~1244px, set-sim ≥0.988, dHash 0): `melt_{0,3}`,
  `jump_{0,1}`, `display_{0,3}`.
- 1 **trim+downscale**: `flying_cam_{0,1}` — clip_0 is a 193-frame 600px cut of the 242-frame
  1244px clip_1 (set-sim 0.977, dHash-bag 4.9).

**Method lesson (user-caught):** v1 gated on *temporally-aligned* per-frame cosine and **missed the
flying_cam trim** (aligned 0.876 < 0.90 gate) — a length edit shifts the same events to different
normalized times, decaying alignment. The order-invariant set-similarity (0.977) and dHash-bag
(4.9) caught it immediately. **Gate dedup on an order-invariant metric; keep aligned cosine only as
temporal-identity context.** Byte-hash alone would miss all four (none are byte-identical).

Dropped clips (the lower-res / shorter twin) moved reversibly to
`data/processed/transitions/<style>/_dup/` (non-recursive glob excludes them). Proof/provenance:
`outputs/eval/exp_053/dedup/{duplicates.json, duplicates.png, flying_cam_0_vs_1.png}`.

Exam impact (core-mask LOO 1-NN): appearance headline **0.927 → 0.865** — the removed clips were
trivial self-retrievals, so the drop is the corpus getting *honest*, not the metric degrading. Two
weak spots now exposed: **jump_transition** had only 2 clips and they were the same clip → now a
**singleton, untestable** (its old recall 1.0 + ceiling were self-retrieval); **flying_cam** recall
falls **0.60 → 0.25** — the dup was propping it up, and it's really a *motion*-defined style whose
appearance is too scene-diverse to retrieve (rely on its motion fidelity, not M3). **Follow-up:
re-run `run_validation.py`** on the 37-clip corpus to refresh motion exam / ceilings / trust flags
(cached exp_052 validation still reflects 41).

**New styles landing.** Two styles were dropped into the corpus mid-session: `air_bending` (4 clips,
15:56) and `firelava` (6 clips) — both with `_manifest.json`, no cached DINO features yet, so both
are excluded from every analysis above until GPU-processed, and each needs its own dedup pass once
features exist. This is the Higgsfield/multi-style curation starting; when the batch settles,
extract features, dedup, and re-validate (which also revisits jump's untestable status and the
motion/ceiling trust flags).

## 7. Full re-validation on the deduped + expanded corpus (exp_054, 2026-07-07)

Executed the §6 follow-up. Ran `run_validation.py` on the **complete current
corpus — 47 clips / 11 styles** (37 deduped originals + the 2 new styles
`air_bending` 4 + `firelava` 6). Job 9366838 (L40S-normal, phase-1 EXIT 0, ~9
min) → `outputs/eval/exp_052/validation/run_0002/`. Full write-up:
`outputs/eval/exp_054_full_revalidation/REPORT.md`.

**Exam (LOO 1-NN, chance 0.213):** effect_appearance **0.851** [Wilson 0.72–0.93],
**d=2.22** (↑ from 2.04 at 41 clips — separation improved as accuracy fell:
the corpus got honest, not the metric worse); motion 0.255; morph_dtw 0.191.
Appearance overall trajectory: **0.927 (41) → 0.865 (37, notebook) → 0.851
(47)** — the last step is purely the 2 new styles being imperfect (0.75/0.83 vs
~1.0), not degradation.

**The 2 new styles discriminate on APPEARANCE only:**
- air_bending: appearance **0.75** (1 miss → shadow_smoke), motion **0.0**.
- firelava: appearance **0.83** (1 miss → air_bending), motion **0.0**.
Both are ceiling-trusted (n≥4) appearance-axis additions; neither is
motion-discriminable via CoTracker fidelity → their motion columns must stay
flagged. New morph profiles for both are in run_0002's `morph_profiles.png`.

**Post-dedup weak spots confirmed exactly as predicted in §6:** jump is a
**singleton (appearance/motion/morph all 0.0, untestable)**; flying_cam
appearance **0.25** (misses scatter → display/firelava/shadow — no coherent
appearance) but motion **0.5** (motion-defined, motion-trusted).

**New-style dedup (`detect_duplicates.py`, full 47-clip):** 0 candidates above
the 0.80 cut. Max within-style set-sim **0.453 (air_bending) / 0.490
(firelava)** — the 600×800 `*_0` clips are distinct kling_motion sources, NOT
resolution-twins (contrast the 0.977–0.992 of the 4 real twins). Nothing moved,
no re-validation. `outputs/eval/exp_053/dedup/duplicates_v2.json` = empty
clusters.

**Trust flags refreshed:** config `validation_run` repointed run_0001 →
run_0002; ladder re-reported via `run_score.py --from-items
outputs/eval/exp_052/ladder/run_0001/items.jsonl --label ladder_v3` →
`outputs/eval/exp_053/ladder_v3/run_0001/`. jump → both flags False (was
self-retrieval-trusted at n=2); flying_cam → motion-trusted (0.5) /
ceiling-trusted, appearance-untrustworthy. shadow_smoke motion 0.40 stays <0.5
→ the ladder's motion column stays `†`-flagged; ladder_v3 headline is therefore
numerically identical to ladder_v2 (the ladder is shadow_smoke-only). New styles
air_bending/firelava enter the trust registry (ceiling-trusted, motion-untrusted).

**Two operational lessons:**
- **A singleton reference style breaks full-compute scoring.** Phase-2 of
  `job_eval.sbatch` (ladder re-score) crashed `ValueError: zero-size array to
  reduction minimum` at `run_score.py:57` — a style's LOO ceiling iterates
  `[o for o in bs if o is not b]`, which is empty for jump (n=1) → `np.min([])`.
  Present in BOTH the exp_052 and exp_053 scorers. `--from-items` re-reports
  dodge it (no ceiling recompute). Guard `fidelity_vs_refs` (or exclude/grow
  jump) before any future full-compute score.
- **`_dup/` reversibility was incomplete for melt.** Three of the four 41→37
  drops were parked in `<style>/_dup/`, but `melt_transition_0` had only been
  removed from the active tree. Recovered byte-identical from the raw source
  `data/processed/higgsfield_transitions/melt_transition/` (600×800 twin of the
  kept melt_3) and placed in `melt_transition/_dup/` with a README. All 4 twins
  now follow the same reversible convention; active exam unchanged.

## Reuse (updated)

- Score a generated set: manifest → `experiments/exp_053_eval_harness_v2/run_score.py
  --manifest M --label L [--judge-summary J]`.
- Judge: `run_judge_gemini.py --manifest M --label L` (login node; resumable via response cache).
- Re-report old items with current tables: `run_score.py --from-items <items.jsonl> --label L`.
- Adversarial re-check after M6 changes: `run_check.py` (checks A/B/C; job_check.sbatch).

## Open questions / next steps

- Judge human-validation set (~50–100 labels, Spearman ≥0.8) — still THE gate before judge
  numbers become headline.
- Trigger/always-on probe (cheap, one GPU-day) — decides the multi-style route.
- Higgsfield curation → 10–15 styles × 10–20 clips + portal class → re-run the exam; flame/jump
  ceilings and the 1-endpoint path stay untrusted/unexercised until then.
- donor_blend sits in the 0.85–0.88 gray zone — if softened-splice recipes ever return, tighten
  the near-copy bar or add a temporal-consistency leak check.
