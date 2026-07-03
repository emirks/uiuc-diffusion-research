# exp_053 — Eval harness v2: adversarial validation, trust flags, decision-grade numbers

## Question

exp_052 certified the instrument on ground truth; this iteration makes its
numbers **decision-grade** before anything gates on them. Three sub-questions,
pre-registered:

1. **Can M3 drop its M1 dependency?** (check A) Effect appearance currently
   runs on core frames, which requires the morph profile's mask. If all-frames
   appearance retrieves style equally well on the 41 real clips AND keeps
   floor/ceiling separation, the mask is not earning its keep and M3
   simplifies to "DINO set-similarity, full video".
2. **Does the anti-cheating axis actually fire on real cheats?** (check C)
   exp_046/048 produced known near-copies (donor-pin: another reference clip's
   smoke spliced into a target; self-recon injection at g=1.0: the clip's own
   middle reproduced). The lerp control covers the "too little transition"
   failure; these cover the opposite one. M6 leakage must scream on them.
3. **Which cells of a score table may a decision trust?** (check B + reporting)
   Motion fidelity passed its exam only on some styles; flame/jump ceilings
   rest on 1 LOO neighbor. Encode this per-style reliability into the standard
   report instead of leaving it in prose, and add the missing uncertainty
   (mean±std per arm, Wilson CIs on exam accuracies) — at n=6/arm, 0.69 vs
   0.78 is currently indistinguishable from noise by eye.

Plus one backend swap: the local Gemma judge sees 8 stills and failed
q2(dynamics)/q5(artifacts) on all 24 ladder items — a motion question graded
from stills. Swap to **Gemini API (`gemini-3.5-flash`, pinned, native video @
8 fps, temperature 0, JSON out, every raw response cached)** and re-judge the
same 24 items.

Deliberately NOT done (evaluated and rejected for now):
- Core-region-restricted motion fidelity v2 — motion's blind spots are covered
  by appearance; no pending decision hinges on per-style motion for
  medium-defined styles.
- Adding water/display endpoint pairs to the exp_051 ladder — motion trust is
  a property of the REFERENCE style (still shadow_smoke), not of the endpoint
  content, so new endpoint pairs would not un-flag the motion column; trusted
  motion cells arrive with multi-style generation after Higgsfield curation.
- Composite scores — still banned.

## Setup

Library changes (`src/diffusion/transition_eval/`):
- `rubric.py` (new): backend-agnostic checklist + `judge_pass_rate` incl.
  `all_pass`; q2/q5 get explicit severity-calibration clauses.
- `judge_gemini.py` (new): native-video judge, pinned model, response cache.
- `report.py`: `wilson_interval`, `trust_flags` (motion exam-certified per
  style; ceiling needs ≥4 ref clips), `score_tables` — the standard
  headline/analysis split every future run lands in. Headline = appearance /
  motion / judge pass / endpoint DINO / seam z / leak max-sim; analysis = M1
  profile+timing scalars (they saturate under endpoint conditioning). Pearson
  dropped from tables (DTW only).
- `morph.py`: `cross_high` flag (endpoint cross-sim > 0.85 ⇒ unstable
  normalization — flagged, not trusted).

Experiment scripts:
- `run_check.py` — checks A/B/C in one GPU job:
  A: all-frames vs core-mask appearance exam (cached features; adds lerp
  floor/ceiling separation per variant). B: motion sanity from the cached
  exp_052 distance matrix — per-style within > cross. C: score the
  adversarial manifest (leakage, endpoint fidelity, seams; no tracker).
- `make_manifest_adversarial.py` — 26 items: src_copy(4), donor_pin(8),
  donor_blend(3), self_inject_g1(3), self_inject_g08(2), and live-branch
  negatives neg_tempblend(2), neg_velguide(4) from exp_046/047/048 outputs.
- `run_score.py` (v2 scorer, forked from exp_052) — manifest-driven endpoint
  windows, cross_high propagation, no inline judge, `score_tables` reporting
  with trust flags + mean±std, per-item scatter figure, `--from-items` mode
  to re-report existing items.jsonl without recomputation.
- `run_judge_gemini.py` — API judge over a manifest; login-node, no GPU.

### Resource plan

- **GPU**: one `HCESC-L40S-normal` job (`job_check.sbatch`) for run_check —
  ~26 new videos through DINO+LPIPS (~20 min), everything else cache hits.
  Walltime 1:30 (×1.5 buffer). Fallback: cluster `secondary`.
- **Login node**: `run_score.py --from-items` (numpy re-aggregation, seconds)
  and `run_judge_gemini.py` (API-bound; 24 items, sequential, backoff on 429).
- **Resumability**: feature cache as before; judge responses cached per item.

## How to run

```bash
cd $LAB/diffusion-research
python experiments/exp_053_eval_harness_v2/make_manifest_adversarial.py
sbatch experiments/exp_053_eval_harness_v2/job_check.sbatch
# ladder re-report (no GPU):
PYTHONPATH=src $LAB/envs/diffusion/bin/python experiments/exp_053_eval_harness_v2/run_score.py \
  --from-items outputs/eval/exp_052/ladder/run_0001/items.jsonl --label ladder_v2
# judge (login node; GEMINI_API_KEY or ~/.config/gemini/api_key):
PYTHONPATH=src $LAB/envs/diffusion/bin/python experiments/exp_053_eval_harness_v2/run_judge_gemini.py \
  --manifest experiments/exp_052_transition_eval_harness/manifest_exp051.json --label ladder
```

## Expected outcome

Pre-registered:
(a) **Check A adoption bar**: adopt all-frames M3 iff 1-NN acc ≥ 0.88 AND
real-vs-lerp paired separation holds (real LOO sim > lerp sim for ≥90% of
clips, Cohen's d ≥ 1.0). Otherwise the core mask stays. Prediction: within a
few points of 0.93 either way — endpoints are ~14% of frames and mean-of-max
is robust; expect ADOPT.
(b) **Check B**: within-style motion fidelity > cross-style for ≥7/9 styles;
failures concentrated in exam-weak styles (earth_wave, flame).
(c) **Check C hard bar**: every src_copy / donor_pin / self_inject_g1 item
has `leak_max_sim_target ≥ 0.88` (exp_051 honest max was 0.78; near-copy
regime ≈0.9+). ANY miss ⇒ M6 is broken for real cheats ⇒ fix before any
other result is used. donor_blend / self_inject_g08 / negatives are reported
without a hard bar (donor_blend likely still flags; velocity-guided items are
the diagnostic of interest — where does "steered toward donor smoke" land?).
Endpoint DINO expected HIGH for these items (their endpoints were pinned to
the target's own anchors) — the near-copy failure mode is leakage, not
endpoints.
(d) **Judge swap bar**: 24/24 parse; q1/q3/q4 directionally consistent with
the local backend; q2/q5 no longer degenerate (not all-false and not all-true
across arms). If still degenerate ⇒ judge stays advisory and the questions —
not the backend — are the suspect.
(e) **Ladder v2 re-report**: same means as exp_052 (identical inputs), now
with ±std; expected verdict: base-vs-LoRA separation survives error bars on
appearance and motion; i2v-vs-c2v appearance difference (0.69 vs 0.78) likely
does NOT (overlapping ±std) — stated as such.

## Outputs

- `outputs/eval/exp_053/checks/run_NNNN/` — checks_report.md, checks.json,
  adversarial.jsonl, leak_scatter.png
- `outputs/eval/exp_053/ladder_v2/run_NNNN/` — report.md (headline/analysis),
  scatter.png
- `outputs/eval/exp_053/judge_gemini_ladder/run_NNNN/` — judge_results.json,
  judge_summary.json (+ response cache under judge_cache/)
- W&B: `exp053_checks`, `exp053_ladder_v2` in `creative-transition-transfer`.

## Outcome

**Checks completed 2026-07-07.** Jobs 9364990 (L40S-normal) + twin 9364995
(secondary H100), both EXIT 0 in ~6 min; twins agree to the 7th decimal.

Against the pre-registration:
(a) **KEEP_CORE_MASK** — all-frames M3 fell to 0.780 [Wilson 0.63–0.88]
vs core-mask 0.927 [0.81–0.97], missing the 0.88 adoption bar decisively
(prediction of ADOPT was wrong; endpoint content costs ~15 points and halves
Cohen's d). M3 keeps its M1 dependency.
(b) **PASS 8/9** — within-style motion > cross-style everywhere except
raven_transition (exam recall 0.25); per-clip fraction 0.88.
(c) **M6_OK after audit** — all 12 unambiguous copies (src_copy min 0.926,
donor_pin min 0.943) clear the 0.88 bar; honest generations max 0.78. The one
initial miss (self_inject_g1 ss1, 0.742) audited to a MISLABELED item:
exp_049 documents ss1 as z1-poor (PSNR 11.88, "nothing to recover") — the
output genuinely diverges, and the harness independently reproduced the
z1-rich/poor dichotomy (ss0 0.951/ss2 0.934 vs ss1 0.742). Audit:
`checks/run_0001/checkC_audit.json`. Bonus: neg_tempblend flags at 0.9 —
correctly, it contains literal donor smoke.
(d) **Judge swap** — gemini-3.5-flash's video path 503'd persistently
("high demand") → pinned `gemini-3-flash-preview` (model_version recorded per
cached response; free-tier 20/day cap stalled at 15/24 until billing was
enabled). **Final: 24/24 parsed, 0 errors — pre-registered bar PARTIALLY met**:
q2/q5 degeneracy fixed (differentiated, timestamp-cited answers) and q1/q3/q4
consistent, BUT q1–q4 now pass every arm including base; only q5 artifacts
discriminates (base 1.00 > t2v 0.83 > c2v 0.67 > i2v 0.50) — a cleanliness
ranking inverted from the style axes. Judge = advisory quality axis, not a
transfer metric; human validation remains the gate.
(e) **Ladder v2** — same means, now with ±std and trust flags; paired
per-cell analysis added: base-vs-LoRA separation SURVIVES (6/6 cells on
appearance, sign p≈0.016/arm), within-LoRA differences do NOT at n=6 (the
pre-registered expectation that i2v-vs-c2v would not survive error bars:
CONFIRMED). Details: `notes/exp/exp_053_eval_harness_v2.md`.
