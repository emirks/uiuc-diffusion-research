# Transition-Eval Harness v3 — positioning & entry points

**Status: CERTIFIED `eval/v3.0.0` (2026-07-14).** The single authoritative
document is **`src/diffusion/transition_eval/SPEC.md`** — purpose, input
contract, metric formulas, lifecycle, the full health-assessment spec (§6),
versioning invariants, and the change protocol (§10). This note is the map,
not a copy; when it disagrees with SPEC, SPEC wins.

## What it is

The one instrument for scoring video-transition generations (LTX-2 IC-LoRA
adapters etc.) against the 39-class / 223-clip labeled corpus
(`data/processed/transitions_std121/corpus_manifest.json`). Metric IDs follow
task anatomy: **S** structure (sidedness-aware core mask) · **M1** transfer
(M1a appearance-to-reference, M1b camera, M1c object) · **M2** integrity
(M2a copy, M2b intrusion, M2c memorization) · **M3** endpoints/seams ·
**M4** judge (advisory). Raw scores + control arms + paired base-twin deltas;
no normalization, no composite score. v2 is retired; v2↔v3 numbers are not
comparable (a per-item bridge exists in the certification artifacts).

## What "certified" means (and doesn't)

Certification tests the **ruler, never the model**: one stamped execution of
`certify/` against bars pre-registered in `certify/bars.yaml`, producing the
committed record that authorizes the git tag. v3.0.0's record
(`src/diffusion/transition_eval/certifications/v3.0.0.md`) is an overall
PASS — a disclosed regrade of the complete draft.8 run. The claims paragraph
in the record is the exact scope; notably it does NOT claim metrics track
human judgment (M4 is advisory until O9), and **σ_seed is still PENDING — it
gates the first model report, not the tag**.

Trust is per-class, per-metric: every model report must consume the exam's
trust map (`outputs/eval/certification/3.0.0-draft.8/exam/trust_map.json`);
n<4 classes are auto-untrusted. Known blind spots are enumerated in the
record (nature_bloom content confound; M1b time-antisymmetric reversal
blindness; content-invariance alarm 0.82, non-gating).

## How following work uses it (the flow)

1. `plan` → `suite.json` (harness, CPU; twins + controls auto-included).
2. `infer` — model side, external; fulfill suite.json to the §2 contract.
3. `score` — `PYTHONPATH=src python -m diffusion.transition_eval.score
   --manifest M --corpus C --label L` from a checkout at tag `eval/v3.0.0`
   with a clean tree; anything else stamps UNCERTIFIED on every row.
   Results: `{results.json, items.jsonl}`, headline table per SPEC §4.
4. Any instrument/corpus change → new version + re-certification (§7/§10);
   bars are frozen — never adjusted after seeing an outcome.

Inspection: every certification run auto-writes `figures/*.png` and
`results_explorer.html` under `outputs/eval/certification/<version>/`.

## Open items before/around first model report

O6 σ_seed (12 items × 5 seeds, gates the first report) · τ_copy pin 0.88 vs
record-calibrated 0.858 (owner decision; adopting = instrument change =
re-run) · O8–O12 post-lock (Δ-novelty roster, judge calibration, archive
rescoring, repo-guideline update, main-checkout hygiene) · PR #2 merge
timing (owner). Certification cost after the perf work: ~35–40 min on one
H100, floor set by the deliberately cold anchors.
