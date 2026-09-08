# EffectData gapper screen — stage A (owner 2026-09-08: "just do 300 effects, one seed; pick wisely")

## Question
Which EffectData effects does prompting alone (base_cond + the effect clause, no reference) FAIL on? On grid v3 the
prompt-only effect level is the only cheap predictor of the DCG-vs-prompt gap (ρ = −0.70 over 76 classes; −0.81 on the
34 EffectData effects; classes with base effect < 80: 16/17 have gap ≥ 10 pp). Video-only descriptors and CLIP
clause/clip alignment do not predict it (misc/2026-09-07_eval_grid_v2/eval/gap_predictors*.csv). So the screen IS the
prompt-only generation.

## Design (stage A only — stages B/C are separate decisions)
- 300 effects, chosen by `scripts/ed_screen/select_effects.py`: 150 exploit (kNN-predicted gap from BGE-large text
  embeddings fitted on the 34 measured effects), 140 explore (k-means medoids over the rest), 10 anchors (5 largest +
  5 smallest measured gaps, NEW subjects) — the anchors calibrate one-seed screen levels against the two-seed grid.
- Per effect: 1 reference clip, 3 SAME-content rows (3 other roster subjects as endpoints, frame-0 anchor), GT pool ≤ 8
  further clips. All clips are S6-roster (known portrait shape, captioned subjects) → no probing, no new captioning.
- Arm: `base_cond_effect_edscreen` (kind base, token stripped, prompt "{S1}. {clause}."), native 81 f,
  `GEN_FRAMES=81 GEN_PREFIX_FRAMES=1`, seed 42 only. 900 clips ≈ 7.5 GPU-h.
- Clauses: `scripts/grid_v3/build_ref_effects_v3.py` machinery (gemini-3.6-flash watches the reference std clip;
  append-only into misc/refvfx_baseline/reference_effects.json). S1 = captions/004 `<subject>|A`.
- Scoring: v4 with `--reference-corpus` (222 pin) and a screen SUPERSET corpus manifest (677 + screen clips) passed via
  `CORPUS=`; kernel ceilings for the 300 classes (`ceilings_screen.json`, used via `LADDER_CEILINGS_EXTRA`).
  Warm-up job extracts every screen clip's bundle once (no 16-shard race on cold clips).
- Readout: per-effect prompt-only level (pool-%), the distribution over 300, exploit vs explore hit rates
  (did the text prior work?), anchors vs their grid-v3 levels. Threshold for stage B: ~80 (grid-v3 calibrated) or a quantile.

## What this does NOT do
No DCG here (stage B), no held-out confirmation (stage C). Levels only; a class's low prompt-only score selects it for
stage B — the reportable gap comes from stage C rows on new subjects.

## Files
selection.json · embeddings.npz · media/ (manifests) · registry_base_cond_effect_edscreen.jsonl (eval_ladder) ·
corpus_manifest_screen.json · ceilings_screen.json · gen/ (ledger, logs) · eval/ (manifests, REPORT.md)
