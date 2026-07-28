# Endpoint expansion — tracked mirror

`data/processed/` is gitignored (this repo is PUBLIC and holds no media), so the expansion
pipeline runs out of `data/processed/ctt_v2_strata/endpoints_v2/_work/` where its outputs sit
next to the clips. That directory is the LIVE copy. This directory is the tracked record:

* `code/` — the pipeline, byte-identical to what ran. `collect_v2.py` (candidate list),
  `build_v2.py` (QC cascade + std121 standardisation; a path-repointed copy of the round-1
  `build.py`), `tighten_v2.py` (aesthetic floor, letterbox guard, tightening policy, both
  similarity guards, review sheets, finalize), `aggregate_verdicts.py` (merge the visual
  verdicts, assert full coverage, cross-check "static" calls against measured motion).
  `detect_v2.py` is NOT copied: it is the round-1 `detect.py` with three path constants
  changed, and copying it would duplicate 280 lines to record a three-line diff.
* `registries/` — the decision records: the frozen train/eval split, the letterbox audit of the
  round-1 bank, the review queue, every reviewer's raw verdicts, the static adjudication, the
  motion floor, and the resulting additive `bank_tightened_v2.json`.

The round-1 bank `data/processed/synth_endpoints/` is READ-ONLY throughout; nothing here mutates
it. New endpoints are delivered as an additive v2 bank with new clip ids.
