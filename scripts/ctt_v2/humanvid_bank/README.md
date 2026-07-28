# HumanVid endpoint-bank screening (ctt_v2)

## Question
Which of the 19,197 downloaded HumanVid-real (Pexels) clips survive the endpoint-bank
QC contract, so they can serve as S2 procedural-transition endpoint contents?

## Setup
Same cascade, detectors and thresholds as the blessed `synth_endpoints` bank
(text/EAST → subject/FasterRCNN+saliency → fit → cut → dedup), with three deltas:
- **center window**: clips are long (median 13.8 s) → cut a ~5.04 s center window at
  source fps and resample 121 frames inside it (tempo ≈ native; no whole-clip fast-forward);
- **tightened pre-filter** (subject area ≥0.15, score ≥0.7) + CLIP farthest-point
  diversity cap (default 1500) BEFORE the expensive standardize/encode;
- **dedup against the existing 227-clip bank first** (vcbench is also Pexels-sourced).

License: Pexels ToS ML-restriction documented in `notes/dataset/humanvid_real.md`;
**owner cleared use 2026-07-27** (dossier §11). Recorded per clip and in the ledger.

## How to run
```
sbatch scripts/ctt_v2/humanvid_bank/job_screen.sbatch
```
Idempotent: detections are cached per clip; requeue resumes. `build.py --cap N` re-runs
selection/encode from cached detections at any cap without re-detecting.

## Outputs
`data/processed/humanvid_bank/` (main tree, gitignored data): `clips/*.mp4` (480×640·121f·24fps),
`manifest.jsonl`, `embeddings.npy` + `embed_ids.json`, `license_ledger.json`,
`bank_sample_sheet.png`, `_work/{candidates,detections,qc_log}.jsonl`,
`_work/build_report.json`.
