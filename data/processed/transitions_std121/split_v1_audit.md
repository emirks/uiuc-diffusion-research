# split v1 — near-duplicate audit report (2026-07-14)

## What was scanned

For every class with test items in the frozen split v1
(`data/processed/transitions_std121/split_v1.json`, rule in its provenance
block), the **M2a copy score** was computed between every cross-boundary
(train, test) clip pair within the class: max cosine similarity between the
two clips' per-frame DINOv2-base features (ALL frames on both sides), using
the certified machinery
(`diffusion.transition_eval.m2_integrity.copy_score` + `features.py` from the
certified worktree `.claude/worktrees/eval-v3-spec`, feature cache
`outputs/eval/split_v1_audit/dino_cache` — separate from the shared harness
cache). Flag threshold **copy_max >= 0.858** = the certified calibrated
tau_copy (certification record `v3.0.0-amendment-1`, which pre-registers this
scan at that value).

- Scanner: `scripts/audit_split_v1.py`, Slurm job **9495508**
  (HCESC-H100-normal, 3:55 elapsed), log
  `outputs/logs/slurm/split_v1_audit-9495508.out`.
- **326 pairs scored over 5 iterations** (260 in the initial split; 66 more
  covering replacement candidates as remediation was applied).
- Full per-pair scores + argmax frame provenance:
  `outputs/eval/split_v1_audit/audit_results.json`. Flag set consumed by the
  split builder: `data/processed/transitions_std121/split_v1_flagged.json`.

## Pairs flagged (44 flagged pairs, 12 flagged test picks, 4 classes)

Pre-registered remediation, applied mechanically: a flagged test clip is
replaced by the next deterministic candidate (same RNG stream, next sample);
if all candidates in a class flag, the class goes all-train.

| class | flagged test pick | worst pair (train, copy_max) | remediation |
|---|---|---|---|
| animalization | animalization_5 (iter 1) | animalization_7, 0.8897 | replaced -> animalization_7 |
| animalization | animalization_7 (iter 2) | animalization_5, 0.8897 | replaced -> animalization_0 (clean; _5/_7 are a mutual near-dup pair, both now train) |
| illustration_scene | illustration_scene_6 (iter 1) | illustration_scene_0, **0.9975** (near-identical take) | replaced -> illustration_scene_7 (clean) |
| polygon | polygon_3 (iter 1) | polygon_0, **0.9976** (near-identical take) | replaced -> polygon_4 (clean) |
| live_concert | live_concert_0, _3 (iter 1); _1, _2 (iter 2); _4, _5 (iter 3); _6, _7 (iter 4) | every cross pair 0.8633–0.9965 | **all 8 candidates flag -> class goes all-train (zero test)** |

live_concert is a wall of mutual near-duplicates (all 28 within-class pairs
that were scored sit at or above ~0.86; the class appears to be regenerations
of one concert scene) — no honest held-out clip exists, so per-class test
claims are impossible for it. All other scored pairs in all other classes
passed below threshold; the scan converged at iteration 5 with zero new flags.

## Final split (post-remediation)

**184 train / 39 test — overall test fraction 17.5%** (of 223 clips).
28 classes have test items; 11 are all-train (10 with n < 4 by rule +
live_concert by remediation).

| class | n | n_train | n_test | test clips |
|---|---|---|---|---|
| air_bending | 4 | 3 | 1 | air_bending_1 |
| animalization | 8 | 6 | 2 | animalization_0, animalization_2 |
| color_rain | 8 | 6 | 2 | color_rain_0, color_rain_2 |
| cotton_cloud | 6 | 5 | 1 | cotton_cloud_0 |
| display_transition | 3 | 3 | 0 | — |
| earth_element | 6 | 5 | 1 | earth_element_4 |
| earth_wave | 5 | 4 | 1 | earth_wave_3 |
| fire_element | 4 | 3 | 1 | fire_element_2 |
| firelava | 6 | 5 | 1 | firelava_4 |
| flame | 2 | 2 | 0 | — |
| flying_cam_transition | 4 | 3 | 1 | flying_cam_transition_4 |
| gas_transformation | 10 | 8 | 2 | gas_transformation_6, gas_transformation_7 |
| giant_grab | 5 | 4 | 1 | giant_grab_3 |
| hero_flight | 10 | 8 | 2 | hero_flight_5, hero_flight_6 |
| hole_transition | 2 | 2 | 0 | — |
| illustration_scene | 10 | 8 | 2 | illustration_scene_7, illustration_scene_4 |
| jump_transition | 1 | 1 | 0 | — |
| live_concert | 8 | 8 | 0 | — (remediated all-train) |
| luminous_gaze | 4 | 3 | 1 | luminous_gaze_3 |
| melt_transition | 4 | 3 | 1 | melt_transition_2 |
| money_rain | 7 | 6 | 1 | money_rain_3 |
| monstrosity | 3 | 3 | 0 | — |
| mystification | 5 | 4 | 1 | mystification_0 |
| nature_bloom | 2 | 2 | 0 | — |
| plasma_explosion | 4 | 3 | 1 | plasma_explosion_2 |
| polygon | 9 | 7 | 2 | polygon_4, polygon_8 |
| portal | 13 | 11 | 2 | portal_1, portal_4 |
| raven_transition | 4 | 3 | 1 | raven_transition_2 |
| run_set_on_fire | 2 | 2 | 0 | — |
| saint_glow | 4 | 3 | 1 | saint_glow_3 |
| sakura_petals | 2 | 2 | 0 | — |
| seamless_transition | 1 | 1 | 0 | — |
| shadow | 15 | 13 | 2 | shadow_10, shadow_3 |
| shadow_smoke | 10 | 8 | 2 | shadow_smoke_7, shadow_smoke_0 |
| super_fast_run | 12 | 10 | 2 | super_fast_run_0, super_fast_run_3 |
| water_bending | 4 | 3 | 1 | water_bending_3 |
| water_element | 5 | 4 | 1 | water_element_4 |
| wireframe | 9 | 7 | 2 | wireframe_0, wireframe_7 |
| wonderland | 2 | 2 | 0 | — |

Reproducibility: `python scripts/build_split_v1.py` (reads
`corpus_manifest.json`, sha256 `e7c867a6a4269679…` recorded in the split's
provenance block, plus `split_v1_flagged.json`) regenerates
`split_v1.json` byte-identically; the audit scanner regenerates the flag set
from scratch. The choice of test clips never used metric scores — only the
seeded RNG rule plus the pre-registered near-duplicate remediation above.
