# Held-out (zero-shot) class identifiability ceiling (C4)

Computed 2026-07-18 from the certified R2 trust map
(`outputs/eval/certification/3.0.0-draft.8/exam/trust_map.json`), BEFORE Stage-2 unblinding.
Per PRE_REGISTRATION.md #4: zero-shot claims are stratified by whether the certified kernel
can even identify the class; margin gains are expected to concentrate on identifiable classes.

| held-out class | camera? | m2b (margin) trust | appearance recall (m1a) | n clips | zero-shot scoring path |
|---|---|---|---|---|---|
| hero_flight | YES | True | 0.90 | 10 | cam_dtw + 2AFC (identifiable) |
| illustration_scene | YES | True | 0.80 | 10 | cam_dtw + 2AFC (identifiable) |
| raven_transition | YES | True | 1.00 | 4 | cam_dtw + 2AFC (identifiable) |
| gas_transformation | no | **False** | **0.20** | 10 | appearance margin (LOW identifiability) |
| hole_transition | YES | False | — | 2 | excluded (n too small) |
| seamless_transition | YES | False | — | 1 | excluded (n=1) |
| jump_transition | YES | False | — | 1 | excluded (n=1) |

## Consequence (governs the Stage-2 verdict reading)
- The APPEARANCE-margin zero-shot headline rests on **gas_transformation alone**, which is
  **low-identifiability** (recall 0.20, m2b untrusted). A certified appearance-margin zero-shot
  number is therefore near-meaningless / severely under-powered — DO NOT headline it.
- The genuine zero-shot evidence is:
  1. **cam_dtw** on the three identifiable camera classes (hero_flight, illustration_scene,
     raven — recall 0.8–1.0) — the metric CAN see these, just not via appearance margin.
  2. **2AFC human eval** (style-performance + content-preservation) on the identifiable
     held-out items — the load-bearing off-instrument confirmation.
- This is the identifiability ceiling: "escape collapse from one demo" is only well-posed
  where the demo identifies the class. For appearance that is essentially no held-out class;
  for camera-trajectory it is three; for human judgment it is the identifiable subset.
- Pre-registered prediction stands: per-item Δmargin (where scorable) correlates positively
  with identifiability. With appearance-scorable n≈1 class, this correlation is reported as
  descriptive only; the camera + 2AFC paths carry the zero-shot claim.

This reframes the paper's zero-shot story from "appearance-margin win" (unavailable) to
"camera-trajectory + human-judged transfer on identifiable unseen classes", which is honest
and defensible. The seen-class (ID) margin remains the primary quantitative headline.
