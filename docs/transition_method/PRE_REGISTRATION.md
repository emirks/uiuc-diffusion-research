# Transition-transfer method — PRE-REGISTRATION

**Committed 2026-07-18, BEFORE exp_067 (Stage 2) certified results are unblinded.**
Governs every stage (exp_067 Stage 2, exp_068 Stage 3a, exp_069 Stage 3b, exp_070 guidance)
and every fresh-idea experiment. Authority: the certified transition-eval harness v3.0.0
(frozen); headline claims come ONLY from its channels. Rationale and idea routing:
`$LAB/misc/advised_method_impl/DOSSIER.md` + `IDEAS_ROUND1.md`.

## Why this exists (the leak hazard)
Stage-2 step-2000 inline validation on the held-out class `hero_flight` shows the zero-shot
output beginning to IMPORT the reference clip's CONTENT (city skyline, fist-POV) rather than
performing its STYLE on the endpoints' own scene. The certified **margin** metric structurally
REWARDS this: pasting reference-class appearance into the output raises similarity to the
target class's real-clip pool even when no style is performed on the target content; and
`copy_max` (near-copy vs the specific reference, τ=0.858) only catches near-copies of that one
clip, not class-content import. Therefore a certified zero-shot margin gain could be a leak
artifact. This pre-registration exists so a leak cannot be counted as a style-transfer win.

## Held-out (zero-shot / tier-C) class audit — CAMERA-BLINDNESS
Holdout classes (verbatim ic3/exp_064): hero_flight, illustration_scene, gas_transformation,
raven_transition, hole_transition, seamless_transition, jump_transition.
Per corpus_manifest `tags`, **6 of 7 are camera-tagged** and M2b (margin) is documented blind
to camera classes:
- CAMERA (excluded from the appearance-margin headline; scored on cam_dtw + 2AFC):
  hero_flight, illustration_scene, raven_transition, hole_transition, seamless_transition,
  jump_transition.
- APPEARANCE-SCORABLE zero-shot class: **gas_transformation ONLY** (object-tagged).
Consequence: the appearance-margin zero-shot headline rests on ~1 class and is under-powered;
camera-class zero-shot performance is claimed via cam_dtw and the 2AFC human check, never via
appearance margin. This list is FIXED now from tags, not after results.

## Pre-registered claims and guards (verbatim, evaluator-adjudicated 2026-07-18)
1. **Tier-1:** certified zero-shot margin (held-out classes, paired per-item vs ic3 on
   identical items) improves by ≥ 0.037 (MDE), with seen-class ID margin within MDE,
   near_copy ≤ 10% (τ=0.858), max_seam_z within MDE.
2. **Tier-2:** zero-shot margin ≥ base·PE (0.175) − MDE (in-context adaptation stops being
   harmful zero-shot).
3. **Floor:** a certified-scored pixel-crossfade arm on the same tier-C items accompanies
   every headline table.
4. **Identifiability stratification:** per-item reference identifiability (leave-own-clip-out
   class retrieval under the M2b kernel, from the certification R2 trust map) is computed
   before unblinding; prediction: per-item Δmargin correlates positively with identifiability;
   headline margins additionally reported on the identifiable subset.
5. **Camera policy:** holdout classes with camera tags (the 6 listed above) are excluded from
   the appearance-margin headline and reported under cam_dtw (M2b is camera-blind); the
   excluded list is fixed now from tags, not after results.
6. **Leak guard:** every zero-shot margin claim is co-reported with copy_max-vs-reference and,
   once defined, the content-leak metric; a margin gain accompanied by a leak-metric rise
   beyond its ID-band is NOT claimed as style transfer.
7. **Guidance discipline:** no intruder-pool-aware negative guidance in any arm, ever (direct
   metric gaming; poisons the margin channel's credibility).
8. **2AFC:** ~20 tier-C pairs, 3–5 raters, two fixed questions per pair — "which performs the
   reference's transition style?" AND "which better preserves the endpoint scenes' content?";
   any disagreement with margin on the winning arm is reported.

## Content-leak metric (to be defined by the disentanglement fable; guard #6 depends on it)
Experiment-side, from the existing DINO / copy_max machinery: must separate "performs style on
the endpoint content" from "imports reference-clip content", computable on existing tier-C
generations, and reportable alongside every zero-shot claim. Until defined, guard #6 uses
copy_max-vs-reference + the interim ID-vs-tier-C copy_max stratification as the leak proxy.
