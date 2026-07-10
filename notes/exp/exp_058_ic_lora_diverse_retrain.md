# exp_058 — diversified mixed-conditioning IC-LoRA retrain (v2)

**Status: COMPLETE 2026-07-08.** Fresh IC-LoRA trained on 460 pairs / 32
classes with per-pair conditioning (one-sided prefix-only, two-sided
prefix+suffix); evaluated by rerunning the exp_057 40-IC-quad suite verbatim
plus new arms. Scoring `outputs/eval/exp_058/quads/run_0001` (+
`analysis_v1v2.md`); viewer `outputs/eval/exp_058/viewer` (PASS).
Pre-registration: `experiments/exp_058_ic_lora_diverse_retrain/design.md`.

## 1. What changed vs exp_056 (v1)

| | v1 (exp_056) | v2 (exp_058) |
|---|---|---|
| pairs / classes | 131 / 10 (all two-sided) | 460 / 32 (116 two-sided + 344 one-sided) |
| conditioning | global prefix(2)+suffix(1) | per-pair mask: [2 start] or [2 start + 1 end] latent frames |
| steps (epochs) | 3000 (~23) | 5000 (~11); all ckpts kept incl. 3000 |
| held out | jump only | hero_flight, illustration_scene, gas_transformation, raven, hole, seamless, jump + quad clips of big trained classes |

Mask mechanism validated bit-exact against v1's prefix+suffix
(`test_mask_conditioning.py`); training H200, 5000 steps, loss →0.18.
Captions: 185 type-blind endpoint captions (126 Gemini-generated from stills
— first API captioning in the project — 2 leak violations hand-fixed).

## 2. Headline: paired v1→v2 on 40 identical suite items

raw app **0.509→0.555 (+0.046, 24/40)**, leak 0.655→0.666 (+0.011),
prefix/suffix DINO unchanged (0.985/0.96), **max seam z 0.12→0.74 (worse,
only 10/40 improved)**. Read jointly: appearance up ~4× more than leak →
mostly genuine transfer gain, endpoint *identity* anchoring intact, but
endpoint *seam smoothness* regressed (see §5 — it's a conditioning-mode
mismatch, not uniform decay).

## 3. Held-out classes (the real test)

- **gas_transformation (vanish, appearance-blind — cleanest probe):
  IMPROVED.** In-class raw 0.411→0.608 and 0.641→0.670 (leak +0.06/+0.02,
  far sub-copy); cross-target 0.358→0.421. Pre-registered expectation (e):
  genuine in-context gain, cannot be texture memorization (class never
  trained, endpoints don't reveal appearance).
- **illustration_scene (novel texture): improved, partly leak-confounded.**
  In-class 0.592→0.827 / 0.713→0.825 but leak 0.700→0.866 on the first —
  filmstrips show stronger comic-panel rendering AND closer tracking of the
  demo's layout. Cross-target (the honest number) 0.331→0.418 at leak
  0.401→0.471: novel-texture transfer moved ~+0.09 raw — the exp_057
  gradient (cousin 0.45 vs novel 0.30) roughly CLOSED to ~0.42, consistent
  with coverage-limited texture transfer (wireframe/polygon in training).
- **hero_flight (camera): NOT improved.** In-class mixed (0.268→0.432,
  0.431→0.367); cross-target flat (0.207→0.199). Supports the pre-registered
  conditioning-conflict hypothesis: camera arcs fail cross-target because
  pinned foreign endpoints contradict the arc, and no amount of camera
  training data fixes that. (8 camera-tagged classes were in training.)
- **Two-sided unseen (hole/seamless): flat** (0.335→0.324, 0.134→0.108) —
  small-n structure transfer unchanged.
- **raven_transition (trained in v1, REMOVED in v2):** v2 in-class raw
  0.346/0.268 at leak ~0.57, seams negative, endpoints 0.97 — semantics
  execute in-context; base twin copies (0.689 raw at leak 0.995, seam +2.0).
  v1 as a *trained* class scored ~0.5-0.7 raw in-class in exp_056 —
  per-class training still buys appearance fidelity (expected outcome (g):
  cost-of-removal is real but structure survives).

## 4. New capability: prefix-only generation (no end frame)

8 IC items, 2 base twins. The v2 adapter executes transitions from start
frames + demo alone: raw 0.495±0.13, prefix DINO 1.000, seams sane. Base
twins on identical inputs: raw 0.776 **at leak 0.992 and seam z +117** —
verbatim reference replay with a hard cut. Two caveats, stated plainly:
(1) prefix-only leak (0.729±0.12) runs above suffix-anchored leak (0.666) —
gas_7 hits 0.913, portal 0.869; without an end anchor the generation tracks
the demo more closely. In-class leak is partly legitimate (exp_057 finding),
and 0.73 ≪ 0.99, but the pinned suffix evidently acted as an anti-copy
constraint. Cross-class prefix-only is the untested cell for v3 eval.
(2) The caption still names the intended end state (training format) — text
intent remains even without pixels.

## 5. The seam regression is a conditioning-mode mismatch

Worst suite seam deltas are ALL one-sided in-class items scored WITH suffix
conditioning (illustration +4.9, plasma +3.5, portal +2.4/+1.8, shadow +1.5)
— exactly the classes v2 trained prefix-only. v2 learned to approach
one-sided endings freely; a pinned suffix now conflicts and the handoff
snaps. Evidence it's not general two-sided decay: the exp_056 anchors keep
seams (−0.57, −0.53 vs v1 −0.57, −0.78) and suffix DINO (0.988/0.989 ==
v1). The anchors DO show raw app −0.058/−0.099 with leak −0.10/−0.06 —
v2 is more conservative on two-sided cross-class transfer (app-per-leak
ratio unchanged: 0.69→0.71). Pre-registered tolerance (±0.05) is exceeded on
one anchor → the designated follow-up is the two-sided rebalance/oversample
variant, not a redesign.

Corollary that applies to §2 as a whole, not just seams: the suite's
one-sided items were generated AND scored WITH suffix conditioning — required
for the paired v1↔v2 comparison on identical inputs, but for v2 that is an
off-training-mode operating point (those classes trained prefix-only). Suite
deltas on one-sided classes are therefore a conservative LOWER BOUND on v2's
native capability; the v2-native (prefix-only) operating point is measured
only by the 8 ic2_prefixonly items, all in-class. A full prefix-only rerun of
the one-sided suite is the cheap missing measurement (v3 candidate 2b).

## 6. In-context dominance (quiet but important)

Classes that moved INTO training with their eval clips EXCLUDED barely
moved: raw 0.565→0.586, leak 0.708→0.689 (n=18). Small classes whose eval
clips WERE in training: 0.559→0.628 with leak +0.05 (mild memorization
bump). Training a class helps its transfer numbers far less than having the
demo in context — the mechanism is reference-reading, not class recall.

## 7. Metric discipline (carried from exp_057, still binding)

Normalized appearance remains unreadable for one-sided styles (floor≥ceiling
7/16, same corpus) — everything above is raw × leak + paired deltas on
identical inputs + endpoint/seam. Prefix-only items: suffix metrics are
structurally N/A (n_endpoints=1 in the manifest; suffix DINO would measure
"landed near the true ending", advisory only). money_rain stays a degenerate
control. The Gemini video judge was NOT run this round (uses API quota;
metrics + viewer carry the conclusions — run `run_judge_gemini.py` on
run_0001 if wanted).

## 8. Honest summary

The diversified mixed-conditioning retrain did what coverage can do and not
what it can't: genuine gains on the appearance-blind vanish probe and on
novel-texture transfer (the exp_057 texture gradient largely closed), a new
prefix-only generation capability with real but bounded demo-tracking, flat
camera cross-target (mechanism-limited, as pre-registered), and a measurable
two-sided cost — conservative cross-class transfer on the anchors and seam
snaps precisely where suffix conditioning contradicts prefix-only training.
Nothing here contradicts the exp_057 core claim: the reference video, not
the class label, carries the transition.

## 9. Artifacts

- Adapter: `outputs/training/exp_058_ic_lora_diverse_retrain/ic2/checkpoints/`
  (500..5000; primary = 5000, budget-matched = 3000 unevaluated).
- Scores: `outputs/eval/exp_058/quads/run_0001/` (items.jsonl, report.md,
  analysis_v1v2.md, ceilings.json, scatter.png).
- Viewer: `outputs/eval/exp_058/viewer` (serve `outputs/eval/` → 056/057/058).
- Data: `experiments/exp_058_ic_lora_diverse_retrain/dataset/` (captions ×185
  + provenance, pairs.json ×460, quads_v2.json ×53, manifest_ic_v2.json).
- Jobs: sanity 9400364 (after 9399580 link-env fix), train 9401247 (+9401248
  insurance, no-op), infer 9405052-58, eval 9405337. W&B `exp058_ic2`.

## Open questions → v3 candidates

1. Two-sided rebalance (oversample two-sided pairs or lower one-sided share)
   — does it recover the anchor raw app without losing one-sided gains?
2. Cross-class prefix-only (unseen ref on foreign start frames) — the
   untested cell of the new capability. 2b. Prefix-only rerun of the 35
   one-sided suite items (v2-native operating point; suite numbers are a
   mismatch lower bound, see §5).
3. Suffix-optional training for one-sided classes (mask duplicate rows) to
   remove the §5 conditioning-mode mismatch.
4. Step-3000 checkpoint eval (budget-matched comparison v1@3000 vs v2@3000).
5. Camera-arc: endpoints that PERMIT the arc (portrait start → aerial end).
