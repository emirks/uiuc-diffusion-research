# exp_058 design — diversified mixed-conditioning IC-LoRA retrain (pre-registration)

Written 2026-07-08, BEFORE training. Motivated by exp_057: the exp_056 adapter
(131 pairs, 10 two-sided classes) transfers transitions in-context but (a)
novel-texture appearance is ~2/3 of cousin-texture (0.30 vs 0.45 raw
cross-target), (b) camera-arc cross-target is weakest (0.30), (c) one-sided
renderings drift toward trained textures. All three are exactly what a
texture/taxonomy/structure-diverse corpus should address — IF the effects are
capacity/coverage-limited rather than conditioning-limited (camera may be the
latter; pre-registered as uncertain).

## 1. Question

Does retraining the IC-LoRA on a diversified corpus — 23 one-sided classes
(prefix-only conditioning) + 9 two-sided classes (prefix+suffix), 460 pairs —
(i) improve unseen-class / novel-texture / camera transfer, (ii) enable
**prefix-only one-sided generation** (no end-frame given), and (iii) hold the
exp_056 two-sided anchors without regression?

## 2. Conditioning scheme (user-specified, validated)

- **Two-sided pairs**: exactly exp_056 — prefix 2 latent frames (first 9 px
  frames) + suffix 1 latent frame (last 8 px frames; inference feeds last 9 px
  → 2 latents, keeps the final one per the suffix pre-cut rule).
- **One-sided pairs**: prefix 2 latent frames ONLY; the model must generate
  the ending (in-place transform / vanish) from reference + caption.
- **Mechanism**: single `MaskConditionConfig` (p=1.0) with per-pair masks
  `masks/K.pt = {"mask": [16,15,20]}` — frames {0,1} always 1, frame {15} 1
  iff two-sided. `test_mask_conditioning.py` proves bit-exact equivalence
  with exp_056's Prefix(2)+Suffix(1) on two-sided masks and Prefix(2)-only on
  one-sided masks, composed with the reference concat (ALL PASS, 2026-07-08).
- Reference = full 121f clip latents, concatenated before target (unchanged).
- Fresh training from base (NOT continued from exp_056 step 3000): clean
  attribution of corpus effect; step-matched checkpoint (3000) kept for a fair
  same-budget comparison alongside the final (5000).

## 3. Corpus allocation

Filters carried from exp_057 (dedup_report.md): 0 exact dups; near-dup/
same-take exclusions giant_grab_5, super_fast_run_11, plasma_explosion_3,
sakura_petals_1, wonderland_1; ALL 320px-source classes excluded (upscale-blur
supervision + metric depression) — that drops eyes_in, mouth_in (also mostly
<121f), disintegration, head_explosion, set_on_fire, turning_metal×3,
earth_zoom_out, thunder_god. run_set_on_fire contributes only its 2 high-res
clips. Temporal standard stays 480×640×121@24: mixed-length training is
mechanically possible at bs=1 but the trainer's reference position scale
factor is inferred globally from one pair (flexible.py) so mixed ref:target
ratios silently mis-position references; native lengths would rescue zero
clips (all short clips are 320px) and only ~29/227 clips are >145f (≤2×
retime, precedented in exp_056's own corpus). Length-bucket variant deferred.

### Held out (never trained) — the eval suite, stratified

| class | n | stratum | why |
|---|---|---|---|
| hero_flight | 10 | camera, one-sided | best camera class; exp_057 quads reusable |
| illustration_scene | 10 | style, NOVEL texture | the novel-texture probe (wireframe moves to training) |
| gas_transformation | 10 | object/vanish, one-sided | purest in-context probe (empty end, appearance-blind) |
| raven_transition | 4(3 hi-res) | two-sided, object+camera | healthy-normalization unseen two-sided; was TRAINED in exp_056 → old-vs-new = cost-of-removal too. Caveat: animalization (trained) is a semantic cousin |
| hole, seamless, jump | 2/1/1 | two-sided unseen | too small to train; exp_057 items reusable |

Additionally, for training classes with n≥7, the exact clips used by exp_057
quads (endpoints AND references) are EXCLUDED from training (shadow −4,
super_fast_run −4, portal −4, wireframe −4, animalization −4, money_rain −2,
shadow_smoke −1) so reused eval items keep unseen endpoints/demos. Small
trained classes (fire_element, plasma_explosion, giant_grab, flame, …) keep
their quad clips in training — those eval items are re-labeled
in-distribution-trained in the analysis.

### Training set — 162 clips, 460 pairs, 32 classes

- **Two-sided (prefix+suffix), 42 clips / 116 pairs**: air_bending 4,
  display 3, earth_wave 5, firelava 6, flame 2, flying_cam 4, melt 4,
  shadow_smoke 9, water_bending 4 (exp_056 minus raven minus 1 quad clip).
- **One-sided (prefix-only), 120 clips / 344 pairs**: exp_057-standardized:
  animalization 4, fire_element 4, giant_grab 5, money_rain 5,
  plasma_explosion 4, portal 9, shadow 11, super_fast_run 8, wireframe 5.
  NEW (to standardize): color_rain 8, cotton_cloud 6, earth_element 6,
  live_concert 8, luminous_gaze 4, monstrosity 3, mystification 5,
  nature_bloom 2, polygon 9, saint_glow 4, sakura_petals 2, water_element 5,
  wonderland 2, run_set_on_fire 2.
- Taxonomy coverage in training: style 9 classes, object 17, camera-tagged 8.
  Texture families now include wireframe/polygon (grid), illustration-adjacent
  (wonderland, polygon), particulate (sakura, color_rain, money_rain, cotton).
- Pairs: circulant within-class, target i ← refs i+1..i+min(3,n−1) mod n
  (n=2 classes → 2 pairs). No cross-class pairs (no ground-truth target).
- Sidedness ratio 25% two-sided / 75% one-sided accepted WITHOUT rebalancing
  (one change at a time); anchor eval directly measures two-sided regression,
  and a rebalance variant is the designated follow-up if anchors regress.

## 4. Captions

Type-blind, exp_056 format: "<scene A>. The scene transforms into <scene B>."
(+ ICTRANS trigger prepended at pair build). ~42 exp_056 captions reused;
~120 new ones generated by Gemini from the standardized first/last frames
(sees ONLY the two stills — mechanism words cannot leak from motion), then
spot-checked (≥20 manual) with a banned-word scrub (class names, effect
mechanism nouns). Pre-registered caveat (from exp_057): a one-sided clip's
last frame reveals the effect's terminal appearance; unavoidable, uniform
across arms, and vanish classes remain the appearance-blind probes.

## 5. Training config

exp_056 recipe otherwise unchanged: LTX-2 19B dev, rank 32/α32 video attn+FFN,
lr 2e-4, bf16, grad ckpt, batch 1, seed 42, 480×640×121@24 (9600-token pairs,
2.62 s/step, 49.5 GB peak → H100/H200 OK). **5000 steps** (≈10.9 epochs of
460 pairs; exp_056 was ~23 epochs of 131), checkpoints every 500, keep all.
Validation every 1000: 1 two-sided in-class, 1 one-sided in-class
(prefix-only), 1 held-out ref (hero_flight → prefix-only).

## 6. Eval plan (after training; scored with the exp_052/053 harness)

1. **Reuse the exp_057 51-quad suite verbatim** (same endpoints/refs/cuts/
   captions/seed): per-quad triplets base / IC-v1 (exp_056@3000) / IC-v2.
   Held-out classes stay unseen probes; trained-class items re-labeled
   in-distribution (split trained-clips vs held-out-clips).
2. **New arms**: (a) raven in-class ×2 + base twin (healthy-normalization
   unseen two-sided); (b) **prefix-only one-sided**: held-out classes
   hero_flight/illustration/gas ×2 each + 2 trained-class items, no suffix
   condition — THE new-capability test; (c) v2 anchors = exp_056 ic_cross
   items (two-sided regression check).
3. **Metric discipline (pre-registered, from exp_057)**: normalized
   appearance is unreliable for one-sided styles (floor≥ceiling 7/16) — 
   conclusions ride raw appearance × leak, base-twin/v1-twin deltas, endpoint
   DINO + seam z, Gemini judge. Prefix-only items: suffix-endpoint metrics
   are N/A by construction; report prefix DINO + M1/M2/M3/M6 only. money_rain
   stays a designed degenerate control.

## 7. Expected outcomes (pre-registered)

(a) Two-sided anchors within ~±0.05 raw of exp_056 (else: rebalance variant).
(b) Novel-texture held-out (illustration_scene) raw cross-target appearance
    improves over v1's 0.30 if the texture gradient is coverage-limited;
    unchanged if it's a capacity/mechanism limit.
(c) Camera held-out (hero_flight): in-class stays good; cross-target may NOT
    improve (conditioning-conflict hypothesis) — either outcome informative.
(d) Prefix-only one-sided works on trained classes (it's the training task);
    on held-out classes it should execute structure (in-place/vanish) with
    appearance quality between v1's suffix-given results and base.
(e) gas_transformation (never trained, appearance-blind): the cleanest
    v1-vs-v2 delta; improvement here = genuine in-context gain, not texture
    memorization.
(f) Base twins keep copying (adapter-independent claim from exp_057).
(g) raven: v2-unseen ≈ v1-trained would be a striking in-context result;
    v2 markedly worse = per-class training still matters (expected).

## 8. Risks / gotchas carried forward

- sbatch cd's into trainer → LoRA checkpoint paths ABSOLUTE (ops-gotcha #5).
- `data.preprocessed_data_root` must exist at config parse (pre-create).
- Per-chunk inference output dirs (ValidationRunner name collisions).
- process_dataset names outputs after the row's target → preprocess unique
  clips once, symlink pair tree (latents/conditions/reference_latents/masks
  share rel-name K).
- Scale-factor auto-inference: ref and target identical dims → factor 1.
- Cluster `secondary` = account campusclusterusers, gres gpu:H100:1;
  HCESC-H200-secondary for training with --requeue + load_checkpoint resume.
