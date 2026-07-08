# exp_057 — design: broad unseen-class eval of the exp_056 IC-LoRA

**Question.** exp_056 probed unseen-class in-context transfer with n=1 class
(jump) × 4 items — an anecdote. With the user-labeled corpus
(`data/processed/transitions/{onesided,twosided}_transitions/`, labels encoded
in dir names as `<sidedness>_<tags>_<class>`), test the *same frozen adapter*
(step 3000, no retraining) on many unseen classes, stratified by:

- **taxonomy** — object (new object forms), camera (camera move carries the
  transition), style (same object, new rendering); user's own labels, verified
  by frame-montage inspection (montages archived in scratchpad session dir);
- **structure** — one-sided (subject transforms/vanishes in place; training
  corpus was 100% two-sided scene A→scene B) vs two-sided;
- **texture familiarity** — "cousins" of trained classes (shadow↔shadow_smoke,
  fire_element↔firelava) vs genuinely novel textures (wireframe, illustration,
  portal…). Prediction: cousins transfer appearance more easily — if so,
  "unseen-class transfer" claims must control for texture overlap.

## Corpus filtering (from inventory.py / dedup_report.md)

339 labeled clips, 0 exact-md5 dups, 0 undecodable. Excluded:

- 17 clips < 121 frames (all of `mouth_in` bar 3, and 4 of `eyes_in`) — those
  two classes are effectively unusable and are not selected;
- `giant_grab_5` = near-dup of `giant_grab_0` (aHash d=5/192, same take);
- `super_fast_run_11` (same-take regen of `_1`) and `plasma_explosion_3`
  (same-take regen of `_0`) — montage-identified; kept OUT of the eval corpus
  so they don't inflate LOO appearance ceilings;
- `sakura_petals_0/1`, `wonderland_0/1` near-dup pairs — classes not selected;
- low-res (320 px) classes (`disintegration`, `head_explosion`, `set_on_fire`,
  `turning_metal*`, `eyes_in`, `earth_zoom_out`, `thunder_god`) — 2.6×
  upscale would confound appearance metrics; vanish-type coverage comes from
  high-res `gas_transformation` / `portal` / `giant_grab` instead.

## Selected classes (14; n = usable clips in eval corpus)

| class | tags | n | notes |
|---|---|---|---|
| hero_flight | camera | 10 | ground→aerial→POV arc |
| super_fast_run | camera | 12 | speed-blur sprint at camera |
| plasma_explosion | camera | 4 | landscape src (crop loses width); beam+blast |
| shadow | style (cousin: shadow_smoke) | 15 | black shroud engulfs subject |
| fire_element | style (cousin: firelava) | 4 | lava-veined figure |
| wireframe | style novel | 9 | neon wireframe render, bg→black |
| illustration_scene | style novel | 10 | photo→pop-art/line-art |
| animalization | object | 8 | person→animal, same scene |
| gas_transformation | object VANISH | 10 | person→vapor→empty scene |
| portal | object VANISH | 13 | green portal swallows subject (4 cartoon-domain clips kept in corpus — ceiling caveat) |
| giant_grab | object vanish-ish | 5 | giant hand removes subject |
| money_rain | object degenerate-endpoint | 7 | bills rain; endpoints nearly identical → floor≈clip, flag |
| hole_transition | two-sided unseen (object+camera) | 2 | camera through donut hole → scene B |
| seamless_transition | two-sided unseen (camera) | 1 | wall-wipe pan reveal; singleton → raw-only |

Corpus additions standardized like exp_056 (480×640×121@24, even-index frame
spread; 131–161f sources play 1.08–1.33× faster — timing metrics caveat) into
`data/processed/transitions_std121/<class>/`, extending the harness reference
corpus (trained 11 classes stay for anchors/leak "others").

## Arms (51 generations)

| arm | n | endpoints | reference | tests |
|---|---|---|---|---|
| ic_os_inclass | 23 | one-sided clip X | same class, clip Y≠X (12 classes ×2, money_rain ×1) | unseen class + structure-OOD targets |
| ic_os_to2s | 12 | trained two-sided (ew0/melt1/wb0 rotation) | one-sided unseen class (×1 each) | unseen effect on foreign two-sided endpoints — comparable to exp_056 ic_cross |
| ic_ts_unseen | 3 | hole_0 in-class; ew0 × hole ref; ew0 × seamless ref | two-sided unseen refs | near-distribution unseen (jump analogue) |
| ic_anchor | 2 | ew0←ss1, melt1←ew1 (exact exp_056 items) | trained classes | run-to-run link to exp_056 |
| base_* twins | 11 | 6 os_inclass, 3 os_to2s, 1 ts_unseen, 1 anchor | same as twins | copy-vs-transfer control |

Same inference recipe as exp_056 quads: seed 42, 480×640×121@24, 30 steps,
CFG 4.0, STG 1.0 stg_v blocks [29], prefix 9f / suffix 8f cond cuts,
type-blind captions + ICTRANS, reference = full standardized clip.

## Pre-registered metric-validity caveats (to be checked, not assumed)

1. **Lerp floor, one-sided classes**: endpoints share the background, so the
   lerp control is a plausible-ish in-place fade → floors HIGH, floor–ceiling
   gap small → normalized scores unstable. Report gap per class; mark
   norm-unreliable where gap is small. money_rain is the extreme (endpoints
   nearly identical).
2. **Leak (M6), in-class arms**: reference shares class appearance —
   leak_max_sim inflated for legitimate reasons (exp_056 saw 0.77–0.81
   in-class); only cross/`os_to2s` leak cleanly separates copy from transfer.
3. **Camera classes**: "effect appearance" (M3) is ill-defined — the look IS
   the scene; M2 motion fidelity + M1 profile carry the signal. Expect M3 to
   be noisy/meaningless there.
4. **Ceiling trust**: hole (n=2) < min_ceiling_clips=4 → †-flagged;
   seamless (n=1) raw-only (singleton guard); fire_element/plasma (n=4) weak.
   Portal ceiling mixes cartoon+live-action domains.
5. **Anchors**: corpus root moves from raw `transitions/` (exp_056 run_0002)
   to `transitions_std121/` — ceilings shift slightly; anchors are
   qualitative links, not exact reproductions.
6. Motion-trust flags come from the exp_054 validation exam which never saw
   the new classes → motion_trusted=False (†) everywhere new; correct and
   expected — report raw + normalized with that flag visible.
