# exp_058 — training/eval allocation summary (one-page reference)

Written post-hoc (2026-07-09) from the actual artifacts — `dataset_exp058.json`
(162 std121 clips), `pairs.json` (460), `quads_v2.json` (53) — not from the
pre-registration. Complements `design.md` (the pre-registered plan) and
`notes/exp/exp_058_ic_lora_diverse_retrain.md` (results).

## Training recipe in one paragraph

460 same-class pairs (reference clip ≠ target clip, circulant pairing: every
clip is a target against every other clip of its class as reference; classes
with only 2 clips give 2 pairs). Each sample = reference latents concatenated
before the target (clean, timestep 0, loss-excluded) + the target
self-conditioning on its OWN endpoints via a per-pair mask: first 2 latent
frames always, plus its own last latent frame iff the class is two-sided.
No third clip is involved. Per optimizer step (batch 1) a single
flow-matching timestep is drawn (`shifted_logit_normal`) for all trainable
target tokens. 5000 steps ≈ 10.9 epochs. Rank 32/α32 attn+FFN, lr 2e-4,
480×640×121@24, trigger ICTRANS, type-blind captions.

Totals: **32 classes / 162 clips / 460 pairs = 116 two-sided + 344 one-sided
(25%/75%)**.

## The three "unseen" tiers (and where they live in the viewer)

| tier | meaning | classes | eval items (ref role) |
|---|---|---|---|
| A — HELD-OUT | class never in training, no clip in any role | gas_transformation, hero_flight, illustration_scene, raven_transition, hole_transition, seamless_transition, jump_transition (no items this round) | 23 |
| B — trained, eval clips EXCLUDED | class trained, but the exact clips used by eval quads (both endpoint and reference role) removed from training — verified 0-leak | shadow, super_fast_run, portal, wireframe, animalization (−4 each), money_rain (−2), shadow_smoke (−1) | 20 |
| C — trained, eval clips SEEN | small classes; 6 in-class items are EXACT training pairs (same target+reference row) | fire_element, plasma_explosion, giant_grab (+ earth_wave/shadow_smoke anchors, melt/water_bending endpoints) | 10 |

**Viewer note:** arms keep exp_057's taxonomy names, so held-out-ness is NOT
an arm. Held-out items appear under `ic_os_inclass` / `ic_os_to2s`
(gas/hero_flight/illustration), `ic_ts_unseen` (hole/seamless),
`ic2_prefixonly` + `base_prefixonly` (gas/hero_flight/illustration/portal/
shadow), and `ic2_ts_heldout` + `base_ts_heldout` (raven only — the only
NEW held-out arm of this round). To browse a held-out class, filter by
**style**, not arm.

## Per-class allocation (from the artifacts)

| class | corpus clips (std121) | in training | pairs | sidedness | tier | eval items ref/ep role |
|---|---|---|---|---|---|---|
| air_bending | 4 | 4 | 12 | twosided | C (all clips) | 0 / 0 |
| animalization | 8 | 4 | 12 | onesided | B (4 excluded) | 3 / 2 |
| color_rain | 8 | 8 | 24 | onesided | C (all clips) | 0 / 0 |
| cotton_cloud | 6 | 6 | 18 | onesided | C (all clips) | 0 / 0 |
| display_transition | 3 | 3 | 6 | twosided | C (all clips) | 0 / 0 |
| earth_element | 6 | 6 | 18 | onesided | C (all clips) | 0 / 0 |
| earth_wave | 5 | 5 | 15 | twosided | C (all clips) | 1 / 7 (anchor) |
| fire_element | 4 | 4 | 12 | onesided | C (all clips) | 3 / 2 |
| firelava | 6 | 6 | 18 | twosided | C (all clips) | 0 / 0 |
| flame | 2 | 2 | 2 | twosided | C (all clips) | 0 / 0 |
| flying_cam_transition | 4 | 4 | 12 | twosided | C (all clips) | 0 / 0 |
| gas_transformation | 10 | 0 | 0 | — | **A HELD-OUT** | 6 / 5 |
| giant_grab | 5 | 5 | 15 | onesided | C (all clips) | 3 / 2 |
| hero_flight | 10 | 0 | 0 | — | **A HELD-OUT** | 6 / 5 |
| hole_transition | 2 | 0 | 0 | — | **A HELD-OUT** | 2 / 1 |
| illustration_scene | 10 | 0 | 0 | — | **A HELD-OUT** | 5 / 4 |
| jump_transition | 1 | 0 | 0 | — | **A HELD-OUT** | 0 / 0 |
| live_concert | 8 | 8 | 24 | onesided | C (all clips) | 0 / 0 |
| luminous_gaze | 4 | 4 | 12 | onesided | C (all clips) | 0 / 0 |
| melt_transition | 4 | 4 | 12 | twosided | C (all clips) | 0 / 5 |
| money_rain | 7 | 5 | 15 | onesided | B (2 excluded) | 2 / 1 |
| monstrosity | 3 | 3 | 6 | onesided | C (all clips) | 0 / 0 |
| mystification | 5 | 5 | 15 | onesided | C (all clips) | 0 / 0 |
| nature_bloom | 2 | 2 | 2 | onesided | C (all clips) | 0 / 0 |
| plasma_explosion | 4 | 4 | 12 | onesided | C (all clips) | 3 / 2 |
| polygon | 9 | 9 | 27 | onesided | C (all clips) | 0 / 0 |
| portal | 13 | 9 | 27 | onesided | B (4 excluded) | 4 / 3 |
| raven_transition | 4 | 0 | 0 | — | **A HELD-OUT** | 3 / 3 |
| run_set_on_fire | 2 | 2 | 2 | onesided | C (all clips) | 0 / 0 |
| saint_glow | 4 | 4 | 12 | onesided | C (all clips) | 0 / 0 |
| sakura_petals | 2 | 2 | 2 | onesided | C (all clips) | 0 / 0 |
| seamless_transition | 1 | 0 | 0 | — | **A HELD-OUT** | 1 / 0 |
| shadow | 15 | 11 | 33 | onesided | B (4 excluded) | 4 / 3 |
| shadow_smoke | 10 | 9 | 27 | twosided | B (1 excluded) | 1 / 0 (anchor) |
| super_fast_run | 12 | 8 | 24 | onesided | B (4 excluded) | 3 / 2 |
| water_bending | 4 | 4 | 12 | twosided | C (all clips) | 0 / 4 |
| water_element | 5 | 5 | 15 | onesided | C (all clips) | 0 / 0 |
| wireframe | 9 | 5 | 15 | onesided | B (4 excluded) | 3 / 2 |
| wonderland | 2 | 2 | 2 | onesided | C (all clips) | 0 / 0 |

("pairs" counts the class's rows as TARGET; every clip also serves as a
reference for its class siblings. 20 classes trained with zero eval items —
pure diversity mass.)

## Eval set (53 items) by arm

| arm | n | what it is | held-out classes present |
|---|---|---|---|
| ic_os_inclass | 23 | exp_057 suite verbatim: one-sided in-class (suffix-conditioned) | gas, hero_flight, illustration |
| ic_os_to2s | 12 | suite: one-sided ref → foreign two-sided endpoints (cross-target) | gas, hero_flight, illustration |
| ic_ts_unseen | 3 | suite: two-sided unseen structure | hole, seamless |
| ic_anchor | 2 | suite: exp_056 two-sided anchors | — |
| ic2_prefixonly | 8 | NEW: prefix-only (no end frame), in-class | gas, hero_flight, illustration |
| base_prefixonly | 2 | NEW: base-model twins of the above | gas, hero_flight |
| ic2_ts_heldout | 2 | NEW: raven two-sided held-out | raven |
| base_ts_heldout | 1 | NEW: raven base twin | raven |

Caveat carried from the note §5: the 35 one-sided suite items (arms
ic_os_inclass/ic_os_to2s minus two-sided endpoints cases) were generated AND
scored WITH suffix conditioning — off-training-mode for v2, so their deltas
are conservative lower bounds. v2-native numbers = the 8 ic2_prefixonly items.
