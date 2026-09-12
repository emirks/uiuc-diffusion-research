# Hardness selectors compared (grid v3, evals/028; levels; 2 seeds)

A = base_cond effect level (prompt-only hardness, lower = harder). B = DCG w6 neutral − base_cond effect (owner target: reference alone vs best text). C = base_cond effect − base_cond neutral (clause contribution).
Per-class means over ALL content rows (ranking only); tier readouts over SAME rows. B and C use outcome arms → selecting on them is selection on the outcome; A is the only selector that does not.

## Higgsfield 121 f — per class, sorted by A (hardest first)

| class | novelty | n all/same | base neu | base eff (A) | ic_gen neu | ctrl neu | DCG neu | DCG eff | B = DCGneu−base eff | C = eff−neu | ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| giant_grab | unseen | 3/1 | 43.2 | 56.7 | 56.5 | 61.6 | 67.3 | 69.0 | +10.6 | +13.4 | 0.88 |
| nature_bloom | unseen | 2/0 | 51.4 | 62.1 | 46.5 | 63.7 | 65.8 | 68.1 | +3.8 | +10.7 | 0.97 |
| animalization | seen/unseen | 10/4 | 47.0 | 62.9 | 56.8 | 101.8 | 103.2 | 102.7 | +40.4 | +15.9 | 0.84 |
| melt_transition | zero_shot | 8/2 | 45.8 | 64.4 | 63.6 | 90.8 | 108.3 | 108.8 | +43.8 | +18.6 | 0.53 |
| saint_glow | zero_shot | 8/2 | 57.3 | 65.2 | 65.8 | 72.0 | 79.5 | 79.1 | +14.4 | +7.9 | 0.95 |
| super_fast_run | seen/unseen | 10/4 | 64.6 | 68.8 | 86.2 | 89.2 | 93.9 | 98.8 | +25.1 | +4.2 | 0.97 |
| train_rush | zero_shot | 6/2 | 55.2 | 71.8 | 52.6 | 72.0 | 87.5 | 91.1 | +15.7 | +16.6 | 0.97 |
| polygon | seen/unseen | 9/3 | 56.2 | 72.2 | 70.5 | 78.8 | 87.4 | 92.5 | +15.2 | +16.0 | 0.88 |
| raven_transition | zero_shot | 7/1 | 40.8 | 75.1 | 63.5 | 96.0 | 97.0 | 97.5 | +21.8 | +34.3 | 1.00 |
| wonderland | unseen | 2/0 | 65.7 | 79.4 | 81.8 | 91.4 | 96.5 | 96.7 | +17.0 | +13.8 | 0.84 |
| air_bending | unseen | 2/0 | 46.3 | 80.6 | 102.1 | 83.5 | 100.7 | 99.2 | +20.1 | +34.3 | 0.95 |
| earth_element | seen/unseen | 10/4 | 50.7 | 82.4 | 71.2 | 85.6 | 94.0 | 98.3 | +11.5 | +31.7 | 0.97 |
| luminous_gaze | zero_shot | 8/2 | 44.9 | 83.4 | 61.3 | 80.3 | 89.3 | 90.8 | +6.0 | +38.5 | 0.98 |
| hero_flight | seen/unseen | 10/4 | 45.7 | 83.6 | 60.3 | 72.8 | 91.8 | 96.8 | +8.2 | +37.9 | 0.93 |
| point_cloud | zero_shot | 6/2 | 60.2 | 84.7 | 77.6 | 89.4 | 88.1 | 95.6 | +3.3 | +24.5 | 0.88 |
| shadow_smoke | seen/unseen | 10/4 | 49.6 | 87.3 | 80.9 | 96.2 | 98.3 | 98.7 | +11.0 | +37.7 | 0.98 |
| gas_transformation | seen/unseen | 10/4 | 59.6 | 88.0 | 80.0 | 87.7 | 89.7 | 94.8 | +1.7 | +28.4 | 0.83 |
| color_rain | seen/unseen | 10/4 | 48.8 | 91.1 | 63.5 | 74.5 | 81.2 | 82.6 | -10.0 | +42.4 | 0.95 |
| plasma_explosion | unseen | 3/1 | 48.9 | 91.9 | 74.1 | 85.5 | 96.8 | 99.1 | +4.9 | +43.0 | 0.93 |
| earth_wave | unseen | 3/1 | 36.8 | 92.3 | 90.1 | 97.5 | 98.9 | 98.9 | +6.6 | +55.5 | 0.98 |
| live_concert | zero_shot | 8/2 | 54.9 | 93.1 | 55.9 | 81.7 | 83.0 | 94.2 | -10.1 | +38.2 | 1.00 |
| explosion | zero_shot | 6/2 | 48.2 | 94.2 | 78.0 | 82.7 | 93.5 | 88.6 | -0.7 | +46.0 | 0.97 |
| firelava | zero_shot | 8/2 | 31.7 | 95.0 | 79.2 | 98.9 | 99.8 | 99.8 | +4.8 | +63.3 | 0.99 |
| portal | seen/unseen | 10/4 | 45.7 | 95.2 | 58.9 | 92.5 | 99.5 | 99.9 | +4.4 | +49.5 | 0.98 |
| display_transition | zero_shot | 7/1 | 27.2 | 95.4 | 37.2 | 93.4 | 98.9 | 98.8 | +3.6 | +68.1 | 0.98 |
| water_element | unseen | 3/1 | 53.2 | 95.8 | 92.2 | 94.4 | 98.6 | 98.6 | +2.8 | +42.6 | 0.99 |
| water_bending | unseen | 2/0 | 34.8 | 96.1 | 97.8 | 99.7 | 100.0 | 99.6 | +3.9 | +61.3 | 0.98 |
| x_ray | zero_shot | 6/2 | 43.6 | 96.3 | 67.9 | 73.0 | 78.9 | 82.2 | -17.4 | +52.7 | 0.98 |
| illustration_scene | seen/unseen | 10/4 | 56.2 | 96.6 | 78.8 | 79.5 | 80.4 | 83.2 | -16.2 | +40.4 | 0.79 |
| monstrosity | zero_shot | 7/1 | 39.4 | 97.0 | 51.3 | 78.6 | 96.6 | 98.5 | -0.4 | +57.6 | 0.83 |
| northern_lights | zero_shot | 6/2 | 67.8 | 97.3 | 70.6 | 72.9 | 78.9 | 82.7 | -18.4 | +29.5 | 0.85 |
| fire_element | unseen | 3/1 | 75.9 | 98.2 | 85.9 | 97.8 | 99.6 | 99.9 | +1.4 | +22.2 | 0.98 |
| cotton_cloud | zero_shot | 8/2 | 43.6 | 99.0 | 68.0 | 93.1 | 98.9 | 99.6 | -0.1 | +55.4 | 0.98 |
| flame | unseen | 6/2 | 30.8 | 99.0 | 96.4 | 93.5 | 98.2 | 99.1 | -0.8 | +68.2 | 1.00 |
| money_rain | seen/unseen | 10/4 | 62.4 | 99.9 | 84.1 | 89.2 | 97.1 | 100.5 | -2.8 | +37.5 | 0.90 |
| mystification | unseen | 3/1 | 75.9 | 100.5 | 102.6 | 97.9 | 106.4 | 108.4 | +5.8 | +24.6 | 0.73 |
| acid | zero_shot | 6/2 | 86.5 | 100.6 | 88.8 | 102.5 | 109.1 | 111.3 | +8.5 | +14.1 | 0.76 |
| shadow | seen/unseen | 10/4 | 62.3 | 104.4 | 97.0 | 112.3 | 115.9 | 115.0 | +11.4 | +42.1 | 0.66 |
| sakura_petals | unseen | 2/0 | 33.7 | 108.9 | 89.0 | 108.7 | 110.7 | 110.7 | +1.8 | +75.2 | 0.88 |
| wireframe | seen/unseen | 10/4 | 56.2 | 111.1 | 82.6 | 98.4 | 108.2 | 115.6 | -2.9 | +54.9 | 0.67 |
| glitch | zero_shot | 6/2 | 80.3 | 113.3 | 85.7 | 90.7 | 94.8 | 99.6 | -18.5 | +33.0 | 0.61 |
| flying_cam_transition | zero_shot | 8/2 | 66.1 | 114.3 | 84.3 | 101.2 | 134.6 | 136.2 | +20.2 | +48.2 | 0.49 |

Spearman over 42 classes: ρ(A, B) = -0.58 · ρ(C, B) = -0.45 · ρ(A, C) = +0.58

### Zero-shot candidates (17 classes) — top-10 under each selector

- A (lowest base effect): ['melt_transition', 'saint_glow', 'train_rush', 'raven_transition', 'luminous_gaze', 'point_cloud', 'live_concert', 'explosion', 'firelava', 'display_transition']
- B (largest DCG neutral − base effect): ['melt_transition', 'raven_transition', 'flying_cam_transition', 'train_rush', 'saint_glow', 'acid', 'luminous_gaze', 'firelava', 'display_transition', 'point_cloud']
- C (smallest clause contribution): ['saint_glow', 'acid', 'train_rush', 'melt_transition', 'point_cloud', 'northern_lights', 'glitch', 'raven_transition', 'live_concert', 'luminous_gaze']
- overlap A∩B = 8/10 · A∩C = 7/10 · B∩C = 7/10

Tier readout on SAME rows (levels), classes as selected above:

| selection | classes | same rows | base neu | base eff | ic_gen neu | ic_gen eff | ctrl neu | ctrl eff | DCG neu | DCG eff | DCG neu − base eff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all zero-shot (v3 tier) | 17 | 31 | 58.9 | 93.9 | 79.9 | 94.1 | 96.0 | 100.7 | 100.6 | 102.9 | +6.7 |
| A top-10 | 10 | 18 | 52.1 | 86.2 | 74.5 | 87.7 | 93.3 | 96.6 | 96.6 | 98.4 | +10.4 |
| B top-10 | 10 | 18 | 51.9 | 87.3 | 78.3 | 89.4 | 96.9 | 100.1 | 101.4 | 103.9 | +14.1 |
| C top-10 | 10 | 19 | 66.7 | 90.9 | 80.1 | 89.0 | 93.2 | 99.3 | 97.8 | 101.5 | +6.9 |
| zero-shot minus A | 7 | 13 | 68.7 | 104.8 | 87.7 | 103.2 | 99.9 | 106.5 | 106.3 | 109.3 | +1.5 |

## EffectData 81 f — per class, sorted by A (hardest first)

| class | novelty | n all/same | base neu | base eff (A) | ic_gen neu | ctrl neu | DCG neu | DCG eff | B = DCGneu−base eff | C = eff−neu | ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ed.Diamond_Footpath | zero_shot | 3/1 | 60.3 | 56.6 | 58.2 | 73.6 | 85.4 | 92.4 | +28.8 | -3.7 | 0.94 |
| ed.Amber_roots_from_feet | zero_shot | 3/1 | 21.8 | 62.6 | 29.5 | 93.5 | 96.1 | 96.2 | +33.5 | +40.8 | 0.95 |
| ed.Arm_data_corruption | zero_shot | 3/1 | 44.2 | 68.7 | 53.0 | 83.1 | 80.2 | 97.2 | +11.5 | +24.5 | 0.86 |
| ed.Ants_flow_from_eyes | zero_shot | 3/1 | 26.7 | 70.7 | 36.7 | 63.8 | 81.3 | 81.9 | +10.7 | +44.0 | 0.97 |
| ed.Fist_heat_blast | zero_shot | 3/1 | 31.6 | 79.1 | 72.6 | 101.3 | 100.1 | 101.5 | +20.9 | +47.5 | 0.91 |
| ed.Bamboo_shield_in_front | zero_shot | 3/1 | 54.2 | 79.3 | 49.2 | 102.7 | 101.7 | 102.1 | +22.4 | +25.1 | 0.85 |
| ed.Avocado_Pulp_Aura | zero_shot | 3/1 | 43.8 | 79.7 | 68.7 | 99.3 | 106.3 | 106.6 | +26.6 | +35.9 | 0.86 |
| ed.Bile_aura_hands | zero_shot | 3/1 | 34.3 | 81.6 | 42.0 | 86.4 | 89.9 | 94.9 | +8.3 | +47.3 | 0.84 |
| ed.Acid_mist_limb_transformation | zero_shot | 3/1 | 27.4 | 82.2 | 45.9 | 70.7 | 89.1 | 90.0 | +6.9 | +54.8 | 0.98 |
| ed.Back_laser_wings | zero_shot | 3/1 | 39.0 | 82.8 | 37.2 | 101.5 | 102.3 | 102.4 | +19.5 | +43.8 | 0.96 |
| ed.Body_gear_field | zero_shot | 3/1 | 29.6 | 85.0 | 32.8 | 83.6 | 100.9 | 100.4 | +15.9 | +55.4 | 0.96 |
| ed.Chaotic_wings_from_shoulders | zero_shot | 3/1 | 38.4 | 87.4 | 43.4 | 88.7 | 96.5 | 96.2 | +9.0 | +49.0 | 0.95 |
| ed.Avocado_Seed_Pendant | zero_shot | 3/1 | 53.5 | 87.5 | 76.4 | 89.6 | 96.9 | 103.9 | +9.3 | +34.0 | 0.75 |
| ed.Chest_fish_burst | zero_shot | 3/1 | 28.7 | 88.2 | 59.6 | 96.0 | 93.5 | 92.6 | +5.3 | +59.6 | 0.99 |
| ed.Amethyst_sword_from_palm | zero_shot | 3/1 | 40.7 | 88.5 | 46.9 | 89.0 | 98.0 | 97.6 | +9.5 | +47.9 | 0.97 |
| ed.Back_Born_Letter_Wings | zero_shot | 3/1 | 38.1 | 91.1 | 55.2 | 93.5 | 99.2 | 99.3 | +8.0 | +53.1 | 0.98 |
| ed.Amber_crystal_wings_on_back | zero_shot | 3/1 | 35.3 | 92.5 | 42.2 | 86.3 | 101.8 | 102.5 | +9.3 | +57.2 | 0.90 |
| ed.Amber_streams_from_eyes | zero_shot | 3/1 | 33.3 | 92.6 | 81.2 | 92.7 | 96.9 | 101.8 | +4.3 | +59.3 | 0.93 |
| ed.Chest_Burst_Ribbons | zero_shot | 3/1 | 28.9 | 93.2 | 50.3 | 98.2 | 99.2 | 99.5 | +6.0 | +64.3 | 0.98 |
| ed.Body_Meteorite_Crystal | zero_shot | 3/1 | 36.8 | 95.3 | 61.5 | 94.7 | 98.0 | 98.6 | +2.7 | +58.5 | 0.99 |
| ed.Arm_chip_serpent | zero_shot | 3/1 | 19.0 | 95.3 | 49.6 | 99.1 | 100.5 | 100.1 | +5.2 | +76.4 | 0.98 |
| ed.Back_bubble_cloak | zero_shot | 3/1 | 42.7 | 96.6 | 67.8 | 94.3 | 92.3 | 96.5 | -4.3 | +53.9 | 0.98 |
| ed.Chest_Erupting_Shield | zero_shot | 3/1 | 41.6 | 97.5 | 56.4 | 100.0 | 100.0 | 99.9 | +2.4 | +55.9 | 0.99 |
| ed.Electric_Clothes_Glow | zero_shot | 3/1 | 51.2 | 97.7 | 63.6 | 95.7 | 99.9 | 99.8 | +2.2 | +46.5 | 0.97 |
| ed.Brass_serpent_coil | zero_shot | 3/1 | 38.4 | 98.0 | 39.1 | 99.1 | 96.4 | 96.9 | -1.6 | +59.6 | 0.99 |
| ed.Dark_mist_wings_from_back | zero_shot | 3/1 | 46.2 | 98.6 | 62.9 | 93.3 | 104.4 | 104.8 | +5.8 | +52.4 | 0.91 |
| ed.Chest_light_field | zero_shot | 3/1 | 34.3 | 100.2 | 69.4 | 103.2 | 103.6 | 102.5 | +3.4 | +65.9 | 0.91 |
| ed.Cucumber_seeds_from_chest | zero_shot | 3/1 | 27.6 | 100.7 | 58.0 | 99.1 | 96.0 | 98.5 | -4.7 | +73.1 | 0.93 |
| ed.Confetti_Burst_from_Hands | zero_shot | 3/1 | 46.4 | 100.9 | 88.6 | 106.1 | 105.4 | 105.8 | +4.5 | +54.5 | 0.90 |
| ed.Eye_Shard_Stream | zero_shot | 3/1 | 42.1 | 103.2 | 73.7 | 106.5 | 104.2 | 107.9 | +1.0 | +61.1 | 0.88 |
| ed.Chest_Laser_Burst | zero_shot | 3/1 | 38.7 | 105.9 | 80.2 | 108.2 | 114.3 | 114.1 | +8.5 | +67.1 | 0.80 |
| ed.Eye_Meteor_Burst | zero_shot | 3/1 | 38.8 | 107.6 | 57.8 | 111.4 | 105.4 | 107.8 | -2.2 | +68.8 | 0.80 |
| ed.Chest_sigil | zero_shot | 3/1 | 41.1 | 109.2 | 51.9 | 104.1 | 116.9 | 118.0 | +7.7 | +68.1 | 0.75 |
| ed.Dark_mist_from_mouth | zero_shot | 3/1 | 51.2 | 109.3 | 79.8 | 92.6 | 97.2 | 102.9 | -12.1 | +58.1 | 0.87 |

Spearman over 34 classes: ρ(A, B) = -0.81 · ρ(C, B) = -0.69 · ρ(A, C) = +0.77

### Zero-shot candidates (34 classes) — top-10 under each selector

- A (lowest base effect): ['ed.Diamond_Footpath', 'ed.Amber_roots_from_feet', 'ed.Arm_data_corruption', 'ed.Ants_flow_from_eyes', 'ed.Fist_heat_blast', 'ed.Bamboo_shield_in_front', 'ed.Avocado_Pulp_Aura', 'ed.Bile_aura_hands', 'ed.Acid_mist_limb_transformation', 'ed.Back_laser_wings']
- B (largest DCG neutral − base effect): ['ed.Amber_roots_from_feet', 'ed.Diamond_Footpath', 'ed.Avocado_Pulp_Aura', 'ed.Bamboo_shield_in_front', 'ed.Fist_heat_blast', 'ed.Back_laser_wings', 'ed.Body_gear_field', 'ed.Arm_data_corruption', 'ed.Ants_flow_from_eyes', 'ed.Amethyst_sword_from_palm']
- C (smallest clause contribution): ['ed.Diamond_Footpath', 'ed.Arm_data_corruption', 'ed.Bamboo_shield_in_front', 'ed.Avocado_Seed_Pendant', 'ed.Avocado_Pulp_Aura', 'ed.Amber_roots_from_feet', 'ed.Back_laser_wings', 'ed.Ants_flow_from_eyes', 'ed.Electric_Clothes_Glow', 'ed.Bile_aura_hands']
- overlap A∩B = 8/10 · A∩C = 8/10 · B∩C = 7/10

Tier readout on SAME rows (levels), classes as selected above:

| selection | classes | same rows | base neu | base eff | ic_gen neu | ic_gen eff | ctrl neu | ctrl eff | DCG neu | DCG eff | DCG neu − base eff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all zero-shot (v3 tier) | 34 | 34 | 35.3 | 88.8 | 57.3 | 89.1 | 96.9 | 98.4 | 99.0 | 100.3 | +10.2 |
| A top-10 | 10 | 10 | 36.8 | 72.7 | 48.4 | 77.9 | 92.5 | 94.9 | 97.2 | 98.4 | +24.5 |
| B top-10 | 10 | 10 | 38.0 | 74.7 | 46.5 | 76.6 | 95.1 | 96.0 | 98.6 | 99.4 | +23.9 |
| C top-10 | 10 | 10 | 40.1 | 72.6 | 53.2 | 80.8 | 95.0 | 96.5 | 99.1 | 100.1 | +26.5 |
| zero-shot minus A | 24 | 24 | 34.7 | 95.5 | 61.0 | 93.8 | 98.7 | 99.9 | 99.8 | 101.1 | +4.3 |

