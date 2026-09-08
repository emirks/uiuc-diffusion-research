# grid v3 — paper arms, v4 pool-% (eval entry `028_grid_v3_paper_arms__dai__2026-09-07`)

pool-% = mean app_ref vs the row's GT pool ÷ class ceiling (certified matrix; ceilings_v3.json overlay for new classes); %_same cells are cross-class comparable, (%_proxy) cells are ranking-only. Levels, not verdicts. ED81 arms: 81 f, frame-0 anchor — copy/core flags not comparable with 121 f.

| arm | rows scored | %_same headline (n) | seen/same | seen/cross | seen/foreign | unseen/same | unseen/cross | unseen/foreign | zero_shot/same | zero_shot/cross | zero_shot/foreign |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `base_cond_neutral_v3` | 282/282 | 61.4 (90) | 57.9 (26) | — | — | 64.9 (33) | (49.2) (52) | (46.9) (52) | 60.7 (31) | (51.2) (44) | (46.0) (44) |
| `base_cond_neutral_v3ed81` | 102/102 | 35.3 (34) | — | — | — | — | — | — | 35.3 (34) | (33.3) (34) | (46.6) (34) |
| `base_cond_effect_v3` | 282/282 | 92.6 (90) | 94.3 (26) | — | — | 90.5 (33) | (88.5) (52) | (84.2) (52) | 93.5 (31) | (92.1) (44) | (86.0) (44) |
| `base_cond_effect_v3ed81` | 102/102 | 88.8 (34) | — | — | — | — | — | — | 88.8 (34) | (91.2) (34) | (90.4) (34) |
| `ic_gen_neutral_v3` | 282/282 | 85.1 (90) | 84.9 (26) | — | — | 88.4 (33) | (76.7) (52) | (66.1) (52) | 81.7 (31) | (68.6) (44) | (56.1) (44) |
| `ic_gen_neutral_v3ed81` | 102/102 | 57.3 (34) | — | — | — | — | — | — | 57.3 (34) | (49.2) (34) | (64.8) (34) |
| `ic_gen_effect_v3` | 282/282 | 95.0 (90) | 95.5 (26) | — | — | 95.9 (33) | (92.3) (52) | (85.8) (52) | 93.6 (31) | (93.2) (44) | (82.1) (44) |
| `ic_gen_effect_v3ed81` | 102/102 | 89.1 (34) | — | — | — | — | — | — | 89.1 (34) | (90.0) (34) | (90.1) (34) |
| `dualforce_control_neutral_v3` | 282/282 | 96.1 (90) | 94.8 (26) | — | — | 97.7 (33) | (87.2) (52) | (83.5) (52) | 95.6 (31) | (88.4) (44) | (78.8) (44) |
| `dualforce_control_neutral_v3ed81` | 102/102 | 96.9 (34) | — | — | — | — | — | — | 96.9 (34) | (95.4) (34) | (90.1) (34) |
| `dualforce_control_effect_v3` | 282/282 | 100.2 (90) | 100.8 (26) | — | — | 99.4 (33) | (96.4) (52) | (91.7) (52) | 100.6 (31) | (94.5) (44) | (88.9) (44) |
| `dualforce_control_effect_v3ed81` | 102/102 | 98.5 (34) | — | — | — | — | — | — | 98.5 (34) | (99.5) (34) | (96.1) (34) |
| `dualforce_dcg_w6_neutral_v3` | 282/282 | 99.5 (90) | 101.1 (26) | — | — | 97.4 (33) | (94.9) (52) | (92.0) (52) | 100.5 (31) | (94.1) (44) | (93.7) (44) |
| `dualforce_dcg_w6_neutral_v3ed81` | 102/102 | 99.0 (34) | — | — | — | — | — | — | 99.0 (34) | (100.3) (34) | (96.2) (34) |
| `dualforce_dcg_w6_effect_v3` | 282/282 | 101.9 (90) | 103.5 (26) | — | — | 99.6 (33) | (96.0) (52) | (96.1) (52) | 103.0 (31) | (97.4) (44) | (94.5) (44) |
| `dualforce_dcg_w6_effect_v3ed81` | 102/102 | 100.3 (34) | — | — | — | — | — | — | 100.3 (34) | (101.7) (34) | (99.2) (34) |

copy_max mean per cell is in summary.json.

## neutral prompts — pooled over all content cells (same + cross + foreign), by reference novelty

| arm | seen (HF) | unseen (HF) | zero-shot (HF) | zero-shot (EffectData 81 f) | all HF rows |
|---|---|---|---|---|---|
| base_cond | 57.9 (26) | 52.1 (137) | 51.7 (119) | 38.4 (102) | 52.5 (282) |
| ic_gen | 84.9 (26) | 75.5 (137) | 67.4 (119) | 57.1 (102) | 72.9 (282) |
| dualforce_control | 94.8 (26) | 88.3 (137) | 86.7 (119) | 94.1 (102) | 88.2 (282) |
| dualforce_dcg_w6 | 101.1 (26) | 94.4 (137) | 95.6 (119) | 98.5 (102) | 95.5 (282) |

## effect prompts — pooled over all content cells (same + cross + foreign), by reference novelty

| arm | seen (HF) | unseen (HF) | zero-shot (HF) | zero-shot (EffectData 81 f) | all HF rows |
|---|---|---|---|---|---|
| base_cond | 94.3 (26) | 87.3 (137) | 90.2 (119) | 90.2 (102) | 89.2 (282) |
| ic_gen | 95.5 (26) | 90.7 (137) | 89.2 (119) | 89.7 (102) | 90.5 (282) |
| dualforce_control | 100.8 (26) | 95.4 (137) | 94.0 (119) | 98.0 (102) | 95.3 (282) |
| dualforce_dcg_w6 | 103.5 (26) | 96.9 (137) | 97.8 (119) | 100.4 (102) | 97.9 (282) |

## Per-class diagnostics (owner request 2026-09-08)

Per donor class over all its rows (same + cross + foreign pooled, so ranking only). Levels, n = rows; 2-3-row classes carry the ~12 pp seed SD.

### T1 Higgsfield 121 f — sorted by base_cond Δ = effect − neutral

| class | n | base neutral | base effect | Δ eff−neu | dualforce ctrl neutral | dualforce ctrl effect | DCG w6 effect | ceiling |
|---|---|---|---|---|---|---|---|---|
| super_fast_run | 10 | 64.6 | 68.8 | +4.2 | 89.2 | 96.5 | 98.8 | 0.97 |
| saint_glow | 8 | 57.3 | 65.2 | +7.9 | 72.0 | 75.4 | 79.1 | 0.95 |
| nature_bloom | 2 | 51.4 | 62.1 | +10.7 | 63.7 | 63.3 | 68.1 | 0.97 |
| giant_grab | 3 | 43.2 | 56.7 | +13.4 | 61.6 | 65.3 | 69.0 | 0.88 |
| wonderland | 2 | 65.7 | 79.4 | +13.8 | 91.4 | 91.7 | 96.7 | 0.84 |
| acid | 6 | 86.5 | 100.6 | +14.1 | 102.5 | 109.7 | 111.3 | 0.76 |
| animalization | 10 | 47.0 | 62.9 | +15.9 | 101.8 | 98.0 | 102.7 | 0.84 |
| polygon | 9 | 56.2 | 72.2 | +16.0 | 78.8 | 83.7 | 92.5 | 0.88 |
| train_rush | 6 | 55.2 | 71.8 | +16.6 | 72.0 | 92.3 | 91.1 | 0.97 |
| melt_transition | 8 | 45.8 | 64.4 | +18.6 | 90.8 | 93.0 | 108.8 | 0.53 |
| fire_element | 3 | 75.9 | 98.2 | +22.2 | 97.8 | 98.9 | 99.9 | 0.98 |
| point_cloud | 6 | 60.2 | 84.7 | +24.5 | 89.4 | 92.7 | 95.6 | 0.88 |
| mystification | 3 | 75.9 | 100.5 | +24.6 | 97.9 | 110.3 | 108.4 | 0.73 |
| gas_transformation | 10 | 59.6 | 88.0 | +28.4 | 87.7 | 96.8 | 94.8 | 0.83 |
| northern_lights | 6 | 67.8 | 97.3 | +29.5 | 72.9 | 86.6 | 82.7 | 0.85 |
| earth_element | 10 | 50.7 | 82.4 | +31.7 | 85.6 | 93.7 | 98.3 | 0.97 |
| glitch | 6 | 80.3 | 113.3 | +33.0 | 90.7 | 104.2 | 99.6 | 0.61 |
| raven_transition | 7 | 40.8 | 75.1 | +34.3 | 96.0 | 97.5 | 97.5 | 1.00 |
| air_bending | 2 | 46.3 | 80.6 | +34.3 | 83.5 | 95.7 | 99.2 | 0.95 |
| money_rain | 10 | 62.4 | 99.9 | +37.5 | 89.2 | 100.8 | 100.5 | 0.90 |
| shadow_smoke | 10 | 49.6 | 87.3 | +37.7 | 96.2 | 96.4 | 98.7 | 0.98 |
| hero_flight | 10 | 45.7 | 83.6 | +37.9 | 72.8 | 93.1 | 96.8 | 0.93 |
| live_concert | 8 | 54.9 | 93.1 | +38.2 | 81.7 | 94.1 | 94.2 | 1.00 |
| luminous_gaze | 8 | 44.9 | 83.4 | +38.5 | 80.3 | 84.6 | 90.8 | 0.98 |
| illustration_scene | 10 | 56.2 | 96.6 | +40.4 | 79.5 | 90.5 | 83.2 | 0.79 |
| shadow | 10 | 62.3 | 104.4 | +42.1 | 112.3 | 112.8 | 115.0 | 0.66 |
| color_rain | 10 | 48.8 | 91.1 | +42.4 | 74.5 | 81.3 | 82.6 | 0.95 |
| water_element | 3 | 53.2 | 95.8 | +42.6 | 94.4 | 97.9 | 98.6 | 0.99 |
| plasma_explosion | 3 | 48.9 | 91.9 | +43.0 | 85.5 | 95.1 | 99.1 | 0.93 |
| explosion | 6 | 48.2 | 94.2 | +46.0 | 82.7 | 83.6 | 88.6 | 0.97 |
| flying_cam_transition | 8 | 66.1 | 114.3 | +48.2 | 101.2 | 114.3 | 136.2 | 0.49 |
| portal | 10 | 45.7 | 95.2 | +49.5 | 92.5 | 100.1 | 99.9 | 0.98 |
| x_ray | 6 | 43.6 | 96.3 | +52.7 | 73.0 | 83.3 | 82.2 | 0.98 |
| wireframe | 10 | 56.2 | 111.1 | +54.9 | 98.4 | 112.7 | 115.6 | 0.67 |
| cotton_cloud | 8 | 43.6 | 99.0 | +55.4 | 93.1 | 98.8 | 99.6 | 0.98 |
| earth_wave | 3 | 36.8 | 92.3 | +55.5 | 97.5 | 94.5 | 98.9 | 0.98 |
| monstrosity | 7 | 39.4 | 97.0 | +57.6 | 78.6 | 90.8 | 98.5 | 0.83 |
| water_bending | 2 | 34.8 | 96.1 | +61.3 | 99.7 | 100.6 | 99.6 | 0.98 |
| firelava | 8 | 31.7 | 95.0 | +63.3 | 98.9 | 99.0 | 99.8 | 0.99 |
| display_transition | 7 | 27.2 | 95.4 | +68.1 | 93.4 | 97.3 | 98.8 | 0.98 |
| flame | 6 | 30.8 | 99.0 | +68.2 | 93.5 | 99.3 | 99.1 | 1.00 |
| sakura_petals | 2 | 33.7 | 108.9 | +75.2 | 108.7 | 110.0 | 110.7 | 0.88 |

### T2 Higgsfield 121 f — score = (DCG w6 effect − base effect) − (base effect − base neutral), descending

| class | n | base neutral | base effect | Δ eff−neu | DCG w6 effect | DCG − base effect | ctrl − base effect | score |
|---|---|---|---|---|---|---|---|---|
| super_fast_run | 10 | 64.6 | 68.8 | +4.2 | 98.8 | +30.0 | +27.7 | +25.8 |
| melt_transition | 8 | 45.8 | 64.4 | +18.6 | 108.8 | +44.3 | +28.6 | +25.7 |
| animalization | 10 | 47.0 | 62.9 | +15.9 | 102.7 | +39.8 | +35.1 | +24.0 |
| saint_glow | 8 | 57.3 | 65.2 | +7.9 | 79.1 | +14.0 | +10.2 | +6.0 |
| polygon | 9 | 56.2 | 72.2 | +16.0 | 92.5 | +20.3 | +11.5 | +4.4 |
| wonderland | 2 | 65.7 | 79.4 | +13.8 | 96.7 | +17.2 | +12.3 | +3.5 |
| train_rush | 6 | 55.2 | 71.8 | +16.6 | 91.1 | +19.3 | +20.5 | +2.7 |
| giant_grab | 3 | 43.2 | 56.7 | +13.4 | 69.0 | +12.3 | +8.6 | -1.1 |
| acid | 6 | 86.5 | 100.6 | +14.1 | 111.3 | +10.7 | +9.1 | -3.4 |
| nature_bloom | 2 | 51.4 | 62.1 | +10.7 | 68.1 | +6.1 | +1.3 | -4.6 |
| raven_transition | 7 | 40.8 | 75.1 | +34.3 | 97.5 | +22.4 | +22.3 | -12.0 |
| point_cloud | 6 | 60.2 | 84.7 | +24.5 | 95.6 | +10.8 | +8.0 | -13.7 |
| air_bending | 2 | 46.3 | 80.6 | +34.3 | 99.2 | +18.6 | +15.1 | -15.7 |
| earth_element | 10 | 50.7 | 82.4 | +31.7 | 98.3 | +15.8 | +11.3 | -15.9 |
| mystification | 3 | 75.9 | 100.5 | +24.6 | 108.4 | +7.9 | +9.7 | -16.7 |
| fire_element | 3 | 75.9 | 98.2 | +22.2 | 99.9 | +1.7 | +0.8 | -20.5 |
| gas_transformation | 10 | 59.6 | 88.0 | +28.4 | 94.8 | +6.8 | +8.8 | -21.6 |
| hero_flight | 10 | 45.7 | 83.6 | +37.9 | 96.8 | +13.1 | +9.4 | -24.8 |
| shadow_smoke | 10 | 49.6 | 87.3 | +37.7 | 98.7 | +11.4 | +9.2 | -26.3 |
| flying_cam_transition | 8 | 66.1 | 114.3 | +48.2 | 136.2 | +21.9 | -0.0 | -26.3 |
| luminous_gaze | 8 | 44.9 | 83.4 | +38.5 | 90.8 | +7.5 | +1.3 | -31.0 |
| shadow | 10 | 62.3 | 104.4 | +42.1 | 115.0 | +10.6 | +8.3 | -31.5 |
| plasma_explosion | 3 | 48.9 | 91.9 | +43.0 | 99.1 | +7.1 | +3.2 | -35.9 |
| money_rain | 10 | 62.4 | 99.9 | +37.5 | 100.5 | +0.6 | +0.9 | -36.8 |
| live_concert | 8 | 54.9 | 93.1 | +38.2 | 94.2 | +1.1 | +1.0 | -37.1 |
| water_element | 3 | 53.2 | 95.8 | +42.6 | 98.6 | +2.7 | +2.1 | -39.9 |
| northern_lights | 6 | 67.8 | 97.3 | +29.5 | 82.7 | -14.6 | -10.7 | -44.1 |
| portal | 10 | 45.7 | 95.2 | +49.5 | 99.9 | +4.7 | +5.0 | -44.8 |
| glitch | 6 | 80.3 | 113.3 | +33.0 | 99.6 | -13.7 | -9.1 | -46.7 |
| earth_wave | 3 | 36.8 | 92.3 | +55.5 | 98.9 | +6.6 | +2.2 | -48.9 |
| wireframe | 10 | 56.2 | 111.1 | +54.9 | 115.6 | +4.5 | +1.6 | -50.5 |
| color_rain | 10 | 48.8 | 91.1 | +42.4 | 82.6 | -8.5 | -9.8 | -50.9 |
| explosion | 6 | 48.2 | 94.2 | +46.0 | 88.6 | -5.6 | -10.5 | -51.6 |
| illustration_scene | 10 | 56.2 | 96.6 | +40.4 | 83.2 | -13.4 | -6.1 | -53.8 |
| cotton_cloud | 8 | 43.6 | 99.0 | +55.4 | 99.6 | +0.6 | -0.1 | -54.7 |
| monstrosity | 7 | 39.4 | 97.0 | +57.6 | 98.5 | +1.5 | -6.2 | -56.1 |
| water_bending | 2 | 34.8 | 96.1 | +61.3 | 99.6 | +3.4 | +4.4 | -57.8 |
| firelava | 8 | 31.7 | 95.0 | +63.3 | 99.8 | +4.8 | +4.0 | -58.5 |
| display_transition | 7 | 27.2 | 95.4 | +68.1 | 98.8 | +3.4 | +2.0 | -64.7 |
| x_ray | 6 | 43.6 | 96.3 | +52.7 | 82.2 | -14.1 | -13.0 | -66.8 |
| flame | 6 | 30.8 | 99.0 | +68.2 | 99.1 | +0.1 | +0.3 | -68.2 |
| sakura_petals | 2 | 33.7 | 108.9 | +75.2 | 110.7 | +1.8 | +1.1 | -73.4 |

### T1 EffectData 81 f — sorted by base_cond Δ = effect − neutral

| class | n | base neutral | base effect | Δ eff−neu | dualforce ctrl neutral | dualforce ctrl effect | DCG w6 effect | ceiling |
|---|---|---|---|---|---|---|---|---|
| ed.Diamond_Footpath | 3 | 60.3 | 56.6 | -3.7 | 73.6 | 78.6 | 92.4 | 0.94 |
| ed.Arm_data_corruption | 3 | 44.2 | 68.7 | +24.5 | 83.1 | 91.0 | 97.2 | 0.86 |
| ed.Bamboo_shield_in_front | 3 | 54.2 | 79.3 | +25.1 | 102.7 | 103.3 | 102.1 | 0.85 |
| ed.Avocado_Seed_Pendant | 3 | 53.5 | 87.5 | +34.0 | 89.6 | 99.6 | 103.9 | 0.75 |
| ed.Avocado_Pulp_Aura | 3 | 43.8 | 79.7 | +35.9 | 99.3 | 103.1 | 106.6 | 0.86 |
| ed.Amber_roots_from_feet | 3 | 21.8 | 62.6 | +40.8 | 93.5 | 94.8 | 96.2 | 0.95 |
| ed.Back_laser_wings | 3 | 39.0 | 82.8 | +43.8 | 101.5 | 100.6 | 102.4 | 0.96 |
| ed.Ants_flow_from_eyes | 3 | 26.7 | 70.7 | +44.0 | 63.8 | 68.0 | 81.9 | 0.97 |
| ed.Electric_Clothes_Glow | 3 | 51.2 | 97.7 | +46.5 | 95.7 | 95.7 | 99.8 | 0.97 |
| ed.Bile_aura_hands | 3 | 34.3 | 81.6 | +47.3 | 86.4 | 93.2 | 94.9 | 0.84 |
| ed.Fist_heat_blast | 3 | 31.6 | 79.1 | +47.5 | 101.3 | 96.7 | 101.5 | 0.91 |
| ed.Amethyst_sword_from_palm | 3 | 40.7 | 88.5 | +47.9 | 89.0 | 94.4 | 97.6 | 0.97 |
| ed.Chaotic_wings_from_shoulders | 3 | 38.4 | 87.4 | +49.0 | 88.7 | 89.3 | 96.2 | 0.95 |
| ed.Dark_mist_wings_from_back | 3 | 46.2 | 98.6 | +52.4 | 93.3 | 102.5 | 104.8 | 0.91 |
| ed.Back_Born_Letter_Wings | 3 | 38.1 | 91.1 | +53.1 | 93.5 | 97.4 | 99.3 | 0.98 |
| ed.Back_bubble_cloak | 3 | 42.7 | 96.6 | +53.9 | 94.3 | 98.9 | 96.5 | 0.98 |
| ed.Confetti_Burst_from_Hands | 3 | 46.4 | 100.9 | +54.5 | 106.1 | 106.0 | 105.8 | 0.90 |
| ed.Acid_mist_limb_transformation | 3 | 27.4 | 82.2 | +54.8 | 70.7 | 74.7 | 90.0 | 0.98 |
| ed.Body_gear_field | 3 | 29.6 | 85.0 | +55.4 | 83.6 | 97.0 | 100.4 | 0.96 |
| ed.Chest_Erupting_Shield | 3 | 41.6 | 97.5 | +55.9 | 100.0 | 99.3 | 99.9 | 0.99 |
| ed.Amber_crystal_wings_on_back | 3 | 35.3 | 92.5 | +57.2 | 86.3 | 93.7 | 102.5 | 0.90 |
| ed.Dark_mist_from_mouth | 3 | 51.2 | 109.3 | +58.1 | 92.6 | 107.7 | 102.9 | 0.87 |
| ed.Body_Meteorite_Crystal | 3 | 36.8 | 95.3 | +58.5 | 94.7 | 99.3 | 98.6 | 0.99 |
| ed.Amber_streams_from_eyes | 3 | 33.3 | 92.6 | +59.3 | 92.7 | 101.3 | 101.8 | 0.93 |
| ed.Chest_fish_burst | 3 | 28.7 | 88.2 | +59.6 | 96.0 | 97.1 | 92.6 | 0.99 |
| ed.Brass_serpent_coil | 3 | 38.4 | 98.0 | +59.6 | 99.1 | 99.5 | 96.9 | 0.99 |
| ed.Eye_Shard_Stream | 3 | 42.1 | 103.2 | +61.1 | 106.5 | 106.0 | 107.9 | 0.88 |
| ed.Chest_Burst_Ribbons | 3 | 28.9 | 93.2 | +64.3 | 98.2 | 99.9 | 99.5 | 0.98 |
| ed.Chest_light_field | 3 | 34.3 | 100.2 | +65.9 | 103.2 | 104.7 | 102.5 | 0.91 |
| ed.Chest_Laser_Burst | 3 | 38.7 | 105.9 | +67.1 | 108.2 | 113.0 | 114.1 | 0.80 |
| ed.Chest_sigil | 3 | 41.1 | 109.2 | +68.1 | 104.1 | 109.6 | 118.0 | 0.75 |
| ed.Eye_Meteor_Burst | 3 | 38.8 | 107.6 | +68.8 | 111.4 | 114.2 | 107.8 | 0.80 |
| ed.Cucumber_seeds_from_chest | 3 | 27.6 | 100.7 | +73.1 | 99.1 | 101.6 | 98.5 | 0.93 |
| ed.Arm_chip_serpent | 3 | 19.0 | 95.3 | +76.4 | 99.1 | 101.0 | 100.1 | 0.98 |

### T2 EffectData 81 f — score = (DCG w6 effect − base effect) − (base effect − base neutral), descending

| class | n | base neutral | base effect | Δ eff−neu | DCG w6 effect | DCG − base effect | ctrl − base effect | score |
|---|---|---|---|---|---|---|---|---|
| ed.Diamond_Footpath | 3 | 60.3 | 56.6 | -3.7 | 92.4 | +35.8 | +22.0 | +39.5 |
| ed.Arm_data_corruption | 3 | 44.2 | 68.7 | +24.5 | 97.2 | +28.5 | +22.2 | +4.0 |
| ed.Bamboo_shield_in_front | 3 | 54.2 | 79.3 | +25.1 | 102.1 | +22.8 | +24.0 | -2.3 |
| ed.Amber_roots_from_feet | 3 | 21.8 | 62.6 | +40.8 | 96.2 | +33.6 | +32.2 | -7.2 |
| ed.Avocado_Pulp_Aura | 3 | 43.8 | 79.7 | +35.9 | 106.6 | +26.9 | +23.4 | -8.9 |
| ed.Avocado_Seed_Pendant | 3 | 53.5 | 87.5 | +34.0 | 103.9 | +16.4 | +12.0 | -17.6 |
| ed.Back_laser_wings | 3 | 39.0 | 82.8 | +43.8 | 102.4 | +19.5 | +17.7 | -24.3 |
| ed.Fist_heat_blast | 3 | 31.6 | 79.1 | +47.5 | 101.5 | +22.4 | +17.6 | -25.1 |
| ed.Ants_flow_from_eyes | 3 | 26.7 | 70.7 | +44.0 | 81.9 | +11.2 | -2.6 | -32.8 |
| ed.Bile_aura_hands | 3 | 34.3 | 81.6 | +47.3 | 94.9 | +13.3 | +11.6 | -34.0 |
| ed.Amethyst_sword_from_palm | 3 | 40.7 | 88.5 | +47.9 | 97.6 | +9.0 | +5.8 | -38.9 |
| ed.Body_gear_field | 3 | 29.6 | 85.0 | +55.4 | 100.4 | +15.4 | +12.0 | -39.9 |
| ed.Chaotic_wings_from_shoulders | 3 | 38.4 | 87.4 | +49.0 | 96.2 | +8.7 | +1.9 | -40.3 |
| ed.Electric_Clothes_Glow | 3 | 51.2 | 97.7 | +46.5 | 99.8 | +2.0 | -2.0 | -44.5 |
| ed.Back_Born_Letter_Wings | 3 | 38.1 | 91.1 | +53.1 | 99.3 | +8.2 | +6.3 | -44.9 |
| ed.Dark_mist_wings_from_back | 3 | 46.2 | 98.6 | +52.4 | 104.8 | +6.2 | +3.9 | -46.2 |
| ed.Acid_mist_limb_transformation | 3 | 27.4 | 82.2 | +54.8 | 90.0 | +7.8 | -7.5 | -47.0 |
| ed.Amber_crystal_wings_on_back | 3 | 35.3 | 92.5 | +57.2 | 102.5 | +10.0 | +1.2 | -47.2 |
| ed.Confetti_Burst_from_Hands | 3 | 46.4 | 100.9 | +54.5 | 105.8 | +4.9 | +5.1 | -49.6 |
| ed.Amber_streams_from_eyes | 3 | 33.3 | 92.6 | +59.3 | 101.8 | +9.2 | +8.7 | -50.1 |
| ed.Chest_Erupting_Shield | 3 | 41.6 | 97.5 | +55.9 | 99.9 | +2.3 | +1.8 | -53.6 |
| ed.Back_bubble_cloak | 3 | 42.7 | 96.6 | +53.9 | 96.5 | -0.1 | +2.2 | -54.0 |
| ed.Body_Meteorite_Crystal | 3 | 36.8 | 95.3 | +58.5 | 98.6 | +3.3 | +4.0 | -55.2 |
| ed.Chest_fish_burst | 3 | 28.7 | 88.2 | +59.6 | 92.6 | +4.4 | +8.9 | -55.2 |
| ed.Eye_Shard_Stream | 3 | 42.1 | 103.2 | +61.1 | 107.9 | +4.7 | +2.8 | -56.3 |
| ed.Chest_Burst_Ribbons | 3 | 28.9 | 93.2 | +64.3 | 99.5 | +6.3 | +6.7 | -58.0 |
| ed.Chest_Laser_Burst | 3 | 38.7 | 105.9 | +67.1 | 114.1 | +8.2 | +7.1 | -58.9 |
| ed.Chest_sigil | 3 | 41.1 | 109.2 | +68.1 | 118.0 | +8.8 | +0.4 | -59.4 |
| ed.Brass_serpent_coil | 3 | 38.4 | 98.0 | +59.6 | 96.9 | -1.1 | +1.5 | -60.7 |
| ed.Chest_light_field | 3 | 34.3 | 100.2 | +65.9 | 102.5 | +2.3 | +4.6 | -63.6 |
| ed.Dark_mist_from_mouth | 3 | 51.2 | 109.3 | +58.1 | 102.9 | -6.3 | -1.5 | -64.4 |
| ed.Eye_Meteor_Burst | 3 | 38.8 | 107.6 | +68.8 | 107.8 | +0.2 | +6.6 | -68.7 |
| ed.Arm_chip_serpent | 3 | 19.0 | 95.3 | +76.4 | 100.1 | +4.8 | +5.7 | -71.5 |
| ed.Cucumber_seeds_from_chest | 3 | 27.6 | 100.7 | +73.1 | 98.5 | -2.2 | +0.9 | -75.3 |

### T3 novelty tables restricted to the score > 0 classes (diagnostic slice, selected on these scores)

Higgsfield: ['animalization', 'melt_transition', 'polygon', 'saint_glow', 'super_fast_run', 'train_rush', 'wonderland'] · EffectData: ['ed.Arm_data_corruption', 'ed.Diamond_Footpath']

**neutral prompts — selected classes only**

| arm | seen (HF) | unseen (HF) | zero-shot (HF) | all HF rows | zero-shot (ED 81 f) |
|---|---|---|---|---|---|
| base_cond | 60.6 (6) | 55.6 (25) | 52.5 (22) | 54.9 (53) | 52.3 (6) |
| ic_gen | 80.3 (6) | 69.8 (25) | 61.4 (22) | 67.5 (53) | 55.6 (6) |
| dualforce control | 94.7 (6) | 89.3 (25) | 78.9 (22) | 85.6 (53) | 78.4 (6) |
| dualforce DCG w=6 | 99.2 (6) | 94.2 (25) | 92.2 (22) | 93.9 (53) | 82.8 (6) |

**effect prompts — selected classes only**

| arm | seen (HF) | unseen (HF) | zero-shot (HF) | all HF rows | zero-shot (ED 81 f) |
|---|---|---|---|---|---|
| base_cond | 85.9 (6) | 64.4 (25) | 66.7 (22) | 67.8 (53) | 62.7 (6) |
| ic_gen | 82.1 (6) | 74.4 (25) | 68.6 (22) | 72.9 (53) | 74.6 (6) |
| dualforce control | 95.2 (6) | 92.4 (25) | 86.4 (22) | 90.2 (53) | 84.8 (6) |
| dualforce DCG w=6 | 99.7 (6) | 97.7 (25) | 93.2 (22) | 96.1 (53) | 94.8 (6) |
