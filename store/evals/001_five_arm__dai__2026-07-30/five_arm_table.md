# refVFX baseline — 5-ARM v4 table, every arm scored on DeltaAI (GH200)

Instrument: `eval-v4-cert` @ `258a990`, `reference_v4.npz` `459fd9a7…e606a8`, corpus `dc2e139a…` (222 clips). τ_copy = 0.858. Controls excluded everywhere.

## 1 · Coverage

| arm | generation frames | planned | scored | error rows | UNSCORED |
|---|---:|---:|---:|---:|---:|
| `refvfx_A` | 33f | 1842 | 1842 | 0 | 0 |
| `refvfx_B` | 33f | 1842 | 1842 | 0 | 0 |
| `ic_gen` | 121f | 1842 | 1842 | 0 | 0 |
| `ctt_v2` | 121f | 1842 | 1842 | 0 | 0 |
| `ctt_v2_leaky` | 121f | 1842 | 1842 | 0 | 0 |

✅ zero error rows, zero unscored planned items, across all four arms.

### `core_degenerate` rate per GENERATION (304 per arm) — 🚩 **mask geometry, NOT model behaviour**

| arm | frames | window (one / two-sided) | core_deg gens | rate | one-sided | two-sided |
|---|---:|---|---:|---:|---:|---:|
| `refvfx_A` | 33f | 24 / 16 | 158/304 | **0.520** | 99/224 | 59/80 |
| `refvfx_B` | 33f | 24 / 16 | 104/304 | **0.342** | 25/224 | 79/80 |
| `ic_gen` | 121f | 112 / 104 | 16/304 | **0.053** | 0/224 | 16/80 |
| `ctt_v2` | 121f | 112 / 104 | 18/304 | **0.059** | 0/224 | 18/80 |
| `ctt_v2_leaky` | 121f | 112 / 104 | 10/304 | **0.033** | 0/224 | 10/80 |

> **Geometry caveat — `core_degenerate` is NOT like-for-like.** `core_mask_v3` sets the flag when the strict core mask holds `< FALLBACK_MIN_FRAMES = 8` frames — an ABSOLUTE count. The refVFX arms are **33-frame** generations, so the un-conditioned window is 24 frames (one-sided) or 16 (two-sided); ours are **121-frame**, window 112 / 104. The same flag therefore means '≥8 of 16-24' for refVFX and '≥8 of 104-112' for ours. `copy_max`'s `mid_mask` has the same asymmetry (52-73 % of a refVFX clip retained vs 86-93 % of ours). Read `core_deg` WITHIN an arm-geometry group, never across.


## 2 · Raw metrics

### Per cell (all 9)

| group | n | arm | margin | app_ref | app_target | copy_max | near_copy@0.858 | core_deg† |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| G-fit | 182 | refvfx_A | 0.0858 | 0.3369 | 0.3784 | 0.2374 | 0 | 99 |
| G-fit | 182 | refvfx_B | 0.1274 | 0.2574 | 0.3853 | 0.1847 | 0 | 56 |
| G-fit | 182 | ic_gen | 0.1437 | 0.7499 | 0.4546 | 0.3160 | 0 | 8 |
| G-fit | 182 | ctt_v2 | 0.1646 | 0.7862 | 0.4633 | 0.3091 | 0 | 0 |
| G-fit | 182 | ctt_v2_leaky | 0.1481 | 0.8462 | 0.4823 | 0.3423 | 0 | 0 |
| G-memo-probe | 182 | refvfx_A | 0.1144 | 0.3744 | 0.4237 | 0.2550 | 0 | 90 |
| G-memo-probe | 182 | refvfx_B | 0.1180 | 0.2904 | 0.4122 | 0.2233 | 0 | 38 |
| G-memo-probe | 182 | ic_gen | 0.1483 | 0.7310 | 0.4814 | 0.3236 | 0 | 0 |
| G-memo-probe | 182 | ctt_v2 | 0.1361 | 0.7332 | 0.4916 | 0.3327 | 0 | 8 |
| G-memo-probe | 182 | ctt_v2_leaky | 0.1586 | 0.8555 | 0.5040 | 0.3751 | 0 | 0 |
| G-ref-control | 194 | refvfx_A | -0.0141 | 0.2711 | 0.3399 | 0.2319 | 0 | 100 |
| G-ref-control | 194 | refvfx_B | 0.0506 | 0.2989 | 0.3698 | 0.2161 | 0 | 47 |
| G-ref-control | 194 | ic_gen | 0.0395 | 0.5991 | 0.3941 | 0.2655 | 0 | 0 |
| G-ref-control | 194 | ctt_v2 | -0.0305 | 0.6173 | 0.3478 | 0.2718 | 0 | 0 |
| G-ref-control | 194 | ctt_v2_leaky | -0.1045 | 0.6259 | 0.3174 | 0.2734 | 0 | 0 |
| G-unseen-same | 182 | refvfx_A | 0.0870 | 0.3956 | 0.4089 | 0.2634 | 0 | 99 |
| G-unseen-same | 182 | refvfx_B | 0.0504 | 0.3018 | 0.3647 | 0.2270 | 0 | 67 |
| G-unseen-same | 182 | ic_gen | 0.1349 | 0.7708 | 0.4707 | 0.3207 | 0 | 8 |
| G-unseen-same | 182 | ctt_v2 | 0.1296 | 0.7914 | 0.4739 | 0.3205 | 0 | 0 |
| G-unseen-same | 182 | ctt_v2_leaky | 0.1455 | 0.8569 | 0.5067 | 0.3613 | 0 | 0 |
| G-unseen-cross | 388 | refvfx_A | -0.1514 | 0.3455 | 0.2337 | 0.2318 | 0 | 212 |
| G-unseen-cross | 388 | refvfx_B | -0.2391 | 0.2198 | 0.1597 | 0.1739 | 0 | 88 |
| G-unseen-cross | 388 | ic_gen | -0.1794 | 0.6325 | 0.2401 | 0.2433 | 0 | 16 |
| G-unseen-cross | 388 | ctt_v2 | -0.1258 | 0.6411 | 0.2631 | 0.2624 | 0 | 16 |
| G-unseen-cross | 388 | ctt_v2_leaky | -0.0321 | 0.8183 | 0.3406 | 0.3240 | 0 | 0 |
| G-unseen-foreign | 388 | refvfx_A | -0.0705 | 0.3325 | 0.1921 | 0.2002 | 0 | 191 |
| G-unseen-foreign | 388 | refvfx_B | -0.1128 | 0.2463 | 0.1074 | 0.1017 | 0 | 87 |
| G-unseen-foreign | 388 | ic_gen | -0.0868 | 0.4817 | 0.1524 | 0.1509 | 0 | 24 |
| G-unseen-foreign | 388 | ctt_v2 | -0.0560 | 0.5082 | 0.1808 | 0.1741 | 0 | 32 |
| G-unseen-foreign | 388 | ctt_v2_leaky | 0.0093 | 0.7373 | 0.2875 | 0.2655 | 0 | 8 |
| G-zs-same | 46 | refvfx_A | 0.1186 | 0.4989 | 0.4342 | 0.3239 | 0 | 21 |
| G-zs-same | 46 | refvfx_B | 0.0456 | 0.2275 | 0.3553 | 0.2187 | 0 | 31 |
| G-zs-same | 46 | ic_gen | 0.0336 | 0.7741 | 0.4230 | 0.3049 | 0 | 0 |
| G-zs-same | 46 | ctt_v2 | 0.0540 | 0.6792 | 0.4018 | 0.3049 | 0 | 4 |
| G-zs-same | 46 | ctt_v2_leaky | 0.1775 | 0.8509 | 0.5427 | 0.4501 | 0 | 4 |
| G-zs-cross | 140 | refvfx_A | -0.1299 | 0.3857 | 0.2684 | 0.2989 | 0 | 98 |
| G-zs-cross | 140 | refvfx_B | -0.2185 | 0.2596 | 0.1804 | 0.2194 | 0 | 72 |
| G-zs-cross | 140 | ic_gen | -0.1656 | 0.6369 | 0.2498 | 0.2941 | 0 | 8 |
| G-zs-cross | 140 | ctt_v2 | -0.1008 | 0.6970 | 0.2956 | 0.3274 | 0 | 0 |
| G-zs-cross | 140 | ctt_v2_leaky | 0.0770 | 0.8253 | 0.4360 | 0.4514 | 0 | 0 |
| G-zs-foreign | 140 | refvfx_A | -0.0615 | 0.4326 | 0.2004 | 0.2038 | 0 | 61 |
| G-zs-foreign | 140 | refvfx_B | -0.1216 | 0.2012 | 0.0913 | 0.1081 | 0 | 69 |
| G-zs-foreign | 140 | ic_gen | -0.0928 | 0.4056 | 0.1272 | 0.1411 | 0 | 19 |
| G-zs-foreign | 140 | ctt_v2 | -0.0805 | 0.5179 | 0.1922 | 0.1945 | 0 | 25 |
| G-zs-foreign | 140 | ctt_v2_leaky | 0.0637 | 0.6958 | 0.3384 | 0.3393 | 0 | 21 |
| **ALL** | 1842 | refvfx_A | -0.0315 | 0.3554 | 0.2916 | 0.2364 | 0 | 971 |
| **ALL** | 1842 | refvfx_B | -0.0643 | 0.2543 | 0.2396 | 0.1739 | 0 | 555 |
| **ALL** | 1842 | ic_gen | -0.0285 | 0.6188 | 0.3024 | 0.2466 | 0 | 83 |
| **ALL** | 1842 | ctt_v2 | -0.0114 | 0.6447 | 0.3184 | 0.2629 | 0 | 85 |
| **ALL** | 1842 | ctt_v2_leaky | 0.0440 | 0.7833 | 0.3857 | 0.3309 | 0 | 33 |

† `core_deg` is **NOT comparable across arms** — see the geometry note.

### By sidedness

| group | n | arm | margin | app_ref | app_target | copy_max | near_copy@0.858 | core_deg† |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| one | 1438 | refvfx_A | -0.0262 | 0.3638 | 0.2995 | 0.2357 | 0 | 659 |
| one | 1438 | refvfx_B | -0.0579 | 0.2592 | 0.2482 | 0.1581 | 0 | 159 |
| one | 1438 | ic_gen | -0.0289 | 0.6203 | 0.3026 | 0.2343 | 0 | 0 |
| one | 1438 | ctt_v2 | -0.0112 | 0.6385 | 0.3166 | 0.2533 | 0 | 0 |
| one | 1438 | ctt_v2_leaky | 0.0462 | 0.7814 | 0.3860 | 0.3308 | 0 | 0 |
| two | 404 | refvfx_A | -0.0502 | 0.3254 | 0.2636 | 0.2388 | 0 | 312 |
| two | 404 | refvfx_B | -0.0870 | 0.2370 | 0.2089 | 0.2303 | 0 | 396 |
| two | 404 | ic_gen | -0.0273 | 0.6138 | 0.3016 | 0.2902 | 0 | 83 |
| two | 404 | ctt_v2 | -0.0123 | 0.6669 | 0.3250 | 0.2974 | 0 | 85 |
| two | 404 | ctt_v2_leaky | 0.0363 | 0.7900 | 0.3844 | 0.3311 | 0 | 33 |
| **ALL** | 1842 | refvfx_A | -0.0315 | 0.3554 | 0.2916 | 0.2364 | 0 | 971 |
| **ALL** | 1842 | refvfx_B | -0.0643 | 0.2543 | 0.2396 | 0.1739 | 0 | 555 |
| **ALL** | 1842 | ic_gen | -0.0285 | 0.6188 | 0.3024 | 0.2466 | 0 | 83 |
| **ALL** | 1842 | ctt_v2 | -0.0114 | 0.6447 | 0.3184 | 0.2629 | 0 | 85 |
| **ALL** | 1842 | ctt_v2_leaky | 0.0440 | 0.7833 | 0.3857 | 0.3309 | 0 | 33 |

† `core_deg` is **NOT comparable across arms** — see the geometry note.

### By %-type (NEVER blended: `proxy` is content-capped, ranking-only)

| group | n | arm | margin | app_ref | app_target | copy_max | near_copy@0.858 | core_deg† |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| same | 786 | refvfx_A | 0.0699 | 0.3524 | 0.3897 | 0.2512 | 0 | 409 |
| same | 786 | refvfx_B | 0.0836 | 0.2838 | 0.3812 | 0.2132 | 0 | 239 |
| same | 786 | ic_gen | 0.1106 | 0.7146 | 0.4477 | 0.3057 | 0 | 16 |
| same | 786 | ctt_v2 | 0.0953 | 0.7272 | 0.4402 | 0.3078 | 0 | 12 |
| same | 786 | ctt_v2_leaky | 0.0893 | 0.7967 | 0.4558 | 0.3436 | 0 | 4 |
| proxy | 1056 | refvfx_A | -0.1069 | 0.3576 | 0.2186 | 0.2254 | 0 | 562 |
| proxy | 1056 | refvfx_B | -0.1744 | 0.2324 | 0.1342 | 0.1447 | 0 | 316 |
| proxy | 1056 | ic_gen | -0.1321 | 0.5476 | 0.1942 | 0.2025 | 0 | 67 |
| proxy | 1056 | ctt_v2 | -0.0908 | 0.5833 | 0.2278 | 0.2296 | 0 | 73 |
| proxy | 1056 | ctt_v2_leaky | 0.0103 | 0.7732 | 0.3335 | 0.3214 | 0 | 29 |
| **ALL** | 1842 | refvfx_A | -0.0315 | 0.3554 | 0.2916 | 0.2364 | 0 | 971 |
| **ALL** | 1842 | refvfx_B | -0.0643 | 0.2543 | 0.2396 | 0.1739 | 0 | 555 |
| **ALL** | 1842 | ic_gen | -0.0285 | 0.6188 | 0.3024 | 0.2466 | 0 | 83 |
| **ALL** | 1842 | ctt_v2 | -0.0114 | 0.6447 | 0.3184 | 0.2629 | 0 | 85 |
| **ALL** | 1842 | ctt_v2_leaky | 0.0440 | 0.7833 | 0.3857 | 0.3309 | 0 | 33 |

† `core_deg` is **NOT comparable across arms** — see the geometry note.

## 3 · Pool-%

### Pool-% — raw · ceiling · %  (app_ref pool mean ÷ GT-class ceiling)

Per-item: per-seed pool mean over the GT-pool references, then mean over seeds, divided by that class's ceiling. Controls excluded. `%_same` cells are cross-class comparable; `(%_proxy)` cells are content-capped, ranking-only.

| cell | %type | n items | arm | raw app_ref | ceiling | % of ceiling |
|---|---|---:|---|---:|---:|---:|
| G-fit | same | 13 | refvfx_A | 0.3463 | 0.8735 | 40.1% |
| G-fit | same | 13 | refvfx_B | 0.2584 | 0.8735 | 29.4% |
| G-fit | same | 13 | ic_gen | 0.7522 | 0.8735 | 86.3% |
| G-fit | same | 13 | ctt_v2 | 0.7861 | 0.8735 | 90.6% |
| G-fit | same | 13 | ctt_v2_leaky | 0.8425 | 0.8735 | 97.3% |
| G-memo-probe | same | 13 | refvfx_A | 0.3653 | 0.8735 | 43.8% |
| G-memo-probe | same | 13 | refvfx_B | 0.2838 | 0.8735 | 33.6% |
| G-memo-probe | same | 13 | ic_gen | 0.7212 | 0.8735 | 83.4% |
| G-memo-probe | same | 13 | ctt_v2 | 0.7275 | 0.8735 | 84.3% |
| G-memo-probe | same | 13 | ctt_v2_leaky | 0.8507 | 0.8735 | 98.5% |
| G-ref-control | same | 13 | refvfx_A | 0.2721 | 0.8735 | 31.7% |
| G-ref-control | same | 13 | refvfx_B | 0.3021 | 0.8735 | 34.7% |
| G-ref-control | same | 13 | ic_gen | 0.6005 | 0.8735 | 68.8% |
| G-ref-control | same | 13 | ctt_v2 | 0.6078 | 0.8735 | 69.2% |
| G-ref-control | same | 13 | ctt_v2_leaky | 0.6155 | 0.8735 | 70.9% |
| G-unseen-same | same | 13 | refvfx_A | 0.4207 | 0.8735 | 47.8% |
| G-unseen-same | same | 13 | refvfx_B | 0.3042 | 0.8735 | 35.0% |
| G-unseen-same | same | 13 | ic_gen | 0.7737 | 0.8735 | 89.1% |
| G-unseen-same | same | 13 | ctt_v2 | 0.7912 | 0.8735 | 90.5% |
| G-unseen-same | same | 13 | ctt_v2_leaky | 0.8586 | 0.8735 | 98.3% |
| G-unseen-cross | proxy | 26 | refvfx_A | 0.3528 | 0.8735 | (40.4%) |
| G-unseen-cross | proxy | 26 | refvfx_B | 0.2191 | 0.8735 | (25.2%) |
| G-unseen-cross | proxy | 26 | ic_gen | 0.6318 | 0.8735 | (73.1%) |
| G-unseen-cross | proxy | 26 | ctt_v2 | 0.6481 | 0.8735 | (74.1%) |
| G-unseen-cross | proxy | 26 | ctt_v2_leaky | 0.8215 | 0.8735 | (94.3%) |
| G-unseen-foreign | proxy | 26 | refvfx_A | 0.3425 | 0.8735 | (38.7%) |
| G-unseen-foreign | proxy | 26 | refvfx_B | 0.2484 | 0.8735 | (28.7%) |
| G-unseen-foreign | proxy | 26 | ic_gen | 0.4852 | 0.8735 | (56.9%) |
| G-unseen-foreign | proxy | 26 | ctt_v2 | 0.5113 | 0.8735 | (59.8%) |
| G-unseen-foreign | proxy | 26 | ctt_v2_leaky | 0.7459 | 0.8735 | (86.1%) |
| G-zs-same | same | 8 | refvfx_A | 0.4822 | 0.8635 | 52.3% |
| G-zs-same | same | 8 | refvfx_B | 0.2546 | 0.8635 | 31.6% |
| G-zs-same | same | 8 | ic_gen | 0.7541 | 0.8635 | 91.1% |
| G-zs-same | same | 8 | ctt_v2 | 0.6510 | 0.8635 | 74.9% |
| G-zs-same | same | 8 | ctt_v2_leaky | 0.8027 | 0.8635 | 91.6% |
| G-zs-cross | proxy | 20 | refvfx_A | 0.3722 | 0.8722 | (41.8%) |
| G-zs-cross | proxy | 20 | refvfx_B | 0.2515 | 0.8722 | (28.5%) |
| G-zs-cross | proxy | 20 | ic_gen | 0.6050 | 0.8722 | (72.2%) |
| G-zs-cross | proxy | 20 | ctt_v2 | 0.6671 | 0.8722 | (78.5%) |
| G-zs-cross | proxy | 20 | ctt_v2_leaky | 0.7948 | 0.8722 | (91.0%) |
| G-zs-foreign | proxy | 20 | refvfx_A | 0.3943 | 0.8722 | (45.3%) |
| G-zs-foreign | proxy | 20 | refvfx_B | 0.1976 | 0.8722 | (24.3%) |
| G-zs-foreign | proxy | 20 | ic_gen | 0.3829 | 0.8722 | (44.1%) |
| G-zs-foreign | proxy | 20 | ctt_v2 | 0.4840 | 0.8722 | (54.0%) |
| G-zs-foreign | proxy | 20 | ctt_v2_leaky | 0.6530 | 0.8722 | (71.9%) |
| **ALL(same)** | same | 60 | refvfx_A | 0.3686 | 0.8722 | 42.4% |
| **ALL(same)** | same | 60 | refvfx_B | 0.2828 | 0.8722 | 33.0% |
| **ALL(same)** | same | 60 | ic_gen | 0.7175 | 0.8722 | 83.1% |
| **ALL(same)** | same | 60 | ctt_v2 | 0.7179 | 0.8722 | 82.5% |
| **ALL(same)** | same | 60 | ctt_v2_leaky | 0.7933 | 0.8722 | 91.3% |
| **ALL(proxy)** | proxy | 92 | refvfx_A | 0.3631 | 0.8729 | (41.3%) |
| **ALL(proxy)** | proxy | 92 | refvfx_B | 0.2297 | 0.8729 | (26.7%) |
| **ALL(proxy)** | proxy | 92 | ic_gen | 0.5304 | 0.8729 | (62.0%) |
| **ALL(proxy)** | proxy | 92 | ctt_v2 | 0.5779 | 0.8729 | (66.7%) |
| **ALL(proxy)** | proxy | 92 | ctt_v2_leaky | 0.7577 | 0.8729 | (86.4%) |

## 4 · refVFX A − B paired contrasts

### ONE-SIDED only (pure leak removal) — refvfx_A − refvfx_B, paired per row (n=1438)

| group | n | d margin | d app_ref | d app_target | d copy_max |
|---|---:|---:|---:|---:|---:|
| G-fit | 150 | -0.0515 | +0.0991 | -0.0090 | +0.0638 |
| G-memo-probe | 150 | -0.0119 | +0.1013 | +0.0133 | +0.0498 |
| G-ref-control | 162 | -0.0702 | -0.0279 | -0.0305 | +0.0292 |
| G-unseen-same | 150 | +0.0320 | +0.1232 | +0.0417 | +0.0586 |
| G-unseen-cross | 324 | +0.1013 | +0.1430 | +0.0856 | +0.0758 |
| G-unseen-foreign | 324 | +0.0387 | +0.0587 | +0.0742 | +0.1013 |
| G-zs-same | 26 | +0.0766 | +0.3343 | +0.0808 | +0.1552 |
| G-zs-cross | 76 | +0.1022 | +0.1294 | +0.1008 | +0.1143 |
| G-zs-foreign | 76 | +0.0862 | +0.2969 | +0.1350 | +0.1447 |
| **ALL** | 1438 | +0.0317 | +0.1046 | +0.0513 | +0.0777 |

### TWO-SIDED only (NOT subtractive: arm A carries no S2) — refvfx_A − refvfx_B, paired per row (n=404)

| group | n | d margin | d app_ref | d app_target | d copy_max |
|---|---:|---:|---:|---:|---:|
| G-fit | 32 | +0.0047 | -0.0124 | +0.0031 | +0.0007 |
| G-memo-probe | 32 | +0.0351 | +0.0032 | +0.0030 | -0.0535 |
| G-ref-control | 32 | -0.0373 | -0.0272 | -0.0275 | -0.0523 |
| G-unseen-same | 32 | +0.0579 | -0.0438 | +0.0559 | -0.0671 |
| G-unseen-cross | 64 | +0.0186 | +0.0376 | +0.0148 | -0.0324 |
| G-unseen-foreign | 64 | +0.0606 | +0.2253 | +0.1375 | +0.0837 |
| G-zs-same | 20 | +0.0683 | +0.1896 | +0.0766 | +0.0401 |
| G-zs-cross | 64 | +0.0723 | +0.1222 | +0.0730 | +0.0381 |
| G-zs-foreign | 64 | +0.0290 | +0.1536 | +0.0782 | +0.0376 |
| **ALL** | 404 | +0.0368 | +0.0884 | +0.0546 | +0.0085 |

## 5 · paired contrasts (ARM-FREE key: cell|endpoint|reference|seed|pool_ref)

### ctt_v2 − refvfx_A — ctt_v2 − refvfx_A, paired per row (n=1842)

| group | n | d margin | d app_ref | d app_target | d copy_max |
|---|---:|---:|---:|---:|---:|
| G-fit | 182 | +0.0789 | +0.4494 | +0.0849 | +0.0717 |
| G-memo-probe | 182 | +0.0216 | +0.3587 | +0.0679 | +0.0777 |
| G-ref-control | 194 | -0.0164 | +0.3462 | +0.0080 | +0.0399 |
| G-unseen-same | 182 | +0.0427 | +0.3958 | +0.0650 | +0.0571 |
| G-unseen-cross | 388 | +0.0256 | +0.2956 | +0.0295 | +0.0305 |
| G-unseen-foreign | 388 | +0.0145 | +0.1757 | -0.0113 | -0.0261 |
| G-zs-same | 46 | -0.0646 | +0.1803 | -0.0324 | -0.0190 |
| G-zs-cross | 140 | +0.0291 | +0.3112 | +0.0272 | +0.0285 |
| G-zs-foreign | 140 | -0.0190 | +0.0853 | -0.0082 | -0.0093 |
| **ALL** | 1842 | +0.0200 | +0.2893 | +0.0268 | +0.0265 |

### ctt_v2 − refvfx_B — ctt_v2 − refvfx_B, paired per row (n=1842)

| group | n | d margin | d app_ref | d app_target | d copy_max |
|---|---:|---:|---:|---:|---:|
| G-fit | 182 | +0.0373 | +0.5289 | +0.0780 | +0.1244 |
| G-memo-probe | 182 | +0.0180 | +0.4428 | +0.0794 | +0.1094 |
| G-ref-control | 194 | -0.0811 | +0.3184 | -0.0220 | +0.0557 |
| G-unseen-same | 182 | +0.0793 | +0.4896 | +0.1092 | +0.0935 |
| G-unseen-cross | 388 | +0.1133 | +0.4213 | +0.1034 | +0.0885 |
| G-unseen-foreign | 388 | +0.0568 | +0.2619 | +0.0734 | +0.0724 |
| G-zs-same | 46 | +0.0084 | +0.4517 | +0.0466 | +0.0862 |
| G-zs-cross | 140 | +0.1176 | +0.4373 | +0.1152 | +0.1080 |
| G-zs-foreign | 140 | +0.0411 | +0.3166 | +0.1009 | +0.0865 |
| **ALL** | 1842 | +0.0529 | +0.3904 | +0.0789 | +0.0890 |

### ic_gen − refvfx_A — ic_gen − refvfx_A, paired per row (n=1842)

| group | n | d margin | d app_ref | d app_target | d copy_max |
|---|---:|---:|---:|---:|---:|
| G-fit | 182 | +0.0579 | +0.4130 | +0.0762 | +0.0786 |
| G-memo-probe | 182 | +0.0339 | +0.3566 | +0.0577 | +0.0686 |
| G-ref-control | 194 | +0.0536 | +0.3280 | +0.0543 | +0.0336 |
| G-unseen-same | 182 | +0.0479 | +0.3751 | +0.0618 | +0.0573 |
| G-unseen-cross | 388 | -0.0280 | +0.2871 | +0.0065 | +0.0115 |
| G-unseen-foreign | 388 | -0.0163 | +0.1492 | -0.0397 | -0.0492 |
| G-zs-same | 46 | -0.0850 | +0.2752 | -0.0113 | -0.0190 |
| G-zs-cross | 140 | -0.0357 | +0.2512 | -0.0186 | -0.0048 |
| G-zs-foreign | 140 | -0.0313 | -0.0271 | -0.0732 | -0.0628 |
| **ALL** | 1842 | +0.0029 | +0.2635 | +0.0108 | +0.0102 |

### ic_gen − refvfx_B — ic_gen − refvfx_B, paired per row (n=1842)

| group | n | d margin | d app_ref | d app_target | d copy_max |
|---|---:|---:|---:|---:|---:|
| G-fit | 182 | +0.0163 | +0.4925 | +0.0694 | +0.1313 |
| G-memo-probe | 182 | +0.0303 | +0.4406 | +0.0692 | +0.1002 |
| G-ref-control | 194 | -0.0112 | +0.3002 | +0.0243 | +0.0494 |
| G-unseen-same | 182 | +0.0845 | +0.4690 | +0.1060 | +0.0937 |
| G-unseen-cross | 388 | +0.0597 | +0.4127 | +0.0804 | +0.0694 |
| G-unseen-foreign | 388 | +0.0260 | +0.2354 | +0.0449 | +0.0492 |
| G-zs-same | 46 | -0.0119 | +0.5466 | +0.0677 | +0.0862 |
| G-zs-cross | 140 | +0.0529 | +0.3773 | +0.0695 | +0.0747 |
| G-zs-foreign | 140 | +0.0288 | +0.2043 | +0.0359 | +0.0330 |
| **ALL** | 1842 | +0.0357 | +0.3645 | +0.0628 | +0.0727 |

### ctt_v2_leaky − ctt_v2 — ctt_v2_leaky − ctt_v2, paired per row (n=1842)

| group | n | d margin | d app_ref | d app_target | d copy_max |
|---|---:|---:|---:|---:|---:|
| G-fit | 182 | -0.0165 | +0.0600 | +0.0190 | +0.0332 |
| G-memo-probe | 182 | +0.0226 | +0.1223 | +0.0124 | +0.0424 |
| G-ref-control | 194 | -0.0740 | +0.0086 | -0.0304 | +0.0015 |
| G-unseen-same | 182 | +0.0158 | +0.0655 | +0.0328 | +0.0408 |
| G-unseen-cross | 388 | +0.0937 | +0.1772 | +0.0775 | +0.0617 |
| G-unseen-foreign | 388 | +0.0653 | +0.2291 | +0.1067 | +0.0914 |
| G-zs-same | 46 | +0.1235 | +0.1717 | +0.1409 | +0.1452 |
| G-zs-cross | 140 | +0.1778 | +0.1283 | +0.1404 | +0.1239 |
| G-zs-foreign | 140 | +0.1442 | +0.1779 | +0.1463 | +0.1448 |
| **ALL** | 1842 | +0.0554 | +0.1385 | +0.0672 | +0.0680 |

### ctt_v2_leaky − ctt_v2, ONE-SIDED only — ctt_v2_leaky − ctt_v2, paired per row (n=1438)

| group | n | d margin | d app_ref | d app_target | d copy_max |
|---|---:|---:|---:|---:|---:|
| G-fit | 150 | -0.0283 | +0.0419 | +0.0129 | +0.0301 |
| G-memo-probe | 150 | +0.0292 | +0.1137 | +0.0127 | +0.0444 |
| G-ref-control | 162 | -0.0679 | +0.0299 | -0.0261 | +0.0057 |
| G-unseen-same | 150 | +0.0212 | +0.0538 | +0.0305 | +0.0467 |
| G-unseen-cross | 324 | +0.0924 | +0.1603 | +0.0724 | +0.0638 |
| G-unseen-foreign | 324 | +0.0597 | +0.2230 | +0.1036 | +0.0938 |
| G-zs-same | 26 | +0.2226 | +0.2867 | +0.2446 | +0.2471 |
| G-zs-cross | 76 | +0.2552 | +0.1866 | +0.2067 | +0.2079 |
| G-zs-foreign | 76 | +0.2065 | +0.3079 | +0.2181 | +0.2521 |
| **ALL** | 1438 | +0.0573 | +0.1429 | +0.0694 | +0.0776 |

### ctt_v2_leaky − ctt_v2, TWO-SIDED only — ctt_v2_leaky − ctt_v2, paired per row (n=404)

| group | n | d margin | d app_ref | d app_target | d copy_max |
|---|---:|---:|---:|---:|---:|
| G-fit | 32 | +0.0387 | +0.1449 | +0.0477 | +0.0480 |
| G-memo-probe | 32 | -0.0083 | +0.1629 | +0.0112 | +0.0329 |
| G-ref-control | 32 | -0.1045 | -0.0992 | -0.0525 | -0.0197 |
| G-unseen-same | 32 | -0.0092 | +0.1200 | +0.0434 | +0.0131 |
| G-unseen-cross | 64 | +0.1002 | +0.2629 | +0.1031 | +0.0508 |
| G-unseen-foreign | 64 | +0.0938 | +0.2599 | +0.1224 | +0.0793 |
| G-zs-same | 20 | -0.0053 | +0.0223 | +0.0061 | +0.0128 |
| G-zs-cross | 64 | +0.0859 | +0.0591 | +0.0617 | +0.0242 |
| G-zs-foreign | 64 | +0.0702 | +0.0236 | +0.0609 | +0.0173 |
| **ALL** | 404 | +0.0486 | +0.1230 | +0.0594 | +0.0337 |

---

## eps ↔ DeltaAI machine term on `ctt_v2` (DIAGNOSTIC, not a gate)

* eps successfully-scored `ctt_v2` item rows: **1841** (2 incremental passes, 4 eps error rows)
* DeltaAI rows: **1842** · **joined per item_id: 1841**
* 🔴 scored on DeltaAI but absent from eps (1): `G-unseen-cross__ctt_v2__color_rain_2__ref_animalization_0__s43__ref_animalization_3`
* eps error row (pre-existing, cache-write race): `G-unseen-cross__ctt_v2__color_rain_2__ref_animalization_0__s43__ref_animalization_3` — EOFError: No data left in file
* eps error row (pre-existing, cache-write race): `G-zs-cross__ctt_v2__money_rain_3__ref_live_concert_1__s43__ref_live_concert_4` — EOFError: No data left in file
* eps error row (pre-existing, cache-write race): `G-zs-foreign__ctt_v2__davis_mallard_water__ref_luminous_gaze_0__s43__ref_luminous_gaze_2` — EOFError: No data left in file
* eps error row (pre-existing, cache-write race): `G-unseen-same__ctt_v2__earth_element_6__ref_earth_element_4__s42__ref_earth_element_3` — EOFError: No data left in file

> Pre-registered bar (PROBE.md §5): **PASS iff max |Δ| < 0.005 on every metric AND zero gate flips** (a gate flip = `copy_max` crossing τ_copy = 0.858).

| metric | n | mean \|Δ\| | max \|Δ\| | rows ≥ bar | signed mean Δ | verdict |
|---|---:|---:|---:|---:|---:|---|
| `margin` | 1841 | 2.027e-03 | **1.928e-02** | 112 | +4.071e-04 | **FAIL** |
| `app_ref` | 1841 | 1.062e-02 | **1.846e-01** | 1002 | +8.452e-05 | **FAIL** |
| `app_target` | 1841 | 1.334e-03 | **2.447e-02** | 54 | +1.920e-04 | **FAIL** |
| `copy_max` | 1841 | 2.840e-03 | **8.066e-02** | 256 | +7.011e-05 | **FAIL** |

* **gate flips at τ_copy = 0.858: 0** (rows ≥ τ — eps 0, DeltaAI 0)
* max `copy_max` over the joined rows — eps 0.8010, DeltaAI 0.7988; min distance to τ = 0.0570
* categorical/flag disagreements: `core_degenerate` 8, `tier` 0, `near_copy` 0, `cross_high` 0, `app_saturated` 2, `cam_valid` 7, `sidedness` 0, `core_mode` 0, `intruder` 30
* worst row per metric: `margin` |Δ|=1.928e-02 on `G-unseen-cross__ctt_v2__earth_wave_3__ref_hero_flight_5__s42__ref_hero_flight_0`; `app_ref` |Δ|=1.846e-01 on `G-unseen-foreign__ctt_v2__davis_hike__ref_shadow_10__s42__ref_shadow_13`; `app_target` |Δ|=2.447e-02 on `G-zs-same__ctt_v2__melt_transition_2__ref_melt_transition_1__s42__ref_melt_transition_3`; `copy_max` |Δ|=8.066e-02 on `G-ref-control__ctt_v2__shadow_smoke_7__ref_hero_flight_5__s43__ref_shadow_smoke_0`

**Bar verdict: FAIL** — reported as a diagnostic; every number in the four-arm table above is DeltaAI-scored, so the machine term cancels by construction.
