# ic_gen_effect + neutral baselines — v4-LANE table, scored on DeltaAI (GH200)

Instrument: `eval-v4-cert` @ `258a990`, `reference_v4.npz` `459fd9a7…e606a8` (file sha, echoed by all 108 task verify blocks), corpus `dc2e139a…` (222 clips). τ_copy = 0.858. Controls excluded everywhere. Ceilings from `distance_matrices.npz` `[m1a_S3]`.

Three arms of the metric_eval 2x2 (adapter × text):
- `ic_gen_effect` — TREATMENT: the ic_gen IC-LoRA adapter (runs/001@5000) + effect-in-prompt (leaky convention). Reads the reference.
- `base_cond_neutral` — baseline: base weights, V-neutral prompt, endpoint anchors, NO demo. Specificity-zero anchor.
- `base_prompt_neutral` — baseline: base weights, V-neutral prompt, NO conditioning, NO demo. Cleanest zero.

## 1 · Coverage

| arm | planned items | scored (registry-joined) | rows written | control twins (dropped) | app_ref null | error rows | shards |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ic_gen_effect` | 1842 | 1842 | 3684 | 1842 | 0 | 0 | 36 |
| `base_cond_neutral` | 1842 | 1842 | 3684 | 1842 | 0 | 0 | 36 |
| `base_prompt_neutral` | 1842 | 1842 | 1842 | 0 | 0 | 0 | 36 |

All 108 task logs echoed `reference_v4 : 459fd9a7…e606a8`; all rc=0; zero tracebacks. Manifests disjoint by generation (no cache write race).

## 2 · Pool-%  (app_ref pool mean ÷ GT-class ceiling)

Per-item: per-seed pool mean over the GT-pool references, then mean over seeds, divided by that class's ceiling. `%_same` cells are cross-class comparable and headline-eligible; `(%_proxy)` cells are content-capped, ranking-only — NEVER blended with `%_same`.

| cell | %type | n items | `ic_gen_effect` | `base_cond_neutral` | `base_prompt_neutral` |
|---|---|---:|---:|---:|---:|
| G-fit | same | 13 | 96.9% | 53.3% | 59.5% |
| G-memo-probe | same | 13 | 94.1% | 62.5% | 60.5% |
| G-ref-control | same | 13 | 68.5% | 66.8% | 60.2% |
| G-unseen-same | same | 13 | 97.1% | 67.5% | 60.2% |
| G-zs-same | same | 8 | 88.4% | 46.6% | 45.0% |
| G-unseen-cross | proxy | 26 | (93.1%) | (52.9%) | (51.8%) |
| G-unseen-foreign | proxy | 26 | (81.1%) | (42.3%) | (43.0%) |
| G-zs-cross | proxy | 20 | (92.4%) | (53.2%) | (49.3%) |
| G-zs-foreign | proxy | 20 | (69.3%) | (39.1%) | (34.2%) |
| **ALL(same)** | same | 60 | **89.1%** | **60.4%** | **58.1%** |
| **ALL(proxy)** | proxy | 92 | (84.4%) | (47.0%) | (44.9%) |

### Raw · ceiling (the numerator/denominator behind ALL)

| arm | ALL(same) raw | ceiling | ALL(same) % | ALL(proxy) raw | ceiling | ALL(proxy) % |
|---|---:|---:|---:|---:|---:|---:|
| `ic_gen_effect` | 0.7755 | 0.8722 | 89.1% | 0.7364 | 0.8729 | (84.4%) |
| `base_cond_neutral` | 0.5256 | 0.8722 | 60.4% | 0.4032 | 0.8729 | (47.0%) |
| `base_prompt_neutral` | 0.5078 | 0.8722 | 58.1% | 0.3907 | 0.8729 | (44.9%) |

`%_same` is a LEVEL, not a pass/fail bar. ic_gen_effect (89.1%) sits between plain `ic_gen` (83.1%, evals/001) and `ctt_v2_leaky` (91.3%) as its leaky-prompt convention predicts; the no-demo neutral baselines (60.4% / 58.1%) are the specificity floor. Paired Δpp for ic_gen_effect is NOT computed here — its base twins are not scored in this entry.
