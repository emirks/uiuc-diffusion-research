# dualforce_s0s1redirect_neutral — transition-eval v4 HEADLINE (2026-08-20 scoring; written 2026-08-22 by the contrastive_training close-out)

S0+S1 COUNTERFACTUAL **REDIRECT TREATMENT** (rank128 one_way, warm-start ctt_v2@10k, redirect-only on same-content×different-operator
S0+S1 cells, cond_A, σ_red∼U[0.5,0.9], seam-masked, λ_red 0.25; step 1000, effective batch 4). Instrument v4.0.0 (eval-v4-cert,
reference_v4 sha `459fd9a71bb5…`) · corpus 222 · τ_copy 0.858 · one machine (DeltaAI GH200). SAME instrument + corpus + warm cache +
152-grid + prompt_sha 0d708175 as evals/012 (control) → directly comparable. Rescored pass (array 2990974) after a raced first pass
(`dualforce_s0s1redirect_neutral__pass1_raced/`, kept as record). Pool-yardstick: raw app_ref · same-class GT ceiling · %.

| cell | %type | n | raw app_ref | % | (control 012 %) |
|---|---|---|---|---|---|
| **pooled same** | same | 60 | 0.7623 | **88.1%** | (89.6%) |
| G-fit | same | 13 | 0.7967 | 91.9% | (90.7%) |
| G-unseen-same | same | 13 | 0.8269 | 95.1% | (97.1%) |
| G-zs-same | same | 8 | 0.8198 | 96.4% | (93.7%) |
| G-memo-probe | same | 13 | 0.7662 | 89.1% | (98.6%) |
| G-ref-control | same | 13 | 0.6239 | 71.2% | (69.3%) |
| G-zs-cross | proxy | 20 | 0.7860 | 92.0% | (95.6%) |
| G-unseen-cross | proxy | 26 | 0.7393 | 85.2% | (86.7%) |
| G-unseen-foreign | proxy | 26 | 0.6649 | 76.6% | (76.4%) |
| G-zs-foreign | proxy | 20 | 0.6789 | 75.9% | (70.4%) |
| pooled proxy | proxy | 92 | 0.7153 | (82.2%) | (82.2%) |

**Reference-dependence gap:** G-fit 91.9 − G-ref-control 71.2 = **+20.7pp** (control +21.4pp).
**P3a swapped-compliance (n=192 each):** redirect **0.8246** vs control 0.8305 (misc/2026-08-20_s0s1_counterfactual/build/eval/compliance_scores).
**Copy-guard (per-gen, n=304):** copy_max mean 0.4207 · near_copy **0/304** · core_degenerate **8/304** (control 8).
**Error-rows:** 0/1842.

**Pre-registered bars (s0s1 dossier R1):** WIN gap ≥ +27.4 …; KILL gap < +23.4 … → **gap KILL with every quality bar held = QUALITY-CLEAN NULL** (the redirect changed nothing measurable). Per that dossier's R1 pre-commitment this closed the twin family; the owner knowingly reopened it once more (2026-08-21 contrastive run, evals/018 — also killed).
