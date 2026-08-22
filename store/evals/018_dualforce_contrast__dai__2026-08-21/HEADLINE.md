# dualforce_contrast_neutral — transition-eval v4 HEADLINE (2026-08-22, DeltaAI)

CONTRASTIVE **TREATMENT** over the dual-force plain-FM control (rank128 one_way, warm-start ctt_v2@10k, + paired-preference
contrast on S0+S1 same-content pairs, β 8 λ 0.25 σ∈[0.5,0.9], ref-anchored; step 1000, effective batch 4). Instrument v4.0.0
(eval-v4-cert, reference_v4 sha `459fd9a71bb5…`) · corpus 222 · τ_copy 0.858 · one machine (DeltaAI GH200). SAME instrument + corpus +
warm cache + 152-grid + prompt_sha 0d708175 as evals/012 (control) and evals/017 (redirect) → directly comparable.
Pool-yardstick: raw app_ref · same-class GT ceiling · achieved-%. n = items (seed-averaged over 42/43).

| cell | %type | n | raw app_ref | % | (control 012 %) | (redirect 015 %) |
|---|---|---|---|---|---|---|
| **pooled same** | same | 60 | 0.6828 | **78.5%** | (89.6%) | (88.1%) |
| G-fit | same | 13 | 0.7084 | 81.5% | (90.7%) | (91.9%) |
| G-unseen-same | same | 13 | 0.7390 | 84.8% | (97.1%) | (95.1%) |
| G-zs-same | same | 8 | 0.7004 | 79.8% | (93.7%) | (96.4%) |
| G-memo-probe | same | 13 | 0.7207 | 83.5% | (98.6%) | (89.1%) |
| G-ref-control | same | 13 | 0.5525 | 63.3% | (69.3%) | (71.2%) |
| G-zs-cross | proxy | 20 | 0.6845 | 79.7% | (95.6%) | (92.0%) |
| G-unseen-cross | proxy | 26 | 0.6533 | 75.1% | (86.7%) | (85.2%) |
| G-unseen-foreign | proxy | 26 | 0.4698 | 54.7% | (76.4%) | (76.6%) |
| G-zs-foreign | proxy | 20 | 0.5030 | 55.1% | (70.4%) | (75.9%) |
| pooled proxy | proxy | 92 | 0.5755 | (66.0%) | (82.2%) | (82.2%) |

**Reference-dependence gap (G-fit − G-ref-control):** contrast **+18.1pp** · control +21.4pp · redirect +20.7pp.
**P3a swapped-compliance (G-ref-control gens vs the DEMO's class pool, raw app_ref, n=192 each):** contrast **0.6718** vs control 0.8305 (Δ -0.1586).
**Copy-guard (per-gen, n=304):** copy_max mean 0.3607 · near_copy(≥0.858) **0/304** · core_degenerate **21/304** (control 8, redirect 8).
**Error-rows:** 7/1842 app_ref-None (control 73/1842).

**Pre-registered bars (dossier §3, vs 012@1000):** WIN gap ≥ +27.4 & pooled-same ≥ 87.6 & compliance ≥ 0.831 & degen ≤ 12 & near_copy ≤ 1 & G-zs-same ≥ 92.7; KILL gap < +23.4 or pooled-same < 86.6 or degen ≥ 16 or near_copy ≥ 3 or compliance < 0.80 while gap improves; GRAY otherwise (quality-clean null). Verdict: advisor (dossier).

Per-item results: `store/evals/018_dualforce_contrast__dai__2026-08-21/dualforce_contrast_neutral/c{0..15}/items.jsonl`.
