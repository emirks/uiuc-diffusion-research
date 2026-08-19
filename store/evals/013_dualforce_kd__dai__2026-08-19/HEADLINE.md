# dualforce_kd_neutral — transition-eval v4 HEADLINE (2026-08-19, DeltaAI)

DUAL-FORCE **KD TREATMENT** (rank128 one_way, warm-start ctt_v2, text-crutch self-distillation, step 1000).
Instrument v4.0.0 (eval-v4-cert @ 258a990, [UNCERTIFIED] by design) · reference_v4 sha256 `459fd9a7…e606a8`
· corpus 222 (raw `5a7a8be9…` / canon-json `dc2e139a…`) · τ_copy 0.858 · one machine (DeltaAI GH200).
SAME instrument + corpus + warm cache as the control eval (evals/012) → directly comparable.
Pool-yardstick: raw app_ref · same-class GT ceiling · achieved-%. n = items (seed-averaged over 42/43).

| cell             | %type | n  | raw app_ref | GT ceiling | %    | note                         | (control %) |
|------------------|-------|----|-------------|------------|------|------------------------------|-------------|
| **pooled same**  | same  | 60 | 0.7222      | 0.8722     | **83.2%** | all 5 same-typed cells   | (89.6%)     |
| G-fit            | same  | 13 | 0.7415      | 0.8735     | 84.9% | heldin fit                   | (90.7%)     |
| G-unseen-same    | same  | 13 | 0.7929      | 0.8735     | 91.2% | unseen same-class            | (97.1%)     |
| G-zs-same        | same  | 8  | 0.7508      | 0.8635     | 86.9% | zero-shot same-class         | (93.7%)     |
| G-memo-probe     | same  | 13 | 0.7513      | 0.8735     | 87.2% | DIAGNOSTIC: memorization probe | (98.6%)   |
| G-ref-control    | same  | 13 | 0.5853      | 0.8735     | 67.3% | DIAGNOSTIC: mismatched-ref     | (69.3%)   |
| pooled proxy     | proxy | 92 | 0.5930      | 0.8729     | (68.4%) | cross/foreign, ranking-only | (82.2%)   |
| G-zs-cross       | proxy | 20 | 0.6813      | 0.8722     | (80.1%) |                            | (95.6%)     |
| G-unseen-cross   | proxy | 26 | 0.6759      | 0.8735     | (77.6%) |                            | (86.7%)     |
| G-unseen-foreign | proxy | 26 | 0.5209      | 0.8735     | (60.1%) |                            | (76.4%)     |
| G-zs-foreign     | proxy | 20 | 0.4905      | 0.8722     | (55.6%) |                            | (70.4%)     |

**Reference-dependence signal (matched − mismatched):** G-fit 84.9 − G-ref-control 67.3 = **+17.6pp** (control gap +21.4pp).

**Copy-guard (per-gen, n=304):** copy_max mean 0.3649 · max 0.7952 · p95 0.6156 · near_copy(≥0.858) **0/304** · core_degenerate **17/304** (control 8/304).

**Δpp vs base:** unpaired here (152/152 base twins registered but not co-scored into this out-root) — pair downstream.

**Caveats:** 35/1842 rows are EOFError cache-race error-rows (fewer than control's 73; cache warmed by evals/012). All 152 items still retain a pool-mean; no gen lost all refs (mean 5.94/gen, min 1). This out-root holds ONLY dualforce_kd_neutral rows (no co-scored controls).

Per-item results: `store/evals/013_dualforce_kd__dai__2026-08-19/dualforce_kd_neutral/c{0..15}/items.jsonl` (all rows are `arm == "dualforce_kd_neutral"`).
