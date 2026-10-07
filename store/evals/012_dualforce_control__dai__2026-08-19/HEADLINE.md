# dualforce_control_neutral — transition-eval v4 HEADLINE (2026-08-19, DeltaAI)

Instrument v4.0.0 (eval-v4-cert @ 258a990, [UNCERTIFIED] by design) · reference_v4 sha256 `459fd9a7…e606a8`
· corpus 222 (raw `5a7a8be9…` / canon-json `dc2e139a…`) · τ_copy 0.858 · one machine (DeltaAI GH200).
Pool-yardstick: raw app_ref · same-class GT ceiling · achieved-%. n = items (seed-averaged over 42/43).

| cell             | %type | n  | raw app_ref | GT ceiling | %    | note                         |
|------------------|-------|----|-------------|------------|------|------------------------------|
| **pooled same**  | same  | 60 | 0.7757      | 0.8722     | **89.6%** | all 5 same-typed cells   |
| G-fit            | same  | 13 | 0.7899      | 0.8735     | 90.7% | heldin fit                   |
| G-unseen-same    | same  | 13 | 0.8452      | 0.8735     | 97.1% | unseen same-class            |
| G-zs-same        | same  | 8  | 0.8026      | 0.8635     | 93.7% | zero-shot same-class         |
| G-memo-probe     | same  | 13 | 0.8477      | 0.8735     | 98.6% | DIAGNOSTIC: memorization probe |
| G-ref-control    | same  | 13 | 0.6034      | 0.8735     | 69.3% | DIAGNOSTIC: mismatched-ref     |
| pooled proxy     | proxy | 92 | 0.7145      | 0.8729     | (82.2%) | cross/foreign, ranking-only |
| G-zs-cross       | proxy | 20 | 0.8212      | 0.8722     | (95.6%) |                            |
| G-unseen-cross   | proxy | 26 | 0.7554      | 0.8735     | (86.7%) |                            |
| G-unseen-foreign | proxy | 26 | 0.6615      | 0.8735     | (76.4%) |                            |
| G-zs-foreign     | proxy | 20 | 0.6237      | 0.8722     | (70.4%) |                            |

**Copy-guard (per-gen, n=304):** copy_max mean 0.4113 · max 0.8141 · p95 0.6450 · near_copy(≥0.858) **0/304** · core_degenerate **8/304**.

**Δpp vs base:** unpaired here (152/152 base twins registered but not co-scored into this out-root) — pair downstream.

**Caveats:** 73/1842 rows are EOFError cache-race error-rows (all 152 items still retain a pool-mean; means over surviving refs, ~5.8/gen). Shards c0-c15 also hold co-scored non-model controls control_hold (1322) / control_lerp (364) — out of scope, auto-filtered.

Per-item results: `store/evals/012_dualforce_control__dai__2026-08-19/dualforce_control_neutral/c{0..15}/items.jsonl` (filter `arm == "dualforce_control_neutral"`).
