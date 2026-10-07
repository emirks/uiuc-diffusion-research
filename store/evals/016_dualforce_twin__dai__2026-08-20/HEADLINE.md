# dualforce_twin_neutral — transition-eval v4 HEADLINE (2026-08-20, DeltaAI)

COUNTERFACTUAL-**TWIN TREATMENT** (rank128 one_way, warm-start ctt_v2, redirect+differential on S2
same-endpoint counterfactuals, step 1000). Instrument v4.0.0 (eval-v4-cert, reference_v4 sha256
`459fd9a7…`) · corpus 222 · τ_copy 0.858 · one machine (DeltaAI GH200). SAME instrument + corpus + warm
cache + 152-grid + prompt_sha `0d708175` as the control eval (evals/012) → directly comparable.
Pool-yardstick: raw app_ref · same-class GT ceiling · achieved-%. n = items (seed-averaged over 42/43).

| cell             | %type | n  | raw app_ref | GT ceiling | %    | note                         | (control %) |
|------------------|-------|----|-------------|------------|------|------------------------------|-------------|
| **pooled same**  | same  | 60 | 0.6957      | 0.8722     | **80.3%** | all 5 same-typed cells   | (89.6%)     |
| G-fit            | same  | 13 | 0.7256      | 0.8735     | 83.7% | heldin fit                   | (90.7%)     |
| G-unseen-same    | same  | 13 | 0.7602      | 0.8735     | 87.0% | unseen same-class            | (97.1%)     |
| G-zs-same        | same  | 8  | 0.6634      | 0.8635     | 78.3% | zero-shot same-class         | (93.7%)     |
| G-memo-probe     | same  | 13 | 0.7201      | 0.8735     | 83.4% | DIAGNOSTIC: memorization probe | (98.6%)   |
| G-ref-control    | same  | 13 | 0.5967      | 0.8735     | 68.5% | DIAGNOSTIC: mismatched-ref     | (69.3%)   |
| pooled proxy     | proxy | 92 | 0.5315      | 0.8729     | (61.2%) | cross/foreign, ranking-only | (82.2%)   |
| G-unseen-cross   | proxy | 26 | 0.6392      | 0.8735     | (72.6%) |                            | (86.7%)     |
| G-zs-cross       | proxy | 20 | 0.6307      | 0.8722     | (74.5%) |                            | (95.6%)     |
| G-unseen-foreign | proxy | 26 | 0.4612      | 0.8735     | (53.2%) |                            | (76.4%)     |
| G-zs-foreign     | proxy | 20 | 0.3839      | 0.8722     | (43.3%) |                            | (70.4%)     |

**Reference-dependence gap (primary metric):** G-fit 83.7 − G-ref-control 68.5 = **+15.2pp** (control gap **+21.4pp**) — gap NARROWED by 6.2pp; G-ref-control barely moved (−0.8pp) while G-fit fell (−7.0pp).

**Copy-guard (per-gen, n=304):** copy_max mean 0.3487 · max 0.7982 · p95 0.6011 · near_copy(≥0.858) **0/304** · core_degenerate **21/304** (control 8/304).

**Error-rows:** 36/1842 app_ref-None (control 73/1842) — twin has FEWER, comparison not voided by differential rate.

**Forward-vs-sampled:** the frozen α-probe rose on the trained checkpoint (α(0.85) 0.0164 frozen → 0.0511 @ step250), i.e. the redirect loss DID increase off-path redirect responsiveness — but that did NOT convert to a wider sampled gap; matched sampling quality fell across every same-typed cell and degeneracy rose. Forward-read↑ / sampling↓.

**Pre-registered bars (from advisor R1/R2, verdict is the advisor's call R3):** WIN needs gap ≥ +26.5pp AND pooled-same ≥ 88.6%; KILL if gap < +23.4pp guards-clean OR pooled-same < 87.6% OR core_degenerate ≥ 13/304. Measured: gap +15.2pp, pooled-same 80.3%, core_degenerate 21/304.

Per-item results: `store/evals/016_dualforce_twin__dai__2026-08-20/dualforce_twin_neutral/c{0..15}/items.jsonl` (filter `arm == "dualforce_twin_neutral"`; co-scored control_hold/control_lerp out of scope).
