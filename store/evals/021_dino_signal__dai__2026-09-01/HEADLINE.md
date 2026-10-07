# DINO-signal production arms — transition-eval v4 HEADLINE (2026-09-01, DeltaAI)

FULL PRODUCTION grid, **MATCHED case only** (owner FINAL 2026-09-01): reference (pixel demo) + per-row DINO
signal (A2 only) + prompt, single **seed 42**, all **152 rows**, both prompts **neutral** (family 001) + **effect**
(family 002). Two arms through the SAME `LTX-2-armA` fork (@32d6e3f), rank/alpha **128**, step **10000**:
- **A2_tokens** — DINO feature-flow appended as 2×-pooled sequence tokens at the target block (route=tokens,
  placement=target, token_pool 2, signal-guidance 1.0, learned type-emb, norm db47be88); pixel ref KEPT. dataset **005_ctt_v2plus_s6reshape**.
- **A0_baseline** — reference-only IC-LoRA, **NO signal port**. dataset **004_ctt_v2plus** (005 rerun deferred).

Instrument v4.0.0 (eval-v4-cert, reference_v4 sha `459fd9a71bb5…`, [UNCERTIFIED] by design) · corpus 222 · τ_copy 0.858 ·
one machine (DeltaAI GH200, torch 2.10.0+cu129). Same instrument + corpus + warm cache + 152-grid + prompt_sha
(neutral 0d708175, effect 35930d7d) as evals/001 (ctt_v2), 012 (control), dcg/flowsig → same-machine comparable (cross-pass caveat).
Pool-yardstick: raw app_ref · same-class GT ceiling · achieved-% (via `eval/report_dino.py`). Numbers NEUTRAL — nothing settled.

## Achieved-% per cell (seed 42; comparators = neutral)

| cell | %type | A2 neutral | A0 neutral | A2 effect | A0 effect | (ctt_v2) | (control 012) |
|---|---|---|---|---|---|---|---|
| **pooled same** | same | **86.0%** | **84.6%** | **90.1%** | **86.8%** | (82.5%) | (89.6%) |
| G-fit | same | 96.0% | 96.6% | 96.3% | 94.3% | (90.6%) | (90.7%) |
| G-unseen-same | same | 93.2% | 91.8% | 97.7% | 95.3% | (90.5%) | (97.1%) |
| G-zs-same | same | 87.5% | 85.7% | 82.8% | 75.7% | (74.9%) | (93.7%) |
| G-memo-probe | same | 91.0% | 86.7% | 101.6% | 99.9% | (84.3%) | (98.6%) |
| G-ref-control | same | 63.0% | 62.5% | 69.3% | 64.5% | (69.2%) | (69.3%) |
| G-zs-cross | proxy | 76.0% | 89.4% | 94.6% | 92.3% | (78.5%) | (95.6%) |
| G-unseen-cross | proxy | 79.6% | 83.7% | 88.4% | 93.3% | (74.1%) | (86.7%) |
| G-unseen-foreign | proxy | 66.1% | 70.9% | 84.7% | 85.1% | (59.8%) | (76.4%) |
| G-zs-foreign | proxy | 54.0% | 59.6% | 72.5% | 73.3% | (54.0%) | (70.4%) |
| pooled proxy | proxy | 69.4% | 76.1% | 85.3% | 86.4% | (66.7%) | (82.2%) |

Foreign-tier %s are donor-class **proxies** — ranking-only, never blended with %_same.

## Reference-dependence gap (G-fit − G-ref-control)
A2 neutral **+33.0pp** · A0 neutral +34.1pp · A2 effect +27.1pp · A0 effect +29.8pp (ctt_v2 +21.3 · control 012 +21.4).

## Copy-guard (per-gen, n=152/arm, seed 42)
| | A2 neutral | A0 neutral | A2 effect | A0 effect |
|---|---|---|---|---|
| copy_max mean | 0.3785 | 0.3964 | 0.4385 | 0.4419 |
| near_copy (≥0.858) | 0/152 | 1/152 | 0/152 | 0/152 |
| core_degenerate | 10/152 | 7/152 | 6/152 | 6/152 |

Error-rows (app_ref=None on a minority of pool-reference comparisons, EOFError/BadZipFile at decode; every one of the
152 gens still landed a pooled score): a2n 16, a2e 25, a0n 36, a0e 36 (of 921 pool-rows/arm). NOT re-scored (owner directive).

## No verdict — bars NOT pre-set for this production case
- **A2-vs-A0 is DATASET-CONFOUNDED** (A0 on 004, A2 on 005) → secondary yardstick only; any "signal beats baseline"
  reading is provisional until an A0-005 rerun.
- The claim-bearing **within-arm** contrast (A2-matched vs A2-shufsignal) is NOT part of this production pass; the
  shufsig registry is stamped and ready if the owner wants it run.
- A **bar-setting consult** is owed before interpreting these numbers.

Per-item results: `store/evals/021_dino_signal__dai__2026-09-01/<arm>/c{0..15}/items.jsonl` (gitignored on disk).
