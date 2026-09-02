# DINO-signal A1 channels-target — transition-eval v4 HEADLINE (2026-09-02, DeltaAI)

FULL PRODUCTION grid, **MATCHED case only**: reference (pixel demo) + per-row DINO signal (A1, channels route) +
prompt, single **seed 42**, all **152 rows**, both prompts **neutral** (family 001) + **effect** (family 002). A1
through the SAME `LTX-2-armA` fork (@32d6e3f) and inference path as evals/021 (A2/A0) and evals/022 (A5), rank/alpha
**128**, step **10000**:
- **A1_channels_target** — route=channels (SPEC §2): the per-row appearance-free 44-ch DINO signal is projected to the
  model inner dim by a **Linear(44, inner_dim)** and **ADDED onto the target token embeddings** (placement=target,
  token_pool 2, signal-guidance 1.0, hidden 256, channels 44, **1 `signal.proj.weight` key loaded strict**, norm
  db47be88, inner_dim 4096); pixel ref KEPT. dataset **005_ctt_v2plus_s6reshape** (SAME as A2 and A5 ⇒ A1-vs-A2-vs-A5
  is a CLEAN same-dataset ROUTE comparison).

Instrument v4.0.0 (eval-v4-cert, reference_v4 sha `459fd9a71bb5…`, [UNCERTIFIED] by design) · corpus 222 · τ_copy 0.858 ·
one machine (DeltaAI GH200, torch 2.10.0+cu129). Same instrument + corpus + warm cache + 152-grid + prompt_sha
(neutral 0d708175, effect 35930d7d) as evals/022 (A5), 021 (A2/A0), 001 (ctt_v2), 012 (control), dcg/flowsig →
same-machine comparable (cross-pass caveat). Pool-yardstick: raw app_ref · same-class GT ceiling · achieved-% (via
`eval/report_dino_a1.py`). Numbers NEUTRAL — nothing settled.

## Achieved-% per cell (seed 42; comparators = evals/022 A5, evals/021 A2/A0, evals/001 ctt_v2)

| cell | %type | A1 neutral | A1 effect | (A5 neutral) | (A5 effect) | (A2 neutral) | (A0 neutral) | (A2 effect) | (A0 effect) | (ctt_v2) |
|---|---|---|---|---|---|---|---|---|---|---|
| **pooled same** | same | **84.8%** | **89.6%** | (84.9%) | (89.8%) | (86.0%) | (84.6%) | (90.1%) | (86.8%) | (82.5%) |
| G-fit | same | 92.9% | 95.8% | (90.3%) | (95.0%) | (96.0%) | (96.6%) | (96.3%) | (94.3%) | (90.6%) |
| G-unseen-same | same | 91.1% | 100.4% | (99.5%) | (101.4%) | (93.2%) | (91.8%) | (97.7%) | (95.3%) | (90.5%) |
| G-zs-same | same | 81.5% | 86.5% | (82.0%) | (92.2%) | (87.5%) | (85.7%) | (82.8%) | (75.7%) | (74.9%) |
| G-memo-probe | same | 93.8% | 101.9% | (87.6%) | (99.2%) | (91.0%) | (86.7%) | (101.6%) | (99.9%) | (84.3%) |
| G-ref-control | same | 63.4% | 62.1% | (63.9%) | (62.3%) | (63.0%) | (62.5%) | (69.3%) | (64.5%) | (69.2%) |
| G-zs-cross | proxy | 78.2% | 95.8% | (83.9%) | (96.6%) | (76.0%) | (89.4%) | (94.6%) | (92.3%) | (78.5%) |
| G-unseen-cross | proxy | 81.5% | 95.6% | (79.8%) | (89.3%) | (79.6%) | (83.7%) | (88.4%) | (93.3%) | (74.1%) |
| G-unseen-foreign | proxy | 66.0% | 86.8% | (65.0%) | (80.9%) | (66.1%) | (70.9%) | (84.7%) | (85.1%) | (59.8%) |
| G-zs-foreign | proxy | 55.4% | 72.9% | (47.7%) | (66.0%) | (54.0%) | (59.6%) | (72.5%) | (73.3%) | (54.0%) |
| pooled proxy | proxy | 70.7% | 88.2% | (69.5%) | (83.5%) | (69.4%) | (76.1%) | (85.3%) | (86.4%) | (66.7%) |

Foreign-tier %s are donor-class **proxies** — ranking-only, never blended with %_same.

## Reference-dependence gap (G-fit − G-ref-control)
A1 neutral **+29.5pp** · A1 effect **+33.7pp** (A5n +26.4 · A5e +32.7 · A2n +33.0 · A2e +27.1 · A0n +34.1 · A0e +29.8 · ctt_v2 +21.3).

## Seen / unseen / zero-shot same-content ladder (G-fit / G-unseen-same / G-zs-same)
A1 neutral **92.9 / 91.1 / 81.5** · A1 effect **95.8 / 100.4 / 86.5**.

## Copy-guard (per-gen, n=152/arm, seed 42)
| | A1 neutral | A1 effect |
|---|---|---|
| copy_max mean | 0.3915 | 0.4499 |
| near_copy (≥0.858) | 0/152 | 0/152 |
| core_degenerate | 10/152 | 5/152 |

Error-rows (app_ref=None on a minority of pool-reference comparisons, EOFError/BadZipFile at decode; every one of the
152 gens still landed a pooled score): a1n 30, a1e 20 (of 921 pool-rows/arm). NOT re-scored (matches the evals/021-022 pattern).

## No verdict — bars NOT pre-set for this production case
- **A1-vs-A2-vs-A5 is a CLEAN same-dataset (005) ROUTE comparison** (channels vs tokens vs signal-as-Q xattn):
  pooled-same neutral A1 84.8 · A2 86.0 · A5 84.9; effect A1 89.6 · A2 90.1 · A5 89.8. A1-vs-A0 stays
  dataset-confounded (A0 on 004). Cross-pass to evals/022/021 + ctt_v2/dcg/flowsig is same-machine + same instrument
  sha (459fd9a7) but a different scoring pass (cross-pass caveat).
- The claim-bearing **within-arm** contrast (A1-matched vs A1-shufsignal) is NOT part of this matched-only production
  pass (no A1 shufsig registry stamped yet).
- A **bar-setting consult** is owed before interpreting these numbers.

Per-item results: `store/evals/023_dino_a1__dai__2026-09-02/<arm>/c{0..15}/items.jsonl` (gitignored on disk).
