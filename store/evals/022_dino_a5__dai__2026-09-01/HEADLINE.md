# DINO-signal A5 xattn-fusion — transition-eval v4 HEADLINE (2026-09-01, DeltaAI)

FULL PRODUCTION grid, **MATCHED case only**: reference (pixel demo) + per-row DINO signal (A5, signal-as-Q
cross-attention) + prompt, single **seed 42**, all **152 rows**, both prompts **neutral** (family 001) + **effect**
(family 002). A5 through the SAME `LTX-2-armA` fork (@32d6e3f) and inference path as evals/021 (A2/A0), rank/alpha
**128**, step **10000**:
- **A5_xattn_fusion** — route=xattn (SPEC R1.4): the pooled per-row DINO signal is the **QUERY**, the clean reference
  latent block is the **K/V**; the fused bank tokens append at the target block (placement=target, token_pool 2,
  signal-guidance 1.0, xattn.hidden 256 / xattn.heads 4, 13 `signal.*` keys loaded strict, norm db47be88); pixel ref KEPT.
  dataset **005_ctt_v2plus_s6reshape** (SAME as A2 ⇒ A5-vs-A2 is NOT dataset-confounded).

Instrument v4.0.0 (eval-v4-cert, reference_v4 sha `459fd9a71bb5…`, [UNCERTIFIED] by design) · corpus 222 · τ_copy 0.858 ·
one machine (DeltaAI GH200, torch 2.10.0+cu129). Same instrument + corpus + warm cache + 152-grid + prompt_sha
(neutral 0d708175, effect 35930d7d) as evals/021 (A2/A0), 001 (ctt_v2), 012 (control), dcg/flowsig → same-machine
comparable (cross-pass caveat). Pool-yardstick: raw app_ref · same-class GT ceiling · achieved-% (via
`eval/report_dino_a5.py`). Numbers NEUTRAL — nothing settled.

## Achieved-% per cell (seed 42; comparators = evals/021 A2/A0, evals/001 ctt_v2, evals/019 control 012)

| cell | %type | A5 neutral | A5 effect | (A2 neutral) | (A0 neutral) | (A2 effect) | (A0 effect) | (ctt_v2) | (control 012) |
|---|---|---|---|---|---|---|---|---|---|
| **pooled same** | same | **84.9%** | **89.8%** | (86.0%) | (84.6%) | (90.1%) | (86.8%) | (82.5%) | (89.6%) |
| G-fit | same | 90.3% | 95.0% | (96.0%) | (96.6%) | (96.3%) | (94.3%) | (90.6%) | (90.7%) |
| G-unseen-same | same | 99.5% | 101.4% | (93.2%) | (91.8%) | (97.7%) | (95.3%) | (90.5%) | (97.1%) |
| G-zs-same | same | 82.0% | 92.2% | (87.5%) | (85.7%) | (82.8%) | (75.7%) | (74.9%) | (93.7%) |
| G-memo-probe | same | 87.6% | 99.2% | (91.0%) | (86.7%) | (101.6%) | (99.9%) | (84.3%) | (98.6%) |
| G-ref-control | same | 63.9% | 62.3% | (63.0%) | (62.5%) | (69.3%) | (64.5%) | (69.2%) | (69.3%) |
| G-zs-cross | proxy | 83.9% | 96.6% | (76.0%) | (89.4%) | (94.6%) | (92.3%) | (78.5%) | (95.6%) |
| G-unseen-cross | proxy | 79.8% | 89.3% | (79.6%) | (83.7%) | (88.4%) | (93.3%) | (74.1%) | (86.7%) |
| G-unseen-foreign | proxy | 65.0% | 80.9% | (66.1%) | (70.9%) | (84.7%) | (85.1%) | (59.8%) | (76.4%) |
| G-zs-foreign | proxy | 47.7% | 66.0% | (54.0%) | (59.6%) | (72.5%) | (73.3%) | (54.0%) | (70.4%) |
| pooled proxy | proxy | 69.5% | 83.5% | (69.4%) | (76.1%) | (85.3%) | (86.4%) | (66.7%) | (82.2%) |

Foreign-tier %s are donor-class **proxies** — ranking-only, never blended with %_same.

## Reference-dependence gap (G-fit − G-ref-control)
A5 neutral **+26.4pp** · A5 effect **+32.7pp** (A2n +33.0 · A2e +27.1 · A0n +34.1 · A0e +29.8 · ctt_v2 +21.3 · control 012 +21.4).

## Seen / unseen / zero-shot same-content ladder (G-fit / G-unseen-same / G-zs-same)
A5 neutral **90.3 / 99.5 / 82.0** · A5 effect **95.0 / 101.4 / 92.2**.

## Copy-guard (per-gen, n=152/arm, seed 42)
| | A5 neutral | A5 effect |
|---|---|---|
| copy_max mean | 0.3798 | 0.4406 |
| near_copy (≥0.858) | 0/152 | 0/152 |
| core_degenerate | 10/152 | 7/152 |

Error-rows (app_ref=None on a minority of pool-reference comparisons, EOFError/BadZipFile at decode; every one of the
152 gens still landed a pooled score): a5n 18, a5e 15 (of 921 pool-rows/arm). NOT re-scored (matches the evals/021 pattern).

## No verdict — bars NOT pre-set for this production case
- **A5-vs-A2 is NOT dataset-confounded** (both on 005). Cross-pass to evals/021 (A2/A0) + ctt_v2/dcg/flowsig is
  same-machine + same instrument sha (459fd9a7) but a different scoring pass (cross-pass caveat).
- The claim-bearing **within-arm** contrast (A5-matched vs A5-shufsignal) is NOT part of this matched-only production
  pass (no A5 shufsig registry stamped yet).
- A **bar-setting consult** is owed before interpreting these numbers.

Per-item results: `store/evals/022_dino_a5__dai__2026-09-01/<arm>/c{0..15}/items.jsonl` (gitignored on disk).
