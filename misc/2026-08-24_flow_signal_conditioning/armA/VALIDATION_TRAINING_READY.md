# VALIDATION — 44-ch DINO signal, training-readiness for `004_ctt_v2plus`

Read-only audit on `gh-login02.delta.ncsa.illinois.edu`; 46,219 consumed (stratum,stem) keys from `004_ctt_v2plus/samples.jsonl`. Bars & checks per fable-advisor 2026-08-28.

## V1 · Coverage — every training row's signal present + shape-matched (bar 100%)

| stratum | consumed keys | feat hit | shape match | gaps |
|---|--:|--:|--:|--:|
| S0 | 139 | 139 | 139 | 0 |
| S2a | 7,577 | 7,577 | 7,577 | 0 |
| S2b | 7,859 | 7,859 | 7,859 | 0 |
| S4 | 2,000 | 2,000 | 2,000 | 0 |
| S6 | 28,644 | 28,644 | 28,644 | 0 |

**V1: PASS**

## V2 · S6 full-open integrity — all 28,644 npz opened (bar: 0 defects)

opened **28,644/28,644**; corrupt 0, shape-bad 0, non-finite 0, dtype≠fp16 0, channels≠CH_NAMES 0, missing 0.  **V2: PASS**

## V3 · S6 raw-cache integrity (best-effort zip test)

ok **28,644/28,644**; unreadable 0, missing 0.  **V3: PASS**

## V4 · Frozen-PCA captured-variance share on S6 (pre-registered bars)

Baseline (fit-set EVR sum) = **0.452**. Bars: pooled ≥ 0.75×base = **0.339**, each shape ≥ 0.7×base = **0.316**.

| scope | share | ×base | bar met |
|---|--:|--:|:--:|
| pooled | 0.388 | 0.86× | ✓ |
| (11, 22, 33) | 0.382 | 0.85× | ✓ |
| (11, 22, 39) | 0.386 | 0.85× | ✓ |
| (11, 33, 22) | 0.379 | 0.84× | ✓ |
| (11, 39, 22) | 0.404 | 0.89× | ✓ |

**V4: PASS — keep frozen basis**

## V5 · S6 channel health (report-only)

u/v window-saturation (|u| or |v| ≥ 2.4, near the R=5 → 2.5-cell ceiling): **0.03%** of cells (n=200 clips).

channels with >5% exact-zero: conf 48.2%, csim_A 9.5%.

conf %exact-zero (fwd-bwd rejected cells): **48.24%**.

## V6 · Eval corpus

eval__ feat present: **223** (held-out instrument; not a training input).

## V8 · Norm-apply smoke (join + NORM v2, the contract the trainer's signal-loader will use)

applied to **604** rows across strata; S6 shapes exercised: ['(11, 22, 33)', '(11, 22, 39)', '(11, 33, 22)', '(11, 39, 22)']. non-finite 0, out-of-[-5,5] 0, unresolved 0.  **V8: PASS**

## Overall: READY — V1 ✓ · V2 ✓ · V3 ✓ · V4 ✓ · V8 ✓. Norm gates G-N1..G-N5 in NORM_REPORT_v2.md.
