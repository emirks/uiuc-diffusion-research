# VALIDATION — 44-ch DINO signal, training-readiness for `005_ctt_v2plus_s6reshape`

Read-only audit on `gh-login02.delta.ncsa.illinois.edu`; 47,352 consumed (stratum,stem) keys from `005_ctt_v2plus_s6reshape/samples.jsonl`; signal root `/taiga/illinois/eng/cs/jrehg/users/emirkisa/cache/armA_signals_005/feat`; S6 keyed to the r832 ROSTER.

## V1 · Coverage — every training row's signal present + shape-matched (bar 100%)

| stratum | consumed keys | feat hit | shape match | gaps |
|---|--:|--:|--:|--:|
| S0 | 139 | 139 | 139 | 0 |
| S1 | 1,225 | 1,225 | 1,225 | 0 |
| S2a | 7,577 | 7,577 | 7,577 | 0 |
| S2b | 7,859 | 7,859 | 7,859 | 0 |
| S4 | 2,000 | 2,000 | 2,000 | 0 |
| S6 | 28,552 | 28,552 | 28,552 | 0 |

**V1: PASS**

## V6 · Eval corpus

eval__ feat present: **223** (bar 223) ✓

## V8 · Norm-apply smoke (join + NORM_dino_v4, the trainer's signal-loader contract)

norm = `NORM_dino_v4.json`. Applied to **800** rows (bar ≥600); S6 grids exercised {'[11, 16, 26]': 276, '[11, 26, 16]': 276} (bar ≥100 each). non-finite 0, out-of-[-5,5] 0, unresolved 0. **V8: PASS**

## V2–V5 · Round-3 artifacts (NOT re-run here)

These passed in Round 3 (advisor-verified 2026-08-30); cited by path:

- **V2 S6 full-open + V-verify** (28,644 feat set-equal, census 14,523/14,121, shape/chan/finite 0, 0 .tmp): `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-08-30_s6_reshape/r3/../STATUS.md` + verify job 3049773 log.
- **V4 PCA health** (pooled 0.3744 ≥0.339; 4 native 0.3703–0.3781 ≥0.316; SECONDARY paired ratio 0.9694): `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-08-30_s6_reshape/r3/health_full.json`.
- **Determinism** (24/24 bitwise): `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/misc/2026-08-30_s6_reshape/r3/determinism.json`.
- **NORM_dino_v4 gates G-N1..G-N5 PASS**, S6 131,074,944 cells, non-S6 moments==v3: `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/store/datasets/003_dino_signals/NORM_REPORT_v4.md`.

## Overall: READY — V1 ✓ · V6 ✓ · V8 ✓ (V2–V5 cited from Round 3).
