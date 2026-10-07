# evals/026 — dualforce `null_contrast` (control+lose, lerp-null ref) vs `dualforce_control`
CERTIFIED co-scored A/B — all 4 arms in ONE v4 pass, seed 42, DeltaAI. **NEUTRAL numbers.**

pooled-same %

| arm | whole | seen | unseen | zs | ref-dep gap | near/core |
|---|---:|---:|---:|---:|---:|---:|
| null_contrast N | 82.3 | 88.0 | 88.6 | 93.9 | +20.6 | 0/10 |
| **control N** | 88.3 | 89.9 | 97.5 | 90.2 | +22.7 | 0/5 |
| null_contrast E | 91.2 | 96.1 | 94.8 | 98.2 | +27.6 | 0/3 |
| **control E** | 93.7 | 101.8 | 99.5 | 96.8 | +32.6 | 0/0 |

Δ (null_contrast − control), same pass = CERTIFIED paired:
- **N: whole −6.0** · seen −1.9 · unseen −8.9 · **zs +3.7**
- **E: whole −2.5** · seen −5.8 · unseen −4.7 · **zs +1.3**

Read (NEUTRAL, → owner/FINDINGS): the lerp-null contrast term did **not** beat the plain control on whole
pooled-same (−6.0 N / −2.5 E); the deficit is in seen/unseen, while **zero-shot rose** (+3.7 / +1.3).
Copy-clean (near_copy 0). Milder than 016's cross-operator contrast (−11.1). Co-scored control N=88.3
matches its published 89.6 (seeds 42/43) within seed-42-only noise ⇒ pipeline validated.
Aggregator: `misc/2026-09-02_null_contrast/build/eval/report_nullc.py`.
