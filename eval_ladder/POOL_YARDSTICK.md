# Pool-reference yardstick — ceilings and achieved-% (standing reporting lane)

Owner-adopted 2026-07-21 (exp_072). From now on, every cross-arm appearance
comparison is also computed and reported on this yardstick. Validation and
numbers: `docs/FINDINGS.md` F-001; machinery: `experiments/exp_072_pool_reference_rescore/`.

## Why it exists

Certified `app_ref` semantics differ per arm family — base/specialists were
scored against the item's **own GT clip** (content-matched, inflated), IC arms
against the **demo** (different clip). Cross-arm appearance deltas were therefore
apples-to-oranges. The reference-swap study showed same-class references are
interchangeable *at class level* (v4: d′ 1.71, top-1 86%, single-ref swap noise
±0.044 = 11% of range), so scoring against the class's reference **pool** gives
one uniform, low-noise setting for every arm — including the foreign tier (donor
pool; no GT needed) — with a per-class **GT ceiling** for interpretation.

## Definitions

- **Pool** per item: all same-class corpus clips, minus the item's own endpoints
  clip, minus the IC demo (so demo-copying can't inflate), deterministic first-8
  by clip name.
- **Pool score** per item: mean `app_ref` over its pool references (v4 kernel;
  rows also carry `app_ref_v3`).
- **GT ceiling** per class: same-class off-diagonal mean of the certified
  pairwise matrix (`m1a_S3` in
  `…/eval-v4-cert/outputs/eval/certification/4.0.0-draft.1/analysis/distance_matrices.npz`)
  — what a *perfect* generation scores in this exact setting.
- **Achieved-%** = pool score / class ceiling.

## Reporting rules (fixed)

1. Always report the triplet **raw · ceiling · achieved-%** (raw with % in
   parentheses, ceiling as its own column) — never % alone.
2. Statistics (deltas, sign tests, MDE) live on the **raw** scale; % is
   presentation. Pool-mean swap noise ≈ σ_ref/√m (v4 σ_ref 0.044).
3. % only for trust-map-trusted classes; camera classes stay appearance-blind.
4. \>100% is annotated "≥ceiling (prototype-mode signature)", not celebrated.
5. Foreign-tier ceilings are donor-class proxies — valid for ranking arms, not
   for absolute distance-from-perfect (margin + 2AFC carry those claims).
6. This lane is SECONDARY: it never displaces the certified margin channel or a
   pre-registered claim bar.

## How to run

```bash
python experiments/exp_072_pool_reference_rescore/build_manifests.py   # pool manifests
sbatch --export=ALL,LABEL=<label> --partition=secondary --account=campusclusterusers \
  --gres=gpu:H100:1 experiments/exp_072_pool_reference_rescore/job_score.sbatch
python experiments/exp_072_pool_reference_rescore/aggregate.py         # raw·ceiling·% table
```

Rows: `outputs/eval/exp_072_pool_v4/`. Current full table:
`outputs/reports/pool_yardstick_v4.txt`. Exact-kernel local recompute (read-only,
validated to 1e-6 against harness rows): `local_pool_pilot_v4.py`;
stability/ceiling analysis: `ref_stability.py` (same dir).

Arms the harness lane doesn't cover (r1k/r1k_ext, ic2, ckpt250 specialists —
plus r0/r3x/ic3_x while their fill jobs queue): `local_pool_fill.py` (same dir)
computes their pool means with the exact v4 kernel from cached features
(validates against harness rows first) and writes the viewer-facing index
`outputs/eval/exp_072_pool_v4/local_fill/pool_index.json`. Locally-filled
values are marked (dashed chip / `~`) until harness rows confirm them.

The ladder viewer (`docs/eval_ladder/build_viewer.py` →
`outputs/reports/ladder_viewer/index.html`) consumes this index and shows the
triplet per generation: raw pool score (details), class ceiling (family badge),
achieved-% (chip). Same reporting rules apply there (trusted classes only,
`≥ceil` annotation, `*` donor-proxy for the foreign tier).

## Current headline (2026-07-21, 2,616 pairs + fills in flight)

| % of GT ceiling | A seen | B unseen | C zero-shot† | X foreign* |
|---|---|---|---|---|
| specialist | 99% | 101% | — | 75%* |
| ic3 (IC-LoRA) | 84% | 96% | 90%† | 62%* |
| base·PE | 73% | ← | ~ | — |
| crossfade floor | 48% | | | |
| freeze floor | 22% | | | |

† low-trust holdout classes, descriptive. * single-donor-ref pending pool fill
(jobs 9609271–72; also adds prompt-only base).
