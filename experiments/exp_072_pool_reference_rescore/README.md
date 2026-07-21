# exp_072 — pool-reference re-score (the common yardstick lane)

## Question

Certified `app_ref` semantics differ per arm family (base/specialists: reference =
the item's own GT clip; IC arms: reference = the in-context demo), so cross-arm
appearance comparisons are apples-to-oranges. The reference-swap study
(2026-07-20, certification distance matrices) showed the m1a kernel recognizes a
class from any same-class reference (top-1 84%, gap 0.20, d′ 1.52) but a single
reference carries ±0.079 judge noise → the valid common yardstick is the **pool
mean**: score each generation against every same-class corpus reference
(leave-own-clip-out; leave-demo-out for IC arms) and average. This experiment
re-scores existing ladder generations on that yardstick. **No generation, no
training — manifest-only re-scoring; features come from the warm certified cache.**

## Pre-registration (declared before any pool row is scored)

- **Lane status: SECONDARY / descriptive.** Certified per-arm rows and every
  pre-registered claim keep their original semantics; this lane never displaces a
  headline. Pool scores are read as % of the per-class GT ceiling (same-class
  off-diagonal mean of the certified m1a__v3_sided matrix,
  `outputs/eval/certification/3.0.0-draft.8/analysis/distance_matrices.npz`).
- **Predictions:** (1) specialist R3 (unseen) lands near ic3_b ≈ 100% of ceiling;
  (2) base·PE lands below both on trained classes; (3) specialist R2 (seen) ≥ R3
  (memorization); (4) arm ranking on the pool lane agrees with certified margin
  ranking within classes (both are class-relative channels).
- **Rules:** pool = all same-class corpus clips minus the item's own endpoints
  clip minus the IC demo, deterministic first-8 by clip name; per-item pool score
  = mean over refs; per-class per-arm = mean over items; MDE for pool means uses
  σ_ref/√m (v3 σ_ref 0.079). Classes with <3 refs reported with n, never
  headline. Camera classes stay appearance-blind (trust map governs, as always).

## Setup

- Items: existing exp_066 eval manifests — arms r1 (150), r2_ckpt2000+r3_ckpt2000
  (132), ic3_a/b/c (165). Same generated videos, new `reference_video` per row;
  `item_id` suffixed `__ref_<clip>`.
- Scoring: **v4.0.0 instrument** (eval-v4-cert worktree; owner directive 2026-07-20:
  v4 is the lane instrument — better d' 1.71 vs 1.52, ref-swap noise 11% vs 30% of
  range). Rows also carry the `app_ref_v3` bridge field (byte-identical to certified
  v3 app_ref), so v3 continuity comes free. H100 lane, shared warm cache; no --training.
  First submission (9603416–22, v3 worktree) was canceled before running.
- PILOT (first, small): classes portal / shadow_smoke / super_fast_run, all five
  arms, ≤4 refs per item — a fast preview of the cross-arm picture.

## How to run

```bash
python experiments/exp_072_pool_reference_rescore/build_manifests.py
# pilot then full chunks (secondary H100):
sbatch --export=ALL,LABEL=pool_pilot --partition=secondary --account=campusclusterusers \
  --gres=gpu:H100:1 experiments/exp_072_pool_reference_rescore/job_score.sbatch
for L in pool_c0 pool_c1 pool_c2 pool_c3 pool_c4 pool_c5; do
  sbatch --export=ALL,LABEL=$L --partition=secondary --account=campusclusterusers \
    --gres=gpu:H100:1 experiments/exp_072_pool_reference_rescore/job_score.sbatch; done
# aggregate:
python experiments/exp_072_pool_reference_rescore/aggregate.py
```

## Outputs

`outputs/eval/exp_072_pool/<label>/items.jsonl` (one row per (generation,
pool-reference) pair) + `aggregate.py` report: per arm × tier × class pool-mean
app_ref, % of GT ceiling, n_items × n_refs.
