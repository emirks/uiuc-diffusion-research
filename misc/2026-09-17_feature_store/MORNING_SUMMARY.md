# Morning summary — grid-v3 metric tables (built overnight 2026-09-18)

**Deliverable:** `papers_drafts/_preview/metrics_gridv3/` — `tab_A.tex`, `tab_B.tex`, `tab_C.tex` (paper style, booktabs, `\ph{}`), `TABLES.md` (same numbers + n + caveats), `preview.pdf`.
Regenerate any time: `/taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/ltx2/bin/python scripts/build_metric_tables.py --strict` (exits 0 now). Ozgur's `tables/*.tex` untouched.

## What every column is (all fixed by me; every one recomputed or reproduced independently)
| column | definition | verified how |
|---|---|---|
| Transport | capped pooled-% (S3 / app_ref ÷ certified class ceiling ×100, cap 100), mean over GENERATIONS | per_gen re-aggregates to evals/028 summary.json exactly; 183-triple slice = draft numbers |
| Identity A/B | DINO cos between the hand-off window (first/last K=fps/3 generated frames) and the last given frame (A) / first given suffix frame (B) | hand recompute 3/3 to 1e-7 |
| Motion A/B | velocity continuity across the hand-off from the gen's own CoTracker tracks (weighted per-tracklet cosine of mean velocity in the last 3 given steps vs first K generated steps; NaN for frame-anchored rows) | hand recompute 4/4 to 1e-7 |
| Seam-free % | share with no temporal-LPIPS spike at the hand-off (z ≤ 3) | from evals/028/030 rows (bit-identical to the paper's) |
| Smooth / Ref sim. (VP) / Motion fid. | the competitor lenses through the store (score_batch --store, impl 8a808635) | Table B cells reproduce the paper's rows_v3 means to 3 dp, all 6 arms |
| Copy % / Copy max | M2a vs the generation's OWN reference (τ = 0.858) | hand recompute 10/10 exact; the pool-scored copy in evals/028 was found INVALID (identical across arms) and replaced |
| Aesthetic | LAION head on CLIP-L/14 per-frame features, mean | shapes verified; comparable across arms (256-px frames → not VBench-absolute) |

## Readings (levels, single seeds ×2; no verdicts)
- Transport: SEGUE leads every tier (97.1 / 92.1 / 93.7) and the shared zero-shot set (92.6 vs VFXMaster 85.8, VAP 81.1, refVFX 62.2).
- The cost side of guidance is now visible: Identity A 0.926 → 0.867 (zs), seam-free 97.1 → 94.1, smoothness 0.986 → 0.982, copy max 0.405 → 0.472 (near-copy still ≤ 1.4 %).
- The no-reference base row calibrates the endpoint columns: best identity/smoothness (a model that barely moves), worst seam-free in zero-shot (64.7).
- Prior works keep the input best (Id A refVFX 0.971, VFXMaster 0.964) and score higher on aesthetic (refVFX 5.31); SEGUE leads transfer on all three instruments (transport, VP ref-sim 0.965, motion fid. 0.226).
- Text dependency Δ (transport): base +42.6, baseline LoRA +26.6, SEGUE w/o guidance +5.1, SEGUE +1.9; on VP ref-sim +0.015 / +0.008 / +0.002 / +0.001.

## Caveats to carry into the paper
- refVFX zero-shot transport is **62.2** (S3, the metric everything else uses); the draft's 75.5 is the Look_u column — fix tab_main.
- Table B copy rates for the externals are under-estimated (max over 49/33 frames vs our 121); disclosed under the table.
- Motion B has a selection effect (arms whose effects engulf the frame lose visible tracklets → smaller n); Motion fid. is NaN where nothing moves (n per cell). Seen-tier B-side cells have n < 10 → shown as n/a.
- Table C has no NEUTRAL rows for the externals on grid v3 (their neutral arms exist only on the old 112-row grid).
- Table D (w-sweep) not built: w = 1/1.5/3 arms do not exist on grid v3. VLM judge / human study not run. Bar-8 (4.0.1 certification) deferred per owner.
- Zero-shot n differs by design: Table A uses all 442 zs generations (HF one+two-sided + ED); Tables B/C the 366-gen one-sided set the prior works can produce.

## Provenance
Store evals: 038_handoff_gridv3, 039_copy_gridv3, 040_lenses_gridv3 (meta.yaml + INDEX lines committed; rows are store artifacts). Feature store: every grid-v3 generation now carries dino_cls, cotracker3, lpips_t, clip_b32, clip_l14 (+ videoprism/raft_mag where the lens pass touched them). GPU used tonight ≈ 18.8 GPU-h (bgjg + bhwp). Full trail: `NIGHT_LOG.md`, `OP2..OP6_REPORT.md`, branch `feature-store`.
