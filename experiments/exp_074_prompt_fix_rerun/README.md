# exp_074 — corrected-prompt regeneration (inference-only)

## Question

How much of the foreign-transfer "failure" and the one-sided zero-shot scores
was caused by the prompt describing the outcome / contradicting the task —
rather than by the models? (Problem statement + evidence:
`docs/eval_ladder/PROMPT_REDESIGN.md`.)

## Setup

Same adapters, same conditioning, same seeds (42/43/44), same sampler settings
as the ladder originals — ONLY the prompt changes:

| rows | arms | old prompt | corrected prompt |
|---|---|---|---|
| R5 one-sided (4×3) | ic3 zero-shot | `ICTRANS S1. The scene transforms into S2.` | `ICTRANS S1. The scene transforms into` |
| R4X (44×3) | ic3 foreign | recipient's full caption (incl. ITS outcome) | `ICTRANS <recipient S1>. The scene transforms into` |
| R3X (44×3) | specialist foreign | same | same |

276 generations, twin-matched 1:1 to the originals (same id scheme, outputs
under this experiment's root). Prompt rule (owner 2026-07-22): withhold only
the OUTCOME, keep the transition marker for TRAINING ALIGNMENT — these are
inference-only reruns on models trained with the full-caption format, and a
dangling `…The scene transforms into` stays closest to the training text
statistics while removing the end-scene information. Two-sided rows are
EXCLUDED: with both anchors given, Scene 2 is endpoint knowledge, so their
corrected prompt equals the original — the existing generations are their own
twins. The transition itself must come from weights (R3X) or the in-context
reference (R4X/R5).

An earlier no-marker variant (`ICTRANS S1` alone) was cancelled before any
output was produced; a hand-drafted `manifest_*_m.json` set ("…The scene
transforms." complete-sentence variant, rung R3XM/R4XM/R5M) sits in dataset/
unlaunched — a possible marker-phrasing ablation arm.

Pre-registered predictions (2026-07-22, before scoring):
1. Foreign forensic flips toward the donor: baseline top-1 "looks-like" is
   recipient 76%/83% (r3x/ic3_x), donor only 14%/8%. Corrected prompts raise
   the donor fraction — how far is the measurement.
2. Foreign pool-% vs donor ceiling rises for both arms (was 75%*/63%*).
3. One-sided R5 appearance may DROP — that is the honest number once the text
   no longer hands over the outcome (leak removal, not regression).

## How to run

```bash
python3 experiments/exp_074_prompt_fix_rerun/build_manifests.py
sbatch --export=ALL,MANIFEST=dataset/manifest_ic3_cx.json,CHUNKS=6 --array=0-17 \
    experiments/exp_074_prompt_fix_rerun/job_gen.sbatch
for cls in <11 classes>; do sbatch --job-name=exp074_r3x_$cls \
    --export=ALL,MANIFEST=dataset/manifest_r3x_$cls.json,CHUNKS=1 --array=0-2 \
    experiments/exp_074_prompt_fix_rerun/job_gen.sbatch; done
```

Scoring (after gens): v4 instrument (eval-v4-cert worktree), out-root
`outputs/eval/exp_074_prompt_fix_v4`; pool yardstick per POOL_YARDSTICK.md
(donor pools for foreign); forensic re-run via apps_top3.

## Outputs

`outputs/videos/exp_074_prompt_fix_rerun/{R5,R4X,R3X}/<id>__s<seed>.mp4`
(2026-07-22: jobs 9617329–9617340, 51 array tasks, secondary/backfill-sized).
