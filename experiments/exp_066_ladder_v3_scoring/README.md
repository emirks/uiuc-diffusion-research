# exp_066 — ladder v3 scoring (certified harness, all arms)

## Question
Score every ladder-v3 arm with the certified transition-eval harness so the
tier table and contrasts C1–C11 can be computed. No new instrument decisions:
every convention below is the pre-registered one.

## Setup
- **Code**: `eval-v3-spec` worktree @ 26e6e72 — verified byte-identical to tag
  `eval/v3.0.0` outside `*.md` (docs-only diff); amendment records 1–3 in-tree
  so the versioning stamp sees the corrected-corpus authorization.
- **Corpus**: MAIN's corrected `corpus_manifest.json` (222 clips, amendments
  2–3, sha 5a7a8be9…). cwd = MAIN so relative `data/` paths resolve.
- **Manifests**: `build_eval_manifests.py` → `dataset/eval_*.json`,
  1,142 rows total. Reference conventions (pre-registered):
  base/specialist arms → own GT clip; ic tiers A/B/C → the demo reference fed
  to the model; X arms → recipient's grid reference (identical across twins).
  Conditions point at full endpoint clips (score.py slices 9/8).
- **Chunking**: class-hash keeps twin pairs co-resident; ≤120 rows/chunk fits
  1h59 with `--controls auto`.
- **Training manifest**: `training_manifest_ic3.json` (151 clips / 403 pairs,
  from exp_064 pairs.json) — pass `TRAIN=training_manifest_ic3` for ic3 arms
  (tier joins + M2c copy-vs-training). ic2 arms scored WITHOUT a training
  manifest (frozen comparison arm; its training clipset predates the split
  alignment — noted, never headlines). Specialist training manifests: not
  built tonight; M2c-vs-training for r3x runs when they are (open item).

## How to run
```bash
P="--partition=secondary --account=campusclusterusers --gres=gpu:H100:1 --requeue"
sbatch --export=ALL,LABEL=base_c0 $P experiments/exp_066_ladder_v3_scoring/job_score.sbatch
sbatch --export=ALL,LABEL=ic3_abc_c0,TRAIN=training_manifest_ic3 $P ...
# verify an arm's videos exist before its wave:
python3 build_eval_manifests.py --verify r2r3
```

## Waves (trigger = generation completion, count-verified via monitor)
| wave | labels | rows | trigger |
|---|---|---|---|
| W1 (submitted 2026-07-16 ~22:20, jobs 9542684-91) | base_c0..c5, ic2, sigma_hero_recheck | 354+60+5 | on disk ✓ |
| W2 | r1k_ext | 30 | base_ext array 9541943 done |
| W3 | r2r3_c0..c3 | 264 | arrays 9541939+9541940 done |
| W4 | r3x_c0..c1 | 132 | arrays 9541941+9541942 done |
| W5 | ic3_abc_c0..c2 (TRAIN=ic3) | 165 | grid 9541944 done |
| W6 | ic3_x_c0..c1 (TRAIN=ic3) | 132 | grid 9541945 done |

## Outputs
`outputs/eval/ladder_v3/<label>/{results.json, items.jsonl}`. Analysis joins
items across labels by item_id (twin pairing by class+clip+seed), reflags
`near_copy` at τ_copy=0.858 (amendment-1), attaches σ_seed MDEs to every
delta, consumes the trust map (n<4 → †). sigma_hero_recheck rescores the
exp_060 hero_flight rows under the corrected corpus (amendment-2 caveat);
compare vs `outputs/eval/sigma_seed`.
