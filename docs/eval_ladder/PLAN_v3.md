# Eval Ladder — OPERATIVE PLAN (v3, tier-first)

**This is the clean current-truth document.** The frozen pre-registration
history (decisions, Amendments 1–2, contrasts in full) lives in `PLAN.md` —
never edited, only amended. Grid of record: `ladder_items_v3.json`.

## 1 · Data

| thing | value |
|---|---|
| corpus | **222 clips / 39 classes** (live_concert_2 removed — owner duplicate ruling; cert amendment-3) |
| split **v1.1** (frozen, tag `split/v1.1`) | **182 train / 40 test**, 29 test-bearing classes; live_concert test = live_concert_0; other 38 classes byte-identical to v1 |
| IC holdout (7, verbatim ic2) | hero_flight, illustration_scene, gas_transformation, raven, hole, seamless, jump — tier-C-eligible: **hero_flight, gas, illustration, raven** (the rest have 0 test clips) |
| universal rules | no adapter trains on a test-band clip · conditioning keyed by owner-final taxonomy everywhere · foreign endpoints sidedness-matched, prefix-only · seeds 42/43/44 · paired items across all models |

## 2 · Models × tiers (all volumes = rows × 3 seeds)

| model | trained on | tier | endpoints | rows | status |
|---|---|---|---|---|---|
| **base** 19B dev | — | P (prompt only) | canonical items | 50 | ✅ 150 on disk (R0) |
| | | PE-keyed | same items | 50+10 | ✅ R1 150 + R1K 54 · ▶ ext 30 |
| **specialists** ×11 | own class, train band, keyed | SEEN (R2) | 2 train clips, ckpt 250+2000 | 44 | ▶ 132 |
| | | UNSEEN·own (R3) | 2 test clips, ckpt 250+2000 | 44 | ▶ 132 |
| | | UNSEEN·foreign (R3X) | 4 donors' test clips, ckpt 2000, prefix-only | 44 | ▶ 132 (96 B8 + 36 ext) |
| **ic3** (exp_064) | 32 classes, train band, owner-final keying | A held-in (R4A) | designated train clips | 15 | ▶ 45 (train first) |
| | | B unseen (R4B) | trained ∩ test-bearing × test items | 33 | ▶ 99 |
| | | C zero-shot (R5) | 4 holdout classes × test items | 7 | ▶ 21 |
| | | X foreign (R4X) | same 11×4 donor pairs | 44 | ▶ 132 |
| **ic2** (frozen comparison arm) | — | old R4/R5 | — | — | ✅ 48+12 on disk; never headlines |

Notes: two_sided classes' PE-keyed base == their R1 rows (keyed ≡ blind);
tier-A includes 7 trained test-less stand-ins; base PE for two_sided = R1.

## 3 · Contrasts (headline set)

C1 R1−R0 (50 paired; middle_only split) · C3 R2−R3 overfit · C4 R3−R1(K) ·
**C5 R3 vs ic3-B, same items, sign n=8 — PRIMARY, clean by construction** ·
C6 ic3-C−R1(K): descriptive n=4 (no test) · C7 R3−ic3-C: descriptive n=3 ·
C8 ic3-B−R1(K) · C9 R3X>R4X, B8-only confirmatory (extension = labeled
exploratory) · C10 ic3−ic2 (value of alignment) · C11 ic3 A−B (overfit gap).
Scoring: certified v3.0.0 + amendments 1–3, corrected corpus manifest.

## 4 · Launch sheet (repo root on any CC login node)

```bash
P="--partition=secondary --account=campusclusterusers --gres=gpu:H100:1 --requeue"
# ic3 train chain: 2 chunks suffice (ic2 ran 500 steps/~25 min => ~4h10 + precompute);
# T3 = free safety chunk (DONE marker fast-exits) so the grids can never fire early.
T1=$(sbatch --parsable $P experiments/exp_064_ic3_aligned_retrain/job_train.sbatch)
T2=$(sbatch --parsable --dependency=afterany:$T1 $P experiments/exp_064_ic3_aligned_retrain/job_train.sbatch)
T3=$(sbatch --parsable --dependency=afterany:$T2 $P experiments/exp_064_ic3_aligned_retrain/job_train.sbatch)
# NOW — no dependencies (all trainings for these are DONE):
sbatch --array=0-29%15 --export=ALL,MODE=r2r3   $P experiments/exp_062_ladder_r2r3_specialists/job_gen_keyed.sbatch
sbatch --array=0-2     --export=ALL,MODE=b1     $P experiments/exp_062_ladder_r2r3_specialists/job_gen_keyed.sbatch
sbatch --array=0-23%15 --export=ALL,MODE=r3x    $P experiments/exp_062_ladder_r2r3_specialists/job_gen_keyed.sbatch
sbatch --array=0-8     --export=ALL,MODE=r3xext $P experiments/exp_062_ladder_r2r3_specialists/job_gen_keyed.sbatch
sbatch --array=0-5 --export=ALL,MANIFEST=dataset/manifest_base_ext.json,CHUNKS=2 $P experiments/exp_065_ladder_v3_grid/job_grid.sbatch
# AFTER ic3 (chained on the safety tail; adapter-assert makes them requeue-safe):
sbatch --array=0-20%15 --dependency=afterany:$T3 --export=ALL,MANIFEST=dataset/manifest_ic3.json,CHUNKS=7   $P experiments/exp_065_ladder_v3_grid/job_grid.sbatch
sbatch --array=0-17%15 --dependency=afterany:$T3 --export=ALL,MANIFEST=dataset/manifest_ic3_x.json,CHUNKS=6 $P experiments/exp_065_ladder_v3_grid/job_grid.sbatch
```
Then: score everything (certified checkout, corrected manifest).

### As-run log — 2026-07-16, cc-login5

Submitted exactly as the sheet above (knobs sized to observed availability: 15
free H100s on secondary, 3 small foreign jobs pending). CHUNKS/throttles are
submission-time knobs only — rows, settings, and seeds are frozen in the
manifests/grid and unchanged.

| what | job id | shape |
|---|---|---|
| ic3 train T1→T2→T3 | 9541860 → 9541861 → 9541862 | 3×3h55 chain, resume + DONE fast-exit |
| specialists SEEN+UNSEEN·own (r2r3) | 9541863 | array 0-29%15 |
| hero_flight (b1) | 9541864 | array 0-2 |
| specialists UNSEEN·foreign B8 (r3x) | 9541865 | array 0-23%15 |
| X-extension recipients (r3xext) | 9541866 | array 0-8 |
| base PE-keyed extension | 9541867 | array 0-5, CHUNKS=2 |
| ic3 A/B/C grid | 9541868 | array 0-20%15, CHUNKS=7, afterany:9541862 |
| ic3 X grid | 9541869 | array 0-17%15, CHUNKS=6, afterany:9541862 |

**21:10 resubmission — walltime 3:55 → 1:59 (backfill fix).** After 40 min at
0 running, diagnosis: the 15 free H100s (ccc0423/24) are PLANNED for a 400G
owner-partition job that starts when the current 256G CPU jobs end (~00:00);
only backfill that *ends before then* is admitted — a 59-min foreign job got
in while every 4h GPU job (ours + ~10 others) sat pending. All our task shapes
need ≤~70 min, so everything was cancelled and resubmitted at `--time=1:59:00`
(same knobs otherwise). Training becomes a 5-chunk resume chain (~1h50 compute
each; DONE marker fast-exits unused chunks). Short walltime also wins every
future backfill window. TIMEOUT tail-risk is covered by skip-if-exists +
straggler resubmit.

| what | job id | shape |
|---|---|---|
| ic3 train T1→…→T5 | 9541934-38 | 5×1h59 chain, resume + DONE fast-exit |
| specialists SEEN+UNSEEN·own (r2r3) | 9541939 | array 0-29%15 |
| hero_flight (b1) | 9541940 | array 0-2 |
| specialists UNSEEN·foreign B8 (r3x) | 9541941 | array 0-23%15 |
| X-extension recipients (r3xext) | 9541942 | array 0-8 |
| base PE-keyed extension | 9541943 | array 0-5, CHUNKS=2 |
| ic3 A/B/C grid | 9541944 | array 0-20%15, CHUNKS=7, afterany:9541938 |
| ic3 X grid | 9541945 | array 0-17%15, CHUNKS=6, afterany:9541938 |

**23:08 ic3 validation-cadence fix.** T1's trainer ETA read 6h55: the config's
initial 4-sample validation (~30 min) plus rounds every 1000 steps don't fit
1h59 chunks (T1 reached only step ~280 and died pre-first-checkpoint; its
progress is redone by T2). Fix, before T2 started: `skip_initial_validation:
true`, validation `interval: 1000 → 2500` (rounds at 2500/5000 — the
lora-train inline ID+OOD+control validation stays in-run). Observability-only:
optimizer trajectory, data order, ckpt interval 500, and every training
hyperparameter untouched (commit in exp_064). Revised chain forecast: ~2,300
net steps/chunk → step_05000 lands in T4 ≈ 05:00–05:30, ic3 grids fire ≈
05:30, last videos ≈ 08:00.

**Scoring (exp_066).** All 20 eval manifests built (1,142 rows, pre-registered
conventions) + `training_manifest_ic3.json`. W1 submitted 22:20 (jobs
9542684-91: base_c0..c5 354 rows, ic2 60, sigma_hero_recheck 5). Waves W2–W6
submit on count-verified generation completion; see exp_066/README.

**04:20 node-health incident (ccc0423/0424).** Both nodes went dark ~23:35 —
every job segment placed there after ~23:40 produced zero output (logs
untouched for hours, requeued segments never write their sbatch preamble)
while jobs elsewhere completed normally. Five hours of apparent-but-fake
progress: R3X frozen at 112/132, base-ext 0/30, ic3 stuck at T1's step-1500
checkpoint. Recovery: ExcNodeList=ccc0423,ccc0424 stamped on every pending
job; 16 zombie running jobs cancelled; r3x/r3xext/base_ext resubmitted with
the exclusion (9544740-42; skip-if-exists dedupes); training chain resumes
from step_01500 via T3 (9541936). Healthy-capacity note: only ~2 free H100s
outside the sick pair at 04:20 — timeline now rides on overnight churn.
