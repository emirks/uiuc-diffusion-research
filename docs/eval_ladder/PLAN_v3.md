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

## 4 · Launch sheet (from cc-login3, repo root)

```bash
P="--partition=secondary --account=campusclusterusers --gres=gpu:H100:1 --requeue"
# NOW — no dependencies (all trainings for these are DONE):
T1=$(sbatch --parsable $P experiments/exp_064_ic3_aligned_retrain/job_train.sbatch)
T2=$(sbatch --parsable --dependency=afterany:$T1 $P experiments/exp_064_ic3_aligned_retrain/job_train.sbatch)
sbatch --array=0-29%6 --export=ALL,MODE=r2r3   $P experiments/exp_062_ladder_r2r3_specialists/job_gen_keyed.sbatch
sbatch --array=0-2    --export=ALL,MODE=b1     $P experiments/exp_062_ladder_r2r3_specialists/job_gen_keyed.sbatch
sbatch --array=0-23%6 --export=ALL,MODE=r3x    $P experiments/exp_062_ladder_r2r3_specialists/job_gen_keyed.sbatch
sbatch --array=0-8    --export=ALL,MODE=r3xext $P experiments/exp_062_ladder_r2r3_specialists/job_gen_keyed.sbatch
sbatch --array=0-2 --export=ALL,MANIFEST=dataset/manifest_base_ext.json,CHUNKS=1 $P experiments/exp_065_ladder_v3_grid/job_grid.sbatch
# AFTER ic3 (chained on the train tail; adapter-assert makes them requeue-safe):
sbatch --array=0-11%6 --dependency=afterany:$T2 --export=ALL,MANIFEST=dataset/manifest_ic3.json,CHUNKS=4   $P experiments/exp_065_ladder_v3_grid/job_grid.sbatch
sbatch --array=0-8%6  --dependency=afterany:$T2 --export=ALL,MANIFEST=dataset/manifest_ic3_x.json,CHUNKS=3 $P experiments/exp_065_ladder_v3_grid/job_grid.sbatch
```
Then: score everything (certified checkout, corrected manifest).
