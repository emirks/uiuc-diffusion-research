# Session summary — 2026-09-07 → 2026-09-09 (grid v3 gen+eval, analyses, EffectData gapper screen)

Written for the owner before compaction. Everything below is committed on `bneck_redesign` (last f716e4e) and, for the scorer,
on the cert worktree branch `eval/v4-metrics` (8734f2a, 4e792c0). Numbers are LEVELS (pool-% of m1a), never verdicts.

## 1. Grid v3 — generation and evaluation of the four paper arms (DONE)
- 16 harness arms `{base_cond, ic_gen, dualforce_control, dualforce_dcg_w6} x {neutral, effect} x {Higgsfield 121 f, EffectData 81 f}`,
  384 rows x 2 seeds -> 6,144 clips (4,290 new + 1,854 hardlinked kept-row clips) in gens/005_base_cond/04-07, 001_ic_gen/03-06,
  013_dualforce_control/03-06, 032_dualforce_dcg_w6/03-06. ONE v4 pass: `store/evals/028_grid_v3_paper_arms__dai__2026-09-07`
  (80,673 rows, 0 error rows). summary.json per arm x cell; REPORT.md (owner tables + per-class diagnostics) in misc/2026-09-07_eval_grid_v2/eval/.
- Viewer `iclora_neutral_effect`: 16 grid-v3 entries; grid chip row (v2 / v3-HF / v3-ED) filters metrics, cards AND the arm panel.
- Tooling: scripts/grid_v3/{launch_gen, gen_dcg_v3, launch_score (register->plan->submit->status, re-pass labels), closeout, class_tables,
  gap_predictors, gap_predictors_clip}.py.
- Defects met and fixed: official stack rejects `attention` -> ic_gen on the ctt fork; gen sbatch swallowed exit codes; scorer null rows for
  references outside the certified 222 population + a cache race between shards (amendment 2: per-clip cams/profiles for superset clips,
  on-the-fly CSLS hub term, row flag `ref_in_v4_population`, atomic+tolerant caches; in-population rows byte-identical); one kept DCG clip
  was a truncated 1 MiB mp4 from 2026-09-03 (regenerated); eval entry forked by date at midnight (merged, launcher fixed); cert-worktree
  class symlinks required for every new class. Queue: score shards waited up to 9.5 h as fairshare fell; moving short shards to bgms
  (best FairShare) with a 30-min walltime started them in minutes (~15 GPU-h on bgms).

## 2. What the grid-v3 numbers say (pooled over all content cells; rows in parentheses)
| arm | neutral HF (282) | effect HF (282) | neutral ED (102) | effect ED (102) |
|---|---|---|---|---|
| base_cond | 52.5 | 89.2 | 38.4 | 90.2 |
| ic_gen | 72.9 | 90.5 | 57.1 | 89.7 |
| dualforce control | 88.2 | 95.3 | 94.1 | 98.0 |
| dualforce DCG w=6 | 95.5 | 97.9 | 98.5 | 100.4 |
- The old base_cond effect headline 84.9 (evals/002) included 13 mismatched-reference control rows at 61; on the 47 genuine same rows it
  was already 91.5; grid v3 (control rows dropped) reads 92.6. Kept rows re-score byte-identically (0.0 pp in 6/8 cells).
- With the effect clause every arm is in the 90s: the effect tier measures the text more than the reference. The yardstick saturates
  (>=100) on low-ceiling classes (glitch 0.61, shadow 0.66, acid 0.76, flying_cam 0.49).
- Where prompting fails (base effect < 80) the reference arms hold: Higgsfield 17 classes / 125 rows, DCG - base ~ +26 pp, and the
  selection REPLICATES across seeds (+24 to +28 out-of-seed). EffectData: 7 classes / 21 rows, +26 in-sample but noisy at 3 rows/class.
- Predictors of the gap: prompt-only EFFECT level rho = -0.70 (HF -0.61, ED -0.81); video-only descriptors from the scorer cache |rho| <= 0.26
  with sign flips; CLIP clause/clip alignment rho ~ 0; the NEUTRAL gap does not predict the effect gap (0.04). Hence the screen below.

## 3. EffectData gapper screen (misc/2026-09-08_ed_gapper_screen; evals/029)
- Stage A: 300 effects picked by scripts/ed_screen/select_effects.py — 150 exploit (kNN text-prior from BGE-large embeddings fitted on
  the 34 measured ED effects), 140 explore (k-means medoids), 10 anchors (measured effects). 674/2,917 eligible (only ~30% of roster
  subjects are portrait). 3,600 clips standardised (13 min), 299 gemini clauses, 900 prompt-only rows (arm base_cond_effect_edscreen,
  prompts/014), seed 42, 81 f frame-0 anchor. Result: median level 70.4; 91 / 149 / 207 of 300 below 60 / 70 / 80.
  The text prior did NOT work (exploit vs explore share<80 0.67 vs 0.73; rho(pred, level) 0.03). Anchors read ~11 pp LOWER than their
  grid-v3 two-seed levels (one seed, new rows): absolute screen levels are pessimistic; the ORDERING is the deliverable.
- Stage B: DCG w=6 effect (arm dualforce_dcg_w6_effect_edscreen, prompts/015) on the 40 lowest effects, ONE paired row each (the screen's
  first endpoint): base 45.4 -> DCG 72.5, delta +27.1 pp mean / +30.8 median, DCG higher on 35/40. Strongest: Fingertip_smoke +71,
  Blood_cloak_aura +65, Petal_shield_from_arm +65, Foot-rising_light_column +63, Honey_skin_patterns +62, Shattering_chains +60,
  Leaf_path_from_feet +51. Negatives: Fireflies_rising -40, Arm_time_phase -17, Diamond_hand_gauntlet -13. Table: eval/STAGE_B.md.
- Caveat: selected on the same rows we compare -> a diagnostic slice, not an estimate. Stage C (new subjects, 6-8 rows/effect, two seeds,
  base + control + DCG) is what makes a reportable number. Every screened effect has 45 subjects (~10 captioned, ~13-15 portrait), so
  6-8 held-out rows per effect are feasible; extra captions via Gemini for non-roster portrait subjects.
- Plumbing (side lane, nothing pinned touched): LADDER_SPLIT_FILE=split_screen.json (prompts.py), LADDER_CEILINGS_EXTRA (run_eval),
  CORPUS= (score_v3.sbatch), screen superset manifest (4,175 clips / 385 classes), kernel ceilings, 288 worktree class symlinks.

## 4. Open decisions (owner)
1. Stage C design: which effects (top-k by stage-B gap, or all 40), rows per effect, seeds, arms (base / control / DCG / ic_gen), and
   whether stage-C rows replace or sit beside the paper's current 34-effect EffectData tier (selection must be disclosed either way).
2. Whether to run dualforce CONTROL on the same 40 rows (40 clips, cheap) to split "reference" from "DCG on top".
3. How to present saturation: ceiling-capped levels, raw app_ref alongside, or the copy metric as the tie-breaker for >=100 cells.
4. Threshold vs quantile for "prompting fails" (80 was grid-v3-calibrated; the screen reads ~11 pp lower at one seed).
5. Accounts: bgms ~35 h left (used ~16 h this campaign), bhwp ~450 h, bgjg ~2,750 h; pick by FairShare, keep short shards off the account
   that carries long generations.
