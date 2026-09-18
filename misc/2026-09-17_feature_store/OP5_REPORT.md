# OP-5 report — metric table builder (`scripts/build_metric_tables.py`)

Operator Op-5. Branch `feature-store`. CPU-only, DeltaAI login node. Python
`$LAB/envs-aarch64/ltx2/bin/python` (3.12.9, numpy 2.4.2); TeX Live 2026 (latexmk 4.88).

> **Follow-up (coordinator, 2026-09-18) — applied; see the section at the end.** Copy rate now
> comes from a fourth input, the M2a copy eval `*_copy_gridv3*` (per_gen's `near_copy` is pool-scored
> and NOT used); any cell with defining `n < 10` renders `n/a` (with its n) and is excluded from
> bolding. Op-2's per_gen and Op-3's hand-off (eval 038) have since landed, so the committed tables
> now carry **real** transport + hand-off numbers (Table B shared set 366/366, zero rows lost); the
> copy and lens columns are still `\ph{--}` pending Op-6 / the lens run.

## What was built
- **`scripts/build_metric_tables.py`** — joins four per-generation feature sources on
  `(item_id, seed)` and emits Tables A/B/C in the paper's booktabs style, a markdown twin
  (`TABLES.md`), a `preview.tex`, and a built `preview.pdf`, written **only** under
  `papers_drafts/_preview/metrics_gridv3/`. Ozgur's `papers_drafts/ctt_iclr2027/tables/*.tex`
  are never touched (verified: `git status -- papers_drafts/ctt_iclr2027/tables/` shows only the
  untracked-dir marker; no writes there).
- **`tests/test_build_metric_tables.py`** — 25 tests (all pass) over a synthetic fixture, incl. the
  min-n rule and the copy-eval join.
- **Outputs** at `papers_drafts/_preview/metrics_gridv3/`: `tab_A.tex`, `tab_B.tex`, `tab_C.tex`,
  `TABLES.md`, `preview.tex`, `build.sh`, `.gitignore` (committed); `preview.pdf` +
  latexmk artifacts + `fixture_demo/` (gitignored, local only).

## Inputs consumed (schemas from the Op-2 / Op-3 / Op-6 briefs)
Joined on `(item_id, seed)`; a fully-absent column renders `\ph{--}` and is listed in `TABLES.md`.
1. **v4 per-gen (Op-2)**: `store/evals/028*/<arm>/per_gen.jsonl`, `store/evals/030*/<arm>/per_gen.jsonl`.
   Keys used: `item_id, seed, tier, sided, cell, content, endpoint, reference, transport_pct,
   max/prefix/suffix_seam_z, prefix_dino`. `base_arm/variant_kind/family` are parsed from the
   arm-directory name (`parse_arm`). **Its `near_copy`/`copy_max` are NOT used** (pool-scored — see 4).
2. **hand-off (Op-3)**: `store/evals/*_handoff_gridv3__*/<arm>/rows.jsonl` — `identity_A, identity_B,
   motion_A, motion_B, seam_free` (+ `n_pre/n_suf/K/fps`). Motion A/B are the v2 (velocity-continuity)
   definition in eval 038; same keys.
3. **lens (later)**: `store/evals/*_lenses_gridv3__*/<arm>/rows.jsonl` — `motion_smoothness,
   videoprism_sim_ref, det_motion_fidelity, aesthetic` (clip/dyn loaded but unused by these tables).
4. **copy (Op-6)**: `store/evals/*_copy_gridv3__*/<arm>/rows.jsonl` — `near_copy, copy_max, ...`
   (M2a vs the gen's OWN reference). **Copy rate % = `100·mean(near_copy)` from here**, landing on the
   record as `copy_near_copy` so per_gen's invalid value can never be read.

**State at rebuild (2026-09-18):** Op-2 per_gen (19 arm files, 7242 rows) and Op-3 hand-off (eval 038,
7242 rows) are present and join 1:1 on `(item_id, seed)`; **lens and copy are not yet present** →
those columns (Motion smooth., Ref sim., Motion fid., Aesthetic, Copy rate) render `\ph{--}`. The
committed tables therefore carry real transport + hand-off numbers with the four remaining columns as
placeholders. A populated **`fixture_demo/`** (synthetic, loud red banner, includes the copy eval) is
generated alongside so the full look/bolding/sub-n/n-a/placeholders are visible; it is git-ignored.

## Table definitions (fixed, as implemented)
- **Table A** — own arms `{base_cond→LTX (no reference), ic_gen→LTX baseline LoRA,
  dualforce_control→SEGUE w/o guidance, dualforce_dcg_w6→SEGUE}` × tiers `{seen, unseen, zero_shot}`,
  **neutral** prompt (HF `_neutral_v3` + ED `_neutral_v3ed81` pooled). Columns: Identity A | Identity B
  | Motion A | Motion B | Seam-free % | Motion smooth. | Copy rate % ↓ | Transport. Block `n` per row
  (gray, on the label); a per-cell `n` where a column is defined on fewer rows — Identity/Motion B on
  the two-sided subset, Motion A only where a clip was given (`n_pre ≥ 9`, i.e. HF not ED). Copy rate =
  `100·mean(near_copy)`, Seam-free = `100·mean(seam_free)`.
- **Table B** — prior works, zero-shot, the **shared one-sided set**: the intersection of
  `(endpoint, reference, cell, seed)` keys across the 6 arms `{VAP, VFXMaster, refVFX, LTX baseline
  LoRA, SEGUE w/o guidance, SEGUE}` (externals = author-native, our arms = neutral, one-sided,
  zero-shot). Target `n = 366` (= 183 triples × 2 seeds). Columns: Identity A | Seam-free % | Motion
  smooth. | Copy rate % ↓ | Transport | Ref sim. (VideoPrism) | Motion fid. (vs ref) | Aesthetic.
- **Table C** — text dependency on the **same shared set**: rows = the 4 own arms × {neutral, effect}
  presented one-row-per-arm, plus the 3 externals (author-native = their effect row; neutral & Δ =
  `\ph{--}`). Columns: Transport neutral | effect | Δ | Ref sim. neutral | effect | Δ | n. Δ = effect − neutral.
- **Table D** — **not built** (the w-sweep arms w=1/1.5/3 are not on grid v3): a one-line note in
  `TABLES.md` and `preview.tex`; the paper's `tab_ablation.tex` draft stands.
- Bold = best per column per block (`min` for Copy rate; `absmin` for the text-dependency Δ columns;
  `max` otherwise); ties bold together. Numbers: transport & rates 1 decimal, similarities 3 decimals.

## `--strict`
Fails (exit 1) if any **present** Table B/C column has mixed `n` across its rows (a fully-absent
column is exempt — it is `\ph{--}`, reported separately). The shared-set intersection dropping own-arm
ED / non-shared rows is **by design** (the externals set the one-sided HF frontier), so it is reported
as a non-fatal coverage note, not a failure. A `shared_n ≠ 366` is a non-fatal coverage warning.

## Decisions / deviations (flagged for the coordinator)
- **Table C uses 4 own arms** (adds `base_cond`), as this brief states; the paper's `tab_isolation.tex`
  shows only 3 (no base_cond). `base_cond` is measured on the fixed shared set; if it lacks some shared
  keys its `n` differs and is reported. Drop it if the paper wants 3.
- **refVFX Transport carries Op-2's `transport_pct` = the S3 / `app_ref` column (~62), not the
  `Look_u` 75.5 in the paper draft.** Compare a system with itself; note stated in `TABLES.md`.
- **Aesthetic** is formatted to 3 decimals (bucketed with the similarities); trivially changed if the
  owner prefers 2.
- The `preview.tex` `\input`s the paper's real `preamble.tex` (all macros: `\ltx \segue \vap
  \vfxmaster \refvfx \ph`) + vendored `natbib` + the vendored `iclr2027_conference` bib style, so
  `\citep{}`, booktabs and the placeholder colour all render exactly as in the paper. The stale
  `_preview/build.sh` (which points at a non-existent `main_v2.tex`) is not used.

## Verification (actual output, post-follow-up rebuild)
- `pytest tests/test_build_metric_tables.py -q` → **25 passed in 0.33s**.
- Canonical build (`--source real`): per_gen 7242 rows, hand-off 7242 rows (1:1 join), lens 0, copy 0;
  Table B shared-set **n = 366, per-arm bench all 366, zero rows lost**; `preview.pdf` **build OK**.
  Table B Transport reproduces the paper's tab_main zero-shot block — VAP 81.1, VFXMaster 85.8,
  refVFX 62.2 (S3/`app_ref`), LTX baseline LoRA 61.6, SEGUE w/o guidance 87.6, SEGUE 92.6.
  Copy rate, Motion smooth., Ref sim., Motion fid. and Aesthetic render `\ph{--}` (copy + lens absent).
  Min-n rule visible: the seen tier's Identity B / Motion B (n = 1..8) render `n/a` and do not bold.
- Fixture demo (`--source fixture`): 19 arm files / 416 rows, hand-off 416, lens 320, copy 416; Table B
  shared-set `n=16` (8 HF triples × 2 seeds; own arms' 4 ED rows dropped and reported); bolding, sub-n,
  the `n/a` rule on the thin two-sided subset, and refVFX's partial-lens `\ph{--}` all render; **build OK**.

## Follow-up — coordinator changes (2026-09-18), applied
1. **Copy rate source.** Added the M2a copy eval `store/evals/*_copy_gridv3__*/<arm>/rows.jsonl` as a
   fourth `(item_id, seed)`-joined input (`--copy-glob`, auto-discovered like handoff/lens). Its
   `near_copy` lands as `copy_near_copy`; the Copy rate columns in Tables A/B read that, never per_gen's
   pool-scored `near_copy` (which is identical across arms and not a generation property). `\ph{--}`
   while the eval is absent. A frame-count caveat is written under Table B in `TABLES.md`: M2a takes the
   max over mid frames, so our 121-f clips have more match chances than the externals' 49-f (VAP/
   VFXMaster) / 33-f (refVFX) — the externals' copy rate is under-estimated, disclosed not corrected.
2. **Minimum n.** `MIN_N = 10`: any cell whose defining n is `< 10` renders `n/a` (with its n) and is
   excluded from the bold contest, in every table. This hits the seen tier's two-sided Identity B /
   Motion B (n = 1..8) and any other thin cell; the sub-n annotation is kept for reportable
   (n ≥ 10) Motion/Identity B cells.
Nothing changed for Motion A/B beyond reading eval 038 (v2 velocity-continuity; same keys).

## Coordinator regeneration command (run once all inputs land)
```
cd /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research
$LAB/envs-aarch64/ltx2/bin/python scripts/build_metric_tables.py --strict
```
Default `--source auto` uses the real store as soon as `store/evals/028*/<arm>/per_gen.jsonl` exists
(else it falls back to the fixture with a banner). Add `--strict` only for the final, complete-input
run (it fails on a half-populated lens column, which is the intended guard). To eyeball the populated
layout at any time: `... --source fixture --out-dir papers_drafts/_preview/metrics_gridv3/fixture_demo`.
The build refreshes `tab_A/B/C.tex`, `TABLES.md`, `preview.tex` and rebuilds `preview.pdf`
(`papers_drafts/_preview/metrics_gridv3/build.sh` rebuilds just the PDF).
