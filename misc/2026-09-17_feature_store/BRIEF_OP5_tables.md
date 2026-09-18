# BRIEF Op-5 — metric table builder (`scripts/build_metric_tables.py`)

Repo /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research, branch `feature-store`. Python `$LAB/envs-aarch64/ltx2/bin/python`.
CPU only. Read first: `papers_drafts/ctt_iclr2027/tables/{tab_main,tab_ablation,tab_isolation}.tex` (the target LOOK: booktabs, \shortstack
headers, `\ph{}` placeholders, macros `\ltx{} \segue{} \vap{} \vfxmaster{} \refvfx{}`), `papers_drafts/ctt_iclr2027/preamble.tex` (macros),
`misc/2026-09-17_feature_store/population_gridv3.json`, the three input schemas below, and `papers_drafts/_preview/` (existing preview flow,
`render_owner_preview.py` if present — the owner reviews tables in a SEPARATE preview PDF; never overwrite Ozgur's `tables/*.tex`).

## Inputs (joined on `(item_id, seed)`; any may be partially present — a missing column renders as `\ph{--}` and is listed in the report)
1. v4 per-gen rows (Op-2): `store/evals/028_…/<arm>/per_gen.jsonl`, `store/evals/030_…/<arm>/per_gen.jsonl` — keys incl. `tier, sided, cell,
   content, transport_pct, copy_max, near_copy, max_seam_z, prefix_seam_z, suffix_seam_z, prefix_dino`.
2. hand-off rows (Op-3): `store/evals/*_handoff_gridv3__dai__*/<arm>/rows.jsonl` — `identity_A, identity_B, motion_A, motion_B, seam_free`.
3. lens rows (later tonight): `store/evals/*_lenses_gridv3__dai__*/<arm>/rows.jsonl` — `motion_smoothness, videoprism_sim_ref,
   det_motion_fidelity, clip_sim_ref, dynamic_degree_mean_mag, aesthetic`.
Arm labels: `base_cond_*` → `\ltx{} (no reference)`, `ic_gen_*` → `\ltx{} baseline LoRA`, `dualforce_control_*` → `\segue{} w/o guidance`,
`dualforce_dcg_w6_*` → `\segue{}`, externals → `\vap{}`, `\vfxmaster{}`, `\refvfx{}` (+ `~\citep{…}` as in tab_main). Variants: `neutral`
(HF `_neutral_v3` + ED `_neutral_v3ed81` pooled) unless stated; `effect` only in Table C.

## Tables (column definitions FIXED; each cell = mean over the rows in scope; `n` shown per block)
- **Table A** (own arms, 3 tiers, neutral): rows per tier = the 4 arms; columns `Identity A | Identity B | Motion A | Motion B |
  Seam-free % | Motion smooth. | Copy rate % ↓ | Transport`. Identity/Motion B and Motion A on the rows where defined (B: two-sided; Motion A:
  n_pre ≥ 9) — show `n` for those cells when it differs from the block n. Copy rate = 100·mean(near_copy). Seam-free = 100·mean(seam_free).
- **Table B** (prior works, zero-shot, the shared one-sided set = exactly the rows present in the externals' grids, matched into our arms by
  (endpoint, reference, cell, seed)): rows `\vap{}, \vfxmaster{}, \refvfx{}, \ltx{} baseline LoRA, \segue{} w/o guidance, \segue{}`; columns
  `Identity A | Seam-free % | Motion smooth. | Copy rate % ↓ | Transport | Ref sim. (VideoPrism) | Motion fid. (vs ref) | Aesthetic`. One n for
  every column (366 per arm) — assert it, and report any row lost in the match.
- **Table C** (text dependency, same shared set): rows = the 4 own arms × {neutral, effect} and the 3 externals (author-native = their effect
  row; neutral row `\ph{--}`); columns `Transport neutral | effect | Δ | Ref sim. neutral | effect | Δ | n`.
- Table D (ablation) is NOT built tonight (the w-sweep arms are not on grid v3) — emit a one-line note instead.
Bold = best per column per tier/block (↓ for copy rate). Numbers: transport/rates 1 decimal; similarities 3 decimals.

## Outputs
`papers_drafts/_preview/metrics_gridv3/{tab_A,tab_B,tab_C}.tex` (self-contained `table` envs in the paper's style) +
`papers_drafts/_preview/metrics_gridv3/TABLES.md` (the same numbers in markdown, plus n and the list of `\ph{--}` cells and why) +
`papers_drafts/_preview/metrics_gridv3/preview.tex` that `\input`s the paper's preamble/macros and the three tables, and build it with
`latexmk -pdf` (PATH `$LAB/texlive/bin/aarch64-linux`) → `preview.pdf`. A `--strict` flag fails if any Table B/C column has mixed n.
Tests: `tests/test_build_metric_tables.py` on a tiny synthetic input (join, n assertion, bolding, `\ph{--}` for missing).
Git: emirks <emirks88@gmail.com>, pathspec staging (script, test, the `_preview/metrics_gridv3/*.tex|md` outputs; `git add -f` the report
`misc/2026-09-17_feature_store/OP5_REPORT.md`), push after every commit. Don't touch the ~66 dirty files, CHANGELOG.md, Ozgur's `tables/*.tex`,
`.claude/worktrees/*`. Build it now against whatever inputs exist (Op-2/Op-3 are running in parallel — if their files are absent, synthesize
a fixture and make the script robust); the coordinator re-runs it when all inputs land. Reply with a concise summary.
