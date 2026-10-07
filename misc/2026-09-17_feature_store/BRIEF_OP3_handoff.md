# BRIEF Op-3 — hand-off metrics from store features (`scripts/handoff_metrics.py`)

Repo /taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research, branch `feature-store`. Python `$LAB/envs-aarch64/ltx2/bin/python`.
CPU only; no GPU; no sbatch. Read first: `store/FEATURES.md`, `src/diffusion/feature_store.py` (`FeatureStore.get(video, ns)`),
`src/diffusion/transition_eval/motion.py` (`motion_fidelity(tracks_a, vis_a, tracks_b, vis_b)` — read its exact signature/semantics),
`src/diffusion/transition_eval/video_io.py` (`probe_fps`), the gens' `grid.jsonl` (`sided`, `endpoint`), `eval_ladder/encode_conditioning.py`
(`cond_paths`: the condition clips are `eval_ladder/conds/<endpoint>_start9.mp4` / `_end9.mp4`), `misc/2026-09-17_feature_store/population_gridv3.json`.

## What to compute (definitions are FIXED — implement exactly)
Per generation, from `dino_cls@dinov2b-r256` (`feats [T,768]`, L2-normalized) and `cotracker3@g20-m384-v2` (`tracks [T,N,2]`, `vis [T,N]`)
of the GEN and of its CONDITION CLIPS:
- Conditioning windows: HF-grid rows (121 f): `n_pre = 9`, `n_suf = 8` if `sided == "two"` else 0. ED-grid rows (81 f, frame-0 anchor):
  `n_pre = 1`, `n_suf = 0`. Externals (VAP/VFXMaster 49 f, refVFX 33 f, frame-0 conditioning): `n_pre = 1`, `n_suf = 0`.
- Hand-off window length `K = max(2, round(fps / 3))` frames (≈ 1/3 s; 8 at 24 fps, 3 at 9.72 fps, 5 at 15 fps). fps via `probe_fps(gen_video)`.
- **identity_A** = mean over t ∈ [n_pre, n_pre+K) of cos(f_gen[t], f_condA[n_pre−1]) where f_condA = dino of `<endpoint>_start9.mp4`
  (the last GIVEN frame; for n_pre=1 that is frame 0). **identity_B** (two-sided only) = mean over t ∈ [T−n_suf−K, T−n_suf) of
  cos(f_gen[t], f_condB[T_B − n_suf]) where f_condB = dino of `<endpoint>_end9.mp4` and T_B = 9 (the first given suffix frame = frame 1 of end9).
  Else NaN.
- **motion_A** = `motion_fidelity` between the gen's tracks/vis sliced to frames [0, n_pre+K) and the start clip's full tracks/vis (9 f) —
  only when `n_pre ≥ 9` (a clip was given); NaN for n_pre = 1 (a frame has no motion). **motion_B** symmetric: gen frames [T−n_suf−K, T)
  vs the end clip's tracks; NaN unless two-sided.
- **seam_free** = 1 if the relevant seam z-scores ≤ 3 (prefix only for one-sided; prefix and suffix for two-sided) — read `prefix_seam_z`,
  `suffix_seam_z` from the v4 rows (they are in `store/evals/028|030/…/items.jsonl`; take them from Op-2's `per_gen.jsonl` if present,
  else from items.jsonl) — so this script emits `seam_free` alongside, for one join.
- Missing feature (e.g. externals before P4 lands) → NaN + a `missing` field naming the namespace; never crash; idempotent per gen.

## Output
A store eval entry: `store/evals/<next free NNN>_handoff_gridv3__dai__2026-09-18/<harness_arm>/rows.jsonl` (one row per gen:
`item_id, seed, arm, n_pre, n_suf, K, fps, identity_A, identity_B, motion_A, motion_B, seam_free, missing`) + `meta.yaml`
(`id, seq, shelf: evals, created, machine: dai (login CPU, numpy over stored features), instrument: handoff_metrics.py @ <git sha>,
definitions (the bullets above verbatim), arms_scored: {arm: {gen, rows}}`) + an INDEX.md line (append; do not renumber anything).
`harness_arm` = the gen's `meta.yaml` `harness_arm`.

## Tests + report
`tests/test_handoff_metrics.py` with synthetic features (known cosines; a track set that is identical vs reversed). Run the script on the
16 paper-arm variants NOW (their gen/cond dino+tracks may partially exist — conds' features arrive from a running GPU job; report coverage) and
again later is fine (idempotent). Report `misc/2026-09-17_feature_store/OP3_REPORT.md` (`git add -f`). Git: emirks <emirks88@gmail.com>,
pathspec staging (script, test, `store/evals/<new>/meta.yaml`, `store/INDEX.md` line), push after every commit; don't touch the ~66 dirty
files, CHANGELOG.md, `.claude/worktrees/*`. Reply with a concise summary incl. coverage per variant.
