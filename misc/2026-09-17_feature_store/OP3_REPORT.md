# OP3 REPORT — hand-off metrics from stored features

**Op-3.** `scripts/handoff_metrics.py` + `tests/test_handoff_metrics.py` + store eval entry
`store/evals/038_handoff_gridv3__dai__2026-09-18/` (meta.yaml + per-arm rows.jsonl + one INDEX.md line).
Branch `feature-store`; run on the DeltaAI login CPU, numpy over stored features (no GPU, no sbatch).
Instrument sha at run time: `21bd667` (contains this session's script commit `e24cc1f`; unchanged since).

## What was built

- **`scripts/handoff_metrics.py`** — per generation, from `dino_cls@dinov2b-r256` (feats [T,768])
  and `cotracker3@g20-m384-v2` (tracks [T,N,2], vis [T,N]) of the GEN and of its condition clips
  `eval_ladder/conds/<endpoint>_{start9,end9}.mp4`, emits `identity_A, identity_B, motion_A,
  motion_B, seam_free` + `n_pre, n_suf, K, fps, missing`. Definitions implemented **verbatim** from
  `BRIEF_OP3_handoff.md` (and copied into the eval meta.yaml):
  - windows: HF (121f) n_pre=9, n_suf=8 iff `sided=="two"`; ED (81f) & externals n_pre=1, n_suf=0.
  - `K = max(2, round(fps/3))`, fps via `probe_fps(gen)` — measured 8 (HF/ED @24fps), 3 (VAP/VFXMaster
    @9.72fps), 2 (refVFX @6.5455fps).
  - identity_A = mean\_{t∈[n_pre,n_pre+K)} cos(f_gen[t], f_condA[n_pre−1]); identity_B (two-sided) =
    mean\_{t∈[T−n_suf−K, T−n_suf)} cos(f_gen[t], f_condB[T_B−n_suf]), T_B=9 → f_condB[1].
  - motion_A = `motion_fidelity`(gen[0:n_pre+K], condA-full) only when n_pre≥9; motion_B (two-sided) =
    `motion_fidelity`(gen[T−n_suf−K:T], condB-full).
  - seam_free = 1 iff prefix_seam_z≤3 (one-sided) or prefix & suffix ≤3 (two-sided); read from the v4
    rows — Op-2's `per_gen.jsonl` if present (it was, for all 19 arms), else the `c*/items.jsonl` shards.
  - Missing feature → NaN + a `missing` entry like `dino_cls@dinov2b-r256:condA` /
    `cotracker3@g20-m384-v2:gen` / `seam`; never crashes; idempotent (rewrites rows/meta each run,
    appends the INDEX line only once). Gen npz are read lazily — only once the matching condition
    feature is present — so a pre-extraction run touches no gen npz.
- **`tests/test_handoff_metrics.py`** — 12 cases (all pass): synthetic cosines (1.0 / 0.0 / 0.5 / empty→NaN),
  the coherent identical-vs-reversed track set (`motion_fidelity` = 1.0 vs −1.0), end-to-end `compute_row`
  over a temp FeatureStore (identity_A/B=1.0, motion_A/B≈1.0), the missing-feature paths, the ED
  motion_A-undefined path, seam thresholds, and INDEX append idempotency.
- **Store entry** `038_handoff_gridv3__dai__2026-09-18`: `meta.yaml` (id/seq/shelf/created/machine,
  instrument+sha, the 6 definitions verbatim, `arms_scored` with per-arm gen/rows/grid_type/seam_source
  and a coverage snapshot) + `<harness_arm>/rows.jsonl` for all 19 arms + one appended INDEX.md line.
  rows.jsonl are store artifacts (.gitignored, not committed), regenerable and idempotent.

## Scope

The population (`population_gridv3.json`) lists **19** gridv3 gen variants: the **16 paper-arm variants**
(4 methods — ic_gen, base_cond, dualforce_control, dualforce_dcg_w6 — × {neutral, effect} × {HF 121f, ED 81f})
+ the **3 author-native externals** (refvfx, vap, vfxmaster). I processed all 19 so Op-5's Table B (which
reads Identity A for the externals from these rows) and the coordinator's re-run get every arm folder;
externals currently have no gen features, so their metrics are NaN + `missing` (reported below).

## Coverage per variant (snapshot — extraction is in progress)

Condition-clip features are still being produced by a running GPU job. At run time: gen dino + gen
cotracker are present for all internal arms; **condition-clip dino is partially landed**
(identity_A/B partial), **condition-clip cotracker has not landed** (motion_A/B all NaN + `missing`),
and seam_free is **complete** (Op-2's per_gen.jsonl present for all 19 arms). Numbers are LEVELS only.

`idA_n` = gens with finite identity_A; `idB_n/def` = finite / defined (two-sided); `mA_n/def`,
`mB_n/def` likewise; `seam_n` = finite seam_free; `*_mean` = mean over finite values; seam rate =
mean(seam_free).

```
arm                               n    idA_n  idA_mean  idB_n  idB_def  idB_mean  mA_n  mA_def  mB_n  mB_def  seam_n  seamfree_rate
ic_gen_neutral_v3                 564  314    0.9285    74     148      0.9335    0     564     0     148     564     0.9592
ic_gen_neutral_v3ed81             204  100    0.9782    0      0        nan       0     0       0     0       204     0.9804
ic_gen_effect_v3                  564  314    0.9259    74     148      0.9345    0     564     0     148     564     0.9628
ic_gen_effect_v3ed81              204  100    0.972     0      0        nan       0     0       0     0       204     1.0
base_cond_neutral_v3              564  314    0.9436    74     148      0.9524    0     564     0     148     564     0.8865
base_cond_neutral_v3ed81          204  100    0.9848    0      0        nan       0     0       0     0       204     0.3725
base_cond_effect_v3               564  314    0.9224    74     148      0.9561    0     564     0     148     564     0.8777
base_cond_effect_v3ed81           204  100    0.9791    0      0        nan       0     0       0     0       204     0.9951
dualforce_control_neutral_v3      564  314    0.9179    74     148      0.9209    0     564     0     148     564     0.9592
dualforce_control_neutral_v3ed81  204  100    0.9414    0      0        nan       0     0       0     0       204     0.9804
dualforce_control_effect_v3       564  314    0.9087    74     148      0.9067    0     564     0     148     564     0.9486
dualforce_control_effect_v3ed81   204  100    0.939     0      0        nan       0     0       0     0       204     0.9853
dualforce_dcg_w6_neutral_v3       564  314    0.856     74     148      0.8552    0     564     0     148     564     0.9238
dualforce_dcg_w6_neutral_v3ed81   204  100    0.8909    0      0        nan       0     0       0     0       204     0.9755
dualforce_dcg_w6_effect_v3        564  314    0.8641    74     148      0.8567    0     564     0     148     564     0.9184
dualforce_dcg_w6_effect_v3ed81    204  100    0.9033    0      0        nan       0     0       0     0       204     0.9755
refvfx_author_native              366  92     0.9726    0      0        nan       0     0       0     0       366     0.9317
vap_author_native                 366  102    0.8337    0      0        nan       0     0       0     0       366     0.7596
vfxmaster_author_native           366  92     0.9651    0      0        nan       0     0       0     0       366     0.8552
```

Reading the snapshot: identity_A is defined for every gen and computed wherever the endpoint's
`start9` dino has landed — HF/ED internal arms 314/564 and 100/204 (same endpoint set across the 16
internal arms), externals 92–102/366. identity_B is defined only for the 148 two-sided HF gens per arm
and computed for 74 of them (the endpoints whose `end9` dino has landed); it is 0/0 for ED and externals
(one-sided). motion_A is defined for the 564 HF gens (n_pre≥9) and 0 for ED/externals (n_pre=1, a frame
has no motion); motion_B is defined for the 148 two-sided HF gens; both are 0 computed everywhere because
condition-clip cotracker is not yet extracted. seam_free is complete (n = rows) for all 19 arms.

## Verification

- `pytest tests/test_handoff_metrics.py` → **12 passed**.
- identity_A hand-check: for `G-fit__ic_gen_neutral_v3__color_rain_3__ref_color_rain_1__s42` the script's
  0.964084876919435 reproduces a from-scratch numpy computation (condA `start9` = 9 frames, anchor =
  frame 8 = the last given frame) to 1e-9.
- seam source: Op-2's `per_gen.jsonl` (keyed (item_id, seed)) agrees **exactly** with the `items.jsonl`
  shards on (prefix_seam_z, suffix_seam_z) — 564/564 common, 0 disagreements — so the source switch is safe.
- All 564 gen videos per HF arm join a unique seam row; identical seam z across an item's pool rows (0
  disagreements) confirms the stem-prefix join is unambiguous.
- meta.yaml parses as valid YAML (19 arms_scored, 6 definitions); rows.jsonl carry exactly the 13 briefed
  keys, seed as int, `missing` as a list, NaN encoded as the JSON `NaN` token (round-trips via python json).

## Idempotency / re-run

Re-running `$PY scripts/handoff_metrics.py` (default population + eval-id) overwrites each rows.jsonl and
meta.yaml deterministically and re-appends nothing to INDEX.md (guarded by the eval id). As the GPU job
finishes the condition-clip `dino`/`cotracker` features, a re-run fills identity_A/B and motion_A/B in
place; when external gen features land, the external arms fill too. The coordinator re-runs later.

## Git

- `e24cc1f` infra: scripts/handoff_metrics.py + tests/test_handoff_metrics.py (pushed).
- `fad68ca` eval: store entry 038 meta.yaml + INDEX.md line (pushed).
- this report committed with `git add -f` (misc/ is gitignored).
- All commits as emirks <emirks88@gmail.com>; pushed after each; CHANGELOG.md, the pre-existing dirty
  files, and .claude/worktrees/* left untouched.

---

# Follow-up — motion v2 (velocity continuity), 2026-09-18

**Coordinator decision.** The v1 motion_A/motion_B (`motion_fidelity` of the gen's tracks vs the
condition clip's tracks) was defined on only ~166/564 HF rows (motion_A) and ~22–48/148 (motion_B)
even though gen and condition cotracker are 100% present: a 9-frame condition clip has almost no
tracklets above `motion_fidelity`'s moving threshold. **motion_A/motion_B are redefined as hand-off
VELOCITY CONTINUITY from the GENERATION's own cotracker3 tracks** (no condition tracks — the given
frames inside the gen ARE the pinned condition frames):

- per tracklet j visible (`vis ≥ 0.5`) on every frame of both windows: `v_before_j` = mean velocity
  over the last 3 given steps (frames [n_pre−3, n_pre]), `v_after_j` = mean velocity over the first K
  generated steps (frames [n_pre, n_pre+K]); `cos_j = cosine(v_before_j, v_after_j)`; weight
  `w_j = |v_before_j|·|v_after_j|`. **motion_A = Σ w_j cos_j / Σ w_j**; NaN if Σ w_j < 1e-6 (nothing
  moves) or **n_pre < 4** (frame-anchored: ED grids + all externals → NaN by construction, matching
  the table's "—").
- **motion_B** symmetric at the suffix (two-sided only): `v_before` = last K generated steps before the
  suffix (frames [T−n_suf−K, T−n_suf]), `v_after` = first 3 given suffix steps (frames [T−n_suf,
  T−n_suf+3]).
- velocities in tracked-resolution pixels (the cosine and weight ratio are scale-free).

The v1 values are kept as **`motion_A_mf` / `motion_B_mf`** (extra columns, low definedness, for
reference). identity_A/B and seam_free are unchanged. The 038 meta.yaml carries `definitions_version:
v2 (2026-09-18)`, `definitions_reason`, and the updated definitions verbatim. Re-run in place with
`--eval-id 038_handoff_gridv3__dai__2026-09-18 --no-index` (INDEX untouched); rows now carry 15 keys
(…, motion_A, motion_B, motion_A_mf, motion_B_mf, seam_free, missing).

**Definedness gain (motion_A):** ~166/564 → ~552–562/564 (v2 vs v1_mf); **motion_B:** ~22–48/148 →
~65–128/148. (The few NaN v2 rows: no tracklet is visible on every frame of the window / nothing moves.)
Condition-clip dino has also fully landed since the first run, so identity_A is now 564/204/366 (complete)
and identity_B 148/148 on the two-sided HF rows.

## v2 coverage + mean per variant

`idA`, `seam` = count of finite values; `idB/mA/mB` shown as finite/defined; `*_mean` = mean over
finite values; `mA_mf/mB_mf` = the v1 reference columns (finite/defined). ED & externals: motion is —
(n_pre<4 / one-sided). LEVELS only.

```
arm                               n    idA  idB(def)  mA(def)  mA_mean  mB(def)  mB_mean  mA_mf(def)  mB_mf(def)  seam
ic_gen_neutral_v3                 564  564  148/148   562/564  0.752    119/148  0.466    166/564     47/148      564
ic_gen_neutral_v3ed81             204  204  0/0       0/0      -        0/0      -        0/0         0/0         204
ic_gen_effect_v3                  564  564  148/148   560/564  0.756    95/148   0.391    165/564     35/148      564
ic_gen_effect_v3ed81              204  204  0/0       0/0      -        0/0      -        0/0         0/0         204
base_cond_neutral_v3              564  564  148/148   552/564  0.722    128/148  0.679    165/564     48/148      564
base_cond_neutral_v3ed81          204  204  0/0       0/0      -        0/0      -        0/0         0/0         204
base_cond_effect_v3               564  564  148/148   554/564  0.725    103/148  0.598    166/564     31/148      564
base_cond_effect_v3ed81           204  204  0/0       0/0      -        0/0      -        0/0         0/0         204
dualforce_control_neutral_v3      564  564  148/148   558/564  0.729    77/148   0.377    166/564     22/148      564
dualforce_control_neutral_v3ed81  204  204  0/0       0/0      -        0/0      -        0/0         0/0         204
dualforce_control_effect_v3       564  564  148/148   559/564  0.725    76/148   0.405    164/564     24/148      564
dualforce_control_effect_v3ed81   204  204  0/0       0/0      -        0/0      -        0/0         0/0         204
dualforce_dcg_w6_neutral_v3       564  564  148/148   559/564  0.716    65/148   0.387    165/564     26/148      564
dualforce_dcg_w6_neutral_v3ed81   204  204  0/0       0/0      -        0/0      -        0/0         0/0         204
dualforce_dcg_w6_effect_v3        564  564  148/148   557/564  0.715    66/148   0.38     166/564     20/148      564
dualforce_dcg_w6_effect_v3ed81    204  204  0/0       0/0      -        0/0      -        0/0         0/0         204
refvfx_author_native              366  366  0/0       0/0      -        0/0      -        0/0         0/0         366
vap_author_native                 366  366  0/0       0/0      -        0/0      -        0/0         0/0         366
vfxmaster_author_native           366  366  0/0       0/0      -        0/0      -        0/0         0/0         366
```

motion_A means sit ~0.72–0.76 across the 8 HF arms (the generated motion tends to continue the given
direction at the prefix hand-off); motion_B means are lower and vary more (~0.38–0.68) with fewer
tracklets visible throughout the suffix window (effects engulf points near the clip end). These are
levels, not verdicts.

## v2 verification

- `pytest tests/test_handoff_metrics.py` → **13 passed** (added `test_velocity_continuity`: identical
  velocity before/after → 1.0, reversed → −1.0, static → NaN, a frame occluded inside the window → NaN;
  `compute_row` and missing-feature tests updated: v2 motion_A needs only gen tracks, motion_A_mf needs
  condA tracks).
- motion_A hand-check: for `G-fit__ic_gen_neutral_v3__color_rain_3__ref_color_rain_1__s42` the script's
  0.8293510671114203 reproduces a from-scratch numpy continuity computation (657 tracklets visible
  throughout [6,17]) to 1e-9.
- meta.yaml valid YAML: `definitions_version: v2 (2026-09-18)`, 7 definitions, 19 arms_scored with the
  motion means + `_mf` counts in each coverage block.

## v2 git

- `f838576` infra: motion v2 = velocity continuity (scripts/handoff_metrics.py + tests; 13 pass) — pushed.
- 038 meta.yaml re-committed with the v2 definitions/coverage; INDEX.md unchanged (—no-index); rows.jsonl
  regenerated (gitignored). This report updated with `git add -f`. emirks <emirks88@gmail.com>; CHANGELOG.md,
  the pre-existing dirty files, and .claude/worktrees/* untouched.
