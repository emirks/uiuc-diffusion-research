# metrics v5 — from the store to the paper's metric table

The paper's metric table (`papers_drafts/_preview/metrics_v2_gridv3*/preview.pdf`, built by
`scripts/family_tables.py`) is reproducible from **store entries alone**. Two owner decisions (2026-09-23) drove v5:

1. **Transport = the three-channel blend at equal thirds** — appearance (Mu) / semantic transport (D) /
   pixel transport (PX), each a rank-similarity vs the 222-corpus population, blended 1/3 each, re-ranked,
   % of the class ceiling. This replaces the v4 `transport_pct`. The blend rule is the Look_u rule
   generalised (`misc/2026-09-02_temporal_dynamics_metric/blend_grid.py`).
2. **Seam at the physical given window everywhere** — seam z recomputed from the stored temporal LPIPS
   (`lpips_t@alex-r256`) with `store_eval_common.windows`, correcting evals/030's 9-frame planned prefix on
   the one-sided prior works to their 1-frame physical prefix.

Everything the table reads now lives in the store as numbered evals.

## The roster

`scripts/metrics_v5/roster.json` is the single source of table rows (22 arms). Each entry: `id`, `role`
(`own_base` | `own` | `own_effect` | `prior` | `prior_teg`), `prompt`, `gens` (store gens variants), `label`/`tex`,
optional `sweep` and `twin`. `family_tables.py` reads the roster and drops the old hardcoded row lists.
The `own_effect` role (added 2026-09-23) holds the three effect-prompt twins of our arms — `ic_gen_effect`,
`dualforce_control_effect`, `dualforce_dcg_w6_effect` — for the paper's N / E / Delta tables. It is **never**
selected by `family_tables._roster_rows`, so Tables 1/2/3 and `paper_table_v5` are byte-identical with or without
it; only `scripts/metrics_v5/paper_tables.py` reads it, for the effect (E) columns of the five paper tables.

## Stages (driver: `scripts/metrics_v5/v5.py`)

| subcommand | deliverable | what it does |
|---|---|---|
| `import-trackdesc` | D1 | fill `trackdesc@cotracker3-s64-v1` (dirs + px derived from the stored cotracker3 tracks): import the `mf_dirs_cache` dirs/px where present, else compute from the tracks |
| `reference` | D2 | bundle `reference_v5.npz` (222-corpus channel matrices + PX + new-class pair distances), pin its sha in `versioning.py` |
| `blend` | D3 | eval `047_transport_v5_gridv3` — pairs + per-gen blend, imported from the `mf_pair_cache` pair caches, scored against `reference_v5` (`blend.py`) |
| `pairs` | D8 | ADD to eval 047 the Mu/D/PX for roster arms WITHOUT a cache (sweep arms + neutral twins), computed from stored features (score_v3_mf's channels; gen `dino_cls`, gen/ref `trackdesc`); skip-if-done, meta merged, never re-imports (`--gen-dino store` default) |
| `seam` | D4 | eval `048_seam_gridv3` — seam z from `lpips_t@alex-r256` at the physical windows (`seam_eval.py`) |
| `flow-action` | R4 | eval `049_transition_flow_action_gridv3` — Flow MSE (`flow_u32@raft-r256-g24x32`) + Action KL (`action@swin3db-k400-u32`) of the reference clip vs the whole generated video (`flow_action_eval.py`, CPU, Pool(8)) |
| `pixel-endpoints` | R6 | eval `050_endpoint_pixel_gridv3` — PSNR/SSIM/LPIPS of the given endpoint frames (pixels, PyAV rgb24; `pixel_endpoint_eval.py`); shardable/resumable (`--shard i/n`, `--device cuda\|cpu`, per-arm O_EXCL locks), `--wait` login guarantor, `--finalize` meta |
| `finalize` | D5 | re-run the four draft evals with numbered ids 043-046 (identity / motion / smooth / text), `--no-index` |
| `tables` | D6 | rebuild `metrics_v2_gridv3{,_clean,_sweep,_sweep_clean}` (roster-driven; Transport = bl_333) |
| `paper-table` | D10 | render `papers_drafts/_preview/paper_table_v5/tab_main.tex` (+ preview.pdf) in the paper's tab_main style from the same `family_tables` levels; A10 = `paper_table_check.py` |
| `status` | — | FAST per-roster-arm namespace coverage (os.scandir, ~11 s) + per-eval 043-050 presence |

`blend.py` = the small library (populations/ceilings from `reference_v5` + `score_cache`); `seam_eval.py`
the seam script; `versioning.py` pins `REFERENCE_V5_SHA256`. `blend_grid.py` and the temporal-dynamics
math are imported, never modified.

## Eval numbers

043 endpoint identity · 044 endpoint motion · 045 smooth matched · 046 viclip text · 047 transport v5
(pairs + per-gen, one entry) · 048 seam · 049 transition flow+action (Flow MSE + Action KL, whole video) ·
050 endpoint pixel (PSNR/SSIM/LPIPS of the given endpoint frames).

## Round-1 scope

Only arms that already have a pair cache are scored in 047 (8 own labels = 16 harness arms incl. ED,
3 author-native, 3 TEG effect). The sweep arms (w=1.5, w=3) and the six neutral twins have no pair cache
yet — their Transport is `--` until Round 2 (`arms_pending` in the 047 meta). trackdesc covers every roster
gen that has stored cotracker3 tracks; the six neutral twins have none yet and are skipped in Round 1.

## Acceptance (re-runnable; see `misc/2026-09-23_metrics_v5/RECORD.md`)

- **A1** trackdesc extractor reproduces the imported cache (`np.array_equal`, float32).
- **A2** `reference_v5` rebuilds blend_grid's pop/fpop/ceil bit-identically (`v5.py reference`).
- **A3** 047 per-gen matches `BLEND_GRID_{cells,teg}_pergen.jsonl` to ≤1e-9 on all 10 weight columns.
- **A4** 048 seam z reproduces the v4 per-gen (028/041/042) at the physical windows; corrects evals/030.
- **A5** 043-046 rows equal the draft rows.
- **A6** the metrics_v2 tables are bit-identical to the current ones except Transport and Seam-free.

## Round 2 (2026-09-23) — the full roster + the paper-style table

`pairs` scores every roster arm without a cache from stored features and ADDs it to 047: the guidance
sweep arms (w=1.5, w=3), and — once their GPU features landed — the six neutral twins. **047
`arms_pending` is now `[]` (32 harness arms, every arm store-derived).** `paper-table` renders the
paper-style `papers_drafts/_preview/paper_table_v5/tab_main.tex` (+ preview.pdf). The GPU featurization of
the roster is the reviewer's sbatch, not a v5.py stage.

- **A8** — the pairs path reproduces an imported cache bit-exactly (`--gen-dino harness` → 0 on
  Mu/D/PX/fid/bl_333); the shipped `store` default equals it for every Round-2 arm (own gens store==legacy
  or no legacy; twins store-only).
- **A10** — every number in the paper table equals metrics_v2_gridv3_clean: final **204 matched cells,
  0 mismatches, 0 unmatched** (`paper_table_check.py`).

**Stage ordering / measurability rules for a new arm's twin or prompt row (learned in Round 2):**
- **Run `038` (handoff) BEFORE `045` (smooth_matched).** 045 reads the fps that 038 records; if 045 runs
  first for a just-added arm, its Smoothness stays `--` even though clip_b32 is present (the six twins hit
  this; fixed by re-running 045 after 038).
- **Motion (044) is not measurable on one-frame conditioning.** `scripts/endpoint_motion.py` applies the
  window rule (n_pre/n_suf < 2 ⇒ side `not_measurable`, `--`): a two-sided arm that happens to carry an
  extracted `raft_flow_win` still shows `--` for Motion start/end when its given clip is a single frame, so
  each neutral twin carries exactly its effect twin's dash pattern.

## Round 4 (2026-09-23) — Flow MSE + Action KL (eval 049)

Two whole-video transition-fidelity metrics of the paper tables, computed over the store:
- **Flow MSE** — RAFT-large optical flow over T=32 uniform steps (33 sampled frames) reduced to a 24×32 grid
  of the frame-diagonal fraction (`flow_u32@raft-r256-g24x32`); `flow_mse` = mean of `(100·(flow_gen −
  flow_ref))²` over `[32,24,32,2]`, unit `(% of frame diagonal)²`. Lower is closer.
- **Action KL** — Video Swin-B Kinetics-400 action-class distribution over 32 sampled frames
  (`action@swin3db-k400-u32`); `action_kl` = KL(p_ref ‖ p_gen) in nats (also `action_kl_rev`, `action_js`,
  and the magnitude-only `flow_mse_mag`). Lower is closer.

Both are whole-video (not the transition window); every video is sampled on its own at fixed fractions and
featurized once, pairing is a store lookup by the grid row's `reference` (`data/processed/transitions_std121/
<clip_class(reference)>/<reference>.mp4`). Extraction = two GPU arrays (24 shards each) over
`population_gridv3.json`; scoring = `v5.py flow-action` (eval 049, CPU, Pool(8)). Both columns are added to the
`metrics_v2_gridv3` Transition-fidelity group (order: Transport, Motion fid., Flow MSE, Action KL, Ref sim.,
Seam-free) and filled into the five paper tables by `paper_tables.py` (both lower-better).

## Round 6 (2026-09-24) — pixel fidelity of the given endpoint frames (eval 050)

Prior works take an image input; this reports the **pixel** fidelity (PSNR / SSIM / LPIPS) of the given
endpoint frames, a perceptual/pixel counterpart to the DINO identity of eval 043. Two pairings (frozen):

- **Frame-level (MAIN, comparable across every system):** out frame 0 vs the given `start9[0]` frame and, on
  two-sided rows, out frame T−1 vs the given `end9[8]` frame. `start9`/`end9` are the
  `eval_ladder/conds/<endpoint>_{start9,end9}.mp4` (480×640, 24 fps, 9 frames) for **every** grid type,
  VACE16 included — the cross-system comparison. These are `psnr_A/ssim_A/lpips_A` (+ `_B`).
- **Window mean (SECONDARY, clip-conditioned systems only; a within-system self-check, not the comparison):**
  the mean over the whole given window with the eval-043 pairing. `(n_pre, n_suf)` =
  `store_eval_common.windows(gtype, sided)`: HF 9/8, VACE16 6/4, ED + externals 1/1 (window ≡ frame). Start/end
  clips = `start9`/`end9` mp4 everywhere except VACE16, whose given frames are the lossless PNGs
  `misc/2026-09-20_teg_baselines/conds_16fps/<endpoint>_{start6,end4}_NN.png` (what the pipeline consumed).
  These are `psnr_Aw/ssim_Aw/lpips_Aw/n_Aw` (+ `_Bw`); they stay in eval 050 (not in family_tables).

Metrics on uint8 [0,255], full native resolution (PyAV rgb24, no resize; the given frame is bilinear-resized to
the output's size and `resized:true` recorded only on a shape mismatch — never here, all frames 640×480). PSNR
`10·log10(255²/MSE)`, MSE 0 ⇒ 100; SSIM Wang 2004 (11×11 Gaussian σ1.5, K1 .01, K2 .03, L 255, valid, per-RGB
mean, torch conv2d groups=3 float64, verified < 1e-6 against a numpy sliding-window loop); LPIPS `lpips.LPIPS(net='alex',
version='0.1')`, input `x/127.5−1`. **Caveat:** PSNR is capped by the H.264 encoding of both the output and the given
clips (~40–45 dB; a perfect copy through a VAE lands lower).

Compute is parallel and resumable: one `rows.jsonl` per harness arm, written atomically only when the arm is
complete; a `store/evals/<id>/_locks/<arm>.lock` (host+pid, O_EXCL) claims an arm so the ghx4 shard array
(`misc/2026-09-23_metrics_v5/jobs/eval050.sbatch`, `--shard $SLURM_ARRAY_TASK_ID/8 --device cuda --workers 12`) and a
login CPU race (`--device cpu --workers 8 --wait`) never double-score one (a lock older than 20 min with no complete
rows is stolen). Every row records `device`; GPU-vs-CPU LPIPS differ at ~1e-6. `v5.py pixel-endpoints --finalize`
writes `meta.yaml` (coverage + producing device per arm) once every arm is complete. The six pixel columns are added
to the **`input_fidelity_gridv3`** family view only (frame-level; the owner places them elsewhere); the paper-style
preview `tab_endpoint_pixel.tex` (+ `preview_pixel.pdf`, `LEVELS_pixel.md`) is `paper_tables.py --pixel-endpoint-table`,
**not** applied to the paper.
