# From a new generation to the metric table — the exact steps

Reproduces `papers_drafts/_preview/metrics_v2_gridv3/preview.pdf` for a new arm on grid v3, same settings as the
paper tables (2026-09-19). Everything below runs on DeltaAI; `PY=$LAB/envs-aarch64/ltx2/bin/python`;
GPU jobs charge `bgjg-dtai-gh` or `bhwp-dtai-gh` (pick by `sshare` LevelFS). Repo: `$LAB/diffusion-research`,
branch `feature-store`.

## 0. Store contract (what a gen entry looks like)

```
store/gens/NNN_<arm>/KK_<variant>__<machine>/
  videos/<cell>__<harness_arm>__<endpoint>__ref_<reference>__s<seed>.mp4   # 121 f (HF) / 81 f (ED), 24 fps
  videos/SHA256SUMS                                                          # tracked
  grid.jsonl        one row per (cell, endpoint, reference): item_id, arm(=harness_arm), cell, endpoint, reference,
                    sided (one|two), ref_novelty (seen|unseen|zero_shot), prompt, gt_pool_class, ...
  meta.yaml         arm, harness_arm, variant, machine, inputs{run,step}, code, prompt_family/sha, videos, features
  features/<stem>/<namespace>.npz + .json                                    # written by step 3
  scores -> ../../../evals/<v4 eval>/<harness_arm>                           # written by step 4
```
`harness_arm` = `<arm>_<tier>_v3[ed81]` (e.g. `dualforce_dcg_w6_neutral_v3`); it is the 2nd `__` field of every
video stem and the join key of every eval. One arm on grid v3 = 4 subentries: neutral/effect × HF/ED.

## 1. Generate + register (grid v3)

Add the arm to `ARMS` in `scripts/grid_v3/launch_gen.py` (gen_dir, run, step, code, account, next_kk), then:
```
$PY scripts/grid_v3/launch_gen.py stamp   --arms <arm>      # registries + arms.yaml entries
$PY scripts/grid_v3/launch_gen.py prepare --arms <arm>      # subentries; hardlinks the 139 kept rows
$PY scripts/grid_v3/launch_gen.py submit  --arms <arm>      # sbatch arrays (neutral wave, then effect)
$PY scripts/grid_v3/launch_gen.py status  --arms <arm>
$PY scripts/grid_v3/launch_score.py register --arms <arm>   # store_register.py per subentry + store_fsck
```
Videos made elsewhere: drop them into `videos/` with the stem grammar above, copy the registry to `grid.jsonl`, and
run `scripts/store_register.py gen <subentry> --registry <registry.jsonl> --run <run> --step <n> --code "<sha>"`.

## 2. Hashes + population

```
$PY scripts/store_features.py sha256sums --gens 'store/gens/NNN_<arm>/0*_v3*__dai'
```
Add the 4 subentries to `misc/2026-09-17_feature_store/population_gridv3.json` (`gen_variants`, `n_gens`); the HF
neutral+effect subentries also to `population_flowwin.json`; all 4 to `population_gridv3_gens.json`.

## 3. Embeddings (GPU, one array; resumable, fills only misses)

```
cd misc/2026-09-17_feature_store
sbatch --account=<acct> --qos=<acct> --array=0-3 --time=02:00:00 \
  --export=ALL,NSHARDS=4,POP=misc/2026-09-17_feature_store/population_gridv3.json,VBENCH_CACHE_DIR=$LAB/cache/vbench,\
NS="dino_cls@dinov2b-r256 cotracker3@g20-m384-v2 lpips_t@alex-r256 clip_b32@r256 clip_l14@r224 videoprism@f16r288 raft_mag@r256 viclip@l14-f8" \
  jobs/extract.sbatch
sbatch ... --export=ALL,NSHARDS=4,POP=misc/2026-09-17_feature_store/population_flowwin.json,NS=raft_flow_win@r256 jobs/extract.sbatch
# metrics v5 R4 (eval 049): two whole-video namespaces, ONE array each, 24 shards, --time=00:45:00 (no throttle);
# download swin3d_b_22k-7c6ae6fa.pth into TORCH_HOME=$LAB/cache/torch/hub/checkpoints on the LOGIN node first (compute nodes are HF_HUB_OFFLINE=1).
sbatch --account=<acct> --qos=<acct> --array=0-23 --time=00:45:00 --job-name=fs_flow_u32 --export=ALL,NSHARDS=24,POP=misc/2026-09-17_feature_store/population_gridv3.json,NS=flow_u32@raft-r256-g24x32 jobs/extract.sbatch  # RAFT-large, ~1.2 s/video
sbatch --account=<acct> --qos=<acct> --array=0-23 --time=00:45:00 --job-name=fs_action  --export=ALL,NSHARDS=24,POP=misc/2026-09-17_feature_store/population_gridv3.json,NS=action@swin3db-k400-u32   jobs/extract.sbatch  # Video Swin-B, ~5 min/shard
$PY scripts/store_features.py coverage --population misc/2026-09-17_feature_store/population_gridv3.json --md
```
Namespaces are frozen pins (`store/FEATURES.md`); never change a tag's settings, add a new tag instead.

## 4. Transport + seam z (metrics v5, CPU over stored features)

Since 2026-09-23 the paper's **Transport** column is the three-channel blend (appearance Mu / semantic
transport D / pixel transport PX, equal thirds, % of the class ceiling) recorded in eval **047**, and
**Seam-free / seam z** is recomputed from the stored temporal LPIPS at the physical windows in eval **048**
— no v4 harness pass for the table. Driver: `scripts/metrics_v5/v5.py`.

```
# reference pin (once): bundles reference_v5.npz + versioning.py, verifies pop/fpop/ceil
$PY scripts/metrics_v5/v5.py reference
# Transport (eval 047): imports the mf_pair_cache pair distances, scores the blend against reference_v5
$PY scripts/metrics_v5/v5.py blend
# Seam (eval 048): seam z from lpips_t@alex-r256 at the physical windows (store_eval_common.windows)
$PY scripts/metrics_v5/v5.py seam
# Flow MSE + Action KL (eval 049): whole-video optical-flow MSE + Kinetics-400 action KL vs the reference clip
$PY scripts/metrics_v5/v5.py flow-action
# Pixel PSNR/SSIM/LPIPS of the given endpoint frames (eval 050, R6): shardable GPU array + login CPU race, per-arm locks
sbatch --array=0-7 misc/2026-09-23_metrics_v5/jobs/eval050.sbatch   # 8 ghx4 shards, LPIPS cuda, 12 CPU-decode workers
$PY scripts/metrics_v5/v5.py pixel-endpoints --device cpu --workers 8 --wait   # login guarantor (races the array; cooperates via the locks)
$PY scripts/metrics_v5/v5.py pixel-endpoints --finalize                        # meta.yaml once every arm has rows.jsonl
# Transport for a brand-new arm (no mf_pair_cache): compute Mu/D/PX from stored features and ADD to 047
$PY scripts/metrics_v5/v5.py import-trackdesc      # first: trackdesc@cotracker3-s64-v1 from the stored tracks
$PY scripts/metrics_v5/v5.py pairs                 # per pending roster arm whose features are ready (skip-if-done)
```

`v5.py pairs` uses score_v3_mf's channel code (pillars raw_pair, fid, dist_concat) but rows come from the
roster: an own arm via `score_v3_cells.load_rows` (+ ic_gen completion); a neutral twin (roster `twin`
field) inherits its effect sibling's (endpoint, reference, seed) -> pool-refs map. Gen DINO from
`dino_cls@dinov2b-r256`, dirs/px from `trackdesc@cotracker3-s64-v1`, PX z-stats from `reference_v5.npz`
(`--gen-dino store` default; `--gen-dino harness` reproduces a legacy import bit-exactly, A8). It merges the
arm into 047's meta and drops it from `arms_pending`; existing arms are never touched. An arm with no pair
cache and no computed pairs renders Transport `--`. The v4 `transport_pct` (evals 028/030/041/042) is kept
only for the semantic_transport preview families (`transport_v4`).

## 5. Hand-off / copy / lenses (evals 038, 039, 040)

Rerun over the full population: idempotent, only the new arm is computed, `meta.yaml` stays complete.
```
$PY scripts/handoff_metrics.py --population misc/2026-09-17_feature_store/population_gridv3.json --eval-id 038_handoff_gridv3__dai__2026-09-18
$PY scripts/copy_metrics.py    --population misc/2026-09-17_feature_store/population_gridv3.json --eval-id 039_copy_gridv3__dai__2026-09-18
sbatch --account=<acct> --qos=<acct> --array=0-7 --export=ALL,NSHARDS=8,EVAL_ID=040_lenses_gridv3__dai__2026-09-18 \
  misc/2026-09-17_feature_store/jobs/lens_pass.sbatch          # GPU; skip-if-exists per gen
$PY scripts/lens_pass_gridv3.py --collect --eval-id 040_lenses_gridv3__dai__2026-09-18
$PY scripts/aesthetic_from_store.py --eval-id 040_lenses_gridv3__dai__2026-09-18
```

## 6. Endpoint identity / endpoint motion / smoothness / text (draft evals, CPU, minutes)

```
export PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1 VBENCH_CACHE_DIR=$LAB/cache/vbench
$PY scripts/endpoint_identity.py --eval-id endpoint_identity_gridv3__dai__2026-09-18
$PY scripts/endpoint_motion.py   --eval-id endpoint_motion_gridv3__dai__2026-09-19
$PY scripts/smooth_matched.py    --eval-id smooth_matched_gridv3__dai__2026-09-18
$PY scripts/viclip_text.py       --eval-id viclip_text_gridv3__dai__2026-09-19
```
Unnumbered id = draft under `store/evals/_draft/`, no index line. To finalize, give a numbered id.

**Two rules for a just-added arm (metrics v5 Round 2, 2026-09-23):**
- Run **038 (handoff) BEFORE 045 (smooth_matched)** — 045 reads the fps that 038 records; running 045
  first leaves that arm's Smoothness `--` even with clip_b32 present.
- **Motion (044) is `--` on one-frame conditioning** — `scripts/endpoint_motion.py` applies the window
  rule (n_pre/n_suf < 2 ⇒ side `not_measurable`): a two-sided arm that carries an extracted
  `raft_flow_win` still shows `--` for Motion start/end when its given clip is a single frame, so a
  neutral twin carries exactly its effect twin's dash pattern.

## 7. Tables

Table rows come from `scripts/metrics_v5/roster.json` (metrics v5) — add the arm there (`id`, `role`,
`prompt`, `gens`, `label`/`tex`), not to `family_tables.py`'s code. Then:
```
$PY scripts/metrics_v5/v5.py tables            # rebuilds metrics_v2_gridv3{,_clean,_sweep,_sweep_clean}
# or directly:
$PY scripts/family_tables.py                                   # 4 folders under papers_drafts/_preview/
$PY scripts/family_tables.py --clean --families metrics_v2_gridv3   # the PI version
$PY scripts/family_tables.py --sweep                           # + the DCG guidance-weight rows (roster sweep:true)
```
Output: `papers_drafts/_preview/metrics_v2_gridv3{,_clean,_sweep,_sweep_clean}/preview.pdf`, `tab_{1,2,3}.tex`, `TABLES.md`.

The paper-style main table (the exact style of `papers_drafts/ctt_iclr2027/tables/tab_main.tex`):
```
$PY scripts/metrics_v5/v5.py paper-table                # papers_drafts/_preview/paper_table_v5/{tab_main.tex, preview.pdf}
$PY scripts/metrics_v5/paper_table_check.py             # A10: every number == metrics_v2_gridv3_clean
```

**A new arm end-to-end = one roster line + the v5.py stages.** Add it to `scripts/metrics_v5/roster.json`
(`id`, `role`, `prompt`, `gens`, `label`/`tex`, optional `twin`/`sweep`), generate + featurize (steps 1-3),
then run, in order: `v5.py import-trackdesc` -> `v5.py seam` -> `handoff_metrics`/`copy_metrics` (038/039)
-> `v5.py finalize` (043-046; keep 045 AFTER 038) + the lens pass collect + `aesthetic_from_store` (040) -> `v5.py pairs` (047) ->
`v5.py tables` -> `v5.py paper-table`. Every stage is idempotent / skip-if-done; poll with `v5.py status`.

## What each column reads

| column | eval | script |
|---|---|---|
| Identity Start / End | 043 endpoint identity | `endpoint_identity.py` |
| Motion Start / End | 044 endpoint motion | `endpoint_motion.py` |
| Consistency (text) | 046 viclip text | `viclip_text.py` |
| Transport | **047** transport v5 `per_gen.jsonl` (`bl_333`) | `metrics_v5/v5.py blend` (+ `blend.py`) |
| Motion fid., Ref. sim., Dynamics, Aesthetic | 040 | `lens_pass_gridv3.py` (+ `aesthetic_from_store.py`) |
| Seam-free % | **048** seam (physical windows) | `metrics_v5/seam_eval.py` |
| Flow MSE | **049** transition flow+action (`flow_mse`) | `metrics_v5/flow_action_eval.py` (from `flow_u32@raft-r256-g24x32`) |
| Action KL | **049** transition flow+action (`action_kl`) | `metrics_v5/flow_action_eval.py` (from `action@swin3db-k400-u32`) |
| Smoothness | 045 smooth matched | `smooth_matched.py` |
| (family view: input fidelity) Pixel PSNR / SSIM / LPIPS, start & end | **050** endpoint pixel (R6; frame-level; window means in 050) | `metrics_v5/pixel_endpoint_eval.py` |
| (family views) hand-off id/motion, copy, seam z, CLIP ref sim, 8 fps smoothness | 038, 039, 040, 045, 048 | same scripts |
