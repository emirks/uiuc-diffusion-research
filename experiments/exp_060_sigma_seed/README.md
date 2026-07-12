# exp_060 — σ_seed measurement for the transition-eval harness (O6)

**Status: DONE 2026-07-14.** Generation 9488544 (H100, 2:01:16, 60/60 videos,
EXIT 0) → certified scoring 9488659 (5:35 warm-cache, 60/60 rows, 0 error
rows, every row stamped CERTIFIED transition-eval/3.0.0) → σ_seed table below
(`outputs/eval/sigma_seed/sigma_seed.json`). This experiment measures the
generation-seed noise of the decision-generating IC-LoRA arm so every future
1-seed model report can attach the σ_seed-derived minimum detectable effect
(MDE). It **calibrates the measured thing's noise**; it does not touch the
instrument or the certification record (σ_seed enters the record at the first
model report — an owner-reviewed step, SPEC §6.4).

## Question

What is the pooled between-seed standard deviation (σ_seed) of each headline
metric for the exp_056 IC-LoRA adapter, and the resulting paired-comparison
MDE at n = 5/10/20/40 items? (SPEC §4 seeds paragraph, §6.4 Block D;
`certify/bars.yaml` `stability.sigma_seed`: 12 items × 5 seeds, adapter arm,
pooled df, gates the first model report not the tag.)

## Setup

### Item selection (deterministic, corpus-only — no cherry-picking)

Rule (frozen in `select_items.py`, run before any GPU time):

1. Restrict to **n ≥ 4-eligible** classes (the exam's trust convention).
2. Strata = (twosided, onesided) × (camera-tagged, object-tagged,
   style/untagged), membership by **tag presence** ('style/untagged' =
   `style` in tags OR no tags). Fixed stratum order:
   twosided-{camera, object, style/untagged}, then onesided-{…}.
   (twosided-style/untagged is empty in this corpus.)
3. Fill 12 slots by **cycling the strata in that fixed order** (round-robin),
   each visit taking the **first-lexicographic** class in that stratum **not
   already picked** globally; skip empty/exhausted strata. (A stratum's second
   visit therefore takes its second-lexicographic class — the "second pass"
   clause.)
4. For each picked class the probe item = the class's **bar pair**
   (max-endpoint-distance pair) **exactly as certification defines it**
   (`certify/probes.sibling_pairs`), read verbatim from the certified
   certification artifact
   `outputs/eval/certification/3.0.0-draft.8/manifests/siblings.json`
   (bar-pair selection is corpus-only + pre-freeze, so the draft.8 artifact is
   byte-identical to the `eval/v3.0.0` tag). **Clip A** = `sib__<class>`'s
   `generated_video` (endpoints / conditioning source), **clip B** =
   `reference_video` (the reference).

The 12 picks (see `dataset/selection.json`):

| # | class | stratum | clip A (cond) | clip B (ref) | n |
|---|-------|---------|---------------|--------------|---|
| 1 | air_bending | twosided-camera | air_bending_2 | air_bending_3 | 4 |
| 2 | earth_wave | twosided-object | earth_wave_0 | earth_wave_4 | 5 |
| 3 | earth_element | onesided-camera | earth_element_4 | earth_element_6 | 6 |
| 4 | animalization | onesided-object | animalization_1 | animalization_3 | 8 |
| 5 | fire_element | onesided-style/untagged | fire_element_1 | fire_element_4 | 4 |
| 6 | firelava | twosided-camera | firelava_0 | firelava_3 | 6 |
| 7 | melt_transition | twosided-object | melt_transition_1 | melt_transition_2 | 4 |
| 8 | hero_flight | onesided-camera | hero_flight_6 | hero_flight_8 | 10 |
| 9 | color_rain | onesided-object | color_rain_0 | color_rain_8 | 8 |
| 10 | illustration_scene | onesided-style/untagged | illustration_scene_1 | illustration_scene_2 | 10 |
| 11 | flying_cam_transition | twosided-camera | flying_cam_transition_2 | flying_cam_transition_3 | 4 |
| 12 | raven_transition | twosided-object | raven_transition_1 | raven_transition_3 | 4 |

Seeds: **42, 43, 44, 45, 46** → 12 × 5 = **60 videos**, adapter arm only.

### Adapter provenance (the decision-generating arm)

- **Checkpoint:** `outputs/training/exp_056_ltx2_ic_lora_transition_transfer/ic/checkpoints/lora_weights_step_03000.safetensors`
  — the exp_056 IC-LoRA, step 3000; the exact adapter scored in the archived
  exp_057 generations (exp_057 README + `run_ic_inference.py`: `LORA=…/lora_weights_step_03000.safetensors`).
  LoRA r=32, α=32, attn+FFN target modules.
- **Base model:** `cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors`;
  text encoder Gemma-3-12B QAT-q4. Inference via the ltx-trainer
  `ValidationRunner` (`uv run --frozen`), identical to exp_057.
- **Recipe (exp_057 verbatim):** 480×640×121 @ 24 fps, 30 steps, CFG 4.0,
  STG 1.0 `stg_v` blocks [29], prefix 9f + suffix 8f, negative prompt
  "worst quality, inconsistent motion, distorted, jittery", prompt =
  `ICTRANS ` + type-blind endpoint caption of clip A.
- **Captions:** 11/12 clip-A captions reused from exp_056/057/058 caption
  files; `illustration_scene_1` was absent from all three and regenerated with
  the **exact exp_058 captioner** (`caption_missing.py`: same PROMPT, Gemini
  `gemini-3.5-flash`, temp 0, two-sentence "…. The scene transforms into …."
  format). See `dataset/captions_extra.json`.

### Generation contract vs scoring contract

Conditions at generation = clip A's `start9.mp4` (first 9 frames) +
`end9.mp4` (last 9 frames, consumed as 8) + clip B as reference
(`include_in_output=False`) — `make_conds.py` cuts them with the exp_057
ffmpeg recipe. For **scoring**, the eval manifest points `condition_prefix` /
`condition_suffix` at the **full clip A** (num_frames 9 / 8); score.py slices
first-9 / last-8 — the identical contract the certified sibling manifest uses,
so M3a endpoint fidelity compares the generation against clip A's endpoints.

## How to run

```bash
# --- prep (login node / diffusion env) ---
python experiments/exp_060_sigma_seed/select_items.py      # -> dataset/selection.json
python experiments/exp_060_sigma_seed/caption_missing.py   # fills any missing clip-A caption
python experiments/exp_060_sigma_seed/make_conds.py        # -> dataset/cond/*.mp4
python experiments/exp_060_sigma_seed/build_manifest.py    # -> dataset/eval_manifest.json + probe_groups.json

# --- generate 60 videos (one H100, single model load, skip-if-exists) ---
sbatch -p HCESC-H100-normal -A hcesc-h100 --gres=gpu:H100:1 --time=03:30:00 \
  experiments/exp_060_sigma_seed/job_infer.sbatch                          # job 9488544

# --- score with the CERTIFIED harness (eval/v3.0.0 worktree) ---
sbatch -p HCESC-H100-normal -A hcesc-h100 --gres=gpu:H100:1 \
  experiments/exp_060_sigma_seed/job_score.sbatch

# --- compute sigma_seed (login node, numpy-only, worktree src) ---
WT=$LAB/diffusion-research/.claude/worktrees/eval-v3.0.0
PYTHONPATH=$WT/src python experiments/exp_060_sigma_seed/compute_sigma.py \
  --items $LAB/diffusion-research/outputs/eval/sigma_seed/adapter/items.jsonl \
  --out   $LAB/diffusion-research/outputs/eval/sigma_seed/sigma_seed.json
```

## Outputs

- `dataset/selection.json` — the 12 picks + rule + bar pairs + captions.
- `dataset/eval_manifest.json` — 60-row eval manifest; `dataset/probe_groups.json` — item_id → class.
- `outputs/videos/exp_060_sigma_seed/adapter/sigseed__<class>__s<seed>.mp4` — 60 generations.
- `outputs/eval/sigma_seed/adapter/{items.jsonl,results.json}` — certified scoring (60 rows).
- `outputs/eval/sigma_seed/sigma_seed.json` — per-metric pooled σ_seed + MDE@n=5/10/20/40 + audit.

## Job IDs & runtimes

- Inference: `9488544` (HCESC-H100-normal, 1× H100) — **2:01:16** (≈16 min
  model load + ~95 s/video × 60; single model load for all 60).
- Scoring: `9488659` (HCESC-H100-normal, afterok-chained) — **5:35**
  (corpus features fully warm from the shared cache; only the 60 generations
  featurized fresh).
- Total ≈ **2.1 GPU-hours** (budgeted ~6).

## Results

`seeds.sigma_seed` (certified `certify/seeds.py`, eval/v3.0.0) over 12 probe
groups × 5 seeds; pooled between-seed std (RMS of per-group ddof=1 stds),
MDE(n) = 1.96 · σ · √(2/n) for a paired comparison over n items:

| metric | σ_seed | MDE n=5 | n=10 | n=20 | n=40 |
|---|---|---|---|---|---|
| app_ref (M1a) | 0.0271 | 0.0336 | 0.0238 | 0.0168 | 0.0119 |
| margin (M2b) | 0.0427 | 0.0530 | 0.0374 | 0.0265 | 0.0187 |
| copy_max (M2a) | 0.0251 | 0.0312 | 0.0220 | 0.0156 | 0.0110 |
| cam_dtw (M1b) | 0.0864 | 0.1071 | 0.0757 | 0.0536 | 0.0379 |
| obj_match (M1c) | 0.0091 | 0.0112 | 0.0079 | 0.0056 | 0.0040 |
| max_seam_z (M3b) | 0.3036 | 0.3763 | 0.2661 | 0.1882 | 0.1330 |

- **df statement (per bars.yaml): POOLED** — 12 groups × (5−1) = 48 df for
  fully-defined metrics; never imply stratified precision.
- **Audit (all in `sigma_seed.json` `_audit`):** 60/60 rows scored, 0 error
  rows, all rows certified; every probe_group has 5/5 finite `app_ref`.
  `cam_dtw` NaN pattern (reported, never imputed — SPEC §3 M1b/M1c):
  air_bending 4/5, earth_wave 4/5 finite; **raven_transition 0/5** (camera
  fit invalid on all its generations) — cam_dtw σ therefore pools over 11
  groups (~42 df). All other metrics 5/5 finite in all 12 groups.
- Base arm assumed same-family for σ purposes (bars.yaml
  `stability.sigma_seed.arm: adapter`, documented there).
- σ_seed **gates the first model report, not the tag** — it enters the
  certification record at that owner-reviewed step; nothing under
  `src/diffusion/transition_eval/` was touched by this experiment.

Reading: appearance/copy/margin deltas ≳ 0.02–0.04 raw are resolvable at
routine suite sizes (n≈10–20 paired items, 1 seed); `max_seam_z` is by far
the noisiest headline (σ 0.30 — consistent with its SPEC §3 "flag, not a
ranker" role); `obj_match` is remarkably seed-stable (σ 0.009).
