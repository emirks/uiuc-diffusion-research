# Phase A report — null-default campaign (operator, 2026-09-13)

Facts only; no readings or verdicts. Phase A (manifest + PIX + smoke) and Phase B prep (extract
script + sbatch) are done. No Slurm job was submitted; nothing committed. Env for all python:
`source $LAB/envs-aarch64/activate`. Login node `gh-login03`.

## 1. Manifest — `manifest.csv` (749 clips)

Built by `scripts/manifest.py` (reuses the Sep-08 `per_clip.csv` join for the S-GRID family, so
md5s match the regression check by construction; GT/probe/sweep md5 computed fresh, 0 errors).

Per stratum × group:

| stratum | group | n |
|---|---|---|
| S-GRID | NULL | 68 |
| S-GRID | GT | 19 |
| S-GRID-F | NULL | 38 |
| S-GRID-START | NULL | 204 |
| S-PROBE | R1 | 120 |
| S-PROBE | R2 | 120 |
| S-PROBE | R3 | 120 |
| S-SWEEP | A_empty / A_word / B / C / D / E | 10 each (60) |

S-GRID family × grid: S-GRID v2=30 v3=38 · S-GRID-F v2=6 v3=32 · S-GRID-START v2=82 v3=122.
S-PROBE tier × group: `high` R1/R2/R3 = 90 each (30 scene-change prompts × 3 seeds); `inplace`
R1/R2/R3 = 30 each (10 in-place prompts × 3 seeds). tier is `''` for every non-probe row.

Windows recorded (0-based, verified against the instrument/probe/sweep contracts):
- S-GRID NULL, S-GRID-F, GT twins: T=121, a_idx=8, b_idx=113, two_sided=True.
- S-GRID-START: T=121, a_idx=8, b_idx=120, two_sided=False.
- S-PROBE R1: a_idx=8, b_idx=120 (one-sided); R2/R3: a_idx=8, b_idx=113 (two-sided).
- S-SWEEP: T=193 (verified per file), a_idx=24, b_idx=168, two_sided=True (25 cond frames/end).

## 2. Dedup and foreign counts

base_cond ignores the reference, so rows sharing (endpoint, seed) are byte-identical; md5-dedup
collapses them. Raw rows → unique md5 (each md5 group has exactly one endpoint and one seed):

| variant | condition | clean rows→uniq | foreign rows→uniq |
|---|---|---|---|
| 02_neutral (v2) | both | 52 → **30** | 28 → 6 |
| 02_neutral (v2) | start | 160 → **82** | 64 → 6 |
| 04_neutral_v3 (v3) | both | 94 → **38** | 54 → 32 |
| 04_neutral_v3 (v3) | start | 278 → **122** | 138 → 80 |

- S-GRID (both, clean) = 30 v2 + 38 v3 = **68** — matches the ledger (v2 30, v3 38).
- S-GRID-F (both, davis-endpoint) = 6 v2 + 32 v3 = **38** (foreign = `endpoint_source=='davis'` or
  `endpoint.startswith('davis_')` or `'foreign' in item`, i.e. score_store.is_foreign).
- S-GRID-START (start, clean) = 82 v2 + 122 v3 = **204**. NOTE: the ledger quotes 80/119 in §2.1
  but 82 (v2) in §4.1 (the distance_vs_cut addendum); my unique-md5 dedup gives 82/122, matching
  §4.1. I did not reconcile the §2.1 80/119 — likely a slightly different dedup scope there.
- GT twins = **19** unique endpoints used by S-GRID (both, clean); all found by recursive glob of
  `data/processed/transitions_std121/**/<endpoint>.mp4` (0 missing).
- Foreign start-only rows exist (6 v2 + 80 v3 unique) but are NOT included: `foreign` is a
  two-sided distinction (S-GRID-F); start-only has no end anchor / GT twin. Flagged for the driver
  to override if a start-only foreign stratum is wanted.

`static_pix` (PIX gap_rel < 0.12, SPEC 3) filled by pix_features: **8 / 749** static, all in
S-GRID-START (near-static start-only clips). Counted, not excluded.

## 3. Probe window facts verified

Probe contract (from `_probe_common.py` / `splice_r1.py` / `score_probe.py`): prefix 9 px
(a_idx=8), suffix 8 consumed (b_idx = T−8 = 113) for the two-sided R2/R3; R1 is one-sided
(b_idx=120). The end anchor is `cut_windows(R1)` frames 112..120 spliced back as `..._end9.mp4`.

Pixel-equality check R1[112:121] vs R3 tail (3 pairs, MAE on 0–255):

    P01H s42: same-index R3[112:121] MAE=6.43 = best over offsets 108..112
    P01H s43: same-index R3[112:121] MAE=7.23 = best
    P01H s44: same-index R3[112:121] MAE=6.38 = best

Reading of the numbers (operator): the best alignment is at the SAME frame indices (no shift),
consistent with b_idx=113; the tail is NOT byte-identical (MAE ≈ 6–7 / 255 ≈ 2.5–3%), consistent
with R3's suffix being the VAE-conditioned reconstruction of R1's ending, not a byte copy. The
a_idx/b_idx the probe contract implies (a=8, b=113 for R2/R3; b=120 for R1) are what the manifest
records.

## 4. Regression check (SPEC 3) — `pix_per_clip.csv`

PIX DR / M / step_share for the 310 matched S-GRID-family main clips (matched by md5 to the
Sep-08 `per_clip.csv` DR_med/M and the distance_vs_cut `cutstats.csv` step_share):

    DR   : n=310, max |dev| = 9.7e-17
    M    : n=310, max |dev| = 9.7e-17
    step_share : n=310, max |dev| = 9.7e-17

i.e. machine epsilon — byte-for-byte the instrument. (This required a SERIAL decode; see §7.)

## 5. Landmark self-checks (SPEC 4) — n=441 landmark-source clips each

| landmark | DR (med / max) | cross (med) | step_share (med) |
|---|---|---|---|
| LERP | 3.6e-7 / 2.2e-5 | 0.500 | 0.0089 |
| CUT50 | 0.0 / 6.8e-6 | 0.503 | 1.000 |
| FREEZE | 0.0 / 0.0 | 1.000 | 1.000 |

LERP DR ≪ 0.02 bar; CUT50 cross ≈ 0.5 and step_share ≈ 1; FREEZE is a single jump (step_share 1,
never crosses → cross 1.0). Landmarks are built at native resolution from each source clip's own
frames a_idx/b_idx and pushed through the same 128px/blur pipeline.

## 6. CPU smoke (SPEC A4) — `extract_features.py --device cpu`

Ran DINO + TRANS on `null_v2__hero_flight_1__s42` (NULL, main only) + `GT__hero_flight_1` (GT,
landmark-source → main + LERP/CUT50/FREEZE): 2 clips → 5 npz in 168 s (CPU). Per-file:

    dino  : (121, 768) float16, L2-norm(row)=1.0000   [SPEC (121,768)]
    trans : (16, 20, 15, 44) float32, all finite      [SPEC (16,20,15,44) for 480w×640h]
    a_idx/b_idx/T stored; TRANS behaves sensibly (0-based ch40=nu: CUT50/FREEZE nu=0, LERP nu>0).

VAE was NOT run on CPU: the encoder checkpoint is `ltx-2-19b-dev.safetensors` (43 GB) and a CPU
encode of one clip would far exceed the 10-min budget (and the RAM ceiling). VAE is verified only
by code inspection against `eval_ladder/train/precompute.py` (same `load_vae`/`encode` calls) and
runs on GPU in Phase B.

## 7. Phase B estimate (`job_extract.sbatch`, ghx4, 1 GPU)

- Main clips: **749**. Landmark-source clips: **441** (GT 19, S-GRID-F 38, S-GRID-START 204,
  S-PROBE R3 120, S-SWEEP 60). NULL/R1/R2 carry no own landmarks (they reference the GT twin /
  R1 witness / R3 anchors per SPEC 1).
- Phase B npz files: **2513** = 749 main + 441 × (LERP + CUT50 + FREEZE + LATLERP).
- Heavy extraction units (DINO-CLS + TRANS + VAE): **2072** = 749 + 441×3. The 441 LATLERP are
  cheap latent interpolations of the main clip's own VAE latents (no extractor).
- Time: TRANS is **~1.1 s/clip** measured on GH200 (armA precedent: 132 clips/164 s, 793/840 s).
  DINO-CLS ≈ 0.5 s/clip (est). VAE encode ≈ 3–4 s/clip (EST, unmeasured — the one uncertain term).
  → ≈ 4.5–6 s/heavy unit → **≈ 2.5–3.5 GPU-h total** + one-time model load (~1–2 min). This
  **exceeds the 2 h single-job wall**, so the sbatch supports `--shard i/n` and `--only-missing`;
  recommend a **2–3 way array** (uncomment `#SBATCH --array=0-2`, set `SHARDS=3`) so each task is
  ~1–1.5 h. Driver must set `--account` (`bgjg-dtai-gh` or `bhwp-dtai-gh` by FairShare).

## 8. Things to know / could not verify

1. **`group` value "NULL" ↔ pandas NaN (IMPORTANT).** "NULL" is in pandas' default na_values; a
   plain `pd.read_csv(manifest.csv)` silently turns every NULL group into NaN (it already clobbered
   the file once mid-run). The canonical reader `scripts/common.load_manifest` reads with
   `keep_default_na=False`. **Phase C must read via `common.load_manifest` (or `keep_default_na=
   False`), or rename the group.** `pix_per_clip.csv` also carries "NULL" and has the same hazard.
2. **Login-node decode contention.** cv2/ffmpeg h264 decode on gh-login03 corrupts frames under
   parallel load (matches the ledger §4.6 note; the Sep-08 scorer ran on a compute node for this
   reason). A first `pix_features --nproc 4` run silently corrupted 22/310 clips + failed 15;
   `--nproc 1` (serial) is clean (regression at machine epsilon, 0 errors). pix_features is left at
   the serial default for the login node.
3. **VAE decoder path (deviation, flagged).** extract_features builds the VAE pixel tensor from
   the same cv2 frames as DINO/TRANS and calls `encode_conditioning.load_vae`+`encode` (the repo's
   encoder), rather than `preprocess()`'s PyAV reader — necessary because preprocess caps at 121
   frames (S-SWEEP is 193) and cannot do the downscale or the synthetic landmark frames. For the
   121-f clips this is a sub-LSB decode difference from PyAV; it keeps a clip's main and landmark
   latents on one decoder. VAE itself is unverified at runtime (no GPU run in Phase A).
4. **SPEC 2.4 TRANS channel indices are 1-based** relative to armA's 0-based `CH_NAMES` (SPEC
   "ch41 nu" = 0-based channel 40 = `nu`; "ch39/40 sA/sB" = 0-based 38/39; "ch33–35 u,v,conf" =
   0-based 32,33,34; "ch36 ent"=35; "ch37 inplace"=36; "ch38 dir"=37; "ch42 dlab"=41). Verified by
   the smoke (CUT50/FREEZE 0-based ch40 nu = 0, LERP nu > 0). Phase C descriptor code must use the
   0-based indices. The stored `trans` field carries all 44 channels, so no re-extraction needed.
5. S-SWEEP is downscaled to 768×512 (W×H) before DINO/TRANS/VAE (and before its PIX 128 matrix);
   S-GRID/S-PROBE/GT stay native 480×640. T_lat = 25 for the sweep, 16 for the 121-f clips.
6. Not done (out of Phase A/B-prep scope): the GPU extraction run (Phase B) and all of Phase C.
