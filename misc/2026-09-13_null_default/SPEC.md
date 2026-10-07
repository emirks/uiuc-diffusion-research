# SPEC — the base model's default transition, measured in four spaces
Campaign dir: `misc/2026-09-13_null_default/`. Written 2026-09-13 by the driver (Fable); implemented by an opus48 operator.
Read together with `misc/2026-09-13_null_ledger/LEDGER.md` (every prior number) — nothing there is inherited as a conclusion.

## 0. Question
What trajectory does base LTX-2 (no adapter) produce between two anchors when the text says nothing about the transition?
Characterise it as a DISTRIBUTION of trajectories, in four feature spaces, against three kinds of reference:
(i) real trajectories with the same endpoints (GT twin), (ii) endpoint-explainable landmarks built from the same anchors
(lerp / cut / freeze / latent-lerp), (iii) the model's own witness trajectory where one exists (probe R1).
No transition family is assumed. Descriptors are family-agnostic. Every reading in the report is attributed ("the operator reads…"); no verdicts.

## 1. Strata — never pooled for an estimate, always reported side by side
The two main sources differ in ways that would confound a pooled estimate, so they are separate strata whose AGREEMENT is the result:

| stratum | clips | end anchor | text | seeds | reference |
|---|---|---|---|---|---|
| S-GRID | base_cond neutral two-sided, grid v2 `store/gens/005_base_cond/02_neutral__dai` + v3 `04_neutral_v3__dai`; md5-dedup (rows sharing an endpoint are byte-identical) | REAL: frames 113–120 of the GT transition clip | start caption only ("{S1}.") | 42, 43 | GT twin = the same GT clip `data/processed/transitions_std121/<class>/<endpoint>.mp4` (recursive glob), scored with the same window; landmarks from its frames a_idx/b_idx |
| S-GRID-F | the davis-endpoint ("foreign") rows of the same gens | real frames, but no GT transition exists | same | same | landmarks only |
| S-GRID-START | base_cond neutral START-only rows, v2+v3 (dedup) | none (scored to own last frame, b_idx = T−1) | same | same | landmarks from own frames a_idx / T−1 |
| S-PROBE | `misc/2026-09-08_collapse_probe/out/r3/...` R3 (both anchors, captions only); R2 (both anchors, full prompt) and R1 (start only, full prompt) as comparators | MODEL-MADE: R1's own last 9 frames | start caption + end caption | 42, 43, 44 | R1 witness (same start, and R1[112:121] == R3's suffix by construction — verify byte/pixel equality); landmarks from R3's anchors. Keep tier `high` (scene change, 30 prompts) and the in-place tier (10 prompts) SEPARATE. |
| S-SWEEP | `outputs/videos/exp_024_ltx2_prompt_sweep/run_0003/<pair>/<cat>/*.mp4` (10 DAVIS pairs × 6 categories × seeds); 193 f, 1024×1536, CFG 3.2, 25 conditioning frames each end | real DAVIS frame | A_empty = "" · A_word = "transition" · B generic · C abstract · D typed · E timed | per file | landmarks only; A_empty/A_word are the null candidates, B–E the text-informed comparators. Different config → side panel, never merged with S-GRID/S-PROBE. |

Conflation guard: S-GRID vs S-PROBE differ in (a) end-anchor provenance, (b) prompt form, (c) endpoint-distance distribution, (d) seed count.
Where the two strata agree on a descriptor, the default is robust to (a)–(d). Where they disagree, the report attributes the gap to (a)–(d) — not to "the null".

Windows (frame indices, 0-based): S-GRID/S-GRID-F/S-PROBE two-sided: a_idx = 8, b_idx = T−8 = 113 (T = 121); start-only: a_idx = 8, b_idx = T−1.
Verify S-PROBE's window against `misc/2026-09-08_collapse_probe/_probe_common.py` (9 px prefix consumed, 9 px suffix cut / 8 consumed) — if it differs, follow the probe's contract and record it.
S-SWEEP: a_idx = 24, b_idx = T−25 = 168 (T = 193) — verify T per file. Interior I = (a_idx, b_idx) open.

## 2. Spaces
1. **PIX** — 128×128 RGB, Gaussian blur σ=1.0, flattened, per frame. Byte-for-byte the Sep-08 instrument (`misc/2026-09-08_collapse_remeasure/instrument.py: load_matrix`). `cv2.setNumThreads(1)` always.
2. **DINO** — DINOv2-base CLS token per frame, 768-d, L2-normalised. Frame → RGB → resize to 224×224 (whole frame, no crop) → ImageNet mean/std. Every frame of the clip. Distances Euclidean on the normalised vectors.
3. **VAE** — LTX-2 video-VAE latents of the whole clip, via the repo's own encoder path (`eval_ladder/encode_conditioning.py: load_vae / preprocess / encode`, trainer package). Per latent timestep, flatten (C·H_lat·W_lat). Frame f → latent index ℓ(f) = 0 if f == 0 else (f−1)//8 + 1; so a_idx 8 → ℓ 1, b_idx 113 → ℓ 15, interior ℓ 2..14; T_lat = 16 for 121 f. S-SWEEP: downscale to 512×768 before encoding (÷32 legal), T_lat = 25, ℓ(24) = 3, ℓ(168) = 21.
4. **TRANS** — the frozen 44-channel DINO-basis signal (`misc/2026-08-24_flow_signal_conditioning/armA/armA_extract.py: compute_clip`, basis `$LAB/cache/armA_signals/pca.npz`; SIGNAL_REFERENCE.html is the definition). Field (T_lat, H_lat, W_lat, 44). NOT a point trajectory; reduce to per-timestep global summaries (raw channel units, no NORM file):
   - `nu_t` = mean ch41 (novelty: content matching neither endpoint)
   - `sA_t`, `sB_t` = mean ch39, ch40; `swap_t = sB_t − sA_t`
   - `trans_t` = mean(conf · sqrt(u²+v²)) over cells (ch33–35); `ent_t` = mean ch36
   - `inplace_t` = mean ch37; `dir_t` = mean ch38; `dlab_t` = mean ch42
   - `local_t` = fraction of cells with ch38 > q90, q90 = the 90th percentile of ch38 over all GT-twin cells (S-GRID) — spatial extent of change
   Descriptors: nu_max, nu_mean, swap_sharp = max_t|Δswap_t| / |swap_{T−1} − swap_0|, swap_pos (argmax Δswap, as a fraction of T_lat), trans_mean, trans_peak, inplace_peak_share = max_t inplace_t / Σ_t inplace_t, local_at_peak (local_t at argmax inplace_t), ent_mean. Time courses are kept (results/trans_profiles.csv) for the figure.

## 3. Common trajectory descriptor (spaces 1–3, per clip per space)
x_t, a = x[a_idx], b = x[b_idx], u = b − a, gap = ‖u‖ (+1e-8), interior I:
- `DR` = median_{t∈I} ‖(x_t − a) − τ_t u‖ / gap, τ_t = ⟨x_t − a, u⟩ / gap²  (the Sep-08 definition)
- `tau_prof` = mean τ in 10 equal bins over I; `M` = frac τ_t ∈ [0.25, 0.75]; `cross` = first t with τ ≥ 0.5, as a fraction of |I| (1.0 if never)
- `nu_t` = min(‖x_t − a‖, ‖x_t − b‖)/gap; `nu_max`, `nu_mean`; `explained` = frac(nu_t < 0.15)
- `step_share` = max adjacent step / path over [a_idx, b_idx]; `step_pos`; `path_over_gap` = path / gap
- `speed_prof` = 10-bin normalised step sizes (Σ = 1)
- `gap`, `gap_rel` (as before), `static` flag = PIX gap_rel < 0.12 (excluded from estimates, counted)
Embedding vector per space (18-d): [DR, M, cross, nu_max, nu_mean, explained, step_share, path_over_gap, tau_prof×10], standardised per space with ONE scaler fitted on the union of GT twins + landmarks of all strata.
Regression check: PIX DR / M / step_share for S-GRID clips must equal `misc/2026-09-08_collapse_remeasure/per_clip.csv` and `misc/2026-08-24_lerp_collapse/distance_vs_cut/cutstats.csv` for the same md5 to 1e-3.

## 4. Landmarks (per unique anchor pair; built in PIXEL space at native resolution from the clip's own frames a_idx and b_idx, then pushed through every extractor exactly like a generated clip)
- `LERP`: x_t = (1−α_t) a + α_t b, α linear 0→1 across I
- `CUT50`: a for the first half of I, b for the second
- `FREEZE`: a for all of I (jump to b at b_idx)
- `LATLERP` (VAE only): linear interpolation of the two anchor LATENTS ℓ(a_idx), ℓ(b_idx) — a landmark in the model's own space; PIX/DINO not defined for it
Prefix/suffix frames of the landmark clip = copies of a / b (only the interior is scored). Self-checks: LERP has PIX DR < 0.02 and DINO DR reported (a pixel blend is NOT on the DINO line — report what it is); CUT50 has cross ≈ 0.5 and step_share ≈ 1 in PIX.

## 5. Analysis (Phase C, CPU)
1. `results/per_clip.csv`: one row per clip × space with every descriptor + stratum/group/tier/endpoint/seed/paths/md5/static.
2. `results/TABLES.md`: per stratum × space × group (GT twin, NULLGEN [= base neutral generations; "NULL" is a pandas na_value, hence the rename], R1/R2/R3, A_empty…E, LERP, CUT50, FREEZE, LATLERP): n, median and bootstrap 95 % CI (2,000 resamples, clustered by endpoint) of DR, M, cross, nu_max, explained, step_share, path_over_gap; TRANS descriptors likewise.
3. Collapse-type measures per space, per null group, each with a bootstrap CI:
   - `landmark_nearest`: fraction of null clips whose nearest neighbour in the standardised 18-d space is a landmark rather than a reference trajectory (GT twin for S-GRID, R1 for S-PROBE), with the family breakdown (LERP/CUT50/FREEZE[/LATLERP]); the same number for the reference clips themselves (their own leave-one-out baseline).
   - `occupancy_entropy`: k-means k=6 fitted on reference + landmarks; assign null clips; normalised entropy of the assignment vs the reference's own.
   - `cross_space_agreement`: per clip, in how many of the 3 point-trajectory spaces it is landmark-nearest (0–3), plus the TRANS signature (nu_max below the GT-twin p5 AND swap_sharp above the GT-twin p95 → "swap-like"). Table of counts.
   - S-PROBE paired: R3 − R1 and R3 − R2 per (prompt, seed) on DR, nu_max, step_share, swap_sharp; Wilcoxon; sign counts.
4. Figures (`results/fig_*.png`, 150 dpi, matplotlib, colour-blind-safe): `fig_pca_<space>.png` 2-D PCA per space fitted on reference + landmarks, points by group, strata as panels; `fig_tau_<space>.png` median τ-profile per group with IQR band; `fig_trans.png` median time courses nu_t / swap_t / trans_t / inplace_t / local_t per group; `fig_agreement.png`; `strips/` 3 representative null clips per nearest-landmark family per stratum (12 frames each, cv2 single-threaded).
5. `results/REPORT.md`: numbers with metric + bar, readings attributed, no verdicts. The driver writes the reading.

## 6. Compute plan and rules
- Phase A (CPU, login node, `source $LAB/envs-aarch64/activate`, login python3 is 3.6 — never use it): manifest, PIX descriptors, landmark construction, smoke of DINO + TRANS on 2 clips (CPU), VAE smoke only if a CPU encode of ONE clip finishes in < 10 min, else leave VAE to Phase B.
- Phase B (GPU, ONE job, 1 × GH200, ≤ 2 h, DeltaAI ghx4): DINO + TRANS + VAE features for every manifest clip and landmark → `$LAB/cache/null_default/features/<clip_id>.npz` (dino fp16 (T,768); vae fp16 (T_lat, C, H, W); trans fp32 (T_lat,H,W,44)). The operator WRITES the sbatch and STOPS — the driver submits (account by FairShare, bgjg-dtai-gh or bhwp-dtai-gh only).
- Phase C (CPU): descriptors → tables → figures → report.
- Rules: no GPU job submitted by the operator; no commits (driver commits, `git add -f` small text/CSV only; features and strips are regenerable and stay uncommitted); no secrets; do not read `papers_drafts/ctt_iclr2027/AGENT_DONT_READ.md`; `cv2.setNumThreads(1)`; pandas `clip` column name collision → use `clip_id`; GT clips live in class subfolders (recursive glob).
