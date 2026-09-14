# REPORT — the base model's default transition, four spaces (Phase C / C4)

Operator, 2026-09-14 (`gh-login03`, DeltaAI, CPU login node). Facts only; every interpretation is
attributed as "the operator reads…" and describes patterns in the numbers, not campaign verdicts.
The driver writes the reading. Read with `SPEC.md`, `PHASE_A_REPORT.md`, and `TABLES.md`.

Numbers are reported with the metric and, where a bar exists, the bar. Each descriptor is a
family-agnostic trajectory summary (SPEC §3 / §2.4); it is NOT a transition-family label.

## 0. What ran, integrity, provenance

- Env `source $LAB/envs-aarch64/activate` (python 3.12.9) for every step. `cv2.setNumThreads(1)`.
  `NULL_FEATURES_DIR` unset → real features dir `$LAB/cache/null_default/features`. No Slurm job
  submitted; nothing committed.
- Pipeline: `scripts/integrity.py` (C4 step 0) → `scripts/descriptors.py` (C1) →
  `scripts/analyze.py` (C2). One new script (`integrity.py`) and one recompute (probe-by-tier)
  were added; **no SPEC descriptor definition was changed** (details in §7).
- **Integrity pass (`results/integrity.txt`): CLEAN.** 2513 expected npz = 2513 on disk = 2513
  checked; 0 bad, 0 absent, 0 orphan. Every main + LERP/CUT50/FREEZE carries dino/vae/trans with
  `dino.shape==(T,768)`, `trans.shape[0]==vae.shape[0]==(T−1)//8+1`, spatial (20,15) for 121-f and
  (16,24) for S-SWEEP, all arrays finite; every LATLERP is vae-only. No truncated decode
  (`dino.shape[0] < T`) anywhere.

## 1. Descriptor tables — row counts (C1)

`results/per_clip.csv` = **8729 rows** (1 row per clip × kind × space), `results/trans_profiles.csv`
= 35312 rows; descriptors.py reported **0 skipped, 0 errors**. Rows per space × kind (match the
manifest exactly):

| space | main | LERP | CUT50 | FREEZE | LATLERP | total |
|---|---|---|---|---|---|---|
| PIX  | 749 | 441 | 441 | 441 | —   | 2072 |
| DINO | 749 | 441 | 441 | 441 | —   | 2072 |
| VAE  | 749 | 441 | 441 | 441 | 441 | 2513 |
| TRANS| 749 | 441 | 441 | 441 | —   | 2072 |

749 main = 68 S-GRID NULLGEN + 19 GT + 38 S-GRID-F + 204 S-GRID-START + 360 S-PROBE (R1/R2/R3 =
120 each) + 60 S-SWEEP. 441 landmark-source clips (GT 19, S-GRID-F 38, S-GRID-START 204, R3 120,
S-SWEEP 60) each contribute LERP/CUT50/FREEZE (all four spaces bar VAE-only LATLERP). PIX rows are
reused verbatim from `pix_per_clip.csv` (machine-epsilon to the Sep-08 instrument, PHASE_A §4);
DINO/VAE/TRANS from the Phase-B npz. `local_t` q90 (ch37 over 19 GT-twin clips) = 0.7363.

## 2. Landmark self-checks (SPEC §5 step 3) — the instrument on known trajectories

n = 441 landmark clips per space. Medians (min/max in parentheses where a bar is one-sided):

| check | value | bar | pass |
|---|---|---|---|
| LERP · PIX · DR | med 0.00000 (max 2.2e-5) | < 0.02 | yes |
| LERP · DINO · DR | med 0.4535 (mean 0.500) | report only — NOT ≈0 | reported |
| LERP · VAE · DR | med 0.2786 (mean 0.288) | report only — NOT ≈0 | reported |
| LATLERP · VAE · DR | med 0.00012 (max 5.7e-4) | ≈ 0 | yes |
| CUT50 · PIX · cross | med 0.5035 | ≈ 0.5 | yes |
| CUT50 · PIX · step_share | 1.000 | ≈ 1 | yes |
| FREEZE · PIX · cross | 1.000 | = 1 | yes |
| FREEZE · PIX · step_share | 1.000 | — | — |

The operator reads: the pixel-space landmark checks all sit on their SPEC bars, so the PIX
instrument reproduces the Sep-08 definitions. A pixel LERP is deliberately NOT a straight line in
DINO (DR 0.45) or VAE (DR 0.28): the operator reads these as the geometry of a pixel blend re-read
through a non-linear encoder, i.e. a per-space *offset* the landmarks themselves carry, not a
property of any generated clip. LATLERP (a straight line built in latent space) does sit at VAE
DR≈0, so the VAE straight-line reference is the LATLERP landmark, not the pixel LERP.

## 3. Per stratum × space × group medians (SPEC §5.2)

Full table with 95% bootstrap CIs (2000 resamples, clustered by endpoint; S-PROBE by prompt_id)
for all 7 point metrics and 9 TRANS descriptors is `results/TABLES.md`. Condensed view of DR /
nu_max / step_share (median only) for each stratum's null group, its reference, and the three
pixel landmarks:

**PIX**

| stratum | group | n | DR | nu_max | step_share |
|---|---|---|---|---|---|
| S-GRID | NULLGEN | 68 | 0.211 | 0.782 | 0.081 |
| S-GRID | GT (ref) | 19 | 0.604 | 1.063 | 0.025 |
| S-GRID | LERP/CUT50/FREEZE | 19 | 0.000 / 0.000 / 0.000 | 0.495 / 0.000 / 0.000 | 0.010 / 1.000 / 1.000 |
| S-GRID-START | NULLGEN | 204 | 0.496 | 0.836 | 0.031 |
| S-PROBE | R3 | 120 | 0.218 | 0.718 | 0.072 |
| S-PROBE | R1 (ref) | 120 | 0.347 | 0.810 | 0.029 |
| S-PROBE | R2 | 120 | 0.320 | 0.809 | 0.034 |

**DINO** (S-GRID NULLGEN DR 0.245 / GT 0.511 / LERP 0.454; S-PROBE R3 0.202 / R1 0.392 / R2 0.358).
**VAE** (S-GRID NULLGEN DR 0.607 / GT 0.741 / LERP 0.279 / LATLERP 0.000; S-PROBE R3 0.612 / R1
0.650 / R2 0.662). See TABLES.md for CIs and all groups.

The operator reads: in every space the generated null groups (NULLGEN, R3) sit at a DR **between**
the landmark floor (LERP/CUT50/FREEZE/LATLERP ≈ 0, or ≈0.28 for the pixel-LERP-in-VAE offset) and
the real-trajectory ceiling (GT / R1). They do not sit on the landmark floor. Their nu_max is high
(0.72–0.91), i.e. at least one interior frame is far from both anchors relative to the gap — the
operator reads this as "the interior leaves the segment", the opposite of a freeze/cut/lerp whose
interior stays on or between the anchors (landmark nu_max ≤ 0.5). step_share for the nulls is small
(0.03–0.13) vs 1.0 for CUT50/FREEZE, i.e. no single dominating jump.

## 4. Collapse-type measures (SPEC §5.3) — `results/collapse_measures.csv`

### 4a. landmark_nearest (fraction of null clips whose NN in the standardised 18-d pool is a
landmark, not a reference), with family breakdown; and each reference's own leave-one-out (LOO)
baseline. Pool = GT+R1 references + all landmarks of all strata (see §7 note 3).

| space | stratum·group | n | frac landmark-nearest | LERP | CUT50 | FREEZE | LATLERP | REF |
|---|---|---|---|---|---|---|---|---|
| PIX | S-GRID·NULLGEN | 68 | 0.000 | 0 | 0 | 0 | 0 | 68 |
| PIX | S-GRID-F·NULLGEN | 38 | 0.000 | 0 | 0 | 0 | 0 | 38 |
| PIX | S-GRID-START·NULLGEN | 204 | 0.010 | 0 | 0 | 2 | 0 | 202 |
| PIX | S-PROBE·R2 | 120 | 0.000 | 0 | 0 | 0 | 0 | 120 |
| PIX | S-PROBE·R3 | 120 | 0.000 | 0 | 0 | 0 | 0 | 120 |
| DINO| S-GRID·NULLGEN | 68 | 0.044 | 3 | 0 | 0 | 0 | 65 |
| DINO| S-GRID-F·NULLGEN | 38 | 0.132 | 5 | 0 | 0 | 0 | 33 |
| DINO| S-GRID-START·NULLGEN | 204 | 0.108 | 22 | 0 | 0 | 0 | 182 |
| DINO| S-PROBE·R2 | 120 | 0.008 | 1 | 0 | 0 | 0 | 119 |
| DINO| S-PROBE·R3 | 120 | 0.042 | 5 | 0 | 0 | 0 | 115 |
| VAE | S-GRID·NULLGEN | 68 | 0.029 | 0 | 2 | 0 | 0 | 66 |
| VAE | S-GRID-F·NULLGEN | 38 | 0.000 | 0 | 0 | 0 | 0 | 38 |
| VAE | S-GRID-START·NULLGEN | 204 | 0.020 | 4 | 0 | 0 | 0 | 200 |
| VAE | S-PROBE·R2 | 120 | 0.000 | 0 | 0 | 0 | 0 | 120 |
| VAE | S-PROBE·R3 | 120 | 0.000 | 0 | 0 | 0 | 0 | 120 |

Reference LOO baseline (a reference's own nearest-landmark rate): PIX GT 0.000, R1 0.000; DINO GT
0.000, R1 0.017 (2/120); VAE GT 0.000, R1 0.000. (S-SWEEP null groups: landmark-nearest = 0.000 in
every point space — see §6.)

The operator reads: across PIX/DINO/VAE, the generated null groups are nearest a *reference*
trajectory, not a landmark, in ≥ 88% of clips (landmark-nearest ≤ 0.132, mostly ≤ 0.05). When a
null clip IS landmark-nearest it is almost always the **LERP** family (in DINO) or scattered
CUT50/FREEZE singletons (in PIX/VAE), never a consistent family. The reference LOO baselines are
~0, so the measure does discriminate (references are essentially never landmark-nearest). The
operator reads the combined picture as: by nearest-neighbour identity in the 18-d descriptor space,
the null default is not collapsing onto any single endpoint-explainable landmark.

### 4b. occupancy_entropy (k-means k=6 on the pool; normalised entropy of the null assignment vs
the reference's own). ref_own per space: PIX 0.659, DINO 0.541, VAE 0.230.

| space | S-GRID | S-GRID-F | S-GRID-START | S-PROBE R2 | S-PROBE R3 |
|---|---|---|---|---|---|
| PIX | 0.626 | 0.808 | 0.789 | 0.619 | 0.459 |
| DINO| 0.650 | 0.712 | 0.690 | 0.574 | 0.759 |
| VAE | 0.390 | 0.243 | 0.223 | 0.236 | 0.324 |

The operator reads: the null groups occupy the 6 pooled clusters with entropy of the same order as
the reference's own (0.23–0.81 vs ref 0.23–0.66) — not a single-cluster collapse (which would read
≈0). S-PROBE R3 in PIX (0.459) is the lowest non-sweep value, i.e. R3 spreads over fewer PIX
clusters than R1/R2, but still occupies several. (S-SWEEP entropies are degenerate — §6/§7.)

### 4c. cross-space agreement (per null clip, in how many of PIX/DINO/VAE it is landmark-nearest,
0–3; plus the TRANS "swap-like" flag: nu_max < GT-twin p5 AND swap_sharp > GT-twin p95).
Thresholds from GT twins: nu_max p5 = 0.174, swap_sharp p95 = 0.394. Full per-clip table
`results/cross_space_agreement.csv` (610 null clips).

| stratum | group | n | agree=0 | 1 | 2 | 3 | swap-like |
|---|---|---|---|---|---|---|---|
| S-GRID | NULLGEN | 68 | 63 | 5 | 0 | 0 | 15 |
| S-GRID-F | NULLGEN | 38 | 33 | 5 | 0 | 0 | 8 |
| S-GRID-START | NULLGEN | 204 | 178 | 24 | 2 | 0 | 23 |
| S-PROBE | R2 | 120 | 119 | 1 | 0 | 0 | 1 |
| S-PROBE | R3 | 120 | 115 | 5 | 0 | 0 | 20 |
| S-SWEEP | A_empty | 10 | 10 | 0 | 0 | 0 | 1 |
| S-SWEEP | A_word | 10 | 10 | 0 | 0 | 0 | 1 |
| S-SWEEP | B/C/D/E | 10 ea | 10 | 0 | 0 | 0 | 0 |

The operator reads: NO null clip is landmark-nearest in all 3 point spaces (agree=3 is empty); only
2 clips reach 2/3. So the small per-space landmark-nearest hits of §4a do not co-occur across
spaces — a clip that looks LERP-like in DINO is not the same clip that looks CUT-like in VAE. The
TRANS "swap-like" flag (low novelty + a sharp sB−sA swap, the signature of an A→B content swap)
fires on a minority that differs by stratum: S-GRID 15/68 (22%), S-GRID-START 23/204 (11%), S-PROBE
R3 20/120 (17%) vs R2 1/120. The operator reads the R3-vs-R2 gap here as the largest single
TRANS-signature difference in the probe, and notes that "swap-like" is a TRANS-only flag that does
not require point-space landmark-nearest (the two are reported separately, not intersected).

## 5. S-PROBE paired tests (SPEC §5.3) — R3 vs its witness R1 and its full-prompt twin R2

Paired per (prompt_id, seed); Wilcoxon two-sided; sign counts n_neg = R3 below comparator. SPEC
requires the `high` (30 scene-change prompts) and `inplace` (10 prompts) tiers **separate**; the
pooled table `results/probe_paired.csv` merges them, the split is `results/probe_paired_by_tier.csv`.

**Pooled (n=120 pairs):**

| space·metric | R3−R1 median | n_neg/n_pos | p | R3−R2 median | n_neg/n_pos | p |
|---|---|---|---|---|---|---|
| PIX DR | −0.062 | 100/20 | 1.0e-15 | −0.035 | 102/18 | 3.3e-15 |
| DINO DR | −0.078 | 98/22 | 1.1e-15 | −0.040 | 95/25 | 1.3e-14 |
| VAE DR | −0.015 | 73/47 | 6.4e-4 | −0.014 | 79/41 | 5.7e-7 |
| PIX nu_max | −0.064 | 81/39 | 2.4e-6 | −0.063 | 79/41 | 8.2e-7 |
| PIX step_share | +0.040 | 15/105 | 1.3e-17 | +0.030 | 18/102 | 2.7e-15 |
| TRANS swap_sharp | +0.205 | 18/102 | 3.6e-16 | +0.144 | 20/100 | 5.9e-15 |

**By tier** (`high` n=90, `inplace` n=30):

| tier | space·metric | R3−R1 med (p) | R3−R2 med (p) |
|---|---|---|---|
| high | PIX DR | −0.100 (9.6e-14) | −0.070 (5.6e-14) |
| high | DINO DR | −0.105 (8.8e-14) | −0.066 (2.6e-14) |
| high | VAE DR | −0.038 (2.6e-5) | −0.038 (2.7e-8) |
| high | PIX step_share | +0.053 (2.8e-15) | +0.045 (6.4e-15) |
| high | TRANS swap_sharp | +0.260 (4.0e-14) | +0.209 (7.1e-14) |
| inplace | PIX DR | −0.029 (4.0e-3) | −0.005 (4.1e-2) |
| inplace | DINO DR | −0.047 (5.4e-3) | −0.002 (3.3e-1) |
| inplace | VAE DR | +0.011 (2.8e-1) | +0.001 (6.0e-1) |
| inplace | PIX step_share | +0.013 (2.8e-3) | +0.002 (1.9e-1) |
| inplace | TRANS swap_sharp | +0.052 (2.8e-3) | +0.014 (2.5e-1) |

The operator reads: in the `high` (scene-change) tier every metric moves the same direction and is
strongly significant — R3 has lower DR (straighter), lower nu_max, higher step_share, and higher
swap_sharp than both R1 and R2, in every space. In the `inplace` tier the same signs appear only
weakly: R3−R1 stays significant for PIX/DINO DR and TRANS swap_sharp but small (|median| ≤ 0.05),
and R3−R2 is largely non-significant (VAE DR even flips sign, +0.001, p=0.60). The operator reads
this as: the R3-vs-witness effect the pooled table shows is carried almost entirely by the
scene-change prompts; on in-place prompts R3, R1 and R2 are close on these descriptors. This is the
tier separation the SPEC asked to preserve; do not read the pooled row as one homogeneous effect.

## 6. S-SWEEP side panel (SPEC §1) — never merged with S-GRID/S-PROBE

Different config (193 f, 1536×1024 native → 768×512 for features, CFG 3.2, 25 cond frames/end),
n=10 per text arm. A_empty ("") and A_word ("transition") are the null candidates; B–E are
text-informed. Medians (PIX unless noted):

| arm | PIX DR | PIX nu_max | PIX step_share | VAE DR | TRANS swap_sharp | TRANS nu_max |
|---|---|---|---|---|---|---|
| A_empty | 0.561 | 0.865 | 0.018 | 0.661 | 0.451 | 0.345 |
| A_word | 0.568 | 0.869 | 0.024 | 0.651 | 0.454 | 0.348 |
| B | 0.570 | 0.929 | 0.015 | 0.690 | 0.395 | 0.343 |
| C | 0.518 | 0.819 | 0.016 | 0.702 | 0.285 | 0.447 |
| D | 0.580 | 0.875 | 0.015 | 0.686 | 0.330 | 0.381 |
| E | 0.544 | 0.836 | 0.017 | 0.674 | 0.311 | 0.341 |

The operator reads: within the sweep, the two null arms (A_empty, A_word) are nearly identical to
each other on every descriptor and sit in the same DR/nu_max band as the text-informed B–E arms;
the largest arm-to-arm move is TRANS swap_sharp (A ≈ 0.45 vs C ≈ 0.29) and TRANS nu_max (C 0.447
highest). With n=10 per arm the CIs (TABLES.md) overlap heavily. In the point-space collapse
measures every sweep arm is landmark-nearest = 0.000 and VAE occupancy_entropy = 0.000 (all 10
clips fall in one k-means cluster); the operator reads these two zeros as an artefact of pooling a
different-config stratum into a pool the 121-f clips dominate (see §7 note 4), NOT as a within-arm
property — read the sweep only through its own DR/nu_max/swap_sharp medians above, not through the
pooled nearest-neighbour / occupancy numbers.

## 7. Could not verify / possible instrument artefacts

The operator flags the following explicitly rather than smoothing them over:

1. **`strips/` is nearly empty (2 PNGs), and this is faithful, not a decode failure.** analyze.py
   picks strip clips by PIX nearest-landmark; in PIX only 2 null clips (both S-GRID-START,
   FREEZE-nearest — matching §4a fam_FREEZE=2) qualify, so exactly 2 strips are written. The
   decode is serial cv2 (single-threaded) and both picks decoded (no swallowed failures). The
   operator reads this as a direct consequence of §4a (nulls are reference-nearest in PIX), and
   notes the strip visual would be richer if keyed off DINO (where LERP-nearest hits are more
   common); the SPEC definition (PIX = "representative space") was left intact.

2. **FREEZE landmark in TRANS behaves differently for two-sided vs start-only windows.** For
   two-sided windows (S-GRID/S-GRID-F/S-PROBE) FREEZE swap_sharp ≈ 1.0; for S-GRID-START (b_idx =
   T−1) FREEZE swap_sharp = 0.000 and inplace_peak_share = 0.000. The operator reads this as the
   FREEZE jump (a→b at the last pixel frame, latent 15) being aggregated away by the VAE/TRANS 8:1
   temporal compression when the jump sits at the final latent, rather than a code fault — the PIX
   FREEZE self-checks (cross=1, step_share=1) still pass everywhere. It means the TRANS FREEZE
   descriptor is not a stable landmark reference in the start-only stratum; do not read S-GRID-START
   TRANS against the FREEZE row.

3. **The nearest-neighbour / scaler pool is global (all strata), not within-stratum.** Per SPEC §3
   one scaler is fit on "GT twins + landmarks of all strata"; the implementation's pool =
   references (GT **and** R1) + every landmark of every stratum, and each null clip's nearest
   neighbour is searched in that same global pool. So an S-GRID null's NN could in principle be an
   S-SWEEP landmark, and "REF" counts any reference (GT or R1), not only the stratum's own. The
   operator reads the landmark-nearest numbers as global-pool identities; they are conservative
   (more landmark families available to match) and the ~0 reference LOO baselines show the pool is
   not degenerate, but this is a cross-stratum pool, not a per-stratum one.

4. **S-SWEEP is a minority, different-config stratum inside a 121-f-dominated pool.** The pool is
   dominated by the 121-f strata (esp. S-GRID-START's 204×3 landmarks); the 60 sweep clips (10
   mains/arm) land as a compact outlier group, which is why every sweep arm reads landmark-nearest
   = 0.000 and VAE occupancy_entropy = 0.000. The operator reads these pooled numbers as not
   informative for the sweep and defers to the sweep's own §6 medians (this is why the SPEC keeps
   S-SWEEP a side panel).

5. **LERP-in-DINO / LERP-in-VAE carry a built-in offset (DR 0.45 / 0.28), not a straight line.**
   Reported per SPEC §5 step 3; the operator reads it as a landmark property (a pixel blend is
   non-linear under these encoders), so in DINO/VAE the straight-line reference is LATLERP (VAE
   DR≈0), and the pixel LERP should not be read as the "no-motion" floor in those two spaces.

6. **VAE reference DR ceiling is high across the board (GT 0.74, R1 0.65, R2 0.66, R3 0.61) and
   `explained` = 0.000 for all main groups in VAE.** The operator reads the near-zero `explained`
   (fraction of interior frames within 0.15·gap of an anchor) as a property of the flattened-latent
   metric — VAE interior latents are rarely within 15% of an anchor latent for ANY group including
   real GT — so in VAE `explained` does not separate groups; use DR / nu_max / step_share there.

7. **Not verified at runtime by this operator:** the GPU extraction itself (Phase B, done by the
   prior operator; here validated only structurally by the integrity pass and by the landmark
   self-checks passing their bars). The VAE decoder-path deviation flagged in PHASE_A §8.3 is
   unchanged.

## 8. Files created / changed by this operator (C4)

Created: `scripts/integrity.py`; `results/integrity.txt`, `results/per_clip.csv`,
`results/trans_profiles.csv`, `results/trans_q90.json`, `results/TABLES.md`,
`results/collapse_measures.csv`, `results/cross_space_agreement.csv`, `results/probe_paired.csv`,
`results/probe_paired_by_tier.csv`, `results/fig_pca_{PIX,DINO,VAE}.png`,
`results/fig_tau_{PIX,DINO,VAE}.png`, `results/fig_trans.png`, `results/fig_agreement.png`,
`results/strips/` (2 PNGs). No SPEC descriptor definition changed; `probe_paired_by_tier.csv` is an
added tier split of the existing pooled test. Nothing committed; no Slurm job submitted.
