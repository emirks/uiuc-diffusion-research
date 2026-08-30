# S6 reshape (r832) — build authority

**Status 2026-08-30.** Goal: re-encode EffectData stratum **S6** at **two orientation grids** so it is
(a) grid-consistent for the DINO signal contract, (b) token-matched with the corpus (4,576 vs 4,800),
(c) pairing-complete (only 92 same-grid singletons drop, vs 2,378 in 004), (d) still a faithful
"same effect, different subject" one-sided demonstration stratum. **Only S6 changes**; S0/S1/S2a/S2b/S4
are byte-identical to 004. Numbers here are measured. Full trajectory: `misc/2026-08-30_s6_reshape/DOSSIER.md`.

---

## 1 · Reshape spec (advisor-locked, DOSSIER Round 1)

| aspect | value |
|---|---|
| Landscape-native S6 (1248×704, 1056×704; 14,523 clips) | target **832×512 px (W×H) × 81 f** → latent **(11,16,26)** `effd_832x512_81f` |
| Portrait-native S6 (704×1248, 704×1056; 14,121 clips) | target **512×832 px × 81 f** → latent **(11,26,16)** `effd_512x832_81f` |
| Frames / fps | 81 f native (all frames), 24 fps, no resample; F_latent = 11 |
| Resize policy | process_videos default: aspect-preserving **scale-to-COVER + center-crop**; NO letterbox/pad, NO rotation |
| Crop offsets (top,left) | 1248×704 → (0,38) · 1056×704 → (21,0) · 704×1248 → (38,0) · 704×1056 → (0,21) — VAE ≈ DINO within <1 px |
| Crop loss | 16:9-native crops the long side 8.27 % (**4.13 %/edge**); 3:2-native crops the short side 7.59 % (**3.79 %/edge**); every clip ≤ 4.13 %/edge |
| Tokens / shift | **4,576 tokens** (95.3 % of corpus 4,800); **shift 2.221875** — DERIVED by root_common (RULED_SHAPES), never restated |
| Pairing | `ring_offset_within_op_shape__k=min(3,n-1)__s6_drop_shape_singletons` with shape = orientation grid; gate: same grid AND different subject |
| Captions / conditions | reused unchanged from 004 (resolution-independent; ≤ 4.13 %/edge crop leaves the A-description valid) |
| Sided | one (frame-0 anchor, prefix_latents=1); mask `f11_h16_w26_p1_onesided` / `f11_h26_w16_p1_onesided` (frame-0 plane sum 416) |

**A1–A3 (Round 1) code changes:** A1 — 2 RULED_SHAPES entries added to `root_common.py`
((11,16,26)→`effd_832x512_81f`, (11,26,16)→`effd_512x832_81f`; git diff 11 insertions, 0 deletions).
A2 — derived roster (below). A3 — the r-driver `armA_extract_s6r.py` + the encode driver reuse the
frozen compute; **`dino_raw` written to /work/nvme scratch only, never $LAB**, deleted after PCA health.

## 2 · Derived roster

`outputs/ctt_v2/encodes/EFFECTDATA_r832/ROSTER.json` (sha256 `c66c6477…`) is derived from the frozen
native roster `outputs/ctt_v2/encodes/EFFECTDATA/ROSTER.json` (sha256 `da2a0842…`): per clip, keep
`native_wh`; set `target_wh`=(832,512) & `latent_fhw`=(11,16,26) if native w>h, else (512,832)/(11,26,16).
28,644 clips — **14,523 landscape (11,16,26) / 14,121 portrait (11,26,16)**.

The stratum inventory `outputs/ctt_v2/inventories/S6_r832.json` is derived from `S6.json` by
`scripts/ctt_v2/s6/derive_inventory_r832.py` — re-points every clip's latents/cond_clean at
`EFFECTDATA_r832/`, keeps group/conditions/caption/endpoints/caption_sources byte-identical
(28,644 clips / 2,917 groups / 57,288 paths, all exist).

## 3 · Encode (VAE) — DeltaAI GH200 (aarch64)

Array **3049217** (`bhwp-dtai-gh`), `--array=0-47` (4 shapes-superseded → 2 grids × shards), one
`process_videos.py` call per (grid,shard) at exact W×H×81. **48/48 COMPLETED, ExitCode 0:0; Σ 7.84 GPU-h**;
MaxRSS 36.4–48.6 GB. Outputs `EFFECTDATA_r832/{latents,cond_clean}/<stem>.pt` (128,11,H,W); cond_clean ==
latents bitwise for the frame-0 anchor. Census verified 28,644/28,644 (14,523 L / 14,121 P; 0 tmp;
latents+cond_clean ≈ 67.2 GB on disk). Decode PSNR 30.2–36.3 dB (pilot 3049192).

## 4 · DINO signal (44-ch operator field) — store 003

Extract array **3049605** (`bgjg-dtai-gh`), `--array=0-95`. **96/96 COMPLETED, ExitCode 0:0; Σ 7.69 GPU-h**;
MaxRSS ≤ 8.83 GB. Frozen `armA_extract` compute + frozen `pca.npz` basis; DINO frames 448×728 (L) /
728×448 (P), 28 px/cell; patch grid pooled 2×2 → latent grid. Signal root
`$LAB/cache/armA_signals_005/feat` = 19,023 inode-verified non-S6 003 hardlinks + **28,644 new S6 r832
fields** + pca hardlink (**47,667** total). `verify` ALL CHECKS PASS (28,644 feat set-equal, census
14,523/14,121, shape/chan/finite 0, 0 tmp). Cross-job determinism 24/24 bitwise.

**PCA health (pre-registered bars):** baseline evr.sum() 0.4521; **PRIMARY** pooled **0.3744** (≥ 0.339),
4 native 0.3703–0.3781 (≥ 0.316); **CROSS-CHECK** META vs V4-raw max|Δ| 3.7e-15 (≤ 1e-6); **SECONDARY**
paired r832/native ratio pooled **0.9694** (≥ 0.90), per-native 0.9645–0.9732 (≥ 0.88). conf %exact-zero
43.47 % (native 48.2 %); u/v saturation 0.026 %. ALL PASS.

**Norm:** `NORM_dino_v4.json` by **addendum** to store 003 (sha256 `db47be88…`; version tag
`dino_signal_norm_v4`). Six strata; n_files 47,444; **S6 131,074,944 cells** (28,644 × 4,576); non-S6
per-stratum moments **bit-identical to v3** (rel Δ 0). G-N1..G-N5 PASS. Committed 1afc7c8 (Round 3).

## 5 · Pairing → root

`assemble_root.py --contract 005_ctt_v2plus_s6reshape --sampler-mix --code-side` pairs S6 within
(effect × orientation grid): **81,779 same-grid pairs** over 28,552 targets; **92** same-grid
same-effect singletons DROPPED (untrained → unseen-subject eval material, MUST NOT be called trained).
Collapsing the 4-grid native zoo to 2 grids **returns 2,286 targets** that 004 dropped
(28,552 − (26,266 − 2,286) = 28,552; net +2,286 vs 004's target set). Root total **138,147**
(S0 385 / S1 3,675 / S2a 22,731 / S2b 23,577 / S4 6,000 / S6 81,779), VERSION
`3.1.0-ctt_v2plus_s6reshape-codeside`, 5 masks. Verification: `verify_code_side_005.py` invariants 1-5 PASS.

## Costs

| stage | job | account | GPU-h |
|---|---|---|---|
| encode (VAE) | 3049217 | bhwp-dtai-gh | 7.84 |
| DINO extract | 3049605 | bgjg-dtai-gh | 7.69 |
| verify + health + fit (R3) | — | bgjg-dtai-gh | ~0.8 |
| Round 4 (root + verify + configs + store) | — | none (CPU on gh-login) | 0 |

Campaign ≈ **16.4 GPU-h** total (bhwp ≈ 7.9, bgjg ≈ 8.5). No full `dino_raw` persisted (1 % raw
health sample only). Native `EFFECTDATA/`, store 004, norms v1–v3, trainer fork, eps all untouched.
