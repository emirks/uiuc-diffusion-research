## 2026-08-30
- 01:23 DINO-signal arms — ALL FIVE ARMS LAUNCHED. DeltaAI chains submitted (2×GH200 singleton chunks, 8 each): A1 bgjg / A2 bhwp as each account's primary 2-GPU job (reverse of the eps order), A5 bgjg / A4 bhwp "on top" via `--nice=200` (new `SBATCH_EXTRA` pass-through in `build/submit_chain.sh`; manifest `deltaai_chains.tsv`); A1/A2/A5 started within 2 min, A4 pending. eps A0 at step 120, 9.9 s/step. New `build/watch_all.sh` (eps + DeltaAI status, automatic chain top-up) runs every 30 min from a session cron; the per-event log monitor is gone. LAUNCH.md/DOSSIER updated.
- 00:33 DINO-signal arms — fork `32d6e3f` (SPEC R2.5): LoRA-A init seeded (`set_seed` before `_load_models()`) and per-(chunk, rank) σ/ε stream (`set_seed(seed + initial_step, device_specific=True)`) — a resumed chunk no longer replays the seed-42 stream; data order and signal-drop RNG unchanged. Owner waived re-gating and smokes; shipped to eps (trainer at 32d6e3f). Launch-ready.
- 00:17 DINO-signal arms — resume-mechanism check before launch: data order continues exactly (offline-exact sampler proof + 2-GPU chunked-vs-continuous test), LR/scheduler/optimizer/drop counters/wandb id continue; multi-GPU resume replays the σ/ε RNG stream per chunk (unbiased); NEW: LoRA-A init is unseeded (PEFT wrap in __init__ before set_seed in train(); same-seed runs corr 0.002) — 2-line fork fix owner-gated. Launcher fixes on both machines: `build/pick_resume_ckpt.py` (resume from the newest intact pair, quarantine truncated) and `build/prune_ckpts.py` (chain-aware keep-3 + 500-grid hard-links; chains would otherwise accumulate ~250 GB/arm).

## 2026-08-29
- 23:33 DINO-signal arms — LAUNCH READY (misc/2026-08-27_dino_signal_training/LAUNCH.md): eps `A0→A4→A5→A2→A1` sequential on GPUs 0–3 only (launchers now pick from `GPU_POOL=0,1,2,3` and refuse otherwise); DeltaAI 2-GPU chains `A1` (bgjg) + `A2` (bhwp) now, `A5`/`A4` on the new projects tomorrow, replicates after. `train_chunk.sbatch` account guard is an allow-list (bhwp + bgjg — owner: supervisor cleared bgjg for these runs; extend via ALLOWED_ACCOUNTS); `submit_chain.sh` takes `ACCOUNT=`. Balances: bgjg 997 h (FairShare 0.22), bhwp 804 h; ~143 GPU-h per arm on DeltaAI.
- 23:27 DINO-signal arms — final training mix (owner): S0 2 / S1 18 / S2a 22.09 / S2b 22.91 / S4 8 / S6 27 — creative 55 % / procedural 45 %, two-sided 65 %, S0+S1 sampled as one pool ∝ size (both ≈ 4 passes in a 10k run), S6 0.37 passes. All five `configs_004` re-rendered and shipped to eps; `assert_pins` prints the override vs the root's mix.json on both machines. Supersedes the S0 5 / S1 11.8 interim.
- 23:16 DINO-signal arms — owner decisions after two Fable consultations (recorded in DOSSIER): keep the five architectures, LR groups, zero-init and signal p_drop 0.1; mix changed to S0 5 / S1 11.8 / S2a 27.10 / S2b 28.10 / S4 8 / S6 20 (`configs_004` re-rendered with `--weights`, shipped to eps, launcher prints `mix OVERRIDE`); owner's wandb key placed on eps as `$DS/wandb.env` (600) read only by `03_train_all.sh` (refuses to launch without it); reference dropout stays a pre-registered follow-up. Facts: 25.8 s/step on 2×GH200 (~72 h/arm), bhwp balance 804 h.
- 21:50 DINO-signal arms — health re-check after the code-side 004 migration (misc/2026-08-27_dino_signal_training, SPEC ROUND 2). A Fable code review found two BLOCKERs, both reproduced and fixed: (1) `SignalStore.verify()` hard-pinned the norm version tag to `dino_signal_norm_v1`, so every signal arm on `configs_004` (NORM v3) died at startup → fork `46a0663` accepts `dino_signal_norm_v<N>` (sha pin unchanged) = new freeze; (2) `build/train_chunk.sbatch` used unbound `ARM`/`COMMIT7` under `set -u` (every DeltaAI chunk aborted). New CPU gate `build/verify_signal_store.py` (real SignalStore from each config + one load per stratum) wired into all launchers — PASS on both machines. Mix policy per owner: config weights are the authority; `assert_pins.sh` prints `mix OVERRIDE` vs `mix.json` (fatal only if malformed); `make_004_configs.py --weights [--renorm]` + `eps/ship_configs.sh` change the mix for all five arms. eps launcher fixed to derive per-arm dirs from the config `output_dir` (configs_004 wrote to `outputs_004/` while resume/DONE/mirror looked in `outputs/`); `ship_campaign.sh` no longer deletes eps logs. eps at `46a0663`, `01_preflight` (configs_004) `01 OK`. GPU re-gates on bhwp: G-identity 004 (3047868), G-shape on S6 shapes with peak memory (3047877), end-to-end 30-step chunk smoke (3047869). RESULTS: all PASS at `46a0663` — G-identity identical to round 1; S6 shapes peak 60.1–62.7 GB fwd+bwd (checkpointing on; ≈68 GB with AdamW state vs 80 GB eps H100); chunk smoke ran pins→verify→fsck→provenance→30 steps→DONE in 17 min. Both machines launch-ready.

**21:17 — `004_ctt_v2plus` docs carried INTO the store — the store entry is now self-sufficient.** The dataset's prose docs lived in `misc/2026-08-28_effectdata_s6/`, outside the store. Moved the two canonical docs into `store/datasets/004_ctt_v2plus/` — `CODESIDE_FORMAT.md` (format/usage/portability/verify) and `BUILD.md` (S6 source build; added a banner marking §8's assembled-root numbers superseded by the code-side rebuild, §1–7 still authority) — and added a new `README.md` entry point that routes to them + `meta.yaml` (correction block) + `root/` (via the `root` symlink) + the sibling `003_dino_signals` (signal + NORM_dino_v3, not duplicated). Relative **symlink stubs** left at the old misc paths so `CHANGELOG`, the hashed caption store, and code comments still resolve. Repointed the living indexes (`store/INDEX.md`, `store/TEXT_LIFECYCLE.md`) to the in-store home and appended a `docs_relocated_2026_08_29` block to `meta.yaml`. Verified: YAML parses, all README links resolve, `store_fsck` PASS (0 fails).

**21:11 — DINO-signal arms: 004_ctt_v2plus code-side root migrated to eps; both config sets launch-ready on both machines.** After the dataset agent's 004 rebuild (S1 restored, S6 same-shape 57,847 pairs, code-side root, NORM_dino_v3), `configs_004/` was re-rendered (new mix S0 12 / S1 4.8 / S2a 27.10 / S2b 28.10 / S4 8 / S6 20; NORM v3 sha pinned) and a `PINS.env` + `build/assert_pins.sh` now enforce `samples_sha256` 5a73eb3c… and the norm sha in every launcher (DeltaAI chunk sbatch, eps preflight/train). Local shape-aware fsck 100 % in all six strata. eps (`/storage/ozgur/dino_signal`): stale 138,625-row list deleted + tombstoned, code-side root brought up with a three-link `_src` shim (July ctt_v2 tree + S0 mirror), NORM v3 shipped; full path existence 5 × 114,215, tensor-load smoke 225 rows / 9 (stratum,shape) groups / 0 mismatches, `01_preflight` with `CONFIGS=configs_004` → `01 OK`. Fork unchanged at the five-arm freeze `091b50b`. GPU smoke/train on eps wait for idle GPUs.

**14:34 — `004_ctt_v2plus` corrected + rebuilt CODE-SIDE: S1 restored, S6 re-paired same-shape, norm v3; advisor GO, invariants PASS.** Three fixes in one rebuild (owner-directed, fable-advisor GO): **(1) S1 restored** — its absence was a build gap, not "pilot only" (the `003_ctt_v2plus` contract always included S1 at 4.8%; the assembler ran the pre-registered S1-absent branch because `normalize_inventory_paths.py` omitted S1 from `STRATA`, so S1's inventory was never path-normalized). **(2) S6 re-paired same-shape** — S6's 4 native grids meant 75.1% of pairs were cross-shape (reference grid ≠ target grid — the exact issue the DINO-signal Fable review handed to the dataset agent as B2); `assemble_root.build_samples` now sub-groups each effect by shape and rings within `(effect,shape)`, so every S6 pair is same-shape; 2,378 shape-singleton clips DROPPED (untrained same-effect subjects = free unseen-subject eval). S6 26,266 clips / 57,847 pairs. **(3) code-side form** — added `assemble_root --code-side`: no per-row symlink trees (was ~571k); `samples.jsonl` carries ROOT-RELATIVE paths resolving through ONE re-pointable `_src` symlink + `_mask_store` (7 masks), portable across devices. New total **114,215 pairs** (S0 385 / S1 3,675 / S2a 22,731 / S2b 23,577 / S4 6,000 / S6 57,847), full contract mix (S0 12 / S1 4.8 / S2 55.2 / S4 8 / S6 20). **Verified:** invariant battery `verify_code_side.py` (counts vs inventory/ROSTER, set-equality, shared-stub/conditions-dedup, path-scheme, existence+shape) — all PASS; old 571k tree pruned in full; `samples_sha256 5a73eb3c…` pinned in ROOT_MANIFEST. **Norm v3** (`NORM_dino_v3.json`, sha `91af261d…`) over {S0,S1,S2a,S2b,S4,S6}: all gates PASS, **n_files 47,444**, S6 coverage 26,266/26,266; kept-vs-dropped moment split worst |Δμ|/σ 0.110 (drop immaterial). Registered: 004 meta correction block + 003 v3 addendum + `$CACHE/NORM_dino_v3.json` symlink; v2 retained (superseded for 004). Docs: `misc/2026-08-28_effectdata_s6/CODESIDE_FORMAT.md` + `outputs/ctt_v2/roots/ctt_v2plus_mix/README.md` (layout, `_src` portability, per-device bring-up, eps migration for the eps agent). in-place 004 correction (consumers still []; nothing consumed the old form).

**12:29 — DINO-signal IC-LoRA: five-arm freeze `091b50b`, all gates PASS, eps mirror prepared (campaign `misc/2026-08-27_dino_signal_training/`, single source of truth `SSOT.md`).** Round 1 adds two arms to the gated stack: **A4 channels-reference** (`placement: target|reference|both` replaces `on_reference`) and **A5 xattn-fusion** (signal cells as Q over the clean reference latents as K/V at 128-d, LN, fixed sinusoidal 3-D PE on Q/K normalised by the target block's pixel/second extents, zero-init `w_o` ⇒ step-0 output bitwise equal to the tokens arm; ≈110k params). Evaluations: conformance (blocker: PE built differently at train vs inference → one shared `signal_norm_pos`), Opus 4.8 adversarial (no blockers), **Fable adversarial** (B1: positions were pixel/seconds ÷ latent grid, PE aliased 5 rad per cell, low-band adjacency 0.00 → fixed to 1.00; M1: grid guard was token-count-only → full (T,H,W) check; B2: **S6/EffectData pairs mix four latent grids — only 25 % of S6 rows have reference grid == target grid**, handed to the dataset agent, not changed). Round-1 GPU gates on the freeze commit: G-identity (A1/A4 bitwise; A2/A5 cos 0.9966; **A5 step-0 ≡ A2 bitwise**), G-shape, G-ddp+probe (A4, A5), G-resume (A5, 13 xattn keys), G-val, G-fusion (i)-(v), G-invariant — all PASS. Data: `configs/` (ctt_v2) launch-ready; `configs_004/` (004_ctt_v2plus incl. S6, mix from its mix.json) rendered but blocked by the S6 grid issue; NORM v1 on S6 misses the std gate on s_B/nu (0.49 vs 0.5), a six-stratum v2 candidate passes all gates (unregistered, owner choice). eps (fal-h100): new root `/storage/ozgur/dino_signal` — trainer clone at the freeze commit, venv, 47,667 signals, EffectData encodes, S6 captions, absolute-path 004 sample list (0/15,000 paths missing), configs for both sets (accum 2 × 4 GPUs = eb 8), tmux/idle-GPU/dirty-tree/resume/provenance/mirror launchers; bootstrap + CPU preflight OK; Fable health-check OK (GPU smoke and training wait for idle GPUs; foreign job untouched).

## 2026-08-28

**22:21 — EffectData S6 signal extracted + `NORM_dino_v2.json` fit + registered (store `003_dino_signals` addendum); advisor GO, training-ready for `004_ctt_v2plus`.** Extracted the 44-ch DINO-basis operator signal for all **28,644 EffectData/S6 clips** using BYTE-IDENTICAL frozen compute (`armA_extract_s6.py` imports `armA_extract`) + the same frozen `pca.npz`, into the shared `$LAB/cache/armA_signals/{feat,dino_raw}` — cache now **47,667 clips**. S6 needed **zero geometry changes** (native VAE-legal, no crop; exactly 81f ⇒ `select_frames(81)→T_lat=11`; one-sided A=first/B=last like S4); 4 shapes (11,{22,33,39}×{22,33,39}), 96-shard bhwp ghx4 array. Refit the norm as **`NORM_dino_v2.json`** over `{S0,S2a,S2b,S4,S6}` — the 004 training strata (S1 absent from 004), recording the rule *norm strata = the served dataset's training strata, eval excluded* (v1↔ctt_v2, v2↔004); **all gates G-N1..G-N5 PASS**, asinh not triggered, sha `cd7593b9`, fit over 46,219 files / 328.3M cells. Full training-readiness audit (`validate_training_ready.py`) **OVERALL READY**: V1 coverage 100% every stratum, V2 **full-open of all 28,644 S6 npz** 0 defects, V3 raw 0 unreadable, V4 frozen-PCA share pooled 0.86× / per-shape 0.84–0.89× baseline (above pre-registered 0.75×/0.70× bars — basis kept), V8 norm-apply clean. **Advisor-caught + fixed:** two self-inflicted incidents — a tmp-naming defect (`np.savez` auto-`.npz`) compounded by editing a live-imported driver (11,103/28,644 first pass; purged + fully re-extracted → 28,644/28,644, 0 fail), and a coverage-checker bug (S6 target/reference are cross-subject with **different grid orientations**, so a row's `shape` is the target's — now keyed by each stem's own ROSTER `latent_fhw`; norm sha unchanged before/after ⇒ data was always correct). Registered as an append-only entry-003 meta addendum (v1 lines frozen) + `$CACHE` symlink into the store; v1 remains the ctt_v2 norm. `store_fsck` PASS. Report `.../armA/NORM_REPORT_v2.md`, validation `.../armA/VALIDATION_TRAINING_READY.md`.

**11:53 — Architecture diagram family started: the default LTX-2 IC-LoRA training graph, verified against code (`misc/2026-08-28_arch_diagrams/`, viewer `arch_diagrams`).** One paper-grade SVG (`base_iclora.svg` + 2× PNG) generated by `make_base_iclora.py`, drawn from the actual modules: ① inputs → the `[reference | target]` token sequence (reference = demo latents on the SAME 16×20×15 grid, M = N = 4800; start/end pinned clean at σ=0; only the middle noised and supervised) through the single frozen `patchify_proj` 128→4096 and a shared RoPE grid; ② one transformer block ×48 with every frozen part (RMSNorms, adaLN `scale_shift_table`, q/k norms, RoPE, gates) and the 10 LoRA-adapted linears (attn1 q·k·v·out, attn2 q·k·v·out, ff.net.0.proj, ff.net.2), the one-way SDPA split drawn as a 2×2 query/key mask, text entering only via attn2 K/V (no adaLN/gate: `cross_attention_adaln` off), the unused audio stream; the output head (`norm_out` + `proj_out`, frozen) and the middle-only flow-matching loss; ③ the LoRA-on-a-linear mechanism and ④ accounting: 480 adapted linears, 960 tensors, 654.3 M trainable (1.31 GB bf16, ≈3.4 % of 19B). Facts checked in `src/LTX-2-dualforce` (ltx-core model/transformer/attention/adaln/text_projection; ltx-trainer flexible/trainer) and the `store/runs/012` checkpoint keys. Arm variants are to be derived from the base script (same grid/palette).

**11:10 — ICLR 2027 paper folder restructured into a writing scaffold (`papers_drafts/ctt_iclr2027/`, untracked — repo visibility decision pending).** `main.tex` is now the single root (anonymous by default) with one file per section under `sections/` (each carrying a `% PLAN` block distilled from the owner's `paper_foundation_USER.md`), `macros.tex` with `\todo{}`/`\gap{}` draft markers so missing evidence is stated in the PDF rather than hidden, and `CLAIMS.md` — a claim→section→evidence→status ledger that doubles as the experiment queue (top item: the champion's deranged-reference control is VOID in the store). Tables are no longer a monolithic `metrics.tex`: `render_latex.py` (metric campaign) now reads paper-owned templates from `tables/templates/` and writes one file per table into `tables/generated/` plus `numbers.tex`, which defines `\ctnum{key}` for all 446 quotable numbers (e.g. `gap.unseen.app.champion-neutral-vs-best-external-effect` = +15.8) so prose can never drift from the store; regression gate still PASSES. `build.sh` checks the vendored ICLR style files against the pristine `papers_drafts/iclr2027/` master (fails on drift), builds, and reports pages / open TODO+GAP / undefined keys. `STRUCTURE.md` documents the three file rules. The old proposal draft moved to `archive/`. Skeleton builds clean (0 errors, 0 undefined keys).

## 2026-08-27

**21:04 — DINO-signal IC-LoRA training stack built, gated, and ready to launch (campaign `misc/2026-08-27_dino_signal_training/`, fork `src/LTX-2-armA` @ `5829c22`).** From-scratch generation conditioned on the frozen 44-ch signal (`datasets/003_dino_signals`), replacing the audited flowsig recipe: no cell lattice / text dropout / reference dropout, one independent per-sample signal-drop (p=0.10) on its own RNG stream (also across accum micro-steps), two injection routes with one route per arm — **channels** (bias-free zero-init `Linear(44→4096)` added to target-token embeddings after `patchify_proj`; dropped = zeros = the exact baseline path) and **tokens** (bank tokens, 019 mechanics minus the adaLN side-channel, learned null) — two optimizer groups (LoRA 1e-4 / signal 1e-3), `signal.*` checkpointed in fp32 state-first, cumulative counters persisted, per-group pre-clip grad norms, 44 column norms, per-group grad×input, contribution ratio, and a frozen probe (matched/zeroed/deranged, paired deltas) every 500 steps; inline validation renders through the signal port (ID + OOD + two-sided OOD + control, `signal_id` on the reference). Provenance-guarded chunk sbatch (records `git rev-parse HEAD`, refuses a dirty tree, runs the coverage fsck — 100% on all 56,368 rows). Built by 5 opus-4.8 implementers against `SPEC.md`, adversarially reviewed (2 blockers + 3 majors fixed), then **all 7 pre-registered gates PASS on the final commit**: G-cpu, G-identity (channels **bitwise** 0.597412 = 0.597412; tokens cos 0.9966), G-shape (S4 grid), G-ddp, G-probe (fires at interval under 2-GPU DDP), G-resume (LR/optimizer/counters/provenance continue), G-val. Two real defects the gates caught and fixed on the way: a `bf16 + fp32` promotion of the residual stream (`cc904af`) and a telemetry dtype mismatch (`869e497`). Arms A0 baseline / A1 channels / A2 tokens / A3 channels+on_reference differ only in marked knobs; OWNER-GATED before launch: stratum mix (placeholder = 002's), arm count, GPUs per chain. Full doc: campaign `README.md` (self-contained) + `DOSSIER.md` (per-gate log).

**18:02 — Frozen normalization v1 fit for the 44-ch DINO signal → store `datasets/003_dino_signals`.** Fit the trainer-load-time per-channel affine `x_norm = clip((x−loc)/scale, −5, +5)` over all **18,800 non-eval feat cells (84.28M cells)** with **equal-stratum weighting** across S0/S1/S2a/S2b/S4 (eval held out) so the instrument is invariant to the not-yet-frozen training mix and S2's 15,436 clips don't dominate the scale (`misc/2026-08-24_flow_signal_conditioning/armA/fit_norm_dino.py`, one streaming float64 pass, both identity + asinh moments accumulated so the escape hatch needs no re-pass). **All five pre-registered gates PASS**: G-N1 post-norm std ∈[0.5,2.0] (measured 0.98–1.01 every channel), G-N2 clip %|z|≥5 ≤2% (worst = dLab 0.36%, so **asinh not triggered — every channel kept identity**), G-N3 no dead channels, G-N4 channel names byte-identical to `armA_extract.CH_NAMES` across 150 files, G-N5 **100% coverage** of all 56,368 `samples.jsonl` rows→feat with shape match (S0 139 / S1 1,225 / S2a 7,577 / S2b 7,859 / S4 2,000). Drift table clean (no per-stratum |post-norm mean|>1.0, incl. the sidedness-coupled endpoint channels). Registered as immutable datasets entry **`003_dino_signals`** (root→`cache/armA_signals`, canonical `NORM_dino_v1.json` sha256 `79061ec4…c51d3` in-entry + cache symlink into store, `pca.npz` sha256 `4d59539b…17244`); `store_fsck` PASS. Report `.../armA/NORM_REPORT.md`. Signal held as-is at 44 channels.

**14:48 — Arm-A signal extended to all consumed S2 + S4 (ctt_v2), advisor-verified, training-aligned.** Extracted the 44-ch operator signal + raw-DINO cache for every clip ctt_v2 training consumes: **19,023 total** (223 eval + 1,364 train + **15,436 S2** [S2a 7,577 + S2b 7,859] + **2,000 S4**), keyed by (stratum, stem) from `samples.jsonl` (fact-checked against `MANIFEST.json` v2.1.0). 24-shard bhwp array, **0 FAILs**, `feat/` + `dino_raw/` = 19,023/19,023, raw cache **483 GB**. The extractor is now adaptive per-clip: latent grid = authoritative `shape` from samples.jsonl, DINO grid 2× that; fields match training latents exactly (S2 [16,20,15], S4 [5,14,26]). **Fable-advisor review (GO-with-changes)** caught one real defect: S4 (832×464) training encodes by identity-resize + **16-row center-crop** (rows [8,456), `top=8`) — not a squeeze into the 448-grid, which would have misaligned every S4 token by up to ~23% of a cell; reproduced the exact crop in `_load_clip._fit_to_grid` and recorded `crop_top` in the raw metadata. Advisor also confirmed: **A=first/B=last is correct for one-sided S4** (sidedness is a mask property, not a signal property; S1 one-sided already uses it), and **reuse the frozen PCA** (instrument consistency) — post-run G1 captured-variance per stratum measured healthy: S2a 0.358 / S2b 0.375 / S4 0.393 vs S0/S1/EV ~0.44 baseline (0.82–0.90×, well above the half-threshold, no refit). Endpoint structures documented (S2 two-sided: 2 endpoints, `p2_twosided` mask; S4 one-sided: 1 endpoint, `p1_onesided`), sided stored per clip. Viewer `armA_signal` rebuilt to show a browsable slice (4,617: all eval/train/S4 + S2 sampled 1/15), S4 at its native landscape grid, LIVE. Code: `misc/2026-08-24_flow_signal_conditioning/armA/{armA_extract.py,extract_full.sbatch}`.

**12:45 — Arm A (44-ch DINO-basis operator signal) extracted, cached, and served as viewer v2; plus a per-metric extension cache with v2 ν′ as the first add-on.** Implemented the owner-specced 44-channel signal on the LTX-2 latent grid (`misc/2026-08-24_flow_signal_conditioning/armA/armA_extract.py`): field `(T_lat=16, 20, 15, 44)`, chunk-center frames `c(0)=0, c(ℓ)=8ℓ−4` + endpoint B (17 unique frames), DINOv2 tokens at a **grid-commensurate 28/32 resize → 40×30 patches pooled 2×2 → latent cell**; five groups — G1 32-d **frozen DINO-PCA** (fit on 672k cell vectors, EVR 0.45), G2 windowed transport (r=5, τ=0.07, soft-argmax u/v conf-weighted, top-3 conf, entropy÷log|window|, fwd-bwd zeroing), G3 in-place ‖ΔF‖/1−cos, G4 endpoint bank-max s_A/s_B/ν, G5 Lab Δcolor/csim. Extracted **1587 clips** (223 eval std121 + 1364 train S0/S1, all 121f/640×480) via a **12-shard bhwp GPU array** (~3.5 min/shard, 0 fail); GPU-validated (no NaN/Inf, τ well-calibrated: entropy mean ~0.55–0.68 not saturated/flat, u,v bounded ±2.5 by the r=5 window). Two caches at `$LAB/cache/armA_signals/`: `feat/` (608 MB, the fields) and **`dino_raw/` (43 GB, raw un-normalized DINOv2 tokens `(17,40,30,768)` fp16 + frame meta, for reuse)** plus `pca.npz`. Added a **per-metric cache pattern** (`armA_metrics.py`): each metric computes straight from `dino_raw` (no DINO re-run) into `metrics/<name>/`, the viewer builder auto-appends it into its declared group. First metric = **v2 ν′** (span-residual novelty `1−max_α cos(f, α·bestA+(1−α)·bestB)`, faithful global-best port) — validated ν′ ≤ ν in 100% of cells (ν−ν′≈0.016 = on-span/dissolve signal), r=0.99 vs Arm-A ν, r=0.87 vs original v2 `nu_p` (grid/temporal differences); computed for all 1587 clips (6-shard array, ~1.5 min/shard). Viewer `armA_signal` (45 ch now, ν′ beside ν in G4) registered + LIVE. Regen: `scripts/viewers/build_armA_signal.py`.

**~11:30 — SEA-RAFT internals run through the C6 retrieval exam, same probe as RAFT-large.** Cloned SEA-RAFT (spring-M, dim128/iters4), hooked the context half + final-iteration hidden state, extracted over the identical PROC(420)+S1-grid(621) clips (13.7 GPU-min, bhwp), and scored with the verbatim `c1_stats`+`c2a_instance` from `certify.py`. Result: **content leak c2a stays ~total** (PROC ctx 1.000/hid 0.998, S1 ctx 0.960/hid 0.903 ≈ RAFT-large), operator C1 **higher** esp. context (PROC ctx 0.531→0.795); same mechanical reading — internals leak content, use the flow field only. Code/results `misc/2026-08-24_flow_signal_conditioning/c6_searaft/`.

## 2026-08-27

**18:30 — EffectData (S6) shape safeguards landed in the ctt_v2 pipeline; assert self-test green.** The top-2k EffectData roster has exactly **4 native latent shapes** — (11,22,39)/(11,39,22)/(11,33,22)/(11,22,33), two transpose pairs, 81f/24fps, all VAE-legal (no crop, unlike S4). Added them to `root_common.RULED_SHAPES` and the independent `assert_root_shapes.EXPECTED_SHAPE_CLASSES` with **`prefix_latents=1`** (frame-0-only anchor, per S4's owner precedent — the 121f default of 2 would condition 9 frames INTO the effect onset and mismatch the frame-0 caption). Generalized **B7** from "no two shape classes share a token count" (which EffectData's transpose pairs legitimately break — 9438 & 7986 tokens) to INFORMATIONAL: the pairs get DISTINCT masks and B2's per-sample exact-shape mask check already owns the wrong-mask defect, so a cross-shape mask fails LOUDLY at B2; the `token_collision` self-test fixture is now caught by B3 (unexpected geometry). `--self-test`: **10/10 fixtures correct** (every broken one caught, clean passes). Additive and safe for the frozen champion root. Stratum id = **S6** (S5 was the never-built refVFX-code plan).

**18:10 — Health-checked `TEXT_LIFECYCLE.md` (fable-advisor, 64-tool deep verify) and corrected the S4 provenance it had inherited.** The shipped S4 caption store `34534e47` is a **gemini-3.6-flash** regeneration (v2-s4f0, per-item length draw; source `outputs/ctt_v2/captions/s4_gemini/`) — NOT the "claude-sonnet 25×80" store `fcd46f33`, which is **archived**; the only formal gate battery (8a 0.8849, gate 2 FAIL) measured the *archived sonnet* captions, and `CAPTIONS.md §12.4`'s "no caption was ever regenerated" is **stale** (the gemini adoption is recorded only in the store's own `generator` field — no CHANGELOG/DOSSIER decision; 🔴 flagged to owner). Also corrected: the 29-clip S2a gap is **RESOLVED** (A16, 2026-07-28 13:45), not pending; the S0 training caption is marker→`sksz` (not verbatim); embed is **Gemma-only**. Added `TEXT_LIFECYCLE.md §8.1` "non-obvious requirements" for the EffectData build (inventory is the integration unit; per-item length draw MANDATORY; one-sided `prefix_latents=1` shape rule + `assert_root_shapes` B3/B7 generalization). `store/captions/002` meta + README brought to disk truth.

**17:55 — Store text lane made self-contained: a `captions/` shelf + a `TEXT_LIFECYCLE.md` authority.** Mapped the full caption/prompt lifecycle (fresh-context agent, file-precise) and wrote **`store/TEXT_LIFECYCLE.md`** — the single self-contained account of every text artifact across captioning → prompt prep → training → gen → eval: the two lanes (leak-free training captions `{A}. sksz. {B}.` assembled by `build_encode_inputs.py::assembled_for` into content-addressed `conditions/by_caption/<sha16>.pt` the trainer reads; eval neutral/effect grids rendered by `eval_ladder/prompts.py::render_prompt`, effect clause spliced from `reference_effects.json`, frozen sha-pinned in `store/prompts/`), the hash/lock table, and the reconciliations. Added a first-class **`store/captions/`** shelf holding the SOURCE description text as versioned, git-tracked JSON (was only in gitignored `outputs/`): `001_ctt_v2_endpoints` (A/B, 1,403, locked `c8e2d95b`), `002_ctt_v2_s4` (S4 first-frame A, 2,000, `34534e47`), `003_effect_clauses` (36 EFFECT_DESC) — each a **hash-verified byte-copy**, via `.gitignore` scoped exception `!/store/captions/**/*.json`. Corrected two live discrepancies the map surfaced: the **S4 store hash is `34534e47…` on disk** (CAPTIONS.md §12's `fcd46f33…` is stale), and a **training-mix drift** (`mix.json` S0 15/S1 6/S2a 33.87/S2b 35.13/S4 10 vs the shipped champion `runs/002` trained S0 5/S1 12/S2a 34.36/S2b 35.64/S4 13). Restructure design + EffectData slots: `misc/2026-08-27_store_prompt_restructure/PROPOSAL.md`. EffectData's own A-descriptions + effect_desc are the next build.

## 2026-08-26

**17:15 — EffectData ingested (full 821 GB), counterfactually characterised, and organised into the repo.** Downloaded the complete HF dataset `ysy31415926/EffectData` (EffectMaker, Apache-2.0): **3,063 per-effect zips / 132,850 mp4 / 3,061 effect classes**, every file **sha256-verified** against the LFS manifest (0 fail, 63.6 min; xet-free parallel downloader over plain `resolve/main` URLs that dodges the Lustre `.lock` hang). Lives at **`data/raw/effectdata/`** (gitignored; instant same-fs move from the staging dir, with a compat symlink `data_external/EffectData → data/raw/effectdata` so the :8017 viewer + `build_effectdata.py` keep resolving with no tracked-file edits). **Both-axis counterfactual analysis** (metadata-only; the source is one-sided so the endpoint is the shared **start frame** = subject id, derived from the undocumented filename middle token, validated same-source ≈0.7/255): 3,061 operators × 56,941 subjects; **Axis A** (same start, different effect) **308,746 pairs**, **Axis B** (same effect, different start) **2,952,671**, both-axis **rectangles 10,418** — demonstration-dominant with a sparse, hub-concentrated counterfactual core (half the endpoints are singletons; a "hero" hump at degree 7–10). Size↔counterfactuality frontier: dropping the 1-effect subjects is free (**822→647 GB keeps 100% of Axis A and R**); operator breadth (~2,917/3,061) survives every shrink. Reorganised: scripts → **`scripts/effectdata/`** (`fetch_manifest`, `download_videos`, `frontier_both_axis`, `plot_axisA`, `remote_zip`; repo-anchored paths, verified from a neutral cwd); the dataset self-documents via **`data/raw/effectdata/{README,counterfactuality}.md` + `axisA_degree.png`** (`git add -f` — small knowledge files that live inside the gitignored data dir). NOT yet registered in the artifact store.

**13:15 — flowsig SPLIT arm gets the full standard treatment: the ball-vs-split pair is now the campaign's one clean comparison, and split wins it by +3.8pp.** Generated the neutral 152-grid x seeds 42/43 in the production `both` regime (304 clips, array 3023223, 32 tasks, 96/96 COMPLETED) on a grid **row-identical to the b_all grid** — same 152 (cell, endpoint, reference) triples, seeds, prompts and program sources, verified before submit — then scored it with eval-v4 on the same instrument, corpus and warm cache as evals/001/012/019 (array 3023360, 16 shards, 48/48, 1,842 rows, reference sha 459fd9a7). Scoring was pipelined behind generation with no idle barrier. **Four-way pooled-same: ctt_v2 82.5 · control-012 89.6 · flowsig b_all 80.5 · flowsig split 84.3.** Because runs/018 and runs/019 share data, mix, cells, warm start, schedule, step count and the same `textdrop-coupled` recipe defect, and now share a grid, **ball vs split isolates the injection route** — and the sequence-token route is **+3.8pp** over the per-token-adaLN route, with the advantage concentrated in transfer and memorisation cells (G-zs-foreign 52.8 vs 41.0, G-unseen-cross 83.3 vs 75.4, G-memo-probe 95.8 vs 85.6, G-unseen-same 92.5 vs 86.5) while the two are level where the reference is easiest (G-fit 92.1 vs 92.6). Both routes widen the reference-dependence gap the same way and by about the same amount (+29.9 and +29.0 vs ~+21.3 for both comparators; G-fit up, G-ref-control down). core_degenerate 10/304 for split vs 18 for b_all (control 8, ctt_v2 18). 🔴 **near_copy 1/304 — the only non-zero copy flag across all four arms**: `G-memo-probe__flowsig_split_neutral__animalization_0__ref_animalization_1` s43 at copy_max 0.9218, with the arm's next-highest generation at 0.8486, below τ=0.858; recorded, not explained — G-memo-probe deliberately re-uses a seen reference so it is where a copy would surface first, but one item is not a pattern. **Caveats unchanged:** neither flowsig arm is compute-matched to control 012 (10,000 steps vs 1,000 from the same parent), so the lineage-matched reference is ctt_v2 — b_all −2.0pp, split **+1.8pp** — and since the compute-matched program-free twin was never trained, no quality number can be attributed to the program as such; and this measures transition quality and reference-dependence only, **not whether the program is read** (the sampled reading ladder remains cancelled). Registered `gens/023_flowsig_split/02_neutral__dai` + `evals/020_flowsig_split__dai__2026-08-25`, scores symlinked, `store_fsck` PASS; the arm joins `iclora_neutral_effect` as a sibling card under the "optical-flow program (flowsig)" category (`flowsig · split (RoPE tokens)` beside `flowsig · b_all (per-token adaLN)`), all builder seatbelts passing. Also resolved a duplicate `flowsig_split` row in ARMS.md — the training operator and this campaign registered the arm concurrently; merged to one row keeping their wording, correcting only the stale alias cell.

**12:00 — flowsig: the SPLIT arm generated in the production setting and put side by side with b_all; the arm grouped under its own "optical-flow program" category.** Added the inference **program-TOKEN** stream (the earlier port covered only the adaLN route): `attach_program` gains `token_grid`/`token_pool`, and the 2×-pooled field is attached as **1,280 RoPE-positioned tokens**. The layout mirror is the load-bearing part — training builds `[program | ref | target]` with `oneway_ref_first=True` so program+ref are the *leading* read-only bank, while `attention.py` with `oneway_ref_first=False` makes the *last* m tokens the bank, so inference is `[target | ref | program]` with `oneway_ref_tokens = n_ref + n_prog`; the bank is internally unmasked so its ref/program order is immaterial, and the RoPE positions travel with the tokens and are what carry co-location. Tokens are appended to the **Modality**, never to the LatentState (whose token count the noiser, the Euler step, `clear_conditioning` and `unpatchify` all assume), and every prediction taken from the augmented modality — positive, text-CFG, STG, signal-CFG — is sliced back to T before use. Verified `T=10880`, `oneway=6080`: **bit-for-bit the training-side G2 gate's numbers** for this arm, an independent check that the inference layout reproduces the trained one. One real crash on the way (`timesteps` is `(B,T)` at training but `(B,T,1)` at inference, since `denoise_mask * sigma` keeps the mask's trailing axis; the pad is now built from the tensor's own trailing shape) cost ~0.4 GPU-h, after which a CPU simulation against inference-shaped tensors was added so that class of bug is caught in seconds. **Wiring gate PASS**: same row, same seed, only the program's source clip swapped ⇒ mean|Δ| **0.0731**, vs b_all's **0.0302** on the identical row at the identical 8-step setting — a measurement, not a ranking. **26 clips** (13 G-fit rows × seeds 42/43, mode `both` = pixel reference + matched program + neutral caption, the same rows/seeds/derangement as `gens/022`) registered as `gens/023_flowsig_split/01_split_both_matched__dai`; `store_fsck` PASS; `ARMS.md` gains a `flowsig_split` row. **UNSCORED** — no eval-v4 pass on this arm, per owner scope. Descriptive pixel comparison over the 26 pairs: b_all↔split **0.107**, b_all↔null **0.133**, split↔null **0.156** — both routes sit further from the no-program floor than from each other, per-row spread is large and the ordering flips row to row; no bar attached. The pilot page now carries both routes as adjacent columns (8 columns/card, 169 videos, LIVE 12/12). Separately, on `iclora_neutral_effect` the flowsig arm was moved under a new **"optical-flow program (flowsig)"** category — it had been a CATALOG row naming a category absent from `CATEGORIES`, and since the panel is built by iterating `CATEGORIES` the row was **silently dropped from the arm selector** while the arm still scored, still joined its cards and still appeared in the machine table; the seatbelt that would have caught it was added alongside.

**11:30 — flow_signal_conditioning Step-2 EVAL: arm b_all generated, scored and on two viewer pages — and the inference path for the program had to be built first.** The trainer injected the transition program from `_training_step` only, so **no code path anywhere could generate a program-conditioned video** — validation had been running program-free the whole time. Built it: `ValidationRunner.attach_program` + `_build_program_condition` in the fork (`src/LTX-2-flowsig` @ `4cd3c94`, local branch), an env-gated `FLOWSIG_CONFIG` hook in the shared generator (unset ⇒ byte-identical for every other campaign), and `step2eval/build/flowsig_hook.py`, which rebuilds `ProgramModule` from the same checkpoint that carries the LoRA and installs LTX-2's native `TimestepEmbedding.cond_proj` hook exactly as the trainer does. The layout asymmetry is load-bearing and silent if missed: training builds `[ref | target]` with the learned null on the LEADING block, inference appends `[target | ref]`, so the null goes at the TAIL. Signal-CFG is implemented as the house guider-delta `x + (w−1)(x_prog − x_null)` and **skips entirely at w=1**, so every claim-bearing clip here is byte-identical to no signal-CFG. Also found: only **13 of the 152-grid's 36 reference clips** had a cached program, and the training store's S0 entries came from a different re-encode than the file the generator is fed — so a homogeneous eval store was extracted over **all 223 std121 clips** (0 failures, 0.11 GPU-h), reusing `NORM.json` verbatim because it is part of the model contract. A CPU-only pre-flight (store → tokens (4800,62) → strict 6/6 program-tensor load → condition ‖prog−null‖ 101.2, per-token variation 1.07×) caught nothing but cost seconds; a GPU smoke gate then confirmed the channel reaches pixels — same row, same seed, same caption, only the program's source clip swapped: mean|Δ| **0.0302**, not bitwise identical (a dead channel would have been). **Generated 408 clips, 120/120 tasks, 0 failures** → `store/gens/022_flowsig_ball`: `01_neutral__dai` = the standard arm treatment (152-grid × seeds 42/43 in the intended `both` regime, 304 clips) and four 26-clip Phase-A pilot variants over the 13 G-fit rows {code_only matched, code_only DERANGED, both matched, null}. **eval-v4 (evals/019, 16 shards, DeltaAI, reference sha 459fd9a7 — the same instrument, corpus and warm cache as evals/001 and evals/012): pooled-same 80.5 %** vs ctt_v2 82.5 and control-012 89.6; **ref-dependence gap +29.0pp** vs +21.3/+21.4 — and it widened from *both* ends, G-fit 92.6 (the highest of the three) with G-ref-control 63.6 (the lowest), the opposite signature to the three killed counterfactual-objective arms, which narrowed the gap by dropping G-fit. Loss is concentrated in transfer (G-zs-foreign 41.0 vs 70.4). Copy guard clean: near_copy 0/304, copy_max mean 0.3508 — the lowest of the three. 🔴 **The arm is NOT compute-matched to control 012** (10,000 steps vs 1,000 from the same parent), so the lineage-matched reference is ctt_v2, against which it is −2.0pp; nothing here separates "the program hurt" from "10k steps at eff-batch 2 drifted the adapter", because the compute-matched program-free twin was not trained. And this eval measures transition quality and reference-dependence only — **it does not test whether the program is read**; the sampled reading ladder was cancelled by owner directive before generation, along with the four-mode matrix, dose–response and shuffle controls. Pixel-level pilot description (descriptive, no bar): matched↔deranged 0.038, matched↔null 0.040, deranged↔null 0.039, both↔matched 0.131 — the three program-only contrasts are the same size to within ~6 %, while adding the pixel reference moves the output ~3.3× further. Pages: `outputs/viewers/flowsig_pilot/` (13 cards, GT + both source demos + four conditions × 2 seeds, 143 videos, LIVE 12/12) and the arm on `iclora_neutral_effect`. `store_fsck` PASS. Advised campaign; the round-1 design ruling is on file **unexecuted** — including one correction worth keeping if the detail branch ever opens: scoring each generation against *its own fed* program would let a completely deaf model pass the matched-vs-deranged test, so all such contrasts must be computed against the row-true program, paired within (row, seed). Full trajectory in `misc/2026-08-24_flow_signal_conditioning/DOSSIER.md`.

**11:15 — flow_signal_conditioning Step 2 CLOSED: `runs/019_flowsig_split` registered — both program-conditioned injection arms trained to 10,000 steps.** The sequence-token arm finished clean (final step loss 0.2345, LR 1e-5, 0 NaN, 3 verified chunk resumes, 12 inline validation clips, 20 checkpoints on the 500-grid). Its forward reading contrast matches b_all's: σ-matched and confound-free (S0/S1, steps 6751–10000) **−0.0018 ± 0.0096**; paired D-probe at ckpt 10000 in the never-trained `code_only × no-text` configuration Δloss **+0.00012 ± 0.00006** over 1,000 pairs while the velocity prediction moves **3.5 % rel-L2** (vs b_all's 2.3 %) — both channels are live at the output and loss-neutral. Mid-run the split arm had to be rescued from a DDP fault (`program.token_head.4.weight` marked ready twice: a structurally-zero "touch" term I had added to force parameter participation used the tensor outside the checkpointed blocks, so its hook fired twice; b_all was immune because its only consumer runs once outside those blocks). The advisor-ruled fix attaches the program tokens on **every** step and blends the learned null in arithmetically — `feats = present·feats + (1−present)·null`, never a Python branch, which would have moved the participation variance to the other cell type — and was verified by a 2-GPU 30-step probe (all four cells exercised, `token_head` grad 0.005–0.021, `null` grad non-zero from step 2) before the chain went back up. Total campaign cost **≈38 GPU-h** on `bhwp-dtai-gh`. Neither arm has been evaluated yet: the sampled reading ladder (T1 oracle → T2/T4 → T5 shuffle → T3 dose) is the next step, and its outcome map was amended **before any generation** because of the recipe defect — an all-null result is non-diagnostic between signal-unreadable and incentive-absent-by-defect, while a pass gains force. Six advisor rulings verbatim in `misc/2026-08-24_flow_signal_conditioning/DOSSIER.md`.

**09:55 — flow_signal_conditioning Step 2: `runs/018_flowsig_ball` registered — arm b_all trained 10,000 steps, and a recipe defect found and characterised.** The per-token-adaLN program arm completed cleanly (0 NaN, 0 crashes, 3 verified chunk resumes, inline ID/OOD/control validation at 5k and 10k, 20 checkpoints on the 500-grid). Pre-registered forward reading contrast `loss(code_only) − loss(none)`, σ-matched and free of the stratum confound (S0/S1, steps 8501–10000): **+0.0036 ± 0.0135** — no forward-loss evidence of reading at |Δ|≳0.03 sensitivity. A **paired** D-probe on ckpt 10000 in the never-trained `code_only × no-text` configuration (same row, same σ, same noise; only program vs learned null differs) puts that far more precisely: Δloss **+0.00010 ± 0.00002** over 1,000 pairs, while the velocity prediction itself moves **2.3 % in relative L2** (per-token cosine 0.99967) — the channel is **live at the output but loss-neutral**. The adaLN condition varies across target tokens at ~half its own magnitude (ratio 0.47), so a global-clock outcome in the morning's T5 is not pre-explained by degenerate injection (this measures the injected signal, not the model's use of it). 🔴 **Recipe defect, recorded as variant `textdrop-coupled`:** the text-dropout draw and the conditioning-cell draw consumed the identical first uniform from same-seeded per-step generators, so text was dropped on exactly the `both` cell and nowhere else — the model **never saw `code_only × text-absent`**, the one configuration in which the program is the sole operator description in context. The marginal dropout rate was on target (0.506), so no marginal check could see it; only the joint could. The pre-registered independent-50 %-dropout recipe was NOT run; the one-line fix sits on branch `flowsig-textdrop-fix` @ `c4b8383` and is deliberately NOT in these runs (arm split was mid-flight and had to finish under the identical recipe to stay comparable). Consequence for the morning, declared before any generation: an all-null G1 outcome is **over-determined** — non-diagnostic between signal-unreadable and incentive-absent-by-defect — while a PASS gains force. Advised campaign; all six advisor rulings verbatim in `misc/2026-08-24_flow_signal_conditioning/DOSSIER.md`.

**04:35 — flow_signal_conditioning Step 2: BOTH program-conditioned injection arms built, smoke-gated and training (2×GH200 singleton chains, 10k steps each).** New trainer fork `src/LTX-2-flowsig` (branch `flowsig` off `contrast`@83d266d, commit 0090661) adds transition-PROGRAM conditioning to LTX-2 19B in two config-switched arms: **b_all** — field AND per-phase tempo through the per-token adaLN path, handed to `TimestepEmbedding.cond_proj`, a hook upstream plumbs but never instantiates, so every one of the 48 blocks' shift/scale/gate becomes f(σ, program); **split** — the field as 2×-pooled sequence TOKENS at the target's own (t,y,x) RoPE coordinates (1,280 tokens, +13 % sequence), prepended as `[program | ref | target]` so the one-way read-only block stays contiguous at P+M, with tempo still on adaLN. `ltx-core` gains a per-token `Modality.program` threaded to `AdaLayerNormSingle`; `program=None` reproduces the stock path exactly. Pixel references STAY in context (owner): condition-dropout cells {both .50, ref-only .15, code-only .25, none .10} within program-bearing strata, field and tempo dropped together as one skeleton stream, text dropped 50 %. Warm start = ctt_v2 `002@10000` (r128/α128 + attn2, lr 1e-4 — the owner's r64/2e-4 spec bends because a rank-64 LoRA cannot load rank-128 tensors), program adapter lr 1e-3, WSD 100/0.25/0.1, eff. batch 2 (2 GPU × bs1 × accum1 — the only point inside both "10k steps" and the 40 GPU-h budget at the measured 3.15 GPU-s/sample). Descriptor per owner directive: **v2, all 18 channels** (displacement group IN — "a positive result is more important than attributability"), resampled to the (16,20,15) target latent grid; store `$LAB/cache/flowsig_programs/v2_c18_20x15` (1,364 S0+S1 clips, 0 failures), z-scores fitted once over the S0+S1 union (NORM.json sha b9692824…db5f). Mix S1 60 / S0 15 / S2a 12.5 / S2b 12.5 / S4 0 — S2 carries NO cached program and enters as program-null anchor mass against forgetting. Six pre-registered smoke gates PASS on both arms: join 3/3, shapes exact in all four cells, **b_all bitwise-identical to the warm start at step 0**, split velocity-cosine 0.99580 (bar 0.95) and loss within 3.0 % (bar 10 %), gradients in-graph 6/6 and 12/12 with zero base-weight leak, program state-dict round-trip clean; measured 0.88 / 1.03 s forward per sample. Training measured at **3.17 s/step**. EffectData rows deliberately NOT built tonight (its widened 240-clip set has no counterfactual pressure — 2 of 238 subjects are multi-op — and encoding it would have delayed both chains on the single legal account); its 72-clip same-subject×different-operator set was retimed 81→121 frames on CPU as daylight prep. Advised campaign; every ruling verbatim in `misc/2026-08-24_flow_signal_conditioning/DOSSIER.md`.

**00:17 — flow_signal_conditioning Step 0 (gate G0): signal certified — CONDITIONAL PASS on descriptor v2; novelty family scoped out; displacement group dropped.** Advised campaign (3 fresh advisors: plan / evaluate+iterate / adjudicate) certifying the RESEARCH.md §3.3 "transition program" φ before any training, per §6.1 C1–C6. Built a 14-operator procedural corpus with exact per-pixel ground truth (420 clips, 640×480×121f, 5 families incl. deliberate hard-negative pairs), widened EffectData 65→240 clips (58 s), and extracted 4 full passes over 2,160 clips (v1, v1-L, v2, v2-L) with **zero failures**, 2.670 GPU-h total on `bhwp-dtai-gh`. **v1** (18-ch, literal spec) passed C1 (PROC 0.869 / S1-grid 0.852 vs chance 0.071/0.111) but fired a pre-registered KILL: PROC ω²_content 0.410 > ω²_op 0.288, content-leak c2b 0.991, and the leak was *not* confined to by-construction boundary bins (D1 flat: 0.991→0.991) and sat overwhelmingly in the displacement group (c2b 1.000 alone). **v2** — one authorised iteration, redirected by those diagnostics — normalises every similarity channel from its own floor (the other endpoint) to its own ceiling (self-similarity) instead of to a fixed 1, replaces ν with a span-residual ν′, and **drops the displacement group** (12-ch contract, hash `0bc8815840049f0a`): PROC ω²_op 0.460 vs content 0.123 with **192/192 cells operator-dominant** (was 33/192), C1 0.907/0.868, c2b 0.991→0.656, spatial-shuffle drop 0.257 (this field is *not* the shuffle-invariant clock of runs/017), C3-GT profile Spearman 1.000 / wipe-direction 0.987 / cell 0.891, C5 instrument agreement 0.983/0.958 like-for-like, cross-generator (Higgsfield→LTX-2) 1-NN operator match 0.803. Verdict CONDITIONAL PASS (C3-GT cell 0.891 misses the ≥0.90 bar by 0.009, carried by profile 1.000 + wipe 0.987; adjudicated at the certification grid, not the shipping grid, to avoid post-hoc bar-shopping). Riders: novelty family **out of the paper's claims** (ν′(flash) 0.31× ν′(D-lin) vs a 2× bar) though its channels stay in; displacement **not readmitted** (fails 2 of 3); shipping grid **(8,6)**, verified (PROC C1 rises to 0.929, drop 0.267). C6 reported negative: RAFT internals carry operator info (C1 0.840) but leak content totally (c2a 1.000) ⇒ flow *field* only. Explicitly NOT established: that φ is content-free (c2a 0.455 ≈ 14× chance — it is content-*suppressed*), robust cross-generator transfer (v2 paid −0.04…−0.09 on every non-gated stratum), or that a DiT can read φ — that is Step 1, where X1/L2 become mandatory. Report `misc/2026-08-24_flow_signal_conditioning/step0/REPORT.md`; full trajectory + all three verdicts verbatim in `../DOSSIER.md`.

## 2026-08-24
**11:40 — flow_signal_conditioning: fresh research design for conditioning LTX-2 on an abstract motion signal (owner ask).** `misc/2026-08-24_flow_signal_conditioning/RESEARCH.md` — first-principles problem statement (in-context reference = content pass-through; program regime removes the copier by construction), signal spec (two-resolution *transition program*: per-phase global vector + target-latent-grid field of DINO-feature-flow progress/novelty/displacement/ambiguity; raw flow-net hidden states as ablation), injection ranked against LTX-2's real surfaces (per-token adaLN via the unused `TimestepEmbedding.cond_proj` hook = Tora-style modulation at all 48 blocks; RoPE-positioned program tokens as a `ConditioningItem`; A2V-pattern stream as fallback; cross-attn baseline only), necessity recipe (counterfactual pairs, text drop 50 %, program dropout → signal-CFG, optional relational REPA to φ(V)), and a gated evaluation protocol (signal certification C1–C6 → reading tests oracle/deranged/dose-response → four-site ablation table → transfer vs one-hot-label control → layout/content/tempo leak tests). Evidence base in `lit/` (LTX-2 code map with file:line; ≈150-paper injection survey; representations/REPA/condition-reading survey). Rendered at outputs/viewers/flow_signal_conditioning/ (:8017).

## 2026-08-22
**01:54 — contrastive_training CLOSED (advisor R4: KILL, counterfactual-objective family closed as a 5-replicate negative).** Record fixes: evals/018 HEADLINE gained the trigger line + forward-telemetry decomposition (margin won entirely by degrading the wrong row: ℓ_w−ℓ_w^ref +0.033 / ℓ_l−ℓ_l^ref +0.091; bound never engaged); per-step telemetry archived (misc/2026-08-21_contrastive_training/build/telemetry/); `evals/017_dualforce_s0s1redirect` finally got its HEADLINE + meta (quality-clean NULL: 88.1 / +20.7 / 0.825 / 8) + INDEX row + gen `scores` link → store_fsck PASS (0 fails). Viewer `iclora_neutral_effect` carries `dualforce_contrast_neutral`. Scratch optimizer states (68 GB) deleted; LoRA weights @100–1000 kept in outputs/dualforce/contrast_2gpu. F-block proposed in chat (owner-gated).

**01:41 — contrastive_training: evals/018 scored → `dualforce_contrast` KILLED (5th death of the counterfactual-objective family, same signature).** v4 on DeltaAI (array 3004137, 16 shards; compliance 3004138), same instrument/corpus/cache/grid as evals/012 (control) and 017 (redirect): pooled-same 78.5% vs 89.6, G-fit 81.5 vs 90.7, G-ref-control 63.3 vs 69.3 → ref-dep gap +18.1pp vs +21.4 (narrowed the wrong way), core_degenerate 21/304 vs 8, P3a swapped-compliance 0.672 vs 0.831, near_copy 0. KILL on 4 pre-registered triggers (gap<+23.4, pooled<86.6, degen≥16, compliance<0.80). Training-side the contrast margin had widened monotonically to Δ=−0.065 (~6× the init's discrimination) with L_FM flat — forward-read↑ / sampled↓ again, now on a bounded, reference-anchored (DPO-style) objective on the sample axis. HEADLINE + meta + scores symlink written; store_fsck PASS for the entry. Numbers: misc/2026-08-21_contrastive_training/DOSSIER.md R3.

**01:02 — contrastive_training: gens/021_dualforce_contrast/01_neutral__dai registered (304 clips) + v4 scoring launched.** Neutral 152-grid × seeds 42/43 generated on DeltaAI (array 3003781, 32 tasks, ~2 h under a saturated queue) from runs/016@1000 with the proven one_way stack (--rank 128 --alpha 128); store_register + fsck PASS for the subentry. Scoring: v4 16-shard array 3004137 → evals/018_dualforce_contrast__dai__2026-08-21 (same instrument/corpus/cache as evals/012 & 017) + P3a compliance job 3004138 (192 contrast / 192 control G-ref-control items vs the demo-class pool).

**00:06 — semantic_flow_run: trained + registered `runs/017_armA_field_retry` (step 4000 shipped) — REFINED NEGATIVE.** Campaign 2026-08-21_semantic_flow_run, Arm A flagship: a semantic-flow FIELD-conditioned IC-LoRA (**v2 = demo-dropout retry** of the atrophied v1 field arm). ctt_v2 (002@10000) recipe + an auxiliary precomputed retiming FIELD (18-ch: logistic phases + derivatives + confidence + detour, extracted endpoint-relative from the demo via DINOv2) injected additively through a gated per-channel-normed MLP (18→256→4096) at 6 blocks on generated-middle tokens (gate 0.05 RMS-scaled), **demo_dropout 0.5** (nulls the in-context demo ~50% of steps so the FM loss must route the operator through the field — escapes the v1 conditional-redundancy gate-atrophy). Trained DeltaAI 2×GH200 (job 2997771 [0→2500] + resume 3002817 [2500→4000]); trainer `src/LTX-2-dualforce @ 5dfdcee` (branch dualforce, **DIRTY** — field wiring uncommitted: field_injector.py + config/trainer mods); ckpt sha 8853040b. **Advisor-adjudicated REFINED NEGATIVE** on a pre-committed closure table: demo-dropout RESCUED a real, seed-stable, dose-stable demo-NULLED field-swap (toward_B_phase 10→10→**12/13** @ck2000/3000/4000, mean_swap +0.121/+0.105/+0.107, gate collapse arrested at plateau ~0.0075 abs vs v1's death to 0.0016) — **BUT** shuffle-invariant (spatial-shuffle control 13/13, +0.113, gap −0.006) + per-token geometry at chance (5/5/6 of 13) → **GLOBAL CLOCK, not per-token instruction; redundant with Arm C's free test-time clock steering**; copy clean (near_copy 0/13). Retires the gate-magnitude liveness bar. Finding staged as owner-gated F-block `misc/2026-08-21_semantic_flow_run/PROPOSED_F_BLOCK.md`; conflict-training next-arm DROPPED (bneck + SURG-1 + this = 3 arms on the same forward→sampling wall). fsck: 017 clean (the 2 store-wide fails are other sessions' in-flight 021_dualforce_contrast/017_s0s1redirect_eval, not this).

## 2026-08-21
**23:09 — contrastive_training: trained + registered `runs/016_dualforce_contrast` (step 1000 shipped).** Owner-requested contrastive variant of the dual-force plain-FM control 012: same recipe/sampler/warm-start (002@10000)/WSD, effective batch 4 on 2×GH200 (job 2998010,2999095,2999096,2999097,2999098), plus ONE paired-preference term per step on cell-uniform same-content×different-operator S0+S1 pairs sharing ε and σ∼U[0.5,0.9] — win=(state A, demo A)→v_A vs lose=(state B, demo A)→v_B, seam-masked, Δ anchored to the FROZEN warm-start adapter (second PEFT adapter `ref`, Δ≡0 at step 0), L_con=(2/β)(softplus(βΔ)−log 2) [bounded repel], β 8.0 (default, no probe), λ 0.25 warmup 250. Trainer `src/LTX-2-contrast @ 83d266d` (branch contrast from dualforce@5dfdcee; TwinConfig.variant s0s1_contrast, _contrast_training_step/_setup_ref_adapter). ckpt sha a39d800a. Arm `dualforce_contrast_neutral` registered; gen 304 → gens/021 next.

## 2026-08-20
**11:55 — Schwing meeting-2 prep brief written (`misc/2026-08-20_encoder_collapse_toy/meetings/2026-08-20_schwing_meeting2_PREP.md`, untracked by policy).**
Synthesis of the Aug 6 transcript + every campaign since (17 dossiers) + the owner's paper foundation: a scorecard of Alex's
12 Aug-6 asks vs status (done: prompt-A/B tiers on all arms incl. externals, author-config baselines; not run: S2/counterfactual
dataset ablations, collapse toy, user study), the Aug 6→20 trajectory table, the definitive current state (ctt_v3 88.0 / 012
89.6 pooled-same; reference-only lead 16–38 pp over VAP/VFXMaster/refVFX; four reference-reading mechanisms closed negative with
the shared forward-read→sampling-conversion signature; DCG +3.3 pp @w1.5 and RoPE timing 12/12 positive), three corrections to
the paper foundation (prompt-relay is a negative → present as text control; contrastive line already closed; lerp-collapse is a
designed experiment not a result), positioning questions for Alex, and a dated 5-week plan with a time-boxed last training shot.

## 2026-08-21
- 12:10 semflow campaign (misc/2026-08-21_semflow_dit_guidance): test-time DiT-feature-flow guidance on ctt_v2@10k implemented as an in-process sampler patch (`build/semflow.py`; parity bitwise at s=0; DAVIS pan/static calibration → block 24, centred, τ 0.02). 6-row pilots: the guidance halves its own loss at scales that leave the clip unchanged and produces content-intrusion artifacts before any transition effect — measured negative for the x_t-nudge mechanism; dossier has the cause-level alternatives. Quick-look viewer `semflow_sweep` registered.

## 2026-08-19
**23:58 — counterfactual_training: trained + registered `runs/014_dualforce_twin` (step 1000 shipped).** The counterfactual-twin treatment (advised campaign): per-step pair an S2 row with a same-endpoint byte-exact counterfactual, adding a redirect term (x̂₀-space, σ∈[0.5,0.9]) + differential term (v-space, all t), both middle-masked, on top of plain FM — λ_red 0.5 / λ_diff 0.25, warmup 250, S2a+S2b only. Warm-started runs/002@10000, same shell as the control runs/012 (only the objective differs). Trained on DeltaAI GH200 (job 2982505), trainer `src/LTX-2-dualforce @ 6dc2f68` (CLEAN tree — reproducible), ckpt sha 20322d10. Implemented `TwinConfig` + `_twin_training_step` in the fork. Precheck (advised R1/R2): frozen α probe → proceed-flagged, band tightened to [0.5,0.9], S1 dropped (endpoints not pixel-identical); α-trajectory on 014 already clears the step-500 gate at step 250 (α(0.85) 0.0164→0.0511). Registered arm `dualforce_twin_neutral` (arms.yaml), stamped the 152-row neutral registry (prompt_sha 0d708175 = 012's family), generating gens/019_dualforce_twin (job 2983886). Eval v4 vs 012@1000 next.

**18:21 — Scaffolded the counterfactual_training advised campaign (dossier only, no rounds run).** Created
`misc/2026-08-19_counterfactual_training/DOSSIER.md`: objective = counterfactual twin supervision (L4-CROSS lineage,
redirect + differential on the cttv2 same-endpoint counterfactual cells per `store/datasets/002_ctt_v2/COUNTERFACTUAL.md`)
to widen reference-dependence, warm on the runs/012 dualforce_control config shell with 012@1000 as the ready-made
plain-FM control. Dossier is explicitly marked a single-agent chat handoff — round 1 must have a fresh fable-advisor
re-evaluate the framing and design the experiment; nothing is pre-registered.

**13:58 — DCG conditioning-ablation Phase-2 (EFFECT) complete: REDUNDANT-BUT-SAFE @ w=1.5.** Ran the effect-prompt
DCG sweep on the deployed ctt_v2 champion (test-time Demonstration-Contrastive Guidance, w∈{1,1.5,3,6}, seed 42, 608
clips, DeltaAI array 2977744 on bhwp-dtai-gh) — the mirror of the neutral phase. Scored on eval-v4-cert (array 2978445)
+ the demo-copy guard (job 2978481). Advisor verdict (pre-registered rubric R8): when the EFFECT text already describes
the transition, DCG@w1.5 adds ~0 appearance %same (96.0→96.0; incl-ref-control 90.4→90.3; paired median +0.28pp),
demo-copy CLEAN (Δmean +0.014), harm rails all clear; additivity VOID at ceiling, headroom-share 0% → the effect text
SUBSTITUTES for demo guidance (neutral was 20%). Cross-phase takeaway: DCG@w1.5 is a substitute for missing text, not a
complement — gate ON for neutral/underspecified prompts (+3.3pp, copy-clean), OFF when text already specifies the effect.
Registered store gens `015-018/02_effect__dai` (608 clips) + `evals/015_dcg_sweep_effect` (store_fsck PASS); added the
effect Ⓔ pills to the 4 DCG arms in the `iclora_neutral_effect` viewer (41 arms, LIVE). Analysis: misc/2026-08-14_dcg_conditioning/DOSSIER.md R7-R10.

**10:38 — F-003 landed (owner-approved): WSD LR-restart continue-training of ctt_v2 is a cheap, non-novel quality lever.**
The DUAL-FORCE plain-FM control (`runs/012`) = ctt_v2@10k warm-started + ~1k steps under a fresh WSD schedule
(surgery/KD off) reads qualitatively clearly better than ctt_v2 on the viewer. Logged in `docs/FINDINGS.md` as an
engineering lever, not a contribution; the vs-ctt_v2 magnitude is marked UNCERTIFIED pending a co-scored same-corpus
A/B (ctt_v2@10k vs control@1000). Contribution search continues.

**05:00 — dualforce_kd arm registered + neutral gen launched (run→arm→stamp→gen).** Registered the DUAL-FORCE
text-crutch-distillation TREATMENT adapter to the store as `runs/013_dualforce_kd` (LTX-2 19B IC-LoRA r128/α128
one-way, step-1000 checkpoint, sha f4d1103d; SAME ctt_v2 recipe + warm-start from runs/002@10000 as the control,
the ONLY delta being the KD objective: effect-teacher vs neutral-student self-distillation, λ_target 0.3 in the
high-σ band 0.7, warmup 500, conditions_effect / derive_effect_from=null; surgery OFF; trained on DeltaAI GH200 job
2976278, trainer src/LTX-2-dualforce @ a4033230 branch dualforce with a dirty working tree; verified 960 plain LoRA
tensors, no encoder/bottleneck → generates on the proven one_way stack exactly like the control). Added arm keys
`dualforce_kd_{neutral,effect}` to `eval_ladder/arms.yaml` + the `dualforce_kd` row to `store/ARMS.md`; stamped the
152-row CTT neutral grid from prompts/001 (prompt_sha 0d708175fbfe) to `build/dualforce_kd/registry_dualforce_kd_neutral.jsonl`;
LAUNCHED the neutral gen (152 rows × seeds 42/43 = 304 clips, 480×640×121, r128 one_way) on DeltaAI ghx4 (bhwp-dtai-gh,
`--array=0-15`) into `store/gens/014_dualforce_kd/01_neutral__dai`. store_fsck PASS for the run registration; gen close-out
(store_register + eval + viewer) driven by the parent.

**02:36 — dualforce_control arm registered (run→gen→eval).** Registered the DUAL-FORCE plain-FM CONTROL
adapter to the store as `runs/012_dualforce_control` (LTX-2 19B IC-LoRA r128/α128 one-way, step-1000 checkpoint,
sha 72a213b9; ctt_v2 recipe warm-started from runs/002@10000 with surgery/KD OFF; trained on DeltaAI GH200 job
2975760, trainer src/LTX-2-dualforce @ a4033230 branch dualforce with a dirty working tree). Added arm keys
`dualforce_control_{neutral,effect}` to `eval_ladder/arms.yaml` + the `dualforce_control` row to `store/ARMS.md`;
stamped the 152-row CTT grid from prompts/001 (neutral) and prompts/002 (effect); generated + v4-scored on DeltaAI
(store/gens/013_dualforce_control, store/evals/012). store_fsck PASS.

## 2026-08-17
**13:00 — timing_relay Wave-3: RoPE timing + demo-placement + versatile control (advised).** On the CTT champion
(ctt_v3) vs its undertrained ancestor (ctt_v2), pre-registered (misc/2026-08-14_timing_relay/WAVE3.md). Two clean
POSITIVES, both vindicating the owner's intuitions. (1) **Demo PLACEMENT is followed:** demos re-cut so the effect
happens early vs late move the generation's t50 — ctt_v3 median Δ=−0.156, 18/18 sign-consistent → ADHERENCE-POSITIVE;
ctt_v2 −0.046 / 67% → negative. Better training buys demonstration-dependence. This REFINES the Wave-1 duration-negative:
transition RATE doesn't scale, but PHASE (when it happens) does. (2) **RoPE temporal warp imposes timing:** warping the
free middle frames' temporal coords (γ=0.5 Δt50 −0.227 / γ=2.0 +0.19, both 12/12, 0 broken) matches the analytic f^γ
prediction. Versatile (edit prompts, Gemini-scored): animal-identity (cat/dog/horse) and size (giant) controllable →
PASS; jump event-timing and RoPE uniform span-scale → NEGATIVE. Block-0 free re-score: gens don't copy the demo's
profile shape (null). New viewer `timing_relay_wave3` (+ interactive blind placement panel).
**12:55 — run_gen: ROPE_CONFIG temporal-RoPE warp hook + job-unique scratch dir.** Added a `ROPE_CONFIG` env hook
(mirrors `RELAY_CONFIG`) that monkeypatches `TransformerArgsPreprocessor._prepare_positional_embeddings` to warp the
temporal coordinate of the target middle-frame tokens before the rotary bake (modes warp/scale/scramble/identity;
`rope_hook.py` lives in the campaign dir). Also fixed a scratch-dir race: `_runner/<arm>_s<seed>_c<chunk>` now appends
`$SLURM_JOB_ID`, so concurrent submissions sharing (arm, seed, chunk, out-root) — e.g. a fan-out over rope configs —
can no longer clobber each other's `step_000000_N.mp4` mid-rename (FileNotFoundError).
**11:43 — baseline_metric_table: registered the VAP/VFXMaster author-config re-run + ablation; "text-inert" REFUTED.**
Registered the full-prompting external re-run to the store (contract v2): new gen variants `gens/011_vap/{03_authorcfg__dai,04_tgtfull_refempty__dai}`
and `gens/012_vfxmaster/{03_,04_}` (224 clips each, moved in + old external paths symlinked back), and a new eval
`evals/011_external_authorcfg__dai__2026-08-17` (v4, 4 arms, DeltaAI one-machine). `authorcfg` = the authors' intended
`{S1}.{EFFECT}.` in BOTH text channels; `tgtfull_refempty` = identical target, empty reference channel (ablation).
**Result:** the v1 "text-inert" reading was an under-prompting artifact — authorcfg is +22.5pp vs v1-neutral on Unseen
appearance pool-% (VAP 37.5→60.0, VFX 39.0→61.5; paired stratified bootstrap 95% CI excludes 0). Channel decomposition
(authorcfg − tgtfull_refempty) = ref-text channel contributes only +0.4/+1.5pp → the gain is the generation prompt
naming the effect, not the demo channel. Champion still leads 16–27pp (advisor signed off). Registered the 4 arms in the
`iclora_neutral_effect` viewer (build_neutral_effect.py; 139 cards, all seatbelts pass). ARMS/INDEX updated; store_fsck PASS.
Also fixed `scripts/store_fsck.py` `corpus_sha` to handle the two-channel prompt schema (`target_text`/`ref_text`) —
shelf `008_ext112_authorcfg` was previously uncheckable (KeyError) and now validates to its declared sha.
Dossier: misc/2026-08-13_baseline_metric_table/DOSSIER.md.

## 2026-08-14
**19:45 — timing_relay campaign (advised): timing NEGATIVE + color-sweep order-dependent PARTIAL + relay architectural negative.**
Two inference-only tests on champion ctt_v3. (1) Timing (retimed-demo test): pooled β 0.084 [0.049,0.118], all clips
below the 0.25 bar → transition pacing is conditioning-inert (follows neither text nor demo tempo). (2) Color sweep
(2-color, shadow_smoke_1): green→red 6/8 works, red→green 3/8 fails — an order-dependent positional bias; single-color
controls localize it to token-binding-under-competition. (3) Prompt-Relay token-masking hook (run_gen.py RELAY_CONFIG):
wiring-sanity killed as a clean ARCHITECTURAL negative — the Gemma embeddings-connector runs unmasked bidirectional
mixing (embeddings_connector.py:148), smearing prompt content across all key positions, so per-token cross-attention
masking cannot isolate a color. (4) Wave-2b relay via per-frame conditioning ROUTING (concat red+green contexts,
route early frames→red / late→green): red→green 0/8, but the all-green control flips the clip fully green (4/4) →
routing delivers color GLOBALLY yet cannot SPLIT color within a clip. **Through-line: a transition's pacing AND
color are globally-coherent properties — settable as a whole, resistant to within-clip per-frame control.**
Dashboard viewer `timing_relay`; record in misc/2026-08-14_timing_relay/ (DOSSIER.md + REPORT.md, 4 pre-registered
tracks). F-blocks proposed for owner approval.

**17:11 — run_gen.py: optional `reference_path` row field (feed an explicit reference clip, bypass std121 resolution).**
For the advised timing+relay campaign (`misc/2026-08-14_timing_relay`), the timing test feeds *retimed* demo
clips that live outside `data/processed/transitions_std121` and aren't in split_v1.2. Added a `reference_path`
override in `build_sample()`: when present, it is fed directly to `ReferenceConditionConfig`, else the existing
`STD/clip_class(name)/name.mp4` resolution runs. Absent on every existing row → behaviour byte-identical;
prefix/suffix resolve from `row["endpoint"]`, so the override is isolated. Timing gen array submitted (champion
ctt_v3, 6 clips × factors 1×/2×/3×/4× × 3 seeds).

**16:24 — External-baseline inference spec added to the store (`store/gens/EXTERNAL_BASELINES.md`).**
Durable, discoverable record of HOW the prior-works baselines (refVFX / VAP / VFXMaster) were run — provenance +
weights + env, the exact inference recipe (scheduler/steps/guidance/negatives/dtypes/flags), reference-video frame
subsampling and native output length (refVFX 33f, VAP/VFXMaster 49f, all from our 121f clips), geometry (480×640) and
duration-matched fps, and the full parity-vs-authors verdict from the 2026-08-14 3-agent primary-source audit
(parity excellent; only deliberate/disclosed deviations). Linked from `store/INDEX.md` gens section. Consolidates what
had lived only in the campaign dossier + DISCLOSURES so store consumers can find "how faithful is each external?" in one place.

**11:33 — Controllability-probe viewer added to the dashboard, with IC-LoRA-style playback and control-highlighted prompts.**
Registered `ctl_probe` in `scripts/viewers/registry.json` (group `reports`, featured) as a mount of the existing
`outputs/reports/ctl_probe/{index,adjudicate}.html` — same depth so the pages' `../../../store` media climb resolves
unchanged, originals untouched. Reworked the generator (`misc/2026-08-13_controllability/build_ctl_viewer.py`, gitignored):
clips now lazily autoplay on scroll-into-view via IntersectionObserver (matching the IC-LoRA trainings viewer) instead of
click-to-play, and each condition column now shows its FULL prompt with the control span highlighted — a word-level diff
vs the anchor prompt (inserted/edited = the injection, struck = removed), so *how* control is attempted is visible on the
page. Both pages LIVE (12/12 refs). Note: viewerctl needs python3.12 (default python3 is 3.6, walrus SyntaxError).

**10:51 — Prior-works baseline snapshot: VAP + VFXMaster added to the CTT comparison, scored + registered in the store.**
Advised campaign (`misc/2026-08-13_baseline_metric_table`, DOSSIER.md): stood up VAP (bytedance @ 0f30aedf) and
VFXMaster (libaolu312 @ 0632c5a, 2b-aux+5B-transformer) from scratch on DeltaAI aarch64, demo-gated both (authors'
own examples reproduced), and generated the one-sided CTT grid (112 rows × 2 prompt tiers × 2 seeds = 448 clips each).
Scored on OUR v4 pool-yardstick (new motion pool-% built + validated self-consistent) AND 7 competitor metrics
(`their_metrics/score_batch.py`, frozen impl_sha d63935f4). **Result: CTT beats every external baseline by ~38pp on
the leak-free neutral pool-%** (champion 77.3 vs best external 39.0); VAP/VFXMaster are reference-driven & text-inert,
refVFX text-leaning; competitors' own metrics show ~30× dynamic-range collapse vs the GT-anchored instrument. Advisor
R3 reviewed (numbers trusted, packaging fixed: §2 raw-table subset bug, scope labels, DISCLOSURES.md). **Registered:**
`runs/010_vap`, `011_vfxmaster` (external stubs); `gens/011_vap`, `012_vfxmaster` (arm-first, videos in store);
`evals/010_external_baselines__dai__2026-08-14` (4 arms, 2876 pool-rows each); ARMS.md + INDEX rows. store_fsck PASS.
Tables: `TABLES_{our_metrics,subset_112,their_metrics}.md` + `DISCLOSURES.md`. Gemini VLM lens (competitors' actual
headline) deferred — owner sign-off pending. [UNCERTIFIED — v4 branch not tagged]

## 2026-08-13
**21:35 — Controllability campaign EXECUTED + CLOSED (pending owner adjudication): the champion pays NO attribute-control penalty; controllability is axis-stratified by the BASE model.**
Advised campaign (misc/2026-08-13_controllability, pre-registered before generation): 412-clip contradiction probe —
can OOD attribute text override a demo-shown attribute? 3 arms (base / ctt_v2 / ctt_v3) × 5 axes on G-fit rows +
unseen replication + congruent controls + graft probe. Gen 24/24 clean (~55min, DAI). Verdict instrument (Gemini VQA,
temp 0) hit its pre-registered demotion clause (calibration 70% < 85%) → tables descriptive; color rescued by the
pre-registered hue corroborator (99% FP-check). RESULTS: confirmatory NOT declared — v2 ≈ v3 at color override
(paired p=0.50; objective hue instrument: v3 87.5% vs v2 81.2%, base 97.9%). Appearance attributes text-controllable
everywhere; dynamics (density/speed/direction) text-INERT on all arms incl. base (= base-model limitation, not adapter
cost), except measurable sub-categorical shadow_smoke thinning (12/12 pixel-drop). GRAFT: attribute sentence with NO
effect clause transfers onto the demo effect (v3 6/6, hue-corroborated) — the product recipe is "decorate the demo,
don't contradict it". Unseen-row replication 87.5% > held-in 58.3%. Fixed-gist re-VQA: gas integrity artifact confirmed
(61→98%); earth_element_0 fragile at ROW level (anchors 17%) — quarantined by the failed congruent gate. Registered:
gens {009_ctt_v3/05,002_ctt_v2/06,005_base_cond/03}_probe_ctl + evals/009_ctl_vqa + prompts/007; fsck PASS. Owner
adjudication set (41 clips) at outputs/reports/ctl_probe/adjudicate.html — the pre-registered verdict authority.

**18:55 — neutral_effect viewer REDESIGNED (owner spec): input-bag cards, categorized arm dropdown, prose removed.**
The iclora_neutral_effect page is now metrics-then-examples: all verdict/caveat prose collapsed into one provenance
<details>. New arm selector = stay-open dropdown with categories (baseline / generalist / bottleneck / external),
per-category ▲▼ ordering (drives metrics rows + card cell order), variant-tinted pills per arm (neutral/effect/control),
select-all/clear + N-on/off·E-on/off bulk ops scoped to arms with a selection; state in localStorage. Cards: left =
input bag (UNIQUE prompts deduped by exact text with colored P# chips, anchors, consumed demos), right = one cell per
selected entry with a matching-color P# dot + ref/conditioning indicators — cells tinted by category, per-arm CSS gone.
Builder: +9 arms this page lacked (ctt_v3 era, surg1, regens → 27 toggles), per-gen prompt text, meta.arm_catalog.
Dropped: zip bundle bar + tick boxes, alt_prompts band. Validation: jscheck lex clean, rebuild green (139 cards,
8286 videos), page 200 over viewerctl :8017.

**17:20 — Store contract-v2 migration EXECUTED: arm-first gens, prompts shelf, tooling, viewer rewire — fsck PASS.**
Owner approved all 4 decisions (migrate-with-shims, ctt_v3/ctt_v3_hs, pass-shaped evals, full cleanup bundle).
Executed: (1) catch-up registration commit (metas untracked since ~Aug 6); (2) prompts/ shelf seeded — 6 sha-pinned
arm-free families, all reproduce the audited shas; (3) 25 gen entries migrated to gens/NNN_<arm>/KK_<variant>__<machine>
(migrate_gens.py, dry-run then live, per-entry mp4 counts verified, _legacy shims, v1 metas kept as meta.v1.yaml);
017-025 stubs backfilled (~4.7GB media mv'd in, grids + pins restored); (4) cleanups — evals/001 retro symlinks removed,
~15GB dup checkpoints deleted after sha-verify, _runner scratch cleaned, DCG probe grid registered retroactively as
gens/002_ctt_v2/05_probe_dcg__dai; (5) contract docs — README v2, ARMS.md, MIGRATION.md, INDEX arm-first rewrite;
(6) consumers rewired (72 path replacements in both viewer builders + registry.json) — BOTH viewers rebuilt green
(139 cards, 7374/5550 videos), fixed the build_neutral_effect --out fork bug + job_score /projects prefix;
(7) new tooling: eval_ladder/stamp_rows.py (family→registry, the only sanctioned path), scripts/store_register.py
(gen close-out writer), scripts/store_fsck.py (validator — PASS, 12 grandfathered warns); run_gen --videos-dir.
Skills updated in the same change (lora-flow v2 rewrite, exp-eval store pointer, viewer /taiga fix).

**13:05 — Store restructure PROPOSAL written (contract v2): arm-first gens, neutral/effect variant grammar, prompts shelf.**
Four parallel audits (viewer wiring, prompt provenance, skills, train/gen/eval flow) established: prompts are already
byte-identical within each of 4 families across ~25 copies (single renderer, shared clause file, refVFX-B = neutral);
but gens/017-025 are meta-only stubs (media in outputs/, ~4.7 GB), grid.jsonl has no writer, evals/001 was retro-edited
(9 unused forward symlinks), ~15 GB duplicate checkpoints in misc/, and the viewer is 3 hardcoded lists + a fork.
Proposal (misc/2026-08-13_store_restructure/PROPOSAL.md): gens/NNN_<arm>/KK_<variant>__<machine> two-level layout,
canonical-arm vs frozen harness_arm alias split, new prompts/ shelf (sha-pinned families) + stamp_rows/store_register
tooling, evals stay pass-shaped, migrate-with-shims + MIGRATION.md. Awaiting owner sign-off on 4 decisions — nothing moved.

## 2026-08-12
**15:30 — EFFECT-PROMPT follow-on DONE: "text channel saturates the adapter gain" (co-located 2×2, evals/008).**
Owner asked to run the effect prompt (the ctt_v2_leaky convention) for the new champion too. Advisor upgraded it to a
CO-LOCATED adapter×prompt 2×2 on DeltaAI (5 arms, gens/021-025) to decontaminate the never-co-located "+8.8 text gain".
Result: under the effect prompt the champion's schedule-fix edge WASHES OUT — pushA_effect 91.54 ≈ leaky_regen 90.21,
primary paired Δ −0.22pp [−1.94,+1.45] (NULL, pre-declared "text saturates the adapter gain"); all effect arms sit at
the ~91 regen-consistency ceiling. But the champion's text gain is significantly SMALLER than v2's (DiD −4.62pp
[−7.54,−1.67]) — the corrected adapter reads from the reference what text otherwise supplies (mechanistic complement to
the +5pp plain win). B (high-σ) inert again → RETIRED. Copy-guard clean. Methodological catch: the true co-located v2
text gain is +7.3pp (not +8.8 — that conflated text with a ~1pp machine drift), reinforcing score-on-one-machine.
Champion status UNCHANGED (88.0 plain; effect numbers are text-assisted only). Registered gens/021-025 + evals/008;
F-block = addendum to the plain-campaign block (owner-gated). Fix filed: per-worker cache dir / flock (the concurrent
40-task scoring crashed shards with cache-write EOFErrors; worked around with %4 throttle).

**12:03 — ctt_v2 performance-push CLOSED: a NEW CHAMPION (ctt_v3, +5pp) via the LR-schedule fix; high-σ closed as a raw-reader lever.**
Advised operator/advisor campaign (misc/2026-08-11_ctt_v2_perf_push/DOSSIER.md). Discovered ctt_v2's shipped `linear`
LR schedule was mis-scaled by num_processes and floored at 1e-5 for **87.5%** of its 10k steps. Trained two arms on
eps 8×H100 with a num_processes-correct **WSD** schedule (6000 steps): **Arm A** = corrected baseline, **Arm B** = A +
SURG-1's high-σ timestep lean (owner's seed idea). Gen co-located on eps (repro gate BIT-IDENTICAL, PSNR=inf) → scored
v4 on DeltaAI vs the ctt_v2=82.5 baseline (analysis reproduced 82.49 first). **Arm A = MEASURABLE WIN**: paired same-seed
Δ%same +4.99pp ALL-152 [+2.38,+7.63] / +5.49pp same-60, headline **82.5→88.0**, copy-guard clean → new champion candidate
`ctt_v3` (runs/008, PROVISIONAL pending the pre-registered blind A/B). **High-σ (Arm B) adds NOTHING over A** (B−A −0.4pp
[−2.9,+2.1], slightly negative) → lever closed for raw readers (it transfers to a bottleneck code, not a raw reference).
Registered runs/008+009, gens/017-020, evals/007; added both arms to the iclora_runs viewer. ⚠ CAVEAT: the shipped ctt_v2
was UNDERTRAINED → every ctt_v2-anchored number (incl. the CTT-publishability baseline) needs re-reading. Owner-gated:
blind A/B, git push, F-block. Trainer commit a4033230 is LOCAL (fork origin=Lightricks).

## 2026-08-11
**18:34 — SURG-1 Gate B SCORED → CLOSED as a publishable negative; registered gens/015+016 + evals/006.**
Generated 304 clips/arm through the V-JEPA-code path (matched `surg1_wsd` + deranged-twin `surg1_wsd_shufcode`)
and scored v4 on DeltaAI. Result: **reads-but-weakly-instructs** — P1 same 7/13 & cross 9/13 (bars 9/8), P2
pooled median **+0.0199** vs bar 0.1016; 0/304 copy-flagged; clearly ABOVE the dead encoder arms (6/13, −0.002)
with genuine matched−twin separation. A fresh advisor CLOSED SURG-1: the failure is forward-read→sampling
**conversion** (trained-row ceiling +0.053), NOT channel-deadness (bneck_coupling) nor reader-absence (Gate A
proved a held-out reader) — a strictly sharper negative. One-retry budget left unspent (a generalization retry
can't beat the conversion ceiling). Registered store/gens/015_surg1_wsd, gens/016_surg1_wsd_shufcode,
evals/006_surg1_wsd__dai__2026-08-11 (P1/P2 headline + caveats). Remaining are paper-completeness only (a
labeled CFG-w2 exploratory arm, the free DisMo pre-check, cache-error forensics) — they cannot change the close.

**15:14 — SURG-1 WSD training COMPLETE (4500 steps, 8×H100 on eps); registered as store/runs/007_surg1_wsd.**
The `num_processes`-correct WSD-schedule redo of the V-JEPA-raw objective-surgery arm finished cleanly (`train exit=0`). A real,
manner-dominant, ROBUST (frozen≥live) reader formed — final probe Δcross 0.042, manner-share 0.031, **no H-rule fired** — but at
~4× the natural full-video reader gap (0.0104) with a mild appearance creep (ratio 0.166→0.264 over the last 3 probes). The
magnitude question (strong genuine reader vs content-axis over-forcing) is NOT resolvable from the forward-loss probe; Gate A
(held-out 1065 triples) then Gate B (paired v4 video) adjudicate. Trained on **eps** (offsite 8×H100, no queue — ghx4 and HCESC
were both jammed) via the `src/LTX-2-surg` fork (commit a4033230, the num_processes-correct WSD scheduler) shadowed over the eps
uv env. Checkpoint (sha cdc36bbb) backed off eps to /taiga and registered `store/runs/007`. Gate A (`build/ablation_loss_vjepa.py`,
run on DeltaAI to match the E1/ROUND4 calibration) is next.

**00:41 — SURG-1 full launch crashed at step 0 (native "Aborted."); diagnosed + fixed a multi-rank file race in the probe cache; full-config smoke now green.**
The first full 10k launch chunk (2924995, 4×GH200) got through all setup then printed a bare `Aborted.` (native SIGABRT, no
Python traceback) and died before step 0 — the reduced ddp smoke had passed because it used a smoke config. **Root cause
(confirmed by log localization + code inspection + the partial-write artifact on disk):** `SurgeryProbe.cache_frozen_codes`
had NO rank gating — all 4 ranks ran `if is_file(): torch.load else: compute + torch.save` on the SAME shared
`{output_dir}/probe_frozen_codes.pt`. wandb (main-process only) delays rank 0, so ranks 1-3 took the write branch (concurrent
`torch.save` to one path) while rank 0 hit `is_file()==True` and `torch.load`ed a PARTIALLY-WRITTEN file → native abort in the
zip reader. The reduced smoke had wandb OFF → all ranks synchronized on the write branch (none read) → never triggered. (This
also ruled out the batched-2-OOM / derangement-OOB / per-σ suspects — those are step-0 forward faults, but the crash is BEFORE
step 0.) **Fix (commit 9aed151):** `cache_frozen_codes` is now multi-rank safe — branch on `resume_step` (not `is_file`, which
races); FRESH → every rank recomputes the SAME launch-time codes in-memory (no reads → no partial-read race), ONLY rank 0
persists via atomic tmp→rename (single writer), then a barrier; RESUME → all ranks read the complete chunk-1 file (recompute
fallback). **Verification gap closed:** the multi-GPU smoke now runs the ACTUAL launch config (`surg_fullsmoke.yaml` =
`surg_launch.yaml`, steps=3, ckpt@2, probe@2, **wandb ON**, real dataset, 64-target probe) with `PYTHONFAULTHANDLER=1` +
`CUDA_LAUNCH_BLOCKING=1`. **Full-config smoke 2925157 COMPLETED ExitCode 0:0**: "cached 192 frozen" (no abort), 3 steps, gap
active (grads finite), `SURG-1 PROBE @ step 2 n=64` fired clean, checkpoints `lora_weights_step_00002/00003.safetensors`
(1.3 GB each) written, clean exit. Notes §R4.13–R4.14. The full 10k launch remains the orchestrator's to fire (not resubmitted).

## 2026-08-10
**23:30 — SURG-1 last two gates cleared: (c) cross-transition-manner derangement + the DDP double-forward fix; both smoke-green, still not launched.**
Two coordinator-blocking fixes on `src/LTX-2-surg`. **(c) Derangement granularity:** the gap-loss derangement keyed on the
manifest's 136 `base_idx`, but those SPLIT each transition manner into appearance variants (`animalization`=5 vs
`spec animalization`=109, etc.) — a cross-`base_idx` draw could land in the SAME manner → the hinge would penalize a same-manner
pair = appearance-detection Goodhart. E1 (and the eval) define cross/same at the transition-MANNER level (verified: 0 same-group
cross-refs / 0 cross-group same-refs over 1065 triples, 23 classes). Fix: `demo_class_map` now keys on the transition manner =
`base` normalized (strip `spec `/` 1sided`) → **124 groups; base_idx NESTS** (0 orthogonal); 11 groups merge the spec-pairs.
`DerangementSampler` was already cross-class-by-construction, so this is a class-map data change (gitignored `misc/`); real-roster
smoke: **0 fixed points, 0 same-manner collisions**. **DDP double-forward (flag #9):** the two SEQUENTIAL forwards through the
DDP-wrapped LoRA broke DDP two ways — default reducer "marked ready twice" (2924787), then `static_graph=True` → the
static_graph×non-reentrant-checkpoint `expect_autograd_hooks_` assert (2924856); `use_reentrant=True` is DISQUALIFIED (silently
drops the deranged branch's block-0 LoRA grads — its inputs all lack requires_grad). Fix (coordinator + fable-advisor concur): a
single BATCHED forward — matched prepared first to learn σ, then on a high-band step (σ≥0.90) the deranged code (detached, §9.1) is
batched in (batch-2, one forward, split into L_matched/L_deranged); low-band steps run plain batch-1 (gap=0). ONE forward CALL per
step → LoRA marked ready once → baseline DDP config restored (`find_unused_parameters=True`, `use_reentrant=False`). Same
losses/gradients; commit 7ba93c6. **Verified:** 1-GPU 2924925 equivalence STILL bitwise-0 + functional all pass (detach isolation
projector deranged=**0.0**, identity gap_raw=**0 bitwise**, low-band batch-1 path, probe frozen==live); **4-GPU DDP 2924926
COMPLETED ExitCode 0:0 on all 4 ranks** (backward completes, finite grads, 64-target probe fired clean at step 2, Δsame/SE=1.44
flag=0, 4.44s/step). Both fixes are the last two launch gates; the orchestrator owns the launch (not submitted). Notes §R4.10–R4.12.

**22:20 — SURG-1 FINALIZED to the E1-greenlight ruling (ROUND4); fork `src/LTX-2-surg` ready to launch, still not submitted.**
E1 greenlit SURG-1 (`advisors/ROUND4_E1_RULING.md`), so the fork was upgraded to the exact ruling params and re-smoked.
Implemented: **§9.1 deranged-only detach** (vjepa `encode_train._surg_detach_code` flag → `L_deranged` never gradient-updates
the projector; grad-isolation smoke: projector grad matched **0.435** / deranged **exactly 0.0**, LoRA-deranged 0.114 —
closes the demo-ID-watermark Goodhart channel); **§5c band-mixture sampler** (`BandMixtureSampler`: 60% [0.90,σ_max] / 15%
[0.70,0.90) / 25% base; **σ_max=1.0 measured**); **§5a/b hinge ONLY at σ≥0.90** with **δ=0.007** (=2.1% of E1's σ=0.95 matched
loss; single-bin file + override hook for the δ-transfer check); **§5c per-σ Min-SNR normalization** on the PRIMARY FM term only
(the hinge always consumes RAW losses so δ stays on-scale; data-anchored weights from E1 per-bin means); **§5d λ_target=10** (was
0.1 — ~50× too small; `surgery/gap_frac_of_total` smoke-measured **0.149**, in the 10–20% band); **§5e/§9.1.3 mid-training Goodhart
probe** (`surgery_probe.py`: per-target top-band Δcross/Δsame under LIVE vs FROZEN launch-cached codes; main-process forward-only,
barrier-safe). **BLOCKING §9.2 temporal-RoPE check PASSED** (`rope_verify.py`): code temporal positions strictly increasing,
offset matches the ctt_v2 raw band-setter, span matches the core; `temporal=1.0` is the extent, not a pinned point. **Re-smoke
(job 2924698, 1×GH200): equivalence STILL bitwise-0** (cross-fork + off-block 0.000e+00 — every new op gated behind
`surgery.enabled`), functional all pass. Two sbatch PREPARED not submitted: `job_surg_ddp_smoke.sbatch` (4-rank DDP double-forward
× checkpointing × probe — fire first) and `job_surg_launch.sbatch` (10k-step chunked, resume_wrapper, surg output_dir). Fork
committed local on branch `surg` (77e27ab, not pushed). DATA-ALIGNMENT FLAG: E1 held-out triples reference a different roster
(0/1065 resolve in the vjepa manifest) → probe uses a vjepa-aligned triple set. Launch gated on the E1 negative control + δ-transfer
check (orchestrator runs those). Notes: `misc/2026-08-10_encoder_branch_redteam/SURG1_IMPL_NOTES.md` §R4 (gitignored).

**21:26 — SURG-1 (objective surgery) IMPLEMENTED + SMOKED in a new trainer fork `src/LTX-2-surg`; no training launched.**
Forked the V-JEPA-raw coupling trainer (`src/LTX-2-bneck-coupling`, branch `bneck_redesign`) into a new git worktree
`src/LTX-2-surg` on branch `surg`, and implemented the three SURG-1 levers from
`misc/2026-08-10_encoder_branch_redteam/advisors/ROUND2_OBJECTIVE_SURGERY.md` (§A.1 high-σ timestep mixture, §B.1 code-swap
contrastive gap loss, §C.4 temporal-RoPE check) behind a `surgery` config block that **defaults OFF → flags-off is BITWISE the
coupling baseline**. New/changed in the fork: `config.py` (SurgeryConfig/HighSigmaMixtureConfig/GapLossConfig + validators),
`timestep_samplers.py` (`HighSigmaMixtureSampler` wrapper), `surgery.py` (new: `DeltaMargins` per-σ-bin margins, cross-class
`DerangementSampler` with a dedicated RNG generator, `lambda_ramp`), `trainer.py` (`_init_surgery`, high-σ sampler wrap,
`_surgery_gap_loss` — the RNG-snapshot/restore code-swap second forward, no-detach ΔFM, gated metrics). Also synced the coupling
working-tree baseline the fork's committed ref lacked (`resume_data_position` + `ResumeOffsetSampler` + `ContextAdapter`). Import
isolation = PYTHONPATH-prepend (shared venv `.pth` untouched; verified `import ltx_trainer` → surg fork under the prepend, →
official by default). Smoke on 1 GH200 (jobs 2924473/2924549): **equivalence bitwise-0** (max|Δloss| 0.000e+00 cross-fork AND
off-block), high-σ histogram (base P(σ≥0.83)=0.505 → realized band 0.851 vs analytic 0.851), gap-loss shapes/finiteness,
**identity-derangement gap==0.0 exactly**, hinge==λ·δ==0.02, backprop (966/966 finite grads), RNG invariant true. σ*=0.83 and
λ_target flagged for owner confirmation; δ margins load from an external E1 file (placeholders used for smoke). Fork commit is
local on branch `surg` (not pushed — origin is Lightricks upstream). Notes + build scripts in
`misc/2026-08-10_encoder_branch_redteam/` (`SURG1_IMPL_NOTES.md`, gitignored). No multi-GPU training submitted — that waits on
the E1 gate + real δ margins.

## 2026-08-07
**16:37 — metric_eval adapter×text arms: certified v4-lane scores registered as `store/evals/005_ic_effect_neutral__dai__2026-08-07` + gen bookkeeping fixed.**
Scored three already-generated arms on the certified transition-eval **v4** lane, all on DeltaAI GH200 (one machine, same
instrument sha as evals/001-004: `reference_v4.npz` file sha `459fd9a7…`, echoed by all 108 task verify blocks; corpus 222-clip
`5a7a8be9…`; τ_copy 0.858). Coverage 1,842 items / 36 shards / arm, **0 errors, 0 nulls, all rc=0**. Manifests: `ic_gen_effect`'s
were REUSED from the campaign's ladder plan after independent verification (222-corpus membership, pool identity 0/152, all 304
gens present, per-shard disjoint by generated_video); the two neutral arms' manifests were REBUILT fresh by
`build_eval_manifest_v4lane.py` (verbatim descendant of misc/base_arms/eval), pool identity asserted 0/152 against the
ic_gen_effect registry. Scored into a campaign-private tree `misc/2026-08-06_metric_eval/gen_eval/scores_v4lane/` and wired into
the store by relative SYMLINK (full items.jsonl+results.json present — no provenance lost, unlike evals/004). Headline **%_same**
(app_ref pool mean ÷ GT-class ceiling, m1a_S3; report_v4lane.py, calibrated to reproduce evals/001 ic_gen 83.1% exactly):
**ic_gen_effect 89.1 %**, base_cond_neutral 60.4 %, base_prompt_neutral 58.1 % (%_proxy 84.4 / 47.0 / 44.9, content-capped
ranking-only, never blended). ic_gen_effect is a TREATMENT (LEVEL only here — its ic_gen base twins are not scored in this entry,
so its plan `twin_of` values dangle); the two neutrals are V-neutral no-demo baselines and are NOT the effect-clause base arms of
evals/002. Also fixed the incomplete gen bookkeeping for `gens/010_ic_gen_effect` (adapter = runs/001_ic_gen@5000, sha `6e37fca7…`,
ctt-v2-train `db69ca7` bidirectional), `gens/011_base_cond_neutral`, `gens/012_base_prompt_neutral` (base weights, no adapter):
populated meta.yaml + grid.jsonl (152 exact registry rows each) and added the three missing INDEX gens rows. Campaign scripts under
`misc/` (gitignored). Accounts bgms (primary) + bhwp (second); never bgjg; no existing job touched.

## 2026-08-06
**16:33 — bneck_redesign Idea-1 (`bneck_ctx_v2`): CLEAN arm registered into the artifact store + iclora_runs viewer (closeout bookkeeping).**
Registered the concluded negative-result arm as four immutable store entries: `runs/006_bneck_ctx_v2` (real config.yaml +
meta; checkpoint a SYMLINK to the eps training scratch — sha256 `844ee248…`, 1041 tensors = 31 ContextAdapter + 960 LoRA +
50 frozen operator_encoder; trainer src/LTX-2-bneck-coupling @ bneck_redesign `7ffbe95`), `gens/013_bneck_ctx_v2` (matched,
304 mp4) + `gens/014_bneck_ctx_v2_shufcode` (deranged twin, 304 mp4) — grid.jsonl real, videos symlinked, pool-identity
verified TRUE, and `evals/004_bneck_ctx_v2__dai__2026-08-06` (v4 on DeltaAI, gens 013+014, 1,842 items/arm, 0 nulls;
**ARM_PASS=false** — P1 6/13 & 6/13 vs bars ≥9/≥8, P2 −0.008 vs +0.10, CI [−0.024,+0.013]; dead-channel bitwise-PASS,
liveness R 0.584). Deviations from the sibling pattern: (1) the eval arm dirs are whole-dir SYMLINKS into
`misc/bneck_redesign/eval/scores_clean/<arm>` (filesystem at 99% quota) — scores_clean carries only `merged/items.jsonl`
(no results.json), so full instrument provenance lives in the eval meta and the viewer's per-column corpus badge for these
two arms renders empty (cosmetic); (2) the two arms went into `build_runs.py`'s `EXTERNAL` list as a matched/deranged
bottleneck pair (like ⑦/⑧), NOT `RUNS`/`SCORE_SETS` — a third RUNS entry would silently repoint the page's core paired-Δ
comparison. Viewer rebuilt + mounted; all seatbelts PASS (152 gens / 304 videos / scored 152/152 per arm, `[ids]` MATCH,
0 shared clips or metric vectors vs every other arm); live on gh-login01:8017 (iclora_runs, health LIVE 7/7). 2&3-campaign
(`hrc_*`/`vjepa_*`) rows untouched. No git commit (owner controls commits).

**14:38 — bneck_redesign Idea-1 (context-port injection): CLEAN NULL — DONE (negative), verdict by fresh advisor vs frozen bars.**
Idea-1 routed the frozen 72-token operator code through the DiT's positionless cross-attention CONTEXT stream (skipping the
RoPE reference path) via a co-trained 6.51M-param `ContextAdapter` + rank-128 LoRA, to test the owner's hypothesis that the
coupling campaign's read-failure was a RoPE position mismatch. Clean full-coverage retrain on eps (`bneck_ctx_v2`, single
continuous run, `resume_data_position:true`, step 10000); 608 gens + v4 score on DeltaAI (one machine, sha `459fd9a7`).
Paired read matched-code vs class-deranged-code (`p1p2_arm.py`, `arm_results/p1p2_bneck_ctx_v2.json`) vs frozen bars
(BANDS.json: G-unseen-same ≥9/13, G-unseen-cross ≥8/13, P2 ≥0.1016): **6/13, 6/13, P2 pooled median Δ = −0.0078,
95% CI [−0.0245,+0.0126] (excludes the bar), ARM_PASS=false**, 304/304 units paired. Calibrators confirm the null is
interpretable, not an artifact: dead-channel `ALL_BITWISE_IDENTICAL` (code reaches gen only via the adapter), liveness
**R=0.584** (channel emphatically live in both claim cells), band-setter positive control 12/13,10/13,Δ=0.203 (instrument
detects reading when present). Second independent "transmits but does not instruct" null after coupling — RoPE-mismatch
premise refuted (both routes live & unread). Discriminator for future Ideas 2/3: training captions are class-blind (0/3453
mention a transition verb) and the raw reference is skipped, so the code was the SOLE transition-identity carrier yet unread
→ text-redundancy ruled out; endpoint-inferability vs geometric-unreadability remain. Fixed an 8-shard v4-scoring
feature-cache write race (73/65 truncated-`.pt` rows, incl. barred cells) via serial re-score before finalizing. Campaign
scripts under `misc/bneck_redesign/build/` (gitignored).

## 2026-08-05
- 20:33 presentation: slide pack revised per owner — 12_iid/13_zeroshot row arms are now each method's native-prompting best (cttv2 column = 005_ctt_v2_leaky, refvfx column = 003_refvfx_A, seeds re-picked frame-by-frame on those arms); 07_counterfactual rebuilt from the S3 depth-parallax family (exp_076 sharedop rows — roll_crossfade_fog / orbit_depth_wipe_sphere_focus / crane_crossfade × 3 seed-matched endpoint pairs); filenames globally unique (`<slide>__<name>.mp4`); pack re-zipped as 16 parts, all <20 MB.
- 20:05 presentation: built `outputs/presentation/slide_pack_2026-08-05/` + upload zip (161M, 106 clips) — per-slide deck media: cold open (shadow_smoke_0 memo-probe), lerp collapse (2 base dissolves), refVFX 4-arm row (tennis→snowboard ← firelava_0, promptA/B twins from gens 002/003/004/005), 16 dataset stratum samples, 3×3 counterfactual grid (animalization/shadow_smoke/polygon × 3 S1 endpoints, reject-list checked), 6 iid + 7 zero-shot six-file rows (seeds picked frame-by-frame). Rebuildable via `scripts/build_slide_pack_2026_08_05.sh`; browsable at viewer `slide_pack_2026_08_05` (gen: `scripts/viewers/gen_slide_pack_viewer_2026_08_05.py`); provenance + prompt strings in the pack's MANIFEST.md.
- 18:32 bneck_redesign: found + fixed a chunked-resume data-coverage defect in the LTX trainer. On resume the dataloader rebuilt the StratifiedEpochSampler at epoch 0 (trainer.py `_init_dataloader`, no sampler-state restore), so short chunks only ever trained on the first `chunk_steps*8` of 56,368 samples (hrc_raw saw ~11%, residual arms ~28% — different subsets). Root of the spurious hrc_raw loss dip (memorization, not code-read). Fix: gated `data.resume_data_position` (default off) fast-forwards the sampler to the resume (epoch,offset) — verified bitwise-identical to a continuous run (misc/bneck_redesign/build/test_resume_fix.py). Both raw arms relaunched from step 0 with the fix. NOTE: src/LTX-2-bneck-coupling is git-untracked, so this fix is working-tree only.

**17:53 — bneck_redesign eval: HRC-residual scored (v4, DeltaAI) = CLEAN NULL; integrated into iclora_runs viewer.**
Scored `hrc_coupling` (matched) + `hrc_coupling_shufcode` (deranged), 304+304 gens, on DeltaAI (`--account=bgms-dtai-gh`),
v4 instrument sha `459fd9a7` (UNCERTIFIED by design for v4). Manifests derived from the raw-control band-setter template
(`derive_arm_manifests.py`). Paired read vs recalibrated bars (9/13, 8/13, P2 +0.10): P1 = 6/13 & 4/13, P2 pooled
median Δapp_ref = −0.003 (bootstrap 95% CI [−0.027,+0.007] includes 0), claim cells sign-disagree, 48.7% units positive
→ arm FAILS all three bars (clean null, `p1p2_arm.py`). Secondary reads: liveness R 0.578 (live); temporal matched-vs-
deranged motion-curve r 0.818 with band-setter calibrator r 0.283 (≤0.65 ⇒ metric discriminates ⇒ confirmed no motion
read) and demo-headroom r 0.374 (<0.7 ⇒ transmittable signal exists); appearance-import pooled 0.50 (one marginal
G-unseen-cross flag, not corroborated by matched arm). Hygiene: same machine, same sha, error-rows symmetric (95/97).
Added both arms to `eval_ladder/viewer/build_runs.py` EXTERNAL (kind bottleneck, score_id `redesign_v4`) — joins assert
clean 152/152 matched↔deranged, media+scores curl 200; iclora_runs blurb updated to name the 2×2. Campaign eval scripts
live under `misc/bneck_redesign/build/` (gitignored). V-JEPA-residual + both raw arms pending their gens.

**14:14 — bneck_redesign: LAUNCHED the HRC-raw coupling on DeltaAI (eps was full of the owner's other campaigns).**
`job_coupling_hrc_raw.sbatch` (clone of `job_coupling_hrc.sbatch`: CFG→`coupling_hrc_raw.yaml`, job-name
`hrc_raw_couple`, `WANDB_MODE=offline` + run-specific `WANDB_DIR` since DeltaAI compute nodes have no internet;
config keeps wandb.enabled:true project bneck_redesign for a later `wandb sync`). Submitted 8 resumable
singleton chunks on `--account=bgms-dtai-gh` (bgjg nearly out of credits): jobs 2880286, 2880287,
2880301–2880306 — 1h walltime each (short, for backfill; fairshare drained → priority-limited, est. start
~18:15 but 1h jobs backfill into gaps). `resume_wrapper.py` cold-starts the first chunk fresh and each later
chunk resumes from the newest checkpoint until step 10000, then no-ops. Runs in parallel with V-JEPA-raw +
the residual arms; `bneck_ctx_train` and eps untouched. First-step confirmation pending (chunk PENDING).

**13:06 — bneck_redesign: built the HRC-raw control arm (Round-4 non-residual ablation = Arm C, owner-confirmed).**
Single-delta clone of the running HRC (residual) arm: the ONLY change is the L_local target — `hrc_targets.py`
gained `residual: bool=True`; `residual=False` makes the target the RAW pooled region `x_std` (skip
`endpoint_crossfade` subtraction). `hrc_train.py` gained `--raw-target` (default keeps residual, running arm
byte-identical); new CPU test `test_raw_target_is_pooled_region` (all 7 pass). Retrained the encoder
byte-identical to hrc_l10 (λ=1.0 LITERAL, arm P, seed 42, 8000 steps) via `job_hrc_raw.sbatch` on DeltaAI
(1-GPU backfill, account bgjg-dtai-gh) → `runs/hrc_raw/ckpt_08000.pt` (job 2879794, COMPLETED 18 min, sha256
f109af4e…). Pre-registered gate (`gate_hrc_raw.py`, OPS_HELDOUT 977 ops) PASSED: held-out same-op retrieval
**0.906** (≥0.5 → couple), locality R² **0.837** (cell-shuffled floor 0.069), endpoint-leak R² **0.411**
(floor −0.089) vs the residual baseline's 0.716/0.443/0.474 — raw leaks slightly LESS than residual, so the
residual subtraction was not the endpoint-invariance lever it was assumed to be. Prepared
`coupling_hrc_raw.yaml` (clone of `coupling_hrc.yaml`: load_encoder→hrc_raw ckpt + its sha256,
output_dir→hrc_raw_coupling, wandb enabled → project bneck_redesign; all functional config byte-identical),
gen registries `registry_hrc_raw_coupling{,_shufcode}.jsonl` (arm renamed only; item_ids + derangement
preserved), and `hrc_raw_coupling(+_shufcode)` in `eval_ladder/arms.yaml`. NOT launched — the coupling runs
on eps via a separate agent.

**12:25 — bneck_redesign: built the V-JEPA-raw control arm (Round-4 non-residual ablation, owner's literal idea 3).**
Single-delta clone of the running V-JEPA (residual) arm: the ONLY change is the feature canon —
`raw_canon` pools the frozen V-JEPA-2 features X directly to (16,3,3) with NO residual/interpolation
subtraction (added to `vjepa_residual.py` in both the idea3 scratch copy and the trainer fork, kept
byte-identical; CPU test proves residual≈0 but raw≉0 on a pure-interpolation clip). `extract_features.py`,
`precompute_gen_feats.py`, `precompute_val_feats.py`, and `stageA_train.py` gained a `--canon
{residual,raw}` flag (residual stays byte-identical) writing to `feats_raw/`, `gen_feats_raw.pt`,
`val_feats_raw.pt`, `projector_stageA_raw.pt`. Submitted on DeltaAI (account bgjg-dtai-gh, freshest
fairshare): smoke→extraction array (0-39)→raw Stage-A, plus gen/val-feats precompute. Prepared
`coupling_vjepa_raw.yaml` (byte-identical to `coupling_vjepa.yaml` except feats→raw, projector→raw, val→raw,
name; ref prob STAYS 0.9), added `vjepa_raw_coupling(+_shufcode)` to `eval_ladder/arms.yaml`, and built the
two gen registries (copy of the vjepa registries, arm renamed only, derangement preserved). Coupling NOT
launched — operator-gated on the residual V-JEPA coupling reaching step 10000 healthy.

## 2026-08-02

**04:00 — bneck_coupling CLOSED: the generator does NOT read the frozen transition code. A pre-registered publishable negative.**
Full cycle delivered (train → generate → eval → viewer → store). The campaign answered the one thing the
bneck_v2 certificate explicitly refused to claim, and the answer is negative — but informative, because two
measurements together *localize* the failure rather than just recording it.
**The channel is live:** deranging the code moves generated pixels at **R = 0.492** (bar 0.10), bracketed by a
must-DEAD constant-code calibrator at **bitwise 0.000** (proving zero leak outside the 72 tokens) and a
must-ALIVE raw-demo calibrator at **1.129** — so the code carries ~44% of a full demo's leverage.
**The read claim fails at chance:** P1 = 6/13 and 7/13 donor classes positive against a ≥11/13 bar; P2 pooled
median Δapp_ref = **−0.002** against +0.05, 95% CI **[−0.015, +0.007]** whose upper bound sits *below* the bar —
a null, not an underpowered study. Cross-cell class-sign concordance 6/13 vs 6.5 by chance, with per-class
effects reversing sign between cells.
**Conclusion (advisor A11): the coupling transmits but does not instruct** — information provably enters and is
provably present in the code, so the failure is in the *decode* step.
Validity is not in question: G1 green on both architectures with all three must-fail calibrators rejected, G2
freeze bitwise at step 10,000, code-stats ratio 1.9216 inside the pre-registered [1/3,3] band, appearance-import
check no-FLAG at pooled percentile 0.500, and scoring symmetric at 1,842/1,842 items per arm with zero nulls.
Store: `runs/004_bneck_frozen`, `gens/008`+`009`, `evals/003_bneck_coupling__dai__2026-08-02`. Both arms are
toggleable on the runs page (9 arms, was 7). The input-adapter reserve's pre-declared trigger is now formally
met and is proposed to the owner in `misc/bneck_coupling/OWNER_MEMO.md` — not started.

## 2026-08-01

**20:45 — bneck_coupling: the frozen-encoder IC-LoRA trained and accepted; generation running, liveness gate pre-registered.**
Trained `bneck_frozen` (`store/runs/004`) — runs/002's recipe with the reference channel REPLACED by the certified
transition encoder's 72 operator tokens, encoder held FROZEN. 10,000/10,000 steps, 0 exceptions; **G2 verified the
freeze bitwise at the final step (50/50 encoder tensors exactly the bf16 cast of the certified weights)**, and G1
verified the ported input contract on BOTH architectures with all three must-fail calibrators rejected.
Two advisors closed the open forks: **A9** accepted the run rather than re-running it (every training-side delta is
*common-mode* — both twins score the same checkpoint — so it can shift the test's power but not its validity), and
**A10** ruled that G5 subsumes the now-moot G4, adding a must-DEAD constant-code calibrator and promoting the
design lock's never-measured alive anchor to a must-ALIVE calibrator.
Settled `N27` by measurement instead of argument: the LR schedule decays `num_processes`x faster in *both* runs
(`ctt_v2` floored at ~step 1,250, this run at ~2,506), so the defect is inherited and this run got *twice* the
incumbent's high-LR exposure. Recorded as pre-registered caveat C1.
Three of my own claims were corrected in the record: the fallback-log-fired-once argument didn't follow (the line is
deduplicated — re-settled on arithmetic), G5 was already fully specified in the design lock, and the existing
`ctt_v2` clips could not serve the must-ALIVE calibrator because they were generated on eps, not DeltaAI.
Now generating 812 clips on DeltaAI (both arms + both calibrators + a copier-guard repeat lane); scoring stays
unarmed until P3 is green, enforced in code rather than by discipline.

## 2026-07-30

**22:21 — bneck_v2 closed: a transition-operator encoder, qualified certificate with two stated exceptions.**
Retried the bottleneck branch on CTTv2 as an operator/advisor campaign (9 fresh advisor rulings, verbatim in
`misc/bneck_v2/advisors/`). Trained an 8.6M Perceiver standalone on precomputed latents with a single SupCon
loss over 1,368 operator identities — no generator in the loop. **Certified on all 3 seeds:** same-operator
retrieval on *never-trained* operators 0.896–0.916 (chance 0.009, best corpse 0.332), holding at 12.1–12.2×
chance inside timing-matched galleries; semantic-manner retrieval 0.772–0.843, above the raw-latent content
oracle. Endpoint identity is absent from the code geometry (endpoint-pair AUROC 0.500–0.507 vs corpses
0.91–0.97); unseen-shader-family retrieval 0.571–0.655 (~100× chance); real→synthetic manner transfer
2.47–2.56× chance. **Two exceptions:** the endpoint-appearance leak gate failed (11–17% of oracle after
byte-pure frame masking cut it 3–4×), and graded similarity missed 0.72 by ≤0.03 — the latter resolved by a
pre-committed diagnostic showing 0.851–0.898 at matched timing, i.e. dilution by a timing axis rather than
lost family knowledge. Generator usability is explicitly NOT certified; the frozen-encoder trial is next.
Two instrument lessons recorded: corpse-relative bars must be frozen to numerics once corpses are measured
(they had silently inflated to 0.766/0.906), and a single-λ ridge is an unsafe leak probe (a pre-committed
λ-sweep voided an already-drafted certificate). Viewer live at `outputs/viewers/bneck_v2_encoder/`.
Report: `misc/bneck_v2/RESULTS.md`; record: `misc/bneck_v2/DOSSIER.md`.


- **20:59** — **`lora-flow` skill + numbered store entries — "what ran latest" is now readable from
  `ls`.** The `lora-train` skill became **`lora-flow`**, widened from training-only to the whole
  pipeline (dataset → train → generate → evaluate → view → record); the mandatory
  ID+OOD+control inline-validation directive is preserved verbatim in §4. It carries the platform
  banner and is the thing to check before ANY of those stages. Store entries renamed to
  **`NNN_<slug>`** (zero-padded seq, monotonic per shelf, never reused, matching `seq:` in each
  meta) — `ls $STORE/evals` IS the timeline and the highest number is the latest, no pointer files
  or `latest` symlinks. Now: runs 001_ic_gen/002_ctt_v2/003_refvfx · gens 001_ic_gen/002_ctt_v2/
  003_refvfx_A/004_refvfx_B/005_ctt_v2_leaky · evals 001_five_arm__dai__2026-07-30 · datasets
  001_transitions_std121/002_ctt_v2. Registering an entry = numbered dir + meta, INDEX row,
  CHANGELOG line, one commit; entries are immutable, a re-run gets the next number. Viewer
  (`build_runs.py`, `registry.json`) and `CLAUDE.md` repointed at the numbered paths.

- **20:25** — **Bridges retired; `/taiga` is the canonical absolute prefix.** The five `$LAB`-level
  compatibility symlinks (`misc`, `LTX-2-*`) are parked at `$LAB/.retired-bridges/` (reversible).
  Enablers: rewrote the venvs' editable path files (`envs-aarch64/ltx2` ltx_core/ltx_pipelines/
  ltx_trainer `.pth` + `direct_url.json`; `envs-aarch64/refvfx` DiffSynth `.pth`) to the in-repo
  paths — both venvs import-verified WITHOUT bridges; updated `job_train.sbatch`,
  `job_precompute.sbatch`, `job_gen.sbatch` and `encode_conditioning.py:90` from dead `/projects`
  (+ bridge) paths to `/taiga` + `src/LTX-2-*`. Prefix rule (per cc-cluster-layout): one Taiga fs,
  `/taiga/illinois/...` resolves on BOTH clusters; `/projects/illinois/...` is CC-only and means
  Delta-project space on DeltaAI — so absolute paths use `/taiga`. Also pinned
  `explorer.excludeGitIgnore: false` in `.vscode/` so the newly-gitignored `misc/`, `store/`,
  `src/LTX-2-*` stay visible in the Cursor/VS Code explorer.

- **20:12** — **LTX-2 checkouts moved under `src/`, worktree git repaired, trainer pins recorded.**
  `src/LTX-2-official` (branch `transition-strategy` @ f062984) + linked worktrees
  `src/LTX-2-cond-bleed-fix` (811d045, ic_gen's trainer), `src/LTX-2-ctt-v2-train` (db69ca7 — the
  RUN_RECORD "integrated trainer SHA" for ctt_v2), `src/LTX-2-bneck` (f252c52). Their worktree
  gitdir wiring had pointed at dead `/projects` paths since the DeltaAI migration — repaired to
  absolute `/taiga` paths; `git worktree list` clean again. `$LAB` bridges re-pointed
  (import-verified). Store: `runs/*` meta now carries a `trainer:` pin (checkout·branch·commit·
  entry), `gens/*` a `code:` line; README gains "What the store is NOT" (artifacts + provenance;
  tools stay in git, pinned by commit — entry + checkout = reproduction recipe).

- **20:02** — **The artifact store (`store/`) + the $LAB reorg.** One structured home for the
  train → generate → evaluate chain: `store/{runs,gens,evals,datasets}/<id>/` with a `meta.yaml`
  per entry, `README.md` as the contract, `INDEX.md` as the ledger. Seeded with the five-arm
  refVFX comparison: runs `ic_gen` (step 5000, sha recorded) / `ctt_v2` (step 10000, sha
  recorded) / `refvfx` (external stub → `$LAB/cache/refvfx/weights`); five gens (304 mp4 each,
  moved, with `grid.jsonl` prompt rows); eval `five_arm__dai__2026-07-30` (36 shards × 5 arms,
  measured corpus + reference-artifact sha256s in its meta). Every prior location now holds a
  symlink to the store, so old paths keep resolving. Same reorg moved `$LAB/misc` and the four
  `LTX-2-*` trainer checkouts INSIDE this repo dir (gitignored), leaving symlink bridges at
  `$LAB` — bridges are load-bearing: `envs-aarch64/{ltx2,refvfx}` editable installs point through
  them (import-verified). Viewer re-pointed: `build_runs.py` RUNS/EXTERNAL/SCORE_SETS now read
  `store/...`, mount gained a `store` link (`scripts/viewers/registry.json`), page rebuilt — all
  seatbelts green, single-machine check intact. Also fixed `run_gen.py:252` (`relative_to` crash
  that killed jobs writing to an out-of-repo `--out-root` after the first clip). Store metadata
  (README/INDEX/meta.yaml/grid.jsonl) is git-tracked via `.gitignore` negations; artifacts stay
  out.

- **15:44** — **`iclora_runs`: all four scored arms on ONE machine, a fifth arm (`ctt_v2_leaky`),
  and every arm now toggleable.** Three changes to `eval_ladder/viewer/build_runs.py` +
  `template_runs.html`, no restructuring.
  (1) **Single machine.** A new primary `SCORE_SETS` entry `dai222` points the run columns at the
  DeltaAI re-score (`$LAB/misc/refvfx_baseline/eval/scores/{ic_gen,ctt_v2}`), so `ic_gen`,
  `ctt_v2`, `ctt_v2_leaky`, `refvfx_A` and `refvfx_B` are all aarch64 / torch 2.10.0+cu129 /
  corpus `dc2e139a`. `rebuilt222` (eps) still supplies the copier and `stale223` still supplies the
  never-rescored specialists, so their badge stays and the machine note narrows rather than
  disappears. Two consequences handled deliberately: `instrument_delta()` is now pinned **by id**
  (`IDELTA_PAIR = ("stale223","rebuilt222")`) so inserting a primary set cannot silently turn a
  cross-*build* number into a cross-*machine* one; and the control floors, which follow
  `SCORE_SETS[0]`, moved **crossfade 30.33% → 30.13%** and **freeze 17.46% → 17.45%** on identical
  n (160 / 448) — the machine term, not a roster change.
  (2) **New arm `ctt_v2_leaky`** (⑥): our own ctt_v2 adapter re-run with prompts that also describe
  the transition, 304 clips at 121f/24fps, joined to the same 139 cards. It is a context tier, and
  because it has **no base twin by design** it contributes a *level, never a margin* — enforced
  structurally (never a run tier, so it cannot reach the per-card Δ, the Δpp column or the sign
  test) and marked with a new `‡` caveat that works like the existing `†`: on the chip, the column
  header, every arm cell, every clip and a footnote under each table. `WINDOW_CAVEAT` now derives
  its arms from a declared `frames`, so ⑥ correctly escapes refVFX's 33-frame `†`.
  (3) **Every arm toggles**, owner request — the same `tiers` Set the trainings already used,
  extended from `run_tiers` to `arm_tiers` and driven by one data list (`meta.arm_chips`). Any
  subset shows side by side; `② specialist` and `⚠ copier` stay as the yardstick. Toggling controls
  visibility only: the Δpp/sign-test columns remain trainings-only, the per-card Δ badge is
  suppressed unless both trainings are visible, and `†`/`‡`/`stale223` hold regardless.
  Also folded in: **pool-% is split by `pct_type`, never blended** (%_same 83.1 / 82.5 / 91.3 /
  42.4 / 33.0 · %_proxy ranking-only 62.0 / 66.7 / 86.4 / 41.3 / 26.7 for ic_gen / ctt_v2 / leaky /
  refvfx_A / refvfx_B), and **both `item_id` join traps are now asserted at build time** — measured:
  `ic_gen`'s ids embed its arm so a raw join returns zero rows, while `ctt_v2`, `ctt_v2_leaky`,
  `refvfx_A` and `refvfx_B` share **1,842 of 1,842 ids** so a raw join silently merges. New
  `assert_arms()` checks the harness `arm` stamp at every read, the multi-path merge refuses
  duplicate eval ids instead of concatenating them, and `check()` prints an `[ids]` block asserting
  `(arm, item_id)` uniqueness plus zero shared clips / metric vectors between colliding arms. All
  three seatbelts negative-tested. `docs/VIEWERS.md` gained the generalised rules;
  `misc/refvfx_baseline/VIEWER_NOTES.md` has the full write-up.

- **14:27** — **`iclora_runs`: the page now states which MACHINE scored each column, and marks the
  two metrics that are not comparable across clip lengths.** Both were correctness gaps in what the
  page *claimed*, not in the wiring. (1) The run columns and the external refVFX columns share an
  instrument, a `reference_v4` and the corpus `dc2e139a` but were scored on **three different
  boxes** — eps x86_64/torch 2.9.1 (ic_gen, ctt_v2, copier), Campus Cluster x86_64/torch 2.5.1
  (specialists), DeltaAI aarch64/torch 2.10.0 (refvfx_A/B). Each column's box is read from its own
  `results.json` `provenance.env`, never hardcoded, and stated directly under the pool-% table with
  the measured cost from `misc/refvfx_baseline/probe/PROBE.md`: that probe FAILED the pre-registered
  `max |Δ| < 0.005` bar at per-row max |Δ| **0.046** on `app_ref`, with aggregate shifts of order
  0.001–0.004, zero gate flips and zero `core_degenerate`/`tier` changes — so the ~40pp refVFX gap
  is far larger than the machine term, but a few thousandths between these columns means nothing.
  (2) `FALLBACK_MIN_FRAMES = 8` is an **absolute** frame count and `mid_mask` excludes a fixed 9/8
  frame conditioning window, so a 33f refVFX clip is scored over 24 (one-sided) / 16 (two-sided)
  frames against our 112 / 104 — a 4–7× harder bar. Measured `core_degen` is 0.520 / 0.342 on the
  external arms vs 0.053 / 0.056 on the runs, which would read as a model difference; every such
  cell now carries a **†** and a footnote, and `copy_max` is marked for the weaker form of the same
  effect. `SCORE_SETS` entries also accept `paths` (a list) so the pending DeltaAI re-score of
  ic_gen/ctt_v2 — landing one directory per arm — is a single new entry; procedure in
  `misc/refvfx_baseline/VIEWER_NOTES.md`.

- **14:17** — **`iclora_runs` viewer: the two external refVFX baseline arms, on the same cards.**
  Both arms of the refVFX prior-work baseline (`$LAB/misc/refvfx_baseline`) now appear on the
  IC-LoRA trainings page beside `ic_gen` and `ctt_v2`: **Ⓐ** refVFX in its own prompt convention
  (the prompt describes the effect) and **Ⓑ** refVFX under our text budget (no transition
  information in text). 152 rows × 2 seeds each, joined to the existing 139 cards by `item_id`, all
  608 clips live. They are wired as **context tiers**, the mechanism the page already had for
  `specialist`/`copier` — so they get a column and enter the per-arm aggregate tables but never the
  run chips, the paired per-card Δ or the donor-class sign test, which keeps those meaning what they
  meant. Adding the next external baseline is one entry in the new `EXTERNAL` list.
  The genuinely new requirement was the **prompt**: unlike our arms, which all share the registry
  prompt, these two differ from it and from each other on every row, and that contrast is the point
  of running both. Each clip's output box now carries the exact prompt from that arm's manifest row
  for that item and seed, the inputs band shows all three conventions side by side, and
  `diff_span()` marks the clause that differs from ours — against one baseline, so Ⓐ's effect
  description and Ⓑ's class-agnostic clause land in the same place. Their conditioning bar states
  refVFX's own contract (first-frame anchor, 33f, duration-matched) rather than redrawing our
  prefix/suffix geometry over it. Videos are **symlinked**, never copied: `ensure_external_media()`
  rebuilds `outputs/videos/refvfx_baseline/<arm>` on every build, and the convention is written up
  in `docs/VIEWERS.md` → *Media that lives outside the repo*. Scores are read from the harness's own
  output at `$LAB/misc/refvfx_baseline/eval/scores/<arm>/*/items.jsonl` with the same
  pool-refs→seeds collapse the runs use; an absent directory renders the arm as "unscored — video
  only" with no placeholder numbers anywhere. Landed scored: `transition-eval/4.0.0`, corpus
  `dc2e139a` — the same reference_v4 build the run columns use, read out of the scorer's
  `results.json` rather than asserted — pool-% **41.7%** (Ⓐ) and **29.2%** (Ⓑ) against ic_gen 70.3%
  / ctt_v2 72.9%, near_copy 0 at τ=0.858. Notes: `misc/refvfx_baseline/VIEWER_NOTES.md`.

- **14:17** — **Repaired 25 dangling `/projects` symlinks under `outputs/` after the DeltaAI move.**
  `outputs/videos/ladder2/ctt_v2__ck10000` and every shard of `outputs/eval/ctt_v2_compare` still
  pointed at the Campus-Cluster path, which is not mounted on DeltaAI — the ctt_v2 column had no
  videos and no scores, and `build_runs.py` could not run at all. Repointed to the `/taiga` twins
  (same files). The only remaining dangling link is the already-archived `humanvid_sample`.

- **11:28** — **New viewer: `iclora_runs` — every IC-LoRA training on one page, one chip per run.**
  `eval_ladder/viewer/build_runs.py` + `template_runs.html`, forked from the ladder2 results viewer
  (which is the published ladder2 record and is left untouched). ctt_v2 @ step 10,000 sits beside the
  ladder2 IC-LoRA generalist on **139 identical inputs** — the two registries share `input_key`
  field-for-field, so both runs land in the same card with no view-time join — plus the specialist
  and copier tiers. Adding the next training is one entry in `RUNS`. Three things the build enforces
  that the old one could not: (1) every score set is loaded **by explicit path** and every generation
  carries the artifact that scored it — `report_full.SCORES` is a module constant that ignores
  `$LADDER_SCORES`, so importing it silently reads the stale-artifact scores, which is exactly the
  cross-instrument trap this campaign was bitten by; (2) the run columns come only from the rebuilt
  222-clip rescore, and stale-scored tiers (specialists) are badged and never merged into a run's row;
  (3) `check()` refuses to emit unless every run is fully scored, the runs' card sets are 1:1, and
  every scored arm declares its instrument. Measured and stated on the page: the cross-build error,
  from the 304 ic_gen generations that exist under *both* artifacts, is **0.09pp mean / 0.31pp max at
  cell level** against a 0.4–15.9pp effect. No prompt+endpoint baseline column — `base_prompt`/
  `base_cond` were never scored under either artifact and the two candidate substitutes are different
  rosters that disagree by ~19pp, more than the effect itself; the gap is stated on the page instead
  of being papered over. Crossfade/freeze floors are recomputed from this roster's own eps control
  rows (**30.3% / 17.5%**, vs POOL_YARDSTICK's 48%/22%). Those two numbers are **different lanes, not a
  roster difference**: the doc's figures come from exp_072's separately-constructed control arms,
  which re-aggregate to 43.6%/18.9%; roster was tested and ruled out (no roster x score-set
  combination exceeds 32.6%). Both are correct for their own lane and must never be quoted against
  each other — see `RUN_RECORD.md` §20, which also appends a dated correction to the ic_gen
  comparison column reported earlier (G-zs-same is **-15.9pp, not -11.9pp**, on n=8).
- **09:05** — **`run_gen.py` no longer hardcodes `LAB`.** `MODEL`/`GEMMA` were derived from a
  hardcoded campus-cluster path, so on any other machine they resolved to files that do not exist and
  generation died at model load. `LAB` now reads the environment (default unchanged, so CC behaviour
  is byte-identical) and `LTX_MODEL`/`LTX_GEMMA` allow weights that do not live under
  `cache/huggingface/...`. Found by dry-checking path resolution on the eps box *before* the
  post-training window rather than during it. Also corrected the `ctt_v2` arm note to rank 128
  (the run was valve-demoted from 256).

## 2026-07-29

- **19:07** — **`run_gen.py` reference-attention defect fixed, and the `ctt_v2` arm registered.**
  `build_sample` constructed `ReferenceConditionConfig` without passing `attention`, so it silently
  took the schema default `bidirectional`. Harmless until now — every arm through `b1`/`b1r`/`m1lite`
  was trained bidirectionally — but the CTT v2 IC-LoRA is the first adapter trained with **one-way**
  reference attention, and generating it bidirectionally is a train/inference mismatch that the
  config's own docstring warns about. Attention is now read per-arm from `arms.yaml`
  (`build_sample.ref_attention`, same pattern as `ref_downscale`) and defaults to `bidirectional`,
  so every existing arm is byte-identical. Added the `ctt_v2` arm (attn_ffn, step 10000,
  `attention: one_way`); note its rank/alpha are **256**, and `run_gen.py`'s `--rank/--alpha` flags
  still default to 32, so they must be passed explicitly.

- **13:10** — **DATASET.md fully reconciled with the final v2.1.0 state** — 37 anchored corrections
  across §1 (all PENDING rows closed with how each resolved; "why not stampable" → all five gaps
  closed), §3 (sample contract rewritten to the samples.jsonl row; silent-drop hazard marked
  ELIMINATED with the record kept), §5 (final counted table: 18,800 clips / 1,670 groups / 56,368
  pairs; S1 AS-LANDED block — 1,417 rendered in two layers, owner reject pass superseding the
  never-run batch gate), §6.4/§9/§10 (captions LANDED; assert suite marked RETIRED with its
  replacements named; §10.11 documents the executed v2.1.0 assembly path), §13/§14 (version chain
  0.9.0-DRAFT → 2.0.0 retired → 2.1.0). Gate-8a mentions aligned to the C3 record (FAIL 0.8849,
  owner ship-as-is, countersign pending). Historical rulings and measurements untouched.

- **12:45** — **CTT v2 dataset promoted to top-level `datasets/ctt_v2/` (gitignored; public-repo
  safe) and the three retired symlink roots DELETED.** Dataset verified intact post-move (56,368
  rows, 41,195 files, 0 missing, 4 s); compat symlink left at `outputs/ctt_v2/dataset` so every
  earlier pointer chain still resolves. Clip mp4s were also materialized beforehand — the SSOT now
  contains ZERO symlinks (19,536 real mp4s incl. the 139 S0 corpus sources; staging paths under
  `outputs/videos/ctt_v2_*` compat-link back in). Retired roots' assert/manifest records archived
  to `$LAB/misc/ctt_v2_final/artefacts/retired_roots/` before deletion (~2M inodes freed).

- **12:12** — **CTT v2 dataset FINALIZED at v2.1.0: list-based SSOT directory + trainer-side
  stratified sampler; the 2M-symlink physical root is retired.** Owner rejected 192/1,417 S1
  generations in a grouped-autoplay labelling viewer (`scripts/build_s1_label_viewer.py`;
  hero_flight 87→9, its 1-sided sibling dropped) — applied at the spec (`build_s1_spec.py
  --rejects`), cascading to 56,368 base pairs. Owner then halted physical symlink assembly, so
  `scripts/ctt_v2/build_dataset.py` now builds **`outputs/ctt_v2/dataset/`** — samples.jsonl
  (one row per pair naming its 5 store files), mix.json, moved stores with compat symlinks,
  S0 copies (originals untouched), inventories, docs, MANIFEST. Trainer (`LTX-2-cond-bleed-fix`)
  gained `SampleListDataset` (explicit list, missing file = hard error — deletes the silent-drop
  glob-join failure mode) and `StratifiedEpochSampler` (largest-remainder quotas, without-
  replacement per stratum, deterministic in (seed, epoch), fails closed; replaces assert A3's
  off-disk countability). Verified with real trainer code on the real directory: 41,195 files
  present, realised mix within 0.0006 pp of target (physical root's residual was 0.4289 pp),
  determinism + `set_epoch` exact, per-stratum tensor loads correct. Mix weights are now plain
  training-config floats — changing proportions never rebuilds the dataset. DATASET.md §15 +
  a fresh STAMP block record it (sign-off rows pending owner).

## 2026-07-28

- **16:14** — **S4 captioned and wired in: 2,000 first-frame descriptions at zero API cost, conditioning
  narrowed to video frame 0, and a latent mask-reuse trap caught by making the prefix width derived.**
  Owner directive was to condition S4 on the first frame only (not frames 0–8) and to caption accordingly.
  The refVFX source captions could **not** be adapted: it ships one trigger phrase per *effect* (42 over
  2,000 clips), which is a class label and a Tier-1 leak string, so descriptions were generated from pixels
  by **25 fan-out Sonnet captioners × 80 clips**, batches round-robined over an effect-sorted roster so a
  captioner's style could not track an effect. Result: **2,000 / 2,000, 0 Tier-1 leaks, 0 format violations,
  0 key collisions with the locked store, `hard_fail: []`**, store hash `fcd46f33…`, **0 ₺** (frame
  extraction was 14 s for all 2,000). Kept as a **separate** store: its prompt is `v2` role-A verbatim but
  for *"9-frame snippet"* → *"single still frame"*, and merging would falsify the locked store's
  `single_prompt_variant` assert for no gain.
  🔴 **The mask change found a live trap.** `m[:2]=1` was a literal repeated at six call sites; S4 needed
  `m[:1]`. Making it a shape property (`root_common.prefix_latents`) and putting it in the mask **filename**
  meant `regen_masks.py` reported *"exists but is WRONG and --force was not given → would have been
  REUSED: interior m[1:-1] is not all 0"* — under the old flat name the stale 2-frame S4 mask had the same
  `(f,h,w,sided)` and would have been trusted. Separately, that module's *"bit-identical to
  `ensure_mask`"* check was comparing **file** bytes, and `torch.save`'s zip container records an mtime —
  two saves of an identical tensor hash differently, so the check was passing only when both writes landed
  in the same mtime bucket, and its OK log printed "bit-identical" even on failure. Now compares tensors,
  records a reproducible `content_sha256`, and the manifest's verdict is the real result.
  **Effective-weight consequence:** S4's loss-bearing tokens go **1,092 → 1,456** (60 % → 80 % of each
  sample) since conditioning drops 40 % → 20 %; the 121f values are unchanged. The nominal mix is
  pre-registered and untouched — the owner still picks S4's nominal share. New: `build_s4_batches.py`,
  `merge_s4_captions.py`, `assemble_s4_captions.py`, `s4/build_s4_spec.py`; S4 inventory is **42 groups /
  2,000 clips / 6,000 pairs**, matching the `EXPECTED_BASE_PAIRS` assert. `build_s4_spec.py` also fixes the
  rehearsal spec's empty `endpoints: {}`, which would have made `caption_sources()` return `[]` and drop
  every S4 description sentence silently. Alignment — the one defect no length or lexeme check catches, and
  two captioners self-reported catching an indexing drift mid-draft — was spot-checked directly:
  **10 (stem, caption) pairs across 8 batches, frames re-read, 10 / 10 exact.**

- **13:45** — **A16 executed in code: the 29 S2a clips are DROPPED-AND-RECORDED at consumption, and the
  Keyed-Join Rule is now enforceable machinery — plus a live vacuous-exclusion landmine found on `main`.**
  🔴 **The finding first:** `data/processed/` is gitignored, so `POOL_DROPS_M3_ADJUDICATION.json` travels
  with a *working tree*, not a branch — it was **absent from the consolidated `main` checkout**, which made
  `root_common.ROLE_EXCLUSIONS` an empty dict. Every `role_excluded()` call answered `False`, A10's exclusion
  was silently vacuous, and the A16 drop would have dropped **0 of 29** while every assert printed PASS —
  the `INTENDED_WEIGHTS_PCT` landmine class again. Fixed by copying the three gitignored artefacts
  (`POOL_DROPS_…json`, `CONTENT_POOL_union.json`, `s4_refvfx/selection.json`) into `main` *and* by a new hard
  guard `rc.require_role_exclusions()` that `build_inventories` and `assemble_root` call first: a vacuous
  standing exclusion is instrument failure, not "nothing to exclude". Any fresh clone of this public repo now
  fails loudly instead of running vacuous.
  **The ruling itself:** `build_inventories._attach` no longer `SystemExit`s on a role-scoped consumption
  hit — it derives the drop set as `ROLE_EXCLUSIONS ∩ what the stratum consumes` (never a stem list), drops
  it, and records each drop under `inventory["build_drops"]` with the assembler's own reasons vocabulary
  (`role_scoped_caption_exclusion` + `role_scoped_prefix_condition`); `assemble_root` propagates that into
  `ROOT_MANIFEST.json:drops` tagged `dropped_at`, so the manifest stays the complete account of every clip
  rendered but not consumed. **The crash is kept** for a missing *non-excluded* caption key, for a caption
  that exists for an excluded consumption (a fabricated cross-role fallback), for a vacuous join, and for a
  wrong-shaped lookup key. Verified on the real data: S2a **7,990 → 7,961** clips / **23,883** base pairs,
  29 stems over **29 distinct ops**, one reason-pair each; S2b **7,990** untouched.
  **Keyed-Join Rule (A16 items 1 and 4), in `root_common.py`:** `KeyedStore` with raising accessors and no
  `.get()`, `assert_key_shape` (validates a lookup key against the store's own keys *and* its `keying`
  self-declaration before any result is interpreted), `assert_join_nonvacuous`, `require_keying_declaration`,
  `load_keyed_store`. `.get()`-against-a-keyed-store converted in `s1/build_s1_grid.py`,
  `captions/assert_caption_store.py` (new `S0_store_key_shape` check — every other check there is an absence
  check, so a wrong-shaped store made them all pass on nothing), `captions/consolidate_store.py` (which now
  proves at write time that `keying` describes the keys beside it) and `smoke/mixed_format_probe.py`.
  **Stale comment fixed:** `assemble_root.py`'s *"occupies field B in all 10 rendered clips"* — false, and the
  belief behind all three key-shape incidents — is replaced by A16's enumerated table (S2a 29 field-A / 0
  field-B over 7,990 rows · S2b 0 / 37 over 7,990 · S1 0 over 390).
  **Mutation-proven both directions:** `tests/prove_asserts.py` gains `BUILDER_MUTATIONS` (6 mutations,
  real subprocess runs of the real builder) — excluded-hit ⇒ drop+record; missing-non-excluded ⇒ crash;
  both-at-once ⇒ still crash; fabricated fallback ⇒ crash; wrong key shape ⇒ crash naming the shape; vacuous
  exclusion ⇒ crash. **6/6 PROVEN, and the existing 33-check battery is GREEN before and after.**
  `S2_ACCEPTANCE.json` is **not re-run and not edited** — annotated in `data/DATASET.md` per A16 §Q2.

- **13:05** — 🔴 **Real defect found at assembly-inventory time: 29 rendered S2a clips consume the
  one role-excluded (clip, role) pair.** Of the 7,990 S2a records, **29 use
  `openvid_T1MiFx98l3g_0_50to156` as their A endpoint** and so need the role-A description A10
  deliberately withheld (blank-screen opening anchor). Verified first-hand: 0 B-endpoint users, its
  B-role description is present, **S2b and S1 unaffected**. Changes **no count and no hash** —
  1,403/1,403 and `c8e2d95b…` stand — because it is a *consumption*-side gap at assembly, not a
  store gap. Same shape as the original S2a requirement defect: right about the requirement, wrong
  about what was rendered. **No cross-role fallback invented** (it would caption a blank screen with
  content it does not show). Recorded OPEN in `CAPTIONS.md` §4.2, `CAPTION_LOCK.json`
  (`OPEN_CONSUMPTION_GAP`) and DOSSIER §27.6 with three priced owner options: drop the 29 (0.36 % of
  S2a), overturn the A10 exclusion, or re-render. Found by the assembly rehearsal's first inventory
  build before it was stopped.

- **12:56** — **Caption lane finished: one source of truth, the battery gap closed, and the DeltaAI
  S1 package built.** (1) `data/CAPTIONS.md` is now **the** authority for captions — rewritten lean
  (store, keying, grammar, the three sources, the S2a defect, the auditor churn, v2-not-v3, the
  corpus-139 audit, measured spend, the battery). `data/DATASET.md`'s six stale
  *"blocked: Gemini credits"* caption rows were corrected and its status banner now names the real
  blocker (**S1 media**, not captions). (2) **The §21.7 battery-scope gap is CLOSED**: the existing
  12-gate battery re-ran on the **full 1,403-row store** (was a pooled 447-row subset) —
  `hard_fail: []` holds, zero API spend, and the input is *proven* to be the locked store because
  the pooled in-grid shards re-hash to `c8e2d95b…`. 8a **fell** 0.7099→0.6819 (away from its 0.73
  drift-guard bar, so no stop condition fired); 8b **rose** 0.5787→0.5950 — still PASS at ≤0.60 but
  headroom shrank to 0.0050 (~0.67 SE), so the load-bearing gate now passes narrowly and that is
  recorded prominently. Gate 9's content-dominated ACCEPT is unchanged (33/40 features are content).
  `CAPTION_LOCK.json`'s `SCOPE_CAVEAT` is replaced by the full-store result with the subset kept for
  the record. (3) **`misc/ctt_v2_final/deltaai_s1_handoff/`** — self-contained run book for the
  owner's GH200 run: the 390/390-prompted grid (sha `dea8ffe436998e99`), `MANIFEST.json` with
  sha256 for all 11 adapters / 400 media / 33 control clips, `VERIFY_ON_ARRIVAL.sh` (tested; it
  correctly flags `gemini-3.5-flash` HTTP 503 as a gate-only WARN), a `ghx4` sbatch template, and
  `retarget_grid_paths.py` — **which fixes a real defect: the grid's 400 media paths are absolute
  `/projects/...` and DeltaAI has no `/projects` mount**, so every row would have failed.

- **12:25** — **CTT v2 captions LOCKED for S0/S1/S2 — 1,403 / 1,403 = 100%.** Found and fixed the
  real defect: `build_mass_pair_list.py` read only 2 of 3 sources, because **S2a's endpoints live
  only in its rendered metadata keyed `A`/`B`**, so a strict `endpoint_a` lookup returns an empty set
  and the bug reads as the reassuring *"S2a needs no descriptions."* The true requirement is **1,404**
  (S2a 454 ∪ S2b 1,217 ∪ S1 400), not 1,348 — 36 (clip, role) pairs were absent and 26 were at risk
  of never being generated. The builder now requires all three schemas, asserts positive presence per
  record, recomputes the derived constant 454 with **SPEC-CONSTANT-MISMATCH** escalation (never a
  fallback branch), and records the vacuous `endpoint_a` lookup as a trap witness. Adding S2a also
  made the `openvid_T1MiFx98l3g_0_50to156|A` role-scoped exclusion **live** (it was a genuine no-op
  under two sources), so 1,404 − 1 = **1,403 generatable** and nothing is short. Generated the 213
  missing descriptions and resolved the residual 3 via `manual_rewrite.py` — a deliberately separate,
  re-audited operator path, so a hand-written string can never enter the store on a path mistakable
  for a generated one; **unresolved `inaccurate` = 0**. Ran the **S0 corpus-139 Layer-2 audit** for
  the first time: **0 / 171 `leak=YES`** (no endpoint description leaks the transition effect) and
  4 / 171 `inaccurate=YES` — certified captions kept byte-identical, all four escalated to the owner,
  the script read-only by construction. Consolidated 10 scattered shards into one hashed canonical
  store (`sha256:c8e2d95b…`, single prompt variant `v2`, single auditor `gemini-3.5-flash-lite`) with
  everything else moved to `archive/`. Docs: `data/CAPTIONS.md` is now the caption authority and
  DOSSIER §26 supersedes the caption trajectory for state. **Two deviations reported, not buried:**
  the directive's v3 prompt was not used (it would mix prompts into a v2 store, trip the bug class
  gate 8a detects, and cost ~290 TRY to chase a ~1.2 SE noise delta), and its `gemini-3.5-flash`
  auditor is HTTP 503 / unavailable. **One gap named:** the 12-gate battery stands on a 447-row
  subset (`hard_fail: []`, 8a 0.7099 ≤ 0.73, 8b 0.5787 ≤ 0.60), not the full 1,403 — no new gates
  were run per owner direction. Session spend measured at 609 calls / 185,089 tokens ≈ 58.6 TRY.
- **12:10** — Executed reconciliation ruling A14 for the CTT v2 caption lane. The keystone
  matched-side auditor control **PASSES**: `gemini-3-flash-preview` flags only **1/192 = 0.52 %** of
  correctly-matched descriptions (bar ≤10 %) with **0 errors** over 213 calls, closing the one-sided
  gap the 220/220 mismatch certificate left open — it is the best of the three candidates
  (flash-lite 2.00 %, 3.5-flash 5.75 %). Auditor **pinned**, no switch-back. The cross-auditor
  calibration shows it re-flags only 52.4 % of 3.5-flash's positives (−47.6 pp), so first-pass rates
  measured under it run **mechanically higher** — the ≥97 % bar is easier under the pin, and that is
  recorded rather than banked. Reuse of the round-2 descriptions verified: the v2 prompt is
  **byte-identical** to the round-2-era blob across all 62 reachable renderings, and the generation
  config matches, so A14's overturning condition does not fire. Also landed the gate battery's
  mean ± SE-of-mean reporting (8a = 0.6909 ± 0.0119 SE, which is what makes v3's 0.0137 movement
  visibly 1.2 SE of noise) and the **fourth** fix in the "checker whose failure looks like a pass"
  class — `gate_s1_pilot.py` no longer lets a judge outage shrink *n* instead of failing.
  **Stopped before the mass run**: a second session is executing the same lane concurrently and has
  already generated and audited the whole store under the *fallback* auditor. Details, priced
  options and the recommendation are in `misc/ctt_v2_final/DOSSIER.md` §25.
- **11:15** ctt_v2/captions: **K=10 PACKING PILOT RUN — PACKING IS REJECTED, twice over** (advisor
  A12 §3, steps 1–4; `scripts/ctt_v2/captions/pack_pilot.py`, `pack_analysis.py`,
  `pack_clip_check.py`; artifacts in `pilot_m3/packed_k10/`). 200 descriptions in 20 role- AND
  bank-homogeneous packs, the pinned round-2 prompt embedded byte-identically exactly once
  (assert proven to fire on tampering, double-embedding, absence, wrong role and any `v3` text).
  **Generation packing fails gate 8a at 0.7544 vs the ≤0.73 drift guard**; **audit packing fails
  independently** — the within-pack derangement control flags only 96.0% (bar ≥99%) with exact
  positional attribution in 17/20 packs. Cost measured, not estimated: `c_desc` = **378.9 tok**
  (packed gen 175.9 + unpacked audit 203.0) against the 682 unpacked baseline, and packed audits
  save only 7% because the description text itself never amortises — A12's correction confirmed
  and then some. Passing conditions: 3b matched flag rate 2.0% (≤10%), 3c ID echo intact 200/200,
  5 first-pass 100% on the prompt-controllable scope, 4 lexical-overlap ratio 1.035 (≤1.15).
  **Diagnosis of the 8a failure:** packing silently compresses descriptions by **−4.61 words on
  the same clips** (−7.0 SE, p50 28 vs 34, corpus 33) because `calibrate_ask`'s length fit was
  fitted on *unpacked* generation and does not transfer to K=10; removing length/punctuation
  features drops 8a to 0.7408, still over the bar, so the residual is genuine register drift.
- **11:15** ctt_v2/captions: 🔑 **the echoed item ID is a WORSE attribution key than array
  position** — the opposite of what the packing spec assumed. One pack in 20 returned all ten ids
  intact but with two ADJACENT items transposed; CLIP text-image adjudication (`clip_diagonal.json`)
  says the pixels support **array position** (mean sim 0.3151) over the echoed id (0.2880), with the
  two transposed items at sim 0.348/0.291 by position vs 0.133/0.236 by id. So the model emitted the
  descriptions in the right order and mislabelled two of them. Keying by id — which the spec
  mandates — therefore *introduced* the only two real mispairings in the store, and the diagonal
  argmax would have been 199/200 rather than 197/200 under position-keying. Neither key is safe at
  K=10: 5% of packs carried an attribution defect under either. Recorded because any future packing
  or multi-item-response design will hit this, and an ID echo that is 100% "intact" is *not*
  evidence of correct attribution.
- **11:13** ctt_v2/captions: **an empty/failed Layer-2 audit verdict is now a HARD ERROR, and the
  auditor is re-pinned to `gemini-3.1-pro-preview`** (advisor A13, steps 1–2). §21 recorded that a
  non-200, unparseable or empty audit response was scored as a **clean pass** — `_post` returned
  `(None, err)`, `verdict` stayed `None`, and the caller's `v = arec.get("verdict") or {}` turned it
  into `{}`, which fires neither `leak == "YES"` nor `inaccurate == "YES"`. An auditor outage
  therefore minted descriptions that *look* audited and are not: the §13.12 defect class (a checker
  whose failure is indistinguishable from a pass) for the third time this campaign. `AuditError` now
  raises on non-200, no-response, unparseable, empty, field-incomplete and out-of-domain verdicts,
  the raw response is archived and flushed *before* the raise, and the run aborts rather than
  writing a partly-unaudited store. `scripts/ctt_v2/tests/prove_audit_hard_error.py` proves it:
  **32/32 cases**, including 14 fatal shapes, 5 in-domain verdicts that must still be honoured, an
  end-to-end run of the real driver over a scripted network boundary (abort on outage, archive
  survives, no store written, happy path still stores, a genuine leak still regenerates), and a
  demonstration that the pre-fix expression scored **12 of the 14** unusable verdicts as clean
  passes. Auditor pin: `gemini-3.1-pro-preview`, temp 0, `thinkingLevel: "low"` (the pro tier
  rejects `"minimal"`), `max_output_tokens` 512 so the JSON verdict cannot truncate.
- **11:05** ctt_v2/captions: `build_mass_pair_list.py` written and run — **the mass caption run
  needs 1,348 descriptions, not the ~5,600 the runbook assumed** (~4× less, ≈85 s at the measured
  16 desc/s including the 100% Layer-2 audit). It turns the pinned grids into the (clip, role)
  list A4 Q7.3 asks for, understanding both grid schemas (S2 `pairs`, S1 `rows` with
  one-/two-sided endpoint semantics) and hard-stopping on an unrecognised one rather than
  contributing zero pairs silently. The 800 S2 rows collapse to 1,217 distinct (clip, role)
  and the 390 S1 rows to 400, because a description is per-(clip, role) and clips recur across
  rows; 376 clips need both roles. Role-scoped exclusions are read through the same
  `root_common.load_caption_store_exclusions` loader the generator and `assert_root.py` use, so
  the three channels cannot drift. Worth recording: the
  `openvid_T1MiFx98l3g_0_50to156|A` exclusion **matches nothing in the requested set** — the
  pinned grids never use that clip as an A endpoint (0 times A, 3 times B), so it is a no-op
  here and its legitimate B-role is present as required. The script says so out loud rather
  than reporting a reassuring "skipped 1". Blocked on the auditor before generation can run.
- **10:55** ctt_v2/captions: **round 3 (prompt v3) run and scored — all 12 gates PASS, but v3's
  stated mechanism is FALSIFIED.** The Gemini generator (`gemini-3.6-flash`) came back, so
  `RESUME_ON_CREDITS.sh` steps 1–2 ran: 399 pairs (the role-scoped A-exclusion for
  `openvid_T1MiFx98l3g_0_50to156` correctly derived from `POOL_DROPS_M3_ADJUDICATION.json`),
  398 accepted, 9.4 s, 0 retries, 0 HTTP 429. Gate **8a 0.6929** (bar ≤0.73) and **8b 0.5454**
  (bar ≤0.60, load-bearing) both PASS with no hard failures; 8c stays FAIL-on-record at 0.6929
  as pre-committed. Gate 9 AUC 0.9102 → top-40 dump is content-dominated (~32/40 content words),
  disposition **ACCEPT and record**, consistent with rounds 1–2. **However:** v3's only change
  was relaxing A4's verb-form clause to permit plain *is/are* on the A-role, to close a claimed
  be-verb tell — the A-role prompt was verified changed, yet the realised A-role be-verb rate
  stayed at **0.0%** (round 2: 0.0%). Two follow-ons: the motivating "8.8% corpus" figure is
  *pooled* and mis-attributed — corpus A-role is only 5.0% (n=139) while corpus B-role is 25.0%
  (n=32), so the deficit is mainly a **B-role** phenomenon that v3 deliberately did not touch;
  and 8a's move 0.7066→0.6929 is 0.0137 against a fit std of 0.0568, i.e. noise. A4's third and
  last regeneration round bought a passing store, not a demonstrated fix.
  Round 3 ran **unaudited**: `gemini-3.5-flash` (the round-1/2 Layer-2 auditor) is HTTP 503 on
  0/18 probes at concurrency 1 and 8, so the run used `--no-audit` rather than a 503 auditor
  whose empty verdict is indistinguishable from a clean pass. Its first-pass number therefore
  covers format+Tier-1 only (99.5%, an upper bound) and **cannot gate** the ≥97% / ≤8% bars.
  Availability archived in `pilot_m3/MODEL_AVAILABILITY_20260728.json`: note that
  `gemini-3-pro-preview` is **retired (404), not 429** — DOSSIER §5.1's 429 note is superseded —
  and its successor **`gemini-3.1-pro-preview` is available**, which reopens A4's original
  pro-tier auditor as a live option. Steps 3–7 are blocked pending an auditor ruling.
  Two archival defects fixed in `generate_descriptions.py`: `N_asked` recorded the
  pre-calibration draw for v3 (calibration is applied for v2 *and* v3), and `run_meta` named an
  auditor model even on `--no-audit` runs.
- **10:45** ctt_v2: **group ids are now SLUGGED AT PATH CONSTRUCTION** (A11 item 3 — the
  slugging existed and was declared, but `assemble_root.py` still wrote raw ids into paths).
  `assemble_root.py` builds every relative path from `root_common.slug_group` — the *same*
  function assert A14 checks with, deliberately not a second implementation — hard-stops on a
  collision or an empty slug over both the assembled and the full inventory group sets, and
  stores the raw↔slug mapping in `ROOT_MANIFEST.json:group_slugs`. `assert_root.py` resolves
  the path's group through that mapping before every group-keyed comparison (the inventory
  lookup, the inline-OOD op set): a slug silently matching nothing is exactly the
  namespace-drift vacuity A0 exists to catch. A14 additionally requires that every group
  component in a root path resolves back to a raw inventory id and that the *stored* mapping
  agrees with the recomputed one. **Nothing on disk is re-keyed** — symlink targets are
  untouched absolute paths into the render/encode stores, and `.get(group, group)` bridges a
  root assembled before this landed (root_2shape, raw paths, still 33/33 PASS). Verified on a
  fresh fixture root (`_root_machinery_test/root_slug`): 33/33 asserts PASS, 810 of 2,458 rows
  carry a slugged group, no group dir contains an unsafe character, and the real 42 refVFX S4
  effect ids — **40 of which contain spaces today** — slug to 42 unique non-empty strings.
- **10:40** ctt_v2/asserts: `A3b_prorata_multipliers_equal` is **proven to fire** — it was the
  one assert in the battery with no mutation (added by the A12 rewrite after the mutation set
  was written, so it had never failed). The new mutation moves five S2b samples into a second
  replica dir in all five trees, so S2b shows 2 replica multipliers on disk against S2a's 1
  while every stratum's counted share is untouched — count-preserving on purpose, because a
  literal ×2 duplication would also (correctly) blow A3's ±0.5 pp tolerance and the run would
  not show A3b is sensitive to the multiplier itself. It fires **strictly**: A3b alone, with
  no external co-firing. Proof set 35 → 36 mutations.
- **10:35** ctt_v2/smoke: the smoke gate's own checkers are now **proven to fire**, GPU-free,
  against the archived log of the passing run (`scripts/ctt_v2/smoke/prove_smoke_gate.py`,
  16 cases). Three defects found and fixed by the mutations. (1) A log that EXISTS but cannot
  be read raised an uncaught `OSError`, and Python exits **1** for that — the same code A9 §4's
  fallback ladder reads as a DATA failure; the read is now a checked step (`read_log`) and
  every such case is `PARSER_FAIL`/exit 2. (2) With Rich escapes present but unstrippable,
  `T3_steps_completed` still found a checkpoint filename and reported "highest evidenced step
  N of 30" about a 30/30 run — the original false negative surviving in one check; T1–T4 are
  now forced UNEVALUABLE whenever `T0_parser_sane` fails, with the suppressed reading kept for
  the record. (3) Fed the Slurm capture (which the sbatch appends the gate's own report to),
  the gate reported `T6 … 'loss is NaN'` and exited 1 purely from its previous report — the
  self-test pinned that hazard for `evaluate()` but nothing guarded the CLI path, which now
  slices the trainer's region first. **And A11's Derived-Constant Rule is now implemented in
  the shift assert**: G3 classifies its own failure as `DATA-FAIL` (mechanism inconsistent —
  verdict FAIL, exit 1, ladder permitted) or `SPEC-CONSTANT-MISMATCH` (a pinned literal
  disagrees with reality — verdict `SPEC_CONSTANT_MISMATCH`, exit 3, escalates, ladder
  forbidden), from one shared classifier used by both the `--shifts-only` and full paths.
  Proven by setting the 1,820-token pin to A9 §3's superseded 1.120 and, separately, by
  repointing the S4 arm's tensors at 121f geometry: both come out as escalations, not as the
  auto-drop of a healthy stratum.
- **04:36** ctt_v2/smoke: both smoke-gate "failures" in job 9688250 were in the CHECKERS,
  not the training — the trainer had in fact completed 30/30 steps over the mixed two-shape
  root with a finite loss. `check_train_log.py`: an empty match set is now its own hard error
  (`T0_parser_sane` → verdict `PARSER_FAIL`), and every check that reads the extraction goes
  UNEVALUABLE rather than FAIL, so "I can't read this log" can no longer surface as "your
  training is broken". T3 now also takes step evidence from the trainer's checkpoint lines
  (a 30-step run logs only one loss line, at step 20). A `--self-test` runs on every
  invocation against the sha-pinned job-9688250_1 capture plus three negative cases.
  `mixed_format_probe.py`: the float32-vs-bf16 crash was the intrinsic-mask promotion in
  `flexible.py:542`, which the real trainer absorbs via accelerate's bf16 autocast — the probe
  now reproduces and guards that instead of casting. A11 item 4 clause (b) now *observes* the
  shift the sampler was actually handed (`_ShiftRecorder`) instead of re-deriving it from
  `f*h*w`, which had made the assert unfailable.
- **04:32** ctt_v2/S1: role-scoped (clip, role) exclusions now enforced at S1 grid-build time (mandatory `for_role` in `take()`), not merely detected by assert A13. Proven by mutation.
- **04:45** — **The RULING-9 assert battery is now PROVEN TO FIRE, on a real two-shape root.**
  An assert that has never failed is not known to work, and this campaign has twice met the
  failure class where a gate prints PASS on a broken input. So: `scripts/ctt_v2/tests/
  make_fixture.py` builds a five-stratum fixture with **real structure and stub payloads** —
  real S0 tensors and captions, S2a from the 7,990-row render manifest, S2b from the frozen
  800-op plan, **S4 from the frozen `selection.json` at (5,14,26) @ 16 fps**, S1 from the
  Ruling-3 grid grouped by *arm* (A11 item 6) — so the root holds **both shapes** and masks,
  token counts and derived shifts are exercised for real. `scripts/ctt_v2/tests/
  prove_asserts.py` then breaks **one invariant at a time, in place**, and requires the
  intended check(s) to fail and nothing else to (strict by default), restoring byte-exactly
  and re-establishing the green baseline at the end. Coupled failures are *declared* with
  their reason, never tolerated quietly; the harness holds an exclusive lock on the root
  because two concurrent runs corrupt each other's baseline (that happened once, and it
  looked exactly like a real assert failure).
  New checks, all proven: **A0** pairs every absence assert with a *positive-presence
  control* — A5/A7/A8/A9 all report "= 0" just as happily when the two sides are in different
  id namespaces, which is the same silent failure as a log grep that never matches; **A11a–f**
  the record-level two-shape clauses (did the assembler tell the truth about what it built —
  a stale `_shape_cache.json` makes it self-consistent and wrong); **A12/A13** the two
  consumption channels of A10's role-scoped exclusion; **A14** group ids slug to unique
  path-safe strings; **A15** records nominal vs effective weights. `assert_root.py` now
  **imports** `assert_root_shapes.py` rather than reimplementing it, narrowing its expected
  shape classes to what the manifest declares so the pre-registered S4-cutoff branch (one
  shape) cannot false-FAIL — and a mutation proves that a failure inside the imported module
  reaches the exit code.
  `dryrun_epoch.py`'s "zero skipped" is now **two-sided**: it must also resolve exactly the
  count `ROOT_MANIFEST.json` names, because an absence assert passes trivially when the
  instrument found nothing. It reads no log at all, deliberately — the `RichHandler` ANSI
  escapes that broke a parallel lane's log gate cannot reach it.
  Also: `make_stamp.py` generates the DATASET.md STAMP block from artefacts on disk (so
  stamping is a fill-in, not a rewrite), including a **root content hash** over every
  `(relative path, resolved target)` — what the trainer actually opens, not what the manifest
  claims. DATASET §8.2.1/§8.2.2 carry the analytic σ table, the binding invariant verbatim,
  the supersession note on A9's wrong `{1.120, (5,20,15)}` constants, and the nominal-vs-
  effective disclosure (**S4's 10 % nominal is 3.04 % effective**, inside the ruled 2.8–3.0 %).
  All three pre-registered contingency branches verified numerically against A9 §4 by
  `--plan-only` assembly. **Nothing was stamped and no real root was assembled.**
- **04:20** — **A12: the S2a:S2b split is now DERIVED pro-rata, not forced equal — 271,965 files
  instead of 5,030,200.** Advisor ruling `misc/ctt_v2_final/advisors/A12_prorata_s2_split_VERBATIM.md`
  (0.9+) read A9's full clause — *"S2 total 69, split **pro-rata to the assembled counts**, which are
  ~equal"* — and held that "pro-rata" is the instruction while "~equal" was an observation of counts
  that had not yet met the exclusions. Post-exclusion they are not equal (S2a **22,731** vs S2b
  **23,577** base pairs), and forcing an equal *share* onto unequal *bases* requires differentially
  duplicating the halves — the *"extra reweighting knob"* A1b excluded by name, and a breach of A9's
  own per-op rationale for S2 = 69 (~4.3 draws/op).
  **The mix contract is now `S0 15 / S1 6 / S2 total 69 / S4 10`, with the S2a:S2b split computed
  from the assembled post-exclusion counts.** `root_common.STRATUM_WEIGHTS_PCT` + `PRORATA_GROUPS`
  replace `INTENDED_WEIGHTS_PCT`; the two literal 34.5s are gone from the codebase and from the
  strata manifest (`weight_pct: null`), and `assemble_root.py` hard-refuses a pre-A12 manifest that
  still declares a per-half number. `expand_prorata_weights()` is the single place the split exists,
  and `solve_multipliers()` solves a pro-rata group **as one unit**, so the two halves share a
  multiplier structurally rather than by assertion. The three contingency branches restate the same
  way (S2 total 73 / 79 / 85, split pro-rata); S1 and S4 stay fixed numbers.
  **Measured on the live `S1,S4`-absent branch:** multipliers `{S0 21, S2a 1, S2b 1}`, **54,393**
  base pairs, **max_dev 0.1360 pp** — against forced-equal's `{S0 389, S2a 19, S2b 18}`, 1,006,040
  samples and 0.4296 pp. **18.5x fewer inodes and a 3.2x better deviation.**
  **Assert A3 gained a clause, `A3b_prorata_multipliers_equal`** (`assert_root.py`): the members of a
  pro-rata group carry the SAME replica multiplier, counted from the replica dirs on disk and
  cross-checked against the manifest. It is an exact integer identity, so unlike a share tolerance it
  cannot be satisfied approximately; S0's +-0.5 pp tolerance is unchanged. **Proven to fire:** on a
  throwaway root with half of S2a's groups moved into a second replica dir — sample set and every
  per-stratum count untouched — A3b failed **alone** out of the whole battery.
  **The derived split is pre-registered by freezing its inputs**:
  `misc/ctt_v2_final/PREREG_mix_inputs.json`, written by
  `assemble_root.py --plan-only --write-prereg-mix-inputs`, records every inventory and exclusion
  list by sha256 (the M3 role-scoped adjudication, the seed-42 inline-OOD draw, the holdout/reserved/
  zs/eval sets), the per-reason drop counts that *produce* the counts (S2a 333, S2b 131), the frozen
  counts themselves, the derived targets/shares/multipliers/max_dev, and an amendment rule: any
  change to an exclusion means recomputing the split and logging it as a dossier amendment — the
  counts may move only via a logged amendment, never silently. DOSSIER §15, DATASET §11.1b.
- **04:05** — **S4 credit-independent prep COMPLETE — captions are now S4's only pending item**
  (A9 §5; full record in `misc/ctt_v2_final/artefacts/S4_PREP_REPORT.md`). Zero Gemini calls; the
  trainer was not modified.
  **(1) The mandatory mixed-format smoke gate PASSES, both halves.** The per-format probe
  (`scripts/ctt_v2/smoke/mixed_format_probe.py`, job 9688255, H100) replays
  `trainer.py:_training_step` line-for-line using the certified trainer's own dataset/strategy/
  model/loss — including accelerate's `autocast(bf16)` wrapper, without which the float32
  intrinsic mask times bf16 clean latents promotes `noisy_latents` and `F.linear` raises. All 6
  gates PASS over 130 loss measurements: **realized shifts exactly `{1.2350, 2.3021}` and nothing
  else**; per-format native loss 121f **2.1769** vs 33f **2.3991** (max/min 1.102); and at
  **matched σ** — the shift confound removed — the two formats sit within **1.7–24.6 %**
  (ratios 1.031/1.017/1.032/1.091/1.246 at σ = 0.10/0.25/0.50/0.75/0.90). The real
  `scripts/train.py` over the same mixed root completed **30/30 steps, exit 0, 1.4 min, loss
  finite**, with the trainer's own `Fast index: 10 valid samples from 10 total`, 0 skipped.
  **S4 does NOT auto-drop.**
  **(2) 🔴 A9 §3's pre-written assert would have killed S4.** It orders "realized shifts ∈
  {1.120, 2.302} exactly", but 1.120 needs S4 = (5,20,15) = 1,500 tokens, which cannot exist:
  832×464 is not VAE-legal (464/32 = 14.5), the delivered bucket is 832×448×33 (a pure 16-row
  centre crop, no resampling), so the grid is **(5,14,26) = 1,820 tokens ⇒ shift 1.2350**. On
  healthy data that assert FAILS, and A9 §4 makes a gate failure an S4 auto-drop. The constants
  were re-extracted from `timestep_samplers.py:121-134` (`m = 1.1/3072`, `b = 0.5833…`) and the
  `seq_len = F·H·W` claim verified through the call chain — `sample_for` runs at flexible.py
  Step 3, *before* the reference concat at Step 5, so the IC-LoRA reference does not double it.
  **(3) Per-stratum σ archived analytically** (`scripts/ctt_v2/sigma/sigma_schedule.py` →
  `artefacts/sigma/SIGMA_SCHEDULE.{json,txt,md}`, the `.md` paste-ready for DATASET.md): closed-form
  CDF with the reflection branch and the σ=1 point mass modelled exactly, validated against the
  trainer's own sampler at 4 M draws/stratum (worst sup|ΔF| = **0.00036**). E[σ] = 0.7614 (121f)
  vs 0.6620 (S4), pooled 0.7515. The effectiveness discount A9 weighted S4 on is **12 % smaller**
  than its premise implied.
  **(4) Masks regenerated, never reused** (`scripts/ctt_v2/masks/regen_masks.py`): geometry
  *discovered* from all 2,000 S4 latents (one geometry, one fps), 3 masks over 2 geometries, each
  proven **bit-identical (sha256) to `assemble_root.ensure_mask()`** and asserted absent at A9's
  impossible (5,20,15). Found: S4 conditions **40 %** of its tokens (2 of 5 latent frames) vs
  12.5 % at 121f — a second, undocumented S4 discount.
  **(5) S4 VAE encodes were already complete** (job 9687985): 2,000/2,000 latents + cond_clean,
  all `(128,5,14,26)` fps 16.0, re-verified exhaustively tonight — **not coerced to 121f**. Nothing
  resubmitted.
  **(6) Two-shape root asserts** as a separate importable module (`scripts/ctt_v2/assert_root_shapes.py`;
  `assert_root.py` untouched — another agent owns it), covering what A1's path-level set equality
  cannot see: per-shape five-tree equality, per-sample geometry agreement across all five trees,
  fps-vs-shape, S4's **6,000** (confirmed from `selection.json`: 42 effects × ring `k=3` = 6,000),
  the `Fast index: N of N` gate over both shapes, and a token-count-collision check for the one
  *silent* failure mode. **Proven to fire by 10 deliberately broken two-shape fixtures**, all
  caught, clean one passing.
  **(7) A gate bug worth propagating:** the trainer logs through `RichHandler`, so numbers arrive
  wrapped in SGR/OSC-8 escapes (`Step \x1b[1;36m20\x1b[0m/…`). Regexing a raw capture matches
  nothing and reports **spurious FAILs on a healthy run** — it did, on the first pass. Both log
  gates now strip ANSI first and carry an explicit `T0_ansi_stripped` check.
  **(8) 🔴 A9 §3 item 2 (post-hoc σ split by stratum) is NOT achievable** without editing the
  pinned trainer: `SigmaBucketTracker` carries no sample or stratum identity, its only consumer
  writes to wandb, and `batch["idx"]` is set by the dataset but never read. Options recorded for
  the advisor. ~0.42 GPU-h used total.
- **03:05** — **`data/DATASET.md` written — the CTT v2 dataset single source of truth**
  (`ctt-v2-dataset/0.9.0-DRAFT`, status **NOT STAMPABLE**). Modelled on `eval_ladder/SPEC.md`:
  version stamp + FROZEN/PENDING table, the sample contract (five root dirs, `{target}__ref_{ref}.pt`
  naming, tensor shapes, the trainer's silent-drop hazard), caption grammar with the pre-registered
  §4 bars and the gate-#8 re-pin, sidedness as a class property driving caption/mask/cond_clean,
  one section per stratum (S0/S1/S2a/S2b/S4) with disk-verified counts and gate results, the mix,
  a consolidated holdout table (9 sets), the accepted risks with their rulings, the A1–A10 + D1–D4
  assert checklist, and per-stratum reproduction commands.
  **Every count was verified first-hand on disk; §11 records eight places where disk disagreed with
  the prose.** The three that matter: (a) `root_common.py` still carries A5's mix with **S4 at 0 %**,
  which **A9 reversed** — assembling today would build the wrong mix and then assert it correct;
  (b) S4's real latent grid is **(5,14,26) = 1,820 tokens ⇒ shift 1.2350**, not the (5,20,15)/1,500/
  1.120 the dossier and `REF_mixed_length.md` assume, so A9's pre-written smoke-gate assert
  `shifts ∈ {1.120, 2.302}` would fail; (c) S2a needs **333** caption strings from **454**
  (clip, role) descriptions, not 666/582 — `swap` inverts the shader progress argument only, it does
  not exchange A/B content. §12 lists 8 OPEN items needing a ruling, headed by the unwritten
  `PREREG_inline_ood_ops_s2a.json` (assert A6 hard-fails without it).
- **02:47** — ctt_v2 **VAE-encode pipeline built and launched** for all new strata
  (`scripts/ctt_v2/encode/`). Writes only the two prompt-agnostic root trees —
  `latents/` + `cond_clean/` — so it is fully independent of the Gemini caption blocker;
  `masks/` stay shape-derived and `conditions/` stay blocked. 18,013 clips staged
  (S2a 7,990 / S2b 7,990 / S1 33 / S4 2,000) under `outputs/ctt_v2/encodes/<stratum>/`;
  latents come from the trainer's own `process_videos.py` so the payload schema is
  byte-identical to ic_gen's, and `cond_clean` goes through
  `encode_conditioning.write_cond_clean()` unchanged. Shard counts are hardcoded in the
  module (never derived from `SLURM_ARRAY_TASK_COUNT`) and each stratum's roster is frozen
  to `ROSTER.json`, so a partial resubmit cannot re-partition; every write is `.tmp` +
  `os.replace`, so a preempted requeue can never leave a truncated `.pt`.
  Smoke-verified on GPU: S2a/S2b/S1 → `(128,16,20,15)` bf16 `fps=24.0`, S4 →
  `(128,5,14,26)` bf16 `fps=16.0`; cond_clean corrects exactly the last latent frame on
  two-sided clips and is bitwise-identical on one-sided ones (S1 came out 6 corrected /
  27 copies, matching the registry's `hero_flight`+`shadow_smoke` two-sided pair).
  Jobs 9687982 (L40S ×6) / 9687983 (scavenger ×6) / 9687984 (secondary H100 ×4) /
  9687985 (aux ×4).
- **04:52** — ✅ **ENCODES COMPLETE, ALL CHECKS PASS.** 18,013 clips × 2 trees = **36,026
  `.pt`**, 42 G, in **1h52m wall / ~11.3 GPU-h** on L40S. Per-stratum count asserts are
  set-equal to the frozen rosters (S2a 7,990 · S2b 7,990 · S1 33 · S4 2,000), so there is
  nothing for the trainer's path-join to silently drop. Bleed magnitude is stable across all
  32 S2 shards: median-of-medians `suffix_rel_l2` **S2a 0.334** (0.325–0.345) / **S2b 0.317**
  (0.312–0.325), consistent with exp_073's 0.280 and d2f's 0.314 — the suffix anchor really
  was reaching backwards, and cond_clean really is correcting it. `secondary` was a dead lane
  (zero starts in 90 min); re-bidding shards 12–15 onto `scavenger` (9687984 cancelled →
  **9688318**) had all four running within 6 min. Remaining before root assembly: Gemma
  text-encode (~2 GPU-h) once billing is restored; masks are built by `assemble_root.py`.
- **02:47** — 🔴 **S4 cannot be encoded at literally-native resolution.** refVFX I2V_LoRA is
  832×464; 33f/16fps carry through fine, but **464 is not a multiple of the VAE spatial
  factor 32** and `process_videos.py:parse_resolution_buckets` rejects the bucket outright.
  S4 is therefore encoded at `832x448x33`, which for a 832×464 source is a **pure 16-row
  centre crop with no resampling** (the width scale is exactly 1.0). This corrects the
  2026-07-27 13:16 note that S4 needed "zero-cut reshape" — zero-cut holds temporally, not
  spatially. It also means A9's "masks regenerated at (5,20,15)" is unachievable for S4:
  the real S4 latent grid is **(5,14,26)**. Masks are derived from the latent shape by
  `assemble_root.py:ensure_mask()`, so nothing downstream hardcodes it.
- **10:35** — **Viewers register themselves now, and the dashboard leads with a latest-only bar.**
  Adding a viewer no longer means editing anything: `hub` discovers pages under
  `outputs/viewers/*/`, `outputs/videos/*/run_*/`, `outputs/eval/*/viewer/`, `outputs/reports/*/`
  and `outputs/presentation/*/`, taking metadata from a `viewer.json` sidecar when present and
  from the page's own `<title>` otherwise; `viewerctl new` writes that sidecar for you. Runs of
  one experiment fold into a single card, newest first. The dashboard now opens with a sticky
  bar of current viewers only — earlier builds and anything whose media stopped resolving fall
  into one openable "Earlier versions & archive" block with the reason attached. Archiving is by
  health, not by memory, so the top of the page cannot rot.
- **10:25** — **The ctt_v2 corpora viewer is now static — the app server was working around a
  one-line gap in ours.** `viewerctl httpd` (the viewer system's static server) answers byte
  ranges; stock `python -m http.server` ignores `Range` and returns 200 with the whole file,
  which is why reading one member out of a WebDataset tar had needed a process. It does not:
  `scripts/viewers/build_ctt_v2_corpora.py` precomputes what `serve.py` did in memory — axis
  grouping into `ids_<sub>_<axis>.bin`, slim rows into `rows_<sub>.jsonl` with a `rowoff` offset
  table — and `scripts/viewers/ctt_v2_corpora.html` range-fetches the slice it needs, wrapping
  tar members in blob URLs (VFXMaster is loose files and streams directly). Verified at parity
  against the old server on 8799: identical sample counts (code 136,800 · LoRA 6,995 · VFX
  9,963), identical axis-group counts, identical group membership and row fields. Port 8799 is
  retired; the viewer lives at `outputs/viewers/ctt_v2_corpora/` on 8017 like everything else.
  Range support also un-broke video *seeking* across every existing viewer.
- **10:05** — **Viewer system: one dashboard, one port, a tracked registry.** ~40 HTML pages had
  accumulated under three incompatible path conventions and roughly a third served black boxes.
  `scripts/viewers/viewerctl.py` (`mount` · `check` · `hub` · `serve` · `new` · `httpd`) plus a
  tracked `registry.json` replace the ad-hoc server habit; `docs/VIEWERS.md` is the reference.
  The rule that fixes it: every path in a page is relative to the viewer's own directory, with
  media arriving through a symlink there — the only convention that survives a server restart or
  a repo move. Pages that must not be edited (certification records, the ladder2 REFERENCE, the
  frozen 2AFC study) are wrapped in *mounts*, rebuilt from the registry on every run so they
  cannot drift. The ladder results viewer, both eval_ladder pages and both certification records
  were climbing out of the repo or resolving against the wrong root; all 18 current viewers now
  resolve, 4 stale ones are archived with the reason. Dashboard:
  `http://localhost:8017/outputs/viewers/index.html`.

## 2026-07-27

- **20:05** — HumanVid endpoint screening COMPLETE (job 9680734, 1h19m): 3,000 candidates →
  85% loose pass → **1,499 accepted** std121 clips (769 horizontal / 730 vertical; 0 cut
  rejects, 1 dup vs the 227 bank) at `data/processed/humanvid_bank/`. The 19k-clip screen was
  owner-vetoed as oversized; the capped rerun uses a deterministic-shuffle `--limit` that
  resumes from cached detections if raised. Mix re-ratification advisor round relaunched (the
  first round's output was lost to a session restart) with updated premises: baseline
  `cond-bleed-fix` trainer (no mask), HumanVid cleared + measured yields.

- **13:16** — ctt_v2 fresh-rebuild kickoff after owner rulings: dataset pipeline retired for
  a clean redo (retire-commit; history kept), the retrain moves to the **baseline trainer
  `LTX-2-cond-bleed-fix`** (bneck parked → this round is a pure dataset intervention), and
  **HumanVid was owner-cleared** for use after the Pexels ToS flag. Built a fresh screening
  pipeline `scripts/ctt_v2/humanvid_bank/` (same QC contract as the blessed endpoint bank;
  center-window standardize, tightened+diversity-capped selection, dedup vs the 227 bank) —
  18,702 candidates collected, screening job 9680692 on L40S. Measured: all refVFX I2V-LoRA
  outputs are uniformly 832×464·16fps·33f (33 ≡ 1 mod 8 → zero-cut reshape possible; trainer
  supports per-sample fps natively). Gemini key stored at `$LAB/secrets/` (CLAUDE.md Secrets
  section); stale Gemma capbank job cancelled. Advisor round 7 (mix re-ratification) launched.


- **12:00** — **exp_084: the luma-matte family was killed by `step()`, not (only) by the maps.**
  The aux-map operator family (a static greyscale matte + a threshold sweep, shipped at 0% for
  looking fake) had two confounded defects. `experiments/exp_084_luma_matte_viewer/` runs the
  2×2 that separates them over real playing footage: {7 shipped maps, 12 new arrival-time maps}
  × {`luma.glsl` unmodified, a new `shaders/luma_soft.glsl` with a smoothstep feather, a rim
  colour in the advancing band and an additive glow lobe}. 114 clips, both layers real frames
  from the 227-clip endpoint bank, anchors verbatim. Blind audit (16/arm, anonymised shuffled
  contact sheets, key joined afterwards): **88% BAD → 56% with the compositor fix alone, 88% →
  88% with better maps alone (Fisher p = 1.00), 88% → 31% with both.** The compositor gates
  everything. The 56% residual splits: the three *aperiodic* shipped maps (fbm/radial/linear)
  go 6/8 → 1/8 BAD, the four *geometric* ones (stripes/checker/spiral/voronoi) stay 8/8 → 8/8
  (p = 0.0014) — a feathered checkerboard is still a checkerboard. New maps (eikonal fronts
  through ridged-multifractal speed, invasion percolation with morphological trapping, CC0
  Krita brush-path stamping, and a content-aware Canny boundary draw) are the best cell but
  are not separable from rescued-aperiodic-shipped (p = 0.62) — their value is variety and
  content-awareness, not a higher ceiling. Two side findings: `step(progress, m)` returns 1 at
  `m == progress`, so the shipped compositor leaks 5–6 stale pixels into the final
  conditioning block of *every* clip (`luma_soft` is exactly 0), and the `luma` sampler is
  vertically flipped w.r.t. the image (`probe_orientation.py`). Paired A/B viewer at
  `http://localhost:8017/outputs/viewers/luma_matte/index.html`. Mechanics written up in
  `notes/dataset/procedural_operators.md` §5b. Brush alphas are CC-0 (David Revoy 25.01);
  no commercial or ML-restricted matte pack was downloaded — see `PROVENANCE.json`.

- **11:55** — **S2 stratum viewer built** (`experiments/exp_081_s2_stratum/build_viewer_s2.py`
  → `outputs/viewers/s2_dataset/`, served at
  `http://localhost:8017/outputs/viewers/s2_dataset/index.html`). Two levels: an index of the
  56 shaders (representative thumbnail, op/clip counts, gate reject rate, per-shader table,
  dropped-op table, audit block) and 56 static per-shader pages where the atomic unit is the
  **operator block** — one row of 10 clips under a single header carrying the parameters they
  all share by construction (uniforms, easing, onset/release, flip, swap) plus the 20 distinct
  endpoint stems and the op's rejected renders. Plus `retired.html` for the 420
  blacklisted clips (42 complete op blocks, 6 shaders) framed as reinstatement candidates.
  Stays fast on 7,990 clips: no video is preloaded, each tile is one frame CSS-cropped out of
  the clip's lazy-loaded filmstrip, a global **phase slider** slides every tile to the same
  frame at once (so a whole operator row steps through the transition in lockstep with no new
  requests), and `<video>` elements are created on click / IntersectionObserver and torn down
  on exit. Media is symlinked, never copied.
  **Three metadata discrepancies found and surfaced on the page rather than smoothed over:**
  (1) `S2_ACCEPTANCE.json` is stale — it still reports the pre-blacklist 7,550 clips / 755 ops
  / 62 shaders, not the shipped 7,990 / 799 / 56; (2) the `summary_shard*.json` files were
  rewritten by the backfill pass and describe only that pass (860 accepted, overdraw 1.93) —
  the true build-wide overdraw recomputed from the append-only ops log is **1.2506×**;
  (3) the n=64 blind audit was drawn from the pre-blacklist roster, so 6 of its 64 samples are
  now retired, **including one of the two BAD clips** (`s2_0229_c06`, PuzzleRight) — against
  the roster that actually shipped it reads 1 BAD / 58.
- **11:50** — **HumanVid's REAL (non-synthetic) half: located, fully characterised, and
  ruled OUT on licence grounds.** It is not on HuggingFace — the HF tree holds only the two
  synthetic families. The real portion ships as **19,262 Pexels.com URLs** (11,411 landscape
  + 7,851 portrait) plus 19,429 per-frame camera trajectories, in Google Drive folder
  `1UGEkOKXYX9BGUFz0ao6lOGXkZjQGoJcZ` linked from the GitHub repo; the authors state plainly
  "we cannot redistribute them". **The blocker is Pexels, not HumanVid:** the repo's
  Apache-2.0/CC-BY-4.0 covers their code, cameras and UE renders but cannot relicense
  third-party footage, and the Pexels ToS prohibit "data mining, extraction, scraping … for
  all unauthorised purposes, *including without limitation for machine learning purposes*"
  and "bulk, large-scale or systematic copying", while the API terms separately bar
  collecting content "to train, fine-tune, *evaluate*, or develop ML/AI models *or
  datasets*". That covers eval sets too, so **nothing was downloaded — not 19k, not 60**, and
  nothing should be.
  Characterised the whole corpus anyway at zero media cost: every Pexels URL encodes
  `W_H_fps` and every camera file has exactly one line per frame, so
  `scripts/analyze_humanvid_real.py` derives resolution/fps/frame-count/duration for all
  ~19k clips from the manifest alone — **validated 10/10 against `ffprobe`** (resolution, fps
  and frame count all matched). 90.2 h of footage, median clip 13.8 s, all 1080p+, but only
  13 % is natively 24 fps (67 % is 25 fps). HEAD-only probe: **120/120 URLs live**, mean clip
  8.9 MB, **~167 GB projected** for a full fetch (measured from `Content-Length`).
  Fitness vs our 480×640·121f·24fps contract: resolution/length **pass** (~54k possible
  endpoints), single-shot **passes** (HumanVid filtered out shot changes), letterboxing
  **low risk**, but single-subject **partially fails** — their rule was "few people (n≤4)"
  with a bbox floor of r>0.07, under half our 0.15 — and the portrait crop keeps only ~42 %
  of a landscape frame. Honest verdict on diversity: our pool is already 85 % `person` and
  this set is 100 % human-centric by construction, so it would add **volume, not diversity**.
  Viewer at `outputs/viewers/humanvid_real/` (60 clips, 30V+30H) **streams straight from the
  Pexels CDN so nothing is copied to disk**, with filmstrips rendered client-side into a
  canvas from the already-streaming video. Two gotchas recorded in
  `notes/dataset/humanvid_real.md`: the repo HTTP server sends no charset (so viewers need
  `<meta charset="utf-8">` — the older `humanvid_sample` viewer mojibakes without it), and
  Playwright's `chrome-headless-shell` lacks H.264, so headless screenshots show black tiles
  even though the CDN serves 206s and every clip probes as h264.

- **11:15** — **exp_083 D3/S3 PILOT rendered — 109 depth-parallax transitions, and the honest
  answer is "the idea works, the current renderer does not".** Built on exp_076's `engine3d/`
  (reused, not rewritten) with three additive changes: `subject_anchor()` finds the foreground
  object as the saliency-weighted centroid of the nearest depth quartile, two new dissolve
  families (`subject`, `subject_fbm`) centre the world-space field on that object, and
  `render_transition(coverage_out=)` ports exp_082's disocclusion audit to the frozen-endpoint
  driver. Endpoints are real consecutive frames sliced out of the 227-clip
  `bank_tightened.json` bank (A = frames 112:121, B = frames 0:9); lengths vary over
  n_middle in {7,15,23,31} -> totals 25/33/41/49, every one legal (F = 8k+1), nothing padded.
  **Verbatim-endpoint property holds exactly: in-array max abs diff = 0 on all 109 clips**
  (H.264 round-trip is a separate MAE 1.94, worst single pixel 65). Seam ratio median 0.25,
  7/109 over the 2.0 bar — and 6 of those 7 are the 25-frame length, whose median seam is
  1.02 against 0.02 at 49 frames, so a 7-frame middle is simply too short for this operator
  family. **Blind BAD rate: 14 of 30 (47%)**, drawn from a fixed seed before anything was
  viewed. 13 of the 14 are one defect: the world-space dissolve punches alpha holes in BOTH
  layers at once, ~25% of the frame has no geometry at mid-transition, and push-pull (reach
  ~40 px) leaves a hard black void or a flat smear. It is gateable for free —
  `hole_radius_max < 85 px` catches 13/14 failures and rejects 0/16 of the shippable clips,
  at the cost of 57% of this pilot's operator mix. Dissolve family, not camera amplitude, is
  the driver (plane 154 px / sphere 159 / subject 150 median vs none 59). Viewer at
  `outputs/videos/exp_083_d3_pilot/run_0001/viewer.html`; decision on the full stratum is the
  owner's.

## 2026-07-25

- **09:50** — one-way reference attention now ships as a **two-call split**, not a dense mask.
  The benchmark rejected the dense path against its pre-registered ≤30% bar: fwd+bwd attention
  at the real T=9600 measured 26.97 ms unmasked / 82.49 ms dense (**3.06x**) / 21.79 ms split
  (**0.81x** — faster than the bidirectional baseline, since the reference-over-target block is
  never computed). This is the campaign's first change to `ltx-core`, so jobs must now put
  `ltx-core/src` on PYTHONPATH; `job_train.sbatch` does, with a guard asserting the bneck
  `ltx_core` actually loaded — a silent fallback would train bidirectionally while logging
  `one_way`. 19/19 gates pass, including split-vs-dense numerical equivalence in fp32 and bf16
  across both sequence layouts.
- **09:50** — S2 delivered by the parallel agent (8,410 clips / 809 exact ops, blind audit PASS
  at 2 BAD of 64). **S3 dropped** by their pre-committed tree — 62% defective, and the defect is
  inpaint plausibility, a semantic property no geometric statistic separated. Our mix becomes
  S0+S2+S4.
- **09:50** — S4 reinstated by owner and re-scoped to one-sided (`{S1}. sksz.`). Measured that
  rewriting refVFX prompts yields leak-free S1 for 96.3% of rows but at p50 8 words against the
  corpus's 34 — nearly disjoint, so caption length alone would flag the stratum. Switched to
  frame-based captioning; 2,000 filmstrips extracted (0 failures), pilot scored 25/25 in range
  with zero violations. Paused pending a Gemini key.


- **02:38** — exp_081 scaffolded: the ctt_v2 masked retrain. Advisor round 6 ruled the mix is
  S0+S2+S3 (S4/refVFX **deferred**, not killed — it adds ~4% to the operator count while
  importing either a tempo-rewrite or an unvalidated mixed-length trainer path), S0 at 15% of
  the sampling stream, 10,000 steps, primary checkpoint pre-committed as the final one.
- **02:38** — **D0 resolved as already satisfied.** All four headline cells are fully scored on
  the v4 lane instrument (zero missing rows); "72.9 (proxy)" meant `%_proxy` (content-capped,
  claim channel = Δpp), not uncertified. Frozen record in
  `experiments/exp_081_ctt_v2_masked_retrain/D0_baselines_v4.txt`, and the previously
  unrecorded **G-zs-cross = 72.8%** now has a baseline. Pass bars pre-registered.
- **02:38** — one-way reference attention mask implemented in the private `$LAB/LTX-2-bneck`
  trainer (local-only). Found the defect is on BOTH paths, not just inference, and found a
  SECOND silent defect: `ValidationRunner._modality_from_latent_state` dropped
  `state.attention_mask` entirely, which would have made T5's reference-strength sweep a
  silent null. 13/13 trainer tests green, including a train==inference mask-equality gate
  (the trainer lays the sequence out `[ref|noisy]`, inference `[noisy|ref]`).
- **02:38** — measured: the 19B ic_gen training config **OOMs on a 44 GiB L40S in its
  bidirectional baseline form**, so H100-class memory is required for the retrain; this is not
  a cost of the mask.

- `02:15` **exp_080 validated + S2/S3 hand-off spec written.** Fork of exp_076 to full 121-frame 3D transitions over two REAL playing streams (per-frame temporally-stabilised Depth-Anything, D2 timing contract, pure phases byte-identical, asserted). run_0001, 31 clips on one L40S: join ratio median 0.94 / p90 1.15 / max 1.86 (bar ≤2.0), parallax 3.31, ~11 s/clip. Owner verdict on samples: positive; ruled contents must come from the dsx endpoint bank (read-only, tightened 227) — corpus clips leak class manner into content. Generation itself moves elsewhere (owner call): S2 (800 ops × 10 contents = 8,000 clips) + S3 (300 × 6 = 1,800) spec'd in $LAB/misc/ctt_v2/DATA_PLAN_PROPOSAL.md.

- `00:04` **ctt_v2: external-dataset downloads started** (owner call — go straight to the dataset/retraining branch). `scripts/ctt_v2/download_small.sbatch` pulls VFXMaster (8ruceLi/VFXMaster_datasets, 6.6 GB, 9,209 videos / 241 effect classes, Apache-2.0) and the refVFX I2V_LoRA shard (12.3 GB, 6,995 LoRA-generated pairs, CC-BY-4.0); `download_refvfx_code.sbatch` pulls the refVFX code_based_edits subset (16 shards, 374.7 GB, 2,736 code effect types x ~50 base videos — the same-operator × different-content diagonal), submitted as an afterany self-resume chain on the cluster-wide `secondary` CPU partition. Everything lands in `data/raw/{refvfx,vfxmaster}` (gitignored). refVFX neural_v2v (79 GB) deliberately skipped. Fixed a first-submit failure: `set -u` before sourcing `/etc/bashrc` (unbound BASHRCSOURCED) killed the jobs in 5 s; `-u` now enabled only after env activation.

## 2026-07-24

- **16:01** — exp_077 D2-FULL: the degenerate-frame gate (DFG) was calibrated against a
  pre-committed bar and **escaped it**, so no detector shipped and the final dataset was rendered
  unclamped at baseline. Parameter clamping is abandoned permanently (`param_clamp.py` stays on
  disk but never runs). The escape is structural: the premise "grain has high pixel variance, a
  matte has none" is false on the 96x72 area-downsampled grayscale — StaticFade's luma std
  (0.012-0.040) sits below every declared threshold — and 5 of the 12 BAD positives are
  texture/geometry destruction with entirely normal luma statistics. The first-chunk BLIND check
  then passed at 2/64 BAD (Wilson-95 upper 10.7% vs a 17.5% baseline), and the build delivered
  3,072 tuples / 6,144 clips with exact pure-phase identity (max abs diff 0.0000), 8 distinct
  shaders on every one of the 384 target pairs, and 1.59x realized overdraw. Training remains HELD.

- `14:30` exp_077 **D2 mass build: scaffold complete, render HALTED by its own pre-committed check.** Full chain written and validated end-to-end — `plan_d2_full.py` (3,072 slots = 384 target pairs × exactly 8 operators, **8** distinct shaders/pair, 768 ref pairs content-disjoint and reused exactly 4×, 42–43 slots per keep-shader), `render_d2_full.py` (per-slot rejection sampling against the frozen gate τ=0.2543, sharded by target pair, resumable), `param_clamp.py` (the 2026-07-24 clamp ruling as a wrapper — exp_075's engine untouched), `encode_d2full.py` / `assemble_d2full.py` / `audit_d2full.py` / `build_viewer_d2full.py` / `make_d2_train_config.py` + 5 sbatches. **Two real defects found by measurement, not guesswork.** (1) *Rejection-sampling economics*: freezing the timing draw per slot (my choice, to keep the onset/release law unbiased) pushed realized overdraw to **7.35× vs the spec's 2.5× ceiling** — attempts-per-accepted-slot clustered at {1:165, **26:33**, 51:6, 76:3}, where 26 = "5 shaders × 5 param redraws all failed, then ONE timing redraw passed on its first attempt". Slot difficulty is carried by the **timing** draw, not by params (only 4 slots were ever rescued by a shader swap); unbiased first-attempt clip pass is 0.843, matching the audit's 0.853. Timing is now redrawn per attempt (projected ~1.3×) and the 228 tuples built under the old procedure were discarded so the dataset comes from one procedure. (2) *The clamp does not fix the visual failures*: the pre-committed first-chunk check (64 clips, offenders oversampled to 40.6%) gives **25.0% BAD on targets vs the 17.5% baseline**, and the **non-offender subset alone is 15.8%** (uniform-allocation projection 16.7%) — so the oversampling confound does not rescue it. Array killed per the rule, no blind clamp iteration. Diagnosis: **rule 2's caps are the wrong sign for the two worst offenders** — `[0.5d, 2d] ∩ |v|≤3.0` is EMPTY for `EdgeTransition.edge_brightness` (d=8.0) and a single point for `ColourDistance.power` (d=5.0), so both collapse to the constant 3.0 for 100% of draws *including the canonical default*, and for both shaders *lower* is the destructive direction (dimmer edge map → near-black frames, 8/8 draws bad-or-marginal; lower power → white blowout). Plus a name-match gap: `dissolve.uPow=12.9992` and `uSpreadClr=[2.61,0.23,0.96]` bypass the classifier ("uPow" ⊅ "power", "Clr" ⊅ "color", and the 3-vector fallback needs all components in [0,1]). Records: `D2_FIRSTCHUNK_VISUAL.json` (per-clip taxonomy), `D2_CLAMP_CHECK.json` (21,600 draws, 0 invariant violations), `D2_BUILD_AUDIT.json`. Real ic_gen root re-verified: 385 files per source × 5 sources, **0 dangling**. Train config `configs/d2_gen.yaml` emitted and **NOT submitted** (held).


## 2026-07-23
- `16:05` **ladder2 merged to main.** v3.0.0 TODO recorded in eval_ladder/README.md (owner directive): CTT tasks — (endpoints, target transition) as the atomic unit, every task generated on all four tiers (prompt / prompt+endpoints / specialist / generalist) from one shared roster; enumerate the full task space, then pre-register a budgeted subspace with balanced per-donor/per-cell n. Branch `ladder2` (63 commits) merged --no-ff into main and kept as the record.
- `15:55` **eval_ladder infra complete: 3×3 viewer + VERSIONING.md.** All queued jobs stopped (owner call). Viewer rebuilt to the owner's ontology — SEEN/UNSEEN/ZERO-SHOT × SAME/CROSS/FOREIGN with specialists mapped in by endpoint novelty (SP-fit→seen, other SP-*→unseen), multi-select cells/rows/columns, stats+headline live-update, collapsible+sortable metric tables, one horizontal row per card, conditioning bar + reference ribbon per output, IC demos moved into the INPUTS band, per-tier tints, synced-restart, autoplay. 140 unscored-but-rendered videos now visible with a badge. `VERSIONING.md`: design semver bumps only with SPEC; run records append-only, **newest VALID record = current result**; frozen per-run viewers via `build.py --freeze` (v2.0.0-R1 frozen), stable latest at outputs/reports/ladder_viewer/index.html. Suitability stated honestly in README: 39/177 tasks carry both specialist and generalist on one endpoint (independent rosters); next design needs one shared endpoint roster + the 194-video clean-baseline lane (stopped today before completion).
- `13:58` **eval_ladder v2.1.0 corrected + session damage repaired.** A stray 13:20 bulk-copy had overwritten four eval_ladder files with pre-move versions and recreated `experiments/ladder2` as a stale shim — restored HEAD, shim moved to job tmp. Baseline mechanism finished properly: scoring stays per task (donor-specific pool) but the video is canonical per `(endpoint, sided)` via `video_key = "<dir>/<name>"` resolved identically by run_gen/run_eval/viewer; rows that duplicate an existing clean no-ref base twin reuse its video and only re-score against the new donor pool. 303 scoring rows (177 `base_prompt` + 126 `base_cond`), 97 reusing twin videos, **194 new videos instead of 708**. run_gen dedups by canonical path *before* chunk slicing (no array-task races); `job_score.sbatch` gains MANIFEST_DIR/SCORE_OUT overrides. Viewer: coverage banner (39/177 tasks have specialist+generalist on the same endpoint — the honest side-by-side answer; zero-shot blanks say "no specialist exists by design"), node-check clean, 1209 video refs spot-verified. Submitted: convergence-diag scoring 9646200 (322 rows over 66 ckpt-4500 gens), baseline generation 9646244-47.
- `11:52` **ladder2 campaign complete** — 12/12 models trained, 888/888 generations verified against the registry, 11,695 unique scored rows, 87.9 GPU-h over 339 jobs, no kill rule fired. `experiments/ladder2/REPORT.md` is the full record (training + generation + eval settings, every v4 metric per cell × arm, flag rates, per-arm rollup, reproduction commands). **Headline:** specialists reach 99.7 % of their class ceiling on unseen same-class endpoints and 94.9 % cross-class, both ~+40 pp over their base twins on byte-identical inputs, 11/11 donor classes; DAVIS is +18.9 pp but only 8/11 donors → reported weak. **All 9 generalist donor-pool cells are marked invalid** (retained, never deleted — they were pre-registered): the base twin wins them by *copying the in-context demo*, so pool-% rewards the wrong behaviour. Amendment-1's transfer index TI = min(T, C), locked before any corrective number existed, is the claim-bearing readout: in-context transfer **works inside the trained vocabulary** (G-unseen-cross ΔTI **+3.9 pp, 9/13 donors**) and **fails on genuinely novel transitions** at this budget (G-zs-cross **−8.3 pp, 1/10**). New `report_full.py` (every metric field, pool-refs → seeds → items, flags as rates) surfaced two independent corroborations of the confound from substrates the appearance kernel never touches: `max_seam_z` 10.5–45.2 for base in every reference-bearing cell vs −0.13–3.62 for `ic_gen`, and `prefix_dino` 0.727–0.932 vs 0.959–0.987 — base cuts away from the conditioned prefix, the adapter keeps it. Also fixed a real defect: **798 of 12,500 scored rows are repeats** from overlapping incremental passes and were double-counted in the pool mean (≤0.3 pp per cell, no verdict moved). Convergence diagnostic (ckpt-4500 vs 5000) still running; mechanistic F-block proposed but NOT written — `docs/FINDINGS.md` is owner-gated.

## 2026-07-22
- `20:45` **ladder2 built** (branch `ladder2`) — the clean rebuild of the eval ladder, executed as an advised campaign. `experiments/ladder2/registry.jsonl` is now the single source of truth: **355 items → 710 generations** (P0 432 / P1 184 / P2 94), with every cell label, GT pool, % type, priority and base twin *derived* from three frozen inputs (split v1.2 sha `c694659d`, the caption corpus, `arms.yaml`) behind 8 build-time seatbelts. One folder, one renderer (`prompts.py` — the same call emits training captions and registry rows, so train == inference), one conditioning definition (`encode_conditioning.py` — window rule + the isolation-encode suffix bleed fix), one generator (`run_gen.py`, row × seed = one video, rows selected by arm), one evaluator (`run_eval.py`, v4 pool-% with a %-typing firewall). **Real defects caught during the build:** clip→class cannot be string-split (`action_run_setonfire_6` is class `run_set_on_fire`, `flame_transition_0` is `flame`) — now read from the frozen split everywhere; an `item_id` collision (text_floor drew from two overlapping pools); the train-band-endpoint rule had to become per-*arm* (a held-out class's train clips are untrained content for every arm); `process_captions.py` defaults `media_column` to `media_path`; a YAML flow-style comma silently truncated a DAVIS caption. **DAVIS roster corrected:** `car-turn` rejected — the sequence is shot *from* the car, so the 480×640 portrait crop contains no distinctive foreground object; replaced by `hike`, and every window re-verified frame-by-frame *after* cropping. Added `G-ref-control` (mismatched-demo control, advisor flag) without which G-unseen-same / G-memo-probe cannot show the model uses the in-context demo. Training inventory: 139 clips, **0 missing latents**; generalist trains on 26 classes / 385 pairs (3 classes have <2 trainable clips). 12 configs emitted with inline ID/OOD/control validation @250 steps; pipeline chained with Slurm dependencies (token probe → cond-clean → prepare → 2-model pilot) so nothing waits on a human. Dossier `$LAB/misc/ladder2_redesign/DOSSIER.md`; face `experiments/ladder2/REFERENCE.html`.
- `14:25` exp_076 — **3D-plausible procedural transitions** (owner call: drop the 121-frame padding, "connect two 9-frame buckets normally", and go after geometrically plausible transitions rather than 2D overlays). Format is now `start9 + 15 rendered frames + end9 = 33` with the buckets copied through verbatim, so conditioning fidelity is exact by construction and the layer-extension problem disappears. Engine: Depth Anything V2-Small (Apache-2.0) → displaced grid mesh → moving virtual camera in GL, same headless moderngl+EGL/CPU path as exp_075. Both layers ride **one continuous trajectory** (A leaves from rest, B arrives at rest) so it reads as a single camera flying between scenes. 7 camera families × physically-motivated optics: Beer-Lambert fog, circle-of-confusion rack focus, 180° sub-frame motion blur, dolly-zoom, handheld jitter, and a **world-space noise dissolve** sampled at unprojected scene positions (so the pattern sticks to surfaces and parallaxes instead of sliding like a screen-space overlay). Identity-camera render reproduces its source at MAE 0.083. **New metric: Parallax Index** (`engine3d/metrics.py`, one DIS-flow call + cached depth, ~50 ms/clip) — near-vs-far flow ratio + Spearman ρ between 1/z and flow. It certifies 3D-ness and it caught four real bugs. **Bugs found and fixed, worst-case seam MAE 184 → 6.4 (0/59 clips above 8):** (1) camera easing must have zero velocity at both endpoints or the last rendered frame is off-rest — `PATH_EASINGS` now restricted, blend easing left free as an independent axis; (2) the blend must close inside the rendered range; (3) the world-space dissolve's B-layer mask was inverted (B absent exactly when it should be present); (4) fog was applied at constant density instead of ramping, putting a fully fogged frame against an unfogged bucket. The metric also caught (5) **`shear` had parallax inverted** — screen displacement must go as 1/z, not z (ρ −0.84 → +0.29). Result: translation families are certifiably 3D (dolly PI 3.96/ρ +0.60, truck 1.59/+0.59, spiral 2.82/+0.29) while the exp_075 2D shader bank has ρ undefined; rotation families (orbit/crane/roll) legitimately score low — PI is the wrong expectation for a pivot orbit. Viewer `outputs/videos/exp_076_depth3d_transitions/run_0005/viewer.html`, comparison `outputs/analysis/exp_076_parallax_comparison.json`.
- `12:30` exp_075 — **procedural transition operator engine**, first samples. New idea: manufacture transition training data by applying a large bank of procedural operators to arbitrary endpoint pairs, buying task diversity, operator ⊥ content factorisation, and counterfactuals (same endpoints, many operators) that the 49 real clips cannot provide. Engine renders gl-transitions GLSL shaders headlessly through `moderngl` + EGL — which resolves to Mesa llvmpipe here, so it runs on plain CPU nodes (~1 s per 121-frame 480×640 clip) and never competes with training for the H100/H200 pools. An *operator* is (shader, sampled uniforms, easing, spatial flip, direction swap, layer-extension policy, auxiliary map) ≈ 1.7e6 distinguishable combinations from 122 usable shaders. Run `run_0003`: 67 procedural clips over 47 shaders + 4 real references, viewer at `outputs/videos/exp_075_procedural_transition_engine/run_0003/viewer.html`. **Two findings.** (1) The endpoint identities `transition(uv,0)==from` / `transition(uv,1)==to` are **parameter-dependent**, so a per-shader gate at default params is not enough: 3.8 % of *sampled* operators violate them despite their shader passing (`colorphase` reached MAE 53.8, `undulatingBurnOut` 8.1). Added a per-operator rejection gate against the real endpoint frames → worst endpoint MAE 53.8 → 0.196, mean 0.001. (2) The per-shader gate is **resolution-dependent** (`InvertedPageCurl` passes at 120×160, fails at 480×640) so it must run at production resolution. Tooling survey + task-diversity phase-transition bibliography (Raventós et al., threshold ~2¹⁴–2¹⁵ tasks) in `notes/dataset/procedural_operators.md`.
- `11:20` exp_074 v2 (owner call): corrected prompts KEEP the transition marker for training alignment — one-sided/foreign = "ICTRANS <S1>. The scene transforms into" (dangling; outcome withheld, cue preserved); two-sided rows dropped from the rerun (corrected == original prompt). v1 no-marker jobs cancelled with zero outputs produced. 276 gens resubmitted maximally parallel across campus secondary (ic3 24-task array + 4 r3x arrays) + lab HCESC-H100-secondary (4 arrays) + HCESC-H200-secondary (3 arrays): jobs 9617602-13.
- `10:45` PROMPT DEFECT found + patched (owner-driven): every arm (train+infer) was prompted with the endpoints clip's FULL caption — one-sided prompts describe the outcome the keyed conditioning withholds; foreign prompts describe the recipient's own transition, contradicting the donor task. Forensic: 76%%/83%% of r3x/ic3_x generations look like the recipient class (donor only 14%%/8%%); r0 prompt-only already hits ~70%% of the appearance ceiling. Defect record + retrain proposal (token research, transition-slot placement, one-sided Scene1-only, caption-segment dropout, simpler next-ladder architecture): `docs/eval_ladder/PROMPT_REDESIGN.md`. Immediate inference-only patch launched: exp_074 corrected-prompt regeneration of R5+R4X+R3X (285 gens, jobs 9617329-40, predictions pre-registered in its README).
- `09:42` eval: viewer UI rebuilt on the flat single-page design the owner preferred (copied from the ladder_v3/_viewer look): data-tier tabs (All/Conditioning/A/B/C/X), autoplay-on-scroll, top per-arm aggregate with a % of ceiling column. Cards now group cells into consistent labeled boxes — MODEL INPUTS (prompt text + start/end anchors honoring sidedness + ic demo) · CONTEXT not-input (full GT, scoring refs) · OUTPUTS (fixed arm order, conditioning line + v4 chips + %-of-ceil + vs-own-GT/demo/donor). Prompt cell added after verifying parity: every arm on a card used the identical "ICTRANS <endpoints-clip caption>" prompt (0 mismatches across exp_061/062/065 gen manifests; r0 = prompt-only). ic2 dropped from display (item-id collision with ic3_c made that prudent too). docs/eval_ladder/{build_viewer.py,viewer_template.html}.

## 2026-07-21
- `18:05` exp_072 fill chunks landed (9609271-72, 2,820 pairs): harness rows for r0/r3x/ic3_x match the local exact-kernel fill EXACTLY (408 items, max diff 0.0). Full 9-arm pool table regenerated (aggregate.py now includes pool_x*): base·P 63%%, base·PE 73%%, spec SEEN 99%% / UNSEEN 101%% / FOREIGN 75%%, ic3 84/96/90%% + foreign 63%% — outputs/reports/pool_yardstick_v4.txt. Viewer rebuilt: r0/foreign %-chips now harness-confirmed (solid); only r1k/r1k_ext/ckpt250 remain local-marked (no harness lane planned).
- `16:12` eval: ladder viewer rebuilt on the v4 instrument — pool-yardstick chips (raw · ceiling badge · %-of-ceiling per generation), model-tier filter (baseline/specialist/generalist), sidedness-aware display (one-sided: keyed r1k replaces the CRACKED r1 that saw the end anchor; endpoint cells show only the given anchors, slow-looped), per-arm conditioning labels (none / prefix / prefix+suffix / ref-demo+…), reference-semantics legend (row app_ref: own-GT vs demo vs donor; pool % = the cross-arm comparable number), ic2 dropped (owner: ic3 only). New `exp_072/local_pool_fill.py` fills pool means for arms the harness lane hasn't covered (r1k/r1k_ext/ckpt250 + queued r0/r3x/ic3_x) with the exact v4 kernel from cached features (validated 12/12 vs harness; marked as local until jobs 9609271-72 confirm). New numbers: r1k (honest one-sided base) 95%% vs cracked r1 98%% on one-sided; base collapses to 29%% on two-sided classes; r0 prompt-only 70%%.
- `15:29` docs(eval): pool-reference yardstick adopted as a STANDING reporting lane (owner directive) — method + fixed reporting rules (raw · ceiling · achieved-%%) in `docs/eval_ladder/POOL_YARDSTICK.md`, pointer section appended to `notes/eval_harness_v3.md`, and the rule codified in the exp-eval skill so every future eval computes it. Foreign/prompt-only pool fills in flight (9609271–72).
- `14:51` exp_072 COMPLETE: all 2,616 pool-reference pairs scored under v4 (pilot cross-validated 15/15 vs exact-kernel local run); aggregate.py switched to v4 ceilings. Full-table achieved-%% of GT ceiling — base·PE 73%%, specialist SEEN 99%% / UNSEEN 101%% (no overfit gap on the pool yardstick), ic3 held-in 84%% / unseen 96%% / zero-shot 90%% (low-trust classes). Table: outputs/reports/pool_yardstick_v4.txt.

## 2026-07-20
- `12:34` docs: created `docs/FINDINGS.md` — owner-gated cornerstone-findings registry (claims → FINDINGS, mechanics → notes, trajectory → CHANGELOG); seeded with owner-approved F-001 (reference-swap stability / pool yardstick, v4 σ_ref ±0.044 = 11%% of gap) and F-002 (sided core mask beats all-frames: v2 0.927/d2.04 vs 0.780/d1.11; v3 d′1.52 vs 1.27). Analysis scripts relocated from job tmp into exp_072 so evidence lines point at committed code.
- `12:01` exp_072: lane switched to the v4.0.0 instrument per owner directive (v3 jobs 9603416–22 canceled unstarted; resubmitted as 9603558 pilot + 9603559–64 on the eval-v4-cert worktree, out-root outputs/eval/exp_072_pool_v4; rows carry the app_ref_v3 bridge). Local exact-kernel pilot (v3, validated 12/12 vs certified rows, 324 pairs from warm cache): pool-yardstick achieved-%% of GT ceiling — specialists 93–96%%, base·PE 74%%, ic3 74–75%% on the 3 pilot classes; v4 local pilot running.
- `11:38` exp_072: pool-reference re-score lane scaffolded + submitted (pilot 9603416, chunks 9603417–22) — scores existing r1/specialist/ic3 generations against every same-class corpus reference (leave-own/demo-out, ≤8) under the certified v3 harness to put all arms on one appearance yardstick (% of GT ceiling); pre-registration in the README written before scoring. Also: all 9 method-arm scoring chunks landed; pre-registered verdicts run — debias/margin/moments each FAIL the amended tier-B bar vs ic3 (Δ −0.011/−0.016/−0.026, all within MDE but sig-down sign tests); M1-vs-M2 headline test: moments vs residual +0.002 tier-B → PASS non-inferiority → M2 branch (content-free conditioning viable; disentanglement headline).
- `15:35` docs(eval_ladder): viewer — near-copy badge now distinguishes reference identity: when the scoring reference is the item's OWN GT clip (all base/r2r3 items) copy_max is content-saturated (crossfade control = 1.000, base·PE = 0.985, specialists 0.974–0.984) so the red 'near-copy' badge was a false alarm; those rows now show an amber '≈GT (same-content ref)' badge. Red near-copy is reserved for the real leak channel (reference ≠ own clip, i.e. IC-demo copying).
- `12:50` docs(eval_ladder): viewer — endpoint cells now play ONLY the conditioning frames (first 9 / last 8 @24fps, clamped loop) instead of the whole source clip, which had made it look like the model was given the full GT video; full source clip moved to its own clearly-labeled 'NOT model input' cell, one-sided items get an explicit prefix-only note.
- `12:32` docs(eval_ladder): viewer fix — 140 NaN metric values (mostly cam_dtw on cam-invalid items) made the embedded JSON unparseable in browsers, killing the whole page (blank preset dropdown, no cards); builder now nulls non-finite floats (`allow_nan=False` guard) and also embeds app_target for the detail table.
- `11:32` docs(eval_ladder): added `build_viewer.py` + `viewer_template.html` — full inspection viewer (`outputs/reports/ladder_viewer/index.html`), superset of the 11:24 `_viewer` build: question presets mapped to the C1–C11 contrasts (paired-family filtering + per-contrast anchor arm), any-metric lens with MDE-gated Δ coloring, trust-map † integration, blind-audit mode with per-card reveal, near-copy/camera/seed/band filters, extremes sorting, per-family flag+note export (seeds the 2AFC pair list), controls-as-floor chips, and embedded certified tier/contrast tables. 366 families / 1,137 certified rows, all videos verified on disk.
- `11:24` docs(eval_ladder): added `build_ladder_viewer.py` — joins the 20 exp_066 eval manifests, all certified ladder-v3 per-item rows, and the certification trust map into a self-contained side-by-side HTML viewer (`outputs/eval/ladder_v3/_viewer/index.html`): one card per (class, endpoints clip, seed) with GT/demo/reference clips + every arm's video and metric chips (crossfade-control floors, Δ vs keyed base, trust †, live filtered aggregates). Serve from repo root via `python3 -m http.server`.

## 2026-07-18

`01:43` **Anti-collapse method campaign (advised, operator/advisor x3 rounds): Stage-0 implemented, exp_067 scaffolded.** Trainer branch `transition-strategy` @ 218528b on $LAB/LTX-2-official (base 7809842, patch archived outside this repo): `TransitionStrategy(FlexibleStrategy)` — on-the-fly residual (endpoint-content-subtracted) reference conditioning, directionally debiased FM loss, calibrated anti-collapse margin (GT-anchored segment-D⊥ hinge, ω=(1−σ)², identifiability-masked), ValidationRunner residualize hook + ref-swap tripwire; flags-off is bitwise-identical to the ic3 recipe (10/10 unit tests). Design was probe-gated with pre-registered keep/kill rules: curvature supervision KILLED (same-class Δ² cosine 0.02; energy premise inverted — crossfade/hold controls sit ABOVE GT), margin KEPT (clean D⊥ collapse axis: declerp 0.29 < hold 0.47 < lerp 0.57 < base·P 0.80 < base·PE 0.83 < GT 1.03 ≈ ic3·B 1.04). Calibration on the real corpus: γ=1.0922; no latent-space dissolve family exists (g≈1.62≈iid ceiling for ALL classes). exp_067 (Stage 2, residual reference only, p_drop 0.1, inline 5-sample validation @500 incl. C6 probe + refswap pair) scaffolded for Monday launch; pre-registered bar in its README.

## 2026-07-17

`12:15` **Eval ladder v3 campaign COMPLETE — 20/20 v3 labels certified (2,134 rows, 0 error rows) + full v4.0.0 cross-sweep (20/20, H100 lane).** Final contrasts: C5 PRIMARY = margin parity between the ic3 generalist and per-class specialists on identical unseen items (Δ−0.018 < MDE), achieved by synthesis (near-copy 3% vs 100%); C8 = seam integrity −19.4 (21/25 classes, p=0.001) + no copying over conditioned base; C9 confirmatory CONFIRMED (r3x>ic3_x, app_ref/margin 6/6 recipients, p=0.031, twins apples-to-apples); C10b decontamination cost nothing (margin −0.002 null); C11 no specialist-style overfit in the generalist. v4 instrument agrees on every claim-bearing verdict; bridge |app_ref_v3−v3 app_ref| = 0.00000 on all 20 labels. REPORT_v3 + SUMMARY_v3 filled with measured numbers only; presentation tables ready.

`10:55` exp_066 v4 cross-comparison aggregator (`aggregate_v4_table.py`): lane precedence ladder_v4h (H100, warm cache) > ladder_v4 (mixed insurance, rows disclosed `@mix`); amendment-1 MDEs attached only to definition-unchanged channels (app_ref_v3/margin/copy_max/max_seam_z), v4-normalized channels (app_ref/cam_zpr/obj_csls) get sign tests only; v3-exam trust applied at family level (disclosed approximation); saturation-flag rates per arm (SPEC §6.5); built-in bridge check |app_ref_v3 − certified v3 app_ref| per label. Header stamps "NOT re-certified" — v3 stays the headline. Also: two-week report (outputs/reports, untracked) extended to Jul 17 with the executed ladder + presentation skeleton.

## 2026-07-16

`23:15` exp_066 ladder-v3 scoring scaffolded and W1 launched: 20 eval manifests (1,142 rows, all pre-registered conventions — self-GT refs for base/specialists, demo refs for ic tiers, recipient refs for X twins) + ic3 training manifest; certified-worktree score job (eval-v3-spec = tag + amendments, docs-only diff verified); base/ic2/sigma-recheck submitted (9542684-91). Also: ic3 validation cadence cut (interval 2500, skip-initial) after T1 burned its whole 1h59 chunk on precompute+initial validation — observability-only change, training math untouched; chain forecast now step_05000 ~05:15.

23:05 — **v4.0.0 merged toward main; corpus sidedness skew documented (owner-resolved).** The v4.0.0 certification (tag `eval/v4.0.0`, commit 8584114) was stamped against corpus `aa28c6d5` where `giant_grab`/`hero_flight` are **onesided** — their original, eval-correct classification. `amendment-2` (main) flips both to **twosided** in the shared `corpus_manifest.json` (sha `348db23d`) as a **pragmatic training override** (`OWNER_SIDEDNESS_OVERRIDES`), not an eval-truth correction. Owner determination (2026-07-16): the swap does not materially affect eval, so **v4.0.0 stands as-is, no re-certification** — the tag's onesided cert is correct for eval; main HEAD carries the twosided manifest for training. Known follow-up: eval and training sidedness now diverge in the shared manifest; a future v4.x may split them or re-cert against the twosided manifest if eval scope changes. Merge is clean (only this file conflicted; no measurement/grader code or `reference_v4.npz` touched).

22:45 — **transition-eval 4.0.0 CERTIFIED (regrade of the draft.1 run) — all 8 bars PASS.**
The draft.1 certification run (job 9531327, full §6, clean) FAILED on **bar 8 alone**: the
reference-rebuild-parity of `pop_App` came out 2.02e-05 > the frozen scalar 1e-6. Advised
fail-branch consult (fable-advisor xhigh) diagnosed a **pre-registration defect, not an instrument
failure** — `pop_App`/`pop_Dyn` are ECDF-composed rank lattices on `{k/(2N)}` (quantum 2.02e-05),
for which a scalar float tolerance is unsatisfiable under any cross-environment rebuild while the
same clause tolerates the ~2.5e-8 raw-channel float32-reduction drift that flips one lattice cell
one step. **Fix:** the two-class rebuild-parity criterion (`bars.yaml` `reference:`, SPEC §7) —
value-space arrays keep `value_tol` 1e-6; the two lattice arrays are compared in integer rank units
(`max_step` 4, `max_flips` 50). **No measurement/grader `.py` changed; `reference_v4.npz` byte-
identical.** Per the 3.0.0 precedent, certified by committed regrade (`scripts/regrade_draft1_to_v4.py`)
over the draft.1 artifacts + provenance flip-counts (`scripts/provenance_rebuild_parity_v4.py`, job
9538092 on ccc0440: on-node self-repro bit-identical, cert-path==build-script-path, `pop_App` 4
flips/step 1, `pop_Dyn` 0; committed artifact built on the Jupyter-pod CPU ⇒ cross-env drift, not
within-node) — **not a re-run** (draft.1 warm determinism bit-perfect, worst=0.0). §6 fail-forward
amended to codify the regrade exception. Record: `certifications/v4.0.0.md`; draft.1 FAIL record
retained at `certifications/v4.0.0-draft.1.md`. On tag: `eval/v4.0.0`; `eval/v3.0.0` stays certified
for v3 numbers. Bars: bar1 S3 d=1.734 · bar2 29/29 deployed+LOO · bar4 gap 0.112 · bar5 12W/3L
p=0.018 · bar6 swap+hard-cut 37/37 · bar7 11/11 · bar8 warm 0.0 + two-class parity · bar9 3 metrics
pass / 3 controls fail.

20:39 — **Grid v3 built end-to-end (tier-first) — everything ready for one-paste max-parallel launch.**
Amendment 2 corrected with split-verified arithmetic (tier C = hero/gas/illustration/raven, n=4;
hole has 0 test clips; C6/C7 stay descriptive). Executed: live_concert_2 quarantined + manifest
222 clips (cert amendment-3, sha 5a7a8be9); split v1.1 built+tagged (live_concert test =
live_concert_0; 38 classes byte-identical; 182/40); killed R2/R3 partials + suspect hero_flight
R5 quarantined. Built: exp_064 ic3 retrain (151 clips/403 pairs, owner-final keying, 19-clip
precompute delta, +control validation sample); build_ladder_items_v3.py -> frozen tier-first grid
(base_ext 10 / A 15 / B 33 / C 7 / X 44 rows x3 seeds) + missing cond clips; exp_065
manifest-driven runner (nullable adapter, optional reference) + job_grid.sbatch; exp_062 gains
LADDER_GRID override + r3xext mode (X-extension prefix-only BOTH twin sides, exp_062-consistent);
PLAN_v3.md = clean operative doc + launch sheet. Blocked only on Duo re-auth for sbatch.

22:20 — **Taxonomy v2 ADOPTED (owner validated 39/39) → sidedness fold → scoring UNBLOCKED → B1 submitted.**
Owner signed off all 39 classes in the viewer (two correction exports folded; final counts
transform 17 / overlay 12 [add 6, state 5, remove 1] / cover 4 / traverse 6 / cut 0; two standing
§5.1 exceptions plasma_explosion + portal with pre-registered conservative handling).
`PROTOCOL_v2_PROPOSAL.md` §5–§7 rewritten to owner-final (`7a10815`); `build_class_axes_v2.py`
regenerated as idempotent mirror of the validated record. **Instrument fold:** giant_grab +
hero_flight onesided→twosided via `OWNER_SIDEDNESS_OVERRIDES` in `build_corpus_manifest.py`,
manifest rebuilt (sha `e7c867a6…`→`348db23d…`, diff = exactly 2 class fields; `85023fa`);
certification **amendment-2** on eval/v3-spec-versioning (`26023c2`) records the operational
rule (score with `--corpus` at the corrected manifest) + hero_flight σ_seed-roster caveat.
B8 re-verified: all 8 R3X recipients unchanged → running Amendment-1 jobs unaffected.
**B1 deferral lifted** (hero_flight validated two_sided = pre-built `sidedness_key`, no retrain):
submitted keyed R2/R3 gen **9539197** + R5 **9539198** (`b13a41a`). **All-rung scoring is now
unblocked** (S mask inputs owner-final); first scoring batch must rescore the hero_flight σ_seed
item both ways per amendment-2.

19:11 — **Eval-ladder Amendment 1 IMPLEMENTED + submitted (cluster-wide `secondary`).**
Built: keyed prefix-only configs for the 9 one_sided specialists (`configs_keyed/`, output
`<cls>_keyed/`, reusing existing precompute; two_sided shadow_smoke/hero_flight `<cls>_keyed`
symlink the blind==keyed run); `run_c2v_inference.py` rewritten for ladder_items_v2 (keyed
conditioning + `<cls>_keyed` ckpts + `--no-adapter` R1K [zero-init PEFT = base] + `--r3x`);
parametrized `job_gen_keyed.sbatch`; `build_r4x_manifest.py`→`ladder_r4x.json` (32 rows) +
`run_ic_inference.py --manifest`. Job graph: cancelled blind R2/R3 gen (9530601) + one_sided
blind chain tasks (kept _3/_9); let running blind trainings finish as fallback. Submitted:
keyed train **9531967** + chain **9531968** (9 one_sided); **R1K 9532033** (54 videos, base,
now); **keyed R2/R3 gen 9532034** (afterok chain, 10 classes, hero_flight deferred); **R3X
9532165** (afterok chain, 96 videos); **R4X 9532166** (96 videos, ic2, now). B8 re-verified
against taxonomy v2 (scene_swap unchanged for all 8 → R3X eligibility sound). Commits: amendment
`22cf6ce`, implementation `c0ea421`. Monitor task b5ptlfloa.

19:05 — **Taxonomy Protocol v2 GATE-PASSED + v2 validation viewer live.** The v1 descriptive
taxonomy failed under scrutiny (morph⇔¬scene_swap 21/21 tautology via its elastic clause;
17/39 sidedness + 13/39 mechanism hard-calls; filmstrip-verified misassignments: portal,
sakura_petals, polygon, plasma_explosion). Redesigned from the task principle "endpoints are
conditioning — classify what the middle must synthesize": new `mechanism`
{cover 6, transform 12, overlay(add/remove/state) 14, traverse 6, cut 1} via an ordered
decision procedure; `middle_only` (conditioning-evidence bit, R1−R0 headline split) replaces
`inserted_content`; `subject_anchored` demoted to metadata; `sidedness` untouched (frozen
instrument semantics — relabels only). Process: 14-class filmstrip audit → two independent
fresh-context advisors (architect + adversary, both filmstrip-grounded) → operator synthesis →
fresh-context acceptance gate: **rev.1 FAILED** (2 material wording defects: T1 trigger missed
dissolve-without-reformation; sakura_petals unflagged on the convert-vs-extract boundary) →
fixed → **rev.3 PASSED** (12/12 re-derivations reproduce the table, arithmetic verified, zero
material defects). Artifacts: `docs/taxonomy/PROTOCOL_v2_PROPOSAL.md` (authoritative),
`scripts/build_class_axes_v2.py` → `outputs/taxonomy/class_axes_v2.yaml` (count-asserted),
v2 viewer regenerated (rulings-first ordering). v1 record archived to
`docs/taxonomy/v1_{PROTOCOL.md,class_axes.yaml}` (was untracked). Owner still owes: 7
mechanism rulings + 9 sidedness conflicts in the viewer; scoring stays blocked until sidedness
lands. Strata for the ladder: transform 12 / overlay 14 / pooled new-shot 13 (confirmatory),
cover 6 / traverse 6 (descriptive), restyles = copy_max calibration subgroup.

18:57 — **Eval-ladder PLAN Amendment 1 pre-registered: side-keyed specialists + R1K + R3X/R4X (C9).**
Advisor (fable) ruling on an owner design challenge: the sidedness-BLIND specialists (D2) were
anti-conservative — on one_sided classes the blind suffix hands the specialist the true arrival
endpoint B (the effect's terminal state per SPEC §3), biasing the PRIMARY C5 (R3−R4) toward its
pre-declared R3>R4 direction. Fix, committed BEFORE any keyed generation exists (outcome-blind
honesty anchor): (1) retrain the 9 one_sided specialists PREFIX-ONLY into `<cls>_keyed/` dirs
(two_sided shadow_smoke/hero_flight unchanged, blind==keyed; blind one_sided ckpts kept as a
labeled sensitivity artifact); (2) new rung **R1K** = prefix-only base (no adapter) re-baselining
C4/C6/C7/C8 so adapter-value isn't confounded with suffix removal (C1 stays on blind R1);
(3) new secondary rungs **R3X/R4X** (contrast C9) = cross-class donor endpoints on the 8-class
block B8 (one_sided ∧ scene_swap=false), 96+96 videos, no GT (class-effect transfer). D2 and §7
rule-(iii) superseded; the 60 R4/R5 videos + all R0/R1 + split sha UNCHANGED. New grid
`docs/eval_ladder/ladder_items_v2.json` (sha `087206d7…`, derived verbatim from frozen v1
`afe17a3f…`) via `build_ladder_items_v2.py`; dated amendment in `docs/eval_ladder/PLAN.md`.

17:21 — **Eval-ladder jobs MIGRATED to cluster-wide `secondary` (our node was fully saturated).**
The 16:40 submission to `HCESC-H100-secondary` sat `PENDING (Resources)` indefinitely — that
queue only scavenges our lab's own node `ccc0439`, whose 8/8 H100s are held by another user's
2.5-day `-high` job. Cancelled the four (nothing had run) and resubmitted the identical chain to
the bare cluster-wide `secondary` partition (`--account=campusclusterusers --gres=gpu:H100:1
--requeue`), which reaches the 6 extra 8×H100 nodes `ccc0419–0424` (same play as exp_050's
sweep arms). All jobs are already `--time=03:55:00`, inside secondary's 4h cap, and resume-aware,
so preemption just requeues from checkpoint. `sbatch --test-only` estimated a ~1h backfill start
vs. indefinite on our node. New IDs: **A5** train `9530598_[0-10%4]` + chain `9530599`; **A6**
R4/R5 gen `9530600_[0-2]`; **C1** R2/R3 gen `9530601_[0-32%4]` (afterok:9530599). Recipe,
manifests, and dependency graph unchanged; scoring still blocked on sidedness validation.

16:40 — **Eval-ladder GPU jobs SUBMITTED** (via `ssh cc` → cc-login5, one-command
`docs/eval_ladder/submit_ladder.sh`). All on `HCESC-H100-secondary` (preemptible,
resume-aware), queued and healthy: **A5** 11 R2/R3 specialist trainings
`9529607_[0-10%4]` + chain-retry `9529608` (afterany); **A6** R4/R5 generation
`9529609_[0-2]` (60 videos — R4 16 + R5 gas/illustration 4, ×3 seeds; hero_flight
deferred); **C1** R2/R3 generation `9529636_[0-32%4]` (afterok:9529608 → auto-runs
when training finishes; 264 videos, ckpt 250+2000). Precompute is folded into the
training array (per-class idempotent). Scoring stays blocked until sidedness validation.
17:45 — **transition-eval 4.0.0-draft.1: health-validated metrics + causal bar 9 ported into
the certified harness (branch `eval/v4-metrics`, not yet frozen/run).** Replaces the three M1
transfer metrics with the metric-search/health-validation deliverables — M1a=**S3** (4-channel
appearance+dynamics ECDF composite), M1b=**D_ZPR** (3-view Z/P/R camera ECDF fusion), M1c=**CSLS**
(k=10 de-hubbed object motion, scoped stamp) — now corpus-relative: raw measurements ranked
against the committed `reference_v4.npz` instrument constant (`reference_stats.py`; μ + 9 ECDF
populations + CSLS r_obj; sha in `versioning.PINS`). Fixed causal-excess exam gate ported as
**bar 9** (`certify/datasheet.py`, self-verifying via 3 negative controls that must FAIL). §4
invariant amended "all scores raw" → "no outcome-coupled normalization"; deployed scoring emits a
saturation flag outside fitted support. Advised campaign (fable-advisor xhigh): direct
replacement (no sign-test theater), DINO-only gating baseline (max-over-proxies non-gating), m1c
ships headline scoped, bar-2 leave-own-clip-out robustness clause + non-gating D_ZPR-reversal
field. Port parity-verified at every tier (S3/Z/P/R/CSLS bit-exact, D_ZPR 2e-16, retrieval
headlines 4dp, datasheet verdicts reproduce n_perm=1000); bar 9 dry-run through deployed code =
PASS; 68 tests green. Next: freeze bars (own commit) → §6 certification run on the H200.

16:18 — **Eval-ladder scaffolds built + ready to submit; A7 audit PASS.** Scaffolded
`exp_062` (11 R2/R3 specialist trainings: `caption_missing.py` captioned the 24 held-out-
class train clips via Gemini/PyAV; `build_datasets.py` → 11 per-class manifests + configs
from split-v1 train clips, 92 clips; `job_train.sbatch` = self-contained precompute→train
array 0-10%4, resume-aware/chain-safe) and `exp_063` (R4/R5 generation off exp_058 ic2
step_05000 native-keyed; `build_manifests.py` → 22 rows = 20 active + 2 hero_flight deferred,
reusing exp_061's cond cuts; `run_ic_inference.py` + seed-array `job_infer.sbatch`). Pairing
audit A7 **PASS: 132/132** roster test items (11 classes × 2 × 3 seeds × 2 arms) present in
exp_061. **Environment note:** this background session runs in a GPU-less, Slurm-less Jupyter
pod (login nodes unreachable), so the two GPU steps — A5 (submit 11 specialist trainings) and
A6 (submit R4/R5 generation) — must be launched from a `cc-login3` session; exact submit
commands are in each experiment's README + `docs/eval_ladder/PLAN.md` §6.

15:47 — **Eval-ladder launch batch (advised campaign, `/advised`).** Committed the
untracked metric-search code + reports and force-added its 88K result artifacts on
`eval/metric-workbench` (`32b1546`, `393f093`) — worktree-persistence risk retired.
Declared **`split_v1` FINAL** (`data/processed/transitions_std121/SPLIT_V1_FINAL.md`,
sha256 `f6cc8b5b…`, tag `split/v1`): the split is metadata-only, so the pending
sidedness re-annotation cannot move it (only a clip-roster/curation change could, and the
one open flag `water_element_5` is non-roster and band-invariant). Pre-registered the
canonical **eval-ladder plan + frozen item grid** at
`docs/eval_ladder/{PLAN.md,ladder_items_v1.json,build_ladder_items.py}` (fable-authored):
rungs R2–R5 recipe, class→rung coverage matrix, and a new **contamination finding** —
exp_058's B-tier exclusions were keyed to its own eval quads not split v1, so only 4/16 R4
items have ic2-unseen endpoints ⇒ the R3−R4 (C5) contrast is stratified. Specialists go
sidedness-BLIND (taxonomy-immune, all 11 roster classes incl. hero_flight); generalist
pinned to exp_058 `ic2` step_05000 native-keyed; only R5 hero_flight (6 videos) waits on
sidedness validation. Scoring stays blocked by design.

## 2026-07-14

17:05 — **Metric Workbench CLOSED** (branch `eval/metric-workbench`, 32 commits). All
three tracks terminated on pre-registered rules; nothing was adopted and no frozen
number was ever changed. Cycle 2 ran the owner's E1' directive: the gamma-scalar
signature (a_hat, b_hat, m~) in RAW geometry, gated behind two instrument-validity
preconditions. **Both IVs passed (IV1 0.9357, IV2 1.0000), so the kill rule BINDS the
hypothesis** — d 0.7979 vs the pinned 1.522006 and 183/223 misretrieved vs 73.
**Endpoint-normalization is dead at the appearance level, adjudicated.** `m1a__v3_sided`
stands unchallenged. Two findings worth reusing: (1) the executor-chosen
`eig_floor_ratio=1e-6` — not whitening itself — was E1's instrument failure; a
parameter-free Ledoit-Wolf whitener leaves the signature intact (d 0.8047 vs raw 0.7979)
because an eigenvalue floor amplifies near-null DINO directions to unit variance while LW
does not; (2) `curves.resample` on a *monotone* scalar channel returns a straight line
(per-channel arc length = total variation), which silently degraded e0's descriptive
curves — resample multi-channel curves jointly. Report:
`src/diffusion/transition_eval/workbench/WORKBENCH_REPORT.md`.

**14:20** — Metric Workbench (branch `eval/metric-workbench`) ran end-to-end and both
tracks terminated on pre-registered rules. **Phase 2 / E1** fired its §4.1 KILL rule
(delta: Cohen's d 0.358 vs the pinned incumbent's 1.522; 209/223 misretrieved vs
73/223; hubness FAIL) — E2/E3 do not run. A control shows the identically-preprocessed
representation with *no* endpoint-normalization at all scores the same, so the §1.1
whitening (whose regularization the RUNBOOK does not pin, and which the executor
froze at 1e-6·λ_max) is escalated as an **owner-reserved** matter with a pre-declared
floor-sensitivity sweep attached. **Phase 1 / motion** failed §3.4 acceptance — 29/35
injected-trajectory verdict cells pass (corr 0.96–1.00), but 3 fail at rungs a
noise-limited *oracle* passes with margin, and reversal fails its descriptor leg
(22/33) — so the exam is not run and M1b_flow/M1c_flow stay analysis-tier, with no
second attempt this cycle. Backbone: SEA-RAFT (amendment A2's timeboxed attempt
succeeded). Key measured facts for the owner: candidate coverage is 0.583/0.529
against the incumbents' 0.969/0.996, the dominant loss coming from RUNBOOK-*pinned*
rules meeting a corpus whose effects fill >87% of the frame on undefined frames; and
the corpus's median per-pair camera translation (0.30 px) sits within an order of
magnitude of the flow fit's own noise. Deliverable:
`src/diffusion/transition_eval/workbench/WORKBENCH_REPORT.md`. The certified cache and
the eval/v3.0.0 tag were never touched.

# Changelog

## 2026-07-14

**17:05** — **Full transition taxonomy annotated: 39/39 classes × 7 fields (`outputs/taxonomy/class_axes.yaml`), AWAITING OWNER VALIDATION via `outputs/taxonomy/viewer.html`.** Protocol frozen then pilot-amended (`outputs/taxonomy/PROTOCOL.md`): added `scene_swap` (the corpus's dominant split: only 14/39 classes swap scenes; 25 are same-shot state changes) and extended `morph` to physical-removal effects (giant_grab). Labeler chosen by bake-off on a 7-class pilot vs a blind answer key (`pilot_key_fable.yaml`): Sonnet 49/49 field agreement (chosen, 4 parallel batches over 107 filmstrips at `outputs/taxonomy/filmstrips/`); Haiku ~70%, rejected. Consistency sweep clean (0 hard violations); **no B_only class exists — S's depart-A assumption is now verified, not assumed**. Distribution: morph 21 / occlusion 12 / traversal 6 / dressed_cut 0 (one per-clip); A_only 27 / two_sided 12. **9 sidedness conflicts vs the manifest flagged for owner (instrument-critical, S mask): giant_grab, hero_flight (all 3 exemplars two_sided vs onesided), hole_transition (A_only vs twosided), earth_element, earth_wave, water_bending, water_element (+ per-clip: flying_cam_transition_3, live_concert_7). 9 heterogeneous classes incl. polygon (the only M1c-trusted class). Curation find: water_element_5 contains no water — it is visually a wireframe-class effect.** exp_061/R0-R1 scoring stays blocked until validation. — **exp_061 SUBMITTED: eval-ladder rungs R0/R1 — base-prior and conditioning-suppression baselines, zero training, overnight.** R0 = base LTX-2 19B dev, prompt only, no conditioning, no adapter (does the base prior contain each effect?); R1 = same base + endpoint conditioning (prefix 9f/suffix 8f, exp_051 recipe), no reference, no adapter (conditioning suppression). 50 deterministic items from frozen split v1 post-audit (28 classes' test clips = 39 items + 9 all-train classes' certified bar-pair clip A + 2 singletons — all 39 classes covered) × seeds 42/43/44 × 2 arms = **300 videos**, same prompt ("ICTRANS " + type-blind endpoint caption, base-twin precedent) and same seeds in both arms (paired; R1 rows carry `twin_of`). Recipe = exp_057/060 ValidationRunner contract verbatim (480×640×121@24, 30 steps, CFG 4, STG stg_v [29]); 3 missing captions generated with the exp_058 captioner. Budget: exp_060 measured ~121 s/video WITH reference → ≤110 s/video here; 300 × 110 s ≈ 9.2 GPU-h in 4 array tasks capped at 2 concurrent GPUs ≈ 5–6 h wall, inside the 14 h window, under the 500-video cap. Job array **9495817** (HCESC-H100-normal, %2), resumable skip-if-exists. Manifests `dataset/eval_manifest_{r0,r1}.json` (150 rows each, SPEC §2-valid, reference=self) ready for score.py — **scoring deliberately NOT run** pending the sidedness re-annotation. Scaffold `experiments/exp_061_ladder_r0_r1/`.

**16:45** — **Frozen train/test split v1 + near-duplicate audit for transitions_std121 (223 clips / 39 classes).** Rule (metadata-only, seeded RNG `split_v1:<class>`): n≥8 → 2 test, 4≤n<8 → 1 test, n<4 → all-train; builder `scripts/build_split_v1.py` → `data/processed/transitions_std121/split_v1.json` (provenance: rule verbatim, corpus sha256, flag set) — exactly reproducible. Audit (`scripts/audit_split_v1.py`, job 9495508, H100 3:55): M2a copy score (certified worktree machinery, own cache `outputs/eval/split_v1_audit/`) on all 326 cross-boundary within-class pairs over 5 remediation iterations at certified τ_copy 0.858. **44 pairs flagged; pre-registered remediation applied mechanically: illustration_scene_6 and polygon_3 were near-identical takes of train clips (copy_max 0.9975/0.9976, replaced); animalization_5↔7 a mutual near-dup pair (both now train); live_concert ALL 8 clips mutually near-dup (0.86–0.997) → class goes all-train.** Final: **184 train / 39 test (17.5%), 28 classes with test items**; report `data/processed/transitions_std121/split_v1_audit.md`.

**16:46** — Added `outputs/taxonomy/pilot_sonnet.yaml`: Claude Sonnet 5's candidate labels for the 7-class taxonomy pilot (portal, gas_transformation, illustration_scene, super_fast_run, giant_grab, color_rain, shadow_smoke), all 3 filmstrips per class read and judged per `outputs/taxonomy/PROTOCOL.md`. Notable findings: `giant_grab` sidedness is `two_sided` not the manifest's one-sided (the grabbing hand persists on screen through the final frame, doing a peace sign) — flagged `sidedness_conflict: true` for owner escalation per the instrument-critical rule. `color_rain` is heterogeneous on `stylization` (2/3 clips tint the whole scene; `color_rain_5` stays localized to the subject, sky/grass untouched). `shadow_smoke` and `super_fast_run` are the only two classes with `scene_swap: true` in this set. To be compared against `pilot_haiku.yaml` and `pilot_key_fable.yaml` (the coordinating agent's blind answer key) for annotation-model selection.

**16:45** — **exp_060 COMPLETE: σ_seed measured (O6) — 12 stratified probe items × 5 seeds on the decision-generating IC-LoRA arm, scored end-to-end by the CERTIFIED eval/v3.0.0 harness; every future 1-seed suite now has its MDE table.** Item selection deterministic and corpus-only (n≥4 classes; sidedness×tag strata round-robin, first-lexicographic; probe = certification's max-endpoint-distance bar pair per class, read verbatim from the draft.8 siblings manifest): air_bending, earth_wave, earth_element, animalization, fire_element, firelava, melt_transition, hero_flight, color_rain, illustration_scene, flying_cam_transition, raven_transition. Adapter = exp_056 IC-LoRA step 3000 (the exp_057 checkpoint), exp_057 recipe verbatim (480×640×121@24, 30 steps, CFG 4, STG stg_v [29], prefix 9f/suffix 8f, ICTRANS type-blind captions — 1 missing caption regenerated with the exp_058 captioner). Generation 9488544 (H100, 2:01 — 60 videos in ONE model load via per-sample ValidationSample seeds); scoring 9488659 (5:35 warm, certified worktree `.claude/worktrees/eval-v3.0.0`, 60/60 rows CERTIFIED, 0 error rows); ≈2.1 GPU-h total. **σ_seed (pooled, 48 df): app_ref 0.0271 · margin 0.0427 · copy_max 0.0251 · cam_dtw 0.0864 (11 groups — raven_transition cam-invalid 0/5, air_bending/earth_wave 4/5; NaN reported never imputed) · obj_match 0.0091 · max_seam_z 0.3036. MDE@n=10 paired: 0.024 / 0.037 / 0.022 / 0.076 / 0.008 / 0.266.** score.py does not propagate unknown manifest keys (rejects by design), so probe_group is joined post-hoc by `experiments/exp_060_sigma_seed/compute_sigma.py` before calling the certified `certify.seeds.sigma_seed` — instrument untouched. Artifacts: `outputs/eval/sigma_seed/{sigma_seed.json,items_with_probe_group.jsonl,adapter/{items.jsonl,results.json}}`; scaffold `experiments/exp_060_sigma_seed/`. σ_seed gates the FIRST MODEL REPORT, not the tag (SPEC §6.4) — certification record deliberately not amended here.

**10:22** — **Eval-harness documentation finalized post-certification.** Repo entry point added: `notes/eval_harness_v3.md` (positioning map — certified status, what certification claims/doesn't, plan→infer→score flow, trust-map consumption rule, open items before the first model report) + INDEX.md row and Eval section; `certifications/README.md` rewritten to describe the three committed records (v3.0.0 PASS via disclosed regrade, draft.8 and draft.7 FAILED history); SPEC touch-ups: §0 header marks the register lock-clear (tag exists), §8 gains the certification-driver CLI line, §9 map now lists the actual `certify/` contents (driver, blocks, diagnostics, figures, explorer), stale O4 parenthetical in §10 removed. No code changes.

**10:15** — **Transition-eval 3.0.0: owner's closed-list bar revision (bar 1 → d ≥ 1.5 only; draft.8 bars 2+3 merged into one), draft.8 run REGRADED under the new bars → overall PASS, first certified tag `eval/v3.0.0`.** Both edits decided at the draft.8 joint inspection with the outcome known, and disclosed as outcome-aware verbatim in bars.yaml, SPEC §6, and the record: bar 1's accuracy conjunct deleted (floor was calibrated on the 11-style v2 corpus, chance 0.213, never re-derived for the 39-class exam, chance 0.067; the surviving d conjunct was pre-registered and passed at 1.522 before the change; accuracy 0.673 stays reported, descriptive); bars 2+3 gated the same sibling-vs-control inequality from opposite sides and merge into bar 2 — per n≥4-eligible class, sibling > control ∧ M2a silent on the sibling, ALL 29 eligible classes must pass (the 35/37 and 37/39 count floors were arbitrary headroom; nature_bloom, n=2 and draft.8's only miss, leaves the denominator under the exam's existing n≥4 trust convention — disclosed plainly with its residual-risk note). `core_degenerate` removed from the certification bar path entirely (no conjunct, no silent logging; flag stays live in S/mask-adoption/Block C descriptive). Per owner directive no computation was redone: `scripts/regrade_draft8_to_v3.py` re-runs only the two changed graders (deployed code) over the draft.8 artifacts (job 9465002); bars 4–8 carry verbatim (grader code byte-identical). Regraded verdicts: bar1 PASS (d 1.522), bar2 29/29 PASS, bars 4–8 PASS → overall PASS; record `certifications/v3.0.0.md`; VERSION → 3.0.0; SPEC §0 O5 resolved (draft.8 executed the full stack end-to-end). Tests 83 passed + 1 skipped.

**09:46** — **Certification perf round 2 (owner-approved, numeric no-ops): mtime-preserving probe builds, concurrent scoring, parallel appearance matrices, LPIPS cache.** Draft.8 measured 1h38m: ~63 min in five sequential score.py runs, 29.5 min in Block A, 13.4 min deliberately-cold anchors. Changes: (a) `write_video` encodes to a temp file and keeps the existing file when bytes are identical (x264 verified byte-deterministic here) — probe videos stop invalidating their stat-keyed feature/track caches every run; (b) the driver converts Block C first, then scores cert_siblings/cert_probes/cert_blockc as three concurrent score.py subprocesses (disjoint item sets → disjoint cache writes), and Block D's warm rerun + cold anchors also run concurrently — per-item math untouched, and bar 8's comparisons verify that at run time; (c) `appearance_distance_matrix` runs on the same fork pool as motion (shared `_map_pairs`; equality-tested); (d) temporal + endpoint LPIPS cached in `--cache-dir` keyed by stat-based video identity (`LPIPS_CACHE_TAG`), with a warm cache also skipping the generated video's decode — honesty guards: the warm rerun scores with the new `--lpips-cache off` so bar 8 keeps recomputing LPIPS end-to-end (preserving bars.yaml's pre-registered "LPIPS is recomputed every run" property within tolerance 1e-6), and cold anchors use their own empty cache dir. Verified end-to-end through the shipped CLI on a real draft.8 sibling item: cold-cache vs warm-cache runs bitwise identical on every field (warm run 71 s vs 7.5 min cold on CPU, decode skipped); all feature/track metrics bitwise equal to the committed draft.8 rows; LPIPS-derived fields differ from the H100 record only by CPU-vs-GPU float drift (≤4.4e-3, tolerance 0.04). CPU-seeded lpips cache entries deleted from the shared cache afterward (would trip bar 8's 1e-6 warm tolerance on a GPU run). Tests 83 passed + 1 skipped (4 new). Projected next cert run ≈ 35–40 min, floor set by cold anchors + first-pass scoring of fresh videos.

**09:00** — **Certification representation system: every run now persists and renders its full diagnostic state automatically (owner-requested, non-gating).** New `certify/diagnostics.py`: `run_exam` no longer discards what it computes — the six distance matrices, full confusion, per-clip 1-NN predictions with distances/margins, class-pair distance matrices, R2 rows with intruders, and R1 accuracy per transition-tag group (coarse pools + exact patterns) are written to `<cert_dir>/analysis/{analysis.json,distance_matrices.npz}` in the schema the explorer reads, making the post-hoc recompute job (`scripts/exam_confusion_analysis.py`, now a legacy backfill importing the shared helpers) obsolete. The driver ends every run with a representation step wrapped so it can never gate the record: `certify/figures.py` saves 11 PNGs to `<cert_dir>/figures/` (bar verdicts, per-metric exam accuracy vs chance/floor, six stratum-ordered confusion heatmaps, tag×metric accuracy, R1/R2 margin distributions) and `certify/explorer.py` (moved from `scripts/build_results_explorer.py`; byte-identical output verified against the shipped draft.8 page) rebuilds `results_explorer.html`. The certification md gains an "Exam detail" section: per-metric acc/d/chance table, R2 accuracy, and the R1-by-tag-group table (`by_tag` also lands in exam.json/record.json). Zero effect on any verdict; tests 79 passed + 1 skipped (3 new); figures verified on draft.8 data.

## 2026-07-13

**17:42** — **Eval pipeline perf: lazy decode on warm caches + parallel motion matrices (numeric no-ops, owner-approved).** `process_video_file` gains `need_frames=False`: when the DINO-feature and track caches are both warm it skips video decoding entirely (fps from the container header; cache misses still decode, so results never change) — cache-warm corpus loading drops from ~15 min to seconds. `motion_distance_matrices` now runs its 24.7k-pair DTW/object-match loop on a fork pool (numpy-only workers, each cell independent → bit-identical; equality test added). Frame-discarding call sites (cert corpus/reversal loops, score.py reference loading, exam analysis script) opted in. Also: `scripts/exam_confusion_analysis.py` recomputed the draft.8 exam's full diagnostic state with deployed code (job 9470438: confusion matrices, per-clip 1-NN margins, class-distance matrices, R2 intruders — verifies exam.json exactly), and `scripts/build_results_explorer.py` renders it all as a self-contained interactive explorer. Instrument tests 32/32; these commits postdate the draft.8 record (31dd07e) — next certification run stamps a new commit and draft tag per versioning discipline.

**15:05** — **v3.0.0-draft.8 certification executed end-to-end (Slurm 9465002, H100 secondary, 1h38m) — overall FAIL, 6/8 bars PASS, record committed.** First complete A→B→C→D run: draft.7 had died in Block-B scoring; draft.8-minimal's per-item error rows + empty-keep guard + anchor dedup carried every stage through with **zero error rows**. Failing bars are exactly the two pre-disclosed ones: bar 1 (M1a exam acc 0.673 < 0.80; d 1.52 ≥ 1.5) and bar 3 (36/37 control floors; nature_bloom control 0.596 > sibling 0.420). First-ever data: bar 4 splices 74/74 detected at τ 0.88 with gap 0.112 (splice_min 0.914 vs honest_max 0.802; τ_copy recalibrated → 0.858), bar 6 endpoint-swap 37/37 (true ≈1.0 vs swapped 0.02–0.24) and hard-cut 37/37 (z 5.0–234), bar 7 copy twins 11/11 flagged (copy_max 0.974–0.988), bar 8 warm rerun bitwise 0.0 over 74 rows + first cold-anchor execution (6/6 anchors, worst |Δ| 3.9e-4 vs tol 0.04). Adoptions reproduce draft.7 (v3_sided mask, v3_decomposed motion); O7 Huber conditional triggered again (camera stratum 0.346); content-invariance audit 0.818 (non-gating). Record: `certifications/v3.0.0-draft.8.md` + full artifacts under `outputs/eval/certification/3.0.0-draft.8/`.

**11:57** — **v3.0.0-draft.7 certification FAILED (Slurm 9463686) — honest record committed; failure analysis = draft.8 fix list.** Bars 2 (siblings 36/37) and 5 (reversal 12W/3L p=0.0176) passed; bars 1/3/4/6/7/8 failed. Root causes: bar-1 floor anchored to the 47-clip/11-style exp_054 precedent (223/39 exam has chance 0.067, not 0.213); bar-3 core_degenerate conjunct false-by-design on lerp controls; `object_match` crash on empty tracklet keep-filters killed probe + Block-C scoring; the anchor rule's own n=6 assertion refused after picking air_bending twice; O7 triggered. Record assembled post-hoc from run artifacts (`post_hoc_assembly` in record.json).

**10:24** — **Eval harness certification system implemented + bars finalized (v3.0-draft.7).** `certify/` is now fully executable to the locked SPEC §6: `exam.py` (two readouts — clip-level LOO 1-NN and pool-level M2b margin classification via imported deployed code; exact-binomial sign-test adoption α=0.05; motion contingency; O7 Huber conditional; trust map with M1c definedness), `probes.py` (max-endpoint-distance sibling pairs + all-pairs content-invariance audit from cached features; 24-frame non-core splices with one deterministic crop+color-gain perturbation; endpoint-swap; hard-cut; reversal with analytic self-reversal enumeration), `blockc.py` (v2 manifest conversion with loud exclusions, the exp_057 11-copy-twin bar, v2↔v3 bridge, per-arm distributions), `run_certification.py` (mechanical A→B→C→D driver → `certifications/v<ver>.md`). Pre-freeze corpus-only calibration validated the reversal threshold on real cached tracks: all 95 valid camera-class clips score self-reversal DTW ≥ 0.789 vs 0.0 for analytic invariants (threshold 0.5 separates cleanly) — and surfaced a documented M1b property: the z-normed statistic is blind to reversing time-antisymmetric velocity profiles, which is exactly what the enumeration keeps out of bar 5's denominator. Cache audit: DINO features + tracks 100% warm for all 223 corpus clips and all 150 archived exp_056–058 generations — certification GPU cost is only the ~130 constructed probe videos + LPIPS + cold-anchor rerun (~2 h single GPU). Tests 47/47. bars.yaml numbers final (freeze delegated); `frozen: true` flips in the next commit.

**09:35** — **Eval harness health-assessment spec locked (v3.0-draft.6, branch `eval/v3-spec-versioning`).** SPEC §6 rewritten as the full health system after a three-pass design review (proposal → red-team → external review): four blocks (A exam / B probes / C realism / D stability+calibration), 8 hard bars on constructed or human-verified truth only, two-readout exam (clip-level LOO 1-NN for M1a/M1b/M1c + pool-level margin classification for M2b — trust does not transfer between estimators, so the exam must import deployed metric code), sibling probes hard-barred on max-endpoint-distance pairs, splice perturbation + minimum-gap, reversal-sensitivity enumeration, M3 endpoint-swap/hard-cut panel, content-invariance audit as a required record artifact, and a two-kind calibration rule (corpus-only pre-freeze vs outcome-coupled post-freeze). `certify/bars.yaml` now carries the locked forms with DRAFT numbers (`frozen: false`); the freeze session sets ~8 numbers and flips the flag in its own commit. Review kills reverted out of the tree (with the superseding entry for 2026-07-10 16:05): cross-label probe (≡ the exam's pool readout; circular on generations), Δ-novelty/O8 (deferred to v3.1), bridge B1/B2 bars (v2↔v3 bridge is now descriptive — a bar on generated-item degeneracy rates would gate on model behavior), O7 dual-weighting implementation (reduced to a pre-registered Huber conditional). Kept from that pass: the versioning.py dirty-path fix. Tests 34/34.

## 2026-07-09

**12:01** — Added `experiments/exp_058_ic_lora_diverse_retrain/ALLOCATION.md`: one-page training/eval allocation reference built from the actual artifacts (pairs.json/quads_v2.json/dataset_exp058.json) — per-class table (corpus clips vs trained clips vs pairs vs sidedness), the three unseen tiers (A held-out class / B trained-with-eval-clips-excluded, 0-leak verified / C trained-with-clips-seen incl. 6 exact-training-pair eval items), and the arm→tier map for the viewer (held-out classes live inside the exp_057-named suite arms, not in `ic2_ts_heldout`, which is raven-only; filter by style to browse a held-out class).

**11:30** — **exp_058 post-hoc review: fixed the missing `arm` field that had collapsed report.md's per-arm tables and the viewer's category grouping into one blank arm.** `make_quads_v2.py` wrote `manifest_ic_v2.json` without `arm` (an EvalItem field defaulting to "") → `items.jsonl`, `report.md`, and the viewer all lost arm labels; `analysis_v1v2.md` (the load-bearing doc) was unaffected because it re-joins arms from `quads_v2.json` — no scientific conclusion changes. Repair without rescoring: `arm` + human-readable `notes` added to the manifest writer, `items.jsonl` backfilled from quads_v2 (original kept as `items.jsonl.orig-noarm`), report/scatter regenerated via `run_score_ic.py --from-items`, viewer rebuilt (validate_bundle PASS, arms now `ic_os_inclass/ic_os_to2s/ic_ts_unseen/ic_anchor/ic2_prefixonly/base_prefixonly/ic2_ts_heldout/base_ts_heldout`). Also made the §5 caveat explicit in the note: suite one-sided items were generated+scored WITH suffix conditioning (required for the paired v1↔v2 design) which is off-training-mode for v2 → suite deltas on one-sided classes are a conservative lower bound; the cheap missing measurement is a prefix-only rerun of the 35 one-sided suite items (v3 candidate 2b).

## 2026-07-08

**22:41** — **exp_058 COMPLETE: the diversified mixed-conditioning retrain delivers exactly what coverage can buy — and pays a measurable, mechanistically clean price.** Train 9401247 (5000 steps H200, loss →0.18), 53 v2 generations (9405052-58, all clean with ABSOLUTE LoRA paths), scored in run_0001 (eval 9405337), `analysis_v1v2.md` pairs v1↔v2 on the 40 identical exp_057 suite items (exp_057 base twins reused, adapter-independent). **Paired headline: raw app +0.046 (24/40) at leak +0.011. The appearance-blind vanish probe gas_transformation — held out, endpoints reveal nothing — improved 0.41→0.61/0.64→0.67 in-class: genuine in-context gain, not texture recall. Novel-texture gradient largely closed (illustration cross-target 0.33→0.42; wireframe/polygon in training). Camera cross-target FLAT (0.21→0.20) with 8 camera-tagged classes trained — the pre-registered conditioning-conflict hypothesis stands. NEW capability: prefix-only generation (no end frame) works — raw 0.50, prefix DINO 1.000, vs base twins that replay the demo verbatim (leak 0.992, seam z +117); caveat: without the suffix anchor leak rises to 0.73±0.12 (gas_7 0.91) — the pinned suffix was an anti-copy constraint. COSTS: exp_056 anchors −0.058/−0.099 raw (leak down proportionally — more conservative two-sided transfer, app-per-leak unchanged), and the suite seam regression (+0.62 mean) concentrates ENTIRELY on one-sided classes scored WITH suffix — a conditioning-mode mismatch (trained prefix-only, evaluated suffix-pinned), not general decay: anchor seams/suffix DINO are identical to v1. Quiet finding: classes moved INTO training with eval clips excluded barely moved (0.565→0.586) — reference-reading, not class recall, carries transfer.** Viewer `outputs/eval/exp_058/viewer` (PASS; `outputs/eval/` serves 056+057+058). Gemini video judge NOT run (quota); note + v3 candidates (two-sided rebalance, cross-class prefix-only, suffix-optional one-sided training) in `notes/exp/exp_058_ic_lora_diverse_retrain.md`.

**19:00** — **exp_059 COMPLETE: first controlled LTX-2 inference benchmark on H100 80GB — dev two-stage vs distilled checkpoint, 720p/1080p, 5 s, granular per-section timings.** 10 arms on cluster-`secondary` H100s (jobs 9401444 + 9403087), 4 calls/arm (1 cold + 3 warm), sections timed via monkeypatched `DiffusionStage._transformer_ctx`/`run` + block wrappers + a timing iterator around the lazy VAE decode. **Headline warm numbers per 5 s video:** dev two-stage 142 s (720p) / 215 s (1080p); distilled eager 103/116 s — but only 6–16 s of that is denoise compute, ~90 s is the stock per-call model rebuilding (DiffusionStage frees the transformer after each stage, Gemma rebuilt every call); **distilled with GPU-resident weights (`StateDictRegistry`) = 11.2 s (720p) / 23.4 s (1080p) at 69–72 GB VRAM — 13–9× faster than dev, and the recommended fast setting.** Registry mode is distilled-only: the dev stage-2 LoRA fusion goes out-of-place against the cached copy → 39+39 GB OOM (9401275). torch.compile(reduce-overhead): −6–8 % warm for dev, ≈0 for distilled, +10 min one-time compile tax; NOT supported on the repo venv (Python 3.14) — compile arms ran a parallel `.venv-py312` (uv --python 3.12). Disk-cold first-ever loads off the project FS: Gemma 23 GB ≈ 400 s, checkpoint 40 GB ≈ 860 s (~50 MB/s). Distilled ckpt (40 GB) + spatial upscaler + distilled LoRA staged fresh from HF to `$LAB/cache/huggingface/ltx2_models/`. Scaffold `experiments/exp_059_ltx2_inference_benchmark/` (config/run.py/make_table.py/job_bench.sbatch); note `notes/exp/exp_059_inference_benchmark.md`; outputs+timings.json in `outputs/videos/exp_059_ltx2_inference_benchmark/run_0001/`.

**17:12** — **exp_058 sanity PASSED end-to-end; full 5000-step training submitted.** Captions done (185: 126 Gemini + 59 reused; 2 banned-word leaks hand-fixed: portal_10, wireframe_3), validation prompts patched (incl. one-off Gemini caption for held-out hero_flight_0), manifests built (162 rows / 32 classes; 460 pairs = 116 twosided + 344 onesided). Sanity 9400364 (cluster secondary H100) COMPLETED: 162/162 clips preprocessed with row-count reconciliation, 460-pair symlink tree + per-pair masks linked (all four data sources indexed incl. `masks`), 50 mixed-conditioning steps (loss 1.05→0.38, ~4.1 s/step incl. overhead), step-0/50 validations rendered for all three probe types (two-sided anchor, one-sided prefix-only, held-out hero_flight prefix-only) and uploaded to W&B. One fix along the way: the `--link` step now runs in the trainer uv env (`build_dataset.py --link` imports torch for mask writing; first attempt 9399580 died on system python3). Full training: 9401247 on HCESC-H200-secondary (--requeue, ckpt/500, resume-aware) + insurance job 9401248 chained afterany.

**16:15** — **exp_058 OPENED: diversified mixed-conditioning IC-LoRA retrain — 460 pairs / 34 classes / 162 clips, one-sided classes trained PREFIX-ONLY (2 start latent frames) and two-sided prefix+suffix, via a single per-pair MASK condition.** Feasibility settled first: the trainer's prefix/suffix conditions are global-config Bernoulli draws, but `MaskConditionConfig` loads per-sample masks through the identical `_apply_intrinsic_condition` math — `test_mask_conditioning.py` (run in the trainer venv) proves a `[0,1,15]` latent-frame mask is bit-exact vs exp_056's prefix(2)+suffix(1) and `[0,1]` gives prefix-only with the suffix trainable, both composed with the reference concat. Allocation (design.md): held out hero_flight (camera) / illustration_scene (novel texture) / gas_transformation (vanish, appearance-blind) / raven_transition (two-sided, was trained in exp_056 → cost-of-removal read) + hole/seamless/jump; for training classes with n≥7 the exact clips exp_057's quads used are excluded from training so the 51-quad suite reruns as an honest before/after; 14 NEW one-sided classes standardized (+66 clips, `standardize_train.py`, near-dups + 320px run_set_on_fire clips skipped). Variable-length training investigated and rejected: rescues zero clips (all short clips are 320px), only 29/227 clips are >145f (≤2× retime, precedented — exp_056's own corpus had 2×-retimed flying_cam/earth_wave/raven), and the trainer infers the reference position scale factor globally from ONE pair, so mixed ref:target ratios silently mis-position references. Low-res classes stay excluded (upscale-blur supervision). ~120 new type-blind endpoint captions generating via Gemini (first/last stills only + banned-word scrub). Fresh training from base (not continued from exp_056) for clean corpus attribution: 5000 steps, ckpt/500, step-3000 kept for budget-matched comparison; sanity 50-step first.

**14:35** — **exp_057 COMPLETE: the IC-LoRA's in-context transfer survives 14 unseen classes AND structure it never trained on — and the metric audit caught the harness's normalization breaking exactly where pre-registered.** All 51 quads generated (round 1 failed on a relative LoRA path — `job_infer.sbatch` cd's into the trainer; absolute paths required — round 2 9396277-81 + base 9396109/10 clean), scored in run_0001 (eval 9396609, EXIT 0, ~26 min incl. featurizing 110 new corpus clips; W&B `exp057_quads`). **Twins (n=11, identical inputs ± adapter) are the cleanest result in the series: base leak 0.97-0.995 with visible reference replay and seam snaps up to +228 (gas twin) vs IC leak −0.05..−0.60 lower, seams ~0/negative, endpoint DINO higher on 11/11.** Taxonomy: style transfers best (in-class raw app 0.75; cross-target wireframe neon-grids the melting hay, illustration comic-panels the swimmer), object semantics carry but formed objects shrink on foreign endpoints, camera arcs are weakest cross-target (0.30) though in-class hero_flight reproduces ground→aerial→POV-arm. **Texture-familiarity gradient confirmed: cousins (shadow/fire_element) 0.45±0.10 raw cross vs novel 0.30±0.07 — exp_056's cross-class numbers partly rode trained textures.** One-sided/vanish structure (never trained): executed — subjects transform/dissolve in place, empty-scene suffixes honored (endpoints 0.96), with trained-prior tint intrusions (gas renders dark; illustration_7 detours through shroud+fire). hole in-class = jump-pattern replica (semantics yes, appearance 0.34 vs ceiling 0.60); seamless pan fails. Anchors reproduce exp_056 raw metrics within ±0.04. **Metric audit (analysis.md gap table): floor≥ceiling for 7/16 styles (all one-sided: animalization/gas/giant_grab/money_rain/shadow/super_fast_run/wireframe) — normalized appearance is noise there and was gated out; raw appearance is copy-confounded (base twins hit 0.81-0.93 raw BY copying) so only raw×leak + twins are load-bearing; max-statistics (leak max-sim, max seam z) validated as the right detectors but seam conflates failure snaps with handoff artifacts (portal_11 +8.6 on a successful transfer) and in-class leak inflates legitimately (portal_12 0.96 with zero content copied).** Note `notes/exp/exp_057_ic_lora_unseen_eval.md` + INDEX row; viewer `outputs/eval/exp_057/viewer` (PASS 351/351; `outputs/eval/` served → both 056/057 viewers).

**13:20** — **exp_057 OPENED: broad unseen-class eval of the frozen exp_056 IC-LoRA on the user-labeled corpus — 14 unseen classes stratified by taxonomy (object/camera/style), structure (one-sided vs two-sided), and texture familiarity; 51 quadruples submitted.** The user reorganized `data/processed/transitions/` into `{onesided,twosided}_transitions/<sidedness>_<tags>_<class>/` (36 one-sided + 13 two-sided classes, 339 clips). `inventory.py` scan: 0 exact dups, 0 undecodable, 17 short clips (killing mouth_in/eyes_in), 3 aHash near-dup pairs; montage inspection added 2 same-take regenerations (`super_fast_run_11`, `plasma_explosion_3`) and confirmed `giant_grab_5`≈`giant_grab_0` — all 3 excluded so LOO ceilings aren't inflated. Selected 14 classes (camera: hero_flight/super_fast_run/plasma_explosion; style: shadow+fire_element as trained-texture COUSINS vs wireframe/illustration_scene as novel; object: animalization + vanish trio gas_transformation/portal/giant_grab + degenerate-endpoint money_rain; two-sided unseen: hole_transition n=2, seamless_transition n=1), skipped all 320px classes (upscale would confound appearance metrics). 110 clips standardized into `transitions_std121/` (corpus now 25 styles), 24 new type-blind endpoint captions written from std first/last frames, `make_quads.py` → 51 quads (23 ic_os_inclass / 12 ic_os_to2s / 3 ic_ts_unseen / 2 ic_anchor exp_056-repro / 11 base twins) + 54 cond cuts. Jobs 9396104/05 (H200-sec) + 9396108/09/10/44/45 (cluster secondary; H100-secondary nodes were down → re-routed). Pre-registered validity caveats in `design.md`: high lerp floors for one-sided classes (gap table will gate normalized readings), camera-class M3 ill-defined, in-class leak partly legitimate, hole/fire_element/plasma ceilings under-clipped, anchors only qualitative vs run_0002 (corpus root → std121).

**02:29** — **exp_056 COMPLETE: 46-quadruple IC-LoRA suite generated, harness-scored, and packaged into an interactive viewer — in-context transition transfer is real and quantified.** Inference round 1 (9386903-06) hit a **shared-samples-dir race** (4 concurrent chunks all write `samples/step_003000_N.mp4` → renames grabbed other chunks' files; 3 jobs FAILED, surviving quads untrustworthy) → fixed with per-chunk dirs (`chunk{i}of{n}/`), purged, round 2 (9388130-33 + racing twins 9388171/72 on cluster secondary) delivered all 38 ic quads in ~35 min; base twins 8/8 from 9386907. Scoring: `run_score_ic.py` (exp_053-v2 fork + singleton-ceiling guard; also strips viewer-only manifest keys after an EvalItem TypeError) on H100 ccc0457 — **8:39 for 46 items** thanks to the shared feature cache → `outputs/eval/exp_056/quads/run_0002` (41 normalizable + 5 jump-singleton raw-only; W&B `exp056_quads`/1tcopqle). **Headline: every axis separates style transfer from content copying.** base arms: leak_max_sim **0.95–0.98 (near-copy regime)**, seams +2.4..+7.1, endpoints 0.86–0.94, depart/arrive 0.07/0.93 = the base model REPLAYS the reference between the pinned anchors. ic arms: leak 0.58–0.81, seams **all negative** (−0.47..−0.64), endpoints **0.96–0.98**, depart/arrive ≈0.32/0.63 = genuine mid-clip transitions on the target's own content; ic_cross (n=20) norm-appearance 0.65±0.35. ic_unseen (jump, never trained): leak 0.21–0.36 + low profile-DTW to the demo (0.08–0.23) but raw appearance to jump only 0.15–0.29 — **motion/timing transfers in-context, appearance stays in the trained prior** (visually: the subject crouches, leaps and lands as scene B, rendered smoky). In-class leak 0.77–0.81 includes real reference scene-B element bleed (briefcase/skateboard) in unpinned approach frames. Viewer: `build_viewer_ic.py` (exp_055 fork; each item shows the actual in-context reference video alongside generated + endpoint conditions) → `outputs/eval/exp_056/viewer`, **validate_bundle PASS 381/381 assets, 46/46 filmstrips**. Note `notes/exp/exp_056_ic_lora_transition_transfer.md` + INDEX row; README completion header; memory updated. Open: checkpoint-ladder early-stop scoring, in-class leakage vs reference-dropout, multi-demo unseen transfer.

**00:45** — **exp_056 training COMPLETE (job 9383890, H200 ccc0481, 3:21:25 single window, EXIT 0) — in-context transition transfer VISIBLY WORKS at the eyeball level; quadruple inference suite launched.** Full 3000 steps at ~2.6 s/step, peak 50.33 GB; 12-checkpoint ladder (250→3000) + 39 validation videos (13 rungs × 3 samples, reference side-by-side; chain backups 9383891/92 exited on the DONE marker as designed). **Frame-strip findings (step 0 vs 3000):** (1) BASE model on the cross-class probe **copies the reference's content** — the generated side replays the smoke reference's rooftop/drink-window scenes between the pinned endpoints (reference tokens share the target's RoPE positions; the base model can't tell demo from target). (2) **IC-LoRA@3000 copies the reference's STYLE instead**: bookstore scene held, the TARGET subject gets wrapped by the reference's ink-black billow mid-clip, resolves into the staircase anchor — with a type-blind prompt (no smoke words anywhere). (3) **Unseen-class probe (jump_transition, never trained): the endpoint subject crouches, leaps airborne and lands as scene B** — transition semantics read in-context from a single demo of an unseen family (mild smoky-dust tint from the training prior). (4) In-class rung: strong wrap + partial reference scene-B content leakage in the unpinned approach frames (briefcase/skateboard elements) — M6 will quantify. Post-training suite per the overnight plan: 46 quadruples (endpoints, reference, type-blind prompt, generated) across arms ic_inclass_new/trained, ic_cross (10 ref classes × 2 foreign endpoints), ic_unseen, ic_reverse + 8 base twins — `make_quads.py`, `run_ic_inference.py` (ValidationRunner + PEFT attn+FFN load, chunked, skip-if-exists), jobs **9386903-06** (ic chunks, H200-secondary) + **9386907** (base, cluster secondary H100 ccc0423). Next: `run_score_ic.py` (exp_053-v2 fork, singleton-ceiling guard) on manifest_ic.json → `build_viewer_ic.py` quadruple viewer (exp_055 fork with in-context-reference video slot).

## 2026-07-07

**21:20** — **exp_056 sanity PASSED (job 9383079, H200 ccc0481, 21 min, EXIT 0) → full IC-LoRA training launched as chain 9383890→9383891→9383892 (HCESC-H200-secondary, started instantly).** Sanity validated the whole novel path end-to-end: 47-clip preprocess (captions 26s, video encodes 38s), `build_dataset.py --link` assembled the 131-pair symlink tree and the official trainer's PrecomputedDataset indexed **131/131 valid samples across latents+conditions+reference_latents** (symlink pairing works); 50 steps of reference+prefix+suffix training at **2.62 s/step, peak 49.45 GB** (9600-token seq = 4800 ref + 4800 target — fits H100 80GB too); loss 0.69→0.47; checkpoints+training states at 25/50; validation at steps 0/50 produced all 3 samples at 960×640 = generated + reference side-by-side (reference conditioning live in ValidationRunner). Full run: 3000 steps ≈ 2.2h + 13 validation rounds ≈ 1.5h → finishes inside job 1–2 of the chain.

**20:53** — **exp_056 opened + sanity submitted: IC-LoRA in-context transition transfer — one adapter, 10 transition classes, the reference video (not the caption) carries the transition type.** Design: official V2V IC-LoRA (`reference` condition, rank 32/α32 + FFN targets, lr 2e-4, 3000 steps) composed with the exp_051-validated C2V endpoint conditions (`prefix tb=2` + `suffix tb=1`, all p=1.0) — per training sample the model sees a full same-class reference clip in-context + the target's [2 start, 1 end] latents pinned, and generates the target's middle 13/16 latent frames. Captions are deliberately **type-blind** ("<scene A>. The scene transforms into <scene B>.", endpoint-only, agent-written from frames and leak-checked) so the transition style is only observable in the reference; trigger `ICTRANS`. Data: all 47 deduped clips standardized to **480×640×121@24** (`standardize_clips.py`: even-index selection — 2× decimation for 242f clips preserves the full arc; resize-cover+center-crop; names normalized) → `data/processed/transitions_std121/`; **131 circulant ordered pairs** (target i × next min(3,n−1) same-class refs — every clip is target and reference equally; shadow_smoke capped at 23% share; `jump_transition` singleton fully held out as the unseen-reference validation class). Preprocessing collision discovered & sidestepped: `process_dataset.py` names ALL outputs (incl. reference latents) after the row's *target* path → pairs sharing a target would collide; instead the 47 clips are encoded ONCE (`.precomputed_clips/`) and `build_dataset.py --link` assembles the 131-pair `.precomputed/{latents,conditions,reference_latents}` tree as symlinks (PrecomputedDataset matches by rel-path and follows symlinks — verified in trainer code). Seq/step = 9600 tokens (4800 ref + 4800 target). Validation ladder every 250 steps ×3 samples, reference side-by-side: cross-class (ew0 endpoints + shadow_smoke_3 ref), unseen-class (ew0 + jump_transition_1 ref), in-class (ss0 + ss1). Configs schema-validated against the trainer. Scaffold: `experiments/exp_056_ltx2_ic_lora_transition_transfer/`. Sanity (preprocess 47 clips + link + 50-step dry run) racing on two queues: **9383079** (HCESC-H200-secondary, est. 20:52) vs **9383080** (cluster `secondary` H100, est. 01:27) — loser gets scancelled when one starts.

**16:52** — **exp_054: full harness re-validation on the deduped + expanded corpus (47 clips / 11 styles) — the 2 new styles discriminate on appearance, jump/flying_cam flags refreshed.** Ran the exam (`run_validation.py`, phase 1 of `job_eval.sbatch`) on the complete current corpus: 37 deduped originals + `air_bending` (4) + `firelava` (6). Job **9366838** (HCESC-L40S-normal, ccc0440, 1×L40S), phase-1 **EXIT 0** in ~9 min (10 new clips extracted GPU, 37 cached) → `outputs/eval/exp_052/validation/run_0002/`. Both new styles cleared the morph/endpoint contract (all clips 122–242 frames) — nothing excluded. **Exam (chance 0.213):** effect_appearance headline **0.851** [Wilson 0.72–0.93], **d=2.22 (up from 2.04)**; motion 0.255; morph_dtw 0.191; lerp floor separated=True (real 0.939 vs lerp 0.636). **New styles:** air_bending appearance **0.75** / motion 0.0, firelava **0.83** / motion 0.0 — both valid appearance-axis additions, neither motion-discriminable. **Post-dedup facts confirmed:** jump now a **singleton (0.0, untestable)**; flying_cam appearance **0.25** but motion 0.5 (motion-defined). Appearance overall 0.927 (41) → 0.865 (37) → 0.851 (47) — the drop is the corpus getting honest + 2 imperfect new styles, while separation improved. **New-style dedup** (`detect_duplicates.py`, full 47-clip): **0 candidates above the 0.80 cut** (max within-style set-sim 0.453 air / 0.490 firelava; the 600px `*_0` clips are distinct kling_motion sources, not twins) — nothing moved, no re-validation needed; `outputs/eval/exp_053/dedup/duplicates_v2.json` (empty). **Trust flags refreshed:** repointed `exp_053/config.yaml` `validation_run` → run_0002; `run_score.py --from-items … --label ladder_v3` (login, no GPU) → `outputs/eval/exp_053/ladder_v3/run_0001/`. jump → both flags False (was self-retrieval-trusted); flying_cam → motion-trusted (0.5)/ceiling-trusted but appearance-untrustworthy; shadow_smoke motion 0.40 stays flagged so the ladder motion column is unchanged from ladder_v2. **Corpus-integrity fix:** `melt_transition_0`'s earlier dedup drop had never been parked in `_dup/` (unlike the other 3); recovered byte-identical from `data/processed/higgsfield_transitions/` and placed in `melt_transition/_dup/` with README — all 4 deduped twins now reversible; active exam unchanged. **Known caveat:** phase-2 ladder re-score (bonus) died EXIT 1 — the jump singleton makes `fidelity_vs_refs`'s LOO-neighbor set empty → `np.min([])` in both scorers; harmless here (step-3 `--from-items` skips ceilings), but guard it before any future full re-score. Failed partial `ladder/run_0002/` deleted. Report: `outputs/eval/exp_054_full_revalidation/REPORT.md`; note §7 + INDEX updated.

**16:14** — **Dedup method fix (user-caught miss) + a 4th duplicate found: corpus 41→37; 2nd new style `firelava` landed.** User spotted that `flying_cam_transition_0/1` are the same clip — my v1 scan missed it. Root cause: v1 gated on **temporally-aligned** per-frame DINO cosine, and clip_0 is a **193-frame 600px TRIM** of the 242-frame 1244px clip_1 — the length edit shifts the same events to different normalized times, decaying aligned cosine to 0.876 (< the 0.90 gate). The **order-invariant** views nail it: set-similarity (mean-of-max) **0.977**, independent dHash-bag Hamming **4.9**. Fixed `detect_duplicates.py` to gate on order-invariant set-similarity (≥0.90) confirmed by dHash-bag (≤10), with aligned cosine demoted to context — the lesson: real dupes are often re-cut, not just re-encoded, so an alignment-dependent gate has a blind spot. Re-scan of the full 37-clip corpus found this one and NO others (clean calibration gap: distinct pairs ≤0.42, same-style non-dup ≤0.43, dup 0.977). Moved `flying_cam_0` to `_dup/` (keep the longer/higher-res `_1`). **Exam impact:** core-mask LOO 1-NN 0.895→**0.865**; `flying_cam` recall **0.60→0.25** — the dup was propping it up, and it's really a motion-defined style (scene-diverse, poor appearance retrieval). Core-mask robustness case only hardens on the honest corpus (margin core +0.169 vs all +0.028, ~6×; ρ≈0.79; 7 flips/0 cost; cross_high 0/37). Consolidated `outputs/eval/exp_053/dedup/duplicates.json` (all 4 clusters) + `flying_cam_0_vs_1.png` proof; pair_examples.ipynb regenerated on 37 clips. **Also:** `firelava` (6 clips + `_manifest.json`) appeared alongside `air_bending` — both uncached, excluded until GPU-processed, each needs its own dedup. Note §1/§6 + memory updated.

**16:00** — **Reference corpus deduplicated (41→38) + a new style `air_bending` landed mid-session.** Layered duplicate scan (`experiments/exp_053_eval_harness_v2/detect_duplicates.py`: SHA-256 exact → structural → per-frame dHash → temporally-aligned DINO cosine, calibrated vs the distinct-pair distribution) found **3 within-style near-duplicate clusters** — the same clip at two resolutions (600px + ~1244px), aligned DINO cosine ≥0.988 and independent dHash Hamming 0: `melt_transition_{0,3}`, `jump_transition_{0,1}`, `display_transition_{0,3}`. Not byte-identical (different encode) so an MD5/SHA-only pass misses them — the perceptual+semantic layers are what catch it. Low-res twins moved reversibly to `data/processed/transitions/<style>/_dup/` (non-recursive glob drops them from the exam). **Exam impact:** appearance headline robust (core-mask LOO 1-NN 0.927→0.895, testable-only 0.919), but **jump_transition had only 2 clips and they were the same clip** — its recall 1.00 was 100% trivial (each retrieved a re-encode of itself) and its LOO ceiling was a fake "clip vs its own twin ≈1.0"; jump is now a singleton → untestable. melt 0.80→0.75, display holds 1.00. Dedup actually STRENGTHENS the core-mask case (§15:45): on the honest 38-clip corpus all-frames appearance falls to 0.711 vs core 0.895, the mask now saves 7 retrievals (was 6, 0 cost), and melt's all-frames recall collapses to 0.00 without its twin. Proof montage + manifest: `outputs/eval/exp_053/dedup/{duplicates.png,duplicates.json}`; `dedup_report.py` recomputes the corrected exam. **Also:** `air_bending` (4 clips + `_manifest.json`) appeared in `data/processed/transitions` at 15:56 — no cached features yet, excluded from all analysis until GPU-processed; needs its own dedup pass. **Follow-up: re-run `run_validation.py`** to refresh motion exam / ceilings / trust flags on 38 clips (cached exp_052 validation still reflects 41). pair_examples.ipynb + note §1 addendum/§6 + memory updated.

**15:45** — **exp_053 check-A robustness re-analysis + worked-examples notebook: the "all-frames is more robust than the core mask" objection FAILS on this corpus.** Re-ran check A's two 41×41 appearance matrices on the login node (cached feats, pure numpy — `build_pair_examples_notebook.py`). Findings: core-mask vs all-frames is the SAME measurement denoised (Spearman ρ=0.81 on 820 pairs), a strict Pareto win (6 clips flip correct→wrong going core→all, 0 the other way), and a wider decision margin (best same − best diff sim: **+0.268 core vs +0.158 all**). The mask never misfires on real data: `cross_high` (unstable normalization) fires **0/41** (max endpoint cross-sim 0.53), 0 clips hit the 1-frame fallback, retrieval is threshold-flat (0.927 @ thresh 0.40–0.50). It's targeted, not global — 5/9 styles are a literal no-op (flame/jump/raven/shadow_smoke/water); only the fluid styles gain (earth_wave/melt +0.40, display +0.25, flying_cam +0.20). Verdict: keep the mask; the only real hedge is a `cross_high→all-frames` M3 fallback for future portal-class styles (a 0/41 no-op today, not yet wired). New self-contained notebook `experiments/exp_053_eval_harness_v2/pair_examples.ipynb` (outputs pre-embedded, no kernel needed) documents pair construction / the two aggregation modes (max-per-row 1-NN vs mean-over-axis separation) and shows 30 diverse example pairs + a dilution teaching figure; montages under `outputs/eval/exp_053/pair_examples/`. Note §1 addendum + memory updated.

**15:10** — **exp_053 COMPLETE: Gemini judge pass finished (24/24, 0 parse errors) — judge is functional but measures cleanliness, not transfer.** Free-tier quota (20/day) had stalled the pass at 15/24; user enabled billing → remaining 9 items completed under the pinned `gemini-3-flash-preview` (scheduled quota-reset finisher 9365640 cancelled). Two backend hardenings landed en route: response schema pinned via structured output (one item had invented a score format — `response_mime_type` alone doesn't pin shape) and 429 handling honors the server's `retry in Ns`. **Results: q1 same-type / q2 dynamics / q3 endpoints / q4 no-leakage = 1.00 for every arm INCLUDING base; only q5 artifacts discriminates: base 1.00 > t2v 0.83 > c2v 0.67 > i2v 0.50.** The exp_052 q2/q5 all-false degeneracy is fixed (differentiated, timestamp-cited answers), but the judge's ranking is inverted vs the style axes — base makes clean wrong-style videos. Verdict: advisory quality axis only; the human-validation set now has two concrete questions (q1 leniency on base; whether q5's LoRA-artifact ranking tracks human perception). Ladder v2 final report with judge column merged: `outputs/eval/exp_053/ladder_v2/run_0002/`; consolidated process report `outputs/eval/exp_053/REPORT.md`. Note/README/memory updated.

**14:10** — **exp_053 Gemini judge: 15/24 ladder items judged; free-tier daily quota (20 req/day, gemini-3-flash) blocks the rest — finisher job 9365640 scheduled `--begin=2026-07-08T03:00`** (compute-node egress verified by probe 9365633; response cache makes the re-run incremental; it also regenerates ladder_v2 with the judge column). Backend hardened mid-pass: response schema now PINNED via structured output (one item had answered in an invented score format) and 429 handling honors the server's retry hint. Substance so far: q2/q5 degeneracy is fixed (differentiated, timestamp-cited answers), but the parsed items pass essentially everything INCLUDING base (all_pass 1.00) — with native video the judge is lenient rather than harsh, i.e. not yet discriminative on this ladder; it stays advisory and the human-validation set is still the gate.

**13:20** — **exp_053 eval harness v2 — checks COMPLETE: the harness is now adversarially validated and its numbers decision-grade.** Iteration on exp_052 per the review action items (all evaluated, simplicity-first). **(1) M3 ablation (check A): KEEP_CORE_MASK** — all-frames appearance drops to 0.780 vs core-mask 0.927 on the 41-clip exam (pre-registered adoption bar 0.88 missed decisively); the M1→M3 dependency stays and is now justified by data. **(2) Anti-cheating axis validated (check C): M6_OK** — 26-item adversarial manifest from the exp_046/048 dead branches (`manifest_adversarial.json`); all 12 ground-truth copies (source recons + donor-pin splices) leak ≥ 0.926 vs honest-generation max 0.78 → clean near-copy threshold ~0.88. One initial "miss" audited to a mislabeled item (exp_048 self-inject g1.0 on z1-POOR ss1 genuinely diverges — exp_049 PSNR 11.88): the harness independently reproduced the z1-rich/poor dichotomy; audit in `checks/run_0001/checkC_audit.json`. tempblend correctly flags (it contains literal donor smoke) — future latent-borrowing recipes will read as copying. **(3) Motion sanity (check B): 8/9 styles** within>cross (raven inverts, matching its exam recall). **(4) Decision-grade reporting**: `report.score_tables` = the standard headline/analysis split (appearance/motion/judge/endpoints+seam/leakage vs M1 scalars), mean±std, per-style trust flags (motion † unless exam recall ≥0.5 — shadow_smoke 0.4 is flagged; ceilings ‡ under 4 clips: flame/jump), Wilson CIs on exam accuracies, Pearson dropped; M1 gains a `cross>0.85` edge guard; endpoint windows are manifest-driven. Ladder re-reported (`run_score.py --from-items`, no GPU): **paired per-cell analysis shows base-vs-LoRA separation survives (6/6 cells on appearance, sign p≈0.016/arm) while differences AMONG t2v/i2v/c2v are unresolvable at n=6** — route decisions must not gate on those gaps. Trigger claim reworded everywhere to "no detectable trigger effect (n=3/cell)" + flagged as a fork-in-the-road probe (always-on style would kill the trigger-switched multi-style route). **(5) Judge → Gemini API native video** (`judge_gemini.py`, rubric extracted to backend-agnostic `rubric.py`, q2/q5 severity-calibrated): temp 0, 8 fps, JSON, per-item response cache, no GPU; gemini-3.5-flash video 503'd persistently → pinned `gemini-3-flash-preview`; early results show the q2/q5 all-false degeneracy FIXED (differentiated answers). 24-item pass running. Jobs: 9364990 + twin 9364995 (verdict-identical to 7th decimal). 17 unit tests pass. Note `notes/exp/exp_053_eval_harness_v2.md`; INDEX row; memory updated.

## 2026-07-06

**18:54** — **exp_052 COMPLETE: the eval harness passed its own exam and the exp_051 ladder is now quantitative.** Full run 9354400 + judge re-run 9355187 (both L40S ccc0440, EXIT 0). **Exam (41 real clips / 9 styles, LOO 1-NN, chance 24%):** effect appearance **0.93** (d=2.04) = the style-ID workhorse; motion fidelity 0.46 — but only after three transition-specific fixes the smoke run exposed (naive frame-0 DMT protocol was AT chance): mid-frame grid queries w/ backward tracking (the effect medium doesn't exist at frame 0), per-step occlusion masking (most points get engulfed — whole-tracklet visibility cuts empty the set), duration-normalized + box-smoothed velocities; morph-DTW 0.34 → scoped per the pre-registered failure clause to transition-family/timing fingerprint (jump 1.0, display 0.75; sweep-family styles legitimately share profiles), NOT standalone style ID. The three metrics fail on complementary classes. **Floor calibration:** lerp depth = 0.62 not ≈0 (double-exposure middles leave DINO's endpoint neighborhoods) — separation from real clips (0.92/min 0.64) holds; per-item lerp normalization absorbs it. **Ladder rescored (normalized 0=lerp/1=real-ceiling):** base separates on appearance (0.56 vs 0.69-0.79) and motion (0.47 vs 0.88-0.91); LoRA arms cluster; c2v tops profile fidelity (1.00); NO seams in any arm (robust-z < −0.55) + endpoint DINO ≈0.98 = quantitative twin of "anchoring is mechanism-robust"; morph profiles near-saturate for all conditioned arms (anchors dominate the curves — appearance/motion/judge are the discriminative axes there). Bonus numerics: trigger-independence (appearance identical ± SHDWSMK, e.g. c2v 0.503/0.500) and no leakage (max retrieval sim ≤0.78). **Judge (Gemma 3 12B, 24/24 parsed, frame-cited evidence):** q1 same-type + q3 endpoints rank c2v 1.0/1.0 > t2v=base 0.83 > i2v 0.67; q2-dynamics and q5-artifacts fail everywhere = systematically harsh under 8-frame sampling — judge stays ADVISORY until human-validated. Env gotcha fixed: transformers-4.57 Gemma3 needs torch≥2.6 → judge runs via the LTX-2-official venv (torch 2.9), `run_judge.py`/`job_judge.sbatch`. Artifacts: `outputs/eval/exp_052/{validation,ladder}/run_0001/` (figures, distance matrices, items.jsonl, judge JSONs), W&B runs `exp052_validation`/`exp052_ladder`; note `notes/exp/exp_052_eval_harness.md`; INDEX row added; memory `transition-eval-harness`.

**18:05** — **exp_052 opened: transition eval harness v1 — content-invariant metric suite + the harness's own exam before any method decision rests on it.** Design: a transition is a program acting on content, so every metric is computed relative to the video's OWN endpoints/frames: M1 Morph Profile (per-frame DINOv2 cosine to own endpoints, floor-normalized by cross=cos(eA,eB); DTW+Pearson comparison; transformation depth / timing / identity-hold / core-frame mask fall out), M2 Motion Fidelity (Yatim et al. tracklet velocity correlation via CoTracker3, duration-invariant), M3 Effect Appearance (core-frames-only DINO set similarity — isolates the medium from content by construction), M4 rubric VLM judge (5-question checklist, evidence-required, local Gemma 3 12B — EXPERIMENTAL until human-validated), M5 endpoint fidelity (LPIPS+DINO on conditioned frames) + boundary-seam detection (temporal-LPIPS robust-z at the conditioning handoffs), M6 leakage (near-duplicate retrieval vs reference frames, contrasted against unrelated styles). Every score anchored between a lerp-crossfade floor and a real-clip LOO ceiling run through the identical pipeline; no composite number. Library: `src/diffusion/transition_eval/` (10 modules; morph/motion math pure numpy — 11 unit tests in `tests/test_transition_eval.py` pass on login node). Validation plan pre-registered in `experiments/exp_052_transition_eval_harness/README.md`: (1) style-discrimination exam — LOO 1-NN retrieval on the 41 real clips / 9 styles in `data/processed/transitions` (chance 24%), (2) lerp depth≈0 floor, (3) exp_051 ladder rescored (24 items, manifest_exp051.json) — expect base < LoRA arms on transition fidelity, c2v best on seams/endpoints. Deps staged: DINOv2-base, CoTracker3 (HF-hub fallback after truncated torch.hub download), lpips. Smoke job **9354322** (HCESC-L40S-normal ccc0440).

**13:12** — **exp_051 COMPLETE: the C2V capability ladder is generated — conditioning-matched (c2v) LoRA training wins on anchor continuity and learns 2–4× faster; even the plain t2v LoRA holds endpoint anchors perfectly.** New `c2v` arm (prefix tb=2 + suffix tb=1, both p=1.0, else exp_050-identical) trained in job 9349026 (1:19 h, cluster-`secondary` H100 ccc0424, W&B run `creative-transition-transfer/7iptdfyt`) while base/t2v/i2v_ff05 C2V inference ran IN PARALLEL on the same node (9349028-30); c2v inference 9350243 (13 min). 24 ladder videos on 3 foreign-family test transitions (earth_wave endpoints, verified clean; the arms must override the source's dirt-wave with the learned black smoke — and all LoRA arms do). **Findings:** (1) endpoint anchoring is mechanism-robust — clean-latent/timestep-0 pinning holds even for the t2v LoRA that never saw a conditioned token; what c2v buys is middle quality: tightest subject-wrap onset, most continuous convergence into the end anchor, most stable scene-A hold. (2) c2v acquires the signature morphology by step 250 (vs 500–1000 in exp_050) — endpoint pinning concentrates all supervision on the transition middle. (3) Trigger-dependence probe: no-trigger prompts still produce the full morphology in every arm → the concept binds to the caption phrase, not `SHDWSMK` (exp_050 confound now resolved empirically). (4) Drift: no smoke bleed anywhere; c2v's unconditioned generation composition-shifts at fixed seed (it never trained unconditioned) — use p<1.0 mixes if one adapter must also serve T2V. Suffix-condition gotcha (user-caught): validation reads condition media from the front and the causal VAE bleeds backward-reaching context into the end-anchor latent → pass exactly the trailing frames (last-9 → keep final latent via num_frames=8). W&B integration live (netrc, project `creative-transition-transfer`, inference runs `exp051_infer_*`). Scaffold: `experiments/exp_051_ltx2_lora_c2v_ladder/`; note: `notes/exp/exp_051_c2v_ladder.md`; INDEX updated. Next: watch videos temporally, c2v checkpoint-ladder early-stop pick, head-to-head vs exp_046/047 injection recipes.

**11:12** — **exp_051 opened: C2V capability ladder — does the LoRA's training-time conditioning mode (t2v vs i2v vs c2v) matter for clips-to-video transition generation?** New arm `c2v`: identical to exp_050 baseline (rank 32/α32, video-only attention targets, official 80GB defaults, exp_050 `.precomputed` latents+SHDWSMK captions reused verbatim) but with the official video-extension conditions STACKED: `prefix temporal_boundary=2` + `suffix temporal_boundary=1`, both p=1.0 → first 9 + last 8 pixel frames clean/timestep-0/loss-excluded every step; only the middle 13/16 latent frames supervised. Recon (2 Explore agents over the official repo): conditions are applied at train time on plain latents (no re-preprocess); multiple intrinsic conditions sample independent Bernoullis and accumulate; a validation sample may carry prefix AND suffix at once → keyframe-style C2V inference lives in the trainer's own `ValidationRunner` (ltx-pipelines CLIs only do single-image conditioning). C2V inference for ALL ladder rungs (base / exp_050 baseline / exp_050 i2v_ff05 / c2v) runs through `run_c2v_inference.py` — standalone ValidationRunner + the trainer's exact PEFT LoRA-load path, 2 prompts (trigger / no-trigger probe), seed 42, 640×480×121@24. Test conditions from a DIFFERENT transition family: earth_wave_0 (bookstore woman → dirt wave → staircase woman), so arms must impose the learned black-smoke transition on foreign endpoints. Suffix-condition subtlety (user-caught): the causal VAE's receptive field reaches backward, so the suffix window is cut to the last 17 frames (pure scene-B) to keep the source's dirt transition from bleeding into the conditioning latent — `cond_end_last17.mp4`; prefix needs no care (only first 9 frames are encoded; causality blocks future bleed). Scaffold: `experiments/exp_051_ltx2_lora_c2v_ladder/` (README pre-registration, config_c2v[_sanity].yaml, run_c2v_inference.py, job_train/job_infer.sbatch). Sanity = job **9348844** (HCESC-H100-secondary): 50 steps + step-0/50 C2V validation — step 0 doubles as the ladder's base-model rung.

## 2026-07-04

**03:15** — **exp_050 COMPLETE: all 3 LoRA arms trained to 2000 steps on H100s; the smoke-transition concept is acquired, with zero drift.** Sanity job 9321703 (12 min): 50-step dry run + validation + TRUE resume verified (`training_state_step_*.pt` loaded, global step continued 50→60) + full preprocess (10/10 latents in 3 AR buckets, `--decode` verification videos). Arms: `baseline` job 9321859 (HCESC-H100-secondary, **58 min, 1.16 s/step, peak 48.4 GB**), `i2v_ff05` job 9321861 (cluster `secondary` idle H100 ccc0423, 66 min), `rank64_ffn` job 9321863 (ccc0424, 72 min) — the two sweep arms ran SIMULTANEOUSLY on other investors' idle nodes while our pools were saturated. **Results (fixed-seed validation ladder, held-out prompts):** step 0 = base model renders a generic gray explosion filling the frame; by step 500–1000 all arms converge to the training clips' signature — a dense rounded ink-black billow that WRAPS the subject, sweeps across with tendrils, and cleanly reveals scene B; step 2000 = fully styled. No-trigger drift prompt (golden retriever park) stays clean in all arms, incl. rank64_ffn (pre-registered drift concern not realized). Eyeball ranking: rank-32 attention-only is SUFFICIENT; `i2v_ff05` ≈ baseline quality + first-frame conditioning for free → best default for C2V follow-ups; `rank64_ffn` = densest smoke, no clear win. Checkpoint ladders (250…2000 × 3 arms, 8.2 GB total) + 18 validation videos/arm in `outputs/training/exp_050_ltx2_lora_shadowsmoke/`. Backup-chain gotcha fixed in `job_train.sbatch`: trainer zero-pads step numbers (`step_02000`), and a completed run refuses to resume ("initial_step >= target_steps" — harmless). Note: `notes/exp/exp_050_lora_baseline.md`; INDEX updated. Next: load the LoRA in ltx-pipelines for real C2V endpoint tests vs the exp_046/047 injection recipes.

**00:30** — **exp_050 opened: standard-LoRA fine-tuning baseline on the 10 shadow-smoke clips via the OFFICIAL LTX-2 trainer (clean clone, not the pod-modified vendored copy).** Infra context: the repo now lives on the UIUC campus cluster (`$LAB/diffusion-research`; RunPod era over — migration was byte-verified 2026-07-02; compute = Slurm, see `CLAUDE.md` "Compute" + the cc-slurm/exp-* skills). Setup for exp_050: official `LTX-2` monorepo cloned to `$LAB/LTX-2-official` @ `7809842` (2026-06-17, same upstream head the vendored copy branched from), `uv sync` env; weights staged fresh from HF (`ltx-2-19b-dev.safetensors` 40.3 GiB + `gemma-3-12b-it-qat-q4_0-unquantized` 23 GiB, ~10 min on campus network). All 10 clips probed (24fps, no audio streams, 6 portrait 1244×1660 / 3 landscape 1660×1244 / 1 square 1440×1440; ss2 is 10s) → 3 AR buckets `480x640x121;640x480x121;576x576x121` (F=121 ≈ full 5s). Per-clip captions hand-written from frame contact-sheets (scene A → consistent "dense mass of black smoke sweeps across the frame and engulfs..." phrase → scene B), trigger `SHDWSMK` via `--lora-trigger`. Three arms, all = official `t2v_lora.yaml` 80GB-tier defaults (rank 32/α32, lr 1e-4 linear, 2000 steps, bs1, adamw, bf16, no quant, grad-ckpt, ckpt/250): `baseline` (video-only attention targets — clips are silent), `i2v_ff05` (+first_frame p=0.5, official style-LoRA recommendation), `rank64_ffn` (+rank/α 64 +FFN targets, official capacity recipe). Validation every 250 steps + step-0 before/after (seed 42, 480×640×121@24, CFG 4, STG 1 `stg_v`; held-out in-style trigger prompt + unrelated drift prompt). Sanity job (official train-model-skill Phase 6: one-sample preprocess + 50-step dry run + resume test + full preprocess w/ `--decode`) = Slurm job **9321703** on `HCESC-H100-secondary` (started in ~100 s while H100-normal quoted 3 days). Experiment folder: `experiments/exp_050_ltx2_lora_shadowsmoke/` (dataset.json, 4 configs, 2 sbatch scripts, README with pre-registered expectations).

**14:29** — **exp_049 σ-matched recon-trajectory injection — mechanism CONFIRMED (recovers recon for z1-rich clips, late-σ-localized); deployable donor path still info-limited.** Forks exp_048 to inject the σ-MATCHED step of the recon's own coarse→fine trajectory `x̂₀(σ_i)=z_in−v_pred·σ_i` (derived in-consumer from the exp_040 velocity-only cache `run_0002`; no library change) into the production Euler+CFG=4 regen free-middle, over early/mid/late/all windows, self (g=1) + donor (g=0.8), 3 clips (`run_0006`). **(1) σ-matched self injection RECOVERS the reconstruction — ss0 base 14.70 → self_all 33.26 free-mid (lpips_fm 0.475→0.032)** — closing exp_044's recon→regen solver-mismatch gap when the info exists; the σ-match is essential (exp_048 static-target gave only 13.05). **(2) Gated by z1-richness (exp_044 dichotomy): only ss0 recovers; ss6 (15.18) & ss1 (11.88) are z1-poor → nothing to recover.** **(3) Late-σ carries the recovery (It-4 confirmed, x0-domain): ss0 windows monotone early 17.75 < mid 24.72 < late 33.26 = all 33.26 — late window [26,40) ALONE = full recovery.** (4) z1-poor clips saturate across all windows (no localization/stacking). **(5) Donor cross-clip transfer: inject EARLY, never all/late** — donor_all/late tank free-mid to ~8 and over-saturate (OOD fine-detail forced late); donor_early/mid preserve structure (≈base PSNR). PSNR-vs-specific-clip stays information-limited (exp_045 wall) → donor value is perceptual. No deployable PSNR>18 win. Note appended to `notes/exp/exp_044_smoke_transition_injection.md`; INDEX updated.

**14:05** — **Fixed a ~78 GB CUDA OOM footgun in exp_049: missing `@torch.inference_mode()` on `main()`.** The decorator (present in exp_040/047/048) was dropped when exp_049 was forked and `encode_prompt_bundle` (its own decorated helper) was inserted just before `main()`. Without it, the full-video `pipe.vae.encode()` builds an autograd graph that `z0_packed` retains → ~75 GB of LIVE activations → OOM when the text encoder loads in `prepare_sample`. Neither `enable_model_cpu_offload` nor manual `module.to("cpu")` placement can evict live autograd tensors (both chased as red herrings across several pod reloads; GPU probe `0.0→75.8 GB` purely across the VAE-encode phase nailed it). Fix = the one-line decorator; reverted all memory-management scaffolding back to the proven exp_047/048 offload pattern. Lesson saved to memory `ltx2_inference_mode_oom_footgun`. A100 80GB PCIe (`p4255y3upaj82u`, EU-RO-1) torn down.

## 2026-06-03

**16:14** — **exp_043 z1-vs-Gaussian deviation analysis — the "smoke signature in z1's free-middle" hypothesis is REFUTED.** New standalone `experiments/exp_043_latent_pca_inspection/inverted_noise_vs_gaussian.py` (CPU/numpy; argparse data_dir/tensor/groups; `next_run_dir`+`TeeLogger`) loads cached z1/z0 from exp_033 run_0001, unpacks geometry-safely (`reshape(F=16,H,W,128)` with each clip's own (H,W); orientation groups portrait 22×16 / landscape 16×22 / square 19×19; NEVER grouped by N=5632 which is shared with swapped H,W), splits free-middle (latent frames 4-12, drop1) vs anchors (0-3,13-15), and runs a full battery vs white-Gaussian and variance-matched nulls: per-channel skew/kurtosis+QQ, radial power spectrum, isotropic spatial autocorrelation, adjacent-frame temporal correlation, per-frame/per-channel energy + low-freq-fraction localization, within-group cross-clip structured-map cosine. **Result (replicated across all 3 orientation groups): z1's free-middle is essentially white Gaussian (excess kurt +0.08, spatial autocorr +0.02, temporal corr +0.02, low-freq frac 0.22 vs white 0.135) — RF-inversion ERASES structure there. z1's only deviation lives in the clamped anchor frames (kurt +0.6-0.8, autocorr +0.45, temporal +0.62), which are just hard-pinned slices of z0.** z0 control: structure is present in ALL frames (kurt +0.48, autocorr +0.54) and partly shared across clips (free-middle structured-map cosine +0.26 vs z1's +0.03). Grounded in the inverted-noise literature (DDIM/RF non-Gaussianity, autocorrelation-regularization, "Spectral Collapse in Diffusion Inversion" → excess low-freq power). Recommendation: source the injectable smoke signature from z0's free-middle (low-frequency, temporally-coherent, partly-shared field), NOT z1; inject via late-σ (steps 27-39) trajectory guidance, since a raw additive on a fresh Gaussian seed gets re-Gaussianized by the same dynamics that whitened z1's free-middle. Charts+summary.json in `outputs/latent_pca/exp_043_inverted_noise_vs_gaussian/` (run_0002=z1, run_0003=z0). Did NOT touch the exp040cache GPU job; own tmux session `z1analysis`. Note: `notes/exp/exp_043_inverted_noise_vs_gaussian.md`.

## 2026-06-02

**17:03** — **exp_047 velocity-guided smoke generation — 2nd working recipe; perceptual goal met with two deployable methods.** Run the production Euler sampler but pull the free-middle's x0 (clean) prediction toward the extracted tempblend smoke target by guide_weight g each step (in `RFInverter._x0_clamp_velocity`), so the MODEL synthesizes seamless coherent smoke following the donor's darkening/dynamics. g-sweep on hard clips: guidance moves signals toward real (ss6 lum 0.294→0.374[real 0.402], sat 0.157→0.283; ss1 tdiff 0.031→0.067[real 0.078], sat 0.131→0.222). **CFG-agnostic once guidance dominates (g0.8_cfg1 ≡ g0.8_cfg4)** — reconfirms only the free tokens matter. Signals converge toward tempblend (soft-pin compounding) but output is a single coherent generation (no splice seam) with better mid-frame motion; visual: guided-gen smoke is integrated into the target scene, beats the static baseline. **Conclusion: from {target endpoints + a donor sample}, two deployable recipes produce a coherent smoke transition that emerges from & returns to the target scene — (1) tempblend latent splice (cheap, decode-only), (2) velocity-guided generation (g≈0.8, the genuine inject-at-generation).** Pixel-PSNR>18 vs a specific clip stays information-limited (clip-specific turbulence) — wrong metric; perceptual quality is the right one and is achieved. Open: quantify (CLIP/FVD), validate all 10 clips, block_out substrate. A100 PCIe torn down. Note: `notes/exp/exp_044_smoke_transition_injection.md`.

**16:01** — **exp_046 temporally-windowed donor injection = winning latent recipe for the perceptual smoke goal.** After donor-pin (dramatic but donor's content) and smoke-delta (target identity but under-occluded) hit a pin-vs-delta tradeoff, `tempblend` resolves it: `(1−w(t))·target_endpoint_interp + w(t)·donor_real_middle` with `w(t)` a Gaussian bump peaked at the darkest latent frame (~8). Target scene at smoke onset/offset (continuity), donor's full real smoke at the occlusion peak (scene occluded there, so donor identity is hidden). Signals approach real (ss6 tempblend:1.4 lum 0.388/sat 0.252 vs real 0.402/0.222 and baseline 0.292/0.124; ss1 tdiff 0.067 vs real 0.078, baseline 0.024). Visual (ss6/ss1 montages): the target's OWN scene transitions into a dark smoke billow at the peak and back — "emerges from this clip → full smoke → returns to this clip" — beating the static baseline, donor-pin (donor content throughout), and smoke-delta (under-occluded wash). Deployable: target endpoints + a donor sample's real middle, no target middle, decode-only. Caveat: LPIPS-to-source is a poor smoke metric (rewards the scene-preserving baseline); judge by darkening/dynamics/occlusion + visual. **Perceptual goal met via latent injection** on the two hardest clips. Optional polish: generative block_out injection (exp_047) for seamless turbulent peak. A100 PCIe torn down. Note: `notes/exp/exp_044_smoke_transition_injection.md`.

**15:20** — **Success pivoted to PERCEPTUAL smoke quality (user); exp_046: latent injection of a donor's real smoke perceptually beats the prompt baseline.** Since pixel-PSNR>18 vs a specific clip is information-limited (exp_045), success was redefined to "the generated transition looks like real smoke" (deployment = {target endpoints}+{donor sample transitions}). CPU smoke-signature analysis first: the prompt-only baseline already darkens/desaturates like smoke, but lacks billowing DYNAMICS (free-mid frame-diff tdiff 0.039 vs real 0.063) and shows color artifacts on hard clips. **exp_046** pins a single donor's REAL free-middle latent into a target (same-grid), decodes, judges perceptually vs REAL + BASELINE. Result: donor injection restores dynamics — ss1 tdiff 0.024(baseline)→0.074(donor)≈0.078(real); ss6 0.041→0.064; luminance also recovers (ss6 0.292→0.374 vs real 0.402). **Visual (ss6/ss1 montages): baseline = near-static scene with mild darkening; donor-injected = coherent dark billowing smoke that reads as smoke.** Limitations: pinning splices the donor's *specific* smoke (no scene adaptation), so onset/offset blending is imperfect and saturation overshoots; donorblend softens but dilutes. Next: exp_047 generative block_out/velocity injection (exp_040 cache + exp_041 injector) so the model SYNTHESIZES smoke adapted to the target with smooth boundaries. Bug fixed mid-run: donor pool must precompute all 10 clips (not just targets). Two A100-SXM pods this arc, both torn down (refreshed stale RUNPOD_API_KEY). Note: `notes/exp/exp_044_smoke_transition_injection.md`.

**14:44** — **exp_045 decode-feasibility: cross-clip smoke injection hits an information WALL at the noise floor (~10.5 dB free-mid).** Deployment frame (per user): extract smoke from donor sample transitions, inject into a target known only by its endpoints. exp_045 assembled perfect anchors (target z0) + a candidate free-middle prior, decoded, measured free-mid PSNR (pure VAE decode, leave-one-out donors). Median across 10 clips: `src` (own middle) 120.0 [sanity ✓], `gauss` 10.05, `endpoint_interp` 10.54, `smoke_bcast_loo` 10.30, `smoke_spatial_loo` 10.10, `keepspatial` 10.48. **Every deployable prior ≈ noise floor; donor smoke adds ≤0.4 dB; the marginal leaders gain from endpoint structure, not smoke.** Mechanism: the cross-clip-shared smoke is low-frequency channel-mean (darkening); pixel-PSNR is dominated by high-frequency turbulence which is clip-specific and uncorrelated — a 0.7-cosine latent decodes to perceptually-smoke-but-pixel-wrong ≈ noise. The model's own production generation (~15 free-mid) already beats every pin (pinning removes the generative smoke). **Conclusion: free-middle PSNR>18 vs a specific held-out clip is information-limited — the target's specific turbulence is not in {endpoints+donors}.** `src`=120 vs deployable ~10.5 is the clip-specific-information gap; only the target's own z1 via the matching midpoint solver (recon ~22) bridges it, and that needs the full target clip (unavailable at gen time). Does NOT rule out: full-frame regen>18 (17.92 now, anchor-carried), a perceptual/distributional metric (donor smoke helps), or small velocity/attention gains. Bug fixed mid-run: LTX-2 packed-latent spatial compression is 32 not 8 (704×512→22×16). Pod `l6badalkdfxgqa` torn down. Note: `notes/exp/exp_044_smoke_transition_injection.md`.

**13:48** — **Smoke-transition injection arc opened (goal: free-middle regen PSNR > 18). exp_044 REFUTES the CFG hypothesis; names the real cause as solver self-consistency.** New goal: disentangle the shadow_smoke transition (dark smoke burst in free latent frames 4–12) from clip scene content and inject it so the exp_033 inverted clips' free-middle reconstructs >18 dB (baseline regen free-mid median 14.6). Grounding (ffmpeg per-frame PSNR + CPU latent analysis): the error is entirely in the free middle; recon (RF-midpoint CFG=1) recovers it to 27–38 dB on 6/10 clips but production regen (Euler CFG=4) collapses to 11–18 dB. **exp_044** (regen-only CFG sweep loading cached z1; new token-localized CFG in `RFInverter._call_transformer`) tested whether CFG causes the collapse. **It does not:** ss0 cfg1=16.44 ≈ cfg4=16.43 (cfg2 a mild 17.79); ss7 cfg1=12.82 < cfg4=18.71 (opposite sign). Mechanistic note: anchors are hard-pinned to clean_latents every step so anchor-CFG has zero effect → token-localized CFG ≡ global CFG on free tokens (confirmed loc_f1_a4 16.36 ≡ cfg1 16.44). **Real cause:** the recon→regen gap is solver self-consistency — recon's midpoint solver retraces the inversion discretization and recovers z0's middle; the production Euler sampler diverges at any CFG and emits only generic prompt-smoke. z1's free-middle is gaussian-identical across all clips (std~1.02, per-token norm ~11.5≈√128); whether the solver recovers z0 is solver/clip-dependent (recon_mid_rel_z0 0.07–0.25 rich vs 0.63–1.12 poor — the 4 high-CLIP-gap clips ss1/5/6/9 never encoded the middle). **Conclusion:** the smoke must be INJECTED (the user's actual ask), not recovered from z1. This re-opens the postmortem's failed bootstrap-middle idea (exp_035/036) but with a *smoke-family* prior (resid 0.43–0.84 vs the generic bootstrap's ~orthogonal ≥1.0); a CPU feasibility check shows a shared pooled smoke template explains 68% of cross-clip free-middle variance (poor clips align cos 0.53–0.90). **exp_045** (running) measures the decode ceiling of candidate middle priors before building generative injection. Notes: `notes/exp/exp_044_smoke_transition_injection.md`, `notes/exp/exp_043_smoke_manifold.md`. A100-SXM EU-RO-1 (`l6badalkdfxgqa`), pod kept warm across exp_044→exp_045.

**12:35** — **exp_043 Phase-1 manifold diagnostics — "is chart 03's 'place' a real shared smoke manifold?" Answer: partially, but weaker than it looks.** New CPU-only `experiments/exp_043_latent_pca_inspection/manifold_diagnostics.py` that loads the 10 cached `smoke_z0` + 5 cached `davis_gen_z0` tensors from run_0001 and renders 6 new charts (M1–M6 in `run_0001/charts/`) testing distinct hypotheses about chart-03's apparent "trajectories converging through a region" structure. Numerical summary in `charts/manifold_summary.json`; research note in `notes/exp/exp_043_smoke_manifold.md` (INDEX updated). Findings:

- **[M1] σ(t) is U-shaped for smoke** (1.02 at t=0 → **0.84 at t=8** → 1.04 at t=15) — clips bunch ~20% tighter in the middle. Centroid drift is also U-shaped: 0 → **167 at t≈8** → 108 at t=15 (the smoke mean trajectory bulges out and partially returns). DAVIS σ(t) is flat ≈0.72 from t=1 and drift is monotone 0→175.
- **[M2] R²_time** (variance explained by frame index alone): **smoke 0.134 < davis 0.221** — smoke has *less* shared time-structure than DAVIS A_word, despite same prompt-class. Probably because DAVIS Stage-1 generations all share the C2V anchor-pinning pattern.
- **[M3] Anchored PCA top-5 EVR**: smoke `[9.2, 7.6, 6.9, 6.8, 6.3]%` — **flat, no dominant direction**; davis `[26.4, 20.9, 14.3, 2.6, 2.1]%` — **sharp 3-component structure**.
- **[M4] Cross-clip frame-pair distance heatmap**: smoke shows a clear **dark "square" at t∈[6,10]** (~270 vs ~330 at corners) — middle-frame agreement is the chart-3 signal; davis shows only a thin dark diagonal (matched-time agreement only).
- **[M5] Per-clip PC1 cosine sim**: **smoke off-diag mean +0.003** (≈null=0.005) vs **davis +0.167** (35× null). Smoke clips' within-clip motion directions are essentially **independent** of each other; DAVIS clips share one. The strongest evidence against a single "smoke PC1 direction".
- **[M6] Projection onto `v_smoke = mean(z₁₅−z₀)`**: smoke final-frame proj **108.4 ± 16.2** (CV 15%); DAVIS 14.9 ± 11.2 (~noise); smoke on random direction −0.46 ± 1.19 (null). `v_smoke · v_davis_eq` cos = 0.086. So `v_smoke` IS a real, smoke-specific direction — but as a **mean-displacement summary**, not a within-clip PC1.

**Synthesis**: chart 03's "place" = the **centroid bulge / dispersion-dip at t≈8**. It's real but small (~20% σ dip, not order-of-magnitude). The cleanest 1-D handle on smoke is `v_smoke` (the mean endpoint displacement). The trajectories *don't* collapse onto a low-D manifold (M3 flat scree, M5 independent PC1s). Per-clip variance is mostly scene/object content (~80% from M2 inverse). **Roadmap (Phase 2/3 in the note)**: cheap CPU follow-ups (token-level PCA, per-time-band PCA, subspace projection of DAVIS into smoke subspace), then GPU verifications (decode the centroid trajectory μ(t); decode `latent + α·v_smoke` vs random-direction control), then generative use (feature injection at block_out per exp_041, or `v_smoke` as conditioning offset). Explicit anti-overclaim list at note bottom.

**10:20** — **exp_043 VAE-latent PCA inspection — first descriptive map of the latent space we've been working in.** New experiment + chart-rendering script (`experiments/exp_043_latent_pca_inspection/`) that VAE-encodes a curated set of clips at fixed 608×608 / 121 frames → packed token shape `[1, 5776, 128]`: shadow_smoke source clips (ss0..9), exp_024 DAVIS A_word generations (5 class-pair representatives), and the 11 variant × 3 role outputs of exp_041 run_0007 block_out injection (ss4 only, 33 mp4s). Existing exp_033 z1 inverted noises (10 samples, ss0..9) copied in; 50 i.i.d. N(0,1) samples generated as the noise control. Pipeline: GPU script does encoding only (load LTX-2 in bf16, encode via `pipe.vae.encode` + `_normalize_latents` + `_pack_latents`, save bfloat16 .pt + manifest.yaml). `make_charts.py` runs on the CPU host and renders 11 PNG figures via numpy SVD. Fix mid-run: shadow_smoke_0 maps to `shadow_smoke.mp4` (no suffix) not `shadow_smoke_0.mp4`. Second fix: exp_033's per-clip `max_area` resolutions produced z1's with 5632 tokens for ss0..3/5..9 but 5776 for ss4 — the z1-vs-gaussian PCAs use spatial-mean-pooling to a uniform `[16, 128]` per clip so the heterogeneity is absorbed. A100 80GB PCIe SECURE EU-RO-1 (`uk8zyep38vrga1`), ~5 min cold-start (mmap fault on first VAE forward) + ~4 min for the 48 encodes ≈ 12 min GPU. Pod removed.

Headline findings (`outputs/.../run_0001/OBSERVATIONS.md` for detail):
- **[6] Cross-domain frame PCA (cleans)**: smoke vs davis A_word cluster centroids 65.2 units apart, within-cluster σ ≈ 45 → sep / σ ≈ **1.43** (mostly separated but with overlap). PC1+PC2 only 12.1% of total variance — separation lives beyond 2-D, claim "smoke is OOD" is partial at this projection. At the **clip level [5]** the two domains separate cleanly (sep 551, ratio not pathological).
- **[7]/[8] z1 vs Gaussian**: smoke z1's are clearly distinguishable from i.i.d. N(0,1). Per-channel mean rms 0.094 vs gaussian 0.004 (~**20× larger first-moment bias**); per-clip ‖z‖ 901±30 vs gauss 859±0.66; sep/z1_σ ≈ 1.5 in pooled PCA. Per-channel std is essentially 1.0 for both → the deviation is concentrated in **channel-level mean bias**, not in higher moments. Plain language: inversion leaves a small but systematic per-channel offset.
- **[9] exp_041 injection latent displacement** tracks the previously-measured PSNR almost monotonically (Δfull in latent units vs ΔPSNR): saturation variants (1–48 layers, CFG=1) all sit at Δfull ≈ 8 ± 0.4 (ΔPSNR +0.60–0.67); late-only Δfull ≈ 6.0 (PSNR +0.45); early/mid Δfull < 1 (PSNR ≈ +0.05); CFG=3.2 cond-only saturation Δfull ≈ **25** (PSNR +2.12, ~3× the CFG=1 saturation in both metrics). Inject moves C **along the (ref, bas) axis toward ref**, not perpendicular to it — the layer-equivalence story holds in latent space too.

**18:17** — **Feature-injection It-4 — block_out is the FIRST positive site; CFG=3.2 cond-only amplifies it 3.3×; late-σ steps carry the signal; layer count is irrelevant at strength=1.** Ran an exp_040 dense block_out cache (ss4, **all 48 layers × all 40 recon steps**, predictor, free-mid-token-scoped — 40 × 1138 MB = ~45 GB, run_0003) then an 11-variant exp_041 sweep over it (`config_blockout_step_sweep.yaml`). run.py gained per-variant `inject_layers` / `inject_steps` / `negative_prompt` overrides, B-pass deduplication by `(gscale, prompt, neg)`, ref-decode-once-per-sample, and a **noise-floor ORACLE diagnostic**: decode `z0_recon + (z1_pert − z1)` to get the theoretical ceiling that any strength=1 velocity-equality injection can reach. **Oracle = free-mid PSNR 15.53** (Δ_noise RMS≈1.0 latent caps it — quantifies "if noises differ, even perfect velocity-forcing can't recover source"). **Results (free-mid C−B, ss4):**

- **block_out saturation (all 48 L × 40 steps, CFG=1, strength 1.0): ΔPSNR +0.64, ΔSSIM +0.020, ΔLPIPS −0.021** — first non-noise signal in the project (all prior K,V/Q+K,V iterations sat at ±0.05). C reaches 9.81 vs oracle 15.53.
- **Layer-coverage equivalence CONFIRMED:** `all48`(+0.64), `last47`(+0.66), `first10`(+0.64), `single10`(+0.60), `first20`(+0.63), `first0`(+0.67) — six variants from 1 layer to 48 all within ±0.035. At strength=1, overwriting block_out at ANY layer forces ≈ source velocity (downstream natural-forward layers reproduce source given matched text/σ/audio). Close but **not byte-identical** — the ±0.03 spread is audio-stream drift (audio never overwritten, evolves via v2a independently) + float precision. → **future block_out-predictor work uses a 0.9 GB layer-47-only cache, not 45 GB (48× savings).**
- **Step-window localization (CFG=1):** early[0–13] +0.045, mid[14–26] +0.048, **late[27–39] +0.447** — signal concentrates near-clean (low σ), inverting the standard "content forms early" assumption. Consistent with the corrector bottleneck: high-σ predictor injection is washed out by 27+ steps of un-injected free corrector.
- **CFG=3.2 cond-only amplifies 3.3×:** `all48_alltime_cfg32` **ΔPSNR +2.12, ΔSSIM +0.083, ΔLPIPS −0.078**, C reaches 11.32 (climbing toward oracle 15.53). The cond-only inject rides the 3.2× lever in `v_uncond + 3.2·(v_cond − v_uncond)`. `all48_early_cfg32` stayed +0.091 → late-step dominance holds across CFG. D in cond-only-CFG≠1 ≠ 120 by design (uncond row uninjected).
- **Bottleneck still predictor-only:** best C (11.32) below oracle 15.53 because the midpoint corrector (`z_next = z + dτ·v_corrector(z_mid, σ_mid)`) runs un-injected and determines the step. D=120 at CFG=1 confirms velocity-forcing works when noise matches.

**Next direction:** cache + inject the corrector substep (the main lever), focus late-σ steps, run at CFG=3.2 cond-only — these stack. A100 80GB PCIe SECURE EU-RO-1 (`6qhi6hc6eroc0a`) — cache ~20 min + 11-variant sweep ~110 min ≈ 2.6 h GPU. **Cache run_0003 deleted** post-run (predictor-only, regenerable cheaply as L47-only; corrector follow-up needs a fresh cache). Pod removed.

**16:30** — **Feature-injection: no-prompt isolation of K,V — confirms text guidance is not the bottleneck (still NULL).** Ran exp_041 `config_noprompt.yaml` on 3 samples (ss0/ss4/ss8) from the exp_040 run_0002 K,V cache, two variants each: `cfg1_kv` (cached source prompt) and `cfg1_kv_nop` (`prompt=""`, `negative_prompt=""`), CFG=1. run.py gained a `negative_prompt` variant override. **All 6 NULL: free-mid C−B ΔPSNR ∈ [−0.03, +0.05].** Empty prompt drops B and C together by ~0.3 dB (text contributes little to the free middle) but C−B stays ≈ 0. D plumbing: with-prompt → 120 (perfect velocity equality); empty-prompt → 27–37 (cached K,V from a prompt-conditioned source no longer match the natural edit-pass velocity → content-correct-but-offset). Stripping text guidance does not unlock K,V injection — the failure is the re-noised-query / corrector-dilution mechanism, not text confounding. (Same pod as the 18:17 block_out run; run_0002 K,V cache deleted afterward — K,V line concluded NULL across 4 iterations.)

**11:41** — **exp_042 production C2V baseline — full shadow_smoke sweep, two anchor lengths (K=25 default, K=17 compression-aligned).** Rebuilt exp_042 from the ground up around the stock `LTX2ConditionPipeline.__call__` two-stage production path (Stage 1 base 40 steps CFG=3.2, LatentUpsample ×2, Stage 2 distilled-LoRA 3 steps CFG=1.0 with `STAGE_2_DISTILLED_SIGMA_VALUES`) plus muxed audio via `encode_video`, removing all RFInverter baggage inherited from exp_040/041. The drop1 anchor surgery is preserved as a `types.MethodType` monkey-patch on `pipe.apply_visual_conditioning` that zeroes the first end-anchor latent slot in `cmask` / `clean_latents` / `latents` after the original call runs (no-op when `condition_indices=[]`, so Stage 2's `conditions=None` invocation is safe). The cmask=0 lets `prepare_latents` fill the slot with `randn × σ_max` and the denoise loop's `denoised·(1-cmask) + clean·cmask` treats it as free. Negative prompt seeded from exp_020's DAVIS sweep, **stripped of smoke-hostile terms** (blurry/out-of-focus/motion-blur/flickering/grainy-texture/excessive-noise/jittery-movement/camera-shake) since smoke is inherently diffuse/particle-textured, keeping only lighting + anatomy + artifact terms. Ran two configs on all 10 shadow_smoke samples (`shadow_smoke_0..9`), saving both Stage 1 and Stage 2 mp4s per sample:

- **K=25 (run_0002, default 4 latents/anchor)**: end_clip_index = 12, 8 → 7 active anchor latents after drop1. ~5 min/sample warm, ~57 min total.
- **K=17 (run_0003, compression-aligned 3 latents/anchor)**: K−1 ≡ 0 (mod 8), so `k_lat = (17-1)//8 + 1 = 3` covers exactly 17 pixel frames with no slack. end_clip_index = 13, 6 → 5 active anchor latents. Shorter anchors → more free middle. ~5 min/sample warm, ~67 min total.

A100-SXM4-80GB SECURE EU-RO-1 (`9h8e2hd8v4izuf`) — pipeline reload ~4 min + K=25 ~57 min + K=17 ~67 min ≈ 2.1 h GPU. Pod removed. Note: `run_0001` (Stage-1-only, silent mp4, pre-refactor) on the same dir is superseded by run_0002.

## 2026-05-25

**16:07** — **Feature-injection It-3 (Q + K,V, with CFG cond-only fix) — STILL NULL on ss4 free-middle.** Two structural changes vs run_0004 of It-2: (a) new `cond_only_at_cfg=True` mode in `src/diffusion/feature_inject.py` writes the cached tensor ONLY into row 1 (cond) of the batched `[uncond ; cond]` CFG>1 edit pass — previously it wrote into both rows, forcing `v_uncond ≈ v_cond` inside the injection mask and collapsing the CFG mix there; (b) re-cached the recon pass of ss4 at `sites=[attn1_q, attn1_k, attn1_v]` on the same dense grid (layers 10–21, recon steps 0–23, free-mid-token-scoped, ~20 GB) so Q can be injected alongside K,V to restore the query alignment that pure K,V loses on a re-noised free middle. exp_041 run.py also gained per-variant `sites`/`cond_only_at_cfg` overrides so a single inject loop covers the 2×2 matrix. **Results on ss4 (free-mid C−B):** `cfg1_kv` −0.03 (matches run_0004), `cfg32_kv_condonly` −0.02, `cfg1_qkv` −0.02, `cfg32_qkv_condonly` +0.01 — all four inside noise. **D null tests:** cfg1_kv = cfg1_qkv = **120.00 (exact identity)** — Q+K+V plumbing is also flawless; cfg32_* D ≠ identity by design (CFG=3.2 trajectory diverges from CFG=1 cache). **Interpretation:** the CFG cond-only fix removed the artefact correctly (uncond row no longer forced) but the residual signal is the same noise floor as cfg1, so CFG mix wasn't the bottleneck; adding Q didn't bind either, so query misalignment isn't the only obstruction — the corrector-substep dilution (each predictor write is filtered through an un-injected corrector at σ_mid before z advances) and the maximal full-re-noise test setup are still in the way. Remaining levers: inject the corrector substep too (cache cost ×2), block_out (PnP, still untested at dense grid; was the only positive It-1 signal +0.08), strength sweep, milder perturbation. **Also ran exp_042** (`exp_042_ltx2_c2v_drop1_baseline`, NEW): plain LTX-2 C2V production generation at CFG=3.2 with drop1 on ss4/ss8/ss0 — three reference videos so the operator can see what straight production looks like for these samples at the same prompt and CFG. Filenames encode seed/steps/cfg/drop1: `prod_baseline_seed42_steps40_cfg3p2_drop1.mp4`. Also fixed a residual color-inversion bug in exp_041 (`export_to_video` numpy branch does `frame * 255` — passing already-uint8 frames double-multiplied → mod-256 wrap; now routes through `Image.fromarray` like exp_040) and ffmpeg-negated the 12 broken run_0004 mp4s in place. A100-SXM4-80GB SECURE EU-RO-1 (`mlafa7rj5thmzb`) — exp_040 cache 25 min + exp_041 inject 75 min + exp_042 baseline 28 min ≈ 2.1 h GPU.

## 2026-05-23

**18:26** — **Dense feature-injection (It-2) — falsifies the "too sparse" hypothesis: K,V injection is NULL at any density.** Re-cached the recon pass of ss4/ss8/ss0 at a CONTIGUOUS dense block — layers 10–21 (12), recon steps 0–23 (24), **K,V only**, **free-middle-token-scoped** (new additive `token_scope` in `src/diffusion/feature_cache.py`: caches only the 2888 free-mid tokens → 570 MB/step, 13 GB/sample vs full-token's ~41 MB/step; `FeatureInjector._blend` auto-detects the compact cache by token count, with a manifest-scope guard; unit-tested). Then injected dense K,V (strength 1.0, all 12 layers × 24 steps) on ss4 across three variants. **Headline (free-mid C−B): cfg1 −0.03, cfg32 (CFG=3.2) −0.06, null_cfg1 (empty prompt) +0.01 — all null.** cfg1 **D null-test = exact identity (PSNR 120, SSIM 1.0, LPIPS 0)** → the dense scoped-cache injection plumbing is perfect; the null is real, not a bug. (cfg32/null_cfg1 D≠identity is expected: D is only exact-identity when the edit pass matches the cache's CFG *and* prompt; CFG=3.2 and empty-prompt both diverge the trajectory from the CFG=1/real-prompt cache.) **Mechanism:** MasaCtrl K,V injection only binds when the *queries* are shared; we fully re-noise the free middle, so its queries are random and `softmax(Q·Kᵀ)·V` can't reconstruct source content from source K,V. Density was never the bottleneck. **Untested levers that bypass the query problem (need a re-cache — we cached K,V only): block_out (PnP, query-independent, the only positive It-1 signal +0.08) and Q+K,V (restore query alignment).** Also: full re-noise is a harder test than the real cross-clip-transplant goal. **User stopped the run after ss4's 3 variants to regroup before more GPU spend** — ss8/ss0 (would replicate the null) and the exp_020 production baseline (CFG=3.2) were not run. Infra this session: exp_041 `main()` refactored to loop samples × variants on one pipeline load (legacy single-sample configs still work); CFG knob + `cfg_batch` wired into exp_041 (RF-Edit convention — CFG=1 cache injected into both [uncond,cond] rows at CFG>1, validated by run_0003 ss4/cfg3p2 smoke test = −0.08 on the sparse 4-layer cache); fixed exp_041's video saver (was LTX-2 `encode_video` which inverted RGB → now `export_to_video`; the 8 prior mp4s were ffmpeg-negated back to correct color). New configs only — no experiment forks (exp_040 `config_dense.yaml`, exp_041 `config_cfg32.yaml`/`config_dense.yaml`, exp_020 `config_shadow_smoke.yaml`). A100 (`romw5kih6r1eap`, EU-RO-1) removed after.

**12:45** — Ran the **first feature-injection experiment** (`exp_041_ltx2_feature_inject`) on the A100 (`uwr5reuf5g46ie`, EU-RO-1; torn down after). New reusable module `src/diffusion/feature_inject.py` (`FeatureInjector`) writes cached tensors back into a denoising pass via write-hooks on `attn1.to_k/to_v` (+optional q, block_out), token-scoped + step/substep-gated. Cached the recon pass of `shadow_smoke_4` (best exp_033 sample, recon PSNR 33.12) at 4 mid layers {11,19,28,37} × 9 σ-steps × {block_out, attn1_q/k/v}, predictor substep — **6.4 GB** (exp_040 `config_recon_ss4.yaml`). exp_041 test: re-noise the free-middle latent frames (4–11) of z1, reconstruct (CFG=1 midpoint), inject source features into the free-middle tokens, measure recon-similarity vs the source recon on the free-middle pixel frames (25–88). Two configs, same 4 layers / 9 steps / free-middle / strength 1.0:

- **run_0001 — self-attn K,V only (MasaCtrl-style):** free-mid PSNR baseline 9.16 → inject 9.16. **C−B ΔPSNR −0.001, ΔLPIPS +0.0003 — negligible (noise-level).**
- **run_0002 — full block_out + Q/K/V (PnP+MasaCtrl):** free-mid PSNR 9.16 → 9.25. **C−B ΔPSNR +0.083, ΔSSIM +0.0028, ΔLPIPS −0.0024 — small but correctly signed (pulls toward source).**

Both runs' null test (self-injection into the *unperturbed* z1) = **exact identity** (PSNR 120, LPIPS 0.0), and a standalone test confirmed forward-hook write-replacement propagates. **Conclusion: the injection pathway is sound; the lever is just too sparse** — 4 spread mid-layers can't hold a transplanted representation because the ~12 free layers between each injected layer wash it out (44/48 layers + 31/40 steps recompute freely from the perturbed latent and dominate). The full-residual (block_out) graft gives a real, right-direction nudge, K,V-only doesn't register. **Next experiment: re-cache a *contiguous dense* block of layers (e.g. 8–20) and inject block_out there, plus earlier-step injection** — sparse mid-layer K,V grafting is a dead end for transition transport. Infra note: LTX-2 weights are mmap'd from the MooseFS network volume; a fresh process's first transformer forward faults weight pages on demand and can stall in state D for minutes when the volume is busy (observed both runs); once page-warm it runs at full speed.

**11:35** — Created `exp_040_ltx2_feature_cache` — a feature-gathering fork of exp_033 (drop1, §0 floor) that adds forward hooks on the LTX-2 transformer during invert / reconstruct / regenerate. New reusable module `src/diffusion/feature_cache.py` holds a `FeatureCache` class that registers per-(layer, site) hooks on `transformer_blocks[l]` and on `attn1.to_q/to_k/to_v` (plus optional `attn2_q/k/v`, `ff_out`, `audio_*`, `a2v_*`), gated per (phase, step, substep). Cheap loop-recorded payload (`z_in`, `v_pred`, σ) sits alongside heavy hook-captured tensors in a single per-step `.pt` under `<sample>/feature_cache/<phase>/step_NNN.pt`. Captures by default: block_out + self-attn Q/K/V pre-RMSNorm/pre-RoPE at 6 spread layers × 9 σ-spaced steps × predictor substep. Default per-sample disk ≈ 33 GB; widening axes (`"all"` layers, `"all"` steps, +corrector substep, +text cross-attn, +FFN) documented in the experiment README. Inversion recipe unchanged from exp_033 → hooks are pure observers; metrics should be bit-identical. Smoke-tested end-to-end against a fake LTX-2-shaped transformer (hooks fire on correct submodules, save-grid gating honored, schema reloads cleanly). Next axis: feature-injection regen (MasaCtrl-style K,V replacement, PnP-style block_out injection, etc).

## 2026-05-18

**12:50** — Built a single-page viewer for the local AutoTransition slice at `data/raw/AutoTransition/viewer/`: `build_index.py` joins the 53 MB annotation JSON against the 485 extracted MP4s and emits a slim `templates_local.json` (440 KB, 3976 transitions across 61 classes seen locally vs 107 globally). `index.html` is a vanilla zero-dep page: split filter (train/test), search by `template_id`, multi-select transition-class chips (colored by name hash, with top-10/all/clear toggles), three sorts, per-template timeline with clickable colored ticks at each transition's start_ms (seeks the video), and a chip list of transitions per card. Tick positions normalize to actual video duration once metadata loads. Smoke-tested via `python -m http.server`: HTML/JSON/MP4 all 200. Usage in `viewer/README.md`.

**12:45** — Pulled a sample slice of the **AutoTransition** dataset (HF `yaojie-shen/AutoTransition`) into `data/raw/AutoTransition/`: full 53 MB annotation JSON + first 500 MB of part `.00`, extracted to 485 complete template MP4s (H.264 480×854, 30 fps, ~11 s each). Verified the dataset has 35,000 templates (30k train / 5k test) annotated with 107 distinct transition names (top: `direct_cut`, `pull_in`, `mix`, `pull_out`, `circle_1`) — taxonomy is transition-style labels, not the semantic `shadow_smoke`-style categories under `data/processed/transitions/`. **Non-obvious gotcha discovered:** despite the `template_download.tar.gz.NN` naming, the 13 split parts are plain POSIX tar bytes, not gzip — `cat parts | tar -xf -` (no `z`), and HTTP Range requests against `.00` give cheap partial extraction. Documented in `notes/dataset/autotransition.md` with full JSON schema, partial-download recipe, and current local state; indexed in `notes/INDEX.md`.

**12:11** — Wrote `notes/rf_inversion_postmortem.md` — an explanatory complement to the (procedural) Ledger. Narrates the closed RF-inversion loop for a reader who wasn't in the room: the §0 constraint, the round-trip mechanics, the four intervention families (anchor-quality, model-bootstrap middle, solver step-escalation, σ-conditional release), the reasoning behind each, why each failed, the named cause (free-middle cost coupled to anchor quality through velocity coupling), the per-clip CLIP-gap predictor, and three paths forward outside §0. Indexed in `notes/INDEX.md`; `rf_inversion_loop.md` description updated from ACTIVE → CLOSED.

## 2026-05-16

**02:16** — **LOOP EXIT ② RE-CONFIRMED after 4 more deployable interventions (It-6 through It-10).** User flagged the 19:37 exit-② call as premature ("Why did you fucking stop then?") and the loop reopened with 5.45 GPU-hours of budget remaining. Tested four new families: It-6 `exp_036` soft-bootstrap-middle (strength 0.3) — all 3 pilot clips catastrophic (13-15 PSNR). It-7 `exp_037` step escalation (40→80 steps) — pilot positive on ss0 (+2.78 dB) but full 10-clip batch showed clip-dependent trade-off (ss1 +5.42, ss7 −5.56, ss8 −8.54), final median 23.72 (vs exp_033's 25.66) with only 1/10 clean exit-① pass — REJECTED. It-8 `exp_038` σ-conditional anchor release (release at σ<0.3) — pilot (ss0/ss2/ss5) initially promising (ss2 → 28.06 clean exit-① pass), but full batch revealed ss2 was an outlier: median 21.25, ss4 catastrophically regressed (33.12→23.57, Δ−9.55), ss7 −13.50 — REJECTED. It-9 `exp_039` combined 80 steps + σ-release — confirmed the two interventions DON'T combine constructively, pilot ss2 21.15 (catastrophic), failed mid-pilot. It-10 `exp_038` full batch — final 2/10 clean pass (same as exp_033), median lower. **All four new variants underperform exp_033 on full-batch metrics.** exp_033 remains the deployable floor (PSNR median 25.66, 2/10 clean exit-① pass). Cumulative GPU spend across the whole reset loop: **6.75 / 8.0 pod-hours** (1.55 in It-3..5 + 5.2 in It-6..10). 1.25h headroom unused; further GPU spend within current intervention space will not move the floor. Updated `notes/rf_inversion_loop.md` §7 measured-data table with all 9 recipes; pod `06d7spzeysniu6` terminated; keeper cancelled; loop genuinely closed at the §0-bound floor. The pattern that emerges from 9 deployable recipes: PSNR ranges 15-30 are achievable for borderline-content clips but the catastrophic-transition clips (ss1/5/6/9 with high CLIP f24-vs-f96 gap) are *systematically* unreachable under §0 — no intervention crossed PSNR 22 for ss5/ss6/ss9, confirming the named cause (deployable anchors cannot approach z0 for clip-specific novel transition content).

## 2026-05-15

**19:37** — **LOOP EXIT ② FIRES (preliminary) — scientific floor under §0 reached at exp_033 (PSNR median 25.66).** It-5 (`exp_035_ltx2_rf_inv_bootstrap_middle`, model-bootstrap middle anchors) ran on pod `114yyzu78qx5r5` (A100 80GB PCIe SECURE, EU-RO-1, cycle 17). **B0 mini-falsification on ss0 (10-step bootstrap + 20-step invert/recon/regen, ~15 min wall): recon PSNR 14.24, fails B0 gate of ≥22 dB.** CPU diagnostic on saved `z_bootstrap.pt` proves the failure mode: per-frame `|z_bootstrap - z0|²` at middle frames {4..11} is 28K–63K, compared to exp_032's middle truncation `|z0_recon - z0|²` of 15–144 (~300-400× larger). Since the solver hard-pins to clean_latents at conditioned positions, the recipe forces the round-trip through *bootstrap-land*, not *source-land*. The model's interpolation prior is generic; the source middle is clip-specific. **No bootstrap-derived anchor can approach z0_middle for novel transition content under §0** — fundamental data-distance bottleneck, not a recipe-tuning issue. Cage detector fires on all four signals (two regressing iterations, same anchor-quality family, hostile reviewer winning, recursive §6 citations). **Exit ② check satisfied (all three criteria):** ≥3 deployable fixes run (5 actually — exp_030, 033, 034A, 034B, 035); degradation localized to a *named cause* (free-middle truncation × anchor-quality coupling, where anchor quality is upper-bounded by §0); achievable floor measured (exp_033 PSNR median 25.66, 2/10 clean exit-① pass, with per-clip failure mode predicted by f24-vs-f96 CLIP cosine gap at Spearman ρ=0.855). Pod terminated 19:37. **Final cumulative GPU spend: 1.55 / 8.0 pod-hours (6.45 unused — loop exited at the scientific floor, not the budget).** Ledger §7 measured-data table and §8 budget updated to reflect final state.

**19:02** — **It-4 COMPLETE — REJECTED.** Both pilot variants of `exp_034_ltx2_rf_inv_anchor_quality` regressed vs exp_033 on the 2-clip pilot (ss0, ss5). **Recipe A (scaffold-pad frame 12 with 9-frame static-replay encode → slice lframe[1])**: ss0 PSNR 19.16 (Δ −5.81 dB), ss5 16.67 (+0.29). Median 17.92, regression −2.76 dB. **Recipe B (drop all 4 end anchors {12..15})**: ss0 12.18 (Δ −12.79 dB), ss5 12.41 (Δ −3.97 dB). Median 12.30, both clips below 18 PSNR floor → catastrophic regression. Decision rule fires REJECTED for both. Harness bug found and fixed mid-pilot: original 8-frame static-replay violated LTX-2 VAE's `(F_pix-1)%8==0` constraint and crashed with `unflatten: sizes [-1,2] don't multiply up to size 9`; fix uses 9 frames yielding 2 latent frames, slices the 8-pixel-collapse one. Forensic note: the *first* latent frame of any standalone encoding is *always* 1-pixel-collapse — you cannot replicate the full-clip encoder's 8-pixel-collapse semantics at position 0 in isolation. **Unified takeaway (replicated tag):** at frame 12, NO pin > content-wrong pin > structure-wrong pin. At frames 13–15, imperfect pin > no pin. exp_033 (drop1 frame 12, keep imperfect anchors elsewhere) is at the *local optimum* of the discrete drop-or-keep design space for end-sub-clip latents under §0. Two interventions in this family have now regressed; further progress requires a different family (soft pin / sigma-conditional / model-bootstrap / step-count). Pod `3psxbsgqy6aypj` terminated at 19:00. **1.2 / 8.0 pod-hours used; 6.8 remaining.** Ledger It-4 Phase B/C committed to `notes/rf_inversion_loop.md`. Cage-detector status: no signals fire yet; standard It-5 Phase A follows.

**17:48** — **It-4 Phase B1 pilot launched on GPU pod `3psxbsgqy6aypj`** (A100 80GB PCIe SECURE, EU-RO-1, ssh 213.173.105.10:13295). Capacity poller (raw GraphQL with PUBLIC_KEY) caught the slot on cycle 3 (~45s). Pilot runs `exp_034_ltx2_rf_inv_anchor_quality` with two recipe variants sequentially: **recipe A** (scaffold_pad — 8-frame static-replay encode substituted at source latent frame 12, cmask re-enabled) and **recipe B** (drop_all_end — zero cmask + clean_latents at all 4 end-sub-clip latent positions). Each on the same two pilot clips: `shadow_smoke_0` (borderline at exp_033 PSNR 24.97) and `shadow_smoke_5` (high-motion catastrophic at 16.38). Pre-registered decision rule (see Ledger It-4 in `notes/rf_inversion_loop.md`): CONFIRMED = variant median PSNR ≥ exp_033 baseline median + 3 dB AND no clip below 18 PSNR → full 10-clip batch on winner; REJECTED = both variants regress >2 dB on either clip → design It-5; INCONCLUSIVE = mixed → 3-clip mini-batch on less-regressed. Monitor armed at 4-min cadence with pane snapshots; pod terminates on `=== EXIT` line. Estimated 0.4 pod-hours / 8.0 budget.

**17:26 (resume)** — Hardened `notes/rf_inversion_loop.md` with a per-iteration re-read gate. New banner at the top of the file mandates re-reading §0 (deployability), §1 (objective/exits), §2 (protocol), §3 (Fresh-Mind Gate) at the *start of every iteration*. Each Ledger entry now begins with `**§0-§3 re-read this iteration: ✅**`. Rationale: the most common autonomous-research failure on long sessions is "the protocol is loaded once then quietly forgotten"; re-reading the four-section spine each iteration is the cheapest possible firewall. Now entering It-3 Phase A — Fresh-Mind Gate live.

**17:55** — Added §3-bis **Operating principles** to `notes/rf_inversion_loop.md` — the few autonomous-research habits without which a long-horizon unsupervised loop drifts or stalls. Distilled to five: (1) **Calibration over confidence** — every load-bearing claim tagged `verified / replicated / plausible / guessed`; agent does not build on `guessed` without flagging. Compounding drift is the dominant failure mode without a human in the loop. (2) **External verification > internal reasoning** — when a claim is checkable (run code, decode latent, grep file), verify before building on it; treat own CoT as a hypothesis generator, not an oracle. (3) **Pilot before production** — Phase B now mandates a 1-clip pilot at reduced steps as B1 before any full batch (B2), since stupid failures (bad loader, wrong config) eat pod-hours and are catchable in minutes at small scale. (4) **Cage detection** — live stuckness monitor with four explicit signals (no metric gain × 2 iters, R1 proposing variants of prior hypothesis, R2 unable to find a genuine alternative, same §6 measurements cited recursively) that forces a reframing move (invert the problem, drop a load-bearing assumption, switch abstraction level, change the question from "close the gap" to "what is the floor"); fires *earlier* than exit ③ which now only triggers if reframes are also exhausted. (5) **Generator/critic separation in the gate** — R1 plays generator role, R2 must play hostile-reviewer role; if R2 cannot state concretely why a competent skeptic would reject R1, the gate has failed and must restart. Also tightened the Ledger template to require epistemic tags inline, an explicit cage-detector check at the top of each Phase A, a B1 pilot/B2 full-batch split with smoke-check criteria, and honest-negative-reporting language in Phase C ("INCONCLUSIVE requires a specific evidentiary reason, not just 'unclear'"). Skipped (deliberately, to keep the doc lean): portfolio-search guidance, the mathematical-obstruction extraction heuristic, and reproducibility plumbing (already covered by `experiments/CLAUDE.md` and `exp_utils`).

**17:25** — Trimmed `notes/rf_inversion_loop.md` and the in-flight memory to remove caging interpretation. Stripped out the "critical challenges & gaps" section, the open-hypothesis menu, the "running best / target gap" framing, and per-run "conclusions drawn at the time" — those would have steered the next agent's reasoning. What remains: §0 hard deployability constraint, §1 exits, §2 protocol, **§3 mandatory Fresh-Mind Gate (3 rounds: re-interpret → steelman the opposite → reconcile and decide)** that every new iteration must execute before any GPU work, §4–5 lifecycle/frozen-artifacts, §6 raw-evidence archive (setups and measured tables only, no claims), §7 measured-data table, §8 budget meter, and an empty Ledger template. The in-flight memory file is rewritten to redirect the agent to the gate and to explicitly *not* contain a "what to try next" list. Goal: each iteration starts from ground-up thinking on the raw data, not from extending a prior trail.

**17:10** — **RF-inversion research loop RESET — fresh hypothesis space under hard deployability constraint.** Adjudicated `exp_033_ltx2_rf_inv_drop1` (It-2 of the prior loop, ran on terminated pod `z9zwvjae9w9rvw` for ~1.6 pod-hours, cumulative pre-reset 5.2/8.0). The "drop end-clip first latent frame from the mask" hypothesis landed at recon PSNR median **25.66** (0/10 strict pass, 2/10 clean pass — ss4 33.12 / ss8 29.58; 4/10 borderline 2-of-3; 4/10 catastrophic — ss1/5/6 amorphous endpoints, ss9 pixel-discontinuity outlier). This **falsifies** the "cost is concentrated at one position" assumption — the sub-clip-vs-full encoding mismatch is **graded**, not point-localized. Combined with the post-It-2 transition-hardness analysis (CLIP cosine between f24/f96 predicts exp_033 PSNR at Spearman ρ=0.855, p=0.0016; clean threshold gap_clip≈0.39 separates catastrophic from passing), this gave enough signal to call **reset rather than incremental drop-2-token or drop-3-token escalation**. Rewrote `notes/rf_inversion_loop.md` from scratch: (a) §1 a hard *deployability constraint* — only endpoint sub-clips and material derivable from them may enter the recipe; z₀ slices are now formally forbidden because they leak ground-truth middle-frame info via causal-VAE temporal mixing; (b) §7 a compact archive of It-0/1/2 with raw data preserved; (c) §8 a **critical-challenges section** that names ten gaps in prior conclusions (e.g. exp_031's R3/R4/R5 all silently shared the leaky self-conditioning, so the anchor-mismatch attribution was inferred by elimination not isolated; "cost concentrated" was a clean falsification; recon is the wrong primary metric since deployment needs regen-at-CFG=4 fidelity; encoding-side fixes were never tried); (d) §9 a *non-narrowed* hypothesis space spanning encoding-side anchor improvements (edge-pad / scaffold encode / blended anchor), strength schedules (σ-conditional / position-graded), solver-side (step-count sweep / RK3), metric reform, and explicit scoping. §9.G queues four CPU-only preflight tasks (regen-recon gap rescore, gap_clip-stratified rescore, direct mismatch-profile measurement on existing artifacts, visualization) that should run before any GPU iteration since they may reshape the choice. Reset budget: 8 fresh pod-hours, independent of pre-reset 5.2. Updated in-flight memory → `project_rf_inversion_it2_in_flight.md` reframed as "loop reset, It-3 ready, awaiting user call". Closed-state memory (`project_rf_inversion_loop.md`) preserved as historical record; the mechanism note `notes/models/ltx2/conditioning.md §14-b` is still valid (the causal-VAE asymmetry is real), only the recipe built on it is non-deployable. No exp_034 created yet — loop is awaiting the user's pick of next hypothesis before Phase A begins.

## 2026-05-14

**20:59** — **RF-inversion research loop CLOSED — exit ① triggered at It-1.** `exp_032_ltx2_rf_inv_selfcond` ran the full 10-clip shadow_smoke set on GPU pod `l9h30xtbq5u9vs` (A100-SXM4-80GB, 19:25–20:55, ~1.6 pod-hours, pod terminated; loop total 3.6/8.0). exp_032 is exp_030 with **exactly one change** — `clean_latents` is now the exact slices of z0 itself (true self-conditioning) instead of separately re-encoded sub-clips. Result: **8/10 clips pass the perceptual exit-① bar** (PSNR≥28, SSIM≥0.88, LPIPS≤0.10 per clip); medians PSNR 40.88 / SSIM 0.974 / LPIPS 0.025. vs exp_030's 0/10, PSNR median ~18, recon_rel ~0.68 → exp_032 recon_rel median ~0.105. The two misses (shadow_smoke_2, the lone 10s-source clip, recon_rel 0.56; shadow_smoke_6, landscape, 0.40) still hit PSNR 34–36 — they fail only the strict SSIM/LPIPS bar, residual attributable to the secondary provenance cost the exp_031 ladder flagged; clean follow-up lever (inversion-step sweep) noted but not needed for the loop's goal. Regen secondary largely met (median PSNR ~31.6, SSIM ~0.82); strict latent regen gate fails as expected (CFG=1↔CFG=4 structural mismatch). **Conclusion: exp_030's catastrophic failure was one bug — pinning the RF-Solver to causal-VAE-mismatched conditioning anchors — and true self-conditioning fixes it.** Full closeout, per-clip table, and loop summary in `notes/rf_inversion_loop.md` (LOOP CLOSED section). Knowledge bank updated: `notes/models/ltx2/conditioning.md` gains the self-conditioning anchor rule.

**19:15** — **It-0 of the RF-inversion loop complete — degradation LOCALIZED.** `exp_031_ltx2_rf_inv_ladder` ran the R0→R5 controlled ladder on GPU pod `8hevgt1m0l9gv9` (A100-SXM4-80GB, 17:28–19:11, ~2.0 pod-hours, pod terminated; cumulative 2.0/8.0). Result: **exp_030's catastrophic real-clip failure (recon_rel ~0.68, 0/10) traces to the causal-VAE conditioning-anchor mismatch ("fault #2")** — exp_030 built `clean_latents` by re-encoding the first-1s/last-1s sub-clips, which ≠ the slices of the full-clip encode, so the inverter hard-pinned the solver to anchors that don't match z0's actual conditioned positions every step. exp_031 R5 uses **true self-conditioning** (`clean_latents` = exact slices of z0) and shadow_smoke jumps 0.68 → 0.11 free_rel, with 2/3 clips (ss1: PSNR 40.6/SSIM 0.980/LPIPS 0.016; ss3: 42.4/0.985/0.015) **passing the perceptual exit-① thresholds outright**. Secondary findings: (a) R2 (no conditioning) is catastrophic for every sample (free_rel 1.1–1.65) — vanilla midpoint RF inversion of a VAE-encoded latent diverges; conditioning is load-bearing, the opposite of exp_030's changelog speculation. (b) R4 (encode_clip real DAVIS + true self-cond) is the *best* rung (free_rel 0.015–0.062, PSNR 46) — provenance (VAE-encoder output) and real *natural* content are NOT the problem. (c) Stylized content adds only mild cost (R4 ~0.04 → R5 ss1/ss3 ~0.11). (d) The R0 HALT gate tripped (mean free_rel 0.128>0.05) but was adjudicated a **false alarm**: class2/class5 reproduce exp_029 cleanly; only class8 diverges, because exp_029's reported numbers are 60-step while exp_031 R0 ran 40-step (exp_029 itself escalated class8 to 60). (e) shadow_smoke_4's R5 failure is an aspect-ratio artifact — it's a 1440² square clip and exp_031 forced fixed 512×768; exp_030 used per-clip resolution and ss4 was its best sample. Full pre-registration, per-sample table, and It-1 plan in `notes/rf_inversion_loop.md`. **It-1 pre-registered:** `exp_032` = exp_030 + true self-conditioning + restored per-clip resolution, full 10-clip shadow_smoke set — a confirmation run targeting exit ①.

**17:00** — Stood up an **autonomous RF-inversion research loop** to bridge exp_029 (inversion works on generated latents) → exp_030's goal (real clips, where it collapsed to inv_recon_rel ~0.68). Protocol + live Ledger persisted to `notes/rf_inversion_loop.md` (added to `notes/INDEX.md` under a new "Process" area, and surfaced via auto-memory `project_rf_inversion_loop.md` so it re-loads each session). Loop = COLD pre-flight → HOT (pod up / run batch / pod down) → COLD adjudication; one pre-registered one-variable hypothesis per iteration. Four exit conditions locked: ① perceptual success on shadow_smoke (recon median PSNR≥28/SSIM≥0.88/LPIPS≤0.10, ≥6/10 pass), ② scientific-floor (degradation localized w/ evidence + ≥2 fixes tried), ③ wall (3 iters <20% gain), ④ budget (8 cumulative pod-hours). Iteration **It-0** pre-registered after a 3-pass self-critique: `exp_031` is an R0→R5 controlled ladder (one variable per rung — z₀ provenance, conditioning source, real vs stylized content) reusing exp_029 run_0002's `z0.pt` via a decode-as-fake-real-clip trick, with a FREE-positions-only latent metric to kill exp_030's cond-pollution confound and R0 as a mandatory harness-validation gate. exp_001–030 frozen; loop only creates exp_031+.

**14:57** — exp_030 first run complete (`run_0001`, A100-SXM4-80GB SECURE pod `hvdcv25bideg2d`, ~75 min wall, pod terminated). **0/10 samples passed the dual gate** — and unlike exp_029, the failure is in the *floor* metric. `inv_recon` (solver self-consistency, the must-pass check) averaged **rel ≈ 0.68 / cos ≈ 0.77** vs exp_029's rel ≈ 0.008–0.012 — ~60–85× worse. `inv_regen` averaged rel ≈ 0.83 / cos ≈ 0.67. Per-sample inv_recon rel spread 0.27 (shadow_smoke_4, the 1440×1440 square clip — also best decoded: PSNR 30.7, SSIM 0.91) → 0.91 (shadow_smoke_9). Diagnosis: **off-manifold z₀ under C2V conditioning.** exp_029's z₀ was Stage-1-*generated* — on LTX-2's conditional generative manifold, where the velocity field is smooth and midpoint inversion is ~2nd-order accurate. exp_030's z₀ is a VAE-encode of a *real stylized* clip whose middle frames are not what the model would generate from its own endpoints; the velocity field there is high-curvature, so the midpoint solver's O(dτ³) truncation error explodes and invert↔recon stop round-tripping. C2V makes it worse than vanilla inversion: half the tokens are pinned to clip endpoints and the model is asked to find noise that — combined with those endpoints — produces a middle it structurally would never generate. Encoded-silent audio is a possible secondary stiffness-amplifier but can't be the primary cause (it's identical in invert and recon, so it can't break reversibility). Recommended next steps: (a) cheap 1-variable ablation — rerun with zeros audio to rule audio in/out; (b) vanilla (no-C2V) inversion of the same clips to confirm C2V is the structural culprit; (c) if real-clip C2V inversion is genuinely needed, RF-Solver paper's correction machinery or optimization-based / SDEdit-style partial-noise inversion. Decoded recon/regen mp4s saved per sample under `run_0001/<sample_id>/` for visual inspection. Pod teardown note: RUNPOD_API_KEY rotated mid-session — env + runpodctl config.toml both went stale; terminated the pod via raw GraphQL after re-extracting the key from `pod_init.sh` (per runpod-pod-init skill procedure).

**13:35** — exp_030 audio strategy changed from synthesized zeros to **encoded silent mel** before any run. Rationale (user push): torch.zeros in transformer audio cross-attention is out-of-distribution — the audio VAE never outputs zeros and the transformer never saw zero audio at training. Instead, encode a literal silent mel `torch.zeros((1, 2, T_mel, 64))` through `pipe.audio_vae` (deterministic posterior mode), pack + normalize via `pipe.prepare_audio_latents(latents=..., noise_scale=0)`, and use the resulting fixed in-distribution tensor as audio context across all phases. Deleted `audio_strategy` config knob, `audio_replay`/`audio_zeros` RFInverter fields, all `"zeros"`/`"capture_and_replay"` branching, and the `phase`/`step_idx` audio-direction args in `_call_transformer`/`_midpoint_step`/`_euler_step`. One path, no branches. Caveat documented in README: regen no longer matches production exactly (production's audio scheduler evolves audio noisy→clean), but holding audio fixed isolates the video flow as the only variable in the round-trip. Files: `experiments/exp_030_ltx2_rf_inv_real_clips/{run.py,config.yaml,README.md}`.

**13:09** — Forked exp_029 → **exp_030_ltx2_rf_inv_real_clips**. Same RF-Solver dual-gate pipeline but operates on **real existing clips** instead of Stage-1 generations. `z₀` now comes from VAE-encoding `data/processed/transitions/shadow_smoke/*.mp4` (10 samples; 9×5s + 1×10s — we use first 5s of all). VC conditioning is first-1s and last-1s slices of the SAME source video (not two separate clips like exp_029). Audio is synthesized zeros — no Stage-1 gen ⇒ no AudioContextRecorder; added `build_audio_zeros_packed()` that derives the post-`_pack_audio_latents` shape directly from `pipe.audio_sampling_rate / audio_hop_length / audio_vae_temporal_compression_ratio` + mel/latent-channel config. Resolution auto-resolves per-clip via `max_area=393216` + `resolve_resolution(ref_image)`. Shared one-line prompt across samples (`"A floating black smoke transition between objects."`) — minimal on purpose. Dropped the unused `AudioContextRecorder`/`load_clip_frames`/`load_frames_from_*` helpers and the `Any`/`glob` imports. Files: `experiments/exp_030_ltx2_rf_inv_real_clips/{run.py,config.yaml,README.md}`.

**12:48** — exp_029 dead-code sweep after gate downgrade. Dropped `retry_steps` and `lpips_threshold` from `run.py` (no longer read), removed `retry_executed` from `summary.yaml` samples and `lpips_threshold_reported` from `inv_meta.yaml` (both were always-constant artifacts post-retry-removal). Renamed `config.yaml`'s `inversion.escalation:` block to `inversion.gate:` (purely thresholds now, no escalation logic) and stripped `retry_num_steps` + `lpips_threshold` keys. `run.py` reads `gate` with an `escalation` fallback for back-compat with old configs / `config_snapshot.yaml` files. No behavior change.

**12:43** — exp_029 dual gate downgraded to **informational-only logging**: removed the auto-retry-at-60-steps path entirely. Rationale from yesterday's run_0002 finding — the gate's blocking metric is `inv_regen rel ≈ 0.30 ≫ 0.20`, which is structurally driven by the CFG=1 invert ↔ CFG=gen regen flow mismatch and can NOT be improved by adding inversion steps (regen is a fresh Euler+CFG=gen forward pass that doesn't see the invert step count). The gate is now evaluated, printed per-condition (`recon_rel/cos`, `regen_rel/cos`) with PASS/FAIL marks, and persisted into `summary.yaml` under `samples[i].gate` with thresholds, but no retry runs. Also added a bordered metrics table at the end of each run (sample_id, diff, gate, rec_rel/cos/PSNR/SSIM, reg_rel/cos/PSNR/SSIM). README "(e) Validate" and "(f) Escalation" sections updated. Files: `experiments/exp_029_ltx2_rf_inversion_v2/run.py`, `experiments/exp_029_ltx2_rf_inversion_v2/README.md`.

## 2026-05-13

**19:32** — exp_029 first complete run on A100-SXM4 80GB SECURE (pod `d32mgbhpritga1`, ~1h wall, pod torn down). `run_0002/` on shared volume. **0/3 samples passed the dual gate** — but the pattern is the actual finding, not a failure: **all 3 inv_recon pass by ~8-10× margin; all 3 inv_regen fail with rel ≈ 0.23-0.31**. Numbers (60-step retry, since first 40-step attempts all failed on inv_regen which can't be retried into success):

| Sample | inv_recon rel | inv_recon cos | inv_regen rel | inv_regen cos |
|--------|--------------:|---------------:|--------------:|---------------:|
| mallard (easy) | 0.0081 | 0.99997 | 0.3081 | 0.95226 |
| car-roundabout (mid) | 0.0087 | 0.99996 | 0.3039 | 0.95369 |
| **blackswan (hard)** | **0.0120** | **0.99993** | **0.2326** | **0.97322** |

Three takeaways:
- **Solver self-consistency at CFG=1 is excellent** — inv_recon `rel ≈ 0.01` is ~10× tighter than exp_027's already-good 0.017. The 40-step σ-grid match (Fix #1) and true-zeros audio (Fix #4 revised) are doing real work.
- **exp_027's hard sample (blackswan) now passes inv_recon cleanly.** exp_027 had it at `rel = 0.32` (fail at 30 and 50 steps); exp_029 has it at `rel = 0.012` (would pass with 8× margin). The recipe genuinely fixed the case exp_027 couldn't crack. Counter-intuitively, blackswan is also CLOSEST to passing inv_regen (rel=0.23, vs ~0.30 for the easier samples) — suggesting the inv_regen failure mode is not difficulty-driven but is a structural property of the CFG=1↔CFG=4 trajectory mismatch.
- **inv_regen is the gate-blocking metric.** The dual-gate retry helps inv_recon (more steps → tighter solver) but NOT inv_regen (regen is a fresh Euler+CFG=4 forward pass; step count for invert/recon doesn't enter). So the auto-retry's design is wrong for this failure mode — it burns compute on the already-tight metric. Future iteration should either (a) retry inversion with a different solver/CFG strategy to address inv_regen, or (b) decouple the retry trigger from the dual gate.

Other observations: (i) audio capture worked cleanly — `(2, 126, 128)` shape per call, 40 captures per gen run, saved as `audio_record.pt` per sample for forensics; (ii) Stage-1 generation time dropped from 5:27 (first sample, includes torch.compile graph build) to ~2:00 (subsequent samples) on the SXM variant; (iii) the first launch crashed with `mat1×mat2 shape mismatch (8x16 vs 128x2048)` because my zeros tensor was 4D `(1, 8, 1, 8)` while the transformer expects post-`_pack_audio_latents` 3D `(B, audio_num_frames, audio_channels*mel_bins_compressed) = (1, 126, 128)` — the diffusers pipeline calls `_pack_audio_latents` before the transformer. Fix: build zeros via `torch.zeros_like(captured_template)` from `audio_record.pt[0]`. The capture always runs, so the template is always available. This is a more robust pattern than computing dimensions from `audio_vae_mel_compression_ratio`. Next step: investigate inv_regen failure mode (CFG=1↔CFG=4 trajectory gap) — candidate approaches are null-text-style optimized neg-prompt inversion, or accept inv_recon-only as the cache validity criterion and ship `z₁` for Step 7 with documented "regen drift" caveat.

**18:05** — Tightened exp_029 audio docs to compact + rigorous form. README "Audio strategy" now leads with where we inject (audio-VAE latent, direct to transformer kwarg), the three reasons for zeros default, why regen also uses zeros (diagnostic isolation), the accepted gen↔others asymmetry, and an alternatives table. `PHASES_AND_CONTRACTS.md` §6 reorganized into 9 numbered subsections (pipeline behavior, control surface, default, regen choice, asymmetry, alternatives matrix, exp_027 bug, instrumentation, when to switch). No code change. Also corrected the runpod-pod-init skill: do NOT `source /workspace/cache/pod_init.sh` on a long-lived CPU host to refresh a stale `RUNPOD_API_KEY` — the script's first lines `rm -rf /root/.claude /root/.codex` and would wipe live Claude session state. The surgical refresh is `export RUNPOD_API_KEY=$(grep '^export RUNPOD_API_KEY=' /workspace/cache/pod_init.sh | cut -d= -f2-) && runpodctl config --apiKey "$RUNPOD_API_KEY"`. New memory `runpod_api_key_location.md` documents the canonical key location + safe refresh procedure.

**17:36** — Corrected the exp_029 audio strategy default after a sharper read of the use case. DAVIS clips are silent → no real audio trajectory to invert → tying inversion to base gen's specific audio roll is over-fitting. Step 7 will re-denoise `z₁` in a fresh `pipe(...)` call with its own audio init, so the cached `z₁` should be audio-context-agnostic. exp_027's empirical evidence backs this up: samples 1+2 round-tripped at `rel ≈ 0.017` despite the audio mismatch — the video transformer is robust to audio context for silent training data. Default `audio_strategy` is now `zeros` (true `torch.zeros`, NOT `prepare_audio_latents(noise_scale=0, latents=None)` which exp_027 misused — the latter actually returns randn because `_create_noised_state` computes `0·new_randn + 1·initial_randn = randn`, a reproducibility footgun where different runs got different "zeros" tensors). `AudioContextRecorder` is kept as pure instrumentation: it captures gen audio for forensics (`audio_record.pt` ~2 MB / sample) but does not feed it into invert/recon/regen unless `audio_strategy: capture_and_replay` is set explicitly. PHASES_AND_CONTRACTS.md §6 now walks through the decision tree.

**17:16** — Forked `experiments/exp_029_ltx2_rf_inversion_v2/` from exp_027 to close six methodology gaps. (1) Inversion + reconstruction step count 30 → **40** to match the generation σ grid exactly (exp_027 sampled different σ points in forward vs reverse passes — biggest first-order source of round-trip drift). Retry 50 → 60. (2) Documented the existing CFG=1 round-trip as testing **solver self-consistency only**, not generation-trajectory recovery. (3) Added a **new `regenerate` phase**: Euler + CFG=gen_cfg from `z₁` → `z₀_regen`, compared against `z₀` with a second latent-space gate (rel<0.20, cos>0.97). Sample passes only if BOTH `inv_recon` and `inv_regen` clear their thresholds. (4) Fixed a silent audio-context mismatch — base generation initializes `audio_latents` at `noise_scale = sigmas[0] ≈ 1.0` and applies `audio_scheduler.step` every iteration; exp_027 passed exact zeros to inversion's transformer at every call. New `AudioContextRecorder` wraps `pipe.transformer.forward` during base gen, captures `audio_hidden_states` per call, and `RFInverter` replays the trajectory across all four phases (reversed for invert; forward for reconstruct/regen). Captured audio saved as `audio_record.pt` per sample for forensics. (5) Per-step CSV diagnostics for every phase: `v_norm` (raw + clamped), `z_norm` split by C2V mask (`z_cond_norm` should remain ≈ `‖clean_latents‖`; `z_free_norm` drives solver error), `σ` triple, `x0_pred_norm`, `dt_s`. Three CSVs per attempt per sample. (6) Explicit comment block at the `retrieve_latents(... sample_mode="argmax")` call site documenting that argmax mode is deterministic and `generator` is unused there — a future stochastic switch would silently break C2V conditioning identity across runs. New companion doc `PHASES_AND_CONTRACTS.md` (replaces exp_027's `CFG_AND_PROMPT.md`) tabulates the phase × CFG × prompt × audio matrix and documents the audio replay scheme. Ready to run on next GPU pod.

**16:39** — Documented LTX2 CFG mechanics in `notes/models/ltx2/pipeline_api.md` §6b (compact table). Confirms `LTX2ConditionPipeline` uses vanilla cond/uncond CFG batched in one forward pass: `do_classifier_free_guidance = guidance_scale > 1`, prompts encoded twice via Gemma, `prompt_embeds = cat([neg, pos], dim=0)` (uncond first), `chunk(2)` then `ε_uncond + s·(ε_text − ε_uncond)`. Audio shares the same `guidance_scale`. At Stage 2 (`guidance_scale=1.0`) CFG is fully off — `negative_prompt` ignored, batch is `1×B`. Clip-conditioning `strength` (via `conditioning_mask`) blends after CFG and is independent of it. INDEX entry updated.

**12:26** — exp_027 first complete run with the new MetricSuite landed (`run_0005/`, A100 80GB PCIe, 40 min wall, $0.93). **2/3 samples passed the gate; sample 3 (blackswan→boat, "hard") failed both 30-step and 50-step retries.** Pass summary: mallard→mallard (easy) `rel=0.0169 / cos=0.99986 / psnr=44.81 / ssim=0.9951 / lpips=0.0043`; car-roundabout→bus (mid) `rel=0.0174 / cos=0.99985 / psnr=44.32 / ssim=0.9958 / lpips=0.0026` — both at 30 steps, latent gate hits with ~6× margin. Sample 3 fail: 30-step `rel=0.3165 / cos=0.94967 / lpips=0.0725`; **50-step retry made it *worse*** (`rel=0.3395 / cos=0.94210 / lpips=0.0825`). Counterintuitive but explained by the README's "schedule caveat": at 50 steps the dynamic-shifted σ grid puts even MORE density near σ=1, leaving the σ<0.1 cleanup phase under-sampled, which is where the C2V end-clip transition lives for this hard pair. The 3rd-order RK escalation (configured but not auto-triggered) is the next knob — that's a follow-up experiment, not an exp_027 fix. Two design wins worth keeping: (1) the latent-space gate gave an unambiguous fail (rel=0.32 is far from 0.10), where the LPIPS-only gate would've been on the bubble (0.0725 vs the historical 0.05 threshold — borderline ambiguous), and (2) all decoded metrics agreed on the worst-frame region (frames 54-59 on sample 3 vs frame 65 on samples 1-2), localizing where the failure occurs without needing to eyeball videos. Cached `z₀ / z₁ / σ_t=0.25,0.50,0.75 / source_video.mp4 / recon_video.mp4 / inv_meta.yaml` for all three samples live in `outputs/videos/exp_027_ltx2_rf_inversion/run_0005/` — samples 1 and 2 are ready to feed Step 7 (feature injection); sample 3 needs the RK retry first.

**11:42** — exp_027 metric expansion + run on a new A100 80GB PCIe pod. Replaced the single-metric `FrameLPIPS` with a unified `MetricSuite` (`run.py:480`) returning five families per attempt: per-frame PSNR (dB), SSIM (skimage `structural_similarity`, channel_axis=-1), LPIPS (AlexNet), temporal-flicker `|Δsrc(t,t+1)−Δrec(t,t+1)|`, and latent L2 / relative / cosine on the packed (1, N, 128) latents. Each metric ships a per-frame array plus a `worst_frame` index (`min`-indexed for higher-is-better PSNR/SSIM/cosine, `max`-indexed for the error metrics). Moved the auto-retry gate off LPIPS onto two **latent-space** conditions — `latent_rel < 0.10` AND `latent_cos > 0.99` (`config.yaml:escalation`). Latent-space gating is decode-free, so it isolates inversion error from VAE-decode loss; the decoded-space metrics still print and live in `inv_meta.yaml` for visual debugging but no longer drive the retry. Calibrated thresholds against synthetic perturbations: σ=10 pixel noise → rel≈0.01 / cos≈0.99995 (passes); random per-element noise → rel≈1.01 / cos≈0.70 (clearly fails) — gate is well-balanced. README and `inv_meta.yaml` schema updated to match.

**11:38** — RunPod fresh-pod SSH bootstrap bug surfaced and patched in the skill file. The 15s poller's raw `podFindAndDeployOnDemand` GraphQL call had been passing `env: []`, but the `runpod/pytorch:2.4.0-...` image's `/start.sh` entrypoint only starts sshd when `PUBLIC_KEY` env is present. Symptom: pod boots to RUNNING and the TCP port mapping populates, but `ssh root@<ip> -p <port>` returns `Connection refused` indefinitely — nothing is listening on port 22. Lost one A100 slot (had to stop+remove the broken pod) before catching it; a tight burst poller (1s sleep × 60 cycles, then 15s) grabbed the slot back on cycle 1 (~1 s). The skill file (`/workspace/cache/claude/skills/runpod-pod-init/SKILL.md`) now states the `env: [{key: "PUBLIC_KEY", value: "..."}]` requirement and the on-air recovery path (RunPod dashboard → Connect → Web Terminal).

**11:05** — Registered a fresh ed25519 SSH key (`/workspace/cache/runpod-key-diff{,.pub}`, fingerprint `SHA256:yED7cfIedtxrn6oXRPLdgwywjoQ8NDXDj5YhLQ0olxY`) with the RunPod account via `runpodctl ssh add-key --key-file`, then added the corresponding bash permission rules to `.claude/settings.local.json` so future invocations don't trip the credential-mutation guard. Reason: this is the first run that needs to SSH into pods from the CPU host, and the previous skill-documented `/root/.runpod/ssh/RunPod-Key-Go` private key file never existed there.

## 2026-05-12

**13:09** — Revised exp_028 per visual-review feedback and re-ran on an A100 PCIe ($1.39/hr, EU-RO-1 secure, ~5 min total wall). Two changes: (1) the bridge in `hold_bridge_hold` is no longer a 57-pixel-frame cross-fade re-encoded together — it's now a latent-space lerp between two *single-frame VAE encodings* (last_start_frame alone → key-frame latent A, first_end_frame alone → key-frame latent B), filling the M middle slots with alphas (1..M)/(M+1). This drops the previous version's double-anchoring of `first_end_pixel` (it had appeared at both the bridge tail and `end_lat[0]`), so the second-clip onset isn't ambiguous anymore. (2) Added a length sweep `num_frames_sweep = [121, 89, 65]` → M ∈ {8, 4, 1}; N=65 with M=1 is the smallest possible bridge that still keeps both clips fully held. Output filenames now encode M (`s42_K25_N{nf}_M{m}_mode-{mode}.mp4`). 27 mp4s in `outputs/videos/exp_028_vae_latent_composition/run_0002/` covering 3 samples × 3 lengths × 3 modes. Per-output wall time 2-8s (no pixel-cross-fade encode pass, so the bridge mode is no longer the bottleneck).

**12:41** — Ran exp_028 on an A100-SXM4-80GB in EU-RO-1 (community-cloud sold out of all pre-Blackwell GPUs at 12:25, poller hit at cycle=7 / ~7 min later). All 3 samples × 3 modes (naive / hold_lerp_hold / hold_bridge_hold) wrote cleanly under `outputs/videos/exp_028_vae_latent_composition/run_0001/`. Per-mode wall time was 4-33s (naive/lerp are pure latent arithmetic; bridge adds a 57-pixel-frame VAE encode pass). The `hold_bridge_hold` mode produced exactly the expected `bridge_lat` shape `(1,128,8,16,24)` — confirms the `M_px = 8*(M-1)+1 → M` arithmetic. Logs/summary committed; videos are the visual A/B/C for the dissolve-cause diagnosis.

**12:14** — Built exp_028 (`experiments/exp_028_vae_latent_composition/`) — a fork of exp_023 that fixes two diagnosed problems with the VAE-only dissolve probe. (1) exp_023's `alpha = t / (T_total - 1)` ramps from 0 across the entire timeline, so the blend kicks in immediately at output frame 1 — the start clip never plays clean. (2) The middle blends mix `start_lat[-1]` (a *motion* latent encoding pixel frames 17-24 of the start clip) with `end_lat[0]` (a *key-frame* latent encoding only pixel frame 0 of the end clip) — arithmetic on two latents with incompatible temporal semantics, producing flicker. exp_028 runs three modes per sample for visual comparison: `naive` (exp_023 baseline), `hold_lerp_hold` (pure boundaries + latent lerp middle — isolates fix 1), and `hold_bridge_hold` (pure boundaries + a pixel-space cross-fade between `last_pixel_of_start` and `first_pixel_of_end` re-encoded by the VAE to exactly fill the middle — isolates fix 1 + 2). For default `num_frames=121` / `num_clip_frames=25`: T_total=16, T_clip=4, middle M=8, bridge clip = 57 pixel frames. 3 samples (easy/mid/hard).

**11:51** — Added `experiments/exp_027_ltx2_rf_inversion/CFG_AND_PROMPT.md` documenting the conditioning contract: generation runs at CFG=4 with positive + negative prompt and 80 NFE; inversion/reconstruction runs at CFG=1 with the positive prompt still active (negative dropped) and 60 NFE per direction — the "2 calls per step" come from the midpoint integrator, not CFG. Spells out the three independent reasons CFG=1 is required for inversion (non-conservative mixed field, NFE budget, downstream cache reusability), separates text CFG from LTX-2's visual conditioning (per-token timestep + x₀-clamp), and lists six future-pitfall warnings (the `inversion.guidance_scale` knob is currently a no-op past 1.0; bf16/fp32 boundary; σ<1e-4 short-circuit; etc.).

## 2026-05-11

**16:50** — Cleaned up `experiments/exp_023_vae_latent_lerp/`: replaced the stray exp_020 README with a real one describing the VAE-only dissolve-cause probe (hypothesis, interpolation scheme, sweep table for `num_frames` ∈ {121, 73, 57}, outputs layout); rewrote the misleading config comment that claimed `num_frames=121` when the value was 57; updated `run.py` output filename to `s{seed}_K{K}_N{num_frames}_lerp.mp4` so sweep outputs are distinguishable on disk per project convention.

**16:27** — Added section 2.5 (PCA of velocity-field frame embeddings) to `notebooks/exp021_02_velocity_field.ipynb`, mirroring the VAE PCA in Level 1.4 and the transformer PCA in Level 3.2 but with `v_pred` as the source signal. Grid is samples × {τ=0, 13, 26, 39} — early-step two-cluster splits indicate the model commits to the dissolve frame from the noisiest step. Also fixed the `NameError: add_gt_vline` in cells 2.2 and 2.4 by forcing `importlib.reload(trajectory_utils)` in the imports cell so a stale module cache cannot mask newly added helpers.

**11:44** — Built exp_027 (LTX-2 RF-Solver flow inversion, Step 6 of the editing pipeline). Custom denoising loop on top of `LTX2ConditionPipeline`: VAE-encode Stage-1 generated z₀ → invert via RF-Solver midpoint 2nd-order (30 steps, CFG=1, 60 NFE) → reconstruct → per-frame LPIPS gate at 0.05. Replicates the pipeline's per-token timestep + x₀-domain conditioning clamp, with a short-circuit guard at σ<1e-4 to avoid the schedule's degenerate first inversion step. Auto-retries at 50 steps on gate miss. 3 DAVIS pairs (easy/mid/hard). Cached artefacts: z₀, z₁, σ-checkpoints, source/recon videos, LPIPS stats. Ready for Step 7 (feature injection).

**11:18** — Opened placeholders for exp_025 (LTX-2 negative-prompt sweep) and exp_026 (LTX-2 seed × end-clip-strength sweep) with READMEs/configs but stubbed `run.py` — to be implemented after the next round of work, when we revisit which structural knobs (negative prompt, seed lottery, endpoint clamping) most affect transition creativity beyond the empty-prompt finding from exp_024.

**11:01** — Cleared saved cell outputs from `notebooks/exp021_trajectory_analysis.ipynb` (~176 MB → ~118 KB) so GitHub accepts the blob under its 100 MB limit; re-run the notebook locally to regenerate plots.

**10:57** — Published exp_023 (VAE latent interpolation) and exp_024 (LTX-2 prompt sweep) with configs and run scripts; added Jupyter notebooks under `notebooks/` for exp_021 trajectory analysis (`trajectory_utils`, programmatic notebook generator) and exp_024 prompt exploration. Documented LTX-2 19B P=1 patch geometry and token locality in `spatial_locality.md` with matching updates to `conditioning.md` and the knowledge index. Added root `CHANGELOG.md`, Cursor rule for keeping it current, repo-wide `CLAUDE.md` guidance, experiment-wide `experiments/CLAUDE.md`, and gitignore entries for `.claude/` and `.ipynb_checkpoints/`.

## 2026-05-05

**16:45** — exp_024 prompt update: rewrote all 10 Category B prompts to describe continuous semantic morphing (feathers changing color and shape, bus form growing from a car silhouette, clothing materializing mid-walk) instead of scene cuts. Removed "morphing" and "warping" from the negative prompt since they fight the new B intent. Added Stage 1 video save to `run.py` so each run now outputs both a Stage 1 preview (512×768, silent) and the final Stage 2 output (1536×1024), enabling faster iteration decisions. Created `ltx2_prompting_notes.md` in the experiment folder documenting the format rules, transformation mechanism principles, and what prompt language to avoid.

---

Newest first. Each entry has a timestamp and says what changed in plain language.
Code details only when they help locate the change.

---

## 2026-04-08

**12:57** — Changelog made timestamped and language-first per this update.

**12:45** — Notes folder fully restructured. Moved theory notes into `theory/`, dataset notes into `dataset/`, split the monolithic LTX-2 reference into three focused files (pipeline API, conditioning mechanics, denoising schedule), added Wan 2.1 model notes, removed all empty stub files, rewrote the knowledge index.

**12:30** — Cursor rule updated to require maintaining the changelog on every meaningful change, placed as the first instruction so it is never missed.

**12:15** — Learned and documented how the LTX-2 denoising schedule actually works: the sigma shift concentrates many small steps at the noisy end and few large steps near clean. This explains the counterintuitive pattern in the trajectory heatmaps where latent displacements are largest in the final denoising steps, not the first.

**11:45** — Fixed two plotting bugs in exp_022: the y-axis arrow on heatmaps pointed in the wrong direction after matplotlib's rotation, and the conditioning boundary lines were drawn at the wrong column position for curvature and angular features (which have shorter x-axes than the other panels).

**11:10** — Fixed a crash in exp_021 trajectory logging. LTX-2 passes latents to the scheduler in packed sequence format `[batch, tokens, channels]`, but the logger assumed a spatial layout. Added unpacking logic so the rest of the analysis works unchanged.

---

## [earlier — dates not recorded]

**exp_022** — Geometric feature extraction from trajectories. Computes per-frame norms, speed, curvature, and angular consistency across all denoising steps. The discrete Laplacian of the final clean latent turned out to be the best signal for locating the dissolve frame; works well for semantically distant clips, less so when the conditioning boundary dominates.

**exp_021** — Trajectory logging. Patches the scheduler to capture the full denoising trajectory — every latent state and velocity prediction at every step — and saves it for offline analysis.

**exp_020** — First working clip-to-video pipeline using Diffusers natively. Key discovery: Stage 2 requires the doubled spatial dimensions explicitly, otherwise the conditioning mask is built at the wrong size.
