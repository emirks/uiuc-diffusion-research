# BRIEF — which null should the CTT / VFX-transfer guidance push away from? (fable investigation, 2026-09-14)

Owner's question (verbatim intent): "I want to get the best null that I can give. The lerps (pixel, VAE, DINO-space or
DINO-transport-space) are informationally very sensible, because a lerp is the path that adds no information beyond the
endpoints. If we can justify one of them, great." The driver (Fable, this session) adds: the base model's measured default
(REPORT.md / READING.md) is NOT a lerp — it is a fast, spatially global appearance swap with little novelty. The null
question therefore has two halves that must not be conflated: (i) what does the model do by default, (ii) what is the
best reference to guide away from, given (i) and the mechanisms actually available. You investigate both, with data.

## What exists (all CPU-readable; no GPU is available to you and none is needed)
Campaign dir `misc/2026-09-13_null_default/` (repo root `/taiga/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research`,
`$LAB=/taiga/illinois/eng/cs/jrehg/users/emirkisa`). Read in this order: `SPEC.md`, `PHASE_A_REPORT.md`,
`results/REPORT.md`, `results/READING.md`, `results/TABLES.md`; then `misc/2026-09-13_null_ledger/LEDGER.md` for every
earlier number (Aug-24 lerp-collapse, Sep-08 re-measure, Sep-08 probe, Sep-12 distance-vs-cut).
- `manifest.csv` (read with `keep_default_na=False`; group label NULLGEN = base neutral generations): 749 clips.
  Groups to focus on: S-GRID NULLGEN (68; real GT anchors, start caption only), S-GRID-F NULLGEN (38; DAVIS anchors, no
  GT twin), S-PROBE R3 (120; model-made end anchor, captions only; tiers `high` = scene change 90 / `inplace` 30),
  S-SWEEP A_empty and A_word (10 each; exp_024, 193 f, DIFFERENT config — side panel only). References: GT (19, the real
  transition with the same anchors), R1 (start-only witness, full prompt), R2 (both anchors, full prompt).
- Per-frame features `$LAB/cache/null_default/features/<clip_id>.npz` and `<clip_id>__{LERP,CUT50,FREEZE,LATLERP}.npz`:
  `dino` fp16 (T,768) L2-normed DINOv2-base CLS per frame; `vae` fp16 (T_lat,128,H,W) LTX-2 latents; `trans` fp32
  (T_lat,H,W,44) the frozen DINO-basis transport signal (0-BASED channels: nu=40, sA=38, sB=39, u/v/conf=32/33/34,
  ent=35, inplace=36, dir=37, dlab=41; definition `misc/2026-08-24_flow_signal_conditioning/armA/SIGNAL_REFERENCE.html`);
  `a_idx`, `b_idx`, `T`. Frame f → latent index 0 if f==0 else (f−1)//8+1 (`scripts/common.lat_index`). Landmarks belong
  to the landmark-source clip (GT for S-GRID NULLGEN via `twin_clip_id`; R3 for R1/R2; own for S-GRID-F/S-SWEEP).
- Pixel trajectories: decode with `misc/2026-09-08_collapse_remeasure/instrument.py: load_matrix` (128 px, blur 1.0,
  `cv2.setNumThreads(1)`, SERIAL — parallel decode on the login node corrupts frames). Videos at `manifest.path`.
- `results/per_clip.csv` (8729 rows: clip × kind × space, the 18-d summaries), `results/trans_profiles.csv` (per latent
  timestep nu/sA/sB/swap/trans/ent/inplace/dir/dlab/local), `results/collapse_measures.csv`, `results/probe_paired_by_tier.csv`.
- Helpers: `scripts/common.py` (`load_manifest`, `lat_index`, `measure_traj`, TRANS summaries). Env: ALWAYS
  `source $LAB/envs-aarch64/activate` (login `python3` is 3.6 and will fail). Keep CPU use modest (single-threaded decode,
  ≤ 4 BLAS threads). Write everything you produce under `misc/2026-09-13_null_default/investigation/`.

## Mechanism facts you must ground the recommendation in (read the files, do not assume)
- Base sampler (`eval_ladder/run_gen.py`, `eval_ladder/arms.yaml`): CFG scale 4.0 against the fixed negative prompt
  "worst quality, inconsistent motion, distorted, jittery", STG on block 29, 30 steps. So today's "null branch" of every
  base and trained generation is a quality-word text negative, not a transition null.
- DCG campaign (`misc/2026-08-14_dcg_conditioning/DOSSIER.md`): test-time guidance on the deployed IC-LoRA with the
  CONDITIONING dropped in the negative branch, w ∈ {1, 1.5, 3, 6}; `store/gens/015-018`, `029-032`. Read what was dropped
  and what moved.
- IC-LoRA reference conditioning (`misc/2026-08-27_dino_signal_training/SPEC.md` null semantics; `lora-flow` skill;
  `notes/INDEX.md`): the effect enters as a reference video (and, in the signal arms A1/A2/A4, as the 44-ch signal with
  "dropped = zeros"). A lerp can therefore enter the TRAINED model in three concrete ways: (a) as the reference VIDEO
  (rendered pixel lerp of the target's own endpoints, VAE-encoded), (b) as the reference SIGNAL (the LERP landmark's
  44-ch field — already computed for every landmark-source clip), (c) as a latent-space guidance target (LATLERP).
  For the BASE model only (c) and text drops exist.
- Eval workbench already renders 223 pixel-lerp nulls (`src/diffusion/transition_eval/workbench/build_cache.py`).
- `store/gens/028_dualforce_null_contrast`, `021_dualforce_contrast`: read their meta.v1.yaml for what "null" meant there.

## What to do (in order; write results as you go)
A. THE DEFAULT AS A TRAJECTORY, NOT A SUMMARY. For every clip in the focus groups + GT + R1 + R2 + landmarks, put the
   frame-by-frame trajectory into an anchor-relative, time-normalised frame per space (PIX, DINO, VAE; TRANS already
   is): progress τ(s) along the A→B chord, off-chord deviation ρ(s) in gap units, similarity/distance to A and to B,
   novelty ν(s) = min(dist to A, dist to B)/gap (and in TRANS the native nu/sA/sB/local curves), with s ∈ [0,1] over the
   interior. Produce mean ± std curves per group per space (raw time AND aligned by crossing time τ = 0.5, so swaps at
   different positions do not blur into a ramp), the per-clip spread, and 2–3 prototype clusters of the normalised
   curves (report cluster shares per group; the driver expects roughly a 20 % swap tail + 80 % body — verify or refute).
   State, in one paragraph with numbers, what the default trajectory IS in each space, and where the lerp of that space
   sits relative to it and relative to GT / R1.
B. GUIDANCE-DIRECTION TABLE. For each candidate null N ∈ {pixel LERP, LATLERP (VAE), DINO-space lerp (feature-space
   straight line — compute it from the cached CLS: no video exists), transport-space lerp (= the LERP landmark's TRANS
   curves), the model's own text-dropped branch (NULLGEN / R3)} and each descriptor d (DR, M, cross, nu_max, path/gap,
   step_share, TRANS nu_max, swap_sharp, local_at_peak, trans_mean): does sign(GT − N) agree with sign(GT − default)?
   I.e. would guiding AWAY from N push the default TOWARD real transitions on that descriptor, or further away? The
   driver's prior from TABLES.md: pixel lerp is softer than GT on swap_sharp (0.12 vs 0.23) and higher on M (0.5 vs 0.4)
   while the default is 0.56 / 0.06 — so pushing away from a lerp would push swap sharpness and mid-line mass in the
   WRONG direction, while it pushes novelty and straightness the right way. Verify with the data, per space, per stratum.
C. INFORMATION ARGUMENT, MADE PRECISE. "No information beyond the endpoints" is exactly true only in the space where
   the lerp is taken. Quantify what each lerp adds when read in the OTHER spaces (pixel LERP read in DINO: DR 0.45,
   TRANS novelty 0.28; LATLERP decoded = ? — you cannot decode, say so; DINO lerp = not renderable). Decide which space's
   lerp is the honest "endpoints-only" reference for each mechanism (a)/(b)/(c) above, and whether the model would see
   it as an in-distribution video (a rendered dissolve is a real editing transition; a decoded latent lerp probably is not).
D. THE RECOMMENDATION. Decide, with the informational argument AND the guidance-direction table AND realizability:
   for the CTT task (reference-conditioned trained arm) and for the base model separately, which null to give and how
   it enters. If a lerp is justified, say which space and which mechanism and what its known weakness is (from B). If
   the model's own dropped branch is better, say what the lerp is still good for (e.g. the eval normaliser, the signal
   arms' null signal). Set success criteria and design the ONE smallest GPU test that would settle it (endpoints, seeds,
   arms, what to measure with these descriptors, what result would falsify the recommendation). Do NOT run any GPU job.
E. Write `investigation/FINDINGS.md`: numbers with metric and bar; your readings attributed to you ("the advisor
   reads…"); a decisions section; a "could not verify" section. Keep it under ~2,500 words; put big tables in CSVs and
   figures in `investigation/fig_*.png` (matplotlib, 150 dpi). List every file you created.

Rules: no GPU, no Slurm submission, no commits, no secrets, never read `papers_drafts/ctt_iclr2027/AGENT_DONT_READ.md`,
`cv2.setNumThreads(1)`, serial decode, aarch64 env only, don't rename columns `clip`. Prior campaign conclusions are
inputs, not decisions — re-derive from the numbers.
