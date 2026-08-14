# Store ledger

One line per entry, in seq order — **the highest number IS the latest**, on disk and here; numbers
are never reused. Registering an entry = numbered dir + meta.yaml (matching `seq:`) + a row here +
CHANGELOG, in one commit. Contract v2: README.md · arms: ARMS.md · v1→v2 map: MIGRATION.md ·
flow: `lora-flow` skill. Gens are `NNN_<arm>/KK_<variant>__<machine>` since 2026-08-13; old ids
resolve via `gens/_legacy/`.

## runs
1. `001_ic_gen` — LTX-2 19B IC-LoRA r32/α32 bidir, step 5000 shipped (ladder2 generalist)
2. `002_ctt_v2` — LTX-2 19B IC-LoRA r128/α128 one-way, step 10000 shipped (CTT v2 final training)
3. `003_refvfx` — EXTERNAL: Wan2.1-FLF2V-14B + refVFX LoRA r1024 + CausVid (weights → $LAB/cache/refvfx)
4. `004_bneck_frozen` — runs/002's recipe with the reference channel REPLACED by 72 frozen operator tokens (the certified transition encoder, held frozen — G2 verified bitwise at step 10000). Carries pre-registered caveats C1 (realized LR schedule, inherited) and C2 (S4 positional factors)
5. `005_vjepa_stageA_idea3` — bneck_redesign **Idea 3**: frozen V-JEPA 2 (vitl-fpc64-256) residual-trajectory features for the 18,800 CTTv2 reference demos (cached fp16 `feats/`) + trainable projector (LN→1024→512→GELU→128, 0.593M) warm-started via Stage-A SupCon τ=0.1 on the pooled code, labels = 146 base manners, hold-out 30 bases (same split as sibling gates). Pre-coupling go/no-go = median held-out base top-1 ≥ 7%. Projector `projector_stageA.pt`; results `stageA_report.json`. Coupling contract: projector→(C=128,F=16,H=3,W=3) per-clip per-channel standardized fp32. NOTE: seq may need renumber vs parallel Idea-2 at commit (coordinator owns commits; not pushed by Idea-3 agent).
6. `006_bneck_ctx_v2` — bneck_redesign **Idea 1 (CLEAN)**: runs/002's r128/α128 one-way recipe with the reference channel REPLACED by the frozen certified encoder's 72 operator tokens, compressed by a co-trained ContextAdapter (K'=16; 31 tensors, zero-init head) and injected as cross-attention CONTEXT (`inject=context`) — the RoPE-free route vs runs/004's concat-reference. Full-coverage single continuous run (`resume_data_position:true`), step 10000. Trainer src/LTX-2-bneck-coupling @ bneck_redesign 7ffbe95; checkpoint sha 844ee248
7. `007_surg1_wsd` — **SURG-1** objective-surgery redo of the V-JEPA-raw cell (runs/005 encoder, 144-tok code): high-σ timestep mixture (60% [0.9,1.0]) + code-swap contrastive gap loss (δ0.007/λ10, cross-class 124 groups) to FORCE the frozen DiT to build a reader; num_processes-correct **WSD** schedule (warmup100/stable1e-4/decay→1e-5), 4500 steps, eff-batch 8, **8×H100 on eps**. Reader formed (Δcross 0.042, manner-dominant, robust frozen≥live, no H-fire) but ~4× the raw-reader gap w/ mild appearance creep (ratio→0.26) — **VERDICT PENDING Gate A/B**. Trainer src/LTX-2-surg @ a4033230 (LOCAL, origin=Lightricks); ckpt sha cdc36bbb
8. `008_ctt_v3_wsd` — **NEW CHAMPION candidate ("ctt_v3")**: runs/002_ctt_v2 recipe (r128, one-way RAW ref, plain sksz prompt) with the LR schedule CORRECTED — ctt_v2 shipped a num_processes-mis-scaled `linear` schedule that floored LR at 1e-5 for **87.5%** of its 10k steps; this num_processes-correct **WSD** retrain (**6000 steps, 40% cheaper**) is a **MEASURABLE WIN**: paired same-seed Δ%same vs ctt_v2 **+5.0pp ALL-152 [+2.4,+7.6] / +5.5pp same-60**, headline **82.5→88.0**, copy-guard clean. Gen on eps co-located (repro gate BIT-IDENTICAL). src/LTX-2-surg @ a4033230 (surgery OFF); ckpt sha aa263cba. **PROVISIONAL** pending blind A/B.
9. `009_pushB_wsd_hs302050` — **the NEGATIVE**: runs/008 + SURG-1's high-σ timestep lean (30/20/50). Beats ctt_v2 on ALL-152 (+4.6pp) but NOT same-60; adds NOTHING over runs/008 (B−A −0.4pp [−2.9,+2.1], point-est negative). **High-σ lever CLOSED for raw readers** (it transfers to a bottleneck code, not a raw ref). ckpt sha 76a0f66d
10. `010_vap` — EXTERNAL: Video-As-Prompt (frozen Wan2.1-I2V-14B + MoT expert), bytedance @ 0f30aedf; weights → `$LAB/cache/vap` (65.87 GB self-contained). One-sided CTT baseline (I2V, no end-frame). Demo-gate PASS
11. `011_vfxmaster` — EXTERNAL: VFXMaster (CogVideoX-Fun-V1.1 **2b-aux + 5B-transformer** — the authors' own train/infer config; README 5b line inconsistent), libaolu312 @ 0632c5a; weights → `$LAB/cache/vfxmaster` (~24.5 GB). One-sided CTT baseline. Demo-gate PASS

## gens  (arm-first since v2; all on the 152-row CTT grid × seeds 42/43 = 304 clips unless noted)
> **External baselines** (refvfx/vap/vfxmaster) inference spec — provenance, exact recipe, frame handling, geometry/fps, parity-vs-authors — is in [`gens/EXTERNAL_BASELINES.md`](gens/EXTERNAL_BASELINES.md).

1. `001_ic_gen` — runs/001@5000 (IC-LoRA generalist) · `01_neutral__cc` 83.1 · `02_effect__dai` 89.1
2. `002_ctt_v2` — runs/002@10000 (CTT v2, superseded) · `01_neutral__eps` 82.5 · `02_effect__dai` (leaky) 91.3 · `03_effect__dai` regen 90.21 · `04_neutral__dai` regen 82.95 · `05_probe_dcg__dai` (300-row DCG cycle-0 probe, off-CTT grid) · `06_probe_ctl__dai` (controllability probe, 136 clips)
3. `003_refvfx` — runs/003 (EXTERNAL Wan2.1-FLF2V; 33f@6.5455fps) · `01_effect__dai` (their convention, leak 35/62) 42.4 · `02_neutral__dai` (our budget, leak-free) 33.0
4. `004_base_prompt` — NO adapter, no anchors · `01_effect__dai` (clause, no sksz) · `02_neutral__dai` (V-neutral) 58.1
5. `005_base_cond` — NO adapter + endpoint conditioning · `01_effect__dai` · `02_neutral__dai` (V-neutral) 60.4; vs 004 prices the anchors · `03_probe_ctl__dai` (controllability probe, 124 clips)
6. `006_bneck_frozen` — runs/004@10000 (72 frozen operator tokens) · `01_neutral__dai` · `02_neutral_shufcode__dai` (load-bearing corpse)
7. `007_bneck_ctx` — runs/006@10000 (16 context tokens) · `01_neutral__dai` · `02_neutral_shufcode__dai`
8. `008_surg1` — runs/007@4500 (V-JEPA 144-tok code, SURG-1) · `01_neutral__dai` · `02_neutral_shufcode__dai` (must-fail twin, scored lower → reads)
9. `009_ctt_v3` — runs/008@6000 (THE CHAMPION, provisional) · `01_neutral__eps` 87.98 · `02_neutral_shufref__eps` VOID (39 clips) · `03_effect__dai` 91.54 · `04_neutral__dai` 88.57 · `05_probe_ctl__dai` (controllability probe, 152 clips)
10. `010_ctt_v3_hs` — runs/009@6000 (high-σ negative, retired) · `01_neutral__eps` · `02_neutral_shufref__eps` VOID · `03_effect__dai` 90.00
11. `011_vap` — runs/010 (EXTERNAL VAP; **one-sided 112 rows** × 2 seeds = 224/variant; 49f @ 9.719fps, 480×640) · `01_neutral__dai` (Unseen app 37.5%) · `02_effect__dai` (36.9%) — text-inert
12. `012_vfxmaster` — runs/011 (EXTERNAL VFXMaster; one-sided 112 rows × 2 seeds; 49f @ 9.719fps, 480×640) · `01_neutral__dai` (Unseen app 39.0%) · `02_effect__dai` (42.5%)

## prompts  (the TWO sources — everything else is a stamp_rows transform, verified 152/152 exact)
1. `001_ctt152_neutral` — `{S1}. sksz. [{S2}.]` · sha 0d708175fbfe · derived: strip_sksz f2ebeedf2187 (base V-neutral) · swap_token_refvfx 11a50d24645a
2. `002_ctt152_effect` — `{S1}. sksz. {EFFECT}. [{S2}.]` · sha 35930d7d7453 · derived: strip_sksz d0460eaace93 (base effect) · template_refvfx b88a248dfafc
   (003–006 retired 2026-08-13 — they were exact transforms; numbers never reused)
7. `007_ctl_probe` — controllability-probe rows (13 rows × conditions, arm-free; file sha dfcbb07b926a; frozen phrase table in the campaign dossier)

## evals
> **Baseline reference scores** (the comparison scale for any coupling/treatment arm; all v4 · DeltaAI · reference sha `459fd9a7`, so mutually comparable). No-demo **floors**: `002` `base_prompt_ctt` / `base_cond_ctt` (CTT prompts) + `005` `base_cond_neutral` / `base_prompt_neutral` (V-neutral). Trained **reference levels**: `001` `ic_gen` 83.1 / `ctt_v2` 82.5, `005` `ic_gen_effect` 89.1. The two `005` neutrals are the V-neutral siblings of `002`'s base_ctt arms (`ic_gen_effect` is a treatment, not a floor).

1. `001_five_arm__dai__2026-07-30` — v4 on DeltaAI, all five gens, 1,842 items/arm, 0 errors; headline: %_same ic_gen 83.1 / ctt_v2 82.5 / ctt_v2_leaky 91.3 / refvfx_A 42.4 / refvfx_B 33.0

2. `002_base_arms__dai__2026-07-31` — v4 on DeltaAI, gens 006+007, 1,842 items/arm; the trainings page's baselines

3. `003_bneck_coupling__dai__2026-08-02` — v4 on DeltaAI, gens 008+009, 1,842 items/arm, 0 nulls. **The frozen-coupling NEGATIVE**: P1 6/13 and 7/13 vs ≥11/13; P2 −0.002 vs +0.05 (95% CI upper bound below the bar). Liveness P3 GREEN (R 0.492) — the channel transmits, the model did not learn to decode

4. `004_bneck_ctx_v2__dai__2026-08-06` — v4 on DeltaAI, gens 013+014, 1,842 items/arm, 0 nulls. **The context-inject Idea-1 NEGATIVE**: P1 6/13 and 6/13 vs bars ≥9/13 & ≥8/13; P2 −0.008 vs +0.10 (95% CI [−0.024,+0.013] includes 0). Dead-channel PASS (bitwise-identical), liveness P3 GREEN (R 0.584) — transmits but does not instruct. Scores symlinked from `scores_clean` (quota); recovered from a shared-cache write race (73/65 rows)

5. `005_ic_effect_neutral__dai__2026-08-07` — v4 on DeltaAI, gens 010+011+012, 1,842 items/arm, 0 errors/0 nulls. The metric_eval adapter×text arms. headline %_same: ic_gen_effect **89.1** / base_cond_neutral **60.4** / base_prompt_neutral **58.1** (%_proxy 84.4 / 47.0 / 44.9, ranking-only). ic_gen_effect is a TREATMENT (LEVEL only — its base twins aren't scored here); the two neutrals are V-neutral no-demo baselines (NOT the effect-clause base arms of evals/002). ic_gen_effect manifests reused+verified, neutral manifests rebuilt; scores symlinked from `scores_v4lane` (campaign-private v4 lane)

6. `006_surg1_wsd__dai__2026-08-11` — v4 on DeltaAI, gens 015 (matched) + 016 (deranged twin), 304 paired units, 0 copy. **SURG-1 Gate B — READS-BUT-WEAKLY-INSTRUCTS**: P1 same 7/13 & cross 9/13 (bars 9/8); P2 pooled median **+0.0199** vs bar 0.1016 (raw·ceiling·% = +0.020·+0.203·9.8%). ABOVE the dead encoder arms (6/13, −0.002) with genuine twin separation, but far under the win bar. Advisor CLOSED SURG-1 as a **publishable negative** — failure = forward-read→sampling **conversion** (G-fit ceiling +0.053), not channel-deadness (bneck) nor reader-absence (Gate A cleared both); retry budget unspent. Caveats: 484 symmetric pool-ref EOFErrors (paired Δ unbiased), [UNCERTIFIED]=branch-not-tagged (reference matches evals/001-004)

7. `007_ctt_v2_push__dai__2026-08-12` — v4 on DeltaAI, arms `ctt_v2_pushA` + `ctt_v2_pushB` vs the ctt_v2=82.5 baseline (evals/001, bit-identical eps-gen → clean paired Δ, ZERO gen-machine term). Analysis reproduced ctt_v2=82.49 first. **Arm A (runs/008 ctt_v3) = MEASURABLE WIN**: paired Δ%same **+4.99pp ALL-152 [+2.38,+7.63] / +5.49pp same-60 [+1.88,+9.38]**, headline **87.98**, COPY-GUARD PASS (near_copy 0/0, copy_max 0.269 vs 0.255). Arm B wins on 152 not same-60; **B−A ≈0 (high-σ inert, Arm C not fired)**. Shufref controls VOID (eps run_gen too old). **The win = the LR-schedule fix, not high-σ.** Advisor a38825fa: provisional champion pending blind A/B. [UNCERTIFIED]=branch-not-tagged

8. `008_ctt_v2_effect_2x2__dai__2026-08-12` — the EFFECT-prompt follow-on: a CO-LOCATED adapter×prompt 2×2 on DeltaAI (gens/021-025) to decontaminate the never-co-located "+8.8 text gain". **Verdict (advisor afefa334): the pre-registered NULL fires — "the text channel SATURATES the adapter gain."** Under the effect prompt the champion's edge washes out (pushA_effect 91.54 ≈ leaky_regen 90.21; primary Δ −0.22pp [−1.94,+1.45]); all effect arms sit at the ~91 regen-consistency ceiling. The champion's text gain is significantly SMALLER than v2's (DiD −4.62pp [−7.54,−1.67]) — the corrected adapter reads from the reference what text otherwise supplies. B retired (null). Copy-guard clean (near_copy 0, copy_max 0.322). Co-located v2 text gain = +7.3pp (vs published +8.8; baselines drift ≤1.1pp). **Champion stays 88.0-PLAIN — effect numbers are text-assisted characterization only.** F-block = addendum to the plain-campaign block. [UNCERTIFIED]

9. `009_ctl_vqa__dai__2026-08-13` — **controllability probe (VQA, NOT the v4 harness — pre-registered deviation)**: 412 clips, 3 arms × 5 axes. Calibration 70% < 85% bar → VQA DEMOTED per pre-registration; color rescued by the hue corroborator (99% FP-check). **Confirmatory NOT declared** — champion ≈ ctt_v2 at attribute override (paired p=0.50; hue instrument even puts v3 ahead 87.5 vs 81.2). Color controllable on all arms; density/speed/direction text-inert everywhere (except sub-categorical shadow_smoke thinning 12/12); GRAFT works v3 6/6; unseen-row replication 87.5%. Owner adjudication pending (adjudicate.html). [DESCRIPTIVE-CORROBORATED]

10. `010_external_baselines__dai__2026-08-14` — v4 on DeltaAI, arms `vap_{neutral,effect}` + `vfxmaster_{neutral,effect}`, **one-sided 112 rows × 2 seeds** (2876 pool-rows/arm). Prior-works baseline snapshot (misc/2026-08-13_baseline_metric_table). **Externals ≪ our arms on GT-anchored neutral pool-%** (best external VFXMaster 39.0 vs champion 77.3 = ~38pp). VAP/VFXMaster text-inert (neutral≈effect); refvfx (evals/001) text-leaning. two-sided cells N/A (structural). Competitor-metric lenses in the campaign dir (impl_sha d63935f4). [UNCERTIFIED]

## datasets
1. `001_transitions_std121` — 222-clip eval corpus (stub → data/processed/transitions_std121)
2. `002_ctt_v2` — 56,368-pair training set (stub → datasets/ctt_v2)
