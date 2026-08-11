# Store ledger

One line per entry, in seq order. Entry dirs are named `NNN_<slug>` — **the highest number IS the
latest**, on disk and here; numbers are never reused. Registering an entry = numbered dir +
meta.yaml (matching `seq:`) + a row here + CHANGELOG, in one commit.
Contract: README.md · flow: `lora-flow` skill.

## runs
1. `001_ic_gen` — LTX-2 19B IC-LoRA r32/α32 bidir, step 5000 shipped (ladder2 generalist)
2. `002_ctt_v2` — LTX-2 19B IC-LoRA r128/α128 one-way, step 10000 shipped (CTT v2 final training)
3. `003_refvfx` — EXTERNAL: Wan2.1-FLF2V-14B + refVFX LoRA r1024 + CausVid (weights → $LAB/cache/refvfx)
4. `004_bneck_frozen` — runs/002's recipe with the reference channel REPLACED by 72 frozen operator tokens (the certified transition encoder, held frozen — G2 verified bitwise at step 10000). Carries pre-registered caveats C1 (realized LR schedule, inherited) and C2 (S4 positional factors)
5. `005_vjepa_stageA_idea3` — bneck_redesign **Idea 3**: frozen V-JEPA 2 (vitl-fpc64-256) residual-trajectory features for the 18,800 CTTv2 reference demos (cached fp16 `feats/`) + trainable projector (LN→1024→512→GELU→128, 0.593M) warm-started via Stage-A SupCon τ=0.1 on the pooled code, labels = 146 base manners, hold-out 30 bases (same split as sibling gates). Pre-coupling go/no-go = median held-out base top-1 ≥ 7%. Projector `projector_stageA.pt`; results `stageA_report.json`. Coupling contract: projector→(C=128,F=16,H=3,W=3) per-clip per-channel standardized fp32. NOTE: seq may need renumber vs parallel Idea-2 at commit (coordinator owns commits; not pushed by Idea-3 agent).
6. `006_bneck_ctx_v2` — bneck_redesign **Idea 1 (CLEAN)**: runs/002's r128/α128 one-way recipe with the reference channel REPLACED by the frozen certified encoder's 72 operator tokens, compressed by a co-trained ContextAdapter (K'=16; 31 tensors, zero-init head) and injected as cross-attention CONTEXT (`inject=context`) — the RoPE-free route vs runs/004's concat-reference. Full-coverage single continuous run (`resume_data_position:true`), step 10000. Trainer src/LTX-2-bneck-coupling @ bneck_redesign 7ffbe95; checkpoint sha 844ee248
7. `007_surg1_wsd` — **SURG-1** objective-surgery redo of the V-JEPA-raw cell (runs/005 encoder, 144-tok code): high-σ timestep mixture (60% [0.9,1.0]) + code-swap contrastive gap loss (δ0.007/λ10, cross-class 124 groups) to FORCE the frozen DiT to build a reader; num_processes-correct **WSD** schedule (warmup100/stable1e-4/decay→1e-5), 4500 steps, eff-batch 8, **8×H100 on eps**. Reader formed (Δcross 0.042, manner-dominant, robust frozen≥live, no H-fire) but ~4× the raw-reader gap w/ mild appearance creep (ratio→0.26) — **VERDICT PENDING Gate A/B**. Trainer src/LTX-2-surg @ a4033230 (LOCAL, origin=Lightricks); ckpt sha cdc36bbb

## gens  (all on the 152-row CTT grid × seeds 42/43 = 304 clips)
1. `001_ic_gen` — runs/001_ic_gen@5000, plain sksz prompts, 121f@24fps
2. `002_ctt_v2` — runs/002_ctt_v2@10000, plain sksz prompts, 121f@24fps
3. `003_refvfx_A` — runs/003_refvfx, their prompt convention (describes effect; text leak 35/62), 33f@6.5455fps
4. `004_refvfx_B` — runs/003_refvfx, our text budget (fixed token; leak-free 0/62), 33f@6.5455fps
5. `005_ctt_v2_leaky` — runs/002_ctt_v2@10000 (same file), prompts + effect clause; leaky mirror, no base twin
6. `006_base_prompt_ctt` — NO adapter (base weights), prompt only (effect clause, no `sksz`), no anchors, no demo
7. `007_base_cond_ctt` — NO adapter, same prompt + endpoint conditioning, still no demo; 006 vs 007 prices the anchors
8. `008_bneck_frozen` — runs/004@10000; the raw reference REPLACED by 72 frozen operator tokens (certified encoder, G2 bitwise). Twin of 009
9. `009_bneck_frozen_shufcode` — SAME adapter file as 008, code source deranged via `code_source_reference` (row's own `reference` untouched, so both twins share a byte-identical GT pool). The load-bearing corpse
10. `010_ic_gen_effect` — runs/001_ic_gen@5000 (SAME adapter as gens/001), effect clause after the `sksz` token (leaky convention); the missing adapter×text 2×2 cell, twin of gens/001_ic_gen. Scored in evals/005
11. `011_base_cond_neutral` — NO adapter (base weights), V-neutral prompt (start scene only; `sksz` + effect clause both removed) + endpoint conditioning, no demo. The specificity-zero anchor. Scored in evals/005
12. `012_base_prompt_neutral` — NO adapter, V-neutral prompt only, no conditioning, no demo. The cleanest zero; 011 vs 012 prices the anchors. Scored in evals/005
13. `013_bneck_ctx_v2` — runs/006@10000; the raw reference REPLACED by the frozen 72-token code injected as 16 CONTEXT tokens (`inject=context`). Twin of 014
14. `014_bneck_ctx_v2_shufcode` — SAME adapter file as 013, code source deranged via `code_source_reference` (row's own `reference` untouched, so both twins share a byte-identical GT pool). The load-bearing corpse
15. `015_surg1_wsd` — **SURG-1 Gate B MATCHED**: runs/007 @ step4500, V-JEPA 144-tok code reference (backbone-free gen, trained projector from ckpt), 152×2. Twin of 016. Reads-but-weakly (see evals/006)
16. `016_surg1_wsd_shufcode` — SAME adapter as 015, code source cross-class deranged (`code_source_reference`), reference untouched → byte-identical GT pool. The must-fail twin (it scored LOWER → the channel reads)

## evals
> **Baseline reference scores** (the comparison scale for any coupling/treatment arm; all v4 · DeltaAI · reference sha `459fd9a7`, so mutually comparable). No-demo **floors**: `002` `base_prompt_ctt` / `base_cond_ctt` (CTT prompts) + `005` `base_cond_neutral` / `base_prompt_neutral` (V-neutral). Trained **reference levels**: `001` `ic_gen` 83.1 / `ctt_v2` 82.5, `005` `ic_gen_effect` 89.1. The two `005` neutrals are the V-neutral siblings of `002`'s base_ctt arms (`ic_gen_effect` is a treatment, not a floor).

1. `001_five_arm__dai__2026-07-30` — v4 on DeltaAI, all five gens, 1,842 items/arm, 0 errors; headline: %_same ic_gen 83.1 / ctt_v2 82.5 / ctt_v2_leaky 91.3 / refvfx_A 42.4 / refvfx_B 33.0

2. `002_base_arms__dai__2026-07-31` — v4 on DeltaAI, gens 006+007, 1,842 items/arm; the trainings page's baselines

3. `003_bneck_coupling__dai__2026-08-02` — v4 on DeltaAI, gens 008+009, 1,842 items/arm, 0 nulls. **The frozen-coupling NEGATIVE**: P1 6/13 and 7/13 vs ≥11/13; P2 −0.002 vs +0.05 (95% CI upper bound below the bar). Liveness P3 GREEN (R 0.492) — the channel transmits, the model did not learn to decode

4. `004_bneck_ctx_v2__dai__2026-08-06` — v4 on DeltaAI, gens 013+014, 1,842 items/arm, 0 nulls. **The context-inject Idea-1 NEGATIVE**: P1 6/13 and 6/13 vs bars ≥9/13 & ≥8/13; P2 −0.008 vs +0.10 (95% CI [−0.024,+0.013] includes 0). Dead-channel PASS (bitwise-identical), liveness P3 GREEN (R 0.584) — transmits but does not instruct. Scores symlinked from `scores_clean` (quota); recovered from a shared-cache write race (73/65 rows)

5. `005_ic_effect_neutral__dai__2026-08-07` — v4 on DeltaAI, gens 010+011+012, 1,842 items/arm, 0 errors/0 nulls. The metric_eval adapter×text arms. headline %_same: ic_gen_effect **89.1** / base_cond_neutral **60.4** / base_prompt_neutral **58.1** (%_proxy 84.4 / 47.0 / 44.9, ranking-only). ic_gen_effect is a TREATMENT (LEVEL only — its base twins aren't scored here); the two neutrals are V-neutral no-demo baselines (NOT the effect-clause base arms of evals/002). ic_gen_effect manifests reused+verified, neutral manifests rebuilt; scores symlinked from `scores_v4lane` (campaign-private v4 lane)

6. `006_surg1_wsd__dai__2026-08-11` — v4 on DeltaAI, gens 015 (matched) + 016 (deranged twin), 304 paired units, 0 copy. **SURG-1 Gate B — READS-BUT-WEAKLY-INSTRUCTS**: P1 same 7/13 & cross 9/13 (bars 9/8); P2 pooled median **+0.0199** vs bar 0.1016 (raw·ceiling·% = +0.020·+0.203·9.8%). ABOVE the dead encoder arms (6/13, −0.002) with genuine twin separation, but far under the win bar. Advisor CLOSED SURG-1 as a **publishable negative** — failure = forward-read→sampling **conversion** (G-fit ceiling +0.053), not channel-deadness (bneck) nor reader-absence (Gate A cleared both); retry budget unspent. Caveats: 484 symmetric pool-ref EOFErrors (paired Δ unbiased), [UNCERTIFIED]=branch-not-tagged (reference matches evals/001-004)

## datasets
1. `001_transitions_std121` — 222-clip eval corpus (stub → data/processed/transitions_std121)
2. `002_ctt_v2` — 56,368-pair training set (stub → datasets/ctt_v2)
