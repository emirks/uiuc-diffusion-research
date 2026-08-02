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

## evals
1. `001_five_arm__dai__2026-07-30` — v4 on DeltaAI, all five gens, 1,842 items/arm, 0 errors; headline: %_same ic_gen 83.1 / ctt_v2 82.5 / ctt_v2_leaky 91.3 / refvfx_A 42.4 / refvfx_B 33.0

2. `002_base_arms__dai__2026-07-31` — v4 on DeltaAI, gens 006+007, 1,842 items/arm; the trainings page's baselines

3. `003_bneck_coupling__dai__2026-08-02` — v4 on DeltaAI, gens 008+009, 1,842 items/arm, 0 nulls. **The frozen-coupling NEGATIVE**: P1 6/13 and 7/13 vs ≥11/13; P2 −0.002 vs +0.05 (95% CI upper bound below the bar). Liveness P3 GREEN (R 0.492) — the channel transmits, the model did not learn to decode

## datasets
1. `001_transitions_std121` — 222-clip eval corpus (stub → data/processed/transitions_std121)
2. `002_ctt_v2` — 56,368-pair training set (stub → datasets/ctt_v2)
