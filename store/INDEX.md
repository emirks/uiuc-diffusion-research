# Store ledger

One line per entry. See README.md for the contract.

## runs
- `ic_gen` — LTX-2 19B IC-LoRA r32/α32 bidir, step 5000 shipped (ladder2 generalist)
- `ctt_v2` — LTX-2 19B IC-LoRA r128/α128 one-way, step 10000 shipped (CTT v2 final training)
- `refvfx` — EXTERNAL: Wan2.1-FLF2V-14B + refVFX LoRA r1024 + CausVid (weights → $LAB/cache/refvfx)

## gens  (all on the 152-row CTT grid × seeds 42/43 = 304 clips)
- `ic_gen` — runs/ic_gen@5000, plain sksz prompts, 121f@24fps
- `ctt_v2` — runs/ctt_v2@10000, plain sksz prompts, 121f@24fps
- `ctt_v2_leaky` — runs/ctt_v2@10000 (same file), prompts + effect clause; leaky mirror, no base twin
- `refvfx_A` — runs/refvfx, their prompt convention (describes effect; text leak 35/62), 33f@6.5455fps
- `refvfx_B` — runs/refvfx, our text budget (fixed token; leak-free 0/62), 33f@6.5455fps

## evals
- `five_arm__dai__2026-07-30` — v4 on DeltaAI, all five gens, 1,842 items/arm, 0 errors; headline: %_same ic_gen 83.1 / ctt_v2 82.5 / ctt_v2_leaky 91.3 / refvfx_A 42.4 / refvfx_B 33.0

## datasets
- `ctt_v2` — 56,368-pair training set (stub → datasets/ctt_v2)
- `transitions_std121` — 222-clip eval corpus (stub → data/processed/transitions_std121)
