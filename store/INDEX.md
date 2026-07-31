# Store ledger

One line per entry, in seq order. Entry dirs are named `NNN_<slug>` — **the highest number IS the
latest**, on disk and here; numbers are never reused. Registering an entry = numbered dir +
meta.yaml (matching `seq:`) + a row here + CHANGELOG, in one commit.
Contract: README.md · flow: `lora-flow` skill.

## runs
1. `001_ic_gen` — LTX-2 19B IC-LoRA r32/α32 bidir, step 5000 shipped (ladder2 generalist)
2. `002_ctt_v2` — LTX-2 19B IC-LoRA r128/α128 one-way, step 10000 shipped (CTT v2 final training)
3. `003_refvfx` — EXTERNAL: Wan2.1-FLF2V-14B + refVFX LoRA r1024 + CausVid (weights → $LAB/cache/refvfx)

## gens  (all on the 152-row CTT grid × seeds 42/43 = 304 clips)
1. `001_ic_gen` — runs/001_ic_gen@5000, plain sksz prompts, 121f@24fps
2. `002_ctt_v2` — runs/002_ctt_v2@10000, plain sksz prompts, 121f@24fps
3. `003_refvfx_A` — runs/003_refvfx, their prompt convention (describes effect; text leak 35/62), 33f@6.5455fps
4. `004_refvfx_B` — runs/003_refvfx, our text budget (fixed token; leak-free 0/62), 33f@6.5455fps
5. `005_ctt_v2_leaky` — runs/002_ctt_v2@10000 (same file), prompts + effect clause; leaky mirror, no base twin

## evals
1. `001_five_arm__dai__2026-07-30` — v4 on DeltaAI, all five gens, 1,842 items/arm, 0 errors; headline: %_same ic_gen 83.1 / ctt_v2 82.5 / ctt_v2_leaky 91.3 / refvfx_A 42.4 / refvfx_B 33.0

## datasets
1. `001_transitions_std121` — 222-clip eval corpus (stub → data/processed/transitions_std121)
2. `002_ctt_v2` — 56,368-pair training set (stub → datasets/ctt_v2)
