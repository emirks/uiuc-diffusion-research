# exp_059 — LTX-2 inference benchmark on H100 (dev two-stage vs distilled checkpoint)

**Status: COMPLETE (2026-07-08). Jobs 9401444 (eager+registry, array 0-3,8-9) + 9403087 (compile arms, py3.12 env, array 4-7), all on cluster-`secondary` H100s (ccc0419/23/24).**

## Question

Granular wall-clock cost (cold vs warm, per pipeline section) of a 5 s
(121f @ 24 fps, joint AV) generation on one H100 80GB:

- `TI2VidTwoStagesPipeline` (dev 19B ckpt; 40 guided steps @ half res → ×2
  spatial upsample → 3 distilled-LoRA steps @ target res)
- `DistilledPipeline` (distilled 19B ckpt; 8 unguided + 3 fixed-sigma steps)
- at 1280×704 ("720p") and 1920×1088 ("1080p"), eager vs
  `torch.compile mode=reduce-overhead`.

## Method

One process per arm; cold call (empty `StateDictRegistry` → disk loads) + 3 warm
calls (registry-cached state dicts). Sections timed with perf_counter +
cuda.synchronize fences by monkeypatching `DiffusionStage._transformer_ctx`
(build/free), `DiffusionStage.run` (denoise), `PromptEncoder`, `VideoUpsampler`,
`AudioDecoder`, plus a timing iterator around the lazy VAE decode inside
`encode_video`. bf16, no offload, no quantization, `max_batch_size=4` (dev),
`expandable_segments:True`. Weights on `$LAB` (project FS).

## Pre-registered expectations

1. Distilled ≈ 3–5× faster end-to-end warm than dev (not the naive 51/11 step
   ratio — upsample/decode/mux/audio are shared costs).
2. Cold − warm gap 1–3 min, dominated by the ~40 GB checkpoint read.
3. 1080p ≈ +2.3× on stage-2/decode sections vs 720p; stage-1 cost scales the same.
4. compile/reduce-overhead helps warm denoise 10–30 %, but per-call transformer
   rebuild forces re-trace → cold call substantially slower; may be net-negative
   at this call pattern.

## Results

Full granular table: `make_table.py --run-dir outputs/videos/exp_059_ltx2_inference_benchmark/run_0001`.
5 s = 121 f @ 24 fps, joint AV, seed-invariant (±5 % across warm seeds). All numbers seconds.

**Warm end-to-end per video (page-cached / steady-state):**

| arm | 720p (1280×704) | 1080p (1920×1088) | peak VRAM |
|---|---|---|---|
| dev eager | 142 | 215 | 38.3 / 42.2 G |
| dev compile (reduce-overhead) | 134 | 198 | 39 / 43.6 G |
| distilled eager | 103 | 116 | 38.2 / 42.1 G |
| distilled compile | 101 | 116 | 38.8 / 43.4 G |
| **distilled + GPU-resident weights (registry)** | **11.2** | **23.4** | 68.9 / 72.2 G |

**Warm section split (eager arms):** dev 720p = prompt_encode 22 + s1_build 31 +
s1_denoise 40 (40 steps @640×352) + upsample 2 + s2_build 37–55 (LoRA fusion) +
s2_denoise 3.4 (3 steps @1280×704) + decode+mux 2.9. dev 1080p: s1_denoise 98
(@960×544), s2_denoise 10.3 (@1920×1088), decode 3.7. Distilled swaps
s1_denoise to 2.3/5.1 (8 unguided steps) — **the model compute is 6–16 s; the
other ~90 s of the eager warm total is per-call model rebuilding** (stock
`DiffusionStage` frees the transformer after every stage; Gemma rebuilt every
call).

**Registry mode** (`StateDictRegistry`, state dicts cached ON the GPU) removes
all rebuild cost: transformer_build 0.27 s, prompt_encode 1.7 s → 11 s/video at
720p, 23 s at 1080p. Only possible for the distilled pipeline: the dev
pipeline's stage-2 LoRA fusion runs out-of-place to protect the registry copy
(39+39 GB) → OOM on 80 GB (job 9401275). Costs +30 GB VRAM (69–72 GB peak).

**Cold (first call in a process):** +25–65 s over warm when files are
page-cached. True disk-cold on an untouched node (first-ever read of the
checkpoint off the project FS): gemma 23 GB ≈ 400 s, checkpoint 40 GB ≈ 860 s →
dist eager cold totals of ~1430 s. Budget ~5–25 min for the first video on a
fresh node depending on cache state.

**torch.compile verdict:** unsupported on the repo venv (Python 3.14, torch
2.9.1 raises) — compile arms ran a parallel `.venv-py312` (uv sync --frozen
--python 3.12). Warm gain: dev −6–8 % (s1_denoise 40→35, 98→92), distilled ≈ 0
(too few steps; loads dominate). One-time compile tax on the cold call:
+10 min (dev 720p cold 787 s). Per-call transformer rebuild forces re-wrap but
the inductor cache holds — warm calls stay compiled. Not worth it below ~10
videos/process; registry mode dominates it anyway for distilled.

**Expectations vs outcome:** (1) distilled 3–5× faster — CONFIRMED only in
registry mode (13–9×); in stock eager mode just 1.4–1.9× (rebuild overhead
swamps the step advantage). (2) cold−warm 1–3 min — page-cache case confirmed;
disk-cold far larger (up to +22 min). (3) 1080p scaling — s2/decode ≈ +3×
tokens as predicted, s1_denoise +2.5×. (4) compile — direction right,
magnitude smaller (6–8 %, not 10–30 %).

Videos: run_0001/<arm>/*.mp4 (40 clips; distilled arms audibly/visually
noisier at 8 steps but coherent; not a quality eval).
