# exp_059 — LTX-2 inference benchmark (H100): dev two-stage vs distilled checkpoint

## Question

What is the granular wall-clock cost (cold and warm, per pipeline section) of
generating a 5 s clip with LTX-2 on a single H100 80GB, for the **dev** production
pipeline (`TI2VidTwoStagesPipeline`: 40 guided steps at half res + ×2 upsample +
3 distilled-LoRA steps) vs the **distilled** checkpoint (`DistilledPipeline`:
8 + 3 fixed sigmas, no guidance), at 720p (1280×704) and 1080p (1920×1088),
eager vs `torch.compile mode=reduce-overhead`?

## Setup

- Official `LTX-2` monorepo @ `7809842`, uv venv (`$LAB/LTX-2-official/.venv`,
  torch 2.9.1+cu128), bf16, **no offload** (`OffloadMode.NONE`), no quantization.
- **Fast-loading levers measured**: warm calls ride the OS page cache (stock
  `DummyRegistry` loader), `expandable_segments:True`, `max_batch_size=4` for
  the dev arms (guidance passes batched in one transformer call), compile arms
  use CUDA graphs with a per-arm `TORCHINDUCTOR_CACHE_DIR`.
  *(`StateDictRegistry` was tried and rejected: it caches state dicts on the
  GPU and LoRA fusion goes out-of-place to protect the cached copy →
  39+39 GB = OOM on 80 GB at the dev stage-2 build. Smoke job 9401275.)*
- One process per arm; per arm: 1 **cold** call (first disk read) +
  3 **warm** calls (seeds 42/43/44). Sections timed with perf_counter +
  cuda.synchronize fences: prompt_encode, transformer_build/denoise/free per
  stage, upsample, audio_decode, vae_decode, mux. Peak VRAM per call.
- 121 frames @ 24 fps ≈ 5.04 s, joint audio+video (production path).
- Weights: `$LAB/cache/huggingface/ltx2_models/` — dev + distilled 19B
  checkpoints, distilled LoRA 384 (strength 0.8, stage 2 of dev arms only),
  spatial upscaler x2; Gemma-3-12B QAT-q4 text encoder.

### Resource plan

- **GPU/queue**: H100 (benchmark target, per user request). Primary:
  cluster-wide `secondary` H100 (fast start, 4 h cap ≥ arm runtime); fallback
  `HCESC-H100-normal` / `-secondary` (ccc0439). One array task per arm so each
  arm gets a dedicated GPU and a cold process; `%N` throttle for etiquette.
- **Walltime**: 3:30 h per arm (worst arm ≈ dev_1080p_compile, est. ≤ 1.5 h + buffer).
- **Preemption-readiness**: skip-if-exists on `<arm>/timings.json` +
  `timings.partial.json` persisted after every call; `--requeue` set.
- **Mem**: 200 G (registry holds ~40 G checkpoint + ~24 G Gemma in CPU RAM), 16 CPUs.

## How to run

```bash
# smoke (tiny, ~10 min, validates harness):
sbatch --partition=secondary --account=campusclusterusers --gres=gpu:H100:1 \
  --time=00:45:00 --export=ALL,ARM=smoke_dist \
  experiments/exp_059_ltx2_inference_benchmark/job_bench.sbatch

# full matrix (8 arms):
sbatch --partition=secondary --account=campusclusterusers --gres=gpu:H100:1 \
  --array=0-7 experiments/exp_059_ltx2_inference_benchmark/job_bench.sbatch
```

## Expected outcome (pre-registered)

- Distilled ≈ 3–5× faster than dev end-to-end warm (11 unguided steps vs 40
  guided + 3), NOT ~7× (step ratio) because upsample/decode/mux and the
  audio branch are shared costs.
- Cold − warm ≈ 1–3 min, dominated by the 40 GB checkpoint read from the
  project filesystem; warm transformer_build a few seconds (CPU→GPU H2D of
  ~38 GB over PCIe).
- 1080p adds ~2.3× stage-2 + decode cost vs 720p (token count 32640 vs 14080);
  stage 1 scales identically (half res).
- compile reduce-overhead: 10–30 % faster denoise warm, but the per-call
  transformer rebuild forces re-trace/graph capture — first call much slower;
  possibly a net loss for one-off calls.

## Outputs

`outputs/videos/exp_059_ltx2_inference_benchmark/run_0001/<arm>/`:
`timings.json` (meta, per-call sections, peak VRAM), `run.log`, one mp4 per call
(`cold0_s42_1280x704.mp4`, `warm1_s42_…`). Table built by `make_table.py`.
