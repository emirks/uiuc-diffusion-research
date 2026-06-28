# exp_050 — Standard-LoRA fine-tuning baseline on the shadow-smoke clips (official LTX-2 trainer)

**Status: COMPLETE — all 3 arms trained to 2000 steps, concept acquired, no drift. 2026-07-04.**

## What was run

First training experiment of the campus-cluster era, and the first use of the **official
LTX-2 trainer** (clean clone at `$LAB/LTX-2-official`, commit `7809842`, 2026-06-17) instead
of the pod-modified vendored `src/LTX-2`. Everything at the official `t2v_lora.yaml`
80GB-tier defaults (rank 32/α32, lr 1e-4 linear→0, 2000 steps, bs 1, adamw, bf16, no
quantization, gradient checkpointing, checkpoints+validation every 250 steps).

- **Data**: the 10 `data/processed/transitions/shadow_smoke/` clips (no audio streams).
  Hand-written per-clip captions from frame inspection — scene A → fixed concept phrase
  ("a dense mass of black smoke sweeps across the frame and engulfs …") → scene B.
  Trigger `SHDWSMK` prepended by `--lora-trigger` at preprocessing.
- **Preprocessing**: 3 AR buckets `480x640x121 / 640x480x121 / 576x576x121` (F=121 ≈ full
  5 s @ 24 fps), batch_size 1 (multi-bucket requirement). Output layout gotcha: latents
  mirror the *absolute* source path under `latents/` — count with `find`, not `ls`.
- **Arms** (1× H100 each): `baseline` = video-only attention targets;
  `i2v_ff05` = + `first_frame` condition p=0.5 (official style-LoRA recommendation —
  trains T2V and I2V jointly); `rank64_ffn` = + rank/α 64 + FFN targets.

## Results (fixed-seed validation ladder, steps 0→2000)

- **Concept acquired by ~step 500–1000, fully styled by 2000.** Held-out trigger prompt
  (raincoat woman on pier → chef): the base model (step 0) already attempts a smoke
  transition from the caption text but renders a generic gray explosion filling the frame.
  By step 1000–2000 all arms produce the training clips' signature morphology — a dense,
  rounded, ink-black billow that **wraps/engulfs the subject** (not the frame), sweeps
  across with tendrils, then reveals scene B cleanly with a dissolving wisp.
- **No drift**: unrelated no-trigger prompt (golden retriever park) stays a clean park
  scene at step 2000 in all 3 arms — no smoke bleed, no quality loss (rank64_ffn included,
  contrary to the pre-registered concern).
- **Arm comparison at eyeball level**: all three succeed; `rank64_ffn` renders the densest
  most-literal smoke mass; `i2v_ff05` ≈ baseline quality while ALSO supporting first-frame
  conditioning → **best default for downstream C2V work**. Rank 32 attention-only is
  sufficient for concept acquisition on 10 clips; capacity increase is not required.
- **Cost**: 58–72 min per arm on one H100 (1.16 s/step at ~4800-token seq len, peak 48.4 GB
  — near-miss for a 48GB L40S). Preprocess of all 10 clips: minutes.

## Infra facts worth reusing

- **Preemptible/idle-node queues did all the work**: `HCESC-H100-secondary` starts in
  ~100 s by preempting scavenger jobs while `-normal` quoted 3 days; the cluster-wide
  `secondary` partition ran both sweep arms simultaneously on other investors' idle H100
  nodes (ccc0423/0424, 4h cap).
- **Resume is real**: checkpoints ship with `training_state_step_*.pt`; `model.load_checkpoint`
  → "✅ LoRA checkpoint loaded" and the global step continues (sanity: 50→60). A completed
  run refuses to resume ("initial_step >= target_steps") — harmless. Trainer zero-pads
  step numbers (`step_02000`) — match patterns accordingly.
- Experiment scaffold: `experiments/exp_050_ltx2_lora_shadowsmoke/` (dataset.json, 4 configs,
  `job_sanity.sbatch`, resume-aware chain-safe `job_train.sbatch`).

## Open questions / next steps

- Checkpoint ladder (250…2000) is saved for all arms — picking an earlier stop (e.g. 1000)
  vs 2000 for best style-vs-fidelity tradeoff needs a proper side-by-side.
- Quantitative eval (CLIP/FVD or vcbench-style) not run — this was a visual baseline.
- Inference-side: load `lora_weights_step_*.safetensors` in ltx-pipelines (or diffusers)
  and test on real C2V endpoint conditioning; compare against the exp_046/047 injection
  recipes — the reference point this baseline exists to provide.
