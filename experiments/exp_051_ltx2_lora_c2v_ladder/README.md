# exp_051 — C2V capability ladder: t2v vs i2v vs c2v LoRA conditioning modes

## Question

How does the conditioning mode used during LoRA training affect **C2V transition
generation** (both endpoint clips given, generate the middle)? We compare, at
identical settings, C2V inference on:

1. **base** — LTX-2 19B, no LoRA (step-0 anchor)
2. **t2v** — exp_050 `baseline` LoRA (no conditioning during training)
3. **i2v_ff05** — exp_050 arm (first_frame p=0.5 during training)
4. **c2v** — NEW arm trained with prefix(2 latent) + suffix(1 latent) conditions,
   both p=1.0 — i.e. trained exactly the way it is used at inference

The test conditions come from a **different transition family** (earth_wave_0:
bookstore woman → dirt wave → staircase woman), so every arm must impose the
learned SHDWSMK black-smoke transition between endpoints it has never seen,
overriding the source clip's own transition type. This is the "creative
transition" capability the ladder measures.

## Setup

- **Trainer/inference**: official `$LAB/LTX-2-official` @ `7809842`; C2V inference
  runs through the trainer's own `ValidationRunner` (`run_c2v_inference.py`) — the
  identical code path as in-training validation, for all arms.
- **c2v arm training** (`config_c2v.yaml`): identical to exp_050 baseline (rank
  32/α32, video-only attention targets, lr 1e-4 linear, 2000 steps, bs 1, bf16,
  ckpt/250) except `training_strategy.video.conditions = [prefix tb=2 p=1.0,
  suffix tb=1 p=1.0]`. Conditioned latent frames get clean latents/timestep 0 and
  are excluded from the loss → supervision only on the middle 13 of 16 latent
  frames. Reuses exp_050 `.precomputed` latents + SHDWSMK captions verbatim (
  conditions are applied at train time; no re-preprocessing).
- **Test transitions ×3** (endpoint-clean, verified by frame inspection):
  earth_wave_0 (16:9: bookstore woman → staircase afro woman), earth_wave_1
  (4:3: man in cap on old-town street → woman tying shoe at market stall),
  earth_wave_2 (portrait, runs at 480×640: hilltop blonde → desert close-up).
- **Conditioning clips** (`dataset/cond_ew{0,1,2}_{start9,end9}.mp4`): 9-frame
  cuts. Prefix: `num_frames: 9` → 2 latent frames at position 0. Suffix: the
  9-frame end clip encodes to 2 latents and `num_frames: 8` keeps only the FINAL
  latent (the true last 8 px frames). The windows are deliberately minimal: the
  causal VAE's receptive field reaches backward, so any longer suffix window
  would bleed the source clip's own dirt-wave transition into the end-anchor
  latent (user-caught issue). The prefix is safe by causality regardless.
  Trade-off noted: training-side suffix latents come from full-clip encodes
  (deeper context); `cond_end_last121.mp4`/`cond_end_last17.mp4` (ew0) are kept
  on disk to A/B window lengths if end-anchor artifacts appear.
- **C2V inference protocol** (same for every arm): 640×480×121 @ 24 fps for the
  landscape clips, 480×640×121 for ew2 (per-sample `video_dims` override), seed
  42, 30 steps, CFG 4.0, STG 1.0 (`stg_v`); 6 samples per arm = 3 transitions ×
  (trigger prompt, no-trigger prompt — trigger-dependence probe). Prompts are
  caption-style: scene A → smoke phrase → scene B, hand-written per test clip.
  The c2v arm's in-training validation uses the ew0 trigger sample plus the
  exp_050 golden-retriever drift prompt, so its 0→2000 ladder accrues during
  training (step 0 = base-model rung); full training logs to W&B project
  `creative-transition-transfer`.
- **Compute**: 1× H100 80 GB per job via `HCESC-H100-secondary` / cluster
  `secondary` (preemptible; job_train.sbatch is resume-aware + chain-safe).

## How to run

```bash
cd $LAB/diffusion-research
# 1. sanity (50 steps + step-0/50 C2V validation; proves the conditioning path)
sbatch --partition=HCESC-H100-secondary --account=hcesc-h100 --gres=gpu:1 --requeue \
    experiments/exp_051_ltx2_lora_c2v_ladder/job_train.sbatch config_c2v_sanity.yaml
# 2. full c2v training (chain 2 copies to survive preemption/4h windows)
J1=$(sbatch --parsable -J exp051_c2v --partition=HCESC-H100-secondary --account=hcesc-h100 \
    --gres=gpu:1 --requeue experiments/exp_051_ltx2_lora_c2v_ladder/job_train.sbatch config_c2v.yaml)
sbatch -J exp051_c2v --dependency=afterany:$J1 --partition=HCESC-H100-secondary --account=hcesc-h100 \
    --gres=gpu:1 --requeue experiments/exp_051_ltx2_lora_c2v_ladder/job_train.sbatch config_c2v.yaml
# 3. inference ladder (base + exp_050 arms can run while c2v trains)
sbatch -J exp051_inf_base ... job_infer.sbatch base
sbatch -J exp051_inf_t2v  ... job_infer.sbatch t2v  $LAB/diffusion-research/outputs/training/exp_050_ltx2_lora_shadowsmoke/baseline/checkpoints/lora_weights_step_02000.safetensors
sbatch -J exp051_inf_i2v  ... job_infer.sbatch i2v_ff05 .../i2v_ff05/checkpoints/lora_weights_step_02000.safetensors
sbatch -J exp051_inf_c2v  ... job_infer.sbatch c2v  .../exp_051_ltx2_lora_c2v_ladder/c2v/checkpoints/lora_weights_step_02000.safetensors
```

## Expected outcome

Pre-registered: (a) **base** should produce a plausible interpolation but a generic
transition (gray explosion / crossfade), not the signature ink-black
subject-wrapping billow; (b) **t2v** LoRA has never seen conditioned tokens —
C2V inference is out-of-distribution for it; it may still style the middle but
risks endpoint discontinuity (visible jumps at frame 9 / frame 113) or ignoring
the anchors; (c) **i2v_ff05** has seen first-frame conditioning (p=0.5) but never
suffix conditioning — expected better start-anchor coherence than t2v, weaker end
anchoring; (d) **c2v** is train/inference matched — expected the cleanest endpoint
adherence and the smoothest anchored transition; its risk is degraded
unconditioned generation (both conditions were always on), observable in the
drift-prompt validation sample. Trigger-dependence probe: if sample 2 (no
SHDWSMK) still produces black smoke, the concept lives in the caption phrase /
conditioning rather than the trigger token.

## Outputs

- `outputs/training/exp_051_ltx2_lora_c2v_ladder/c2v/` — checkpoints, in-training
  validation samples (step 0…2000 C2V ladder + drift), logs.
- `outputs/videos/exp_051_ltx2_lora_c2v_ladder/{base,t2v,i2v_ff05,c2v}/samples/` —
  final ladder: 6 samples per arm (ew0/ew1/ew2 × trigger/no-trigger).
- W&B: training run `creative-transition-transfer/runs/7iptdfyt`; inference runs
  `exp051_infer_*`.
- Slurm logs: `outputs/logs/slurm/exp051_*.out`.

**Completed 2026-07-06.** Jobs: sanity 9348958 (11:32); c2v training 9349026
(1:19:01, cluster `secondary` H100 ccc0424); inference base/t2v/i2v 9349028-30
(ran in parallel with training on the same node); c2v inference 9350243 (13:02).
**Outcome:** all pre-registered expectations except (b)-risk: (a) base = generic
frame-filling smoke curtain (with a fire hallucination on ew2), no subject
interaction; (b) t2v transfers the signature subject-wrapping billow with NO
endpoint discontinuity — the conditioning mechanism holds anchors even for a
LoRA that never saw conditioned tokens (pre-registered risk not realized);
(c) i2v_ff05 ≈ t2v, marginally cleaner sweeps; (d) c2v = tightest subject-wrap
onset, most continuous approach into the end anchor (scene-B composition
converges to the anchor framing earliest), and most stable scene-A hold. Plus
two unregistered findings: **c2v acquires the concept by step 250** (vs 500–1000
for exp_050 t2v — endpoint pinning concentrates all supervision on the
transition middle), and **the concept rides the caption phrase, not the
trigger** (no-trigger samples show the full morphology in every LoRA arm).
Drift: no smoke bleed in any arm; c2v's unconditioned generation shows
composition shift at fixed seed (it never trained unconditioned — expected
specialization cost; use p<1.0 if one LoRA must also serve plain T2V).
Details: CHANGELOG 2026-07-06, `notes/exp/exp_051_c2v_ladder.md`.
