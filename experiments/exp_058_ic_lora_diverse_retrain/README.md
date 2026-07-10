# exp_058 — diversified mixed-conditioning IC-LoRA retrain

**COMPLETE 2026-07-08.** Sanity 9400364 (after link-env fix in 9399580),
train 9401247 (5000 steps, H200, loss →0.18), infer 9405052-58 (53/53), eval
9405337 → `outputs/eval/exp_058/quads/run_0001` + `analysis_v1v2.md`; viewer
PASS. **Findings: held-out vanish probe improved (0.41→0.61, appearance-blind
→ genuine), novel-texture gradient largely closed (0.33→0.42 cross), camera
flat (mechanism-limited), prefix-only generation works (leak 0.73 vs base
copy 0.99) — cost: anchors −0.06/−0.10 raw + seam snaps where suffix
conditioning meets prefix-only-trained classes. See
`notes/exp/exp_058_ic_lora_diverse_retrain.md`.**

## Question

Does retraining the transition IC-LoRA on a diversified corpus — 23 one-sided
classes with **prefix-only** conditioning + 9 two-sided classes with
prefix+suffix (460 pairs, 32 classes, 162 clips) — improve unseen-class /
novel-texture / camera transfer, enable prefix-only one-sided generation, and
hold the exp_056 two-sided anchors? Full pre-registration: `design.md`.

## Setup

- Conditioning: single per-pair `mask` condition (start 2 latent frames
  always; +final latent frame iff two-sided) — proven bit-exact vs exp_056's
  prefix(2)+suffix(1) by `test_mask_conditioning.py` (run in the trainer uv
  env). Reference concat unchanged.
- Held out: hero_flight, illustration_scene, gas_transformation,
  raven_transition, hole, seamless, jump + the exp_057 quad clips of large
  training classes. Training: fresh from base, rank 32/α32 attn+FFN, lr 2e-4,
  5000 steps, ckpt every 500, seed 42, 480×640×121@24.
- Captions: type-blind endpoint captions; ~120 new ones via Gemini from
  standardized first/last frames + manual spot-check + banned-word scrub.

### Resource plan

- Preprocess (162 clips encode + caption embeds): 1× H100 `secondary`,
  ~30 min walltime 1h. Skip-if-exists (idempotent).
- Sanity: 1-pair 50-step dry run incl. one validation sample, `secondary`,
  walltime 1h — validates the mask path + VRAM before the full run.
- Training: `HCESC-H200-secondary` (fallback H100-normal), walltime 4h/window,
  `--requeue` + `model.load_checkpoint` auto-resume, ckpt interval 500
  (~22 min max loss). 5000 steps ≈ 3.6 h + overhead → expect 1–2 windows.
- Eval inference (~66 gens): 5–6 chunk jobs, cluster `secondary`
  (campusclusterusers, gpu:H100:1), per-chunk output dirs, ABSOLUTE LoRA path.
- Scoring: 1× L40S `HCESC-L40S-normal` (feature cache warm from exp_057).

## How to run

```bash
cd $LAB/diffusion-research
# 1. corpus + captions + pairs (login-node safe)
python experiments/exp_058_ic_lora_diverse_retrain/standardize_train.py
python experiments/exp_058_ic_lora_diverse_retrain/caption_train.py      # Gemini
python experiments/exp_058_ic_lora_diverse_retrain/build_dataset.py --link
# 2. preprocess + sanity + train (Slurm)
sbatch experiments/exp_058_ic_lora_diverse_retrain/job_preprocess.sbatch
sbatch experiments/exp_058_ic_lora_diverse_retrain/job_sanity.sbatch
sbatch experiments/exp_058_ic_lora_diverse_retrain/job_train.sbatch
# 3. eval (after training) — see design.md §6
```

## Expected outcome

Pre-registered in `design.md` §7: anchors hold (±0.05 raw); novel-texture and
vanish held-outs improve iff coverage-limited; camera cross-target may not
(conditioning-conflict hypothesis); prefix-only one-sided is the new
capability under test.

## Outputs

- `outputs/training/exp_058_ic_lora_diverse_retrain/ic2/` — checkpoints
- `outputs/videos/exp_058_ic_lora_diverse_retrain/` — eval generations
- `outputs/eval/exp_058/` — harness runs + viewer
- W&B `creative-transition-transfer`, run `exp058_ic2`
