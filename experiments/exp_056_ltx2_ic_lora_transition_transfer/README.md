# exp_056 — IC-LoRA in-context transition transfer (official LTX-2 trainer)

**COMPLETE 2026-07-08.** Training 9383890 (H200, 3:21 h, EXIT 0); 46-quadruple
suite generated (jobs 9386907 base + 9388132/33 + racing twins); harness scores
`outputs/eval/exp_056/quads/run_0002` (W&B `exp056_quads`); interactive viewer
`outputs/eval/exp_056/viewer` (381/381 assets). **Outcome: in-context transfer
works** — cross-class appearance 0.65±0.35 at leak 0.61 with negative seams and
0.97 endpoint DINO, while the BASE model near-copies the reference's content
(leak 0.95–0.98, seams +2.4..+7.1); unseen-class (jump) transfers motion
semantics but not appearance. Details: `notes/exp/exp_056_ic_lora_transition_
transfer.md`, CHANGELOG 2026-07-08.

## Question

Can a single IC-LoRA learn **in-context transition transfer**: given the
endpoints of one clip ([2 start + 1 end latent] conditioning) and a **whole
other clip of the same transition class as an in-context reference**, generate
the target's middle by *reading the transition style off the reference* rather
than off the text? The captions are deliberately type-blind (fixed generic
phrase "The scene transforms into", no mention of smoke/fire/melt/... as an
effect), so the reference video is the **only** carrier of the transition type
— if the model uses it, one adapter should apply any demonstrated transition,
including ones never trained (jump_transition is fully held out as an
unseen-class reference probe).

## Setup

- **Trainer**: official `$LAB/LTX-2-official` @ `7809842` (uv env) — the
  **V2V IC-LoRA** mode (`reference` condition of the `flexible` strategy)
  composed with the exp_051-validated C2V endpoint conditions
  (`prefix temporal_boundary=2` + `suffix temporal_boundary=1`), all
  `probability: 1.0`. Verified against `flexible.py`: intrinsic conditions
  apply first (endpoint latents pinned clean/timestep-0/loss-excluded), then
  the reference latent sequence is concatenated BEFORE the target (clean,
  timestep 0, loss-excluded, full bidirectional attention, positions identical
  to the target's grid). Supervision = middle 13/16 latent frames of the
  target. Loss slices `pred[:, -target_len:]`.
- **Base model**: `ltx-2-19b-dev.safetensors` + Gemma-3-12B QAT text encoder
  (same as exp_050/051), staged in `$LAB/cache/huggingface/`.
- **Data standardization** (`standardize_clips.py`): all 47 clips in
  `data/processed/transitions/` → `data/processed/transitions_std121/` at
  exactly **480×640 (W×H portrait 3:4) × 121 frames @ 24 fps** (x264 crf14).
  121 evenly-spaced frame indices (122f → trim; 242f → exact 2× decimation,
  arc preserved at 2× speed; the 150f raven clip → even spread); resize-to-
  cover + center crop (13 landscape clips lose lateral content — accepted,
  transitions are center-weighted). Names normalized (`shadow_smoke.mp4` →
  `shadow_smoke_0.mp4`, `raven_transiton_2` typo fixed).
- **Captions** (`dataset/captions.json`): per-clip, endpoint-only —
  "<scene A>. The scene transforms into <scene B>." Written from the
  standardized clips' first/last frames; **no transition-mechanism words**.
  Trigger `ICTRANS` prepended at preprocessing (`--lora-trigger`).
- **Pairs** (`build_dataset.py` → `dataset/pairs.json`): within each class of
  n≥2 clips, circulant ordered pairs — target *i* takes clips *i+1..i+min(3,n−1)*
  (mod n) as references → **131 pairs** / 10 classes (shadow_smoke 30,
  firelava 18, earth_wave 15, air/water/raven/melt/flying_cam 12 each,
  display 6, flame 2). Every clip appears as target AND as reference equally
  often; max class share 23% (vs 43% if all 208 ordered pairs were used).
  jump_transition (n=1) excluded → unseen reference class.
- **Preprocessing trick — encode once, pair by symlink**: `process_dataset.py`
  names every output after the row's *target* path (verified in
  `process_videos.py`), so pair rows sharing a target would collide. Instead:
  the 47 unique clips are preprocessed ONCE (`dataset/.precomputed_clips/`,
  single bucket `480x640x121`, `--skip-audio --decode`), then
  `build_dataset.py --link` assembles `dataset/.precomputed/
  {latents,conditions,reference_latents}/<class>/<target>__ref_<ref>.pt` as
  symlinks (target latent, target caption, reference latent).
  `PrecomputedDataset` matches sources by identical relative path and
  `torch.load` follows symlinks. Sequence per step = 4800 (ref) + 4800
  (target) = **9600 tokens** (exp_050 measured 48.4 GB peak @ 4800; H100 80GB
  expected to fit — sanity job confirms before the full run).
- **Training config** (`config_ic.yaml`): official `v2v_ic_lora.yaml` defaults
  — rank 32/α32, video attention + FFN targets, lr 2e-4 linear, 3000 steps,
  bs 1, adamw, bf16, no quant, grad ckpt, ckpt/250 (keep all). W&B project
  `creative-transition-transfer`.
- **Validation during training** (every 250 steps + step 0 = base-model rung;
  seed 42, 480×640×121@24, 30 steps, CFG 4.0, STG 1.0 `stg_v`; reference
  side-by-side in output):
  1. **cross-class**: earth_wave_0 endpoints + shadow_smoke_3 reference —
     must impose smoke morphology on foreign endpoints;
  2. **unseen-class**: same endpoints + jump_transition_1 reference (family
     never trained) — the pure in-context probe;
  3. **in-class**: shadow_smoke_0 endpoints + shadow_smoke_1 reference (a
     training pair) — training-fit gauge.
  Endpoint condition clips pre-cut per the exp_051 suffix rule
  (`cond_*_start9.mp4` `num_frames: 9`; `cond_*_end9.mp4` `num_frames: 8` —
  causal-VAE backward bleed avoided).
- **Resource plan**: 1× H100 80GB via `HCESC-H100-secondary` (+cluster-wide
  `secondary` fallback, gres `gpu:H100:1` case-sensitive). Sanity+preprocess
  job ~1.5–2h (`job_sanity.sbatch`, time 3:00). Full run: 3000 × ~3 s/step
  ≈ 2.5h + 13 validation rounds × 3 samples ≈ 1.5–2h → ~4.5h ⇒ **chain 3
  copies** of resume-aware `job_train.sbatch` (--requeue, DONE marker,
  resume from latest checkpoint). Preemption loses ≤250 steps (~13 min).

## How to run

```bash
cd $LAB/diffusion-research
mkdir -p outputs/logs/slurm
# 0. (login node, done) standardize + captions + manifests:
#    python experiments/exp_056_ltx2_ic_lora_transition_transfer/standardize_clips.py
#    python experiments/exp_056_ltx2_ic_lora_transition_transfer/build_dataset.py
# 1. preprocess + pair tree + 50-step sanity (proves memory & the 3-condition path)
sbatch --partition=HCESC-H100-secondary --account=hcesc-h100 --gres=gpu:1 --requeue \
    experiments/exp_056_ltx2_ic_lora_transition_transfer/job_sanity.sbatch
# 2. full training (chain 3 copies to survive 4h windows / preemption)
J1=$(sbatch --parsable -J exp056_ic --partition=HCESC-H100-secondary --account=hcesc-h100 \
    --gres=gpu:1 --requeue experiments/exp_056_ltx2_ic_lora_transition_transfer/job_train.sbatch config_ic.yaml)
J2=$(sbatch --parsable -J exp056_ic --dependency=afterany:$J1 --partition=HCESC-H100-secondary --account=hcesc-h100 \
    --gres=gpu:1 --requeue experiments/exp_056_ltx2_ic_lora_transition_transfer/job_train.sbatch config_ic.yaml)
sbatch -J exp056_ic --dependency=afterany:$J2 --partition=HCESC-H100-secondary --account=hcesc-h100 \
    --gres=gpu:1 --requeue experiments/exp_056_ltx2_ic_lora_transition_transfer/job_train.sbatch config_ic.yaml
```

## Expected outcome

Pre-registered:
(a) **In-class validation** (sample 3) should converge fastest — endpoints +
reference + trained pairing; expect recognizable class morphology by step
250–500 (exp_051's c2v arm acquired its concept by 250 with endpoint pinning;
the reference adds signal, but the *task* — infer the effect from a second
video — is harder than memorizing one style).
(b) **Cross-class transfer** (sample 1) is the experiment's core claim: if the
adapter reads the reference in-context, the smoke morphology should appear on
earth_wave endpoints WITHOUT smoke words in the prompt. Failure mode to watch:
the model ignores the reference and produces a generic crossfade (then
captions-only training signal was enough to satisfy the loss, and the
reference tokens were ignored), or it *copies* reference content
(subjects/background leaking from the reference video instead of its
transition style — M6-style leakage).
(c) **Unseen-class reference** (sample 2, jump_transition): genuine in-context
generalization. Expected weakest; any systematic imitation of the jump
transition's timing/geometry would be a strong positive result.
(d) Endpoint anchoring should hold everywhere (mechanism-robust per exp_051).
(e) Specialization: with all conditions p=1.0 the adapter needs reference +
endpoints at inference; plain T2V behavior will drift (accepted, matches
exp_051's c2v arm).

## Outputs

- `outputs/training/exp_056_ltx2_ic_lora_transition_transfer/sanity/` — 50-step
  sanity run (step-0/50 validation ×3 samples).
- `outputs/training/exp_056_ltx2_ic_lora_transition_transfer/ic/` — checkpoints
  (`lora_weights_step_*.safetensors` + training states, every 250), validation
  ladder `samples/step_*` (3 samples × 13 rungs, reference side-by-side), logs.
- **Post-training quadruple suite** (added at completion): `make_quads.py` →
  `dataset/quads.json` + `dataset/manifest_ic.json` (46 quadruples, 10 arms);
  `run_ic_inference.py` + `job_infer.sbatch` (chunked ValidationRunner
  inference — NOTE: chunks need separate output dirs, see the note);
  generations in `outputs/videos/exp_056_ltx2_ic_lora_transition_transfer/
  {ic_lora,base}/quads/`. Harness: `run_score_ic.py` + `eval_config.yaml` +
  `job_eval.sbatch` → `outputs/eval/exp_056/quads/run_0002/`. Viewer:
  `build_viewer_ic.py` + `viewer_template_ic.html` →
  `outputs/eval/exp_056/viewer/` (serve: `python -m http.server`).
- `dataset/.precomputed_clips/` — 47 clip latents + text embeddings (+
  `decoded_videos/` verification decodes); `dataset/.precomputed/` — 131-pair
  symlink tree.
- W&B: `creative-transition-transfer`, tags `exp_056`.
- Slurm logs: `outputs/logs/slurm/exp056_*.out`.
