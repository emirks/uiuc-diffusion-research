# External baseline inference spec

Authoritative record of **how the prior-works baselines were run** — provenance, the exact inference
recipe, frame handling, geometry, fps, and every parity decision vs the authors' originals. Scope: the
three externals in the CTT baseline table — **refVFX** (`runs/003`, `gens/003_refvfx`), **VAP**
(`runs/010`, `gens/011_vap`), **VFXMaster** (`runs/011`, `gens/012_vfxmaster`). EffectMaker = code
unreleased, cite-only.

**Where this disagrees with the workers/manifests on disk, the disk wins.** Depth + reasoning:
`misc/2026-08-13_baseline_metric_table/DOSSIER.md` (§FULL INFERENCE-PARITY AUDIT) and `…/DISCLOSURES.md`.
Verified by a 3-agent primary-source (file:line) parity audit, 2026-08-14.

---

## 0. The one-line verdict

Parity with the authors is **excellent**. Every sampling knob (scheduler, steps, guidance, negatives,
dtypes, flags) MATCHES each model's released recipe. The only content-changing deviations are **deliberate
and disclosed**: portrait geometry (480×640, to keep our content undistorted and uniform across all arms)
and, for the neutral/ablation tiers, an emptied reference-text channel. All three run at **each model's
NATIVE output length** (refVFX 33f, VAP/VFXMaster 49f), with the reference video uniformly subsampled to
match. Our own method arms are 121f — so externals are shorter by design (see §5).

---

## 1. Provenance (what each model IS)

| arm | method | upstream @ commit | weights (durable) | env |
|---|---|---|---|---|
| **refvfx** | Wan2.1-FLF2V-14B-720P + **refVFX LoRA** (rank 1024, step-10000) + **CausVid** few-step LoRA (rank 32) + swapped pipeline units. First-last-frame (two-sided). | `maxwelljones14/refVFX` @ `e62c2c04…` — **UNOFFICIAL CMU reimpl** of arXiv:2601.07833 (disclose: no official release) | `$LAB/cache/refvfx/weights` (87 GB) | `$LAB/envs-aarch64/refvfx` |
| **vap** | Wan2.1-I2V-14B (frozen) + **MoT expert**, fused. Image-to-video, start-frame only (one-sided). | `bytedance/Video-As-Prompt` @ `0f30aedf…` · arXiv 2510.20888 | `ByteDance/Video-As-Prompt-Wan2.1-14B` rev `f0d6ab47` (65.87 GB self-contained) | `$LAB/envs-aarch64/vap` |
| **vfxmaster** | CogVideoX-Fun-V1.1 **2b-InP aux** (VAE/T5/sched) + **5B VFXMaster transformer** (ckpt-40000, in_ch33). I2V, start-only (one-sided). | `libaolu312/VFXMaster` @ `0632c5a9…` · arXiv 2510.25772 · adapter `8ruceLi/VFXMaster` | base `alibaba-pai/CogVideoX-Fun-V1.1-2b-InP` + adapter (`$LAB/cache/vfxmaster`, ≈24.5 GB) | `$LAB/envs-aarch64/vfxmaster` |

Workers (FROZEN — reused verbatim across v1 and the Phase-2 author-config re-run):
`$LAB/external/{vap,vfxmaster}/gen_worker_*.py`, `$LAB/diffusion-research/misc/refvfx_baseline/gen_worker.py`.
Note the 2b-aux+5B config for VFXMaster is the authors' **scripts'** config; their README's "5b-aux" line is
inconsistent with the scripts (disclosed). VFXMaster's `DDIM_Origin` silently drops `snr_shift_scale=3.0`
on BOTH sides — parity-preserving; do NOT "fix" it to `CogVideoXDDIMScheduler`.

---

## 2. Inference recipe (authors vs ours — all MATCH unless noted)

| param | refVFX | VAP | VFXMaster |
|---|---|---|---|
| scheduler | refVFX flow sampler, `sigma_shift 5.0` | `UniPCMultistepScheduler`, flow_shift 3.0 (NOT the docstring's FlowMatchEuler) | `DDIM_Origin` (DDIMScheduler; v-pred, zero-SNR, trailing) |
| steps | **6** | 50 | 50 |
| guidance | `cfg 6.0 · cfg_ref 2.0 · cfg_input 0.0` | `guidance_scale 5.0` (plain CFG) | `guidance_scale 6.0` + `use_dynamic_cfg=True` |
| negative prompt | "static, blurry, worst quality, low quality" | default 392-char Wan string (×2: main + mot_ref) | default 167-char string (byte-exact) |
| dtypes | base recipe | img-enc **fp32**, VAE **fp32**, transformer+T5 **bf16** | transformer / VAE / T5 all **bf16** |
| special | `strict_end_image=True`, `empty_context_for_ref=False`, **`control_video=None`** (never leak GT middle), LoRA+CausVid stack | `frames_selection="evenly"`, `last_image=None`, caption-CFG off, `use_vfx_token` n/a | `use_vfx_token=False`, `use_dynamic_cfg=True`, noise_aug σ=0.0563 |
| offload/tiling | cpu_offload off (96 GB GH200) | none | none |
| seeds | 42, 43 | 42, 43 | 42, 43 |

**Verdict:** every substantive knob matches the authors' recommended recipe. Deviations that exist are
cosmetic/justified (see §4).

---

## 3. Frame handling — reference subsample + output length (each at the model's NATIVE spec)

Our source clips are **121f @ 24fps (5.04 s)**, uniform 480×640. Per model:

| arm | reference video | start conditioning | frames GENERATED | export fps → duration |
|---|---|---|---|---|
| **refvfx** | 121 → **33**, uniform 4n+1 (`sample_frames`, gen_worker.py:89) | **first + last** frame (FLF2V, two-sided) | **33** (native) | 6.545 → 5.04 s |
| **vap** | 121 → **49**, evenly (`select_frames "evenly"`, gen_worker_vap.py:122) | first frame only (I2V) | **49** (native) | 9.719 → 5.04 s |
| **vfxmaster** | 121 → **49**, evenly (`select_frames "evenly"`, gen_worker_vfxmaster.py:61) | first frame only (I2V) | **49** (native) | 9.719 → 5.04 s |
| *our method arms* | (full) | first+last (endpoint) | **121** | 24 → 5.04 s |

Subsampling to the native length is REQUIRED, not optional: VFXMaster concatenates the reference latent
onto the target latent along the frame axis with no internal check, so a non-49 reference mis-sizes its
rotary embeddings / crashes. fps is chosen to **duration-match** all clips to 5.04 s for cross-arm motion
comparability (metadata only — restampable without regenerating).

---

## 4. Geometry, fps, and the deviations (all disclosed)

- **Geometry = 480×640 portrait for all three** (our content's native, uniform across every arm and every
  reference). VAP's single training bucket is 480×832 landscape and its pipeline *stretches* to target h/w
  — so **480×640 is the ONLY geometry where VAP's stretch-preprocessing is an identity op** (pixel-faithful
  conditioning); 480×832 would corrupt portrait content ~2.3× (anamorphic) and pillarboxing wastes ~57% of
  the frame. refVFX's 480×640 is within its own `max_pixels=399360`. VFXMaster is native-resolution (no
  bucket). **Decision (advisor 2026-08-14): keep 480×640** — also pinned by the pre-registered paired-vs-v1
  analysis (the v1 anchors are 480×640; the bitwise repro-probe confirms 0 drift, so any Δ is prompt-only).
- **fps: 9.719 (49f) / 6.545 (33f), duration-matched** — canonical. Authors' native stamps (VAP 16, VFXMaster
  8, refVFX 15) are a robustness column for any *time-based* metric only; frame-index metrics are stamp-invariant.
- **Cosmetic/unmatchable:** seeds {42,43} vs authors' single 42 (superset; @42 reproduces authors); VFXMaster's
  ref noise-aug (σ=0.0563) draws from global RNG which the authors leave unseeded (nondeterministic) — we seed
  it (better practice, exact match impossible); VAP per-row generator ≡ authors' global-seed for seed 42.
- **control_video is NEVER set** (refVFX) — passing it would leak the GT middle. First+last frames on two-sided
  rows are the endpoint-task definition, not a leak.

---

## 5. Cross-cutting rules for anyone scoring/comparing these

- **Score ALL arms (externals + our method) on ONE machine** with the pinned shas (v4 `reference_v4`
  `459fd9a7`, corpus 222, τ_copy 0.858; competitor impl `d63935f4`). eps↔DeltaAI does not reproduce at the
  0.005 bar — mixing machines breaks the comparison.
- **`core_degenerate` / `copy_max` are NOT comparable across frame counts** (externals 33f/49f vs our 121f) —
  mask-geometry artifact, not model quality. Exclude or footnote for externals.
- Externals are **one-sided only** except refVFX (two-sided). VAP/VFXMaster populate the 112 one-sided grid
  rows; two-sided cells are structurally N/A (a finding, not a gap).
- refVFX arm A gives it MORE text than our method's arms receive (disclosed) — it is refVFX "at its strongest,"
  the peer of the VAP/VFXMaster author-config arms.

---

## 6. Prompting per arm-variant (text channels)

Two text channels per model: a **target** channel (`prompt`) and a **reference** channel
(`prompt_mot_ref` for VAP, `ref_prompt` for VFXMaster; refVFX folds effect into its single prompt).

| arm-variant | target `prompt` | reference channel | prompt shelf |
|---|---|---|---|
| `vap`/`vfxmaster` **`authorcfg`** (Phase-2, author-intended) | `{S1_endpoint}. {EFFECT}.` | `{S1_reference}. {EFFECT}.` | `prompts/008_ext112_authorcfg` (sha `73787305eb4a`) |
| `vap`/`vfxmaster` **`tgtfull_refempty`** (channel-decomposition ablation) | `{S1_endpoint}. {EFFECT}.` | *(empty)* | 008 (ref blanked) |
| `vap`/`vfxmaster` **`neutral`** (v1 — S1-only anchor) | `{S1_endpoint}.` | *(empty)* | `prompts/001 ·ext` |
| `vap`/`vfxmaster` **`effect`** (v1 — 1-line ref) | `{S1_endpoint}.` | genericised effect clause | `prompts/002 ·ext` |
| `refvfx` **A / effect** (author-faithful) | `{S1}. Make it so that the beginning of the scene is unchanged, but during the video {effect}.` | — | `prompts/002 ·template_refvfx` |
| `refvfx` **B / neutral** (de-texted control) | `…during the video the visual effect is applied.` | — | `prompts/001 ·swap_token_refvfx` |

`{EFFECT}` = the reference clip's genericised operator clause (`misc/refvfx_baseline/reference_effects.json`,
keyed by reference). The refVFX template IS refVFX's own preset sentence (their green_fog/pixelated/statue
presets), a faithful adaptation. The `authorcfg` construction was an owner directive (2026-08-14): both channels
`{S1}.{EFFECT}.` — the reference channel carries the effect because the eval_ladder `captions()` are motion-blind
and inconsistently convey it. Parity disclosure: the effect clause then appears in BOTH external channels vs the
champion's single `{S1}.sksz.{EFFECT}` — more effect-text exposure, accepted as the author-faithful choice.

---

## 7. Two-endpoint (TEG) baselines — grid-v3 TWO-SIDED rows (added 2026-09-20)

A separate baseline family for the paper's "both endpoints given" (TEG) block: prior-work / base
backbones run over the grid-v3 **two-sided** rows (`sided=="two"`) — 38 zero-shot + 36 seen/unseen =
74 rows × seeds {42,43} = 148 clips per system. The paper TEG block uses the **zero-shot n=76** subset;
seen+unseen is supplementary (queued behind ZS with a Slurm `afterok`). Campaign record + workers/manifests:
`misc/2026-09-20_teg_baselines/RECORD.md`.

**Prompt (all TEG baselines): the EFFECT prompt byte-equal to what the LTX-2 `base_cond` arm received**
— pulled verbatim from `store/gens/005_base_cond/06_effect_v3__dai/grid.jsonl` ("<start scene>. <effect
clause>. <end scene>.", trained token stripped), keyed by (endpoint, reference, cell, sided, ref_novelty).
No prompt extension/rewriting. fsck corpus_sha over the 148-row grid = `6b55919bcb5e` (ZS-76 subset =
`35b81da20630`). This is a **text-budget-matched** choice (same text every TEG system gets), so it differs
from the refVFX one-sided `author_native` tier (§6), which uses refVFX's own template.

**Endpoint conditioning (all TEG baselines):** `input_image` = frame 0 of `eval_ladder/conds/<endpoint>_start9.mp4`
(the prefix our own arms condition on); `end_image` = last frame of `<endpoint>_end9.mp4` (= `end9[8]` =
frame 120 of the 121-frame target). Geometry 480 wide × 640 tall.

| arm / gen | backbone | reference? | recipe | frames · out fps | negative prompt |
|---|---|---|---|---|---|
| **refvfx** `refvfx_effect_v3` (`gens/003_refvfx/04_effect_v3__dai`) | Wan2.1-FLF2V-14B-720P + refVFX LoRA(step-10000) + CausVid | **yes** (121→33, uniform, by the FROZEN worker) | refVFX fast path (unchanged): 6 steps, cfg 6.0, cfg_ref 2.0, cfg_input 0.0, sigma_shift 5.0, strict_end_image | 33 f · 6.5455 fps | "static, blurry, worst quality, low quality" (refVFX's own) |
| **wan_flf2v** `wan_flf2v_effect_v3` (`gens/042_wan_flf2v/01_effect_v3__<machine>`) | Wan2.1-FLF2V-14B-720P **BASE** (no LoRA, no CausVid) | **no** (pure first-last interpolation, `use_reference:false`) | Wan2.1 flf2v defaults: 50 steps, cfg 5.0, sigma_shift 16, no offload, tiled VAE; `control_video=None` | 81 f · 16 fps | DiffSynth/Wan canonical default (Chinese, 137 chars) |
| **wan_vace** `wan_vace_effect_v3` (`gens/043_wan_vace/01_effect_v3__<machine>`) | Wan2.1-VACE-14B **BASE** (DiT+VACE from the 7 shards), first-last CLIP extension | **no** (`use_reference:false`; endpoints baked into `vace_video`+mask) | Wan2.1 vace defaults: 50 steps, cfg 5.0, sigma_shift 16, no offload, tiled VAE; `vace_reference_image=None`, `vace_scale=1.0` | 81 f · 16 fps | DiffSynth/Wan canonical default (Chinese) |

- **refVFX two-sided** reuses the frozen worker `misc/refvfx_baseline/gen_worker.py` verbatim (same recipe
  as the existing refVFX arms); only the manifest (two-sided rows, base_cond effect prompt) is new.
- **wan_flf2v** worker `misc/2026-09-20_teg_baselines/gen_worker_flf2v.py` mirrors the refVFX authors' own
  base-model runner (`run_base_model.py::build_base_pipeline`): a plain `WanVideoPipeline.from_pretrained`
  over the FLF2V bundle (DiT shards + T5 + VAE + CLIP image encoder + umt5 tokenizer), default units, no
  LoRA; call = the `--pure_flf2v` branch (input+end image, `control_video=None`). Deviations from that
  script's argparse defaults (documented in the campaign RECORD): negative → Wan canonical default; sigma_shift
  5.0 → 16 (the official DiffSynth FLF2V inference example's value); fps 15 → 16; geometry 480×832 → 480×640.
- **Disclosures:** refVFX = UNOFFICIAL CMU reimpl of arXiv:2601.07833 (as §1). wan_flf2v = 480p on a
  720p-trained model. Both run at their own native output length (refVFX 33 f, wan_flf2v 81 f), fps chosen to
  hold the 5.04 s duration (refVFX) / per Wan's flf2v convention (wan_flf2v).
- **wan_vace** worker `misc/2026-09-20_teg_baselines/gen_worker_vace.py` = the stock DiffSynth VACE path
  (provenance `examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B.py`): `from_pretrained` loads DiT+VACE
  from the 7 VACE shards (`$LAB/cache/wan_vace/Wan2.1-VACE-14B`, 63.3 GB), reusing the FLF2V bundle's T5/VAE
  (sha256 byte-identical: T5 `7cace0da…`, VAE `38071ab5…`) and its umt5 tokenizer; no CLIP. **Endpoint
  conditioning** (owner decision): endpoints resampled to 16 fps and given as `vace_video` (81 frames) —
  out 0..5 = start9 idx [0,2,3,5,6,8], out 77..80 = end9 idx [4,5,7,8] (end9[8]=target frame 120), frames
  6..76 = mid-gray (127,127,127); `vace_video_mask` keep(0=black) on the 10 given frames, generate(1=white)
  elsewhere. The VACE unit pools the mask to (81+3)//4=21 latents (nearest-exact): latents 0,1 fully given &
  keep, latent 20 fully given & keep. The 6 start + 4 end given frames per endpoint are saved at
  `misc/2026-09-20_teg_baselines/conds_16fps/` (PNGs + small mp4s) for the metrics. Disclosure: 480p portrait,
  base VACE (no fine-tune), effect conveyed by text only, endpoints downsampled to 16 fps.

## 8. NEUTRAL-prompt twins of every grid-v3 external arm (added 2026-09-21)

The paper compares every system under two text conditions — **neutral** (the start-scene caption only, no
effect text; the reference video, where the system takes one, is the only effect signal) and **effect**.
Until 2026-09-21 the prior works had only their *effect-side* grid-v3 entries (`author_native` on the
one-sided zero-shot set, `effect_v3` on the two-sided TEG rows). This section adds their neutral twins.
Campaign record, builder, manifests and sbatch files: `misc/2026-09-21_neutral_baselines/` (RECORD.md).

**Construction rule — everything identical, only the text changes.** Each neutral manifest is a row-by-row
clone of the frozen manifest that produced the existing entry (`build_neutral_manifests.py` asserts that the
only differing keys are the text channels plus `harness_arm`/`arm`/`variant`/`item_id`/`out_name`):
same rows, seeds {42,43}, endpoint frames/clips, reference clips, geometry, frame counts, fps, recipe,
negative prompts and FROZEN workers.

- `prompt` ← the NEUTRAL text the LTX-2 `base_cond` arm received, verbatim (`store/gens/005_base_cond/04_neutral_v3__dai/grid.jsonl`
  for HF rows, `05_neutral_v3ed81__dai` for EffectData rows; = `prompts/010` with the task token stripped),
  keyed by (cell, endpoint, reference). One-sided rows: the start-scene caption S1. Two-sided rows: "S1 S2"
  (both scene captions, no effect clause). Identical to the grid-v2 neutral text where the endpoints overlap
  (checked: 224/224 VAP rows).
- VAP `prompt_mot_ref` ← `""`, VFXMaster `ref_prompt` ← `""` (zero effect text in ANY channel — the same rule as
  the grid-v2 neutral entries `gens/011_vap/01`, `gens/012_vfxmaster/01`, §6).
- refVFX: the prompt is the base_cond neutral text ONLY. Its own template clause ("Make it so that the
  beginning of the scene is unchanged, but during the video the visual effect is applied") used by the grid-v2
  `refvfx_B` entry is NOT added — this keeps the one-sided neutral twin text-identical to VAP's/VFXMaster's and
  the two-sided twin consistent with `refvfx_effect_v3`, which already uses base_cond text without the template.
  (Disclose: differs from `gens/003_refvfx/02_neutral`.)

| arm / gen | twin of | rows | prompt channels | machine |
|---|---|---|---|---|
| **vap** `vap_neutral_v3` (`gens/011_vap/06_neutral_v3__dai`) | `05_author_native` | one-sided ZS 183 pairs × 2 = 366 (81 HF + 102 ED) | prompt = S1; prompt_mot_ref = "" | dai (same as its twin) |
| **vfxmaster** `vfxmaster_neutral_v3` (`gens/012_vfxmaster/06_neutral_v3__dai`) | `05_author_native` | 366 | prompt = S1; ref_prompt = "" | dai |
| **refvfx** `refvfx_neutral_v3` (`gens/003_refvfx/05_neutral_v3__dai`) | `03_author_native` (first frame + reference, end_image None) | 366 | prompt = S1 | dai |
| **refvfx** `refvfx_neutral_v3_teg` (`gens/003_refvfx/06_neutral_v3_teg__cc`) | `04_effect_v3__cc` (reference + first + last frame) | two-sided 74 pairs × 2 = 148 (76 zs + 72 su) | prompt = S1 S2 | cc (same as its twin) |
| **wan_flf2v** `wan_flf2v_neutral_v3` (`gens/042_wan_flf2v/02_neutral_v3__cc`) | `01_effect_v3__cc` (first + last frame, no reference) | 148 | prompt = S1 S2 | cc |
| **wan_vace** `wan_vace_neutral_v3` (`gens/043_wan_vace/02_neutral_v3__cc`) | `01_effect_v3__cc` (6+4 given frames at 16 fps, no reference) | 148 | prompt = S1 S2 | cc |

- Each twin is generated on the SAME machine as the entry it mirrors, so neutral-vs-effect within an arm carries
  no cross-machine confound (scoring stays on one machine regardless).
- Two-sided twins follow the TEG rule: zero-shot arrays first; seen/unseen arrays only after every zero-shot
  array of every arm is complete.
- For the metrics nothing new is needed: `scripts/store_eval_common.grid_type` keys the given windows on the
  store arm (`refvfx`/`vap`/`vfxmaster`/`wan_flf2v` → 1 (+1) frame, `wan_vace` → VACE16 6 (+4)), which the
  twins share. `scripts/family_tables.py` needs the new harness arms added to its GENS/EXT maps to render them.

---

_Last updated 2026-09-21 (added §8 neutral twins of every grid-v3 external arm; 2026-09-20: §7 TEG baselines). Change here + the source workers together; this doc is
documentation, not the contract (store/README.md is the contract). Per-arm rows: `store/ARMS.md`. Ledger:
`store/INDEX.md`._
