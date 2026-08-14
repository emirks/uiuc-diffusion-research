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

_Last updated 2026-08-14. Change here + the source workers together; this doc is documentation, not the
contract (store/README.md is the contract). Per-arm rows: `store/ARMS.md`. Ledger: `store/INDEX.md`._
