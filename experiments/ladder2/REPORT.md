# ladder2 — complete campaign report

**Run:** 2026-07-22 evening → 2026-07-23 midday, UIUC Campus Cluster.
**Mode:** `/advised` — operator (Opus) executes and verifies; `fable-advisor` owns every judgment call.
**Single source of truth:** `experiments/ladder2/registry.jsonl`. Dossier: `$LAB/misc/ladder2_redesign/DOSSIER.md`. Code: branch `ladder2`.

**Completion:** training **12/12** · generation **888/888** (verified against the registry, 0 missing) · scoring **11,695 unique** (generation × pool-reference) rows + 2,211 recipient-pool rows + 420 dominance rows. **No kill rule (K0–K3) fired on any model at any point.**

**Cost:** 87.9 GPU-hours over 339 Slurm jobs — 43.4 h training, 37.3 h generation, 7.1 h scoring, 0.2 h analysis.

Contents: [1 Question](#1-what-the-campaign-asks) · [2 Design](#2-design) · [3 Training settings](#3-training-settings) · [4 Generation settings](#4-generation-settings) · [5 Eval settings](#5-evaluation-settings) · [6 Gates](#6-gates-passed-before-training) · [7 **Full results**](#7-full-results) · [8 Copy confound](#8-why-every-generalist-row-is-invalid--the-copy-confound) · [9 Amendment-1](#9-amendment-1--the-corrected-claim-bearing-readout) · [10 Finding](#10-the-mechanistic-finding-owner-gated) · [11 Bugs](#11-bugs-found-and-fixed-during-the-run) · [12 Process](#12-process-notes) · [13 Open](#13-open-items)

---

## 1. What the campaign asks

Does a transition learned from a corpus *transfer* — to new content, to a new demo of a known transition, and to transitions never trained on? And how much of that is capability versus lookup?

Two axes:

- **Reference novelty** — `seen` (exact demo trained) → `unseen` (new demo of a trained class) → `zero-shot` (held-out class). Specialists have no reference axis; their transition lives in the weights.
- **Content** — `same` / `cross` / `foreign` (DAVIS, off-distribution real footage).

Endpoints are always untrained content, strictly sidedness-matched, except two deliberate fit anchors.

**Four tiers, all compared on byte-identical inputs** (keyed by `input_key` = endpoint + prompt + sidedness + reference):

| tier | gets | rows |
|---|---|---|
| `text_floor` | prompt only, no conditioning | 12 |
| `base` | prompt + conditioning (+ the same reference where its twin has one), no adapter | 203 |
| specialist ×11 | + transition baked into weights | 77 |
| `ic_gen` | + transition supplied in-context as a demo | 152 |

444 registry rows × 2 seeds = **888 generations**.

---

## 2. Design

### 2.1 The one thing that changed vs every prior ladder

Every previous run prompted with the **full caption**, which described the outcome:

> `"ICTRANS <S1>. The scene transforms into <S2>."`

That leaks the answer. ladder2 renders leak-free prompts where a neutral token holds the transition slot:

> one-sided `"{S1}. sksz."` · two-sided `"{S1}. sksz. {S2}."`

The outcome half is dropped entirely. Training captions come from the **same** `render_prompt()` call as the registry rows, so train == inference by construction.

### 2.2 Cell inventory

| cell | n | what it tests |
|---|---|---|
| SP-fit | 11 | specialist on a *trained* endpoint — sanity anchor |
| SP-same | 22 | specialist, unseen endpoint, own class — **claim cell** |
| SP-cross | 22 | specialist, endpoint from a different class — **claim cell** |
| SP-foreign | 22 | specialist on DAVIS footage |
| G-fit | 13 | generalist, reference it trained on |
| G-memo-probe | 13 | generalist, trained reference + trained endpoint — lookup probe |
| G-unseen-same | 13 | generalist, new demo of a trained class, same content |
| G-unseen-cross | 26 | ditto, cross content — **claim cell** |
| G-unseen-foreign | 26 | ditto, DAVIS content |
| G-zs-same | 8 | generalist, **held-out class**, same content |
| G-zs-cross | 20 | ditto, cross content — **claim cell** |
| G-zs-foreign | 20 | ditto, DAVIS content |
| G-ref-control | 13 | generalist with a **mismatched** demo — does it just follow any clip? |
| text_floor | 12 | prompt-only leak-proof floor |
| base:* | 203 | one keyed twin per treatment row |

Sidedness: 336 one-sided, 108 two-sided. Priority waves: P0 216, P1 92, P2 136 (generated hardest-first).

---

## 3. Training settings

Emitted by `train/make_configs.py` — 12 configs, no hand-editing. Base model `ltx-2-19b-dev`, text encoder `gemma-3-12b-it-qat-q4_0-unquantized`, bf16 mixed precision, gradient checkpointing on, AdamW, linear schedule, batch 1 × grad-accum 1, max-grad-norm 1.0, `shifted_logit_normal` timestep sampling, seed 42.

| | 11 specialists | generalist `ic_gen` |
|---|---|---|
| LoRA rank / alpha / dropout | 32 / 32 / 0.0 | 32 / 32 / 0.0 |
| target modules | **attn** (8): `attn1/2.to_{q,k,v,out.0}` | **attn+FFN** (10): + `ff.net.0.proj`, `ff.net.2` |
| learning rate | 1e-4 | 2e-4 |
| steps | 2000 | 5000 |
| checkpoint interval | 250 | 500 |
| conditioning | prefix (tb=2) + suffix (tb=1, two-sided only) | reference latents + mask, prefix/suffix from the root |
| `cond_clean_latents_dir` | two-sided only | always |
| pinned inference checkpoint | step 2000 | step 5000 |

**Checkpoints were pinned in `arms.yaml` before any score existed** — fixed-checkpoint selection, no post-hoc best-checkpoint picking.

### 3.1 Mandatory inline validation (the `lora-train` triad)

Every config runs the full triad every **250 steps** with `skip_initial_validation: false`, 30 inference steps, 480×640×121, guidance 4.0, STG 1.0 on block 29, seed 42:

| sample | specialists | generalist |
|---|---|---|
| **ID** | train-band clip of the trained class | `shadow_smoke_1`←`shadow_smoke_0` (two-sided) and `color_rain_1`←`color_rain_2` (one-sided) |
| **OOD** | train-band clip of a *different* class (`mystification_0` one-sided / `air_bending_0` two-sided) | `saint_glow_1`→`saint_glow_0` — a **held-out class**, i.e. a live zero-shot preview every interval |
| **control** | same S1 prompt, **no token, no conditioning** | same |

The control doubles as the token's own placebo test. OOD samples are drawn from the **train band** on purpose — eval endpoints stay untouched by anything the training loop renders.

### 3.2 Resume wiring (load-bearing)

```python
"model": {..., "load_checkpoint": str(OUT_TRAIN / name / "checkpoints")},
"checkpoints": {"interval": ckpt_every, "keep_last_n": -1, "no_resume": False},
```

`no_resume: false` **alone does not resume** — `trainer._load_checkpoint()` returns immediately when `model.load_checkpoint` is null, so a requeued job silently restarts at step 0. Pointing it at the run's own checkpoint dir makes `_find_checkpoint()` pick the latest `step_*.safetensors`. On a first run the directory is empty → returns None → clean start at 0.

### 3.3 Measured training cost

| model | wall clock | partition |
|---|---|---|
| spec_color_rain | 1:00:53 | secondary |
| spec_wireframe | 1:09:30 | secondary |
| spec_illustration_scene | 1:11:13 | secondary |
| spec_super_fast_run | 1:12:01 | secondary |
| spec_gas_transformation | 1:18:16 | secondary |
| spec_polygon | 1:22:09 | secondary |
| spec_animalization | 1:26:04 | secondary |
| spec_shadow | 1:40:03 | HCESC-H100 |
| spec_portal | 1:42:35 | HCESC-H100 |
| spec_shadow_smoke | 1:44:12 | HCESC-H200 |
| spec_hero_flight | 1:45:43 | HCESC-H100 |
| **ic_gen** | 3:55:00 (TIMEOUT) **+ 2:28:41 resume** = **6:24** | secondary |

Plus `ladder2_prepare` 11:29, `ladder2_precomp` 0:50, token probe 0:02:32 + 0:50:28.

`ic_gen`'s ~6.4 h vs exp_073's ~4.8 h is **entirely validation cadence**: 21 inline validation rounds vs 1. Measured 18.3 min per 250-step round of which ~14.4 min is pure training — matching the lineage exactly. The resume across the walltime boundary picked up correctly from checkpoint 2500.

---

## 4. Generation settings

`run_gen.py` consumes registry rows for one arm. **Row × seed = exactly one video.** Nothing is decided in the generator: the row already carries the prompt, conditioning, reference and arm.

| | value |
|---|---|
| resolution | 480 × 640 × 121 frames (portrait, matching the corpus), 24 fps |
| inference steps | 30 |
| guidance scale | 4.0 |
| STG | scale 1.0, block [29], mode `stg_v` |
| negative prompt | `worst quality, inconsistent motion, distorted, jittery` |
| seeds | 42, 43 (≥2 per eval item, registered in `arms.yaml`) |
| prefix window | 9 px frames (`PX_PREFIX`) |
| suffix window | 9 px frames encoded, 8 generated (`SUFFIX_GEN_FRAMES`) |
| reference condition | `downscale_factor=1, temporal_scale_factor=1, include_in_output=False` |
| adapter load | PEFT `LoraConfig(r=32, alpha=32)`, `diffusion_model.` prefix stripped |

Conditioning is a **pure function of the row** — the same rule the eval mask uses, so train, generate and score never disagree about which frames are conditioned.

**Causal-VAE suffix fix.** The prefix is clean by causality (rel-L2 8.3e-5) but the suffix bleeds (0.28) when encoded in context. `encode_conditioning.write_cond_clean()` encodes each window in isolation and asserts bitwise; both trainer and generator read the same `cond_clean_latents_dir`.

Skip-if-exists on the output path makes every generation job requeue-safe on the preemptible `secondary` queue.

---

## 5. Evaluation settings

Instrument: **transition-eval v4** (`4.0.0-draft.1`), run from the pinned `eval-v4-cert` worktree — owner directive 2026-07-20, v4 is the lane instrument. Appearance kernel `m1a_S3`. Feature cache shared warm across all scoring jobs.

### 5.1 The yardstick

**Pool percentage.** Score each generation against every same-class corpus clip of the **GT pool** (the donor class — the class the arm was supposed to produce), average, and divide by that class's **GT ceiling** (the same-class off-diagonal mean of the v4 distance matrix). The ceiling carries the same class-spread penalty as the score, so it cancels — which is what makes % comparable *across* classes.

Pool references are **copy-guarded**: never the reference clip, never the endpoint. Deterministic first-8 by clip name.

### 5.2 The %-typing firewall

| type | when | status |
|---|---|---|
| `%_same` | endpoint class == donor class | fair, cross-class comparable, **headline-eligible** |
| `%_proxy` | cross / foreign content | content-capped — the generation can never fully look like the donor class because its *content* comes from elsewhere. Absolute level is **ranking-only**; the claim is the margin Δpp vs the base twin, where the identical cap cancels |

`report()` prints % everywhere and **refuses to aggregate across % types** (hard assert).

### 5.3 Unit of analysis

The unit is the **donor class**, not the item: per-donor mean of the margin, then a sign test. `positive` requires ≥ 80 % of donor classes positive (e.g. 9/11), else `weak`.

### 5.4 Keyed join (seatbelt 4)

Every treatment row joins its base twin by `input_key` — **exact**, never folded, never silently dropped. A treatment row with no base twin is a hard exit. *This is the defect that flipped two verdicts in the previous ladder.*

### 5.5 Dedup

Scoring ran as several incremental passes under distinct labels. A row planned twice — a generation landing between one pass's plan and its score — is written to both passes' `items.jsonl`. **798 of 12,500 written rows are such repeats**; counting them twice reweights that generation's pool mean. `report()` and `report_full.py` both dedup on the eval id. Effect on every headline number: ≤ 0.3 pp — no verdict changes, but the tables below are the deduped ones.

---

## 6. Gates passed before training

| gate | bar | result |
|---|---|---|
| token `sksz` inert on base | effect/noise ≤ 1.0 | **0.21** |
| causal-VAE suffix bleed fix | reproduce exp_073 | rel-L2 **0.284** vs 0.280 |
| base accepts *and uses* an IC reference | > seed noise | **1.48×** |
| root assembly | 12 roots, equal source counts | 0 path mismatches |
| no embedding reuse | must differ from old leaky embeddings | rel-L2 **0.966** |
| leak audit | 0 leaks | **0 / 83** endpoints |
| **new-variable gate** | effect visible by step 1000 | **passed at step 250** — 4× early |

The new-variable gate: identical prompt, prefix and seed — the untrained base shows no effect; after 250 LoRA steps the class effect is clearly present. The neutral token carried the trigger.

---

## 7. Full results

All numbers below are regenerated by `experiments/ladder2/report_full.py` (deduped). Base rows carry their own cell labels (`base:<cell>`) because a single base twin can serve more than one treatment cell.

### 7.1 Headline — the donor-pool yardstick

| cell | n | %type | level | Δpp vs base | donors + | verdict |
|---|---|---|---|---|---|---|
| **SP-same** | 22 | same | **99.7 %** | **+40.0** | **11/11** | **CLAIM PASSES** |
| **SP-cross** | 22 | proxy | (94.9 %) | **+39.2** | **11/11** | **CLAIM PASSES** |
| SP-fit | 11 | same | 100.4 % | +39.5 | 11/11 | sanity anchor |
| SP-foreign | 22 | proxy | (63.1 %) | +18.9 | 8/11 | weak (below 9/11 bar) |
| text_floor | 12 | same | 67.4 % | — | — | leak-proof floor |
| G-fit | 13 | same | 86.3 % | −12.0 | 4/13 | ⚠ invalid |
| G-memo-probe | 13 | same | 83.1 % | −15.3 | 1/13 | ⚠ invalid |
| G-unseen-same | 13 | same | 88.7 % | −11.2 | 2/13 | ⚠ invalid |
| G-unseen-cross | 26 | proxy | (72.9 %) | −27.2 | 0/13 | ⚠ invalid |
| G-zs-same | 8 | same | 90.8 % | −14.1 | 1/8 | ⚠ invalid |
| G-zs-cross | 20 | proxy | (72.8 %) | −30.3 | 0/10 | ⚠ invalid |
| G-ref-control | 13 | same | 69.0 % | +4.1 | 10/13 | ⚠ invalid |
| G-unseen-foreign | 26 | proxy | (56.8 %) | −42.1 | 0/13 | ⚠ invalid |
| G-zs-foreign | 20 | proxy | (44.3 %) | −55.2 | 0/10 | ⚠ invalid |

**Specialist reading.** A specialist reaches its class ceiling on unseen endpoints of its own class (**99.7 %**) and holds ~**95 %** transferring to a different class's endpoints — both ~**+40 pp** over base on identical inputs, **11/11** donor classes. `SP-fit ≈ SP-same` means there is no train-vs-test endpoint gap. DAVIS is positive on average (+18.9 pp) but only 8/11 donors, so it misses its sign-test bar and is reported **weak**.

**Every `G-*` row is marked invalid** — see §8. The numbers are retained, never deleted, because they were pre-registered.

### 7.2 Raw · ceiling · % (the reporting rule)

| cell | arm | n | raw `app_ref` | GT ceiling | pool-% | %type |
|---|---|---|---|---|---|---|
| G-fit | ic_gen | 13 | 0.7521 | 0.8735 | 86.3 % | same |
| G-memo-probe | ic_gen | 13 | 0.7187 | 0.8735 | 83.1 % | same |
| G-ref-control | ic_gen | 13 | 0.6019 | 0.8735 | 69.0 % | same |
| G-unseen-cross | ic_gen | 26 | 0.6302 | 0.8735 | 72.9 % | proxy |
| G-unseen-foreign | ic_gen | 26 | 0.4841 | 0.8735 | 56.8 % | proxy |
| G-unseen-same | ic_gen | 13 | 0.7711 | 0.8735 | 88.7 % | same |
| G-zs-cross | ic_gen | 20 | 0.6086 | 0.8722 | 72.8 % | proxy |
| G-zs-foreign | ic_gen | 20 | 0.3838 | 0.8722 | 44.3 % | proxy |
| G-zs-same | ic_gen | 8 | 0.7523 | 0.8635 | 90.8 % | same |
| SP-cross | specialist | 22 | 0.8192 | 0.8627 | 94.9 % | proxy |
| SP-fit | specialist | 11 | 0.8685 | 0.8627 | 100.4 % | same |
| SP-foreign | specialist | 22 | 0.5483 | 0.8627 | 63.1 % | proxy |
| SP-same | specialist | 22 | 0.8560 | 0.8627 | 99.7 % | same |
| base:G-fit | base | 13 | 0.8616 | 0.8735 | 98.3 % | same |
| base:G-memo-probe | base | 13 | 0.8623 | 0.8735 | 98.4 % | same |
| base:G-ref-control | base | 13 | 0.5674 | 0.8735 | 64.8 % | same |
| base:G-unseen-cross | base | 26 | 0.8651 | 0.8735 | 100.1 % | proxy |
| base:G-unseen-foreign | base | 26 | 0.8552 | 0.8735 | 98.9 % | proxy |
| base:G-unseen-same | base | 13 | 0.8635 | 0.8735 | 99.9 % | same |
| base:G-zs-cross | base | 20 | 0.8769 | 0.8722 | 103.1 % | proxy |
| base:G-zs-foreign | base | 20 | 0.8402 | 0.8722 | 99.5 % | proxy |
| base:G-zs-same | base | 8 | 0.8792 | 0.8635 | 104.9 % | same |
| base:SP-cross | base | 16 | 0.4852 | 0.9130 | 53.8 % | proxy |
| base:SP-fit | base | 11 | 0.5277 | 0.8627 | 60.9 % | same |
| base:SP-foreign | base | 6 | 0.4145 | 0.9127 | 45.3 % | proxy |
| base:SP-same | base | 18 | 0.5287 | 0.8570 | 62.1 % | same |
| text_floor | text_floor | 12 | 0.5806 | 0.8717 | 67.4 % | same |

**The tell is already here.** `base` in the reference-bearing cells scores **98–105 % of ceiling** — a *raw appearance match to the donor class at or above what two real clips of that class score against each other*. On DAVIS content it scores 98.9 % and 99.5 %. That is not a model doing a transition on foreign footage; that is a model reproducing the demo. §8 confirms it.

### 7.3 Reference-relative metrics (mean over the same-class GT pool)

`app_ref` ↑ (M1a) · `cam_zpr` ↓ (M1b) · `obj_csls` ↓ (M1c) · `copy_max` ↓ (M2a). `cam_dtw`/`cam_corr`/`obj_match`/`app_ref_v3` are analysis fields.

| cell | arm | n | M1a app_ref | M1b cam_zpr | M1c obj_csls | M2a copy_max | cam_dtw | cam_corr | obj_match | app_ref_v3 | cross |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G-fit | ic_gen | 13 | 0.752 | 0.396 | 0.151 | 0.316 | 1.182 | 0.034 | 0.120 | 0.260 | 0.395 |
| G-memo-probe | ic_gen | 13 | 0.719 | 0.348 | 0.155 | 0.316 | 1.183 | 0.029 | 0.132 | 0.232 | 0.491 |
| G-ref-control | ic_gen | 13 | 0.602 | 0.381 | 0.174 | 0.264 | 1.204 | 0.035 | 0.107 | 0.187 | 0.555 |
| G-unseen-cross | ic_gen | 26 | 0.630 | 0.452 | 0.177 | 0.244 | 1.186 | 0.039 | 0.112 | 0.186 | 0.489 |
| G-unseen-foreign | ic_gen | 26 | 0.484 | 0.617 | 0.186 | 0.150 | 1.225 | −0.018 | 0.112 | 0.111 | 0.563 |
| G-unseen-same | ic_gen | 13 | 0.771 | 0.362 | 0.156 | 0.317 | 1.193 | 0.040 | 0.131 | 0.254 | 0.577 |
| G-zs-cross | ic_gen | 20 | 0.609 | 0.419 | 0.158 | 0.291 | 1.141 | 0.058 | 0.115 | 0.206 | 0.382 |
| G-zs-foreign | ic_gen | 20 | 0.384 | 0.620 | 0.185 | 0.140 | 1.212 | −0.035 | 0.113 | 0.096 | 0.389 |
| G-zs-same | ic_gen | 8 | 0.752 | 0.377 | 0.174 | 0.314 | 1.129 | 0.018 | 0.084 | 0.243 | 0.397 |
| SP-cross | specialist | 22 | 0.819 | 0.381 | 0.172 | 0.326 | 1.165 | 0.039 | 0.099 | 0.284 | 0.416 |
| SP-fit | specialist | 11 | 0.868 | 0.288 | 0.127 | 0.384 | 1.128 | 0.063 | 0.141 | 0.338 | 0.570 |
| SP-foreign | specialist | 22 | 0.548 | 0.572 | 0.173 | 0.183 | 1.224 | 0.002 | 0.108 | 0.161 | 0.480 |
| SP-same | specialist | 22 | 0.856 | 0.292 | 0.131 | 0.357 | 1.139 | 0.068 | 0.118 | 0.318 | 0.425 |
| base:G-fit | base | 13 | 0.862 | 0.312 | 0.137 | 0.400 | 1.129 | 0.050 | 0.122 | 0.331 | 0.143 |
| base:G-memo-probe | base | 13 | 0.862 | 0.324 | 0.144 | 0.393 | 1.145 | 0.046 | 0.129 | 0.326 | 0.107 |
| base:G-ref-control | base | 13 | 0.567 | 0.431 | 0.193 | 0.212 | 1.204 | −0.011 | 0.081 | 0.143 | 0.133 |
| base:G-unseen-cross | base | 26 | 0.865 | 0.326 | 0.147 | 0.370 | 1.159 | 0.028 | 0.108 | 0.297 | 0.103 |
| base:G-unseen-foreign | base | 26 | 0.855 | 0.338 | 0.148 | 0.363 | 1.147 | 0.019 | 0.112 | 0.292 | 0.114 |
| base:G-unseen-same | base | 13 | 0.864 | 0.342 | 0.146 | 0.390 | 1.185 | 0.001 | 0.113 | 0.309 | 0.140 |
| base:G-zs-cross | base | 20 | 0.877 | 0.251 | 0.122 | 0.445 | 1.010 | 0.189 | 0.121 | 0.372 | 0.155 |
| base:G-zs-foreign | base | 20 | 0.840 | 0.278 | 0.124 | 0.428 | 1.039 | 0.161 | 0.118 | 0.352 | 0.119 |
| base:G-zs-same | base | 8 | 0.879 | 0.264 | 0.140 | 0.459 | 1.047 | 0.164 | 0.088 | 0.371 | 0.135 |
| base:SP-cross | base | 16 | 0.485 | 0.406 | 0.171 | 0.208 | 1.162 | 0.028 | 0.102 | 0.149 | 0.624 |
| base:SP-fit | base | 11 | 0.528 | 0.390 | 0.155 | 0.297 | 1.203 | −0.023 | 0.115 | 0.203 | 0.663 |
| base:SP-foreign | base | 6 | 0.415 | 0.591 | 0.184 | 0.135 | 1.206 | −0.002 | 0.111 | 0.107 | 0.421 |
| base:SP-same | base | 18 | 0.529 | 0.375 | 0.155 | 0.260 | 1.178 | −0.006 | 0.094 | 0.165 | 0.536 |
| text_floor | text_floor | 12 | 0.581 | 0.415 | 0.165 | 0.269 | 1.190 | 0.031 | 0.081 | 0.181 | 0.911 |

Specialists move **camera** and **object** motion toward the donor class too, not just appearance: `cam_zpr` 0.375 → 0.292 and `obj_csls` 0.155 → 0.131 on SP-same vs its base twin (both ↓ = better). So the specialist claim is not appearance-only.

### 7.4 Per-generation metrics — where the confound becomes visible

`margin` ↑ (M2b intrusion — positive = the intended class wins the corpus-wide retrieval) · `prefix_dino` ↑ / `prefix_lpips` ↓ (M3a endpoint fidelity vs the conditioning clip) · `max_seam_z` ↓ (M3b handoff flag; z ≳ 3 = a snap).

| cell | arm | n | M2b margin | app_target | M3a pre_dino | M3a pre_lpips | **M3b seam_z** | depth | depart | arrive | core_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G-fit | ic_gen | 13 | 0.135 | 0.453 | 0.824 | 0.1352 | 3.62 | 0.674 | 0.343 | 0.568 | 0.204 |
| G-memo-probe | ic_gen | 13 | 0.141 | 0.482 | 0.987 | 0.0126 | −0.13 | 0.674 | 0.420 | 0.589 | 0.125 |
| G-ref-control | ic_gen | 13 | 0.042 | 0.396 | 0.984 | 0.0157 | 0.26 | 0.713 | 0.336 | 0.566 | 0.251 |
| G-unseen-cross | ic_gen | 26 | −0.183 | 0.238 | 0.966 | 0.0320 | 1.64 | 0.712 | 0.383 | 0.609 | 0.190 |
| G-unseen-foreign | ic_gen | 26 | −0.088 | 0.151 | 0.959 | 0.1025 | 0.99 | 0.767 | 0.282 | 0.693 | 0.331 |
| G-unseen-same | ic_gen | 13 | 0.134 | 0.472 | 0.983 | 0.0157 | 1.21 | 0.783 | 0.296 | 0.600 | 0.262 |
| G-zs-cross | ic_gen | 20 | −0.171 | 0.239 | 0.984 | 0.0131 | −0.09 | 0.774 | 0.330 | 0.603 | 0.256 |
| G-zs-foreign | ic_gen | 20 | −0.098 | 0.123 | 0.964 | 0.0887 | 0.99 | 0.732 | 0.310 | 0.586 | 0.237 |
| G-zs-same | ic_gen | 8 | 0.041 | 0.431 | 0.980 | 0.0175 | −0.02 | 0.860 | 0.326 | 0.640 | 0.315 |
| SP-cross | specialist | 22 | −0.019 | 0.363 | 0.987 | 0.0138 | 0.20 | 0.733 | 0.364 | 0.590 | 0.260 |
| SP-fit | specialist | 11 | 0.270 | 0.621 | 0.987 | 0.0163 | 0.02 | 0.885 | 0.259 | 0.705 | 0.456 |
| SP-foreign | specialist | 22 | −0.012 | 0.214 | 0.966 | 0.0931 | 1.26 | 0.820 | 0.294 | 0.633 | 0.311 |
| SP-same | specialist | 22 | 0.240 | 0.550 | 0.984 | 0.0137 | 0.46 | 0.712 | 0.385 | 0.574 | 0.196 |
| base:G-fit | base | 13 | 0.254 | 0.637 | 0.887 | 0.0930 | **23.44** | 0.733 | 0.075 | 0.379 | 0.313 |
| base:G-memo-probe | base | 13 | 0.279 | 0.651 | 0.932 | 0.0530 | **25.50** | 0.710 | 0.076 | 0.399 | 0.330 |
| base:G-ref-control | base | 13 | −0.436 | 0.189 | 0.920 | 0.0628 | **31.52** | 0.738 | 0.072 | 0.469 | 0.408 |
| base:G-unseen-cross | base | 26 | 0.277 | 0.612 | 0.901 | 0.0826 | **29.65** | 0.739 | 0.075 | 0.451 | 0.384 |
| base:G-unseen-foreign | base | 26 | 0.237 | 0.583 | 0.727 | 0.2448 | **12.88** | 0.700 | 0.062 | 0.466 | 0.424 |
| base:G-unseen-same | base | 13 | 0.273 | 0.607 | 0.892 | 0.0870 | **45.22** | 0.756 | 0.091 | 0.471 | 0.409 |
| base:G-zs-cross | base | 20 | 0.322 | 0.683 | 0.923 | 0.0553 | **22.93** | 0.876 | 0.075 | 0.632 | 0.649 |
| base:G-zs-foreign | base | 20 | 0.284 | 0.653 | 0.785 | 0.1997 | **10.45** | 0.819 | 0.064 | 0.603 | 0.619 |
| base:G-zs-same | base | 8 | 0.320 | 0.690 | 0.930 | 0.0616 | **20.91** | 0.865 | 0.086 | 0.625 | 0.629 |
| base:SP-cross | base | 16 | −0.236 | 0.195 | 0.959 | 0.0384 | 2.66 | 0.754 | 0.364 | 0.625 | 0.226 |
| base:SP-fit | base | 11 | 0.113 | 0.426 | 0.987 | 0.0165 | 1.92 | 0.813 | 0.423 | 0.642 | 0.212 |
| base:SP-foreign | base | 6 | −0.086 | 0.145 | 0.964 | 0.0906 | 0.53 | 0.814 | 0.371 | 0.628 | 0.250 |
| base:SP-same | base | 18 | 0.057 | 0.346 | 0.981 | 0.0143 | 15.39 | 0.746 | 0.328 | 0.571 | 0.198 |
| text_floor | text_floor | 12 | 0.011 | 0.314 | — | — | 2.83 | 0.862 | 0.233 | 0.709 | 0.483 |

**Three independent signals, all pointing the same way.** M3a and M3b are *not* built on the appearance kernel and know nothing about the donor class, yet within matched conditioning:

1. **`max_seam_z`** — base in **every** reference-bearing cell: **10.5 – 45.2**. `ic_gen` on the identical inputs: **−0.13 – 3.62**. A z of 20–45 means the frame-to-frame distance at the handoff index spikes 20–45 MADs. The generation does not continue from the conditioned prefix — it cuts away from it.
2. **`prefix_dino` / `prefix_lpips`** — base 0.727–0.932 DINO with LPIPS up to 0.245; `ic_gen` 0.959–0.987 with LPIPS 0.013–0.135. The adapter *keeps the endpoint*; the bare base does not.
3. **`scalar_depart`** — base 0.062–0.091 vs treatments 0.26–0.42. Base leaves the start state essentially immediately.

M3b is valid only within matched conditioning modes — and it is matched here, because base and its twin share `input_key` byte-for-byte. Note the contrast with the *non*-reference cells: `base:SP-cross` 2.66, `base:SP-foreign` 0.53, `base:SP-fit` 1.92. Give base a reference and it snaps; take the reference away and it mostly doesn't.

### 7.5 Flag rates (fraction of scored generations)

| cell | arm | n | near_copy | cross_high | app_sat | core_degen | intruder |
|---|---|---|---|---|---|---|---|
| G-fit | ic_gen | 13 | 0.000 | 0.077 | 0.000 | 0.038 | 0.192 |
| G-memo-probe | ic_gen | 13 | 0.000 | 0.115 | 0.000 | 0.000 | 0.077 |
| G-ref-control | ic_gen | 13 | 0.000 | 0.077 | 0.000 | 0.000 | 0.423 |
| G-unseen-cross | ic_gen | 26 | 0.000 | 0.058 | 0.000 | 0.038 | 0.885 |
| G-unseen-foreign | ic_gen | 26 | 0.000 | 0.231 | 0.002 | 0.058 | 0.904 |
| G-unseen-same | ic_gen | 13 | 0.000 | 0.115 | 0.000 | 0.038 | 0.115 |
| G-zs-cross | ic_gen | 20 | 0.000 | 0.025 | 0.008 | 0.075 | 0.850 |
| G-zs-foreign | ic_gen | 20 | 0.000 | 0.150 | 0.008 | 0.150 | 0.900 |
| G-zs-same | ic_gen | 8 | 0.000 | 0.062 | 0.000 | 0.000 | 0.375 |
| SP-cross | specialist | 22 | 0.003 | 0.091 | 0.000 | 0.000 | 0.545 |
| SP-fit | specialist | 11 | 0.035 | 0.000 | 0.000 | 0.000 | 0.000 |
| SP-foreign | specialist | 22 | 0.000 | 0.227 | 0.000 | 0.000 | 0.659 |
| SP-same | specialist | 22 | 0.017 | 0.045 | 0.000 | 0.000 | **0.000** |
| base:G-fit | base | 13 | 0.031 | 0.038 | 0.000 | 0.000 | 0.038 |
| base:G-memo-probe | base | 13 | 0.031 | 0.000 | 0.000 | 0.000 | 0.000 |
| base:G-ref-control | base | 13 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| base:G-unseen-cross | base | 26 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 |
| base:G-unseen-foreign | base | 26 | 0.000 | 0.000 | 0.000 | 0.000 | 0.088 |
| base:G-unseen-same | base | 13 | 0.000 | 0.000 | 0.000 | 0.000 | 0.038 |
| base:G-zs-cross | base | 20 | 0.054 | 0.000 | 0.000 | 0.000 | 0.000 |
| base:G-zs-foreign | base | 20 | 0.046 | 0.000 | 0.000 | 0.025 | 0.050 |
| base:G-zs-same | base | 8 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 |
| base:SP-cross | base | 16 | 0.000 | 0.406 | 0.000 | 0.188 | 0.938 |
| base:SP-fit | base | 11 | 0.034 | 0.318 | 0.000 | 0.136 | 0.182 |
| base:SP-foreign | base | 6 | 0.000 | 0.167 | 0.000 | 0.250 | 0.917 |
| base:SP-same | base | 18 | 0.000 | 0.306 | 0.000 | 0.056 | 0.333 |
| text_floor | text_floor | 12 | 0.010 | 0.833 | 0.000 | 0.042 | 0.333 |

Two things to read here:

- **`near_copy` never fires on the copying arm.** Base's reference-bearing cells flag at **0.000–0.054** while Pass A shows 88–100 % of those same generations are reference-dominated. τ_copy = 0.858 is calibrated for verbatim *frame* copies; scene reproduction at cos ≈ 0.5 sails under it. This is a documented limitation of M2a, not a scoring error — and it is exactly why Amendment-1 was needed.
- **`intruder` on SP-same and SP-fit is 0.000** — across 33 specialist generations on their own class, the intended class won the corpus-wide retrieval every time. `cross_high` at 0.833 for `text_floor` is expected: with no conditioning the two endpoints are unconstrained.

### 7.6 Per-arm rollup

| arm | items | raw app_ref | M2a copy_max | near_copy | M3a pre_dino | M3b seam_z |
|---|---|---|---|---|---|---|
| base | 203 | 0.7522 | 0.347 | 0.018 | 0.890 | **19.30** |
| ic_gen | 152 | 0.6040 | 0.244 | 0.000 | 0.960 | 0.99 |
| spec_animalization | 7 | 0.8533 | 0.364 | 0.063 | 0.978 | 0.96 |
| spec_color_rain | 7 | 0.6828 | 0.173 | 0.000 | 0.987 | 0.48 |
| spec_gas_transformation | 7 | 0.7142 | 0.208 | 0.019 | 0.977 | 0.35 |
| spec_hero_flight | 7 | 0.7647 | 0.311 | 0.000 | 0.983 | −0.19 |
| spec_illustration_scene | 7 | 0.6298 | 0.286 | 0.018 | 0.978 | 0.50 |
| spec_polygon | 7 | 0.6992 | 0.254 | 0.018 | 0.975 | 2.25 |
| spec_portal | 7 | 0.8728 | 0.423 | 0.000 | 0.978 | 0.11 |
| spec_shadow | 7 | 0.5502 | 0.214 | 0.000 | 0.981 | 1.57 |
| spec_shadow_smoke | 7 | 0.9689 | 0.377 | 0.000 | 0.978 | −0.42 |
| spec_super_fast_run | 7 | 0.9665 | 0.480 | 0.000 | 0.981 | 0.11 |
| spec_wireframe | 7 | 0.6503 | 0.238 | 0.000 | 0.984 | 0.35 |
| text_floor | 12 | 0.5806 | 0.269 | 0.010 | — | 2.83 |

Per-specialist raw `app_ref` spans 0.55 (`spec_shadow`) to 0.97 (`spec_shadow_smoke`, `spec_super_fast_run`) — but this column mixes cells with different % types and different ceilings, so it ranks *classes*, not models. The class-normalised view is §7.1, where all eleven are positive.

*11,695 unique scored (generation × pool-reference) rows over 1,764 (generation × seed) pairs, 444 registry items.*

---

## 8. Why every generalist row is invalid — the copy confound

The generalist appeared catastrophic (−11 to −55 pp, near-zero donors positive). Before reporting that, one pair was inspected directly (held-out donor `cotton_cloud`, demo `cotton_cloud_1`, recipient endpoint `animalization_0`):

- **`ic_gen` did the task** — kept the actual endpoint (woman, red bomber jacket, blue backdrop) and bloomed pink cotton-cloud material around her: donor *manner* on recipient *content*.
- **`base`, same input, no adapter, abandoned the endpoint** and reproduced the demo's own scene (a man on a couch) almost verbatim.

Since pool-% measures resemblance to the donor class, **base is rewarded for copying the demo and `ic_gen` is punished for honouring the endpoint.**

### Pass A — clip-level dominance (420 rows, both arms, every reference-bearing cell)

`ep_align` / `ref_align` = mean over generation *middle* frames of the best match to the endpoint clip / to the reference's non-core frames. Mean-of-best-match, not max-of-max — which is exactly why M2a's absolute `near_copy` flag never fired at cos ≈ 0.5.

| cell | arm | n | ep_align | ref_align | ref_dominated |
|---|---|---|---|---|---|
| G-fit | base | 26 | 0.292 | 0.838 | **96 %** |
| G-fit | ic_gen | 22 | 0.662 | 0.212 | 9 % |
| G-memo-probe | base | 26 | 0.321 | 0.864 | **100 %** |
| G-memo-probe | ic_gen | 26 | 0.837 | 0.241 | 0 % |
| G-ref-control | base | 26 | 0.143 | 0.900 | **100 %** |
| G-ref-control | ic_gen | 26 | 0.786 | 0.160 | 0 % |
| G-unseen-cross | base | 52 | 0.141 | 0.866 | **98 %** |
| G-unseen-cross | ic_gen | 52 | 0.762 | 0.156 | 2 % |
| G-unseen-same | base | 26 | 0.260 | 0.852 | **100 %** |
| G-unseen-same | ic_gen | 26 | 0.819 | 0.207 | 0 % |
| G-zs-cross | base | 40 | 0.171 | 0.857 | **100 %** |
| G-zs-cross | ic_gen | 40 | 0.763 | 0.167 | 0 % |
| G-zs-same | base | 16 | 0.408 | 0.846 | **88 %** |
| G-zs-same | ic_gen | 16 | 0.773 | 0.243 | 12 % |

A complete inversion, systematic across 7 cells: base's middle frames are made of the **demo**; `ic_gen`'s are made of the **endpoint**. §7.4's seam and endpoint-fidelity metrics corroborate this from a completely different substrate.

**`G-ref-control` is the cleanest single case.** Its demo is *deliberately mismatched* — the wrong transition. Base still reproduces it (100 % ref-dominated, intruder rate 1.000, meaning the wrong class won retrieval on every item). `ic_gen` ignores it and keeps the endpoint (0 % ref-dominated). The adapter is not merely "using" the reference; it is treating it as an instruction it can decline.

**Specialists are structurally immune** — verified **0 of 77** specialist base twins carry a reference, so there is nothing to copy. That is why their margins are clean and why §7.1's specialist verdicts stand unamended.

---

## 9. Amendment-1 — the corrected, claim-bearing readout

Formulas, thresholds and headline wording were **locked in the dossier before any recipient-pool score existed**.

- **T** (donor manner arrived) = `clip01( (D% − D_ep%) / (D_ref% − D_ep%) )`
- **C** (endpoint content kept) = `clip01( (R% − R_ref%) / (R_ep% − R_ref%) )`
- **TI = min(T, C)** — min, not mean: transfer is a conjunction, and averaging lets copying buy back score
- 2×2 quadrants at (0.5, 0.5); anchor denominators < 5 pp → `anchor_degenerate`, excluded

| cell | arm | n | T | C | **TI** | quadrants |
|---|---|---|---|---|---|---|
| G-unseen-cross | **ic_gen** | 25 | 0.393 | 0.332 | **0.224** | mush 11, ref-won 7, endpoint-won 5, transfer 2 |
| G-unseen-cross | base | 25 | 0.847 | 0.199 | 0.185 | **ref-won 21**, transfer 2, mush 2 |
| | | | | | **ΔTI +3.9 pp** | **donors positive 9/13** ✅ |
| G-zs-cross | **ic_gen** | 20 | 0.360 | 0.291 | **0.156** | mush 9, ref-won 6, endpoint-won 4, transfer 1 |
| G-zs-cross | base | 20 | 0.898 | 0.261 | 0.239 | **ref-won 18**, mush 1, transfer 1 |
| | | | | | **ΔTI −8.3 pp** | donors positive 1/10 ❌ |

**The result, both directions:**

> In-context transfer **works within the trained transition vocabulary** — with a new demo of a class the model trained on, the IC-LoRA beats its base twin (**+3.9 pp, 9/13 donors**). It **does not generalise to genuinely novel transitions** at this budget — with a held-out class it loses (**−8.3 pp, 1/10 donors**).

Both verdicts stand. The instrument was fixed before the numbers existed, precisely so the negative one could not be explained away.

**Caveat on magnitude.** `ic_gen`'s absolute TI is low everywhere (0.156–0.224), with most items in "mush" — neither strongly donor-flavoured nor strongly endpoint-preserving. Base's high T (0.85–0.90) is bought entirely by copying, which is why the conjunction (min) and not the mean is the right combiner: base's C collapses to 0.20–0.26 and takes TI with it.

---

## 10. The mechanistic finding (owner-gated)

> **Without the adapter, LTX-2 treats an in-context clip as *content to continue*. The IC-LoRA converts it into *an instruction to imitate*.**

Quantified on byte-identical inputs, four independent measurements:

| measurement | base | ic_gen |
|---|---|---|
| `ref_align` (Pass A) | 0.86 | 0.19 |
| `ep_align` (Pass A) | 0.22 | 0.78 |
| reference-dominated rate | 88–100 % | 0–12 % |
| `max_seam_z` (M3b) | 10.5–45.2 | −0.13–3.62 |
| `prefix_dino` (M3a) | 0.727–0.932 | 0.959–0.987 |

420 dominance rows over 7 cells, plus the full M3 panel on all 355 reference-bearing generations. The advisor called this "arguably the campaign's best" finding. It is a statement about what the adapter *does*, not about how well it scores.

Proposed as an F-block; **not written to `docs/FINDINGS.md`** — that file is owner-gated.

---

## 11. Bugs found and fixed during the run

Each was caught before it corrupted a reported result.

| # | bug | consequence if missed |
|---|---|---|
| 1 | clip→class derived by string-splitting | `action_run_setonfire_6` → class `run_set_on_fire`, `flame_transition_0` → `flame`; silently mislabelled rows |
| 2 | `item_id` collision in text_floor (overlapping class pools) | seatbelt #1 caught it at build time |
| 3 | train-band endpoint rule was per-root, needed per-**arm** | held-out classes' train clips are untrained content for every arm |
| 4 | `process_captions.py` defaults `media_column` to `media_path` | chained prepare job would have died |
| 5 | YAML flow-style comma truncated a DAVIS caption mid-sentence | silent prompt corruption |
| 6 | `--export` is comma-separated → `CELLS` truncated to its first element | base generation silently covered 11 of 50 rows |
| 7 | **`no_resume:false` does not enable resume** — `load_checkpoint` must point at the ckpt dir | 5 running jobs would have restarted from step 0 |
| 8 | that fix then crashed first runs (`_find_checkpoint` *raises* on a missing path) | `ic_gen` failed 7 min in; fixed by pre-creating the dir |
| 9 | scoring label reused per pass → **overwrote** prior `items.jsonl` | destroyed ~3000 scored rows (recovered via incremental re-plan) |
| 10 | `set -eo pipefail` before `source ~/.bashrc` | 4 scoring chunks died in 7 s with an empty log |
| 11 | stale `eval_c*.json` chunks re-scored | wasted GPU on already-scored rows |
| 12 | `car-turn` in the DAVIS roster has **no subject** in the portrait crop | it's filmed *from* the car; replaced by `hike` |
| 13 | smoke assert judged sidedness per root | would have killed `ic_gen` at startup (its tree mixes both) |
| 14 | **798 duplicate scored rows** across incremental passes were double-counted in the pool mean | ≤0.3 pp on every cell; fixed in `run_eval.report()` and `report_full.py` |

---

## 12. Process notes

- **Hardest-first generation** (owner call) — zero-shot and reference-control were generated before the easier cells, so the most valuable results landed first.
- **DAVIS expanded** from a 16-item token gate to 68 items (owner call). This *changed a conclusion*: at n=5 SP-foreign read 4/5 donors positive; at full n=22 it is **8/11 — below bar**. The gate-sized sample would have been reported as a clean positive.
- **Priority inversion, 3×** — our own bulk generation out-competed critical-path training on `secondary`. Standing rule adopted: whenever `ic_gen` is not running, base/text_floor generation stays held.
- **Amendment discipline** — two pre-registration amendments, both recorded as amendments with reasoning, neither swapped quietly. The 4500-vs-5000 inline eyeball was replaced by a scored checkpoint diagnostic; the donor-pool margin was replaced by the transfer index.
- **One false alarm, corrected** — 0 % GPU utilisation was read as a stall; it was a host-to-device weight load (memory climbing 29.5 → 39.7 GB).
- **8 orphan videos** on disk from a pre-`hike` registry build are not in `registry.jsonl` and therefore were never scored. Harmless; listed here so a future reader is not confused by the file count (896 files vs 888 registry generations).

---

## 13. Open items

1. **Convergence diagnostic** (ckpt-4500 vs ckpt-5000 on G-unseen-same + G-zs-cross) — **running now**, jobs 9644216/9644217, ~34 generations. Pre-declared bar: **UNDERSHOT** if Δ ≥ +2.0 pp **and** ≥ 2/3 of items improve → authorises a labelled follow-up run and flags ic_gen-vs-specialist as budget-confounded; **CONVERGED** if |Δ| < 2.0 pp or non-systematic. This decides whether §9's zero-shot loss is a real limit or a budget artifact.
2. **F-block proposal** for §10 — awaiting owner approval; `docs/FINDINGS.md` is owner-gated.
3. **2AFC** on the claim cells, if wanted — must use a content-aware question ("which shows *this transition* applied to *this scene*"), never "which looks more like class X", which inherits the same defect this campaign had to amend around.

---

## Reproducing every number here

```bash
# headline table (§7.1)
python experiments/ladder2/run_eval.py --mode report

# every metric, all cells x arms (§7.2-7.6)
python experiments/ladder2/report_full.py

# copy confound (§8) and transfer index (§9)
python experiments/ladder2/dominance.py --mode reportA
python experiments/ladder2/dominance.py --mode reportB
```
