# CTT v2 training dataset — DATASET SPEC

**Design version: `ctt-v2-dataset/0.9.0-DRAFT`**
**Status: `NOT STAMPABLE` — the caption lane is blocked (Gemini prepayment credits depleted), so
no `conditions/` tree exists, no root has been assembled, and no assert has been executed against a
real root.** See §1.2 for the exact blocking set.

This file is authoritative for the dataset. Where a script, README, dossier paragraph or advisor
ruling disagrees with this document, **this document wins and the other thing is a bug** — with one
exception, recorded here because it is load-bearing: where *this document* disagrees with **the
disk**, the disk wins and §11 records the correction. Every number below was counted first-hand on
2026-07-28; none was copied from prose.

**Governing authority chain** (each supersedes the one before it where they collide):

| # | document | governs |
|---|---|---|
| 1 | `misc/ctt_v2_final/advisors/A1b_dataset_design_FINAL_VERBATIM.md` | grid, counts, endpoints, pairing, holdouts |
| 2 | `misc/ctt_v2_final/advisors/A5_SYNTHESIS_RULING_VERBATIM.md` | the campaign — mix, gates, captions, asserts, the stamp |
| 3 | `misc/ctt_v2_final/advisors/A7_copy_gate_bars_VERBATIM.md` | copy-gate amendment-2 |
| 4 | `misc/ctt_v2_final/advisors/A8_gate8_captions_VERBATIM.md` | caption gate #8 re-pin |
| 5 | `misc/ctt_v2_final/advisors/A9_s4_final_VERBATIM.md` | **S4 reinstated; final mix weights** (reverses A5 Ruling 2) |
| 6 | `misc/ctt_v2_final/advisors/A10_full_occlusion_VERBATIM.md` | full-occlusion shader family |
| — | `misc/ctt_v2_final/DOSSIER.md` | the operative campaign record; §1 ground truth, §4 caption bars |
| — | `misc/ctt_v2_final/REF_root_format.md` | the on-disk root contract |
| — | `eval_ladder/SPEC.md` | the eval ontology this dataset is measured under; **frozen, not ours to change** |

---

## 1. Status

### 1.1 FROZEN vs PENDING

`FROZEN` = the artefact exists on disk, its count has been verified, and changing it now is an
amendment. `PENDING` = it does not exist yet. `PINNED` = a number or rule fixed before the data it
judges existed (pre-registration), not yet exercised.

| item | state | evidence |
|---|---|---|
| S0 root (139 clips / 385 pairs / 26 classes) | **FROZEN** | counted, §5.1 |
| S0 captions (139, byte-identical, never re-written) | **FROZEN** | `conditions_token.json` = `{"token":"sksz","n":139}` |
| S2a render (7,990 clips / 799 ops) | **FROZEN** | `S2_ACCEPTANCE.json` verdict PASS, §5.4 |
| S2b render (7,990 clips / 799 ops) | **FROZEN** | `S2_ACCEPTANCE.json` verdict PASS, §5.5 |
| S2a blind audit (2/64 BAD, bar ≤3) | **FROZEN** | `AUDIT_RESULT.json` |
| S2b blind audit (0 consensus-BAD, bar ≤3) | **FROZEN** | `AUDIT_RESULT.json` + `AUDIT_RATERS_RAW.md` written, §11.7 |
| S4 selection (2,000 samples / 42+5 triggers) | **FROZEN** | `selection.json`, §5.6 |
| Union content pool (1,146 train / 120 reserved) | **FROZEN** | `CONTENT_POOL_union.json`, gates PASS |
| Full-occlusion family tags (1,730 / 15,980) | **FROZEN** | `s2_full_occlusion_tags.json` |
| Corpus 9-frame anchors (222 clips / 444 mp4) | **FROZEN** | `corpus_anchors_index.json` |
| S1 grid (390 rows, seed 42) | **FROZEN** | `misc/ctt_v2_final/S1_GRID.json` |
| Holdouts: 10 zs classes · 10 shader families · 120 reserved clips | **FROZEN** | §7 |
| Caption grammar + §4 distributional bars | **PINNED** | measured from the 139 corpus captions before any new caption existed |
| Copy-gate admissibility + bars | **PINNED** | `VERIFY_copy_ref_discriminator.md` verdict PASS; A7 amendment-2 |
| Mix weights 15 / 6 / 34.5 / 34.5 / 10 (+ 3 contingency branches) | **PINNED (ruled), IN CODE** | A9 / A11 item 3; `root_common.INTENDED_WEIGHTS_PCT` + `ABSENT_BRANCH_WEIGHTS_PCT`, §11.1 |
| Pairing rule (ring offset, k=min(3,n−1)) | **PINNED** | `root_common.PAIRING_RULE` |
| VAE latents + cond_clean for S1/S2a/S2b/S4 | **IN FLIGHT** | jobs 9687982–9687985 running, §5 per-stratum |
| S1 full render (390 clips) | **PENDING** | 33 pilot clips exist; gate is credit-blocked |
| S1 pilot batch gate (blind 11-way Gemini) | **PENDING (blocked: Gemini credits)** | |
| Caption store (all strata) | **PENDING (blocked: Gemini credits)** | §6 |
| Caption round 3 (be-verb defect fix) | **PENDING (blocked: Gemini credits)** | A8 ordered; staged in `RESUME_ON_CREDITS.sh` |
| Corpus-139 Layer-2 leak audit | **PENDING (blocked: Gemini credits)** | anchors built; A4/A8 require it *before assembly* |
| S4 blind-guess caption gate (seed 44, n=150) | **PENDING (blocked: Gemini credits)** | A9 |
| `conditions/` (Gemma text embeds) | **PENDING (blocked: captions)** | ~2 GPU-h once captions land |
| The 8 pre-registered S2a inline-OOD ops | **FROZEN (pre-registered)** | `PREREG_inline_ood_ops_s2a.json`, advisor-ratified 2026-07-28, §11.4 |
| Assembled root + `ROOT_MANIFEST.json` | **PENDING** | machinery built, never run against real inventories |
| Assert battery A1–A10 executed | **PENDING** | §9 |
| Mixed-format smoke gate (2 shapes) | **PENDING** | A1b Q3 / A9; non-negotiable before S4 trains |
| **THE STAMP** | **PENDING** | §1.2 |

### 1.2 Why this is not stampable

A5 Ruling 9 defines "frozen" as: strata + exact counts + grid definitions + **caption store hash,
model version strings, raw-response archive paths and battery results** + holdout lists + mix
weights **intended and realized (counted)** + pairing rule + seeds + gate results. Five of those do
not exist:

1. **No caption store.** The Gemini project's prepayment credits are depleted (HTTP 429
   `RESOURCE_EXHAUSTED`, verified independently three times, most recently DOSSIER §10.3 at 02:15).
   This is billing exhaustion, not rate limiting; only the owner can clear it.
2. **No `conditions/` tree**, because every caption costs a Gemma-3-12b text-encode pass.
3. **No assembled root**, therefore no counted realized mix and no `ROOT_MANIFEST.json`.
4. **No executed assert battery.** §9 is a checklist, not a record.
5. **One pre-registration hole left** — the S4 caption gates. (The 8 inline-OOD ops are now
   pre-registered and ratified, §11.4.)

Additionally, **three of the numbers this document must stamp were wrong in the governing prose and
right on disk** (§11.1, §11.2, §11.3). §11.1 and §11.2 are now **reconciled by A11**; §11.3 is a
caption-count estimate that changes no byte the trainer reads. **Every §12 row now carries a
ruling** (stamp precondition 2 of §13.3 is met); what remains is execution, not decision.

---

## 2. Principles

Inherited from `eval_ladder/SPEC.md` §1 and re-stated only where the dataset changes their force.

1. **Derive, don't author.** Every set in this document is recomputed from a frozen source on disk
   (`root_common.py` holds no hand-kept list). A literal list is a list that will drift.
2. **Seatbelts are asserts, not comments.** The trainer joins the five root trees by identical
   relative path and **silently drops mismatches** — so a design invariant that is not machine-checked
   at assembly time is a wish. §9 is the machine-checked set.
3. **Freeze before you look.** Caption bars, copy bars and the mix were pinned before the data they
   judge existed. Post-hoc changes are **amendments**, written down and labelled.
4. **Never delete an inconvenient number.** Gate #8 failed at 0.7066 against its original ≤0.65 bar;
   that failure is recorded (§6.3), not erased, alongside the re-pin that superseded it.
5. **Op identity must reach the model only through the reference video.** No class names, no shader
   names, no refVFX trigger tokens in any caption (A1b Q3). A per-op text token would recreate
   refVFX's own text-routing recipe and let the model ignore the demo entirely.
6. **Captioning is instrument matching, not description quality.** The target is the certified
   corpus's own register, because every eval prompt is written in it (A4; ratified A8).
7. **Selection never touches the eval substrate.** No DINOv2 / harness feature may gate which
   training data is kept (A5 Ruling 3(i)). S1's per-clip rejects are mechanical only.
8. **Accumulate, never replace.** A batch that passed its pre-registered audit is not re-cut on a
   criterion invented after seeing the clips (A10).

---

## 3. The sample contract

### 3.1 What one training sample IS

Five files, one per root dir, **at the identical relative path in each**:

```
<root>/latents/<stratum>_r<NN>/<group>/<target>__ref_<reference>.pt
<root>/conditions/…            same relative path
<root>/cond_clean_latents/…    same relative path
<root>/reference_latents/…     same relative path
<root>/masks/…                 same relative path
```

| element | meaning |
|---|---|
| `<stratum>_r<NN>` | **replica directory.** The mix is realized by duplicating whole stratum replicas, never by a sampler — so the realized ratio is a property of the root on disk and can be *counted* (A3-F8.3 / A5 Ruling 4) |
| `<group>` | the class (S0) or the op id (S1 / S2 / S4) — the unit the ring-offset pairing runs inside |
| `<target>__ref_<reference>` | **the pairing is recorded in the filename.** Nothing inside any `.pt` names the reference. This is the operative record the trainer sees |

`latents`, `conditions`, `cond_clean_latents`, `reference_latents` are **per-file symlinks** into the
physical encode trees; `masks` are **real files**, deduplicated through a `_mask_store/` keyed on
`(f, h, w, sidedness)` and symlinked in. Verified on the S0 root: 385 symlinks in each of the four,
385 real files in `masks/`, over 26 class subdirs.

### 3.2 Tensor payloads (verified by loading, not by documentation)

| dir | keys | S0/S1/S2 shape | S4 shape |
|---|---|---|---|
| `latents` / `reference_latents` / `cond_clean_latents` | `latents, num_frames, height, width, fps` | `(128,16,20,15)` bf16; `16, 20, 15, 24.0` | `(128,5,14,26)` bf16; `5, 14, 26, 16.0` |
| `masks` | `mask` | `(16,20,15)` float32 ∈ {0,1} | `(5,14,26)` float32 ∈ {0,1} |
| `conditions` | `video_prompt_embeds`, `prompt_attention_mask`, `audio_prompt_embeds` | `(1024,3840)` bf16, `(1024,)` int64, `(1024,3840)` bf16 | identical (text shape is format-invariant) |

Pixel → latent: 480×640×121 @24fps → 16×20×15; 832×448×33 @16fps → 5×14×26. Spatial factor 32,
temporal factor 8 with `(F−1)/8+1`.

### 3.3 The silent-drop hazard — why exact set-equality is mandatory

`ltx_trainer/datasets.py:202-228`: `latents` is the primary source; for every `latents/**/*.pt` the
loader requires an **exact relative-path match** in each of the other four trees, else the sample is
dropped and training proceeds on the reduced set. Three verified gaps:

- **Gap 1** — `break` on first miss: only ONE missing source is reported per file.
- **Gap 2** — nothing is ever a `WARNING` and **nothing raises** unless *every* sample drops. The
  aggregate is one `INFO` line: `Fast index: N valid samples from M total (K skipped)`.
- **Gap 3, the invisible direction** — a sample present in `masks/` or `reference_latents/` but
  **absent from `latents/`** is never enumerated at all: not in `valid_count`, not in
  `len(data_files)`, not in the skipped arithmetic. It simply does not exist.

Consequence, stated plainly: if a new stratum's samples land in `latents/` but its `masks/` are
written at a slightly different relative path, **the entire new stratum is dropped and the run trains
on the old data while looking completely healthy** — same loss curve, same everything. Counts alone
are not sufficient; **A1 asserts set-equality of relative paths**, and `dryrun_epoch.py` promotes any
skip to a job failure. Both are mandatory (§9).

`batch_size` **must be 1**. There is no `collate_fn` in the repo, so mixed shapes crash
`default_collate`; and even at equal shapes `flexible.py:603-608` reads
`num_frames/height/width/fps` from **element 0 only**. Upstream says the same
(`process_videos.py:1452`). This is the correct setting, not a workaround.

### 3.4 Reference semantics

`ReferenceConditionConfig` (`flexible.py:96-108`): reference latents are **prepended**
(`torch.cat([cond_latents, noisy_latents], dim=1)`) with `cond_timesteps = zeros` and
`cond_loss_mask = zeros` — clean tokens in bidirectional self-attention. `probability` on that config
**is** reference dropout; it is applied by a single batch-wide draw at `flexible.py:638`
(`if torch.rand(()) >= config.probability: return`) which gates **only** the reference-token
concatenation. Endpoint (prefix/suffix) conditions are applied in a separate, earlier loop
(lines 432–443) at probability 1.0, and endpoint tokens are clean, timestep-0 and excluded from loss
on every step (`loss_mask = loss_mask & (mask == 0)`, line 544) — dropped step or not. **There is no
gradient channel through endpoint tokens in either regime** (A6, verified in code).

Pinned for this round: **reference dropout p = 0.1 in both arms**, *conditional* on PARTIAL-B
remaining in the frozen decision rule; if PARTIAL-B is deleted, **go to p = 0** (A6).

---

## 4. Sidedness

**Sidedness is a class property, not a caption style.** Source of truth
`data/processed/transitions_std121/corpus_manifest.json`, cross-checked against
`outputs/taxonomy/class_axes_v2.yaml`; disagreement is a hard error (`prompts.py:79-95`).

It drives **three coupled things at once**:

| # | thing | one-sided | two-sided |
|---|---|---|---|
| 1 | caption form | `{S1}. sksz.` | `{S1}. sksz. {S2}.` |
| 2 | **mask** (`assemble_root.py:161-174`) | `m[:2]=1` | `m[:2]=1` **and** `m[-1]=1` |
| 3 | **cond_clean** (`encode_conditioning.py:129-158`) | **bitwise copy** | last latent frame replaced by a standalone encode of the trailing 9 pixel frames |

`mask = 1` ⇒ the token is conditioned (clean latent, timestep 0, excluded from loss).
`mask = 0` ⇒ noised, contributes to loss. **The mask is a pure function of the conditioning, never
of the class label** (seatbelt #5 of the ladder2 design).

Window constants (`encode_conditioning.py:38-45`): `PX_PREFIX=9`, `PX_SUFFIX=9`,
`SUFFIX_GEN_FRAMES=8`, `STD_FRAMES=121`. Measured causal-VAE bleed: suffix rel-L2 median **0.280**,
prefix median **8.3e-5** — this is why `cond_clean` exists at all.

Per-stratum sidedness, verified:

| stratum | sidedness | source |
|---|---|---|
| S0 | 299 one-sided / 86 two-sided **pairs**; 107 / 32 **clips**; 7 of 26 classes two-sided | `inventory.json`, counted |
| S1 | per specialist — 9 one-sided (270 clips) / 2 two-sided (120 clips) | forced by `run_gen.py`: only `spec_hero_flight` and `spec_shadow_smoke` take a suffix anchor |
| S2a / S2b | **100 % two-sided** — a true A→B between two different contents | render contract |
| S4 | **100 % one-sided** — refVFX I2V is A → effect(A) | A1b Q3; keeps S4 entirely on the bitwise-copy `cond_clean` path |

The two two-sided S0 classes owning the most pairs: `hero_flight` 24, `shadow_smoke` 24; then
`earth_wave` 12, `giant_grab` 12, `air_bending` 6, `water_bending` 6, `flame` 2.

**The residual risk this creates is accepted, not solved** — see §8.1.

---

## 5. Strata

Sample counts below are **planned, pre-exclusion**: they apply the ring-offset rule to the verified
clip counts. Assembled counts are PENDING (§1.2).

### 5.0 Summary

| stratum | clips (disk) | groups | ring-offset samples | format | sidedness | captions |
|---|---|---|---|---|---|---|
| S0 | 139 | 26 classes | **385** (fixed by inventory) | 480×640×121 @24 | 107 one / 32 two | 139, **frozen, byte-identical** |
| S1 | 33 of 390 rendered | 11 specialists | 1,170 planned | 480×640×121 @24 | 270 one / 120 two | PENDING |
| S2a | 7,990 | 799 ops | 23,970 | 480×640×121 @24 | all two | PENDING (333 strings) |
| S2b | 7,990 | 799 ops | 23,970 | 480×640×121 @24 | all two | PENDING (800 strings) |
| S4 | 2,000 | 42 triggers | 6,000 | **832×448×33 @16** | all one | PENDING |
| **total** | **18,152 on disk** (+357 S1 pending) | **1,677 groups** | **≈ 55,495** | two shapes | | |

Ring offset, everywhere, one rule (A1b Q5 / A5 Ruling 4):

```python
k = min(MAX_REFS_PER_TARGET, n - 1)      # MAX_REFS_PER_TARGET = 3
for i, target in enumerate(stems):        # stems = the group's clips, sorted
    for j in range(1, k + 1):
        ref = stems[(i + j) % n]
```

Groups with fewer than 2 trainable clips after exclusions are dropped whole.

---

### 5.1 S0 — the certified real-VFX corpus

| | |
|---|---|
| **What** | The 139-clip held-in half of the `transitions_std121` corpus — the same data the ladder2 `ic_gen` adapter was certified on. Reference = a different clip **of the same class**. |
| **Provenance / licence** | Higgsfield-sourced VFX transition clips, standardised to 121f. Research use; the frozen split is `data/processed/transitions_std121/split_v1.2.json`, sha `c694659d`. |
| **Counts (verified)** | 139 clips · 26 classes · **385 pairs**. Root: 385 symlinks in each of `latents`/`conditions`/`reference_latents`/`cond_clean_latents`, 385 real files in `masks`, 26 class subdirs each. |
| **Class arithmetic** | `inventory.json` `held_in` = **29**; 3 classes dropped for <2 trainable clips (`hole_transition`, `jump_transition`, `seamless_transition`) ⇒ **26** in the root. `held_out` = 10 (§7). |
| **Format** | 480×640 × 121f @24fps → latent `(128,16,20,15)`. |
| **Built by** | `eval_ladder/train/inventory.py` → `precompute.py --mode cond-clean` → `precompute.py --mode text` → `assemble_roots.py`. Physical root `eval_ladder/dataset/roots/ic_gen/` (main tree). |
| **Gates** | None of its own — S0 *is* the certified reference distribution. Its captions are the instrument the other strata are matched to. |
| **Holdouts** | The 10 zero-shot classes are never in this root; 11 zs-audited endpoint clips and the 42 test clips are eval-side (§7). |
| **Captions** | 139 records in `eval_ladder/dataset/captions/dataset_captions.json`, `{caption, video}`. **Byte-identical, never re-written** — A8 rejected re-captioning S0 *more strongly than A4 did*: every eval prompt is old-register and frozen, so re-captioning S0 converts a correlational cue into a full train/eval register shift at the decision point. |
| **Accepted risk** | The captioner-identity residual (§8.4). |

### 5.2 S1 — specialist counterfactuals

| | |
|---|---|
| **What** | Generated clips: for each of 11 pinned specialist LoRAs, run its transition on **synthetic-bank endpoints that are not corpus content**. The only stratum that punishes appearance-copying *in the real-VFX visual domain* while decoupling class-manner from class-typical content. |
| **Provenance** | Endpoints from the union content pool (§5.3); adapters `outputs/training/ladder2/{arm}/checkpoints/lora_weights_step_02000.safetensors`. |
| **Counts (grid, verified in `S1_GRID.json`)** | 390 rows = 9 one-sided specialists × 30 + 2 two-sided × 60. Per-arm: 30 each except `spec_hero_flight` 60 and `spec_shadow_smoke` 60. **400 distinct endpoint clips**; endpoint_a bank split exactly **195 synth / 195 humanvid**; a **10-endpoint probe set shared by all 11** arms (110 of the 390 rows) gives the same-content × diff-op diagonal. |
| **Counts (rendered)** | **33 pilot clips on disk** (3 per specialist × 11), `outputs/videos/ctt_v2_s1/spec_*/`. The remaining 357 are PENDING. |
| **Format** | 480×640 × 121f @24fps, identical to S0. |
| **Pairing group — RULED (A11 item 6, 2026-07-28)** | The inventory's group key is the **ARM (specialist)**, never the endpoint, and **1,170 stands**. The endpoint reading is not merely different, it is **wrong**: group=endpoint would pair same-content × different-op, violating the standing rule *"reference = same operator, different content."* A1b is explicit both ways — Q5 ("endpoints unique within a specialist… within-op endpoint disjointness guarantees ref shares no content with target") and the count itself (9×30×3 + 2×60×3 = 1,170 only works with the specialist as the ring group). **Inventory schema: 11 groups**, one per specialist. And yes — **a shared-probe row may reference a non-probe row**: within an arm, probe endpoints are ordinary distinct clips, the ring runs over the arm's sorted stems irrespective of probe membership, and the probe set's cross-arm role (the same-content × diff-op diagonal) is a *diagnostic across arms*, not a pairing constraint within one. **No count change.** §12.6 closed. |
| **Sidedness** | Native per specialist — 9 one / 2 two. Forced by the mechanism, not chosen: `run_gen.py` appends a `SuffixConditionConfig` only when `row["sided"] == "two"`, and 9 of 11 specialists are one-sided, so they *cannot* produce a true A→B pair. |
| **Gates** | **Batch gate (A5 Ruling 3(i)):** blind 11-way Gemini class identification, `gemini-3.5-flash`, temp 0, `max_output_tokens ≥ 2000`, bar **top-1 ≥ 80 %** (chance 9.09 %), **with a 33-clip control arm of real corpus clips of the same classes**. Verdict rule: `PASS` = batch ≥80 % **and** control ≥80 % **and** mechanical rejects ≤3/33; `FAIL_S1_DROPS` = batch <80 % with passing control; `INSTRUMENT_INVALID` = control <80 % (re-adjudicate, do not blame S1). **Result: PENDING (blocked: Gemini credits).** |
| | **Per-clip mechanical rejects only (A5 Ruling 3(ii)):** decode corruption (frame count ≠ 121 or geometry ≠ 480×640); frozen (mean abs inter-frame delta over frames 9–120 < 1/255) or black (mean luma < 8/255 on ≥10 % of frames); endpoint identity — prefix rel-L2(gen[0:9], anchor) > **τ = 0.12790240**. **No DINOv2, no harness substrate, anywhere in selection.** |
| | **τ provenance (measured, CPU):** p95 over n=198 prefix-conditioned inline-validation clips across all checkpoints; p50 0.0526, p90 0.0997, max 0.1397. Artefact `outputs/ctt_v2/s1/tau_endpoint.json`. |
| | **Acceptance-by-bank differential audit**, flag if synth vs humanvid acceptance differs >15 pp (A1b Q1e). |
| **Holdouts** | None of its own — the contrast is free: 11 held-in classes get S1 amplification, 15 do not. Pre-registered as an **observational diagnostic, never a selection criterion**. |
| **Disjointness (asserted, PASS)** | S1's 400 endpoints ∩ {74 eval endpoints, 11 zs-audited endpoints, 42 test clips, 9 DAVIS eval source sequences, 222 corpus clips} = **∅**. Recorded in `S1_GRID.json:HARD_ASSERT_endpoint_disjointness`. Eligibility: 1,146 pool training clips → **1,120 eligible**, 26 excluded (5 DAVIS eval sequences + 21 near-duplicate-pair members). |
| **Captions** | PENDING. 144 of 390 rows are renderable today from the M3 pilot store; 246 need the full store. `S1_GRID.json:prompts.provisional` requires **all 390 to re-render from the final pinned caption store before the S1 root is assembled** — pilot prompts are provisional. |
| **Pre-registered fallback** | If the 33-clip pilot fails its gates, **S1 drops entirely** and the mix renormalizes. The schedule does not slip. |
| **Accepted risk** | Generated data trained on generated data; mitigated by the class-ID batch gate and by S1 being ≤6 % of stream. |

### 5.3 The union content pool (shared input to S1 and S2b)

`data/processed/ctt_v2_strata/CONTENT_POOL_union.json` + `content_pool_emb_union.npy`
(1,266 × 512, L2-normalised).

| | training | reserved | total |
|---|---|---|---|
| synth (v1 187 + v2 104) | 291 | 20 | 311 |
| humanvid | 855 | 100 | 955 |
| **total** | **1,146** | **120** | **1,266** |

- **Gates, after trim:** `gate_a_mean_cos` **0.5200** (bar ≤0.52) · `gate_b_matched_pr` **50.56**
  (bar ≥42.82, n=187, 300 draws, seed 20260725) · `participation_ratio_pass: true`. Before trim
  gate A failed at 0.5569; 544 humanvid clips were trimmed. The 291 synth clips are a **protected
  floor** (exp_081 shipped them at A 0.5008 / B 42.83); only humanvid was removable.
- **Near-duplicate handling:** 15 pairs pinned to training. Exactly one cross-bank Pexels-ID
  collision exists (`vcbench_painting_3805736_2160x4096` ≡ `humanvid_3805736`, cos 0.9952) and the
  builder already caught it — 0 missed. The *within-synth* bank is the dirtier one (1 pair ≥0.95,
  9 pairs ≥0.90) vs humanvid's zero.
- **Licence:** humanvid = Pexels via HumanVid URL lists — Pexels ToS restricts ML use;
  **owner cleared use 2026-07-27**, recorded in `license_ledger.json` and `$LAB/misc/ctt_v2/DOSSIER.md §11`.
  synth = vcbench (Pexels via VC-Bench, 118) + DAVIS-2017 research (65) + OpenVid-1M CC-BY-4.0,
  underlying Panda-70M/YouTube (148).
- **Format contract:** every clip is asserted on the decoded array, not on metadata —
  `arr.shape == (121, 640, 480, 3)`, 24 fps, H.264.
- **Two M3 pool drops (adjudicated, `POOL_DROPS_M3_ADJUDICATION.json`):** the pool is
  **byte-unchanged**. `humanvid_10344332` is a **no-op** — it was never in the pool. For
  `openvid_T1MiFx98l3g_0_50to156` (blank white A-anchor: grayscale mean 250.2, std 0.79) the
  whole-clip drop was **HELD** in favour of a **role-scoped A-only exclusion**, because the clip
  occupies the **B field in 10/10 rendered rows** and B-role uses frames 112–120, which are normal
  content (std 74.05). The caption store must not contain an A-role description for it. §11.6.

### 5.4 S2a — procedural transitions, synth-bank endpoints

| | |
|---|---|
| **What** | GLSL shader transitions rendered between two *different* real content clips. Endpoints are byte-exact by construction; the middle is a parametric operator. This is the stratum that breaks the "reference content ≈ target content" correlation S0 has trained into the model for three years. |
| **Provenance** | Shader bank `$LAB/misc/gl-transitions/transitions` (125 `.glsl`); contents from `CONTENT_POOL.json` (291 training clips: davis 34 / openvid 98 / vcbench 55 / vcbench_v2 104). |
| **Built by** | `experiments/exp_081_s2_stratum/` — `plan_s2.py` → `render_s2.py` (env-var driven: `SHARD=k NSHARDS=n`) → `accept_s2.py`. moderngl + EGL/llvmpipe, **CPU only, never a GPU**. Engine symlink to `exp_075_procedural_transition_engine/engine`; `engine_git_commit` is recorded per clip row. |
| **Counts (verified by direct count of `meta/clips_shard*.jsonl`)** | **7,990 clip rows · 799 complete ops · 56 shaders · 333 content pairs · 291 distinct content clips · exactly 10 clips/op (min = max = 10)**. `ops_shard*.jsonl` holds **809** finalised op rows = 799 complete + 10 dropped. `videos/` and `filmstrips/` each hold 7,990 files; `retired_blacklisted/videos/` holds 420. |
| **How it got there** | 800 planned ops → 6 shaders blacklisted at >50 % measured rejection → 78 ops / 420 clips retired → 713 surviving → 87 backfill ops → **799 / 7,990**. Policy v1 shipped this batch (62 trainable shaders). |
| **Format** | 480×640 × 121f @24fps, CRF 18. Latent `(128,16,20,15)`. |
| **Op definition** | `(shader, uniforms, easing, onset/release, flip, swap)` — **timing is part of the id**, sha1-hashed. `flip ∈ (none,h,v,hv)` p=0.6 · `swap ∈ bool` p=0.5 · uniforms jittered p_vary=0.85 · 7 easings · window [8,112], onset ∈ [8,28.8], release ∈ [91.2,112] ⇒ duration ≥ 62.4 frames. **`swap` inverts the shader progress argument only; it does NOT exchange A/B content** — see §11.3. |
| **Byte-purity (the invariant the stratum rests on)** | `i0 = floor(onset)`, `j0 = ceil(release)`; the renderer asserts `clip[:i0+1] == a_src[:i0+1]` and `clip[j0:] == b_src[j0:]`. ⇒ **frames 0–8 are byte-identical to source A and frames 112–120 byte-identical to source B in every clip, unconditionally.** This is what makes per-content-clip A-role/B-role captioning exact. |
| **Gates (`S2_ACCEPTANCE.json`, verdict PASS)** | `pure_phase_max_abs_diff` **0.0** (bar ≤0.5) · `seam_max` **1.9984** (≤2.0) · `m1_p10_min` **0.2547** (≥τ 0.2543) · `m2_max_dq` **0.4916** (≤0.5) · `m1_min_flag_count` 35 · overdraw **1.2506** (≤2.5) · attempts min/med/max 10/12/25 · shaders over 50 % rejection: **none** · all six hard invariants true. Plus per-clip pre-render gate-2 (endpoint identity at the op's params, `max(d0,d1) ≤ 0.5`). |
| **Blind audit (`AUDIT_RESULT.json`, PASS)** | n=64 shader-stratified, two independent blind raters + operator adjudication, bar ≤3 consensus-BAD. rater1 [30,40] · rater2 [41,45] · **consensus []** · adjudicated **2 BAD** (`PuzzleRight` s2_0229_c06, `Slides` s2_0270_c08) · agreement 60/64 = 93.8 %. |
| **Holdouts** | 10 shader families never rendered (§7). **8 inline-OOD ops pre-registered and excluded at assembly** — `PREREG_inline_ood_ops_s2a.json`, seed 42, 8 distinct shaders, 80 clips (~1 % of S2a); their encodes stay on disk for the inline lane. §11.4. |
| **Captions** | PENDING. **333 distinct caption strings** from **454 (clip, role) descriptions** over 291 clips (163 clips occupy both roles). §11.3 corrects the dossier's 666/582. |
| **Encode state** | `outputs/ctt_v2/encodes/S2a/` — roster frozen at 7,990, `nshards=16`; 922 latents written at 02:56, jobs running. |
| **Accepted risk** | The full-occlusion family, 870/7,990 = 10.89 % (§8.3). |

### 5.5 S2b — procedural transitions, union-bank endpoints

Identical machinery, new operators, new contents. `experiments/exp_082_s2_humanvid/`, seed **20260727**.

| | |
|---|---|
| **What differs from S2a** | (a) **NEW operators** — task count is the declared lever, so re-running S2a's ops was rejected; 799→1,598 ops doubles it. (b) **Policy v2** — 56 trainable shaders; the 6 that S2a retired at >50 % rejection (`PuzzleRight, SimpleZoom, SimpleZoomOut, StripDatamoshGlitch, splitSlideOutHorizontal, swap`) are pre-blacklisted. (c) **Union pool** contents with a bank quota. |
| **Counts (verified)** | **7,990 clip rows · 799 complete ops (800 finalised, 1 dropped) · 56 shaders · 800 content pairs · 919 distinct content clips · exactly 10 clips/op.** |
| **Bank quota (A1b Q2d / A5 Ruling 4), realized and verified** | pair rows: synth–synth 200 / cross 400 / humanvid–humanvid 200 = **25 / 50 / 25 exactly**. Clip-level: 2,001 / 3,977 / 2,012 = **25.0 % / 49.8 % / 25.2 %**. Endpoint slots: **humanvid 8,001 / synth 7,979** of 15,980. **Bank-pure ops: 0** — verified independently by joining every clip row's A and B against the pool's bank field. Per-op minority-bank endpoints min/median/max 2/8/10. |
| **Gates (`S2_ACCEPTANCE.json`, verdict PASS)** | `pure_phase_max_abs_diff` **0.0** · `seam_max` **2.0** (bar ≤2.0 — passes, but this is the one number with **no headroom**) · `m1_p10_min` **0.255** · `m2_max_dq` **0.4996** · `m1_min_flag_count` 13 · overdraw **1.1549** (better than S2a's 1.2506) · attempts 10/11/25 · shaders over 50 % rejection: **none**. Policy v2 did its job: S2a needed a 78-op retirement and an 87-op backfill; S2b needed neither. |
| **Blind audit** | n=64, same protocol, two independent blind raters (fresh Claude agents — Gemini was credit-blocked; for a *rater* the relevant independence is from each other and from the campaign's expectations, both preserved). **rater1 1 BAD (#055) · rater2 0 BAD · consensus 0** ⇒ **PASS** (bar ≤3). Worst case, adjudicating the single disputed clip against us gives 1/64. ⚠ Recorded in DOSSIER §10.6; **no `AUDIT_RESULT.json` on disk** (§11.7). |
| **Reference pairing** | Recorded in the plan as *dynamic at train time*: ref ≠ target drawn from the same `op_id`'s 10 clips. Within-op endpoint disjointness (20 distinct endpoint clips per op, asserted) guarantees ref shares **no content** with target. The assembler realizes this as the same ring-offset rule. |
| **Encode state** | roster frozen at 7,990, `nshards=16`; encoding in flight. |
| **Accepted risk** | Full-occlusion family 860/7,990 = 10.76 % (§8.3); person-only semantic coverage in the humanvid half (§8.5). |

### 5.6 S4 — refVFX I2V real-VFX effects

**S4 is IN, by A9, reversing A5 Ruling 2.** Confidence 0.75. It is the only stratum besides S0 with
real-VFX effect operators — the direct counter to A1b's "shader demo ⇒ read manner; VFX demo ⇒ copy"
conditional-policy hole — and it is insurance: if the S1 pilot fails its (credit-blocked) gate,
excluding S4 leaves the central claim with **zero** VFX-domain anti-copy signal.

| | |
|---|---|
| **Provenance** | refVFX `I2V_LoRA` release. **ONE tar shard**, `data/raw/refvfx/data/I2V_LoRA/shard-00000.tar`, 12.32 GB; all 2,000 selected samples reference shard 0, **0 missing**. (An earlier note said `shard-{00000..00079}`, conflating the tar with the 80 caption `batches/` dirs; reconciled.) Each sample ships `output_video.mp4`, a clean `NNNNNN.input_image_or_video.png`, and a `.json` sidecar. |
| **Sidecar ground truth** | 5 keys — `prompt`, `effect_type`, `mask_type` (always null), `orientation`, `data_subset`. `effect_type` = leet trigger + plain descriptor (`h01k green hulk transformation`); `prompt` names the effect in the clear. **Neither may ever reach a caption.** |
| **Counts (verified in `selection.json`)** | **2,000 samples · 42 train triggers · 47–50 samples each · 5 held-out triggers · 2,000 filmstrips on disk.** |
| **Format — RATIFIED (A11 item 4, 2026-07-28)** | Native 832×464 · 33f · 16fps. **Encoded at 832×448×33** — 464 is not a multiple of the VAE spatial factor 32 (464/32 = 14.5) and `process_videos.py:parse_resolution_buckets` rejects it. The 832×448 bucket is a **pure 16-row centre crop with NO resampling** (width scale exactly 1.0, 3.4 % of height). Verified latent on disk: `(128, 5, 14, 26)`, fps 16.0 ⇒ **1,820 tokens, shift 1.2350**. **The latents on disk stand — do NOT re-encode.** A1b Q3's "no crop" *as literally written* is VAE-impossible, and a ruling premised on a physical impossibility is void on that word: it is amended to its **intent** — no resampling, no letterbox, no retiming, i.e. protect dynamics, which is half the frozen metric's definition of manner. 832×448 is the **minimal** such amendment; the 832×480 alternative resamples (3.4 % upscale) *and* crops width, strictly worse on the ruling's own intent, and re-encoding would spend 0.3 L40S-h to replace a correct artefact with a worse one. **A1b Q3 wording is amended to:** *"832×448×33@16fps: 16-row centre crop, zero resampling — the minimal VAE-legal bucket; literal 'no crop' is VAE-impossible."* §11.2 / §12.4 closed. |
| **Sidedness** | 100 % one-sided (owner ruling; structurally correct for I2V) ⇒ entirely on the bitwise-copy `cond_clean` path, never touching the suffix machinery. |
| **RoPE** | S4 references are **S4-native** (ring within op ⇒ same trigger), so reference and target share the ~2 s span. No cross-span mismatch. Downstream normalisation divides by a fixed 20-second horizon: 121f@24 → 5.0 s → 0.25; 33f@16 → 2.0 s → 0.10. Both in range, on the same physical time axis. |
| **Caption source** | **Frames 0–8, exactly as A4 specced.** Neither of the owner's two counter-proposals is adopted, on measurement (§11 of DOSSIER, A9). |
| **Caption-leak evidence (the decisive measurement)** | Blind-guess attack — give a judge only the caption plus the 41-way effect menu, score against a 20,000-draw label-permutation null, n=150 each: input image **2.0 %** vs null 2.67 % (0.75×, below chance) · output frame 0 **2.7 %** vs 2.30 % (n.s.) · **frames 0–8 4.0 %** vs 2.55 % (1.57×, **p=0.18, n.s.**) · real-footage floor 1.3 % · **⟨positive control⟩ LAST frame 71.3 %** vs 2.81 % (**25.4×, p=5e-5**). The positive control is what makes the null interpretable: the attack finds the effect at 25× chance when it is genuinely there. Named-effect judge: **0/150** on all three prefix conditions, 58 % on last-frame. |
| **Withdrawn** | DOSSIER §5's onset-audit argument. It does not reproduce (65.0 % → **56.0 %** same seed; control 2.5 % → **0.0 %**; n=200 seed 43 → 53.8 %), its config was never archived, and — the real error — it measured *"do the pixels show an onset"* and **inferred caption leak from pixel onset**. The direct measurement refutes the inference. |
| **Rejected alternatives** | Frame-0 captioning is statistically identical to frames 0–8 (144/150 clips wrong under both, paired) and discards 8/9 of the evidence for no measurable gain. Clean-subset filtering would cost **≈925 of 2,000 clips (CI 789–1,063)** *non-uniformly*, wiping out five effect classes entirely (pirate / electricity / baby / squish / princess, all 100 % onset) — a silent re-weighting of the stratum. |
| **Its own caption gates (A9; A5 Ruling 5's premise broke)** | S4's frames 0–8 are another model's outputs, not byte-pure source, so: the **full 12-gate battery including gate #8 run on S4 captions separately, not pooled**; a **blind-guess gate** (fresh seed 44, n=150, bar = permutation p ≥ 0.05 **AND** top-1 ≤ null + 3 pp, **with a mandatory last-frame positive control at ≥10× null to prove power**); and the Layer-2 named-effect judge on **100 %** of S4 captions as a tripwire. Blind-guess does **not** subsume gate #8 — blind-guess tests *reader* identifiability, gate #8 tests *encoder-exploitable statistical association*. **Keep both.** All PENDING (blocked: Gemini credits). |
| **Mandatory before S4 trains** | The mixed-format smoke gate: mini-root of ~20 corpus + ~20 native-S4 samples, 100–200 steps, asserting (i) no silent skipping — per-format consumed counts logged and exact; (ii) finite, comparable per-format loss; (iii) shapes flow through RoPE in bf16 for both; (iv) one train==inference equivalence probe. **Additionally: the corrected two-clause shift assert — §9 D3** (pin check at t ∈ {1820, 4800} → {1.2350, 2.3021} within 1e-3, **plus** a realized check against the observed token counts). A9's `{1.120, 2.302}` is struck. ~1–2 GPU-h. Placeholder captions suffice. **Credit-independent — should run as soon as a GPU frees.** |
| **Masks** | `(5, 14, 26)` = **1,820 elements** (A9's `(5,20,15)`/1,500 prose was the corpus-grid conflation and is **struck**; disk already agrees). Regenerated for the S4 shape, **never reused** (a reused 16-frame mask is a loud `RuntimeError` at `flexible.py:533`, which is the good failure mode). `assemble_root.ensure_mask` keys the store on `(f,h,w,sided)`, so it adapts automatically. |
| **Schedule ruling — S4 rides for free or not at all** | The launch is **NEVER** held for S4. Cutoff pre-registered at **root-assembly time**: when every other stratum's captions and encodes are done and asserts pass, if S4's caption lane has not passed its gates, **assemble without S4** under the branch weights (§6.1). No slip, no debate. |
| **Accepted risks** | The bimodal noise schedule (§8.2) and A3's format/latent-grid stratum signature — the two objections that survive; the caption leg is refuted. |

---

## 6. Captions

### 6.1 Grammar

Rendered, never authored. `eval_ladder/prompts.py:render_prompt()` is the **only** renderer and it
produces the training captions *and* the eval registry rows, so train == inference by construction.

```
one-sided   "{S1}. sksz."
two-sided   "{S1}. sksz. {S2}."
```

- Token = `sksz` (`eval_ladder/arms.yaml:7`; recorded in `dataset/conditions_token.json`). It is
  **2 Gemma subwords**. It must be probe-verified inert on the base model before any training.
- **The trigger occupies the middle sentence slot — it *is* the transition slot.** Asserted at
  `precompute.py:114`: `assert all(f" {tok}." in r["caption"] for r in rows)`.
- **A-role**: uppercase-initial, ends `.` — 139/139 in the corpus.
- **B-role**: lowercase-initial (it was mid-sentence in the source corpus caption), ends `.`,
  a participial noun phrase — 32/32 lowercase, 29/32 with an `-ing` participle.
- **Leak guard** (`precompute.py:112-113`): the outcome marker `"The scene transforms into "` must be
  ABSENT from every training caption. Pre-ladder2 `conditions/` trees are the leaky captions and must
  never be reused.
- The **matching unit is the DESCRIPTION, not the caption**. The caption-level p90 of 68 words is
  entirely the two-sentence effect.

### 6.2 Pre-registered distributional bars (DOSSIER §4)

Measured from the 139 certified corpus captions **before a single new caption existed** — that
ordering is what makes them pre-registered. Full output `M1_corpus_caption_stats.txt`; the empirical
length list for the per-call sampler is `M1_length_empirical.json` (171 values).

| statistic | corpus value | bar for new strata | type | round-2 measured |
|---|---|---|---|---|
| POOLED description words p10/p25/**p50**/p75/p90 | 21 / 29 / **33** / 36 / 39 (min 13, max 47, mean 31.9) | p50 ∈ [29, 36] | HARD | **34** ✅ |
| description words p10 · p90 | 21 · 39 | p10 ∈ [16,26] · p90 ∈ [34,44] | HARD | **24 / 40** ✅ |
| A-role opens with determiner | 96.4 % (134/139) | ≥ 86.4 % | HARD | **99.5 %** ✅ |
| B-role opens with determiner | 96.9 % (31/32) | ≥ 86.9 % | HARD | **98.5 %** ✅ |
| B-role lowercase-initial | 100 % (32/32) | 100 % | HARD | **100 %** ✅ |
| B-role has `-ing` participle | 90.6 % (29/32) | ≥ 80.6 % | HARD | **100 %** ✅ (LLM 100 %) |
| A-role uppercase-initial | 100 % (139/139) | 100 % | HARD | **100 %** ✅ |
| colour-term density | 3.158 words/description (92.4 % ≥1) | ∈ [1.579, 4.737] | REVIEW | **3.735** ✅ |
| camera-phrase rate | 3.5 % (6/171) | within ±10 pp | FLAG | **2.3 %** ✅ |
| audio words | 0/171 | **0** | HARD | **0** ✅ |
| exact duplicate descriptions | 0 | < 2 % | HARD | **0 %** ✅ |
| `-ing` verb present | 88.9 % (152/171) | — | reference | — |

Per-role detail: A-role (n=139) 18/29/**34**/37/40; B-role (n=32) 22/28/**32**/35/35.

**Detector calibration reproduces the pinned M1 numbers bit-for-bit** before touching new text
(word counts exact; determiner 134/139 & 31/32; `-ing` 29/32; audio 0/171; camera 3.51 %). Only
colour density is inexact (3.199 vs 3.158, +1.3 %) — **reported, not tuned.**

### 6.3 Gate #8 — recorded FAILED, then re-pinned (A8)

The original bar `≤0.65` on a function-word-only classifier probe **FAILED** at 0.7139 (round 1) →
0.7066 (round 2, full remediation). It is recorded as failed, never quietly replaced. The controls
are what made it adjudicable:

| control | balanced acc | what it establishes |
|---|---|---|
| **NULL — corpus vs itself, random halves** | **0.5059** | the probe is correctly calibrated; no inflation from n, CV or features |
| **corpus A-role vs corpus B-role** | **0.6419** | two registers *inside* the certified corpus, same captioner. The 0.65 bar ≈ "no more separable than A is from B within S0" |
| corpus vs round 1 | 0.7139 | the pinned gate — FAIL |
| corpus vs round 2 | 0.7066 | after moving commas 0.78→1.74, p50 29→34, colour 2.31→3.74, camera 11.0 %→2.3 % onto the corpus — **the gate did not move** |
| **round 1 vs round 2** | **0.7233** | **decisive** — same model, prompt delta only, separates *more* than corpus-vs-new ⇒ no prompt reaches 0.65 |
| **round 2 synth vs humanvid** | **0.5518** | 🟢 the two new banks are mutually near-blind — the entire signature is "old captioner vs new captioner", NOT a stratum cue |

Robust across three vocabularies (0.712/0.707, 0.712/0.692, 0.712/0.690). Ablation: removing
punctuation+length makes it **worse** (0.7208) ⇒ the residual is not commas or length. The
coefficients are a preposition profile plus the copula — a captioner-generation fingerprint no prompt
instruction reaches.

**🔒 Re-pinned gate #8:**

| | bar | function |
|---|---|---|
| **8a HARD** | corpus vs new (pooled mass store) **≤ 0.73** | **drift guard** — above 0.73 cannot be the known fingerprint ⇒ a bug (mixed prompts, wrong round's store, contamination) ⇒ stop and investigate |
| **8b HARD** | synth-vs-humanvid function-word probe **≤ 0.60** | the load-bearing replacement: **stratum-internal style blindness**. With the per-(clip, role) shared store this guarantees the only style cue in the stream is S0-vs-rest, never S1-vs-S2 or bank-vs-bank |
| **8c** | original ≤0.65 | **recorded FAILED** with the full control table |

**Why acceptance is safe, and the argument matters:** S0 samples are themselves demo-following
samples on corpus content, and **every eval prompt is old-register**. So "old register ⇒ S0 mode" at
eval routes toward *baseline-like* behaviour — the shortcut **suppresses** the candidate's new-data
capability. It is a **false-FAIL mechanism, not a false-PASS one.** The false-pass channel
(demo-ignoring, corpus-typical output) is already gated by `copy_ref` and both cross-liveness controls.

Gate #9 (full-vocab AUC **0.9192**, bar 0.80 for INVESTIGATE) is REVIEW ⇒ INVESTIGATE.

### 6.4 Store shape and generation contract

The store is **per-(clip, role)**, written from **byte-pure 9-frame anchors only**, and shared by
symlink — exactly as the S0 root already shares 139 condition embeds across 385 samples.

| stratum | (clip, role) descriptions | distinct caption strings |
|---|---|---|
| S0 | — (frozen, 139 captions) | 139 |
| S1 | 400 — each endpoint occupies exactly one role (270 one-sided rows use A only; 120 two-sided rows use A + B) | **300** (190 one-sided + 110 two-sided) |
| S2a | 454 over 291 clips (163 clips dual-role) | **333** |
| S2b | 1,217 over 919 clips (298 dual-role) | **800** |
| S2 union (verified: S2a's 291 clips ⊂ S2b's 919) | **1,297 over 919 clips** | 1,133 |
| S1 ∪ S2 | **1,405 over 972 clips** | — |
| S4 | 2,000 (A-role) | 2,000 |
| **total new descriptions** | **≈ 3,405** | |

- **Generator** `gemini-3.6-flash`; **Layer-2 auditor** `gemini-3.5-flash` (the pro tier 429s on this
  key for every pro model — a genuinely different model, so generator/auditor independence is
  preserved). `thinkingLevel: "minimal"` pinned as part of the auditor config (at default thinking,
  flash burns ~111 thought tokens and A4's 120-token cap truncates mid-word; `thinkingBudget: 0`
  returns HTTP 400).
- **Auditor validated with no slack:** re-auditing each description against a *different* clip
  (n=391) returns **100.0 % inaccurate** vs 5.75 % matched.
- **Video payloads, not filmstrips.** A 9-frame 480×640 mp4 anchor costs **63 prompt tokens**
  (86 with text) vs 1,197 for a 3×3 filmstrip JPEG. Throughput ~16 descriptions/s at 120-way, zero
  429s across ~2,300 flash calls; the full store costs ~6 min per round including a 100 % audit.
- **First-pass governance (A8):** the ≥97 % bar **covers prompt-controllable failures only ⇒ PASSES
  at 99.25 %** (0 Tier-1 hits/400 · 1 format violation · 2 leak=YES at 0.50 % · 23 `inaccurate` at
  5.75 %). `inaccurate` gets its own governance: **≤8 % first-pass (REVIEW)**, **0 unresolved in the
  final store (HARD)**.
- **Content-borne leak rule (new):** a reproducing `leak=YES` whose language describes byte-pure
  visible content ⇒ mark `leak_content_borne`, **keep**, log. If it describes change or onset ⇒ drop
  the clip.
- **Tier-1 is DEMOTED from audit to tripwire (A9 process ruling (b)).** It caught **0/150** on
  captions leaking at 71.3 % — it is zero evidence of semantic cleanliness and **must never again be
  cited as such**. It stays only for the mechanical catastrophe class it can catch: trigger tokens,
  shader basenames, `sksz` misuse. **The blind-guess attack becomes the standard leak audit**, with a
  permutation null and a **mandatory positive control**, for every caption lane with an enumerable
  label space (S4 effects, S1 classes, S2 shader families). Lanes without a label menu rely on gate #8.
- **Config archiving is MANDATORY** for any measurement that gates a design decision (A9 process
  ruling (a)): verbatim prompts, model ID **plus per-call echoed `modelVersion`**, all decoding params,
  seed and sample manifest, **raw per-call responses (JSONL)**, the scoring/null-model code, and a
  one-script regeneration of the headline table. **`misc/ctt_v2_final/_verify_s4_inputs/` is the
  named STANDARD. An unarchived measurement cannot gate.**
- **Round 3 is ORDERED** (A8), scoped as an instrument-matching **defect fix**, not gate-chasing:
  A4's prompt mandates present/progressive verbs and thereby forces be-verbs to **0.0 % against a
  corpus at 8.8 %** — a prompt-manufactured categorical tell. A-role prompt only, replace the
  verb-form clause with *"then the subject's action in present tense (plain 'is/are' constructions
  are fine)"*. Nothing targeting `while` or any other coefficient — that would be mimicry. B-role
  untouched. **Pre-committed: round 3 need not and will not reach 0.65**; report the number for the
  record.
- **Caption paths — PENDING (blocked: Gemini credits).** When the store lands it goes to
  `outputs/ctt_v2/captions/<round>/` with `descriptions.json`, `records.json`, `run_meta.json`,
  `raw_generation_responses.jsonl`, `raw_audit_responses.jsonl`, `gate_report.json`. **Store hash,
  model version strings and archive paths must be written into §1.1 before the stamp.**

---

## 7. Holdouts and exclusions

**This is the table someone will check before claiming generalization.** Every set is *derived* at
assembly time from the frozen source named in the last column — never a hand-kept list
(`root_common.load_exclusions`). Counts verified 2026-07-28.

| # | set | n | members / rule | derived from | asserted by |
|---|---|---|---|---|---|
| H1 | **S0 zero-shot classes** | **10** | `cotton_cloud`, `display_transition`, `firelava`, `flying_cam_transition`, `live_concert`, `luminous_gaze`, `melt_transition`, `monstrosity`, `raven_transition`, `saint_glow` — 5 one-sided / 5 two-sided, a balanced zs set | `split_v1.2.json:generalist_holdout` (sha `c694659d`) | **A9** |
| H2 | **S2 shader families** | **10** | `DefocusBlur`, `VerticalOpen`, `burn0`, `directionalwarp`, `fadegrayscale`, `parametric_glitch`, `randomsquares`, `ripple`, `scale-in`, `wipeUp`. Principle: **every held-out shader keeps a same-genre cousin in training.** Never rendered into either S2 batch | `exp_082_s2_humanvid/HOLDOUT_S2_UNION.json` | **A7** |
| H3 | **Reserved union-pool clips** | **120** | 20 synth + 100 humanvid; `reserved[]` in the pool contract; never trained, eval-only | `CONTENT_POOL_union.json:reserved` | **A8** |
| H4 | **S4 held-out triggers** | **5 selected, 3 usable** | **CLEAN (the diagnostic universe):** `1ung13 jungle transformation`, `5en3m venom transformation.`, `s31lf13 taking a selfie with their younger self`. **`CONTAMINATED — excluded from the diagnostic universe` (A11 item 2):** `cr34sh crash zoom out effect`, `cr4n3 crane down camera motion` — §8.6 | `s4_refvfx/selection.json:held_out_triggers` | excluded at selection; not in the root |
| H5 | **Pre-registered S2a inline-OOD ops** | **8** | 8 ops from 8 distinct shaders, RNG seed 42, drawn from the otherwise-trainable S2a ops; supply the inline-validation OOD demos | `root_common.select_inline_ood_ops` | **A6** — ⚠ **the file does not exist; §11.4** |
| H6 | **Eval-endpoint universe** | **92** | union of: 74 registry endpoints + 36 registry references + 9 DAVIS eval source sequences mapped to pool ids + 11 zs-audited endpoints + the 42 pre-registered test clips, deduplicated | `eval_ladder/registry.jsonl`, `davis.yaml`, `split_v1.2.json` | **A5** |
| H7 | **S1 eligibility exclusions** | **26** | 5 DAVIS eval source sequences + 21 near-duplicate-pair members, removed from the 1,146 pool ⇒ **1,120 eligible** | `S1_GRID.json:eligibility` | asserted PASS in the grid |
| H8 | **S0 classes dropped for <2 trainable clips** | **3** | `hole_transition`, `jump_transition`, `seamless_transition` — held-in but unusable | `eval_ladder/train/inventory.json` | structural |
| H9 | **Role-scoped caption exclusion** | **1** | `openvid_T1MiFx98l3g_0_50to156` — **A-role only** (blank white A-anchor). B-role legitimate and kept | `POOL_DROPS_M3_ADJUDICATION.json` | caption-store rule; **no assert exists — §12.5** |

**Zs-audited endpoint clips (H6 component, 11):** `cotton_cloud_0`, `display_transition_1`,
`firelava_4`, `flying_cam_transition_4`, `live_concert_0`, `live_concert_4`, `luminous_gaze_3`,
`melt_transition_2`, `monstrosity_0`, `raven_transition_2`, `saint_glow_3`.

**DAVIS eval source sequences (H6 component, 9):** `bear`, `blackswan`, `elephant`, `hike`, `lucia`,
`mallard-water`, `rhino`, `snowboard`, `tennis`.

Class resolution for every one of these goes through `eval_ladder/prompts.py:clip_class()` against
the frozen split — **never by string-splitting a clip name.** `action_run_setonfire_6` belongs to
class `run_set_on_fire`; `flame_transition_0` to `flame`.

---

## 8. Known accepted risks

Each is a *recorded decision*, not an oversight. None may be re-litigated on impressions.

### 8.1 Sidedness is a near-perfect stratum signature — and it lives in the mask

**Ruling:** accept; no engineering; **pre-register the detector.** (A1b Q4, ratified A5 Ruling 4.)

S2 is 100 % two-sided, S4 100 % one-sided, S1 69.2 % one-sided, S0 77.0 % one-sided at clip level
(77.7 % at pair level). Under
A9's weights the stream lands roughly one-quarter one-sided, and **P(S2 | suffix anchor) ≈ 94 %**.
Sidedness is not merely a caption feature: it changes the **mask**, so the shortcut is available in
the conditioning tensor itself, not only in text.

Rejected fixes and why: S2 **cannot** be one-sided — an unrelated B is unpredictable from A + manner,
so you would be training hallucination. Cutting S2's weight guts the primary lever. The affordable
counterweights are S1's deliberate two-sided doubling (the 2 two-sided specialists get 60 rows each,
not 30 — they are the only VFX-domain two-sided samples in the mix) and S0's 86 two-sided pairs.

**Pre-registered detector:** report **every eval cell broken down by class sidedness**. If two-sided
zs/unseen classes systematically underperform one-sided ones **with shader-styled outputs**, the
shortcut fired and that becomes the headline diagnostic for the next round.

### 8.2 S4's bimodal noise schedule and format signature

**Ruling:** keep native, **log it analytically, caveat it — do NOT resample, do NOT patch the
sampler.** (A9.)

`ShiftedLogitNormalTimestepSampler._get_shift_for_sequence_length` makes sigma depend on the
**target token count**, `m = (2.05−0.95)/(4096−1024) = 1.1/3072`, `b = 0.95 − m·1024 = 0.5833`.
It raises nothing and logs nothing.

| population | latent grid | tokens | **shift** |
|---|---|---|---|
| S0 / S1 / S2a / S2b | (16, 20, 15) | 4,800 | **2.3021** |
| **S4 as encoded** | **(5, 14, 26)** | **1,820** | **1.2350** |
| *S4 as assumed by the prose* | *(5, 20, 15)* | *1,500* | *1.1204* |

**σ caveat re-derived at 1.2350 (A11 item 4, 2026-07-28).** A9 wrote this caveat against the assumed
1.1204; re-deriving it at the true 1.2350 is **mechanical and changes nothing qualitative** — the
low-σ concentration still discounts S4's effectiveness. If anything **1.2350 vs 1.1204 slightly
*narrows* the bimodality** (the two populations sit 1.067 apart in shift rather than 1.182), which
weakens the objection rather than strengthening it. The pre-written report caveat stands with the
number corrected.

The two populations train at materially different noise levels — short clips mostly at low noise,
long clips mostly at high noise. **Why it is not exclusion-grade once disclosed:** the shift is a
deterministic function of the data, identical in both training arms *and* in the certified baseline,
so it confounds **no claim the round reports**; it muddies only stratum-level sub-attribution, which
no pre-registered claim isolates. Its real bite is an **effectiveness discount** — low-σ concentration
attenuates S4's anti-copy signal — which is an argument against over-weighting, not against inclusion.

**Prescription:** (1) compute and archive exact per-stratum σ distributions **analytically** from the
root manifest, stamped here; (2) post-hoc split `sigma_tracker` output by stratum as a training-health
diagnostic; (3) the mixed-format smoke gate asserts per-format finite comparable loss and the exact
realized shifts; (4) the report caveat is pre-written. **Do not pad or resample** (3.7× interpolation
invents frames; retiming corrupts dynamics, half the frozen metric's definition of manner) and **do
not patch the sampler** (that is a recipe change smuggled inside a "pure dataset intervention" claim).

**A3's format/latent-grid signature objection also survives** and is accepted on the ground that
excluding S4 for identifiability while keeping S2 — perfectly identifiable by demo appearance plus
compression signature — is not a consistent principle; and the binding claims are measured at 121f
corpus format where S4's format cue is **absent**, so a format-keyed policy cannot fabricate a pass.
Worst case S4's 10 % is wasted.

### 8.3 The full-occlusion shader family

**Ruling: KEPT in both halves, unchanged. Tagged. One cheap diagnostic pre-registered with its
trigger and response committed now.** (A10, confidence 0.85 keep-beats-drop / 0.9 no-new-gate.)

Family = **exactly six shaders by name**: `ButterflyWaveScrawler`, `CrossZoom`, `GridFlip`,
`StaticFade`, `flyeye`, `squeeze`. The raters' looser "noise/glitch/wave/mosaic/blur" wording is
explicitly **not** the definition; the six names are the replicated intersection of two blind raters
and are operationally crisp.

Verified exposure (`s2_full_occlusion_tags.json`, recounted from the 15,980 tag records):

| half | in family | of | pct |
|---|---|---|---|
| S2a | **870** | 7,990 | 10.89 % |
| S2b | **860** | 7,990 | 10.76 % |
| **total** | **1,730** | 15,980 | **10.83 %** |

Both raters independently converged on the same pattern, naming 5 of the same clips — replication
across two blind raters makes this signal, not rater noise.

**Why keeping it is correct, not lenient:** rater 2's gating premise was *"if downstream training
needs at least one source to remain legible at every timestep."* Success here is defined as *"donor
manner transferred AND **endpoint content** kept."* These clips keep endpoint content **byte-exactly**
(`pure_phase_max_abs_diff 0.0`, asserted per clip). **Nothing in the task contract requires
mid-transition legibility.** And the "teaches destruction" worry dissolves on the pairing rule:
reference = same operator, different content, enforced by the 10-clips-per-exact-op structure, so
**every full-occlusion target arrives with a full-occlusion demo.** The gradient signal is *"produce
occlusion WHEN THE DEMO SHOWS OCCLUSION"* — manner transfer, precisely the target capability.

**And the M1 mush gate does not see them** — `m1_min_flag` False for all seven exemplars, `m1_p10`
0.368–0.664, all far above τ=0.2543. This is the banked lesson *"M1 can't tell displacement from
destruction"* empirically demonstrated: 331 S2b clips score **lower** than the worst clip a rater
flagged, and the sole BAD sits at 0.650 against a batch median of 0.692. **Consequence for the stamp:
the M1 gate is a mush *floor*, not a quality *proxy*, and the blind audit is doing independent work.
Neither verdict may be overridden by pointing at the other.**

**🔒 Pre-registered diagnostic, committed before any candidate is scored:**

| | |
|---|---|
| **statistic** | per generated eval clip, over the **core-mask frames**, per-frame `max(zNCC(F, A_endpoint), zNCC(F, B_endpoint))`; record the per-clip **min** = `leg_min`. Existing M1 machinery pointed at outputs. CPU, minutes. |
| **report** | per cell, median `leg_min` for the candidate vs median `leg_min` for that cell's GT donors |
| **TRIGGER** | the family is implicated **iff** the candidate FAILS the pre-registered pass bars **AND** candidate median `leg_min` < GT median `leg_min` − **0.15** on `G-unseen-cross` **or** `G-zs-cross` |
| **RESPONSE** | iff triggered, ONE ablation: retrain with all `full_occlusion_family` clips excluded from both halves (~1,730 clips; renormalize **within S2 only**, holding the 34.5/34.5 stratum weights so the mix contract is preserved), score the same cells (~6.4 H100-h + ~6 GPU-h) |
| **CLOSURE** | if the trigger does not fire, **the family is exonerated and the question is permanently closed** — it may not be reopened on rater impressions alone |
| **negative** | **no new measurement is required before training. Do not build a legibility gate.** |

### 8.4 The captioner-identity residual

**Ruling:** accepted; bar superseded, captions accepted (A8). See §6.3 for the full control table.
Residual = corpus-vs-new function-word separability **0.7066**, against a corpus internal A-vs-B
register distance of **0.6419** and a NULL of **0.5059**. It is a captioner-*generation* fingerprint
(preposition profile plus the copula), not a fixable style defect; chasing it further would be
Goodhart. Bounded by the re-pinned ≤0.73 drift guard, mitigated by the interventional register-swap
detector, and — the load-bearing point — it is a **false-FAIL mechanism, not a false-PASS one**.

**🆕 Pre-registered interventional register-swap lane:** for all 46 cross-cell rows, render a *second*
prompt through the unchanged `render_prompt` path with descrA/descrB regenerated by the round-3
pipeline from the same corpus clips' byte-pure 9-frame anchors; same demos, same 2 seeds, ` sksz.`
asserts intact. 46 × 2 = **92 extra gens/arm ≈ 4.1 GPU-h/arm**, riding the main arrays.
**FIRE CRITERION:** declared FIRED in a cell if mean Δ > 2 pp + 0.84 × SE_paired
(Δ = score(new-register) − score(old-register), SE from per-item paired diffs). Opposite sign at the
same magnitude ⇒ REVIEW, not fired. **If fired:** the claim of record still uses old-register numbers
(the instrument is the claim), with the swap lane reported alongside as evidence the cross numbers
**understate** capability. Corpus anchors (job 9686746, 222/222 clips, 444/444 mp4s) serve this lane
and the corpus-139 audit.

**Caption-battery scope for this lane — RULED (A11 item 7, 2026-07-28): EXEMPT from the 12-gate
battery; three per-item checks are MANDATORY.** The distributional gates cannot coherently apply:
the lane's whole purpose is that its register distribution *differs* from corpus, so gate 8a's
separability is the **treatment, not a defect**, and "fixing" the swap descriptions to pass
distributional bars would break the intervention's representativeness of the round-3 pipeline. The
lane is diagnostic-only — A8 pre-registers that the claim of record uses old-register numbers even if
it fires. What *is* load-bearing is the lane's internal validity, and A8's own spec already names it
("same filters, Tier-1/Layer-2 audited"):

| | per-item check on all 46 × 2 swap descriptions | why |
|---|---|---|
| (i) | **render-path asserts intact** — ` sksz.` exactly once, outcome marker absent | the lane must go through the unchanged `render_prompt` path |
| (ii) | **Tier-1 mechanical tripwire** — zero trigger tokens, zero shader basenames | a leaked trigger/shader in a swap prompt would confound Δ with leakage |
| (iii) | **standard Layer-2 accuracy audit, 0 unresolved-inaccurate** | an inaccurate description confounds Δ with *accuracy* rather than *register*; replace via the pipeline's normal retry |

**Nothing distributional, nothing else.** §12.7 closed.

### 8.5 HumanVid is person-only

100 % of the 1,499 HumanVid clips carry `subject.label = "person"`, against the synth bank's mixed
DAVIS/vcbench/OpenVid content. **The geometric fear is measurably wrong** — HumanVid's internal mean
cosine is only +0.032 above synth's (0.5331 vs 0.5008), it has the **lowest max internal cosine of
any bucket (0.8261, zero pairs ≥0.90)**, and its participation ratio is **higher** (60.82 vs 46.63).
The 36.3 % trim (1,499→955) is an artefact of the pool-level mean-cosine bar, not of redundancy.
**The residual concern is semantic (all clips contain people), not geometric — and it is *aligned*
with eval content.** Recorded; spend nothing (A5 Ruling 4).

### 8.6 The S4 held-out trigger split leaks at the token level

`cr34sh` appears in both held-out (`crash zoom out`) and train (`crash zoom in`); `cr4n3` appears in
held-out (`crane down`) and twice in train (`crane over the head`, `crane up`). **Any "held-out
effect" claim on those two is contaminated** — and the leak is worse than token-level, since
"crash zoom out" vs "crash zoom in" is a semantic near-duplicate. **3 of 5 held-out triggers are
clean.** A1b Q6 ordered a redraw from the 6,995 raw pool (no token overlap AND no semantic
near-neighbour in the train 42); A5 Ruling 2 mooted it by excluding S4; **A9 reinstated S4 without
restating the redraw.**

**RULED (A11 item 2, 2026-07-28): NO REDRAW. Strike the 2 contaminated triggers from the diagnostic
universe.** A1b Q6's redraw is **formally superseded by a change of purpose**: at A1b time the
held-out set was a design-level generalization diagnostic; post-A9 its *only* consumer is A2 Q4's
winner-arm, non-claim lane, which **already** restricts itself to "only the 3 token-clean held-out
triggers" (3 × 4 endpoints × 2 seeds = 24 gens). **The root is untouched either way** — held-out
triggers were excluded at selection and the frozen 2,000 are all train-trigger clips.

The token-leak framing above understates it in one direction and overstates it in another: the
*lexical* leak is moot for the root, but the **semantic** leak stands — "crash zoom out" vs trained
"crash zoom in" makes the model near-ID on that manner — so `cr34sh`/`cr4n3` are unusable as
"unseen effect" probes **regardless of any redraw**. Redrawing 2 replacements would buy n=5 instead
of n=3 on a **non-claim diagnostic**, at the price of operationalizing a fuzzy "no semantic
near-neighbour" criterion. Simple wins.

**What this means operationally:** H4 tags the two as `CONTAMINATED — excluded from the diagnostic
universe` (§7); any "held-out effect" sentence in the report may cite **only the 3 clean triggers,
with n=3 disclosed**. The frozen selection is **not re-opened**. §12.2 closed.

---

## 9. The pre-launch assert suite

A5 Ruling 9. **Every check is a hard failure.** There is no warnings-only mode, no `--skip`, no
severity flag: `assert_root.py` exits non-zero if a single check fails and the launch scripts gate on
that exit code. Executable as a checklist:

```bash
cd $LAB/diffusion-research/.claude/worktrees/bottleneck-branch
PY=$LAB/envs/diffusion/bin/python
ROOT=outputs/ctt_v2/roots/ctt_v2_mix

$PY scripts/ctt_v2/assert_root.py  --root $ROOT      # A1–A10, writes ASSERT_REPORT.json
$PY scripts/ctt_v2/dryrun_epoch.py --root $ROOT      # zero-skip epoch, promoted to job failure
```

| # | check | why it exists | status |
|---|---|---|---|
| **A1** | **set-equality of RELATIVE PATHS across all 5 root dirs** — not counts | the trainer joins by relative path and silently drops mismatches (§3.3). Counts are not sufficient (A3-F8.1) | ☐ |
| A1b | root non-empty | | ☐ |
| **A2** | inventory integrity — each stratum inventory's sha256 matches what the root was assembled from | provenance; a changed inventory means the root no longer describes itself | ☐ |
| A2b | every relative path is `<stratum>_r<NN>/<group>/<target>__ref_<reference>.pt` | | ☐ |
| A2c | every sample resolves to a verified inventory entry | | ☐ |
| **A3** | realized mix within **±0.5 pp** of intended, **COUNTED from the assembled root** | the mix is realized by symlink duplication precisely so it can be counted, not asserted by construction (A3-F8.3) | ☑ **solvable within tolerance** — a `--plan-only` assembly over the real S0+S2a+S2b inventories lands `max_dev 0.4296 pp` (tol 0.5) on the ruled `S1,S4`-absent branch, §11.1 |
| **A4** | every caption contains ` sksz.` **exactly once**; the outcome marker is absent; zero Tier-1 leak strings; no caption in an inventory is missing from `CAPTIONS.json` | | ☐ |
| **A5** | S1/S2 endpoints ∩ {eval endpoints, zs-audited endpoints, the 42 test clips} = ∅, classes resolved by `prompts.clip_class()` | A3-F5b | ☐ |
| **A6** | the 8 pre-registered S2a inline-OOD ops are absent — **and a vacuous exclusion on a root containing S2a is itself a hard failure** | prevents the exclusion silently doing nothing | ☑ **non-vacuous; PASSES in a `--plan-only` assembly** (8/8 ops resolve, exactly 8 groups / 80 clips dropped with reason `inline_ood_op`, 0 survive), §11.4 |
| **A7** | the 10 `HOLDOUT_S2` shader families are absent | | ☐ |
| **A8** | the 120 reserved union-pool clips are absent | | ☐ |
| **A9** | the S0 zero-shot classes are absent — as a group class *and* via any corpus-resolvable endpoint | | ☐ |
| **A10** | the Day-0 copy-gate admissibility check is recorded as **PASSED** — absent file or non-PASS verdict ⇒ FAIL | A5 Ruling 1 makes it a training blocker | ☑ currently reads PASS from `VERIFY_copy_ref_discriminator.md` |
| **D1** | **dry-run epoch: ZERO skipped samples.** Reproduces `_discover_samples` exactly, then resolves and metadata-loads every sample. Any skip — join miss, dangling symlink, unreadable tensor, wrong keys, disagreeing shapes — exits non-zero | the trainer only debug-logs a skip, which is how a silently truncated epoch reaches the claim table | ☐ |
| **D2** | grep the trainer's own index line immediately after launch: `Fast index: N valid samples from N total`, **N == expected**, and it must count S4's 6,000 | the only in-band confirmation the trainer agrees with us | ☐ |
| **D3** | **mixed-format smoke gate** — 2 shapes, 100–200 steps, per-format consumed counts exact, finite comparable per-format loss, RoPE in bf16 for both, one train==inference prefix-anchor probe, **realized shifts asserted in two clauses (below)** | three silent trainer defects are on record; this is non-negotiable (A1b Q3, 0.9 confidence) | ☐ |
| **D4** | `cond_clean` smoke assert logged before step 250 (kill rule K0) | | ☐ |

#### D3's shift assert — CORRECTED (A11 item 4, 2026-07-28)

A9 pre-wrote `realized shifts ∈ {1.120, 2.302} exactly`. **That assert would have FAILED on a
correct encode** (§11.2). It is replaced by **two clauses**, because one clause alone catches only
one of the two drift directions:

```python
from ltx_trainer.timestep_samplers import ShiftedLogitNormalTimestepSampler as S
shift = S._get_shift_for_sequence_length

# clause 1 — PIN CHECK: catches TRAINER-CONSTANT drift.
#   m = (2.05-0.95)/(4096-1024) = 1.1/3072,  b = 0.95 - m*1024 = 0.5833...,  NO CLAMP,
#   so 4,800 tokens legitimately extrapolates above max_tokens.  Verified first-hand in
#   timestep_samplers.py:122-134.
assert abs(shift(1820) - 1.2350) < 1e-3      # S4        5*14*26
assert abs(shift(4800) - 2.3021) < 1e-3      # corpus   16*20*15

# clause 2 — REALIZED CHECK: catches ENCODE-GEOMETRY drift.
#   the set of shifts observed in the mixed-format smoke run must equal the function's
#   outputs at the REALIZED token counts, exactly.
assert observed_shifts == {shift(5*14*26), shift(16*20*15)}
```

**Why both.** Clause 1 alone passes even if S4 were re-encoded to a different grid (the constants
never moved); clause 2 alone passes even if upstream changed `min_shift`/`max_shift` (observed and
expected move together). `root_common.shift_for_tokens()` reproduces the trainer function verbatim
and `root_common.RULED_SHAPES` carries the two ruled grids, so the numbers above are **derived from
the shape, never restated** — the arithmetic that produced 1,500/1.120 cannot recur.

**Pre-assembly, additionally required by A4/A8 and not covered by the battery:**

| | check | status |
|---|---|---|
| C1 | corpus-139 Layer-2 leak audit (anchors ready, `corpus_anchors/`) | ☐ **PENDING (blocked: Gemini credits)** |
| C2 | caption round 3 passes the re-pinned 12-gate battery | ☐ **PENDING (blocked: Gemini credits)** |
| C3 | S4's separate 12-gate battery + blind-guess gate + 100 % Layer-2 tripwire | ☐ **PENDING (blocked: Gemini credits)** |
| C4 | S1 pilot batch gate (blind 11-way Gemini + control arm) | ☐ **PENDING (blocked: Gemini credits)** |
| C5 | `eval_ladder/prompts.py` diff resolved | ☑ committed `3bed923` |
| C6 | certified instrument hygiene — `reference_v4.npz` sha matches the pin | ☑ restored; sha `e6ea4011…a8ad2818` matches |

---

## 10. Reproduction

Ordered commands that rebuild each stratum from scratch. Absolute interpreter paths throughout — the
worktree depth makes relative paths silently wrong (an exit-127 lesson from 2026-07-27).

```bash
LAB=/projects/illinois/eng/cs/jrehg/users/emirkisa
W=$LAB/diffusion-research/.claude/worktrees/bottleneck-branch
M=$LAB/diffusion-research
PY=$LAB/envs/diffusion/bin/python
TRPY=$LAB/LTX-2-official/.venv/bin/python                     # owns ltx_trainer (VAE, video reader)
export PYTHONPATH=$LAB/LTX-2-cond-bleed-fix/packages/ltx-trainer/src
```

### 10.1 S0 (already built; rebuild only if the split changes)

```bash
cd $M
$PY eval_ladder/train/inventory.py                                   # -> eval_ladder/train/inventory.json
$PY eval_ladder/train/precompute.py --mode cond-clean --device cuda  # GPU
$TRPY eval_ladder/train/precompute.py --mode text --device cuda      # GPU: render captions + Gemma encode
$PY eval_ladder/train/assemble_roots.py
```
`--mode text` shells out to `scripts/process_captions.py` with
`--text-encoder-path .../gemma-3-12b-it-qat-q4_0-unquantized`. **Every new caption costs a
Gemma-3-12b text-encode pass on GPU.**

### 10.2 Union content pool

```bash
cd $W && $PY experiments/exp_082_s2_humanvid/build_content_pool_union.py
# -> data/processed/ctt_v2_strata/CONTENT_POOL_union.json + content_pool_emb_union.npy
```

### 10.3 S2a (batch 1, policy v1)

```bash
cd $M/experiments/exp_081_s2_stratum
$PY plan_s2.py                                       # -> PLAN_S2.json
MODE=smoke $PY render_s2.py                          # 3 ops, ~2 min, hard asserts -> smoke/SMOKE.json
sbatch --partition=secondary --account=campusclusterusers --array=0-19 \
       --cpus-per-task=4 --mem=32g --time=03:55:00 --requeue job_render_array.sbatch
$PY accept_s2.py --stage verify                      # -> S2_ACCEPTANCE.json
$PY accept_s2.py --stage sheets --n 64               # blind audit sheets, bar_max_bad 3
```
`render_s2.py` has **no argparse** — env vars + `config_s2.yaml`. Required env:
`PYOPENGL_PLATFORM=egl`, `LP_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`. Sharding is `i % nshards == shard` over op index — **NSHARDS must be identical
across a resume** (deriving it from `SLURM_ARRAY_TASK_COUNT` silently repartitions). **Not idempotent
across a plan change**: a second batch MUST get a new `outputs.dir`.

### 10.4 S2b (batch 2, policy v2, union pool)

```bash
cd $W/experiments/exp_082_s2_humanvid
sbatch job_plan.sbatch                               # -> PLAN_S2_UNION.json (seed 20260727)
sbatch job_smoke.sbatch
sbatch job_render_array.sbatch                       # 20 shards; --exclude=ccc0424 (broken EGL)
# resume: job_render_resume.sbatch  (HARDCODES NSHARDS=20 — never derive it)
$PY accept_s2.py --stage verify && $PY accept_s2.py --stage sheets --n 64
$PY bank_rejection_audit.py                          # gate-rejection-by-bank differential, flag >10 pp
```

### 10.5 S2 full-occlusion tagging

```bash
cd $W && $PY scripts/ctt_v2/s2_tag/tag_full_occlusion.py
# -> data/processed/ctt_v2_strata/s2_full_occlusion_tags.json  (asserts all six shaders present in
#    both halves, so a stale definition fails loudly)
```

### 10.6 S1

```bash
cd $W
$PY scripts/ctt_v2/s1/measure_tau_endpoint.py        # -> outputs/ctt_v2/s1/tau_endpoint.json
$PY scripts/ctt_v2/s1/build_s1_grid.py               # -> misc/ctt_v2_final/S1_GRID.json (seed 42)
sbatch scripts/ctt_v2/s1/job_s1_pilot.sbatch         # 33 clips, 3 array tasks, H100
$PY scripts/ctt_v2/s1/gate_s1_pilot.py               # mechanical rejects + blind 11-way Gemini gate
# then the full 390-row array, gated on the pilot verdict
```

### 10.7 Corpus 9-frame anchors (for the Layer-2 audit + register-swap lane)

```bash
cd $W && sbatch data/processed/corpus_anchors/build.sbatch    # secondary, ffmpeg -threads 1
```
⚠ **ffmpeg at 12-way on a login node dies** with `pthread_create() failed`. All media cutting goes to
a `secondary` CPU node with `-threads 1`.

### 10.8 Captions — **BLOCKED**

```bash
bash $W/scripts/ctt_v2/captions/RESUME_ON_CREDITS.sh     # step-by-step, guarded; does NOT auto-run
```
Step 0 of that script is a single cheap call that must return 200 before anything is spent.

### 10.9 VAE encodes (prompt-agnostic — NOT blocked on captions)

```bash
cd $W
$PY scripts/ctt_v2/encode/encode_strata.py stage         # CPU: freeze rosters, extract S4 from the tar
sbatch --partition=HCESC-L40S-normal --account=hcesc-l40s --gres=gpu:L40S:1 \
       --array=0-15%8 --export=ALL,GROUP=s2  scripts/ctt_v2/encode/job_encode.sbatch
sbatch --partition=HCESC-L40S-normal --account=hcesc-l40s --gres=gpu:L40S:1 \
       --array=0-3%4  --export=ALL,GROUP=aux scripts/ctt_v2/encode/job_encode.sbatch
$PY scripts/ctt_v2/encode/encode_strata.py verify        # hard count assert + shape/fps spot check
```
`NSHARDS` is hardcoded per stratum (`S2a 16, S2b 16, S1 1, S4 4`) and is **never** derived from
`SLURM_ARRAY_TASK_COUNT`. Every write is `.tmp` + `os.replace`, so a preempted task cannot leave a
truncated `.pt` that skip-if-exists would then trust. Measured throughput ~1,750 clips/L40S-h.

### 10.10 Assemble + assert

```bash
cd $W
# per-stratum inventories.  S2a/S2b are built from the RENDER MANIFESTS, not the encodes,
# so they do not block on the in-flight VAE jobs (sources are attached later).
$PY scripts/ctt_v2/build_inventories.py s0 --out outputs/ctt_v2/inventories/S0.json
$PY scripts/ctt_v2/build_inventories.py s2meta --stratum S2a --sided two --no-require-sources \
    --meta-glob 'outputs/videos/ctt_v2_s2/full/meta/clips_shard*.jsonl' \
    --out outputs/ctt_v2/inventories/S2a.json          # 799 groups / 7,990 clips / 23,970 pairs
$PY scripts/ctt_v2/build_inventories.py s2meta --stratum S2b --sided two --no-require-sources \
    --meta-glob 'outputs/videos/ctt_v2_s2_humanvid/full/meta/clips_shard*.jsonl' \
    --out outputs/ctt_v2/inventories/S2b.json          # 799 groups / 7,990 clips / 23,970 pairs
# S1 groups are the 11 ARMS (A11 item 6), not the endpoints; S4 groups are the 42 triggers.
$PY scripts/ctt_v2/assemble_root.py --init-manifest outputs/ctt_v2/strata_manifest.json
$PY scripts/ctt_v2/assemble_root.py --manifest outputs/ctt_v2/strata_manifest.json --plan-only
$PY scripts/ctt_v2/assemble_root.py --manifest outputs/ctt_v2/strata_manifest.json
$PY scripts/ctt_v2/assert_root.py  --root outputs/ctt_v2/roots/ctt_v2_mix
$PY scripts/ctt_v2/dryrun_epoch.py --root outputs/ctt_v2/roots/ctt_v2_mix
```
`assemble_root.py` is idempotent: it computes the desired `(path → target)` map, deletes anything not
in it, creates only what is missing. Re-running is a no-op.

---

## 11. Where disk disagreed with the prose

Recorded because the dossier already contains one such correction (S2a is 7,990/799, not the
inherited 8,410/809) and **that pattern must not repeat silently**. Ordered by consequence.

> **Status after A11 (2026-07-28):** §11.1 **RESOLVED** (code + manifest corrected, commit `674ecf1`
> + this one) · §11.2 **RESOLVED** (crop ratified, assert rewritten, `REF_mixed_length.md` corrected)
> · §11.3 recorded, no action beyond the caption-count estimate · §11.4 **RESOLVED**
> (`PREREG_inline_ood_ops_s2a.json` written and ratified) · §11.5 recorded · §11.6 **RESOLVED**
> (role-scoped exclusion now derived + machine-checked in three places) · §11.7 **RESOLVED**
> (`AUDIT_RESULT.json` written) · §11.8 recorded.

### 11.1 The assembly code carries A5's mix, which A9 superseded — **RESOLVED**

| source | S0 | S1 | S2a | S2b | S4 |
|---|---|---|---|---|---|
| **A9 / DOSSIER §12 — the governing ruling** | 15 | 6 | **34.5** | **34.5** | **10** |
| `scripts/ctt_v2/root_common.py:INTENDED_WEIGHTS_PCT` | 15 | 6 | 39.5 | 39.5 | **0.0** |
| `outputs/ctt_v2/strata_manifest.json` | 15 | 6 | 39.5 | 39.5 | 0.0 (`present: false`) |

`root_common.py:63` still reads *"RULING 4 — S0 15 / S1 6 / S2 79; S4 OUT this round (RULING 2)"*.
A5 Ruling 2 was **reversed by A9**. Assert **A3** (realized mix ±0.5 pp) compares against these
constants, so **an assembly run today would assemble the wrong mix and then assert it as correct.**
The `present` flag is a clean toggle (`--set-present S4=true`) but the weights are not toggled with it.
**Must be corrected before assembly.**

**RESOLVED (A11 item 3).** `root_common.INTENDED_WEIGHTS_PCT` now reads
`{S0 15, S1 6, S2a 34.5, S2b 34.5, S4 10}` behind a sum-to-100 guard; `assemble_root.py` **derives**
the manifest's weights from it instead of restating literals; `_S_PRESENT` marks S2b and S4 present;
and `outputs/ctt_v2/strata_manifest.json` was regenerated from that single source. A9's three
pre-registered contingency branches are recorded next to the constants as
`root_common.ABSENT_BRANCH_WEIGHTS_PCT`, each guarded to sum to 100 and to cover exactly the
complement of its own key:

| absent | S0 | S1 | S2a | S2b | S4 |
|---|---|---|---|---|---|
| — (headline) | 15 | 6 | 34.5 | 34.5 | 10 |
| **S1** (S1-fail) | 15 | — | 36.5 | 36.5 | **12** |
| **S4** (S4-cutoff) | 15 | 6 | 39.5 | 39.5 | — |
| **S1 + S4** | 15 | — | 42.5 | 42.5 | — |

⚠ **A latent second landmine, found while doing this:** the manifest's `absent_weight_overrides`
previously held `{"S1": {S0 15, S2a 42.5, S2b 42.5}}` — that is A9's **both-absent** branch sitting
under the **S1-only** key, so dropping S1 alone would have silently deleted S4 from the mix as well.
Fixed by the table above. **A9's 34.5 is confirmed PER S2 HALF ⇒ S2 total 69 %** (A11 item 3, 0.98).

### 11.2 S4's on-disk geometry is not what any ruling says — **RESOLVED**

| claim | source | disk |
|---|---|---|
| "native everything (832×464 · 33f · 16fps, **no crop**, no interpolation, no letterbox)" | A1b Q3 | encoded at **832×448** — a 16-row centre crop, forced by `464/32 = 14.5` being VAE-illegal |
| "Masks regenerated at **(5,20,15)**" | DOSSIER §12 (A9) | true grid is **(5,14,26)** — verified by loading an encoded latent |
| "S4 native … 1,500 tokens … shift **1.120**" | `REF_mixed_length.md` | 5·14·26 = **1,820 tokens** ⇒ shift **1.2350** |
| "the smoke gate asserts realized shifts ∈ {1.120, 2.302} **exactly**" | DOSSIER §12 (A9) | **this assert would FAIL.** The pair is {**1.2350**, 2.3021} |

Root cause: `REF_mixed_length.md`'s S4 row took the **frame count** from refVFX and the **spatial
grid** from the corpus, producing (5,20,15) — which is the shape S4 would have had if it *had* been
reshaped to 480×640, the very thing A1b ruled out. The encode script found the real constraint and
documented it correctly in its own header. The crop is 16 rows of 464 (3.4 % of height), pure, no
resampling, and `S4_BUCKET` is a one-line change costing ~0.3 L40S-h to revisit — but **"no crop" is
now false and the σ caveat A9 pre-wrote quotes the wrong number.**

**RESOLVED (A11 item 4).** The 832×448 encode is **ratified; the latents on disk stand and are not
re-encoded.** A1b Q3's "no crop" is amended to its intent (§5.6); the masks row is corrected to
`(5,14,26)` and A9's `(5,20,15)` prose is struck; `REF_mixed_length.md`'s 1,500-token/1.120 row is
corrected to 1,820/1.2350 in place, with the root cause recorded there; the σ caveat is re-derived at
1.2350 (§8.2); and A9's one-clause smoke-gate assert is replaced by the **two-clause** assert in §9.
`m = 1.1/3072`, `b = 0.5833`, **no clamp** — re-verified first-hand against
`ltx_trainer/timestep_samplers.py:122-134`.

### 11.3 S2a needs 333 caption strings, not 666 — `swap` does not exchange A/B content

DOSSIER §1.10 records *"distinct DIRECTED (first,second) pairs: 666 … 582 per-clip-role descriptions
(291 clips × 2 roles)"*. Counted from the 7,990 manifest rows:

| quantity | dossier | **disk** |
|---|---|---|
| directed (A,B) pairs | 666 | **333** |
| unordered pairs | 333 | **333** |
| pairs appearing in both orders | — | **0** |
| (clip, role) descriptions | 582 | **454** (163 of 291 clips are dual-role, 128 are single-role) |

The inference that `swap` doubles the directed pair count is falsified by the render contract itself,
as later established in `POOL_DROPS_M3_ADJUDICATION.json`: *"`swap` inverts the shader progress
argument only, it does NOT exchange A/B content."* Same correction applies to S2b (800 directed = 800
unordered, 0 both-orders). This **halves** the S2a caption-string estimate and changes the store size
by ~128 descriptions.

### 11.4 The 8 pre-registered S2a inline-OOD ops do not exist

`root_common.PREREG_INLINE_OOD` points at `misc/ctt_v2_final/PREREG_inline_ood_ops_s2a.json`.
**The file is absent.** `load_exclusions()` therefore returns `inline_ood_ops = set()`, and
`assert_root.py` explicitly treats that as a hard failure when the root contains S2a:

> *"the root contains S2a but only 0 inline-OOD ops are pre-registered (expected 8) — the exclusion
> would be vacuous"*

The deterministic selector `select_inline_ood_ops()` exists (8 ops from 8 distinct shaders, seed 42)
and its own output is stamped `"status": "operator-derived, awaiting owner ratification"`. So this is
a **live, correctly-detected pre-registration hole**, not a bug — but assert A6 will fail until the
file is written and ratified. §12.1.

**RESOLVED (A11 item 1, 2026-07-28).** The file is written and advisor-ratified. The draw was run
against the frozen S2a inventory (`outputs/ctt_v2/inventories/S2a.json`, sha
`5cd883e2add1641d…`, 799 groups / 7,990 clips), which is built from the render manifests
`meta/clips_shard*.jsonl` — **not** from the still-running VAE encodes, so nothing about the draw
waits on GPU.

| shader | pre-registered op |
|---|---|
| `BowTieWithParameter` | `BowTieWithParameter_d8b50f918c` |
| `EdgeTransition` | `EdgeTransition_1560f76ce8` |
| `FilmBurn` | `FilmBurn_bce3e2cb2d` |
| `LinearBlur` | `LinearBlur_be1e988437` |
| `Overexposure` | `Overexposure_ee3010899f` |
| `Slides` | `Slides_a8d71c73fe` |
| `morph` | `morph_07a0c1cc6a` |
| `randomNoisex` | `randomNoisex_a5f68f8fd2` |

8 ops · 8 distinct shaders · drawn from 56 eligible shaders / 799 eligible ops · seed 42 · 80 clips
(8 × 10, ~1 % of S2a). **Excluded from the assembled root, not merely held** — their encodes stay on
disk for the inline lane's demos, and per A2 inline scores never gate anything. **S2a-only**: S2b's
operators are all-new, so no S2b op shares an excluded op's id. The 8 are **complementary to** the 10
held-out shader families (H2), not overlapping — H2 is a *family*-level holdout with zero rendered
clips and stays eval-side; these are *op*-level near-OOD within trained families. Verified: **zero
overlap with H2 and zero with the full-occlusion family** — and the draw was **not** post-filtered on
either, since post-draw curation is exactly what the seed-42 procedure precludes.

**Why this is a legitimate late pre-registration, not a contaminated one:** the property being
protected is *"the trained model never saw these ops"*, which is fixed at **training** time, not
data-creation time. No training step has run and no candidate has been scored. The file carries a
verbatim timing declaration to that effect plus `source_inventory_sha256`, so the draw can be
re-verified against the exact bytes it came from forever.

### 11.5 REF_root_format.md's per-dir file count is mis-stated

It reads *"Current: 26 classes × 385 = 1,925 files per dir."* On disk each of the five dirs holds
**385** files spread over 26 class subdirs; **1,925 is the total across all five dirs** (5 × 385).

### 11.6 The two M3 pool drops did not happen as the dossier describes

DOSSIER §9 states *"This clip is in `CONTENT_POOL_union.json` and must be dropped, along with
`humanvid_10344332`."* The adjudication that followed found: `humanvid_10344332` was **never in the
pool** (it appears only in `trim_log`, removed by the diversity trim for an unrelated reason) — a
**no-op**; and `openvid_T1MiFx98l3g_0_50to156`'s whole-clip drop was **HELD** because the clip is
already rendered into 10 S2b clips, occupies the **B field in 10/10** of them, and its defect
(blank white) is confined to its **A-anchor**, which B-role never uses. The pool is **byte-unchanged
at 1,146 / 120**, and the fix is a **role-scoped A-only caption exclusion**. Gate arithmetic for the
counterfactual whole-drop was computed anyway (gate A 0.519960, gate B 50.46) and would have passed.

**RESOLVED (A11 item 5, 2026-07-28) — role-scoping KEPT, whole-clip drop REJECTED.** Dropping the
clip whole would re-cut a frozen, audited S2b batch on a post-hoc criterion (violating A10 /
principle 8), discard 10 good renders, and the defect is *provably* role-confined (A-anchor blank at
std 0.79; B-role frames 112–120 normal at std 74). What was missing was never the decision — it was
the **machine check**. Now enforced in three places, all reading the *same* derived list from
`POOL_DROPS_M3_ADJUDICATION.json` (`role_scoped_exclusions_for_caption_store`), never a hand-kept
list:

1. **`root_common.load_exclusions()`** gains `role_scoped_captions {clip_id: {roles}}` +
   `clip_level_captions`, derived; an **absent** adjudication file is a hard failure, not an empty
   exclusion, so the rule can never go silently vacuous.
2. **Caption-store assert** — `scripts/ctt_v2/captions/assert_caption_store.py`, runs when the store
   lands: `descriptions.json` must contain **no A-role** description for
   `openvid_T1MiFx98l3g_0_50to156` (presence = hard FAIL) **and must contain its B-role** one
   (absence = hard FAIL, guarding against an over-broad skip silently dropping the legitimate role).
   Both directions proven to fire. The caption pipeline consumes the same derived list to skip
   generation (`generate_descriptions.apply_role_scoped_exclusions`).
3. **Root assert (§9)** — no assembled sample's caption may draw on an excluded (clip, role)
   description. Passes **vacuously today** (the clip is B-field in 10/10 S2b rows; 0 occurrences
   anywhere in `S1_GRID.json`) and permanently guards re-renders and future grid changes.

### 11.7 S2b's blind audit has no result artefact

S2a has `outputs/videos/ctt_v2_s2/full/AUDIT_RESULT.json` with raters, adjudication and verdict.
S2b's directory holds `AUDIT_KEY.json` and `audit_sheets/` but **no `AUDIT_RESULT.json`** — the
1/64 · 0/64 · consensus-0 PASS lives only in DOSSIER §10.6 prose. Under A9's own config-archiving
ruling *(an unarchived measurement cannot gate)*, this must be written to disk before the stamp.

**RESOLVED.** `AUDIT_RESULT.json` + `AUDIT_RATERS_RAW.md` are written beside the S2b audit sheets and
mirrored to `misc/ctt_v2_final/artefacts/s2b_audit/`. §12.8 closed.

### 11.8 Minor numeric disagreements, recorded for completeness

| quantity | value A | value B | note |
|---|---|---|---|
| gate 8b, round 2 (synth vs humanvid) | **0.5518** (`gate_report_repinned.json`, the gate battery) | **0.5579** (`gate8_controls.json`, DOSSIER §9.1) | both far under the ≤0.60 bar; the gate battery's own number should be the one of record |
| S2a `m2_max_dq` | **0.4916** (refreshed `S2_ACCEPTANCE.json`) | 0.4961 (`REF_s2_pipeline.md`) | the acceptance file was re-run 2026-07-27 23:59 and now reads 7,990/799; `REF_s2_pipeline.md`'s "S2_ACCEPTANCE says 7,550/755 — STALE" is itself now stale |
| S2a overdraw | **1.2506** (refreshed) | 1.327 (`REF_s2_pipeline.md`) | same cause |
| synth endpoint bank size | **331** mp4s in `synth_endpoints/clips/` | "227 clips, blessed" (DOSSIER §1.3) | 227 = `bank_tightened.json` `n_kept` of 331 candidates. Both true; the pool draws 291 training + 20 reserved from the v1 (187) and v2 (104) banks combined |

---

## 12. CLOSED — every row now carries a ruling

Genuine gaps where no recorded decision existed when this document was written. **All eight were
ruled by `advisors/A11_seven_open_items_VERBATIM.md` (2026-07-28) plus the operator-owned 12.8**, and
each row below records the ruling and where it landed. Nothing here remains open.

| # | ruling | confidence | where it landed |
|---|---|---|---|
| **12.1** | **RATIFIED.** Pre-register the 8 inline-OOD ops NOW via the seed-42 selector; **full root exclusion** (not merely held); **S2a only**. A legitimate late pre-registration — the protected property ("the model never saw these ops") is fixed at *training* time, and no training step has run and no candidate has been scored. Draw ratified **as-is**; never post-filtered. | 0.85 | `misc/ctt_v2_final/PREREG_inline_ood_ops_s2a.json`; §11.4; assert A6 |
| **12.2** | **NO REDRAW.** A1b Q6 is superseded by a change of purpose; the two contaminated triggers are struck from the diagnostic universe and tagged `CONTAMINATED`. "Held-out effect" claims may cite only the 3 clean triggers, n=3 disclosed. Frozen selection not re-opened. | 0.85 | §7 (H4); §8.6 |
| **12.3** | **CONFIRMED: 34.5 is per S2 half ⇒ S2 total 69 %.** Code + manifest reconciled to `15 / 6 / 34.5 / 34.5 / 10`; S4 `present: true`; A9's three contingency branches recorded next to the constants. | 0.98 | `root_common.INTENDED_WEIGHTS_PCT` + `ABSENT_BRANCH_WEIGHTS_PCT`; `strata_manifest.json`; §11.1 |
| **12.4** | **RATIFY 832×448. Do not re-encode.** A1b's literal "no crop" is VAE-impossible and is amended to its intent (no resampling/letterbox/retiming). The 16-row centre crop is the *minimal* amendment; 832×480 is strictly worse on the ruling's own intent. The smoke-gate assert is rewritten with **two clauses**, derived from the trainer's own function. | 0.9 | §5.6; §8.2; §9 (D3); §11.2; `REF_mixed_length.md` |
| **12.5** | **KEEP the role-scoping; enforce by derivation + machine checks. Whole-clip drop REJECTED.** The defect is provably role-confined; dropping the clip would re-cut a frozen audited batch on a post-hoc criterion. Enforced in three places off one derived list. | 0.9 | `root_common.load_exclusions()`; `captions/assert_caption_store.py`; `generate_descriptions.py`; root assert; §11.6 |
| **12.6** | **S1's pairing group is the ARM. 1,170 stands.** group=endpoint would pair same-content × different-op, violating "reference = same operator, different content" — it is wrong, not merely different. 11 groups. A shared-probe row **may** reference a non-probe row. | 0.95 | §5.2 |
| **12.7** | **Swap-lane prompts are EXEMPT from the 12-gate battery; three per-item checks are mandatory** (render-path asserts, Tier-1 tripwire, Layer-2 accuracy with 0 unresolved-inaccurate). The gates cannot coherently apply — the register difference *is* the treatment. | 0.85 | §8.4 |
| **12.8** | **Written.** `AUDIT_RESULT.json` + `AUDIT_RATERS_RAW.md` beside the S2b sheets, mirrored to `misc/ctt_v2_final/artefacts/s2b_audit/`. | — | §11.7 |

<details>
<summary>The original questions, preserved verbatim for the record</summary>

| # | question | why it blocks | who can decide |
|---|---|---|---|
| **12.1** | **Ratify the 8 S2a inline-OOD ops.** The selector is deterministic (8 distinct shaders, seed 42) and its output is stamped *"awaiting owner ratification"*. Until `PREREG_inline_ood_ops_s2a.json` exists, assert **A6 hard-fails** and the inline-validation OOD lane has no demos. | blocks assembly | owner (it is a pre-registration) |
| **12.2** | **The S4 held-out trigger redraw.** A1b Q6 ordered it; A5 Ruling 2 mooted it by excluding S4; **A9 reinstated S4 without restating it.** Do the 5 held-out triggers stand as-is (3 clean, 2 contaminated, reported as such), or is the redraw back on? | affects only post-hoc off-cell diagnostics, not the root | advisor |
| **12.3** | **Reconcile the mix weights in code with A9** (§11.1) — and state whether A9's 34.5/34.5 is *per S2 half* (⇒ S2 total 69 %) as the ruling's prose implies. | assert A3 compares against these constants | mechanical, but it is a ruled quantity — record the reconciliation |
| **12.4** | **S4 geometry: accept the 16-row centre crop, or re-bucket?** (§11.2.) A1b ruled "no crop"; the VAE makes 464 illegal. Accepting means amending A1b's wording and re-deriving A9's σ caveat at shift **1.2350**; the alternative bucket choices each cost ~0.3 L40S-h. **The A9 smoke-gate assert `shifts ∈ {1.120, 2.302}` must be rewritten either way.** | blocks the smoke gate and the pre-written report caveat | advisor |
| **12.5** | **How is the role-scoped A-exclusion for `openvid_T1MiFx98l3g_0_50to156` enforced?** It is a caption-store rule with **no assert behind it** — the battery has no per-(clip, role) check. A silent A-role description for that clip is exactly the failure mode the campaign's own discipline says must be machine-checked. | a correctness hole in the store | operator may implement, but the *requirement* should be ruled |
| **12.6** | **S1 sample count under the pairing rule.** A1b Q5 states 1,170 (= 9×30×3 + 2×60×3), which treats the **specialist** as the pairing group. But S1's design gives each specialist 30 *unique* endpoints, so a ring-offset reference is a different endpoint's clip of the *same* specialist — i.e. same op, different content, which is correct. Confirm the group key for S1 is the **arm**, not the endpoint, and that a shared-probe-set row may reference a non-probe row. | changes the S1 sample count and the inventory schema | advisor |
| **12.7** | **Does the register-swap lane's second prompt need its own caption-battery pass?** A8 specifies it is generated by the round-3 pipeline from corpus anchors, but does not say whether those 46×2 descriptions must clear the 12 gates before use. | small, but it is caption text entering a scored lane | advisor |
| **12.8** | **What is the authoritative S2b blind-audit artefact?** (§11.7.) Under A9's config-archiving ruling an unarchived measurement cannot gate — and this one gates a FROZEN stratum. | blocks the stamp | operator writes it; the *form* is already ruled |

</details>

---

## 13. Versioning

Two things are versioned, differently and on purpose.

### 13.1 The dataset design — semver, in this file's header

| bump | when |
|---|---|
| **MAJOR** | a stratum enters or leaves the mix; the sample contract, caption grammar, or pairing rule changes. A model trained across this boundary is not comparable. |
| **MINOR** | a stratum's contents change (re-render, count change, holdout change); a new gate; a weight change. |
| **PATCH** | corrections, documentation, reproduction commands — nothing that changes a byte the trainer reads. |

`0.9.0-DRAFT` = the design is complete and the data is two-thirds built; the `-DRAFT` suffix drops
only at the stamp.

### 13.2 A build — `ROOT_MANIFEST.json`

Each assembled root carries its own manifest with the strata present, the inventories and their
sha256, the counted realized mix, the drop record with reasons, the mask store, and the filesystem
result. **A build never bumps the design version.** The design is what is versioned; a root is an
instance of it.

### 13.3 What "stamped" will mean

`ctt-v2-dataset/1.0.0` requires, in this order:

1. Every PENDING row in §1.1 resolved or explicitly ruled out of scope.
2. Every item in §12 answered by a recorded ruling.
3. §11.1, §11.2, §11.3 reconciled — code and prose agreeing with disk.
4. `assert_root.py` **and** `dryrun_epoch.py` executed against the real root, both exit 0, reports
   committed.
5. The caption store hash, generator/auditor `modelVersion` strings, raw-response archive paths and
   full battery results written into §6.
6. The counted realized mix written into §6 beside the intended mix.
7. Per-stratum σ distributions computed analytically from the root manifest and written into §8.2.
8. Owner sign-off on the amendments already outstanding: copy-gate amendment-2, the gate-#8 re-pin,
   the ≥97 % scope ruling, and the certification choice for the scoring pass.

---

## 14. Lineage

- **ctt v1** — the ladder2 `ic_gen` root: 385 pairs from 139 corpus clips, 26 classes, one stratum.
  Certified, and the incumbent D0 this round is measured against. Its defect is not a bug but a
  correlation: reference class == target class, so copying the demo's appearance was approximately
  correct for three years of gradient.
- **ctt v2 (this design)** — five strata, 1,677 groups, ≈55.5k samples. The lever is **task count**
  (Raventós pressure: ~1,600 S2 operators at 4–5 exposures each is unmemorizable, forcing a
  read-manner policy) plus **content/operator decorrelation** (reference = same operator, different
  content, enforced structurally by within-op endpoint disjointness).
- The pre-registered gate this dataset is built to be scored under was found to be **passed by a pure
  copier** (`VERIFY_base_passes_gate.md`), which is why the copy discriminator exists at all. That
  finding is the reason for this campaign, and it is recorded in `FBLOCK_DRAFT_F003.md` pending the
  owner-gated findings registry.
