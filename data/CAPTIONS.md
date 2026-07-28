# CTT v2 captions — S0 / S1 / S2 — **LOCKED**

**1,403 / 1,403 in-scope descriptions. One prompt variant, one auditor, one content hash.**

| | |
|---|---|
| lock | `outputs/ctt_v2/captions/CAPTION_LOCK.json` |
| canonical store | `outputs/ctt_v2/captions/CAPTION_STORE.json` |
| content hash | `sha256:c8e2d95bd448e4392ab9c1f55f7eed2c8a26eda46556895961e9402ab4d23396` |
| **S4** | **OUT OF SCOPE — deferred by owner. Zero S4 descriptions exist.** |
| **S1 generation** | **DEFERRED to DeltaAI.** Captions/prompts complete here. |

This document is the authority for the caption lane. Where it disagrees with the **disk**, the disk
wins. Every number was counted first-hand.

---

## 1. What was done

Five things, briefly:

1. **Fixed the requirement.** The pair-list builder read only 2 of 3 sources — S2a was missing, so
   36 (clip, role) pairs were absent from the requirement. Added S2a with hard asserts (§3.1).
2. **Generated the 213 missing descriptions** and merged them into the existing store; nothing that
   already existed was regenerated.
3. **Resolved the residual 3** that exhausted their regenerations, via operator manual rewrite,
   re-audited clean (§7.1).
4. **Ran the S0 corpus-139 Layer-2 audit** — never previously run (§6).
5. **Consolidated** 10 scattered shards into one hashed canonical store, and locked it.

---

## 2. Current state of the lock

### 2.1 Coverage — complete

| | count |
|---|---|
| requirement: union of three sources | **1,404** |
| role-scoped exclusion (§4) | −1 |
| **generatable requirement** | **1,403** |
| **in-scope descriptions present** | **1,403 → 100%** |
| orphans (paid-for, unconsumed) | 76 |
| total paid-for | 1,479 |

Coverage is measured against **1,403**, not 1,404. The 1,404th pair is **excluded by adjudication,
not missing** — nothing is short.

### 2.2 The store

Keyed `"{clip_id}|{role}"`. Three top-level maps:

- **`descriptions`** — the in-scope 1,403. **This is the only thing assembly reads.**
- **`orphans`** — 76 paid-for descriptions no current pinned grid consumes (residue of an
  `S1_GRID.json` sha drift, `eb4a88f3…` → `126fcac9…`). **Kept, never deleted**, and held *out* of
  `descriptions` so they can neither ship nor inflate coverage.
- **`provenance`** — per description: originating shard, prompt variant, generator + auditor model,
  the auditor's **echoed** model version, raw-response archive paths, acceptance attempt, word count,
  Tier-2 flags, audit verdict.

`content_hash` covers the **in-scope `descriptions` map only** (sorted keys, compact separators), so
it is stable against orphan and provenance churn. `orphans_hash` is separate.

The 10 shard directories under `outputs/ctt_v2/captions/store/` remain on disk as the **audit
trail** — the canonical file is what consumers read.

### 2.3 Homogeneity — the load-bearing assert

| | |
|---|---|
| prompt variant | **`v2`, single** (asserted, not coincidence) |
| generator | `gemini-3.6-flash` (temp 0.7, thinkingLevel `minimal`, maxOutputTokens 120) |
| Layer-2 auditor | **`gemini-3.5-flash-lite`, single** (temp 0, thinkingLevel `minimal`, maxOutputTokens 512) |

**Never mix prompt variants in one store** — gate 8a names *mixed prompts* as a bug class it
detects. One shard (`final3`) used `v3` + a pro auditor but **accepted 0 descriptions**, which is why
homogeneity holds: nothing v3-generated or pro-audited ever entered.

Two deviations, recorded:

- The directive named `gemini-3.5-flash` as auditor. It returns **HTTP 503** and is unavailable
  (verified on probe). `flash-lite` is two-sided validated (mismatch 99.74% ≥99, matched 2.0% ≤10).
- 🔒 **No-switch-back.** This is **one** recorded instrument change. If `gemini-3.5-flash` returns,
  **do not revert.**
- 🔒 **First-pass rates are NOT a like-for-like series** across any auditor change. No trend claims.

### 2.4 Residual loop — closed

**Unresolved `inaccurate` in the final store = 0** (A8's HARD bar). Three pairs needed operator
manual rewrite; all three re-audited clean (§7.1). The pre-registered **content-borne leak rule was
never invoked** — 0 `leak=YES` in the final pass, so no clip was dropped.

### 2.5 Tier-2 review queue

**111 of 1,403 (7.9%)** carry a Tier-2 flag (class-word / shader-escape / drift). These are
**recorded, not auto-rejects** — a house on fire is a house on fire. Policy is 100% operator review.
The certified corpus's own base rate is 13.5%, so ~8% is unremarkable. Full pair list is in the lock.

### 2.6 Spend — measured, not estimated

**This session:** 609 calls, **185,089 tokens ≈ 58.6 TRY** (at the ledger's 316.79 TRY/M; ≈68.7 TRY
at a 371 TRY/M reference). Breakdown: gap213 generation 220 calls/103,948 tok; gap213 audit
215/44,901; manual3 audit 3/640; corpus-139 audit 171/35,600.

**Campaign cumulative:** 5,515 calls / 2,825,223 tokens; 895 TRY invoice ⇒ **316.79 TRY per million
as an UPPER bound** (unarchived calls also contributed, so the true rate is lower). Production rate:
**682 tokens per finished description**.

⚠ `cost_ledger.py` applies **one blended rate to every model** and therefore **cannot price a
model-choice decision** (understates pro ~8×, overstates flash-lite ~2×).

---

## 3. The requirement: 1,404 from THREE sources

| source | file | distinct (clip, role) | sha256 |
|---|---|---|---|
| **S2a** | `outputs/videos/ctt_v2_s2/full/meta/clips_shard*.jsonl` (20 shards, 7,990 records) | **454** | `52fab84f477229d0…`¹ |
| **S2b** | `experiments/exp_082_s2_humanvid/PLAN_S2_UNION.json` (800 pairs) | **1,217** | `f58765acde984ac3…` |
| **S1** | `misc/ctt_v2_final/S1_GRID.json` (390 rows: 270 one-sided, 120 two-sided) | **400** | `126fcac9cf56bdd6…` |
| | **union = the TRUE requirement** | **1,404** | |

¹ S2a has no single plan file, so its sha is a **rollup**: `sha256` over sorted
`"<shard name> <shard sha256>"` lines. The 20 per-shard shas are in `mass_pairs_manifest.json`.

By role: A 750 / B 653. By bank: humanvid 836 / synth_endpoints 567. 968 distinct clips.

### 3.1 ⚠ The S2a schema trap — keep these asserts

**S2a's endpoints exist only in its rendered metadata, keyed `A` / `B` — not
`endpoint_a` / `endpoint_b`.** A strict `endpoint_a` lookup returns an **empty set** (verified: 0 of
454), so the bug does not look like a bug — it looks like *"S2a needs no descriptions."*

`build_mass_pair_list.py` now enforces:

1. **positive presence** — every record must carry non-empty `A` and `B`; a missing key is
   instrument failure, never a skipped row;
2. **non-empty** — a zero-pair S2a contribution is a hard stop;
3. **derived constant** — the set must equal `S2A_EXPECTED_CLIP_ROLES = 454`, recomputed from the
   shards; mismatch raises **SPEC-CONSTANT-MISMATCH**, which escalates and **never** selects a
   fallback branch;
4. **trap witness** — the vacuous `endpoint_a` lookup runs on purpose and its `0` is recorded;
5. **all three** source schemas must be present, or the build refuses.

---

## 4. The one exclusion

`openvid_T1MiFx98l3g_0_50to156` is excluded in role **A only** — its opening anchor is a blank
screen, defective for prefix use; its B-role window is fine and kept. Authority: A10 (whole-clip drop
**denied**), sidecar `POOL_DROPS_M3_ADJUDICATION.json` sha `91686fb8127d…`. Standing rule: **defects
are dispositioned at the unit of consumption — (clip, role) — not of storage.** The exclusion is
*derived* through the same loader the generator and `assert_root.py` use, so channels cannot drift.

⚠ **Adding S2a made this exclusion live.** With two sources it matched nothing and was a reported
no-op; S2a uses the clip as an A endpoint.

---

## 5. Keying and grammar — why (clip, role)

A rendered clip is a **splice**: frames `0–8` are byte-identical to one pool clip's opening window,
frames `112–120` to a *different* pool clip's closing window, shader transition in between.

| role | frames | describes |
|---|---|---|
| **A** | `0–8` | the opening anchor |
| **B** | `112–120` | the closing anchor |

A whole-clip description would be **wrong about one end by construction**. Roles anchor to *frames*,
never to grid position.

One description per **(clip, role)**, never per sample: a clip used as the A endpoint of 40 rows gets
exactly **one** A-role description. This is why the store is 1,403 and not ~5,600 — and it is also a
*style* guarantee: because the identical string is shared across every row consuming that endpoint,
the only style contrast the stream can carry is S0-vs-rest, never S1-vs-S2 or bank-vs-bank (gate 8b
measures exactly this).

```
one-sided : "{A-role description}. sksz."
two-sided : "{A-role description}. sksz. {B-role description}."
```

`sksz` is the trigger and **the sole carrier of the transition**. Endpoint descriptions describe
*endpoints*; mentioning the change, onset, or effect is a **leak** — it turns the caption into a soft
trigger token. Stored without trailing period; A-role uppercase-initial, B-role lowercase-initial
(the corpus's own registers: A a finite sentence, B a participial NP). `render_prompt` /
`precompute.py` are the **only** assemblers.

---

## 6. The S0 corpus-139 Layer-2 audit

139 certified captions → **171 audit units** (139 A-side + 32 B-side), each against its own byte-pure
9-frame anchor. Report: `outputs/ctt_v2/captions/CORPUS_139_LAYER2_AUDIT.json`.

**🟢 `leak=YES`: 0 / 171.** This is the question that mattered: **no endpoint description leaks the
transition effect.**

**🟡 `inaccurate=YES`: 4 / 171 (2.3%).** Positive-presence control satisfied (171/171 verdicts
parsed, 0 errors) — so the zero-leak result is a real zero, not a shrunken denominator.

🛑 **These captions are CERTIFIED and remain BYTE-IDENTICAL. All four hits ESCALATE to the owner;
nothing was edited.** `audit_corpus_139.py` is read-only by construction — no `--fix`, no rewrite
path, no store output. Verbatim:

| clip\|role | auditor's stated error |
|---|---|
| `polygon_1\|A` | "The man is wearing a tan baseball cap with a 'C' logo, not a brown one as typically described or implied, but the main error is the cap color or style detail." |
| `portal_2\|A` | "The character is not wearing glasses." |
| `sakura_petals_0\|A` | "The visor is not purple, it is black with reflections."; "The astronaut is standing on a concrete rooftop edge, not a metal framework of a skyscraper." |
| `water_bending_1\|B` | "The man is described as having curly hair, but his hair is short and straight or slightly wavy, not curly." |

**Operator reading (not a ruling):** all four are fine-grained perceptual attribute errors — cap
logo, eyewear, visor colour, hair texture. That is the class A8 §4 placed **outside** the ≥97% bar's
scope ("generator perception, not prompt"), and none describes change, onset, or effect. Owner
adjudicates.

---

## 7. Filters

- **Format (hard):** single sentence, no trailing period, A uppercase-initial / B lowercase-initial,
  no `sksz`, no markup, 8–70 words.
- **Audio words (hard, gate 6):** the LTX-2 audio layer is banned (corpus 0/171). Visible speech
  *actions* ("talks", "singing") are legitimate visual content — tracked, **not** gated.
- **Tier-1 (hard):** banned leak strings, zero tolerance.
- **Tier-2 (recorded):** §2.5.
- **Content-borne leak rule (pre-registered):** a `leak=YES` reproducing across regeneration
  escalates to an operator view of the 9-frame window. Describes **byte-pure visible content** ⇒ mark
  `leak_content_borne`, **keep**, log. Describes **change or onset** ⇒ drop the clip.
  **Never invoked — 0 leaks.**

### 7.1 The 3 operator rewrites

| pair | why | fix |
|---|---|---|
| `humanvid_7086282\|A` | both attempts tripped `audio:music` ("sheet **music** … **music** stand") | → "printed sheets on a black metal stand" |
| `humanvid_8134899\|A` | attempt 1 tripped `sound`, attempt 2 `audio` ("**audio** mixing console") | → "a mixing console" |
| `openvid_XMr5MkOHB5o_2_59to276\|A` | `inaccurate=YES` twice on one attribute — auditor: dress is navy, not white | → "sleeveless **navy** dress" |

Each hand-edited **minimally** from its own last generated attempt, re-validated through the
identical mechanical filters, and **re-audited on the production path** — all clean.
`manual_rewrite.py` is deliberately separate: a hand-written string must never enter the store on a
path mistakable for a generated one. Each carries its rationale and full pre-rewrite history.

🔒 Hand-editing to move a **distributional gate** remains **banned**. These edits fix auditor-named
content errors and mechanical format violations only.

---

## 8. Gate results on the record

⚠ **Scope, stated plainly: the 12-gate battery has been run on a POOLED 447-row SUBSET
(`chunk1` + reused in-grid pilot rows), NOT on the final 1,403-row store.** The owner directed that
no new gates be run, so this is reported as-is and the gap is **named**. Re-running is CPU-only and
costs no API budget: `gate_battery.py --store <shard…> --restrict-to-grid mass_pairs.json`.

`outputs/ctt_v2/captions/store/chunk1/gate_chunk1_plus_reused.json`, n=447 — **`hard_fail: []`**:

| gate | value | bar | verdict |
|---|---|---|---|
| 1 word-count p50 | 34.0 (corpus 33) | p50 ∈ [29, 36] | PASS |
| 2 p10 / p90 | 24.0 / 41.0 | [16,26] / [34,44] | PASS |
| 3 determiner (A/B) | 98.6% / 98.1% | ≥86.4% / ≥86.9% | PASS |
| 4 B lowercase / A uppercase | 100% / 100% | both 100% | PASS |
| 5 B participial-NP | 100.0% | ≥80.6% | PASS |
| 6 audio / markup | 0 / 0 | 0 | PASS |
| 7 exact-dup within stratum | 0.0% | <2% | PASS |
| **8a corpus-vs-new (DRIFT GUARD)** | **0.7099 ± 0.0091 SE** (fold-std 0.0454, 25 fits) | **≤0.73 HARD** | **PASS** |
| **8b stratum-internal blindness** | **0.5787 ± 0.0126 SE** (NULL 0.506) | **≤0.60 HARD** | **PASS** |
| **8c original absolute bar** | 0.7099 | ≤0.65 | **FAIL — recorded, never quietly replaced** |
| 9 full-vocab classifier | AUC **0.9124** (bacc 0.8314) | ≥0.80 ⇒ investigate | **INVESTIGATE → ACCEPT** |
| 10 colour density | 3.87 (corpus 3.158) | [1.579, 4.737] | PASS |
| 11 camera-phrase rate | 3.58% (corpus 3.51%) | ±10 pp | PASS |
| 12 near-dup (Jaccard >0.8) | 0 pairs | report only | REPORT |

### 8.1 What 8a means — a drift guard, not a style bar

The original **≤0.65 is recorded as FAILED** and is **never** quietly replaced. It was mis-pinned as
an absolute number before the anchors existed: the corpus's own two registers, same captioner,
separate at **0.6419**, and a mere prompt delta *within the same model* opens **0.7233** — so the bar
sat in an unreachable reference frame. The residual (~0.69–0.71 with every marginal statistic matched)
is a **captioner-generation fingerprint**, not a fixable style defect.

**8a's function is a drift guard: a value above 0.73 cannot be the known fingerprint and means a BUG
— mixed prompts, wrong round's store, contamination ⇒ stop and investigate, do NOT tune.** The
load-bearing replacement is **8b**, which guarantees the only style cue in the stream is S0-vs-rest.

Why acceptance is safe: S0 samples are themselves demo-following samples on corpus content, so
"old register ⇒ S0 mode" at eval routes toward *baseline-like* behaviour on old-register prompts —
which is what **all** eval prompts are. The shortcut therefore **suppresses** new-data capability at
eval; it is a **false-FAIL** mechanism, not a false-PASS one. The false-pass channel is separately
gated by `copy_ref` and both cross-liveness controls.

### 8.2 Gate 9 accepted — the signal is content

Per A8's pre-committed rule (content-dominated ⇒ accept), ~33 of the 40 top features are **content**:

- **toward CORPUS:** `sunglasses −1.05`, `sky −0.95`, `stands −0.88`, `the −0.85`, `warm −0.82`,
  `on −0.81`, `young −0.70`, `is −0.66`, `golden −0.58`, `night −0.56`, `sunlight`, `trousers`,
  `city`, `under`, `of`, `street`, `concrete`, `slicked`, `skirt`, `sits`
- **toward NEW:** `short 1.24`, `daylight 1.10`, `brown 1.06`, `light 1.01`, `room 0.89`, `lit 0.86`,
  `white 0.85`, `shirt 0.82`, `indoor 0.82`, `skinned 0.80`, `hair`, `soft`, `brightly`, `wooden`,
  `with`, `gray`, `fair`, `by`, `holds`, `t`

The few function words (`the`, `on`, `is`, `under`, `of`, `with`, `by`) are the already-known gate-8
fingerprint. A person-only bank against a 26-class VFX corpus is *expected* to be content-separable.

### 8.3 Standing prohibitions

- 🔒 **Gate 8a/8b movement may NEVER be cited as evidence an intervention worked.** They are
  pass/fail drift guards only. Read movement against **SE-of-mean (≈0.011)**, not the fold-level fit
  std.
- ⛔ **Nobody may record that the be-verb tell was "fixed".** The `v3` intervention permitted plain
  `is/are` on the A-role and the realized rate stayed **0.0%**; A8's motivating 8.8% was pooled and
  mis-attributed (corpus A-role **5.0%**, B-role **25.0%**), so the deficit is chiefly a **B-role**
  phenomenon v3 did not touch. 8a's 0.7066 → 0.6929 is ~**1.2 SE** = noise, **not attributable**.
  The residual B-role tell is an **accepted-and-recorded risk**; **no round 4**; injecting be-verbs
  by hand is banned.
- 🔒 **`v3` is an archived paid-for negative result and enters no store.** It also ran `--no-audit`,
  so it has no Layer-2 audits and cannot gate. Production = `v2`.
- ⛔ **Audits are ALWAYS per-item, never packed** (packed audits failed independently: deranged flag
  rate 96% < 99%, positional attribution 85%).

---

## 9. S1 and S4

**S1 captions are COMPLETE and in scope** — all 400 S1 endpoint (clip, role) descriptions are in the
store. **S1 *generation* is DEFERRED to DeltaAI**, along with the blind Gemini 11-way class-ID batch
gate (top-1 **≥80%** vs 9% chance, with the real-corpus control arm for instrument validation).

Handoff artifact: `misc/ctt_v2_final/S1_GRID_deltaai.json` — **390/390 rows with prompts embedded**,
sha `dea8ffe436998e99…`. Prompts must be embedded or the run hits the
`has no prompt -- rebuild the grid` assert, which is **correct behaviour and must not be weakened**.

🔑 **The S1 class-ID gate is consequential:** if it FAILS, the pre-registered branch **drops S1
entirely** and the mix renormalises to **S0 15 / S2 85**. That is a legitimate pre-registered
outcome, not a failure of the caption work.

**S4: OUT OF SCOPE, deferred by owner. Zero S4 descriptions exist.** If S4 misses the assembly
cutoff, the pre-registered branch sets the mix to **S0 15 / S1 6 / S2 79**.

---

## 10. Files

**There is one master artifact set. Everything else is archive.**

`outputs/ctt_v2/captions/` — the master:

| path | what |
|---|---|
| `CAPTION_STORE.json` | **canonical store** — the only thing assembly reads |
| `CAPTION_LOCK.json` | the lock |
| `mass_pairs.json` | the 1,403 generatable requirement |
| `mass_pairs_manifest.json` | three source shas + the S2a trap witness |
| `CORPUS_139_LAYER2_AUDIT.json` (+ `_raw_responses.jsonl`) | S0 audit + verbatim hits |
| `COST_LEDGER.json` | measured spend |

`outputs/ctt_v2/captions/archive/` — **audit trail only; nothing here is read by assembly:**
the 10 per-shard store dirs (`store/chunk1-4`, `gap213`, `manual3`, `s1_regrid`, `s2a_gap`, `tail`,
`final3`) with their raw generation/audit response archives; the pilot rounds
(`pilot_m3`, `pilot_m3_round2`, `pilot_m3_round3`); the auditor validation lanes
(`auditor_v31pro`, `auditor_v3flash`, `auditor_v3flash_matched`); the per-chunk pair lists; and
`V2_PROMPT_IDENTITY.json`.

Scripts, `scripts/ctt_v2/captions/`:

| path | what |
|---|---|
| `build_mass_pair_list.py` | requirement builder (3 sources, S2a asserts) |
| `generate_descriptions.py` | generation + per-item Layer-2 audit |
| `manual_rewrite.py` | operator residual loop |
| `consolidate_store.py` | shards → canonical hashed store |
| `audit_corpus_139.py` | S0 corpus-139 audit (read-only) |
| `gate_battery.py` | the 12-gate battery |
| `cost_ledger.py` | measured spend |

### 10.1 Where these files live — and why `git merge` was never going to move them

`outputs/` is **gitignored**, so merging `ctt-v2` into `main` moved the **code** and could never move
the **artifacts**. Git has no merge operation for files it was told to ignore.

This repo's existing answer to that is **symlinks**, not git: in a worktree, nearly every `outputs/`
subdirectory (`analysis`, `videos`, `training`, `signals`, `logs`, …) is a symlink back to the main
tree, so both trees read and write one physical copy.

`outputs/ctt_v2/` was an **exception** to that convention, and that was the actual defect. CTT v2
artifacts were split across two unlinked directories:

- main tree: `encodes/`, `masks/`, `smoke/`
- `ctt-v2` worktree: `captions/`, `inventories/`, `roots/`, `s1/`, and two manifests

**Fixed** — the worktree's subdirectories were moved into the main tree's `outputs/ctt_v2/` (no
collisions) and the worktree now holds a symlink, matching the convention used by every other
`outputs/` subdirectory. One physical copy, reachable from both trees; verified by identical
`sha256` of `CAPTION_STORE.json` and `CAPTION_LOCK.json` read through the symlink.

Two related facts:

- `misc/ctt_v2_final/` (the DOSSIER, advisor verbatim files, `S1_GRID*.json`) is **outside the repo
  entirely** — never merged, never committed. That is deliberate: it is the campaign's durable record.
- `/projects/illinois/.../diffusion-research` and `/taiga/illinois/.../diffusion-research` are the
  **same directory** (identical inode) — two mount paths for one filesystem. `git worktree list`
  prints the `/taiga` form; that is cosmetic, not a second copy.

All paths in `CAPTION_LOCK.json` are **repo-relative**, so the lock does not depend on any worktree
existing. The content hash covers `descriptions` only, so it is **unchanged** by both the archive
reorganisation and this relocation (verified before and after each).
