# CTT v2 captions — the single source of truth

**S0 / S1 / S2 captions are DONE and LOCKED: 1,403 / 1,403. One prompt variant, one auditor,
one hash. S4 is now DONE too: 2,000 / 2,000 first-frame descriptions in a SEPARATE store —
see §12.**

This file is the authority for the caption lane. Where it disagrees with the disk, the disk wins.
`misc/ctt_v2_final/DOSSIER.md` §9 and §20–§26 are the *reasoning record*, not the state — read them
for **why**, read this for **what**.

| | |
|---|---|
| canonical store | `outputs/ctt_v2/captions/CAPTION_STORE.json` |
| content hash (the `descriptions` map) | `sha256:c8e2d95bd448e4392ab9c1f55f7eed2c8a26eda46556895961e9402ab4d23396` |
| lock | `outputs/ctt_v2/captions/CAPTION_LOCK.json` |
| coverage | **1,403 / 1,403 = 100 %** |
| prompt variant | **`v2`**, single (asserted) |
| Layer-2 auditor | **`gemini-3.5-flash-lite`**, single across all 1,403 |
| unresolved `inaccurate` | **0** |
| distributional battery | **full 1,403-row store, `hard_fail: []`** |
| S1 *generation* | deferred to DeltaAI → `misc/ctt_v2_final/deltaai_s1_handoff/` |
| S4 | **DONE, separate store** — 2,000 / 2,000, `S4_CAPTION_STORE.json`, hash `sha256:fcd46f3308f9f52f…` (§12) |

## 1. Files — one master set, everything else is archive

`outputs/ctt_v2/captions/`

| file | what |
|---|---|
| `CAPTION_STORE.json` | **the store. The only thing assembly reads.** |
| `CAPTION_LOCK.json` | the lock: coverage, hashes, models, gates, spend |
| `mass_pairs.json` | the 1,403 (clip, role) requirement |
| `mass_pairs_manifest.json` | the three source shas + the S2a trap witness |
| `GATE_BATTERY_FULL_1403.json` | the 12-gate battery on the full store |
| `CORPUS_139_LAYER2_AUDIT.json` (+ `_raw_responses.jsonl`) | the S0 audit |
| `COST_LEDGER.json` | measured spend |
| `archive/` | **audit trail only, read by nothing**: the 10 per-shard store dirs with 31 `raw_*.jsonl` response archives, the 3 pilot rounds, the auditor-validation lanes, the per-chunk pair lists |

Scripts, `scripts/ctt_v2/captions/` — `build_mass_pair_list.py` (requirement),
`generate_descriptions.py` (generate + per-item audit), `manual_rewrite.py`,
`consolidate_store.py`, `audit_corpus_139.py`, `gate_battery.py`, `cost_ledger.py`.

⚠ `outputs/` is **gitignored**, so `git merge` moved the code and could never move these
artifacts. One physical copy exists, in the main tree; the worktree reaches it by symlink like
every other `outputs/` subdir. All lock paths are repo-relative.

## 2. Keying: one description per **(clip, role)** — and why a whole-clip caption would be wrong

A rendered clip is a **splice**. Frames `0–8` are **byte-identical** to one pool clip's opening
window; frames `112–120` are byte-identical to a *different* pool clip's closing window; the shader
transition is in between. The byte-identity is asserted, not assumed —
`assert1: mae_pure_a = mae_pure_b = 0.0`.

| role | frames | describes |
|---|---|---|
| **A** | `0–8` | the opening anchor |
| **B** | `112–120` | the closing anchor |

So a single whole-clip description would be **factually wrong about one end by construction**, and
measurably so: the two windows of one clip sit at **median rel-L2 0.402** against a **0.620**
between-different-clips control. The two ends of one clip are ~**65 %** as different as two
unrelated clips, and only **6 %** of clips have ends that are essentially identical. Roles anchor to
*frames*, never to grid position.

**One description per (clip, role), never per sample.** A clip used as the A endpoint of 40 rows has
exactly **one** A-role description. That is why the store is 1,403 and not ~5,600 — and it is also a
*style guarantee*: because the identical string is shared by every row consuming that endpoint, the
only style contrast the stream can carry is S0-vs-rest, never S1-vs-S2 or bank-vs-bank. Gate 8b
measures exactly that.

## 3. Grammar

```
one-sided : "{A-role description}. sksz."
two-sided : "{A-role description}. sksz. {B-role description}."
```

`sksz` is the trigger **between the two sides** and the sole carrier of the transition. Endpoint
descriptions describe *endpoints*; mentioning the change, onset or effect is a **leak** — it turns
the caption into a soft trigger token. Stored without a trailing period; A-role uppercase-initial
(a finite sentence), B-role lowercase-initial (a participial NP) — the corpus's own two registers.
`render_prompt` / `precompute.py` are the **only** assemblers.

## 4. The requirement: 1,404 from THREE sources → 1,403 generatable

| source | file | (clip, role) | sha256 |
|---|---|---|---|
| **S2a** | `outputs/videos/ctt_v2_s2/full/meta/clips_shard*.jsonl` (20 shards, 7,990 records) | 454 | `52fab84f477229d0…`¹ |
| **S2b** | `experiments/exp_082_s2_humanvid/PLAN_S2_UNION.json` (800 pairs) | 1,217 | `f58765acde984ac3…` |
| **S1** | `misc/ctt_v2_final/S1_GRID.json` (390 rows) | 400 | `126fcac9cf56bdd6…` |
| | **union = the true requirement** | **1,404** | |

¹ S2a has no single plan file, so its sha is a rollup over sorted `"<shard> <sha256>"` lines; the 20
per-shard shas are in `mass_pairs_manifest.json`.

By role A 750 / B 653. By bank humanvid 836 / synth_endpoints 567. 968 distinct clips.

**Minus one role-scoped exclusion = 1,403 generatable**, all 1,403 present.
`openvid_T1MiFx98l3g_0_50to156` is excluded in role **A only** (its opening anchor is a blank
screen); its B-role window is fine and kept. Authority A10 — a whole-clip drop was **denied**.
Standing rule: **defects are dispositioned at the unit of consumption — (clip, role) — not of
storage.** The 1,404th pair is excluded by adjudication, not missing: **the store is not short
against its requirement.**

### 4.2 🔴 OPEN — 29 rendered S2a clips consume the excluded pair. Owner decision, not a caption bug.

The store is complete *against the requirement*, but the requirement **excluded a pair that 29
already-rendered S2a clips still reference**. Verified first-hand: of the 7,990 S2a rendered
records, **29 use `openvid_T1MiFx98l3g_0_50to156` as their A endpoint** and therefore need the
A-role description that A10 deliberately withheld. Its **B-role is present**, 0 records use it as a
B endpoint, and **S2b (0 pairs) and S1 (0 rows) are unaffected** — this is S2a-only.

This is the **same shape as the §4.1 defect**: correct at the requirement level, wrong against what
was actually rendered. It changes **no count and no hash** — 1,403/1,403 stands, `c8e2d95b…` stands —
because it is a *consumption-side* gap, not a store gap. What it means concretely is that at
assembly those 29 clips have no A-role caption available.

⚠ **There is no cross-role fallback and one must not be invented.** Substituting the B-role text, or
any other clip's text, would caption a blank-screen anchor with content it does not show — the exact
failure A10's exclusion exists to prevent. The options are the owner's:

1. **Drop the 29 clips** from the S2a roster at assembly (29 / 7,990 = **0.36 %** of S2a) — cheapest,
   costs nothing but 29 samples, and is consistent with A10.
2. **Overturn A10's role-A exclusion** for this clip — note A10 denied a *whole-clip* drop precisely
   because the A window is defective, so this reopens an adjudicated question.
3. **Re-render the 29** against a different A endpoint — correct but costs render time.

Recorded rather than papered over, and **not** decided here.

### 4.1 🔴 The S2a defect — recorded so nobody reintroduces it

`build_mass_pair_list.py` originally read only **two** of the three sources.
**S2a's endpoints live only in its rendered metadata, keyed `A` / `B` — not
`endpoint_a` / `endpoint_b`.** A strict `endpoint_a` lookup therefore returns an **EMPTY set**
(verified: 0 of 454), and the bug does not look like a bug — **it reads as the reassuring
*"S2a needs no descriptions."*** That cost **36 absent (clip, role) pairs**, 26 of which were at
genuine risk of never being generated at all.

Now enforced, and these asserts must stay:

1. **positive presence** per record — non-empty `A` and `B`; a missing key is instrument failure,
   never a skipped row;
2. **non-empty** — a zero-pair S2a contribution is a hard stop;
3. **derived constant** — the set must equal `S2A_EXPECTED_CLIP_ROLES = 454`, recomputed from the
   shards; a mismatch raises **SPEC-CONSTANT-MISMATCH**, which escalates and **never** selects a
   fallback branch;
4. **trap witness** — the vacuous `endpoint_a` lookup is run on purpose and its `0` is recorded;
5. **all three** source schemas present, or the build refuses.

Adding S2a is also what made the §4 exclusion *live*: under two sources it matched nothing and was
correctly reported as a no-op.

## 5. Models, and the auditor churn

| | |
|---|---|
| generator | `gemini-3.6-flash` (temp 0.7, thinkingLevel `minimal`, maxOutputTokens 120) |
| Layer-2 auditor | `gemini-3.5-flash-lite` (temp 0, thinkingLevel `minimal`, maxOutputTokens 512) |

The auditor identity **moved four times** as availability shifted:

```
gemini-3-pro-preview → gemini-3.5-flash → gemini-3.1-pro-preview → gemini-3.5-flash-lite
```

- The **pro tier is RETIRED / 404, not rate-limited.** That corrects an earlier operator claim of
  429/rate-limiting: it is a different failure with a different remedy — there is no waiting it out.
- `gemini-3.5-flash` returns **HTTP 503** — unavailable, verified on probe, still.
- `gemini-3.5-flash-lite` carries a two-sided validation: mismatch **99.74 %** (bar ≥99), matched
  **2.0 %** (bar ≤10). It audited **all 1,403** — one auditor across the whole store.
- 🔒 **No switch-back.** This is *one* recorded instrument change. If `gemini-3.5-flash` returns,
  **do not revert.** (`gemini-3-flash-preview` was separately certified — mismatch 220/220, matched
  0.52 % — and is the pin for *future* work; switching mid-store would have fragmented audit
  provenance across the very set that carries the claim.)
- 🔒 **First-pass rates are NOT a like-for-like series** across any of these changes. No trend
  claims; any report must carry the per-round auditor identity.

## 6. Prompt variant: `v2`, and why v3 was not merged

Production is **`v2`**, single, asserted. A session directive named **v3**; it was deliberately not
followed, because merging v3 text into a 1,190-row v2 store would

1. violate the **never-mix-prompts** pin,
2. trip the exact bug class **gate 8a exists to detect**, and
3. cost ~290 TRY to chase a delta measured at **~1.2 SE of noise**.

v3 is an **archived, paid-for negative result** and enters no store. Its be-verb mechanism is
**falsified** (realized A-role rate 0.0 % → 0.0 %); A8's motivating 8.8 % was pooled and
mis-attributed (corpus A-role 5.0 %, B-role 25.0 %), so the deficit is chiefly a **B-role**
phenomenon v3 never touched. One shard (`final3`) did fire v3 at 3 production rows and accepted
**0** — so homogeneity survived by a Layer-1 rejection rather than by design, and all three of those
rows were A-role, i.e. sitting exactly on v3's only delta. **No round 4. Nobody may record that the
be-verb tell was "fixed". Hand-injecting be-verbs is banned.**

## 7. Filters and the residual loop

- **Format (hard):** one sentence, no trailing period, A uppercase- / B lowercase-initial, no
  `sksz`, no markup, 8–70 words.
- **Audio words (hard):** the LTX-2 audio layer is banned (corpus 0/171). Visible speech *actions*
  ("talks", "singing") are legitimate visual content — tracked, not gated.
- **Tier-1 (hard):** banned leak strings, zero tolerance.
- **Tier-2 (recorded, not auto-reject):** **111 of 1,403 = 7.9 %**, policy 100 % operator review.
  The certified corpus's own base rate is 13.5 %, so ~8 % is unremarkable. List is in the lock.
- **Unresolved `inaccurate` = 0** (A8's hard bar). Three pairs exhausted their regenerations and
  were **operator hand-edited** — minimally, from their own last attempt, re-filtered and re-audited
  clean on the production path: `humanvid_7086282|A` and `humanvid_8134899|A` (both tripped the
  audio-word filter: "sheet **music**", "**audio** mixing console") and
  `openvid_XMr5MkOHB5o_2_59to276|A` (auditor: the dress is navy, not white). `manual_rewrite.py` is
  a deliberately separate path so a hand-written string can never enter the store on a path
  mistakable for a generated one. 🔒 Hand-editing to move a **distributional gate** stays banned.
- **Content-borne leak rule (pre-registered): never invoked** — 0 leaks in the final pass, so no
  clip was dropped.
- **Orphans: 76** paid-for descriptions no current pinned grid consumes (residue of an
  `S1_GRID.json` sha drift `eb4a88f3…` → `126fcac9…`). **Kept, never deleted**, held *out* of
  `descriptions` so they can neither ship nor inflate coverage.

## 8. The S0 corpus-139 Layer-2 audit

139 certified captions → **171 audit units** (139 A-side + 32 B-side), each against its own
byte-pure 9-frame anchor.

- 🟢 **`leak=YES`: 0 / 171.** The question that mattered: **no endpoint description leaks the
  transition effect.**
- 🟡 **`inaccurate=YES`: 4 / 171 (2.3 %)** — `polygon_1|A` (cap logo), `portal_2|A` (eyewear),
  `sakura_petals_0|A` (visor colour + rooftop), `water_bending_1|B` (hair texture).
- Positive-presence control satisfied (171/171 verdicts parsed, 0 errors), so the zero is a **real
  zero**, not a shrunken denominator.
- 🛑 **These captions are CERTIFIED and stayed BYTE-IDENTICAL. All four ESCALATE to the owner;
  nothing was edited.** `audit_corpus_139.py` is read-only by construction — no `--fix`, no store
  output. All four are fine-grained perceptual attribute errors, the class A8 §4 placed *outside*
  the ≥97 % bar's scope; none describes change, onset or effect. Owner adjudicates.

## 9. The 12-gate battery — full 1,403-row store

`outputs/ctt_v2/captions/GATE_BATTERY_FULL_1403.json` · **`hard_fail: []`** · **zero API spend**
(gate 5 in its pinned regex form, identical to the earlier run). Same bars, 3.1× the denominator;
no gate was added, retuned or reinterpreted. The battery input is **provably the locked store**: the
pooled in-grid shard set is text-identical to `CAPTION_STORE.json`'s `descriptions` and re-hashes to
`c8e2d95b…`.

| gate | full store (n=1,403) | 447-row subset | bar | verdict |
|---|---|---|---|---|
| 1 word-count p50 | 34.0 | 34.0 | ∈ [29, 36] | PASS |
| 2 p10 / p90 | 24.0 / 40.0 | 24.0 / 41.0 | [16,26] / [34,44] | PASS |
| 3 determiner A / B | 97.9 % / 96.3 % | 98.6 % / 98.1 % | ≥86.4 / ≥86.9 | PASS |
| 4 B lowercase / A uppercase | 100 % / 100 % | 100 % / 100 % | both 100 % | PASS |
| 5 B participial-NP | 99.7 % | 100.0 % | ≥80.6 % | PASS |
| 6 audio / markup | 0 / 0 | 0 / 0 | 0 | PASS |
| 7 exact-dup in stratum | 0.0 % | 0.0 % | <2 % | PASS |
| **8a corpus-vs-new (drift guard)** | **0.6819 ± 0.0113 SE** | 0.7099 ± 0.0091 | **≤0.73** | **PASS** |
| **8b stratum-internal (load-bearing)** | **0.5950 ± 0.0075 SE** | 0.5787 ± 0.0126 | **≤0.60** | **PASS** |
| 8c original absolute bar | 0.6819 | 0.7099 | ≤0.65 | **FAIL — recorded, never replaced** |
| 9 full-vocab classifier | AUC **0.8963** | 0.9124 | ≥0.80 ⇒ investigate | INVESTIGATE → ACCEPT |
| 10 colour density | 3.811 | 3.87 | ∈ [1.579, 4.737] | PASS |
| 11 camera-phrase rate | 1.71 % | 3.58 % | ±10 pp of 3.51 % | PASS |
| 12 near-dup (Jaccard >0.8) | 0 pairs | 0 pairs | report only | REPORT |

**What moved, stated plainly:**

- **8a fell** 0.7099 → 0.6819 (−0.0280, ≈2.5 SE) — *away* from the bar. Above 0.73 would have meant
  a bug (mixed prompts, wrong store, contamination), and the pre-agreed response was **stop, do not
  tune**. It did not fire.
- **8b rose** 0.5787 → 0.5950 (+0.0163, ≈2.2 SE) — *toward* the bar. It PASSES, but headroom shrank
  from 0.0213 to **0.0050 (~0.67 SE)**. 8b is the **load-bearing** gate; expect it to be the one
  that moves first under any future change to the store.
- 🔒 **Neither movement is evidence of anything.** 8a/8b are pass/fail **drift guards** and their
  movement may **never** be cited as an intervention working. Read movement against the
  **SE-of-mean**, never the fold-level fit std.

**8a's meaning.** The original **≤0.65 is recorded as FAILED** and never quietly replaced — it was
mis-pinned as an absolute number before the anchors existed: the corpus's own two registers, same
captioner, separate at **0.6419**, and a mere prompt delta *within one model* opens **0.7233**, so
the bar sat in an unreachable reference frame. The residual is a **captioner-generation
fingerprint**, not a fixable style defect. Acceptance is safe because the shortcut *suppresses*
new-data capability at eval (every eval prompt is old-register), making it a false-**FAIL**
mechanism, not a false-pass one; the false-pass channel is gated separately by `copy_ref` and both
cross-liveness controls.

**Gate 9 accepted** on A8's pre-committed content-dominated rule, with the **identical** reading on
the full store: **33 of 40** top features are content, and the 7 function words (`the`, `on`, `is`,
`of`, `under`, `in`, `while`) are the already-known gate-8 fingerprint. A person-only bank against a
26-class VFX corpus is *expected* to be content-separable.

⛔ **Audits are ALWAYS per-item, never packed** (packed audits failed independently: deranged flag
rate 96 % < 99 %, positional attribution 85 %).

## 10. Spend — measured, not estimated

From `cost_ledger.py` over archived `usageMetadata`:

| | calls | tokens |
|---|---|---|
| final session | 609 | 185,089 |
| **campaign cumulative** | **5,515** | **2,825,223** (2,593,780 prompt / 231,443 output) |

Against the **895 TRY** invoice ⇒ **316.79 TRY per million tokens, as an UPPER bound** (unarchived
calls also contributed to that invoice, so the true rate is lower). Final session ≈ **58.6 TRY**.
Production shape: **682 tokens per finished description** (generation + audit).

⚠ `cost_ledger.py` applies **one blended rate to every model** and therefore **cannot price a
model-choice decision** (understates pro ~8×, overstates flash-lite ~2×).

## 11. S1 and S4

**S1 captions are complete** — all 400 S1 endpoint (clip, role) descriptions are in the store.
**S1 *generation* is deferred to DeltaAI**, and the blind 11-way class-ID gate goes with it because
that gate scores *generated* clips, which exist only there. Run book, grid, adapter shas and
preflight: **`misc/ctt_v2_final/deltaai_s1_handoff/`** (`S1_GRID_deltaai.json`, 390/390 prompts
embedded, sha `dea8ffe436998e99…`). 🔑 If that gate **fails, S1 drops** and the mix renormalises —
a legitimate pre-registered outcome (A5 Ruling 3), **not** a failure of the caption work.

**S4 is DONE — see §12.** It is a separate store on purpose: the locked store's homogeneity assert
is *one prompt variant*, and S4's differs in one clause (§12.2). Merging them would break that
assert for no gain, since assembly reads per-stratum inventories anyway.

---

## 12. S4 — 2,000 first-frame descriptions

| | |
|---|---|
| store | `outputs/ctt_v2/captions/S4_CAPTION_STORE.json` |
| content hash | `sha256:fcd46f3308f9f52f…` |
| assembled training captions | `outputs/ctt_v2/captions/S4_CAPTIONS_ASSEMBLED.json` (`sha256:1e9ecd0350d89ad7…`) |
| coverage | **2,000 / 2,000 = 100 %** |
| keying | **`<stem>\|A`** — role A only |
| prompt variant | **`v2-s4f0`** (§12.2) |
| generator | Claude Sonnet vision, 25 fan-out batches × 80 clips |
| spec | `outputs/ctt_v2/captions/S4_CAPTION_SPEC.md` |
| Tier-1 leaks | **0** · format violations **0** · key collisions with the locked store **0** |
| Tier-2 review flags | 185 (9.25 %) — review-only, same policy as the locked store's 111 |
| cost | **0 ₺** — no Gemini calls |

### 12.1 One description per clip, not two — because only frame 0 is conditioned

S4 is **one-sided**. The owner's 2026-07-28 decision is to condition on **video frame 0 alone**,
which is **latent frame 0 alone** (`prefix_latents((5,14,26)) == 1`). So:

- there is no B endpoint and no suffix sentence: the assembled caption is `{description}. sksz.`
- every clip is **its own A endpoint** (`endpoints[c] == [c]`), unlike S2 where the endpoints are
  two *other* pool clips spliced into a render
- the description covers **exactly** the conditioned pixels. Under the earlier fixed 2-latent
  prefix it would have described 9 frames of a 33-frame transition — a third of the clip, most of
  which the model is supposed to invent

That last point is why the prefix width is now a **shape property** (`root_common.prefix_latents`)
rather than a literal `m[:2]` repeated at six call sites.

### 12.2 The prompt delta, and why the stores stay separate

`v2-s4f0` is prompt `v2` role-A **verbatim**, with one clause changed: *"a 9-frame snippet"* →
*"a single still frame"*. That is forced by §12.1 — asking for a 9-frame description of a 1-frame
conditioning would be asking for text about pixels the model never sees.

The locked store asserts `single_prompt_variant`. Adding `v2-s4f0` to it would flip that assert to
false, so S4 gets its own store and its own hash. Assembly reads per-stratum inventories, so
nothing downstream needs them merged.

### 12.3 The source dataset's captions could not be edited

refVFX ships **one trigger phrase per effect** — 42 phrases over 2,000 clips
(`0rb4it 360 degree orbit`, `3xp105ion huge explosion`, …). That is a class label, not a per-clip
description, and every one of them is a **Tier-1 leak string**: naming the effect in the caption
hands the model the transition it is supposed to invent. So the descriptions are generated from
pixels. The effect label is withheld from the captioner and stored only under
`effect_of_clip_NOT_FOR_CAPTIONING`.

### 12.4 Measured register gap vs the corpus — disclosed, not gated

| statistic | S4 (n=2,000) | locked store role-A (n=750) | Δ |
|---|---|---|---|
| words p50 | 30 | 34.5 | **−4.5** |
| words p10 / p90 | 27 / 34 | 26 / 41 | narrower |
| commas / description | 2.111 | 2.288 | −0.18 |
| colour terms / description | 3.103 | 3.997 | **−0.89** |

The word and colour deltas are close in size to the round-1 failure that lost gate #8 (§9). They
are **reported, not gated**, for a reason: gate #8 asks whether a discriminator can separate corpus
captions from new ones, which is diagnostic only when both describe **the same footage domain**. S4
is AI-generated single-subject VFX footage — overwhelmingly close-ups on plain or bokeh backgrounds
— against DAVIS/HumanVid street and park scenes. A studio headshot genuinely contains fewer
distinct coloured objects than a city street, so part of that −0.89 is content, and closing it
would mean instructing the captioner to invent colour that is not in the frame. That is a worse
defect than the gap.

What *would* have been a real confound was checked and is absent: **per-effect word p50 spans only
28 – 33.5** across all 42 effects. Batches were round-robined over an effect-sorted roster
precisely so a captioner's style could not track an effect.

### 12.5 Alignment was the real risk, and it was spot-checked

Two of the 25 captioners self-reported catching a stem↔image indexing drift mid-draft (writing
captions from memory after reading a whole block) and both re-read every frame in small groups
before writing. A drift like that survives every length, comma and lexeme check — the captions are
individually well-formed and simply describe the wrong clip.

So it was checked directly: **10 (stem, caption) pairs across 8 batches, frames re-read and
compared. 10 / 10 exact**, including a wall shadow, an orange-dyed fringe and a dusk horizon band.
