# CTT v2 captions — the single source of truth

**S0 / S1 / S2 captions are DONE and LOCKED: 1,403 / 1,403. One prompt variant, one auditor,
one hash. S4 is out of scope (zero S4 descriptions exist).**

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
| S4 | **OUT** — deferred by owner |

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
storage.** The 1,404th pair is excluded by adjudication, not missing. **Nothing is short.**

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

**S4: out of scope, deferred by owner. Zero S4 descriptions exist.** Nothing here changes if S4 is
later authorized — it is an additive lane keyed the same way.
