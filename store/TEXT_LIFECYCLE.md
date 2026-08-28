# The text lifecycle — store-resident authority

**What this is.** The single self-contained account of *every text artifact* in the project —
from captioning → prompt preparation → training → generation → evaluation — anchored in the
store. Read this to understand where each piece of text lives, what produces it, what consumes
it, and which hashes pin it. Where this disagrees with disk, **disk wins** (as in `data/CAPTIONS.md`).

> Companion contracts: `store/README.md` (shelf contract), `store/ARMS.md` (arm registry),
> `store/captions/README.md` (the description shelf). Deep reasoning record (not state):
> `misc/ctt_v2_final/DOSSIER.md`.

---

## 0. The one thing to understand: there are TWO text lanes

They share a surface grammar (`{A}. sksz. {B}.`) and coincide only at S0. Everything else about
them — where the text lives, what assembles it, what reads it — is different.

```
  LANE A — TRAINING captions (leak-free; the model must READ the operator from the demo)
    A/B/S4 descriptions ─► assembled "{A}. sksz. {B}." ─► Gemma embed (content-addressed)
      (captions shelf)        (build_encode_inputs.py)      conditions/by_caption/<sha16>.pt
                                                                     │ trainer reads via samples.jsonl.caption_key
                                                                     ▼   TRAINING

  LANE B — EVAL / GENERATION prompts (neutral vs effect; tests demonstration-dependence)
    corpus captions ─► render_prompt ─► neutral grid "{S1}. sksz. [{S2}.]"   ─┐
      (eval corpus)     (prompts.py)     + effect grid  "{S1}. sksz. {EFFECT}. [{S2}.]"
                                         frozen in store/prompts/ (sha-pinned) │ stamp_rows.py
                                                                     arm registry ─► GENERATION ─► EVAL
```

**Training never contains effect text** — naming the effect is a Tier-1 *leak*. Effect text
appears only in Lane B's *effect* grid, to test whether naming the effect can substitute for the
demonstration. That asymmetry is the whole point of the task.

---

## 1. Vocabulary

| term | meaning |
|---|---|
| **endpoint** | the source content an operator acts on (its byte-exact anchor frames) |
| **role A / B** | A = the opening anchor (frames 0–8); B = the closing anchor (frames 112–120) |
| **A / B description** | one leak-free sentence describing role A / role B — the training text unit |
| **EFFECT_DESC** | a leak-*ful* clause naming the effect/transition — used ONLY in eval effect prompts |
| **sksz** | the trigger token between the two sides; the sole textual carrier of "a transition happens" (2 Gemma subwords) |
| **one- / two-sided** | one-sided = only role A conditioned (`{A}. sksz.`); two-sided = both (`{A}. sksz. {B}.`) |
| **neutral / effect prompt** | eval prompt without / with the `{EFFECT}` clause |

Keying: **one description per `(clip, role)`, never per sample** — a clip used as endpoint A in 40
rows has exactly one A-description. That is why the S0/S1/S2 store is 1,403, not ~5,600.

---

## 2. Lane A — training captions

### 2.1 Sources (the description stores)

| store | keyed | coverage | content hash | lock | generator |
|---|---|---|---|---|---|
| `outputs/ctt_v2/captions/CAPTION_STORE.json` (S0/S1/S2) | `<clip>\|<role>` | 1,403/1,403 | **`c8e2d95b…`** | `CAPTION_LOCK.json` LOCKED, `single_prompt_variant=v2` | gemini-3.6-flash (temp 0.7, 120 tok); auditor **gemini-3.5-flash-lite**, no-switch-back |
| `outputs/ctt_v2/captions/S4_CAPTION_STORE.json` (S4, role A only) | `<stem>\|A` | 2,000/2,000 | **`34534e47…`** (the SHIPPED store) | separate store (variant `v2-s4f0`) | **gemini-3.6-flash**, per-item length draw — regenerated from the earlier claude-sonnet store `fcd46f33…` (now archived) to fix gate 2 |

### 2.2 The prompt (variant `v2`) and the length target

One-sentence description of a 9-frame snippet. A-role = a finite sentence (uppercase-initial);
B-role = a lowercase participial noun phrase. **Hard prohibitions** (the leak fence): no language
about the scene changing/transforming/beginning/ending/shifting/revealing; **no visual-effect,
editing, or animation-style name**; no sounds/music/speech; no referring to video/frames as
objects; plain literal colours ("red dress", not "vibrant red dress"). Verbatim template:
`scripts/ctt_v2/captions/generate_descriptions.py` (`_PROMPT_A_TEMPLATE` / `_PROMPT_B_TEMPLATE`,
`v2` adds a real corpus exemplar + appositive-comma + plain-colour clauses).

**Length** (calibrated to a 139-caption corpus, `M1_corpus_caption_stats.txt`): per-endpoint
pooled p10/p50/p90 = **21 / 33 / 39**, drawn per-item from the 171-value empirical list; hard
format bar **8–70 words**. S4 variant `v2-s4f0` = `v2` role-A with "a 9-frame snippet" → "a single
still frame".

### 2.3 Assembly → conditions the trainer reads

- **Assemblers (NOT `render_prompt`):** `scripts/ctt_v2/captions/build_encode_inputs.py::assembled_for()`
  (S1/S2a/S2b) and `assemble_s4_captions.py` (S4). Grammar built by string-concat with
  `root_common.TRIGGER_SENTENCE = " sksz."` (`root_common.py:229`):
  `one-sided → "{A}. sksz."`, `two-sided → "{A}. sksz. {B}."`. S0 uses its certified caption with the
  outcome marker replaced by ` sksz. ` (not literally verbatim — a literal marker would fail the
  marker-absent check in `caption_violations()`).
- Every assembled caption is validated by `root_common.caption_violations()`: exactly one ` sksz.`,
  outcome marker absent, zero Tier-1 leak strings (`caption_common.LeakFilter`).
- **Content-addressed:** a caption is a function of its endpoints (the shader is unnameable), so
  clips sharing an endpoint pair share a caption. Key = **`sha256(caption)[:16]`**; the Gemma embed
  lands at **`datasets/ctt_v2/conditions/by_caption/<sha16>.pt`** (S0 → `conditions/s0_corpus/<class>/`).
- **What the trainer reads:** `datasets/ctt_v2/samples.jsonl` rows carry `caption_key` (= the sha16)
  and `paths.conditions = conditions/by_caption/<caption_key>.pt`. Reached through the store as
  `store/datasets/002_ctt_v2/root/conditions/…` (root → `datasets/ctt_v2`). `captions.json`
  (`caption_key → text`, 3,453 rows) is provenance/debug only — the trainer never parses text.
- **Gates:** the 12-gate distributional battery, `hard_fail: []` (`GATE_BATTERY_FULL_1403.json`).

---

## 3. Lane B — eval / generation prompts

### 3.1 The renderer

`eval_ladder/prompts.py::render_prompt(clip, sided, token)` is the **only** neutral renderer
(store contract rule 5). It loads a corpus caption, splits it on the leak marker
**`"The scene transforms into "`** into `(S1, S2)`, and emits:
`one-sided → "{S1}. {token}."`, `two-sided → "{S1}. {token}. {S2}."`. It has **no effect branch.**

### 3.2 The effect clause

`{EFFECT}` is spliced **downstream** of `render_prompt`, from **`misc/refvfx_baseline/reference_effects.json`**
(one clause per reference; house style ≈ 20 words, subject-generic "the subject…", transition-describing).
This is the only place effect text legitimately enters a prompt.

### 3.3 Frozen families in the store (`store/prompts/`) — FOUR pinned sources

| entry | grammar | rows | `prompt_corpus_sha` | source |
|---|---|---|---|---|
| `001_ctt152_neutral` (canonical) | `{S1}. sksz. [{S2}.]` | 152 | `0d708175fbfe` | `eval_ladder/registry_ctt_v2.jsonl` |
| `002_ctt152_effect` (canonical) | `{S1}. sksz. {EFFECT}. [{S2}.]` | 152 | `35930d7d7453` | `store/gens/005_ctt_v2_leaky/grid.jsonl` |
| `007_ctl_probe` (special) | controllability probe | 76 | `dfcbb07b926a` | non-001/002 |
| `008_ext112_authorcfg` (special) | author-config 2-channel | 112 | `73787305eb4a` | non-001/002 |

Rows are **arm-free**; `eval_ladder/stamp_rows.py` derives arm-stamped registries (recorded
`derived:` shas, e.g. `strip_sksz`, `swap_token_refvfx`). A gen pins `prompt_family` + `prompt_sha`
and stores the exact rows in its own `grid.jsonl`. The generator reads the stamped registry — never
`store/prompts/` directly.

---

## 4. Where the lanes meet: S0 (and S1's s0cf layer)

S0's certified corpus captions are shared by both lanes: **training** replaces their outcome marker
with ` sksz. `; **eval** renders from the same 222-clip corpus, split on that marker. Corpus-derived
text also enters training via **S1's s0cf layer** (it draws on the certified 139), so the lanes are
not strictly disjoint apart from S0. Everywhere else they are independent artifacts.

---

## 5. Store layout — current and target

**Current:** the store owns only Lane B (`store/prompts/`). All Lane-A description text lives
OUTSIDE the store in gitignored `outputs/ctt_v2/captions/` + `datasets/ctt_v2/`. This doc's
restructure closes that gap.

**Target (this is what "self-contained in the store" means):**

```
store/
├── TEXT_LIFECYCLE.md              ← this file (the authority)
├── captions/                      ← NEW shelf: SOURCE descriptions, git-tracked text
│   ├── README.md
│   ├── 001_ctt_v2_endpoints/      A/B (clip,role) descriptions   [byte-copy, hash c8e2d95b — LOCKED]
│   ├── 002_ctt_v2_s4/             S4 first-frame A descriptions   [hash 34534e47]
│   ├── 003_effect_clauses/        EFFECT_DESC clauses            [from reference_effects.json]
│   └── 004_effectdata/            EffectData (S6) A descriptions, per SUBJECT   [hash 4796ca7b — DONE 2026-08-28]
├── prompts/                       ← flat ids preserved; a `lane:` field carries the gen_eval/training split
│   ├── 001_ctt152_neutral   (lane: gen_eval)   002_ctt152_effect (lane: gen_eval)
│   ├── 007_ctl_probe (lane: gen_eval)          008_ext112_authorcfg (lane: gen_eval)
│   └── 0NN_ctt_v2_train_corpus  (lane: training)  {A}. sksz. {B}. over the full corpus   [NEW]
└── datasets/002_ctt_v2/root/conditions/   ← the embeds the trainer reads (already store-resident)
```

Rationale for a `lane:` field (not physical `gen_eval/` + `training/` subdirs): gens pin prompt
ids as `prompts/002_ctt152_effect`; repathing would break every consumer + need MIGRATION. A
one-line `lane:` in each meta separates the two logically, non-breakingly.

---

## 6. Hashes & locks (preserve byte-for-byte)

| value | pins | authoritative source |
|---|---|---|
| `sha256:c8e2d95bd448e4392ab9c1f55f7eed2c8a26eda46556895961e9402ab4d23396` | locked S0/S1/S2 `descriptions` map | `CAPTION_STORE.json` |
| `sha256:34534e47c291c95211e7ee5a4c9308bdc80db0aab99833ca90b0191663c5f658` | S4 description store (disk-authoritative) | `S4_CAPTION_STORE.json` |
| `0d708175fbfe` / `35930d7d7453` | prompts 001 neutral / 002 effect corpus | `store/prompts/00{1,2}/meta.yaml` |
| `dfcbb07b926a` / `73787305eb4a` | prompts 007 / 008 corpus | `store/prompts/00{7,8}/meta.yaml` |
| `1a086911…` / `7f4709097a…` / `5c33ab1b…` | ctt_v2 samples.jsonl / captions.json / mix.json | `datasets/ctt_v2/MANIFEST.json` |
| source shas `52fab84f`(S2a) `f58765ac`(S2b) `126fcac9`(S1) | caption requirement provenance | `CAPTION_LOCK.json` |
| model pins: gen `gemini-3.6-flash`; auditor `gemini-3.5-flash-lite` (no-switch-back) | caption provenance | `CAPTIONS.md §5` |

---

## 7. Reconciliations (known discrepancies this doc settles)

1. **S4 store provenance:** the shipped store `34534e47…` is a **gemini-3.6-flash** regeneration
   (variant `v2-s4f0`, per-item length draw over the 171-value corpus list), source
   `outputs/ctt_v2/captions/s4_gemini/`. The earlier **claude-sonnet** store `fcd46f33…` is ARCHIVED
   at `outputs/ctt_v2/captions/archive/s4_gate_measurements/`. Training consumed the **Gemini**
   captions. `CAPTIONS.md §12/§12.4` still describe the sonnet store and state "no caption was ever
   regenerated" — **STALE vs disk**; the regen's authorization is recorded only in the store's own
   `generator` field (no CHANGELOG/DOSSIER decision) — flagged to the owner.
2. **Two assemblers vs "one renderer":** `render_prompt` owns Lane B *only*. Lane A is assembled by
   `build_encode_inputs.py` / `assemble_s4_captions.py`. The "one renderer" rule is about eval prompts.
   (`eval_ladder/prompts.py`'s module docstring still claims it is the one place a prompt is made for
   "training AND inference" — stale ladder2-era text, superseded here.)
3. **`render_prompt` has no effect clause:** the effect grid's `{EFFECT}` is spliced from
   `reference_effects.json` downstream — "single renderer" covers the neutral skeleton only.
4. **Mix drift:** `datasets/ctt_v2/mix.json` rules **S0 15 / S1 6 / S2a 33.87 / S2b 35.13 / S4 10**,
   but the shipped champion `store/runs/002_ctt_v2` trained **S0 5 / S1 12 / S2a 34.36 / S2b 35.64 /
   S4 13**. Any "what the champion saw" claim must cite the run's mix, not mix.json.
5. **29-clip consumption gap — RESOLVED** (not pending): A16 dropped-and-recorded the 29 S2a clips at
   consumption (CHANGELOG 2026-07-28 13:45); `samples.jsonl` has zero S2a rows using
   `openvid_T1MiFx98l3g_0_50to156` as an A endpoint. `CAPTIONS.md §4.2`'s "OPEN" flag is stale.
6. **S4 gate attribution:** the only FORMAL battery (`GATE_BATTERY_S4.json` — 8a 0.8849, gate 2 FAIL)
   measured the ARCHIVED **sonnet** captions. The shipped **gemini** regen fixed gate 2 (width now
   inside the [16,26]/[34,44] bars); its 8a is comparison-measured ≈0.8913 (still FAIL) with no formal
   battery of its own. So "S4 FAILS 8a at 0.8849 + width" is wrong for the shipped store on both numbers.

---

## 8. Adding a stratum — EffectData worked example (one-sided)

> **EXECUTED 2026-08-28 as stratum S6.** Full build authority: `misc/2026-08-28_effectdata_s6/BUILD.md`.
> The plan below was followed with these DIVERGENCES worth flagging:
> - Lane A landed as a **single subject-keyed store** `store/captions/004_effectdata/EFFECTDATA_CAPTION_STORE.json`
>   (schema `ctt_v2_s6_caption_store/v1`, keyed `<subject>|A`, hash `4796ca7b`), matching the S4 store
>   schema — not an `A/` subdir. Each clip reaches it via explicit `caption_sources=[[subject,"A"]]`.
> - **EFFECT_DESC (step 2) and eval grids (step 4) are DEFERRED** — eval-only, not on the training path.
> - **Caption QA gap vs S4 (OPEN, for advisor):** the 2,000 captions passed a generic leak/length/format
>   gate (`build_caption_store.py validate`: 0 hard leaks, 0 format/length failures) plus each fan-out
>   batch's own self-validation. The **formal S4 gate battery — the blind-guess gate and the 100%
>   Layer-2 audit tripwire (§8.1) — was NOT run.** The captioner *did* surface the failure mode it
>   guards against (mislabeled/hybrid animal ids) and described visible content instead, but that is not
>   the formal gate. Whether S6 needs S4's full battery before training is an advisor question.

The template is S4. For EffectData's chosen subset (advisor-recommended **top-2,000 counterfactual
subjects**, additive, S2 kept):

1. **A descriptions** → `store/captions/0NN_effectdata/A/` — first-frame scene descriptions,
   **leak-free** (effect name withheld, `v2-s4f0` prompt), **one per subject** (subjects share their
   start frame ⇒ ~2,000, not per-clip). Vision captioning (Opus 4.8), S4-style batching + the 12 gates.
2. **EFFECT_DESC** → `store/captions/0NN_effectdata/effect/` — one clause per effect, trimmed from
   EffectData's own **`instruction_en`** (the per-effect canonical ~26-word description; `prompt_en`
   is per-clip and NOT used) into the `reference_effects.json` house style (~20 words, "the subject…").
   Text-only (Opus 4.8), no leak concern (eval `{EFFECT}` is *meant* to name the effect).
3. **Training corpus** → append `{A}. sksz.` rows to `store/prompts/0NN_ctt_v2plus_train_corpus`,
   assembled by `assembled_for()`, content-addressed → new dataset conditions.
4. **Eval grids** (optional) → `store/prompts/0NN_effectdata_{neutral,effect}` via `render_prompt`
   + the EffectData effect clauses.
5. **Dataset** → `store/datasets/0NN_ctt_v2plus` (new one-sided stratum + its conditions); the
   trainer reads `conditions/` exactly as today.

**Never in training:** EffectData's `prompt_en`/`vfx_en`/`abstract_en` — they name the effect and
are Tier-1 leaks (identical situation to refVFX's 42 trigger phrases, withheld under
`effect_of_clip_NOT_FOR_CAPTIONING`).

### 8.1 Non-obvious requirements (learned from S4 — do not skip)

- **The integration unit is an INVENTORY, not a caption store.** `assembled_for()` reads
  `outputs/ctt_v2/inventories/<stratum>.json` (clips / groups / sided / `caption_sources`, with an
  inline-caption override). A new stratum must first exist as an inventory (built via
  `build_inventories.py` + a `<stratum>_spec.json`, cf. `s4/build_s4_spec.py`) — that is the actual
  wiring step behind "assembled by `assembled_for()`".
- **The per-item length draw is MANDATORY.** Draw each caption's target length by
  `rng.choice(empirical)` over the 171-value corpus list — never a single fixed target. A fixed
  target is exactly what made the *sonnet* S4 captions fail gate 2 (p10/p90 collapsed to 27/34); the
  *gemini* regen fixed it by restoring the per-item draw. Applies to whatever model captions EffectData.
- **One-sided masks are shape-ruled.** The mask family is `masks/f<F>_h<H>_w<W>_p<prefix>_onesided.pt`,
  where `prefix = root_common.prefix_latents((F,H,W))`. EffectData's shapes (F=11) are **unruled** →
  `DEFAULT_PREFIX_LATENTS = 2` would condition 9 video frames INTO the effect onset (the exact trap S4
  avoided with `prefix_latents=1` = frame-0-only). Every EffectData shape MUST be added to
  `RULED_SHAPES` with `prefix_latents: 1` before staging (owner-gated, as S4's was), or
  `assemble_root` raises on the unruled shape. `assert_root_shapes.py` B3 (two-shape hardcode) and B7
  (no token-count collision — fails on transpose pairs like 704×1248 vs 1248×704) must be generalized first.
- **Vision-captioned strata need S4's extra gates**, not just the 12-gate battery: the blind-guess
  gate and the 100% Layer-2 audit tripwire (`DATASET.md` C3) — a captioner writing from memory can
  pass every lexical/length gate while describing the wrong clip.
