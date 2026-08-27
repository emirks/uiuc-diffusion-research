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
    A/B/S4 descriptions ─► assembled "{A}. sksz. {B}." ─► T5/Gemma embed (content-addressed)
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
| `outputs/ctt_v2/captions/S4_CAPTION_STORE.json` (S4, role A only) | `<stem>\|A` | 2,000/2,000 | **`34534e47…`** (disk-authoritative; `CAPTIONS.md §12`'s `fcd46f33…` is STALE) | separate store (variant `v2-s4f0`) | Claude vision, 25 batches × 80 |

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
  `one-sided → "{A}. sksz."`, `two-sided → "{A}. sksz. {B}."`. S0 uses its certified caption verbatim.
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

## 4. Where the lanes meet: only S0

S0's certified corpus captions are the sole text shared by both lanes (training uses them verbatim;
eval renders from the same 222-clip corpus, split on the marker). Everywhere else the lanes are
independent artifacts with independent provenance.

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
│   └── 0NN_effectdata/            EffectData A (per subject) + effect_desc (per effect)   [FUTURE]
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

1. **S4 store hash:** disk `34534e47…` is authoritative; `CAPTIONS.md §12`'s `fcd46f33…` is stale.
2. **Two assemblers vs "one renderer":** `render_prompt` owns Lane B *only*. Lane A is assembled by
   `build_encode_inputs.py` / `assemble_s4_captions.py`. The "one renderer" rule is about eval prompts.
3. **`render_prompt` has no effect clause:** the effect grid's `{EFFECT}` is spliced from
   `reference_effects.json` downstream — "single renderer" covers the neutral skeleton only.
4. **Mix drift:** `datasets/ctt_v2/mix.json` rules **S0 15 / S1 6 / S2a 33.87 / S2b 35.13 / S4 10**,
   but the shipped champion `store/runs/002_ctt_v2` trained **S0 5 / S1 12 / S2a 34.36 / S2b 35.64 /
   S4 13**. Any "what the champion saw" claim must cite the run's mix, not mix.json.
5. **29-clip consumption gap:** 29 rendered S2a clips reference the A-role of the excluded
   blank-anchor pair `openvid_T1MiFx98l3g_0_50to156`; no cross-role fallback; owner decision pending
   (`CAPTIONS.md §4.2`). Changes no count/hash.
6. **S4 ships past a pre-registered gate:** S4 captions FAIL gate 8a (0.8849 vs ≤0.73) and the width
   gate; shipped by explicit owner decision (`CAPTIONS.md §12.4`). Any report citing S4 carries this.

---

## 8. Adding a stratum — EffectData worked example (one-sided)

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
