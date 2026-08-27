# `captions/` shelf — the source description text

The **single canonical home for description text**: the A/B endpoint descriptions, the S4
first-frame descriptions, and the EFFECT_DESC clauses. Git-tracked (the source text was
previously only in gitignored `outputs/`). This shelf holds *source text*; the *rendered*
prompts live in `prompts/`, and the *embeds the trainer reads* live in `datasets/…/conditions/`.

**Authority for the whole flow:** [`../TEXT_LIFECYCLE.md`](../TEXT_LIFECYCLE.md).

## Entries

| id | role | keyed | count | hash / lock |
|---|---|---|---|---|
| `001_ctt_v2_endpoints` | A/B descriptions (S0/S1/S2), **Lane A training text** | `<clip>\|<role>` | 1,403 | `descriptions` map `sha256:c8e2d95b…` — **LOCKED** (`single_prompt_variant=v2`) |
| `002_ctt_v2_s4` | S4 first-frame A descriptions (one-sided), **Lane A** | `<stem>\|A` | 2,000 | `sha256:34534e47…` (disk-authoritative) — variant `v2-s4f0`, ships past gate 8a by owner decision |
| `003_effect_clauses` | EFFECT_DESC clauses, **Lane B EVAL ONLY** | reference id | 36 | `reference_effects.json` (append-only) |

## The two lanes (see TEXT_LIFECYCLE.md)

- **Lane A (training):** `001` + `002` → assembled `{A}. sksz. {B}.` by
  `build_encode_inputs.py::assembled_for()` → content-addressed embeds
  `datasets/…/conditions/by_caption/<sha16>.pt` the trainer reads. **Leak-free** — effect text
  never enters training.
- **Lane B (eval/gen):** `003` supplies `{EFFECT}`, spliced into `prompts/002_ctt152_effect`
  (`{S1}. sksz. {EFFECT}. [{S2}.]`). Effect text is *meant* to name the effect here — that is the
  demonstration-dependence test.

## Provenance & integrity

Each entry is a **byte-copy** of its locked source (`outputs/ctt_v2/captions/*` /
`misc/refvfx_baseline/reference_effects.json`); the copy's hash was verified equal to the lock at
registration. The originals remain the pipeline's current read-path; consolidating them to
symlinks INTO this shelf (so the store is the sole physical home, per the store contract) is a
planned follow-up — until then the store copy is the **tracked canonical** and the two are
byte-identical (the sources are immutable/locked, so they cannot drift).

## Adding a stratum (EffectData)

A new `0NN_effectdata/` entry will hold:
- **`A/`** — first-frame scene descriptions, leak-free (`v2-s4f0`), **one per subject** (~2,000);
- **`effect/`** — one clause per effect, trimmed from EffectData's `instruction_en` into the
  `003` house style.

`prompt_en`/`vfx_en`/`abstract_en` are **never** used for training A-descriptions — they name the
effect (Tier-1 leak). See `../TEXT_LIFECYCLE.md §8`.
