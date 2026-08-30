# EffectData → cttv2 stratum **S6** — build authority

**Status 2026-08-28.** Goal: add EffectData as a new **one-sided breadth stratum "S6"** to the
cttv2 training root, **additively** (S0/S1/S2/S4 untouched), advisor-ratified plan of
**top-2,000 counterfactual subjects (28,644 clips), native encode (no crop), 81 frames,
prefix_latents=1 (frame-0 anchor).**

This file is the single authority for the S6 build. It mirrors, for S6, what
`misc/ctt_v2_final/REF_root_format.md` is for the root contract and what
`store/TEXT_LIFECYCLE.md` is for text. Numbers here are measured, not planned.

> **Relocated into the store 2026-08-29** — this file now lives at
> `store/datasets/004_ctt_v2plus/BUILD.md` (a symlink remains at the old
> `misc/2026-08-28_effectdata_s6/BUILD.md` path). **§8's assembled-root numbers are SUPERSEDED** by
> the 2026-08-29 code-side rebuild (138,625 → 114,215 pairs, S1 restored, S6 re-paired same-shape):
> see the sibling `./CODESIDE_FORMAT.md` and the `correction_2026_08_29` block in `./meta.yaml`.
> **§1–7** (S6 selection / shapes / encode / captions / conditions) remain the authority for how the
> S6 source was built and are unaffected by the rebuild.

---

## 1 · Selection (Axis-A counterfactual)

`data/processed/effectdata/selection_top2000.json` — the 2,000 subjects with the highest
Axis-A degree (most distinct effects on the same first frame), min degree 10.

| quantity | value |
|---|---|
| subjects | 2,000 |
| clips | 28,644 |
| distinct effects | 2,917 |
| clips/subject (p50) | 13 |
| smallest effect-group | 4 clips |

Roster frozen at `outputs/ctt_v2/encodes/EFFECTDATA/ROSTER.json` (per-clip stem/subject/effect/
video_path/w/h/latent_fhw; `n_bad=0`).

## 2 · Shapes (native, VAE-legal, no crop)

EffectData subjects come at 4 native resolutions, all 81f / 24fps, all VAE-legal (unlike S4,
whose 464-height needed a 16-row centre crop). Registered in `root_common.RULED_SHAPES` and
`assert_root_shapes.EXPECTED_SHAPE_CLASSES` (commit 117daa0), each `prefix_latents=1`:

| shape idx | WxHxF | latent_fhw (F,H,W) | tokens | clips |
|---|---|---|---|---|
| 0 | 704x1056x81 | (11,33,22) | 7986 | 7,347 |
| 1 | 704x1248x81 | (11,39,22) | 9438 | 6,774 |
| 2 | 1056x704x81 | (11,22,33) | 7986 | 6,763 |
| 3 | 1248x704x81 | (11,22,39) | 9438 | 7,760 |

(Two transpose pairs. B7 token-count collision was generalised to informational — B2 owns the
per-sample exact-shape mask, so a transpose sharing a token count is fine.)

## 3 · Encode (`latents/` + `cond_clean/`) — GPU, DeltaAI

Driver `scripts/ctt_v2/s6/encode_s6.py` (DeltaAI-native: aarch64, `src/LTX-2-official` trainer,
node-local `/tmp` clip staging). Two S6-specific facts:

- **Per-shape buckets.** A shard is single-shape; `process_videos.py` is invoked once per shape
  at that shape's exact `WxHx81`. Native==bucket ⇒ identity crop (scale 1.0, 0-px crop), the
  same trick S4 used for its width.
- **One-sided ⇒ cond_clean is a bitwise copy** (`write_cond_clean(correct_suffix=False)`), so
  the step never loads the VAE. The causal VAE makes the prefix (frame-0 anchor) clean *for
  free* — measured prefix rel-L2 8.3e-5 (`eval_ladder/encode_conditioning.py`); only the
  *suffix* of a two-sided clip needs isolation-encoding, and S6 has no suffix.

Pilot `pilot.sbatch` (job **3040828**, PASS, 2m04s): all 4 shapes correct (latent shapes match
`latent_fhw`, fps 24.0, cond_clean==latents bitwise); **no OOM on the largest shape without
`--vae-tiling`** on the GH200. Encode rate ~0.83 s/clip.

Full run `encode_array.sbatch` (job **3040833**, `--array=0-47`, 4 shapes × 12 shards, bhwp,
ghx4, ~9 GPU-h projected). Idempotent (skip-if-exists, .tmp+replace). Verify:
`python encode_s6.py verify`.

## 4 · Captions — Lane A (training text)

Shelf **`store/captions/004_effectdata`** (self-contained, per the store text lifecycle).
Per-**subject** A-descriptions (the first frame is identical across a subject's clips on Axis A,
so 2,000 descriptions serve 28,644 clips), keyed `'<subject>|A'`.

- Generator: **claude-opus-4-8 vision**, prompt variant `v2-s4f0` (S4 role-A prompt verbatim,
  spec `CAPTION_TASK.md` carried into the shelf), per-item length draw over the corpus spread.
- Fan-out: pilot (40) + 24 batches, each validated leak/length/format-clean.
- Gate (`build_caption_store.py validate`): 2000/2000 coverage, 0 format, 0 length-out-of-band,
  **0 hard leaks**; 20 state-word soft-watch (glow/beam-of-sunlight) accepted as literal still
  state per spec.
- `content_hash = sha256:4796ca7b7ebef8f7b0849ccba49ac207156fb8d6655586b354f447a66a559782`.

## 5 · Conditions — Gemma text embeds

`build_encode_inputs.py --strata S6` assembles `"{A-description}. sksz."` (one-sided grammar,
one authority: `root_common.TRIGGER_SENTENCE`) and content-addresses by `sha256(text)[:16]`:
**28,644 clips → 2,000 distinct embeds (14.32×)**, input hash `6d7e4521`. Written to the SHARED
`outputs/ctt_v2/conditions/by_caption/` tree (keyed by caption hash; no collision with S0–S4).
GPU job `conditions.sbatch` (job **3040897**, Gemma-3-12b, ~2,000 embeds, ~31.5 GB). Embed shape
matches the existing tree: `video_prompt_embeds (1024,3840) bf16`.

## 6 · Spec + inventory

`build_s6_spec.py` → `outputs/ctt_v2/inventories/S6_spec.json`: groups by effect, `sided=one`,
`endpoint_disjointness=False`, `endpoints[clip]=[subject]`, explicit
`caption_sources[clip]=[[subject,"A"]]` (so the per-subject store serves every clip without a
per-clip copy). `build_inventories.py spec` → `outputs/ctt_v2/inventories/S6.json` (2917 groups,
28,644 clips).

## 8 · DONE — the training-ready dataset (2026-08-28)

**`store/datasets/004_ctt_v2plus`** (shelf seq 004; mix-contract id `003_ctt_v2plus`). Root:
`outputs/ctt_v2/roots/ctt_v2plus_mix` — a **sampler-mix** physical root (693,125 symlinks
referencing sources; NO data moved, so it is additive over 002 and portable across clusters).

- **`samples.jsonl` 138,625 rows** — S0 385 / S2a 22,731 / S2b 23,577 / S4 6,000 / **S6 85,932**
  (S1 absent, pilot only). One row per base pair; each carries `id` + a 5-key `paths` object.
- **Mix (`mix.json` `stratum_weights_pct`): S0 12 / S2a 28.67 / S2b 29.73 / S4 9.6 / S6 20** —
  realized at train time by `StratifiedEpochSampler` (the mix is a config knob; the dataset is
  never rebuilt to change it). Contract = 002 weights ×0.80 + S6 20 (advisor 2026-08-28).
- **S6 row shape (verified):** target conditioned one-sided on its OWN frame-0
  (`cond_clean==latents`, mask `p1_onesided`); `reference` = a **same-effect, different-subject**
  demonstration; `conditions` = the per-subject caption's Gemma embed `(1024,3840)`.
- **Health:** all 5 paths/row resolve to real tensors (sampled across strata); latents
  `(128,11,H,W)` fps 24; 693,125 symlinks created / 0 replaced / 0 pruned; 0 S6 exclusion drops.
- **Pipeline:** `build_s6_spec` → `build_inventories` → `attach_captions`+`attach_conditions` →
  `normalize_inventory_paths` (/projects→/taiga) → `assemble_root --sampler-mix --contract
  003_ctt_v2plus` → `build_trainer_samples` (samples.jsonl + mix.json).

## 9 · Eval axis — seen / unseen / zero-shot is AUTOMATIC (no held-out carve)

The operator (effect) axis falls out of membership; the held-out material already exists in the
untouched rest of EffectData (full 3,061 effects / 132.8k clips, all downloaded):

| cell | operator | reference/endpoint | source |
|---|---|---|---|
| seen | in the 2,917 trained | in the 2,000 subjects | training subset |
| unseen | in the 2,917 (seen class) | subject NOT in the 2,000 | rest of ED |
| zero-shot | one of the **144** untrained effects (or external) | any | rest of ED |

So no split is carved from training (all 28,644 → training), and effect-description clauses are
NOT pre-built — the core breadth eval is demonstration-driven (neutral arm, effect comes from the
reference, not text); clauses are generated at eval time only for the chosen eval slice.

## 7 · RESOLVED — mix registration (advisor decision; realized in §8)

The mix contract (`root_common.STRATUM_WEIGHTS_PCT = {S0 15, S1 6, S2 69, S4 10}`, `MIX_STRATA`,
`ABSENT_BRANCH_WEIGHTS_PCT`) is the **certified 002_ctt_v2 contract** (rulings A5/A9/A11/A12),
guarded by sum-to-100 and set-equality asserts and a documented history of "mix drift". Adding
S6 touches this. The open question, for the advisor:

1. **Weight** for S6 (owner guidance: additive, ≤25–30%).
2. **New dataset `003_ctt_v2plus` vs. mutating the 002 globals** — reproducibility/certification
   impact. `assemble_root` reads the *global* `STRATUM_WEIGHTS_PCT`/`MIX_STRATA`, so S6 in the
   mix currently *requires* touching those globals (or parameterising them per-dataset).
3. **Re-apportionment** if weights must still sum to 100 (reduce S0–S4 pro-rata, or a separate
   additive mix).
4. A **paired-arm eval gate** (per the S2-swap advisor) before trusting S6 in training.

Steps 1–6 are done/in-flight and touch NOTHING in 002. Step 7 is the only contract-affecting
change and is held for the advisor + owner.
