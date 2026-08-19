# CTT v2 — counterfactual structure (per tier)

Measured 2026-08-19 from `root/samples.jsonl` (consumed clips) and, for S2, the generation
meta `outputs/videos/ctt_v2_s2{,_humanvid}/full/meta/clips_shard*.jsonl` (true endpoint =
`pair_id`/`A,B`, not the rendered-clip name).

**Definitions**
- **operator** = the transition applied (S0/S1 class or manner · S2 shader+params `op_id` · S4 refVFX trigger).
- **endpoint** = the source content the operator acts on (its byte-exact anchor frame(s)).
- **demonstration axis** = same operator × different endpoint (reference pairing). All tiers have this.
- **counterfactual axis** = **same endpoint × different operator**. This is what the table below measures.
- **operators per endpoint** = the counterfactual degree (1 ⇒ no counterfactual).

## Summary

| tier | operators | endpoints | ops/endpoint (mean) | % endpoints ≥2 ops | fidelity | counterfactual |
|---|---|---|---|---|---|---|
| **S0+S1** | 26 (11 applied) | 334 | 3.76 (7.17 on S0's reals) | 42% (95% of S0 reals) | byte-exact | yes — creative manners |
| **S2** | 1,590 (56 families) | 1,100 | 14.0 (S2a 23.8) | **100%** | byte-exact | **yes — parametric, dense** |
| **S4** | 42 | 2,000 | 1.00 | 0% | — | **none** |

Training pairs: S0 385 · S1 3,675 · S2a 22,731 · S2b 23,577 · S4 6,000 = 56,368.

---

## Tier S0 + S1 — real corpus + spec counterfactuals

S0 is real transitions (one operator per clip, **no** counterfactual). S1 re-renders other
manners onto S0's own endpoints, manufacturing the counterfactual S0 lacks.

| | S0 alone | S1 alone | S0+S1 (on S0's 139 real endpoints) |
|---|---|---|---|
| operators | 26 | 11 applied | 26 total / 11 applied |
| endpoints | 139 | 334 | 139 |
| ops/endpoint (mean) | **1.00** | 3.62 | **7.17** (median 9, max 10) |
| % endpoints ≥2 ops | 0% | 37.4% | **95.0%** (77% ≥3) |
| endpoints/operator (mean) | 5.35 | 109.8 | — |
| fidelity | — | byte-exact | byte-exact |

- All 139 S0 contents are reused as S1 endpoints; S1's 11 manners ⊆ S0's 26 classes.
- **Applied-operator alphabet = 11.** 15 of the 26 classes appear only as real transitions, never
  as an applied counterfactual manner (`air_bending, earth_element, earth_wave, fire_element,
  flame, giant_grab, money_rain, mystification, nature_bloom, plasma_explosion, run_set_on_fire,
  sakura_petals, water_bending, water_element, wonderland`).
- S1's extra 195 endpoints (DAVIS/new) extend the counterfactual beyond the corpus.

## Tier S2 — procedural 2D-shader transitions (S2a + S2b)

Sparse factorial: the same endpoint pair driven through many parametric shader operators;
endpoints byte-identical by construction (`assert1.mae_pure = 0`). **The counterfactual core of cttv2.**

| | S2a (synth pool) | S2b (humanvid pool) | S2 combined |
|---|---|---|---|
| operators (`op_id`) | 791 | 799 | 1,590 |
| endpoints (A,B pairs) | 318 | 785 | 1,100 |
| ops/endpoint (mean) | **23.83** | 10.01 | **14.03** |
| ops/endpoint (min/med/max) | 16 / 24 / 31 | 2 / 10 / 17 | 2 / 11 / 36 |
| % endpoints ≥2 ops | **100%** | 100% | **100%** (99% ≥5) |
| endpoints/operator (mean) | 9.58 | 9.84 | 9.71 |
| fidelity | byte-exact | byte-exact | byte-exact |

- Shader-**family** level: ~56 shaders, ~20 shaders/endpoint, ~117 endpoints/shader.
- Operators are parametric (shader + params + easing + timing + flip/swap) ⇒ both **exact** and
  **one-param-apart near-miss** counterfactuals are free.

## Tier S4 — refVFX I2V-LoRA (external)

Remade-AI Wan2.1 I2V effect LoRAs; **self-conditioned** in cttv2 (`endpoints[c]=[c]`, one-sided,
frame-0). Each clip has its own input image — **no shared endpoints, no counterfactual.**

| | S4 |
|---|---|
| operators (triggers) | 42 |
| endpoints (clips) | 2,000 |
| ops/endpoint (mean) | **1.00** (max 1) |
| % endpoints ≥2 ops | **0%** |
| endpoints/operator (mean) | 47.6 (var 0.47) |
| fidelity | — |

- Verified: full I2V_LoRA source = 6,995 distinct input fingerprints, **0 shared across its 48
  effects**. Perceptual near-dup check on the 2,000 selected inputs → nearest cross-effect pairs
  differ by median 21.6/255 (not the same source; EffectData same-source is 0.99/255).
- The shared-content counterfactual refVFX *does* have lives in its **code-based** subset
  (planned S5, **not ingested**): 136,800 samples, 133,749 contents, **3,019 reused across ≥2 effects**.

---

## What this means

- cttv2's counterfactual strength is **S2** (100% coverage, mean ~14–24 ops/endpoint, byte-exact,
  parametric) plus **S1** (byte-exact over 11 creative manners, 95% of S0's reals covered).
- **S0 and S4 are demonstration-only** (0% counterfactual).
- The one axis cttv2 lacks is **semantic-effect operator breadth** — S2's operators are geometric
  shaders, S1's are 11 creative manners. That, not counterfactual density, is what an external
  semantic source (e.g. HF `ysy31415926/EffectData`, 3,061 effects) would add.

**Reproduce:** bipartite counts from `root/samples.jsonl` (operator = `group_slug`, endpoint =
clip stem; for S1 split `spec_<op>__<content>__<seed>`); for S2, join clip stems to the gen meta
and use `pair_id`/`(A,B)` as the endpoint and `op_id` as the operator.
