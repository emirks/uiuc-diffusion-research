# 005_ctt_v2plus_s6reshape — dataset entry point

**cttv2 + EffectData (S6, RESHAPED), CODE-SIDE form.** Successor of `004_ctt_v2plus` in which **only
stratum S6 changes**: the 4-grid native EffectData zoo is re-encoded at **two orientation grids**
(832×512 / 512×832 × 81 f → latent (11,16,26) / (11,26,16), **4,576 tokens each**). This makes S6
grid-consistent for the signal contract, token-matched with the corpus (4,800), and pairing-complete
enough that **2,286 targets dropped in 004 return**. A **138,147-pair** LTX-2 transition-training set
across six strata, portable across devices via one `_src` symlink. This directory is **self-sufficient**.

## Quick facts

| | |
|---|---|
| pairs | **138,147** (S0 385 / S1 3,675 / S2a 22,731 / S2b 23,577 / S4 6,000 / **S6 81,779**) |
| mix (sampler weights) | S0 12 / S1 4.8 / S2a 27.10 / S2b 28.10 / S4 8 / S6 20 — sums to 100 (a RUN knob; configs_005 override to S0 2 / S1 18 / S2a 22.09 / S2b 22.91 / S4 8 / S6 27) |
| form | **code-side** — `samples.jsonl` with root-relative paths; no per-row symlink trees |
| `samples_sha256` | `048d1ef45d8ec98664291c8bce1c8c4b1f756435ae180abc89979ba323747d98` (pin at run start) |
| S6 | 28,644 clips / 2 grids; 28,552 trained targets (92 same-grid singletons dropped = unseen-subject eval material) |
| signal + norm | 44-ch DINO operator signal at the r832 grids, `NORM_dino_v4` → **`../003_dino_signals/`**; signal root `$LAB/cache/armA_signals_005/feat` |
| version | `3.1.0-ctt_v2plus_s6reshape-codeside` |
| predecessor | `004_ctt_v2plus` (UNTOUCHED) |

## Read this — in order

| doc | what it is | when to read |
|---|---|---|
| **[`CODESIDE_FORMAT.md`](CODESIDE_FORMAT.md)** | **the primary doc** — the deltas vs 004's format (2 S6 grids, 5 masks, 138,147 rows, `_src` bring-up now also needs `EFFECTDATA_r832`), row schema, verification battery | to use / move / rebuild the dataset |
| [`BUILD.md`](BUILD.md) | the **S6 r832 build** — spec table (A1–A3), derived roster, encode/extract arrays + GPU-h, health, norm v4, pairing 81,779 / 92 singletons / 2,286 returned | to trace how the reshaped S6 was built |
| [`meta.yaml`](meta.yaml) | the store registry record | for the canonical registry entry |
| `root/README.md` + `root/ROOT_MANIFEST.json` | the physical root's bring-up card + machine-readable ground-truth (via the `root` symlink) | for exact counts / provenance |
| **`../003_dino_signals/meta.yaml`** | the 44-ch DINO signal + `NORM_dino_v4` addendum (a sibling store entry) | for the signal & normalization |

## Layout of this store entry

```
store/datasets/005_ctt_v2plus_s6reshape/
├── README.md            ← you are here (entry point / index)
├── CODESIDE_FORMAT.md   the format deltas vs 004 + usage + verify doc
├── BUILD.md             the S6 r832 build (spec, encode/extract, health, pairing)
├── meta.yaml            registry record
└── root ──────────────► ../../../outputs/ctt_v2/roots/ctt_v2plus_s6reshape_mix
                         (samples.jsonl · mix.json · ROOT_MANIFEST.json · CAPTIONS.json
                          · _mask_store/ (5) · _src · VERSION · README.md · _shape_cache.json)
```

## Train / relaunch

```yaml
data:
  sample_list: <device>/ctt_v2plus_s6reshape_mix/samples.jsonl   # SampleListDataset does the rest
```

DeltaAI: `configs_005/` (rendered by `misc/2026-08-27_dino_signal_training/build/make_005_configs.py`)
points `data.sample_list` here, `signal.root` at `$LAB/cache/armA_signals_005/feat`, `signal.norm`
at `NORM_dino_v4.json`. Assert the `samples_sha256` above at run start (`build/assert_pins.sh`).

## Verification (Round 4, all PASS)

- `misc/2026-08-24_flow_signal_conditioning/armA/CODESIDE_VERIFY_005.md` — invariants 1–5 PASS
  (counts 138,147; S6 same-grid + different-subject on all 81,779; existence FULL; 5 masks).
- `misc/2026-08-24_flow_signal_conditioning/armA/VALIDATION_TRAINING_READY_005.md` — V1 100%
  coverage / V6 eval 223 / V8 norm smoke PASS.
- `misc/2026-08-30_s6_reshape/r4/COMPARE_004_005.md` — non-S6 rows multiset-identical to 004;
  S6 81,779 / 28,552 targets / 2,286 returned / 92 dropped; mask frame-0 sum 416.
- `build/signal_fsck.py` 100% hit in every stratum; `build/verify_signal_store.py` PASS ×5 arms.

## eps ship (Round 5, 2026-08-30)

Shipped to eps (`fal-h100`, `/storage/ozgur/dino_signal/`) in **NEW sibling dirs** (004 + the live A0-004
run untouched): root `datasets/ctt_v2plus_s6reshape_mix/` (`samples.jsonl` `048d1ef4`, `_src` → `srcroot`
== 004's, **no `_shape_cache.json`**), S6 tensors `…/encodes/EFFECTDATA_r832/{latents,cond_clean}` (28,644×2),
`signals_005/feat` (47,667 = 28,644 real S6 + 19,023 **hardlinked** non-S6; NORM_dino_v4 `db47be88`, pca
`4d59539b`), `campaign/configs_005/`, and an `env.sh` `SIGNALS` knob (set `SIGNALS=$DS/signals_005`). All eps
CPU verification PASS (assert_pins ×5, verify_signal_store ×5 v4, `01_preflight` `01 OK` fsck S6 81,779/0 miss).
Details + verification table in **`CODESIDE_FORMAT.md` → eps ship**; full trajectory in the campaign DOSSIER Round 5.

## `_shape_cache.json`

`root/_shape_cache.json` is a **regenerable** local build cache (keys `realpath|size|mtime` → `[F,H,W]`);
the assembler does not rewrite it on a 100% hit and it is **excluded from the eps ship**. It is not part of
the dataset contract — `samples_sha256` / `ROOT_MANIFEST.json` are. `samples_sha256` / `samples_rows` are now
emitted **natively** by `assemble_root.py` (Round-6 commit); the 004/005 manifests were stamped post-assembly
(Round 4) and Round 6's determinism proof verified native emission EQUAL. See `CODESIDE_FORMAT.md`.

## Note on S6 (reshaped)

S6 pairs are **same-shape same-effect, different-subject**, paired *within effect × orientation grid*.
**81,779** pairs over 28,552 targets; **92** clips had no same-grid same-effect partner and are
**dropped** (valid unseen-subject eval material, not trained). Collapsing the 4-grid native zoo to 2
grids returned **2,286** targets that 004 had dropped. See `CODESIDE_FORMAT.md` / `BUILD.md`.
